"""Exploratory, athlete-specific pace/context associations, not causal credits.

One observation per physical run. Missing context does not delete an otherwise
usable run. Calendar-local controls separate changing fitness from context as
far as the available overlap permits; ridge shrinkage limits small-n overfit.
This reader neither changes the released pace model nor calls the planner.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import math

import numpy as np

from .model_windows import load_overlapping_model_windows
from .objective_modeling import _robust_weighted_speed, _wbgt_c, pace_to_speed_mps, speed_to_pace_min_mile
from .run_feedback import _infer_workout_type
from .session_feedback import with_session_feedback
import json


VERSION = 'performance-factors-v1'
# These are modeling assumptions, not physiological laws or user preferences.
CALENDAR_DAYS = 56
RIDGE = 4.0
NUMERIC = (
    ('heat_c', 'Heat and humidity', '°C shade heat index'),
    ('headwind_mph', 'Headwind versus tailwind', 'mph'),
    ('crosswind_mph', 'Crosswind', 'mph'),
    ('grade_percent', 'Net uphill/downhill', '% grade'),
    ('hills_percent', 'Hilliness', '% absolute grade'),
    ('duration_minutes', 'Run duration', 'min'),
    ('previous_7d_miles', 'Recent volume', 'mi in preceding 7 days'),
    ('previous_28d_miles', 'Established volume', 'mi in preceding 28 days'),
    ('days_since_run', 'Time since last run', 'days'),
    ('days_since_hard', 'Time since demanding run', 'days'),
    ('hour_sin', 'Time of day (daily cycle)', 'sine of local hour'),
    ('hour_cos', 'Time of day (daily cycle)', 'cosine of local hour'),
)
CATEGORICAL = (('workout_type', 'Workout type'), ('device', 'Recording device'), ('health_tag', 'Reported health'))
FEEDBACK = (('perceived_exertion', 'Post-run effort', '0–10'), ('workout_feel', 'Post-run feeling', '0–100'))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _day(row):
    value = datetime.fromisoformat(row['start_time_utc'])
    if value.tzinfo is None:
        raise ValueError('Timezone-aware activity times required')
    return value.timestamp() / 86400


def load_factor_observations(connection, config):
    """Broader than the fitness trend: retain quality context and short runs.

    Reuse physical sensor/stop checks, but not the trend's Z2-only, normal-health,
    GPS-only, exact-20-minute or 60-minute-duration acceptance restrictions.
    The source activity eligibility check is retained and disclosed in coverage.
    """
    settings = deepcopy(config)
    model = settings['model']
    model['minimum_reliable_hr_bpm'] = float(config['resting_hr']) + .25 * (float(config['max_hr']) - float(config['resting_hr']))
    model['maximum_reliable_hr_bpm'] = float(config['resting_hr']) + .95 * (float(config['max_hr']) - float(config['resting_hr']))
    model['maximum_reliable_segment_minutes'] = 24 * 60  # no long-duration cutoff; duration is analyzed
    windows, diagnostics = load_overlapping_model_windows(connection, settings,
        window_seconds=int(model['primary_window_seconds']), stride_seconds=int(model['primary_window_stride_seconds']))
    grouped = defaultdict(list)
    for row in windows:
        grouped[row['activity_id']].append(row)
    metadata = with_session_feedback(connection, connection.execute('''
        SELECT a.id,a.start_time_local,a.creator,m.calculated_moving_time_s,m.analysis_distance_m,
               m.hr_zone_seconds_json,o.perceived_exertion,o.health_tag AS reported_health
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''').fetchall())
    metadata = {r['id']: r for r in metadata}
    output = []
    for identity, rows in grouped.items():
        meta = metadata[identity]
        # Downweight transitions continuously instead of discarding workout types.
        weights = np.array([math.exp(-abs(r.get('heart_rate_change_bpm') or 0) / float(model['primary_hr_change_weight_scale_bpm']))
            * math.exp(-abs(r.get('speed_change_fraction') or 0) / float(model['primary_speed_change_weight_scale_fraction']))
            for r in rows])
        def measured(key, *, absolute=False):
            pairs = [(abs(r[key]) if absolute else r[key], w) for r,w in zip(rows,weights) if _finite(r.get(key))]
            return float(np.average([p[0] for p in pairs], weights=[p[1] for p in pairs])) if pairs else None
        heat = []
        for row, weight in zip(rows, weights):
            try:
                value = _wbgt_c(row)
                if _finite(value): heat.append((value, weight))
            except (ValueError, TypeError, KeyError):
                pass
        minutes = meta['calculated_moving_time_s'] / 60 if meta['calculated_moving_time_s'] else None
        miles = meta['analysis_distance_m'] / 1609.344 if meta['analysis_distance_m'] else 0
        zone_seconds = json.loads(meta['hr_zone_seconds_json'] or '{}')
        role = _infer_workout_type(rows[0]['workout_type'], None, miles, minutes or 0, zone_seconds).value
        if role in {'bike', 'hike', 'other'}: continue
        try:
            local = datetime.fromisoformat(meta['start_time_local'])
            hour = local.hour + local.minute / 60
        except (TypeError,ValueError):
            hour = None  # Unknown local time must not silently become UTC.
        device = str(meta['creator'] or 'unknown').strip().lower()
        device = {'fr245':'forerunner 245', 'fr265 small':'forerunner 265s'}.get(device,device)
        grade_rows = [r for r in rows if r.get('grade_energy_ratio') is not None]
        valid_grade = {id(r) for r in grade_rows}
        def grade_mean(absolute=False):
            pairs = [(abs(r['average_grade_percent']) if absolute else r['average_grade_percent'], w)
                for r,w in zip(rows,weights) if id(r) in valid_grade and _finite(r.get('average_grade_percent'))]
            return float(np.average([p[0] for p in pairs], weights=[p[1] for p in pairs])) if pairs else None
        output.append(dict(activity_id=identity, start_time_utc=rows[0]['start_time_utc'],
            observed_pace_min_mile=speed_to_pace_min_mile(_robust_weighted_speed(
                [pace_to_speed_mps(r['moving_pace_min_mile']) for r in rows], weights.tolist())),
            sampled_hr_bpm=measured('average_hr_bpm'), sampled_minutes=measured('moving_minutes_into_run'),
            heat_c=float(np.average([p[0] for p in heat], weights=[p[1] for p in heat])) if heat else None,
            headwind_mph=measured('headwind_signed_mph'), crosswind_mph=measured('crosswind_mph'),
            grade_percent=grade_mean(), hills_percent=grade_mean(True),
            duration_minutes=minutes, previous_7d_miles=rows[0]['previous_7d_miles'],
            previous_28d_miles=rows[0]['previous_28d_miles'], days_since_run=rows[0]['days_since_previous_run'],
            days_since_hard=rows[0]['days_since_previous_hard_run'],
            hour_sin=math.sin(hour*math.tau/24) if hour is not None else None,
            hour_cos=math.cos(hour*math.tau/24) if hour is not None else None,
            workout_type=role, health_tag=meta['reported_health'] or 'unreported', device=device,
            perceived_exertion=meta['perceived_exertion'], workout_feel=meta['workout_feel'],
            feedback_conflict=meta['feedback_conflict'], effort_source=meta['effort_source'],
            feeling_source=meta['feeling_source'], distance_miles=miles,
            grade_window_coverage=len(grade_rows)/len(rows), sampled_windows=len(rows)))
    exclusions = dict(connection.execute('SELECT COALESCE(exclusion_reason,\'none recorded\'),count(*) FROM activity_metrics GROUP BY 1').fetchall())
    return sorted(output, key=lambda r: (_day(r), r['activity_id'])), dict(
        imported_activities=connection.execute('SELECT count(*) FROM activities').fetchone()[0],
        usable_runs=len(output), source_model_eligible_runs=connection.execute('SELECT count(*) FROM activity_metrics WHERE model_eligible=1').fetchone()[0],
        recorded_exclusion_reason_counts=exclusions, window_diagnostics=diagnostics,
        sampling='Stable-weighted submaximal sections; one observation per run, not independent windows',
        missing_context='Available sections averaged; optional missing factors retained as missing, not flat/cool/recovered')


def _prepare(rows):
    valid, skipped, seen = [], Counter(), set()
    for row in rows:
        identity = row.get('activity_id')
        if identity is None: skipped['missing_identity'] += 1; continue
        if identity in seen: raise ValueError('Each physical activity must occur only once')
        seen.add(identity)
        try: _day(row)
        except (ValueError, TypeError, KeyError): skipped['invalid_time'] += 1; continue
        if not all(_finite(row.get(k)) for k in ('observed_pace_min_mile','sampled_hr_bpm','sampled_minutes')) or row['observed_pace_min_mile'] <= 0:
            skipped['missing_pace_hr_or_position'] += 1; continue
        valid.append(row)
    return sorted(valid, key=lambda r: (_day(r),r['activity_id'])), dict(skipped)


def _design(rows, spec=None):
    """Training-only scales/categories; no future medians or unseen-device fit."""
    if spec is None:
        spec = []
        for key,label,unit in NUMERIC:
            values = [r[key] for r in rows if _finite(r.get(key))]
            if len(values) < 2 or np.std(values) < 1e-8: continue
            spec.append(dict(key=key,label=label,unit=unit,kind='numeric',center=float(np.mean(values)),
                scale=float(np.std(values)),low=float(np.quantile(values,.25)),high=float(np.quantile(values,.75)),
                observed_runs=len(values)))
            if len(values) != len(rows): spec.append(dict(key=key,kind='missing'))
        for key,label in CATEGORICAL:
            counts = Counter(str(r.get(key) or 'unknown') for r in rows)
            baseline = sorted(counts, key=lambda v:(-counts[v],v))[0]
            for level in sorted(counts):
                if level != baseline:
                    spec.append(dict(key=key,label=f'{label}: {level}',unit=f'versus {baseline}',kind='category',
                        level=level,baseline=baseline,low=0.,high=1.,observed_runs=counts[level]))
    columns = []
    for feature in spec:
        key, kind = feature['key'], feature['kind']
        if kind == 'numeric': columns.append([(r[key]-feature['center'])/feature['scale'] if _finite(r.get(key)) else 0. for r in rows])
        elif kind == 'missing': columns.append([float(not _finite(r.get(key))) for r in rows])
        else: columns.append([float(str(r.get(key) or 'unknown')==feature['level']) for r in rows])
    return np.array(columns,dtype=float).T.reshape(len(rows),len(spec)), spec


def _core(rows, spec=None):
    values = np.array([[r['sampled_hr_bpm'],r['sampled_minutes']] for r in rows])
    if spec is None: spec = (values.mean(axis=0),np.maximum(values.std(axis=0),1.))
    x = (values-spec[0])/spec[1]
    return np.column_stack((x,x[:,0]**2)), spec


def _fit(rows, *, ridge=RIDGE, period=CALENDAR_DAYS, bootstrap=400):
    if len(rows)<2: return dict(status='more_observations_needed',runs=len(rows))
    x,spec = _design(rows)
    core,core_spec = _core(rows)
    days = np.array([_day(r) for r in rows]); groups = np.floor(days/period).astype(int)
    calendar=[]
    for group in np.unique(groups):
        mask=groups==group
        calendar.extend([mask.astype(float),np.where(mask,(days-days[mask].mean())/period,0)])
    z=np.column_stack((*calendar,core))
    y=np.log([r['observed_pace_min_mile'] for r in rows])
    ry=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    rx=x-z@np.linalg.lstsq(z,x,rcond=None)[0]
    rank=int(np.linalg.matrix_rank(z))
    if len(rows)-rank<2 or not spec:
        return dict(status='limited_within_period_comparisons',runs=len(rows),calendar_blocks=len(np.unique(groups)))
    inverse=np.linalg.inv(rx.T@rx+ridge*np.eye(x.shape[1]))
    beta=inverse@rx.T@ry
    context=np.linalg.lstsq(z,y-x@beta,rcond=None)[0]
    baseline=np.linalg.lstsq(z,y,rcond=None)[0]
    residual=ry-rx@beta
    effective=float(np.trace(inverse@rx.T@rx))
    correction=math.sqrt(len(rows)/max(1.,len(rows)-rank-effective))
    rng=np.random.default_rng(20261013)
    # Calendar-block wild perturbations: descriptive conditional ranges, not
    # calibrated physiological confidence or independent individual windows.
    signs=rng.choice([-1.,1.],size=(bootstrap,len(np.unique(groups))))
    lookup={g:i for i,g in enumerate(np.unique(groups))}
    perturb=residual[:,None]*signs[:,[lookup[g] for g in groups]].T*correction
    draws=beta[:,None]+inverse@rx.T@perturb
    median_pace=float(np.median([r['observed_pace_min_mile'] for r in rows]))
    effects=[]
    for index,feature in enumerate(spec):
        if feature['kind']=='missing': continue
        contrast=(feature['high']-feature['low'])/(feature['scale'] if feature['kind']=='numeric' else 1)
        separation=float(np.sum(rx[:,index]**2))
        effect=dict(feature,within_period_variation=separation,
            status='tentative_association' if separation>1e-8 else 'confounded_with_time_or_effort')
        if separation>1e-8:
            values=np.expm1(np.clip(draws[index]*contrast,-10,10))*median_pace*60
            effect.update(seconds_per_mile=float(math.expm1(float(beta[index]*contrast))*median_pace*60),
                range_95_seconds_per_mile=np.quantile(values,[.025,.975]).tolist() if len(np.unique(groups))>1 else None,
                range_status='conditional_block_perturbation' if len(np.unique(groups))>1 else 'single_calendar_block',
                log_pace_coefficient_standardized=float(beta[index]),
                direction='slower' if beta[index]*contrast>0 else 'faster')
        else: effect.update(seconds_per_mile=None,range_95_seconds_per_mile=None,log_pace_coefficient_standardized=None)
        effects.append(effect)
    return dict(status='estimated_associations',runs=len(rows),calendar_blocks=len(np.unique(groups)),
        effective_context_parameters=effective,calendar_and_effort_rank=rank,ridge=ridge,period_days=period,
        factors=effects,median_observed_pace_min_mile=median_pace,
        _beta=beta,_spec=spec,_core_spec=core_spec,_context=context[-3:],_baseline=baseline[-3:])


def _chronological_check(rows):
    keys=sorted({math.floor(_day(r)/CALENDAR_DAYS) for r in rows})
    errors=[]; folds=[]; unstable_total=0; unseen_total=0
    for left,right in ((.5,.65),(.65,.8),(.8,1.)):
        start,end=int(len(keys)*left),int(len(keys)*right)
        train=[r for r in rows if math.floor(_day(r)/CALENDAR_DAYS) in keys[:start]]
        fit=_fit(train,bootstrap=2)
        folds.append(dict(training_runs=len(train),test_blocks=end-start,fit_status=fit['status']))
        if fit['status']!='estimated_associations': continue
        for key in keys[start:end]:
            test=[r for r in rows if math.floor(_day(r)/CALENDAR_DAYS)==key]
            x,_=_design(test,fit['_spec']); core,_=_core(test,fit['_core_spec'])
            for name,_ in CATEGORICAL:
                known={str(r.get(name) or 'unknown') for r in train}
                unseen_total+=sum(str(r.get(name) or 'unknown') not in known for r in test)
            block=[]
            for i,first in enumerate(test):
                for j in range(i+1,len(test)):
                    if _day(test[j])-_day(first)>7: continue
                    baseline=float((core[j]-core[i])@fit['_baseline'])
                    full=float((core[j]-core[i])@fit['_context']+(x[j]-x[i])@fit['_beta'])
                    actual=test[j]['observed_pace_min_mile']
                    if max(abs(baseline),abs(full))>2:
                        unstable_total+=1;block.append((None,None));continue
                    block.append((abs(first['observed_pace_min_mile']*math.exp(baseline)-actual)*60,
                                  abs(first['observed_pace_min_mile']*math.exp(full)-actual)*60))
            valid=[p for p in block if p[0] is not None]
            if valid: errors.append(dict(block=key,pairs=len(valid),baseline=float(np.mean([p[0] for p in valid])),
                                         factors=float(np.mean([p[1] for p in valid])),unstable_pairs=len(block)-len(valid)))
    return dict(status='descriptive_out_of_time_check' if errors else 'not_enough_chronological_contrasts',
        folds=folds,held_out_blocks=len(errors),pairs=sum(r['pairs'] for r in errors),
        baseline_mae_seconds_per_mile=float(np.mean([r['baseline'] for r in errors])) if errors else None,
        factors_mae_seconds_per_mile=float(np.mean([r['factors'] for r in errors])) if errors else None,
        unstable_pairs=unstable_total,unseen_category_values=unseen_total,
        scope='Later nearby-run pace contrasts, anchored to first recorded pace; not an independently validated absolute fitness trend')


def analyze_performance_factors(observations):
    rows,skipped=_prepare(observations)
    fit=_fit(rows)
    public={k:v for k,v in fit.items() if not k.startswith('_')}
    sensitivity=[{k:v for k,v in _fit(rows,ridge=alpha,bootstrap=100).items() if not k.startswith('_')}
                 for alpha in (1.,16.)]
    calendar_sensitivity=[{k:v for k,v in _fit(rows,period=period,bootstrap=100).items() if not k.startswith('_')}
                          for period in (28,112)]
    latest=max((_day(r) for r in rows),default=0)
    cohort_sensitivity=[]
    for label,cohort in (
        ('easy_and_long',[r for r in rows if r.get('workout_type') in {'easy','long','recovery'}]),
        ('trailing_365_days',[r for r in rows if _day(r)>=latest-365])):
        cohort_sensitivity.append(dict(cohort=label,**{k:v for k,v in _fit(cohort,bootstrap=100).items() if not k.startswith('_')}))
    coverage=[]
    for key,label,unit in (*NUMERIC,*FEEDBACK):
        values=[r[key] for r in rows if _finite(r.get(key))]
        coverage.append(dict(key=key,label=label,unit=unit,observed_runs=len(values),missing_runs=len(rows)-len(values),
            observed_range=[min(values),max(values)] if values else None))
    # Feedback describes the completed experience, not an independent pre-run
    # explanation. Retain its data without allowing it to 'explain away' fitness.
    feedback=[]
    for key,label,unit in FEEDBACK:
        pairs=[(r[key],r['observed_pace_min_mile']) for r in rows if _finite(r.get(key)) and not r.get('feedback_conflict')]
        feedback.append(dict(key=key,label=label,runs=len(pairs),
            status='recorded_post_run_context_not_adjustment',
            conflict_runs=sum(bool(r.get('feedback_conflict')) for r in rows if _finite(r.get(key)))))
    return dict(version=VERSION,status=fit['status'],observations=len(rows),skipped=skipped,
        outcome='Observed submaximal pace, jointly controlling sampled HR, within-run position and local calendar trend',
        primary=public,shrinkage_sensitivity=sensitivity,calendar_sensitivity=calendar_sensitivity,
        cohort_sensitivity=cohort_sensitivity,
        chronological_check=_chronological_check(rows),
        coverage=coverage,feedback=feedback,
        unavailable_factors=['Shoes/equipment tags','Sleep','Pre-run hydration/nutrition','Validated activity-HRV features'],
        adjusted_trend_promoted=False,
        assumptions=dict(calendar_days=CALENDAR_DAYS,ridge=RIDGE,optional_missing='Observed-mean centering plus explicit missingness indicator; missing is not a measured zero',
            contrast='Observed lower to upper quartile; categories versus most common category',
            uncertainty='Conditional calendar-block perturbation range; shrinkage bias and unmeasured influences remain',
            acceptance='Useful tentative associations are shown even when ranges include zero; no p-value feature selection or forced signs'),
        limitations=['Associations are not causal pace credits or training prescriptions.',
            'Workout type, HR lag, effort choice and sensor differences can still confound the result.',
            'Unseen categories in chronological checks use the reference prediction and are counted, not treated as established matches.',
            'Time-of-day sine/cosine are one joint cyclic effect, not two independent causes.',
            'No feedback, sleep or equipment data is invented; post-run feedback is not readiness.'])
