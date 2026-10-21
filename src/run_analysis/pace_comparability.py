"""Bounded, literature-informed pace normalization with athlete calibration.

This is a comparison estimate, not measured VO2 or a fresh-athlete forecast.
Raw observations are retained. Missing context never removes the whole run.
Uncertain priors work immediately; run-centered data refine HR, drift and wind,
and calendar-local data refine heat without treating seasons as fitness proof.
"""
from collections import defaultdict
from datetime import datetime
import math

import numpy as np

from .objective_modeling import (
    _primary_window_weight, _robust_weighted_speed,
    _shared_parameter_eligible, _pre_quality_indexes, _wbgt_c, _reference_wbgt_c,
    pace_to_speed_mps, speed_to_pace_min_mile,
)
from .model_windows import load_model_window_sets, load_overlapping_model_windows
from .quality_phases import detect_continuous_quality_phase


VERSION = 'comparable-pace-v2-terrain-quality'
PRIOR_MEAN = np.array([1., 0., .5])  # HRR elasticity, drift/10 min, uncertain wind scale
PRIOR_SD = np.array([.6, .025, .5])
PARAMETER_BOUNDS = np.array([[.25,2.],[-.05,.05],[0.,1.]])
AIR_COST = .002  # Population approximation per (m/s)^2, not measured runner CdA.
HEAT_MINIMUM_SE = .001
STAGES = ('observed','terrain','wind','heart_rate','duration','weather')


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _features(row,config):
    if not all(_finite(row.get(key)) for key in ('moving_pace_min_mile','average_hr_bpm','moving_minutes_into_run')) or row['moving_pace_min_mile']<=0:
        raise ValueError('Finite positive pace, HR and within-run time required')
    speed=pace_to_speed_mps(row['moving_pace_min_mile'])
    ratio=row.get('grade_energy_ratio')
    # The loader's ratio already includes ascent AND descent, before aggregation.
    supplied_coverage=row.get('grade_complete_fraction',1.)
    coverage=float(supplied_coverage) if _finite(ratio) and ratio>0 and _finite(supplied_coverage) else 0.
    coverage=min(1.,max(0.,coverage))
    if (row.get('terrain_quality') or {}).get('usable') is False:
        coverage = 0.
    effective_grade=1.+coverage*(float(ratio)-1.) if _finite(ratio) and ratio>0 else 1.
    head,cross=row.get('headwind_signed_mph'),row.get('crosswind_mph')
    wind_log=0.;wind_known=_finite(head) and _finite(cross)
    if wind_known:
        forward=speed+head*.44704
        relative=math.hypot(forward,cross*.44704)
        cost=(1.+AIR_COST*forward*relative)/(1.+AIR_COST*speed**2)
        if cost>0:
            wind_log=math.log(cost)
        else: wind_known=False  # No extrapolated free-energy estimate in extreme assisting wind.
    reserve=float(row['average_hr_bpm'])-float(config['resting_hr'])
    target=float(config['target_hr'])-float(config['resting_hr'])
    if reserve<=0 or target<=0: raise ValueError('HR comparison must be above resting HR')
    hr_log=math.log(reserve/target)
    time=(float(row['moving_minutes_into_run'])-float(config['reference_conditions']['within_run_minutes']))/10.
    try:
        heat=max(0.,_wbgt_c(row)-_reference_wbgt_c(config))
        if not math.isfinite(heat):heat=None
    except (ValueError,TypeError,KeyError):heat=None
    return dict(speed=speed,grade=effective_grade,grade_coverage=coverage,
        wind_log=wind_log,wind_known=wind_known,hr_log=hr_log,time=time,heat=heat)


def _calibration_rows(rows):
    selected=defaultdict(list)
    for row in rows:
        # Contradictory altitude must not teach the shared response that bogus
        # climbs caused a run's HR/pace relationship. The run itself is retained.
        quality = row.get('terrain_quality') or {}
        if quality.get('status') == 'rejected' or quality.get('calibration_eligible') is False:
            continue
        if _shared_parameter_eligible(row):selected[row['activity_id']].append(row)
    return [sorted(group,key=lambda r:r['moving_minutes_into_run']) for _,group in sorted(selected.items())]


def fit_comparison_calibration(rows,config):
    groups=_calibration_rows(rows)
    xparts=[];yparts=[];wparts=[];ids=[]
    for group in groups:
        features=[_features(r,config) for r in group]
        x=np.array([[f['hr_log'],f['time'],-f['wind_log']] for f in features])
        y=np.log([f['speed']*f['grade'] for f in features])
        w=np.array([_primary_window_weight(r,config) for r in group]);w/=w.sum()
        # Every physical run has total weight one. A long run is not dozens of independent runners.
        xparts.append(x-np.average(x,axis=0,weights=w));yparts.append(y-np.average(y,weights=w))
        wparts.append(w);ids.extend([group[0]['activity_id']]*len(group))
    beta=PRIOR_MEAN.copy();cov=np.diag(PRIOR_SD**2);sigma=.05
    if xparts:
        x=np.vstack(xparts);y=np.concatenate(yparts);base=np.concatenate(wparts)
        robust=np.ones(len(y));precision=np.diag(1/PRIOR_SD**2)
        for _ in range(8):
            residual=y-x@beta
            sigma=max(.02,1.4826*float(np.median(abs(residual-np.median(residual)))))
            robust=np.minimum(1.,1.345*sigma/np.maximum(abs(residual),1e-12))
            weight=base*robust/sigma**2
            cov=np.linalg.inv(x.T@(weight[:,None]*x)+precision)
            beta=cov@(x.T@(weight*y)+precision@PRIOR_MEAN)
            beta=np.clip(beta,PARAMETER_BOUNDS[:,0],PARAMETER_BOUNDS[:,1])
        scores=[];ids=np.array(ids)
        for key in np.unique(ids):
            mask=ids==key;scores.append(x[mask].T@(weight[mask]*(y[mask]-x[mask]@beta)))
        sandwich=cov@np.array(scores).T@np.array(scores)@cov
        scale=np.sqrt(np.maximum(1.,np.diag(sandwich)/np.maximum(np.diag(cov),1e-12)))
        cov=cov*scale[:,None]*scale[None,:]
    heat=fit_heat_response(groups,config,beta)
    return dict(version=VERSION,target_hr_bpm=float(config['target_hr']),
        reference_minutes=float(config['reference_conditions']['within_run_minutes']),
        parameters=beta.tolist(),covariance=cov.tolist(),
        parameter_names=['hr_reserve_elasticity','log_speed_drift_per_10_minutes','wind_adjustment_scale'],
        prior_mean=PRIOR_MEAN.tolist(),prior_sd=PRIOR_SD.tolist(),parameter_bounds=PARAMETER_BOUNDS.tolist(),
        contributing_runs=len(groups),within_run_log_speed_dispersion=sigma,heat=heat,
        assumptions=dict(air_cost_per_relative_speed_squared=AIR_COST,
            wind='Pugh-style squared relative-air-speed cost; exposure is uncertain and athlete-calibrated, not measured at the runner',
            hr='HR reserve / oxygen-uptake-reserve analogy supplies a local submaximal prior, not an exact pace law',
            duration='Run-centered drift toward the same reference minute, not recovery or durability improvement',
            terrain='Minetti metabolic-cost ratio on smoothed short sections; partial unknown terrain is neutral only in the point estimate',
            heat='Race-weather coefficient is a broad transfer prior; local run data update it, not independent causal validation',
            uncertainty='Conditional parameter/coverage sensitivity plus run dispersion; not calibrated physiological confidence'))


def fit_heat_response(groups,config,beta):
    settings=config['model']
    mean=float(settings.get('fixed_heat_loss_fraction_per_c',.002))
    sd=float(settings.get('fixed_heat_loss_uncertainty_fraction_per_c',.002))
    if mean<0 or sd<=0:raise ValueError('Nonnegative heat prior and positive uncertainty required')
    rows=[]
    for group in groups:
        known=[r for r in group if _features(r,config)['heat'] is not None]
        if not known:continue
        w=np.array([_primary_window_weight(r,config) for r in known]);w/=w.sum()
        fs=[_features(r,config) for r in known]
        log_speed=np.array([math.log(f['speed']*f['grade'])+beta[2]*f['wind_log']-beta[0]*f['hr_log']-beta[1]*f['time'] for f in fs])
        time=datetime.fromisoformat(group[0]['start_time_utc']).timestamp()/86400
        rows.append((time,float(np.average([f['heat'] for f in fs],weights=w)),
            -float(np.average(log_speed,weights=w)),
            [group[0].get(k) for k in ('previous_7d_miles','previous_28d_miles')]))
    evidence=dict(prior_mean=mean,prior_sd=sd,mean=mean,sd=sd,runs=len(rows),
        personal_estimate=None,personal_se=None,personal_weight=0.,reference_shade_wbgt_c=_reference_wbgt_c(config),
        basis='population transfer prior; no independent local heat contrast')
    if len(rows)<2:return evidence
    days=np.array([r[0] for r in rows]);heat=np.array([r[1] for r in rows]);y=np.array([r[2] for r in rows])
    blocks=np.floor(days/56).astype(int);columns=[]
    for key in np.unique(blocks):
        mask=blocks==key
        columns.extend([mask.astype(float),np.where(mask,(days-days[mask].mean())/56.,0.)])
    for index in range(2):
        known=[r[3][index] for r in rows if _finite(r[3][index])]
        center=float(np.mean(known)) if known else 0.
        scale=max(1.,float(np.std(known))) if known else 1.
        columns.append(np.array([(r[3][index]-center)/scale if _finite(r[3][index]) else 0. for r in rows]))
        if len(known)<len(rows):columns.append(np.array([float(not _finite(r[3][index])) for r in rows]))
    z=np.column_stack(columns)
    rh=heat-z@np.linalg.lstsq(z,heat,rcond=None)[0];ry=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    denominator=float(rh@rh);dof=len(rows)-int(np.linalg.matrix_rank(z))-1
    if denominator<1e-8 or dof<=0:return evidence
    estimate=float(rh@ry/denominator);residual=ry-estimate*rh
    # Use blocks, not window or reused-pair counts, as the uncertainty unit.
    scores=[float(np.sum(rh[blocks==key]*residual[blocks==key])) for key in np.unique(blocks)]
    se=max(HEAT_MINIMUM_SE,math.sqrt(sum(s*s for s in scores))/denominator*math.sqrt(len(rows)/dof),
           sd if len(scores)<2 else 0.)
    data_weight=sd**2/(sd**2+se**2)
    posterior=max(0.,min(.006,mean*(1-data_weight)+estimate*data_weight))
    return {**evidence,'mean':posterior,'sd':math.sqrt(1/(1/sd**2+1/se**2)),
        'personal_estimate':estimate,'personal_se':se,'personal_weight':data_weight,
        'calendar_blocks':len(np.unique(blocks)),'remaining_heat_sd_c':float(rh.std()),
        'basis':'broad population prior updated by calendar-local HR/terrain/drift/wind-normalized run associations'}


def compare_run_windows(rows,weights,config,calibration,*,precision_seed=None,_prepared_features=None):
    if not rows or len(rows)!=len(weights) or any(w<=0 or not math.isfinite(w) for w in weights):
        raise ValueError('Aligned reliable windows and positive finite weights required')
    beta=np.array(calibration['parameters']);cov=np.array(calibration['covariance'])
    heat=calibration['heat'];features=[_features(r,config) for r in rows] if _prepared_features is None else _prepared_features
    if len(features)!=len(rows):
        raise ValueError('Prepared features must align with all supplied windows')
    temperature = calibration.get('temperature_hr')
    weather_gradients=[];weather_known=[];weather_adjustments=[];population_weather=[]
    stages={key:[] for key in STAGES};vectors=[];unmeasured=[]
    for row,f in zip(rows,features):
        value=f['speed'];stages['observed'].append(value)
        value*=f['grade'];stages['terrain'].append(value)
        value*=math.exp(beta[2]*f['wind_log']);stages['wind'].append(value)
        value*=math.exp(-beta[0]*f['hr_log']);stages['heart_rate'].append(value)
        value*=math.exp(-beta[1]*f['time']);stages['duration'].append(value)
        if temperature:
            from .pace_temperature import hr_shift
            delta=hr_shift(row,config);reserve=row['average_hr_bpm']-config['resting_hr']
            known=bool(delta is not None and reserve-delta>0)
            log_hr_ratio=math.log1p(-delta/reserve) if known else 0.
            # Personalization scales the tested log-pace burden, not HR twice.
            # This preserves the exact offline candidate at window level.
            population_weather.append(value*math.exp(-beta[0]*log_hr_ratio))
            value*=math.exp(-beta[0]*temperature['scale']*log_hr_ratio)
            gradient=-beta[0]*log_hr_ratio
            vectors.append([-f['hr_log']-temperature['scale']*log_hr_ratio,-f['time'],f['wind_log']])
            weather_gradients.append(gradient);weather_known.append(known)
            weather_adjustments.append(reserve*(-math.expm1(temperature['scale']*log_hr_ratio)) if known else None)
        else:
            value*=math.exp(heat['mean']*(f['heat'] or 0.))
            vectors.append([-f['hr_log'],-f['time'],f['wind_log']])
            weather_gradients.append(f['heat'] or 0.);weather_known.append(f['heat'] is not None)
        stages['weather'].append(value)
        # Unknown is not evidence of flat/cool/calm. Keep its point estimate
        # neutral while carrying a separate uncertainty sensitivity.
        gps=float(row.get('gps_complete_fraction',1.) or 0.)
        shortfall=max(0.,1.-gps/max(float(config['model']['minimum_gps_coverage']),1e-9))
        distance_sd=float(config['model'].get('device_distance_fallback_uncertainty_seconds_per_mile',30))/60/float(row['moving_pace_min_mile'])/1.96
        unmeasured.append((.03*(1-f['grade_coverage']))**2+(.02 if not f['wind_known'] else 0.)**2
            +(.03 if not weather_known[-1] else 0.)**2+(distance_sd*shortfall)**2)
    pace={key:speed_to_pace_min_mile(_robust_weighted_speed(values,weights)) for key,values in stages.items()}
    def temperature_pace(selected_weights,indices=None):
        ix=list(range(len(rows))) if indices is None else indices
        duration=_robust_weighted_speed([stages['duration'][i] for i in ix],selected_weights)
        population=_robust_weighted_speed([population_weather[i] for i in ix],selected_weights)
        return speed_to_pace_min_mile(duration*math.exp(temperature['scale']*math.log(population/duration)))
    if temperature:
        # Match the audited run-level log-burden contract. Robust aggregation
        # and exponentiation do NOT commute, especially on mixed-effort runs.
        pace['weather']=temperature_pace(weights)
    contributions={key:(pace[key]-pace[previous])*60 for previous,key in zip(STAGES,STAGES[1:])}
    w=np.array(weights)/sum(weights);v=np.average(np.array(vectors),axis=0,weights=w)
    mean_heat=float(np.average(weather_gradients,weights=w))
    if temperature:
        mean_heat=math.log(_robust_weighted_speed(population_weather,weights)/_robust_weighted_speed(stages['duration'],weights))
    weather_sd=temperature['conditional_scale_sd'] if temperature else heat['sd']
    common_variance=float(v@cov@v)+(mean_heat*weather_sd)**2+float(np.average(unmeasured,weights=w))
    # Terrain-model transfer and unmeasured local wind are shared uncertainties,
    # not thousands of independent trackpoints that average to certainty.
    common_variance+=float(np.average([abs(math.log(f['grade'])) for f in features],weights=w)*.5)**2
    common_variance+=float(np.average([abs(f['wind_log']) for f in features],weights=w)*.5)**2
    center=math.log(pace_to_speed_mps(pace['weather']))
    dispersion=1.4826*float(np.median(abs(np.log(stages['weather'])-center)))
    # Keep run dispersion whole here; overlapping windows do not prove precision.
    uncertainty=pace['weather']*math.expm1(1.96*math.sqrt(common_variance+dispersion**2))
    precision=None
    if precision_seed is not None:
        from .aerobic_sampling import sampling_precision, robust_speed_samples
        def sampled_pace(indices):
            return temperature_pace([weights[i] for i in indices],indices)
        def sampled_paces(indices):
            duration=robust_speed_samples(stages['duration'],weights,indices)
            population=robust_speed_samples(population_weather,weights,indices)
            return 1609.344/(60*np.maximum(.1,duration*np.exp(temperature['scale']*np.log(population/duration))))
        _,sample=sampling_precision(rows,weights,np.array(stages['weather']),256,precision_seed,
            estimator=sampled_pace if temperature else None,
            batch_estimator=sampled_paces if temperature else None)
        precision=dict(sampling_precision=sample,log_variance_parts=dict(
            parameter=float(v@cov@v)+(mean_heat*weather_sd)**2,
            missing_conditions=float(np.average(unmeasured,weights=w)),
            terrain_transfer=float(np.average([abs(math.log(f['grade'])) for f in features],weights=w)*.5)**2,
            wind_transfer=float(np.average([abs(f['wind_log']) for f in features],weights=w)*.5)**2,
            whole_run_dispersion=dispersion**2))
    # Audited sensitivity only: broader sampling was not better overall. Keep
    # the canonical score/precision intact and expose representativeness rather
    # than quietly changing the reference or presenting this span as a CI.
    settings=config['model']
    broader=[];ranges=[]
    for row in rows:
        stability=math.exp(-abs(row.get('heart_rate_change_bpm') or 0)/settings['primary_hr_change_weight_scale_bpm']
            -abs(row.get('speed_change_fraction') or 0)/settings['primary_speed_change_weight_scale_fraction'])
        stopped=float(row.get('stopped_time_s') or 0.)
        elapsed=float(row.get('elapsed_time_s') or 120.)
        broader.append(max(1e-8,stability*max(.05,(1-stopped/max(1.,elapsed))**2)))
        half=(elapsed-stopped)/120
        ranges.append((row['moving_minutes_into_run']-half,row['moving_minutes_into_run']+half))
    union=0.;end=-math.inf
    for low,high in sorted(ranges):
        union+=max(0.,high-max(low,end));end=max(end,high)
    broader_pace=speed_to_pace_min_mile(_robust_weighted_speed(stages['weather'],broader))
    if temperature:broader_pace=temperature_pace(broader)
    return dict(version=VERSION,pace_min_mile=pace['weather'],observed_pace_min_mile=pace['observed'],
        sampling_support=dict(retained_minutes=union,
            moving_minute_range=[min(a for a,b in ranges),max(b for a,b in ranges)],
            weighted_moving_minute=float(np.average([r['moving_minutes_into_run'] for r in rows],weights=w)),
            broader_sampling_pace_min_mile=broader_pace,
            broader_sampling_difference_seconds=60*(broader_pace-pace['weather']),
            interpretation='Sampling sensitivity, not an alternative score or confidence interval'),
        comparison_precision=precision,
        stages_min_mile=pace,contributions_seconds_per_mile=contributions,
        estimated_range_95_min_mile=[max(.1,pace['weather']-uncertainty),pace['weather']+uncertainty],
        uncertainty_95_min_mile=uncertainty,
        # Presentation-only metadata for comparing periods under the same fit.
        # The existing run estimates, intervals and calibration are unchanged.
        shared_log_speed_gradient=[*v.tolist(),mean_heat],
        terrain_coverage=float(np.average([f['grade_coverage'] for f in features],weights=w)),
        wind_coverage=float(np.average([float(f['wind_known']) for f in features],weights=w)),
        weather_coverage=float(np.average(weather_known,weights=w)),
        temperature_response=(dict(version=temperature['version'],scale=temperature['scale'],
            mean_hr_adjustment_bpm=float(np.average([v or 0. for v in weather_adjustments],weights=w)),
            delta_seconds_per_mile=contributions['weather'],
            applied_windows=sum(weather_known),original_windows=len(rows),
            missing_context='Original HR retained; no temperature credit') if temperature else None),
        target_hr_bpm=float(config['target_hr']),reference_minutes=float(config['reference_conditions']['within_run_minutes']),
        sampled_hr_bpm=float(np.average([r['average_hr_bpm'] for r in rows],weights=w)),
        gps_coverage=float(np.average([float(r.get('gps_complete_fraction',1.) or 0.) for r in rows],weights=w)),
        basis=('Estimated pace at common HR, reference minute, flat terrain, calm wind and reference air temperature'
            if temperature else 'Estimated pace at common HR, reference minute, flat terrain, calm wind and reference shade heat index'),
        not_corrected=['Fatigue/readiness','Illness','Shoes','Sleep','Unmeasured sun exposure','Technical surface'])


def build_pace_comparability(connection,config):
    """Read-only preview: normalize reliable submaximal running, not Z2 alone.

    Continuous relevance weights replace exact-reference/zone acceptance gates.
    The recorded comparison always uses the identical selected sections.
    """
    sizes=(int(config['model'].get('window_seconds',300)),)
    sets,_=load_model_window_sets(connection,config,sizes,check_elevation=True)
    calibration=fit_comparison_calibration(sets[sizes[0]],config)
    rows,diagnostics=load_overlapping_model_windows(connection,config,
        window_seconds=int(config['model'].get('primary_window_seconds',120)),stride_seconds=int(config['model'].get('primary_window_stride_seconds',60)),check_elevation=True)
    grouped=defaultdict(list)
    for index,row in enumerate(rows):grouped[row['activity_id']].append(index)
    from .progress import prepare_progress_data, _trend_evidence_weight, _workout
    prepared=prepare_progress_data(connection)
    distances={s.activity_id:s.distance_miles for s in prepared.sessions}
    results=[]
    for ordinal,(identity,indexes) in enumerate(grouped.items()):
        chosen=list(indexes)  # The window loader already enforces the reliable submaximal HR domain.
        phase=detect_continuous_quality_phase(connection,config,identity) if rows[indexes[0]].get('workout_type')=='tempo_threshold' else None
        chosen=_pre_quality_indexes(rows,chosen,phase)
        selected=[rows[i] for i in chosen]
        if not selected:continue
        role=prepared.details.get(identity,{}).get('workout') or _workout(selected[0]['workout_type'])
        # Reuse role/health policy, not the old score's absence as an exclusion.
        # The purpose of this reader is to estimate newly supported comparisons too.
        weight=_trend_evidence_weight(selected[0]['health_tag'],role)
        result=compare_run_windows(selected,[_primary_window_weight(r,config) for r in selected],config,calibration,
            precision_seed=20261014+ordinal if weight>0 else None)
        results.append(dict(activity_id=identity,start_time=selected[0]['start_time_utc'],
            continuity=selected[0].get('comparison_continuity'),
            workout_type=role.value,health_tag=selected[0]['health_tag'],distance_miles=distances.get(identity,0.),
            trend_weight=weight,included_in_trend=weight>0,
            terrain_quality=selected[0].get('terrain_quality'),**result))
    return dict(version=VERSION,calibration=calibration,runs=results,diagnostics=diagnostics,
        live_model_changed=False,planner_changed=False,
        status='comparison_estimate_preview',
        assumptions=dict(heat_maximum_fraction_per_c=.006,heat_se_floor=HEAT_MINIMUM_SE,
            missing_terrain_log_sd=.03,missing_wind_log_sd=.02,missing_weather_log_sd=.03,
            terrain_transfer_fraction=.5,wind_transfer_fraction=.5,
            source_basis='Published relationships transferred with explicit engineering priors and bounds; athlete data refine estimates'))


def cached_comparability(connection, config):
    """Share evidence extraction between Progress and the detailed comparison."""
    import json
    from .read_cache import invariant_read
    from .elevation_quality import VERSION as TERRAIN_VERSION, tile_signature
    inputs = {key: config.get(key) for key in (
        'max_hr','resting_hr','target_hr','zones','reference_conditions',
        'model','moving_time','segmentation','elevation')}
    return invariant_read(connection, (VERSION, 'shared-gradient-sampling-v1', TERRAIN_VERSION, json.dumps(inputs, sort_keys=True), tile_signature(connection, config)),
                          lambda: build_pace_comparability(connection, config))


def _comparison_performance(rows, window_days, as_of):
    """Use the existing evidence rules for the displayed, normalized samples.

    This does not replace stored model scores or feed VO2/planner calculations.
    """
    from .analytics import build_fitness_analytics
    from .progress import _change_evidence, _trend
    from .web.schemas import ConfidenceLevel, FitnessTrend
    analysis = build_fitness_analytics(rows, window_days, evaluation_time=as_of)
    current = analysis.get('current')
    change = analysis.get('change_prior_window')
    slope = analysis.get('within_window_trend')
    confidence = ConfidenceLevel.UNAVAILABLE
    direction = FitnessTrend.INSUFFICIENT_DATA
    period = trajectory = None
    if current:
        evidence = analysis.get('comparison_evidence_quality', analysis.get('evidence_quality'))
        confidence = {'good': ConfidenceLevel.HIGH, 'moderate': ConfidenceLevel.MODERATE}.get(evidence, ConfidenceLevel.LOW)
        if change:
            direction = _trend(change['direction'])
            if confidence == ConfidenceLevel.LOW and direction in {FitnessTrend.IMPROVING, FitnessTrend.DECLINING}:
                direction = FitnessTrend.UNCERTAIN
            prior = change['prior']
            period = _change_evidence({**change, 'run_count': current['run_count'],
                'coverage_fraction': min(current['coverage_fraction'], prior['coverage_fraction'])},
                basis=change.get('comparison') or f'preceding {window_days} days',
                confidence=confidence, comparison_run_count=prior['run_count'])
        if slope:
            fresh = analysis.get('days_since_latest_scored_run', float('inf'))
            slope_confidence = (ConfidenceLevel.HIGH if slope['run_count'] >= 6 and slope['coverage_fraction'] >= .5 and fresh <= 7
                else ConfidenceLevel.MODERATE if slope['run_count'] >= 3 and slope['coverage_fraction'] >= .25 and fresh <= 21
                else ConfidenceLevel.LOW)
            trajectory = _change_evidence(slope, basis=slope.get('basis', f'trend within {window_days} days'), confidence=slope_confidence)
    return dict(current=current, confidence=confidence.value, fitness_trend=direction.value,
        last_observation=max((r['start_time_utc'] for r in rows), default=None),
        period_change=period.model_dump(mode='json') if period else None,
        within_window_trend=trajectory.model_dump(mode='json') if trajectory else None)


def comparison_view(report,window_days,as_of):
    """Matched-sample pace presentation, separate from canonical model outputs."""
    from datetime import timedelta
    from .progress import _trend_series
    if as_of.tzinfo is None:raise ValueError('Timezone-aware comparison time required')
    eligible=[r for r in report['runs'] if datetime.fromisoformat(r['start_time'])<=as_of]
    end=as_of;start=end-timedelta(days=window_days)
    def analytic(raw=False):
        return [dict(start_time_utc=r['start_time'],standardized_pace=r['observed_pace_min_mile'] if raw else r['pace_min_mile'],
            uncertainty_95=r['uncertainty_95_min_mile'],trend_weight=r['trend_weight'],distance_miles=r['distance_miles'])
            for r in eligible if r['included_in_trend']]
    def trend(raw=False):
        history = _trend_series(analytic(raw),28)
        predecessor = next((p for p in reversed(history) if p.as_of < start), None)
        return [p.model_dump(mode='json') for p in history
                if start<=p.as_of<=end or p is predecessor]
    selected=[r for r in eligible if start<=datetime.fromisoformat(r['start_time'])<=end]
    points=[dict(activity_id=r['activity_id'],start_time=r['start_time'],
        standardized_pace_min_mile=r['pace_min_mile'],raw_pace_min_mile=r['observed_pace_min_mile'],
        uncertainty_95_min_mile=r['uncertainty_95_min_mile'],distance_miles=r['distance_miles'],
        trend_weight=r['trend_weight'],workout_type=r['workout_type'],
        measurement_quality='device_distance_fallback' if r.get('gps_coverage',1.)<=0 else 'partial_gps_device_distance' if r.get('gps_coverage',1.)<.9 else 'full_sensor',
        observed_average_hr_bpm=r['sampled_hr_bpm'],terrain_quality=r.get('terrain_quality'),contributions_seconds_per_mile=r['contributions_seconds_per_mile']) for r in selected]
    return {**report,'runs':selected,'series':points,'trend':trend(),'recorded_trend':trend(True),
        'performance':_comparison_performance(analytic(),window_days,as_of),
        'as_of':end.isoformat(),'window_days':window_days,'available_windows':[14,28,90,365],
        'all_scored_runs':len(eligible),'trend_runs_in_view':sum(r['included_in_trend'] for r in selected)}


def validate_comparability(rows,config):
    """Earlier-run calibration, later-run first-half anchored stage checks.

    This cannot prove weather attribution or between-run fitness accuracy. Heat
    barely varies within a run; its transfer/association evidence is separate.
    No stage or cutoff is chosen by the result of this check.
    """
    groups=sorted((group for group in _calibration_rows(rows) if len(group)>=4),
                  key=lambda group:(group[0]['start_time_utc'],group[0]['activity_id']))
    errors={key:[] for key in STAGES};between={key:[] for key in STAGES};pair_blocks=[];folds=[]
    for left,right in ((.5,.65),(.65,.8),(.8,1.)):
        start,end=int(len(groups)*left),int(len(groups)*right)
        if not start:continue
        training=[row for group in groups[:start] for row in group]
        calibration=fit_comparison_calibration(training,config)
        folds.append(dict(training_runs=start,held_out_runs=end-start,
            last_training_start=groups[start-1][0]['start_time_utc'],
            first_test_start=groups[start][0]['start_time_utc'] if start<end else None))
        for index in range(start,end):
            group=groups[index]
            previous=groups[index-1]
            gap=(datetime.fromisoformat(group[0]['start_time_utc'])-datetime.fromisoformat(previous[0]['start_time_utc'])).total_seconds()/86400
            if 0<gap<=7:
                first,second=[compare_run_windows(g,[_primary_window_weight(r,config) for r in g],config,calibration) for g in (previous,group)]
                for key in STAGES:between[key].append(abs(first['stages_min_mile'][key]-second['stages_min_mile'][key])*60)
                pair_blocks.append(math.floor(datetime.fromisoformat(group[0]['start_time_utc']).timestamp()/86400/56))
            stages=[compare_run_windows([row],[1.],config,calibration)['stages_min_mile'] for row in group]
            split=len(group)//2
            for key in STAGES:
                level=float(np.median([pace_to_speed_mps(value[key]) for value in stages[:split]]))
                per_run=[]
                for row,value in zip(group[split:],stages[split:]):
                    factor=pace_to_speed_mps(value[key])/pace_to_speed_mps(row['moving_pace_min_mile'])
                    prediction=speed_to_pace_min_mile(level/factor)
                    per_run.append(abs(prediction-row['moving_pace_min_mile'])*60)
                errors[key].append(float(np.mean(per_run)))
    difference=np.array(between['weather'])-np.array(between['observed'])
    unique=sorted(set(pair_blocks));interval=None
    if len(unique)>1:
        rng=np.random.default_rng(20261013)
        blocked=[difference[np.array(pair_blocks)==key] for key in unique]
        draws=[float(np.mean(np.concatenate([blocked[i] for i in choice])))
               for choice in rng.integers(len(blocked),size=(1000,len(blocked)))]
        interval=np.quantile(draws,[.025,.975]).tolist()
    return dict(status='evaluated' if errors['observed'] else 'not_enough_held_out_runs',
        eligible_runs=len(groups),held_out_runs=len(errors['observed']),folds=folds,
        mean_absolute_error_seconds_per_mile={key:float(np.mean(value)) for key,value in errors.items() if value},
        between_run_check=dict(pairs=len(difference),calendar_blocks=len(unique),maximum_gap_days=7,
            mean_absolute_disagreement_seconds_per_mile={key:float(np.mean(value)) for key,value in between.items() if value},
            full_minus_recorded_mean_seconds=float(difference.mean()) if len(difference) else None,
            block_perturbation_range_95_seconds=interval,
            scope='Later consecutive easy/long run comparisons with earlier-run calibration; lower disagreement is not proof of latent-fitness accuracy'),
        scope='Within-run conditional prediction with earlier-run calibration, not a fixed-HR or weather validation veto')
