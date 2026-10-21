"""Paired signal-injection audit; no writes to activity evidence or live models.

This is a sensitivity experiment, not a physiological validation or an estimate
of detection power. Each replay is compared with the SAME real history without
the injection. Recorded dates, scatter, uncertainty, missing runs and contextual
weights are retained. The experiment starts after sensor-window extraction:
it cannot validate movement classification, GPS or HR acceptance gates.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np

from run_analysis.aerobic_progress import build_aerobic_progress
from run_analysis.analytics import _window_estimate
from run_analysis.config import load_config, resolve_project_path
from run_analysis import pace_comparability as comparable
from run_analysis.progress import prepare_progress_data
from run_analysis.read_cache import invariant_snapshot
from run_analysis.vo2_estimation import reserve_fitness_level, vo2_series
from run_analysis.web.schemas import FitnessTrendPoint


DAY = 86400
VARIANTS = ('deployed', 'no_huber', 'equal_uncertainty', 'equal_context', 'equal_weights_no_huber')


def at(row):
    return datetime.fromisoformat(row['start_time'])


def speed_factor(when, onset, gain, ramp_days=0):
    if when < onset:
        return 1.
    fraction = min(1., (when-onset).total_seconds()/DAY/ramp_days) if ramp_days else 1.
    return math.exp(math.log1p(gain)*fraction)


def inject(rows, onset, gain, ramp_days=0):
    # Hold the recorded uncertainty fixed to isolate the response to the
    # changed measurements; recalculated uncertainty is tested separately.
    return [dict(r, pace_min_mile=r['pace_min_mile']/speed_factor(at(r), onset, gain, ramp_days)) for r in rows]


def analytic(rows, variant='deployed'):
    selected = [r for r in rows if r.get('included_in_trend') and r.get('trend_weight', 0)>0]
    return [dict(start_time_utc=r['start_time'], standardized_pace=r['pace_min_mile'],
        uncertainty_95=1. if variant in ('equal_uncertainty', 'equal_weights_no_huber') else r['uncertainty_95_min_mile'],
        trend_weight=1. if variant in ('equal_context', 'equal_weights_no_huber') else r['trend_weight'],
        distance_miles=r.get('distance_miles', 0.)) for r in selected]


def estimate(rows, end, days=28, variant='deployed'):
    return _window_estimate(analytic(rows, variant), end, days,
        robust=variant not in ('no_huber', 'equal_weights_no_huber'))


def response(base, changed):
    return 100*(base['pace_min_mile']/changed['pace_min_mile']-1)


def replay_case(rows, onset, gain, ramp_days=0, variant='deployed'):
    changed = inject(rows, onset, gain, ramp_days)
    dates = sorted({at(r) for r in rows if r.get('included_in_trend') and r['trend_weight']>0
                    and onset<=at(r)<=onset+timedelta(days=56+ramp_days)})
    points = []
    for end in dates:
        base, modified = estimate(rows, end, variant=variant), estimate(changed, end, variant=variant)
        # Compare like-for-like. Existing real improvement is not counted as
        # successful detection of the synthetic change.
        points.append(dict(as_of=end.isoformat(), days_since_onset=(end-onset).total_seconds()/DAY,
            response_percent=response(base, modified), injected_endpoint_percent=100*(speed_factor(end,onset,gain,ramp_days)-1),
            run_count=modified['run_count']))
    def crossing(fraction):
        return next((dict(days=p['days_since_onset'], subsequent_run_count=i+1,
                          as_of=p['as_of']) for i,p in enumerate(points)
                     if p['response_percent']/ (gain*100)>=fraction), None)
    settled = [p for p in points if p['days_since_onset']>=ramp_days+28]
    return dict(onset=onset.isoformat(), gain_percent=gain*100, ramp_days=ramp_days, variant=variant,
        half_response=crossing(.5), ninety_percent_response=crossing(.9),
        settled_response_percent=settled[0]['response_percent'] if settled else None, points=points)


def summarize(cases):
    def distribution(values):
        values=[v for v in values if v is not None]
        return dict(n=len(values), minimum=min(values) if values else None,
            median=median(values) if values else None, maximum=max(values) if values else None)
    return dict(cases=len(cases), half_response_days=distribution([c['half_response']['days'] if c['half_response'] else None for c in cases]),
        ninety_percent_response_days=distribution([c['ninety_percent_response']['days'] if c['ninety_percent_response'] else None for c in cases]),
        settled_response_percent=distribution([c['settled_response_percent'] for c in cases]),
        half_not_reached=sum(c['half_response'] is None for c in cases),
        ninety_percent_not_reached=sum(c['ninety_percent_response'] is None for c in cases))


def covariance(report):
    cov=np.zeros((4,4)); cov[:3,:3]=report['calibration']['covariance']
    cov[3,3]=report['calibration']['heat']['sd']**2
    return cov


def current_state(report, end):
    return build_aerobic_progress(report['runs'], as_of=end, covariance=covariance(report), history_days=1)


def thin_state(state, config):
    return dict(pace_min_mile=state['current']['pace_min_mile'],
        fitness_level=reserve_fitness_level(state['current']['pace_min_mile'], state['current']['uncertainty_95_min_mile'], config),
        monthly_change=state['change'], longer_change=state['change_90d'])


def build_from_windows(connection, config, sets, overlapping, calibration=None):
    # Reuse production aggregation, phase selection and evidence policy, not a
    # locally reimplemented normalization pipeline. Patches exist only here.
    with patch.object(comparable, 'load_model_window_sets', return_value=(sets, {})), \
         patch.object(comparable, 'load_overlapping_model_windows', return_value=(overlapping, {})):
        if calibration is None:
            return comparable.build_pace_comparability(connection,config)
        with patch.object(comparable, 'fit_comparison_calibration', return_value=calibration):
            return comparable.build_pace_comparability(connection,config)


def inject_windows(rows, onset, gain, ramp_days=0):
    return [dict(r, moving_pace_min_mile=r['moving_pace_min_mile']/
        speed_factor(datetime.fromisoformat(r['start_time_utc']),onset,gain,ramp_days)) for r in rows]


def audit(database, config, end):
    started=perf_counter()
    with sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro',uri=True) as connection:
        connection.row_factory=sqlite3.Row
        with invariant_snapshot(connection):
            size=int(config['model'].get('window_seconds',300))
            sets,_=comparable.load_model_window_sets(connection,config,(size,),check_elevation=True)
            overlapping,_=comparable.load_overlapping_model_windows(connection,config,
                window_seconds=int(config['model'].get('primary_window_seconds',120)),
                stride_seconds=int(config['model'].get('primary_window_stride_seconds',60)),check_elevation=True)
            # Pin extraction to the audit date before any fitting.
            sets={key:[r for r in values if datetime.fromisoformat(r['start_time_utc'])<=end] for key,values in sets.items()}
            overlapping=[r for r in overlapping if datetime.fromisoformat(r['start_time_utc'])<=end]
            report=build_from_windows(connection,config,sets,overlapping)
            before=deepcopy(report); rows=report['runs']
            eligible=sorted([r for r in rows if r['included_in_trend']],key=at)
            latest=at(eligible[-1])
            # All recent feasible onset dates, not whichever onset looks best.
            onsets=[at(r) for r in eligible if latest-timedelta(days=90)<=at(r)<=latest-timedelta(days=28)
                and sum(at(r)-timedelta(days=28)<=at(p)<at(r) for p in eligible)>=3]
            if not onsets:
                raise ValueError('Insufficient real history for paired 28-day replay')
            step=[]; summary=[]
            for gain in (.03,.05,.10,-.05):
                for variant in VARIANTS:
                    cases=[replay_case(rows,onset,gain,variant=variant) for onset in onsets]
                    step.extend(cases)
                    summary.append(dict(gain_percent=gain*100,variant=variant,**summarize(cases)))
            # A recent gradual gain exposes unavoidable trailing-window lag.
            ramp=[]
            for duration in (28,56):
                onset=latest-timedelta(days=duration)
                for gain in (.03,.05,.10):
                    case=replay_case(rows,onset,gain,ramp_days=duration)
                    case['last_response_percent']=case['points'][-1]['response_percent']
                    ramp.append(case)
            # Compare frozen adjustment calibration with actual refitting after
            # a coherent within-run speed improvement at unchanged HR/weather.
            pipeline=[]
            pipeline_cases=[('recent',duration,latest-timedelta(days=28 if duration==0 else duration),gain)
                for duration in (0,28,56) for gain in (.03,.05,.10)]
            pipeline_cases.extend(('historical_step',0,onset,.05) for onset in onsets)
            for case_kind,duration,onset,gain in pipeline_cases:
                changed_sets={key:inject_windows(values,onset,gain,duration) for key,values in sets.items()}
                changed_windows=inject_windows(overlapping,onset,gain,duration)
                fixed=build_from_windows(connection,config,changed_sets,changed_windows,report['calibration'])
                refit=build_from_windows(connection,config,changed_sets,changed_windows)
                base_state=current_state(report,end);fixed_state=current_state(fixed,end);refit_state=current_state(refit,end)
                fixed_rows={r['activity_id']:r for r in fixed['runs']}
                refit_rows={r['activity_id']:r for r in refit['runs']}
                pipeline.append(dict(case_kind=case_kind,gain_percent=gain*100,ramp_days=duration,onset=onset.isoformat(),
                    frozen_calibration_response_percent=response(base_state['current'],fixed_state['current']),
                    refitted_calibration_response_percent=response(base_state['current'],refit_state['current']),
                    baseline=thin_state(base_state,config),frozen=thin_state(fixed_state,config),refitted=thin_state(refit_state,config),
                    heat_calibration_baseline=report['calibration']['heat'],heat_calibration_refitted=refit['calibration']['heat'],
                    parameters_baseline=report['calibration']['parameters'],parameters_refitted=refit['calibration']['parameters'],
                    same_run_ids=set(fixed_rows)==set(refit_rows)=={r['activity_id'] for r in rows},
                    same_context_weights=all(fixed_rows[r['activity_id']]['trend_weight']==refit_rows[r['activity_id']]['trend_weight']==r['trend_weight'] for r in rows),
                    pace_headline_baseline=comparable._comparison_performance(analytic(rows),28,end),
                    pace_headline_refitted=comparable._comparison_performance(analytic(refit['runs']),28,end)))
            # Measure the old card's mapping separately: a 5% speed change is
            # not necessarily a 5% change on its pooled VO2 scale.
            prepared=prepare_progress_data(connection)
            old=[]
            base_rows=[r for r in prepared.analytics_rows if datetime.fromisoformat(r['start_time_utc'])<=end]
            def old_value(items):
                value=_window_estimate(items,end,14,minimum_runs=2)
                if not value:return None
                point=FitnessTrendPoint(as_of=end,**{k:value[k] for k in ('pace_min_mile','uncertainty_95_min_mile','run_count')})
                curve=vo2_series([point],config=config,as_of=end)
                return dict(pace_min_mile=value['pace_min_mile'],vo2=curve[0].value_ml_kg_min if curve else None)
            base_old=old_value(base_rows)
            for gain in (.03,.05,.10):
                modified=[dict(r,standardized_pace=r['standardized_pace']/(1+gain)
                    if datetime.fromisoformat(r['start_time_utc'])>end-timedelta(days=14) else r['standardized_pace']) for r in base_rows]
                old.append(dict(gain_percent=gain*100,baseline=base_old,modified=old_value(modified)))
            # One anomalous observation versus a sustained improvement.
            single=[]
            for gain in (.10,.30):
                changed=[dict(r,pace_min_mile=r['pace_min_mile']/(1+gain)) if r['activity_id']==eligible[-1]['activity_id'] else dict(r) for r in rows]
                single.append(dict(single_run_gain_percent=gain*100,response_percent=response(estimate(rows,latest),estimate(changed,latest))))
            quality_only=[dict(r,pace_min_mile=r['pace_min_mile']/1.1)
                if r['workout_type'] in ('intervals','tempo_threshold','race') else dict(r) for r in rows]
            quality_response=response(estimate(rows,latest),estimate(quality_only,latest))
            # Production-oracle check for the point-estimate adapter used above.
            for gain in (0,.03,.05,.10):
                modified=inject(rows,latest-timedelta(days=28),gain)
                oracle=build_aerobic_progress(modified,as_of=end,covariance=covariance(report),history_days=1)
                assert math.isclose(estimate(modified,latest)['pace_min_mile'],oracle['current']['pace_min_mile'],abs_tol=1e-10)
            assert report==before
            assert connection.total_changes==0
    return dict(version='progress-responsiveness-v1',as_of=end.isoformat(),last_eligible_run=latest.isoformat(),
        source_database_read_only=True,production_model_changed=False,physiological_validation=False,
        run_counts=dict(comparable=len(rows),included=len(eligible),onset_dates=len(onsets)),
        step_summary=summary,step_replays=step,gradual_replays=ramp,pipeline_replays=pipeline,
        original_card_mapping=old,single_run_checks=single,
        quality_only_gain_percent=10.,quality_only_response_percent=quality_response,
        elapsed_seconds=perf_counter()-started,
        assumptions=[
            'Injected percentages mean speed improvement at unchanged HR; pace is divided by 1+gain.',
            'Step changes persist; gradual changes are log-linear to the endpoint over 28 or 56 days.',
            'Paired response subtracts the same real history: real pre-existing improvement is not detection.',
            'Post-estimate replays freeze per-run uncertainty; pipeline replays recompute aggregation and uncertainty.',
            'Both pipeline variants use original extracted sections, GPS/movement screens, role and health tags; no raw-FIT/gate sensitivity claim.',
            'Frozen calibration isolates normalization; refitted calibration uses the production HR, drift, wind and calendar-local heat fit.',
            'Equal-weight and non-robust variants are diagnostic controls, not proposed production changes.',
            '28-day level responsiveness is distinct from significance, monthly change and the selected-window headline.',
            'No claim about false-positive rates or physiological effect sizes; recorded residuals are not a known stationary null.',
            'No extra sensor data, new production coefficients, live refit, database update or planner call.'])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.now(timezone.utc))
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output path')
    if args.as_of.tzinfo is None:parser.error('Timezone-aware --as-of required')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    result=audit(resolve_project_path(root,config['paths']['database']),config,args.as_of)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('run_counts','step_summary','single_run_checks','elapsed_seconds')},indent=2))


if __name__=='__main__':main()
