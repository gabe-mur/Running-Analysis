"""Shared, read-only aerobic comparison evidence for application displays.

Raw HR/pace/load and prescribed workout predictors are not rewritten. This
reader ties the tested terrain, HR timing, sampling and temperature estimates
together without importing research scripts or depending on audit artifacts.
"""
from copy import deepcopy
from datetime import datetime
import json
import math

import numpy as np

from .aerobic_hr_timing import build_hr_aligned_comparability, VERSION as TIMING_VERSION
from .pace_comparability import cached_comparability, compare_run_windows
from .pace_temperature import VERSION as WEATHER_VERSION, fit_temperature
from .read_cache import invariant_read

VERSION = 'shared-aerobic-comparison-v1'


def cached_comparable_evidence(connection, config):
    from .elevation_quality import VERSION as TERRAIN_VERSION, tile_signature
    inputs = {k: config.get(k) for k in ('model', 'moving_time', 'segmentation', 'elevation',
        'max_hr', 'resting_hr', 'target_hr', 'zones', 'reference_conditions')}
    def build():
        from .model_windows import window_preparation_scope
        with window_preparation_scope(connection, config):
            return build_comparable_evidence(connection, cached_comparability(connection, config), config)
    return invariant_read(connection, (VERSION, TIMING_VERSION, WEATHER_VERSION, TERRAIN_VERSION,
        json.dumps(inputs, sort_keys=True, default=str), tile_signature(connection, config)),
        build)


def build_comparable_evidence(connection, base, config):
    contexts = {}
    def collect(identity, rows, weights, context):
        contexts[identity] = (rows, weights, context)
    # Successful aligned rows are rescored with the final temperature model
    # below. Their intermediate bootstrap is never returned or consumed.
    # Missing-response rows retain the original complete base precision.
    aligned = build_hr_aligned_comparability(connection, base, config, window_collector=collect,
        include_precision=False)
    calibration = deepcopy(base['calibration']); calibration['heat']['mean'] = 0.; calibration['heat']['sd'] = 0.
    population = {**calibration, 'temperature_hr': fit_temperature([])}
    meta = {r['id']: dict(r) for r in connection.execute('SELECT id,creator,start_time_local FROM activities')}
    areas = {r['activity_id']: (math.floor(r['latitude']*10), math.floor(r['longitude']*10)) for r in connection.execute('''
        SELECT activity_id,latitude,longitude FROM trackpoints WHERE id IN
        (SELECT min(id) FROM trackpoints WHERE gps_valid=1 GROUP BY activity_id)''')}
    labels = {v: str(i) for i, v in enumerate(sorted(set(areas.values())))}
    observations = [];features_by_id={}
    for original in aligned['runs']:
        identity = original['activity_id']
        if identity not in contexts:
            continue
        rows, weights, context = contexts[identity]
        from .pace_comparability import _features
        features_by_id[identity]=[_features(row,config) for row in rows]
        none = compare_run_windows(rows, weights, config, calibration,_prepared_features=features_by_id[identity])
        air = compare_run_windows(rows, weights, config, population,_prepared_features=features_by_id[identity])
        # Partial unknown weather is retained in the score, not fitted as a
        # known neutral weather example. No strict history count is imposed.
        if air['weather_coverage'] < 1.:
            continue
        local = datetime.fromisoformat(meta[identity]['start_time_local'] or original['start_time'])
        hour = local.hour+local.minute/60
        observations.append(dict(activity_id=identity, start_time_utc=original['start_time'],
            sampled_hr_bpm=none['sampled_hr_bpm'], sampled_minutes=float(np.average([r['moving_minutes_into_run'] for r in rows], weights=weights)),
            duration_minutes=context['coverage']['moving_seconds']/60,
            device=(meta[identity]['creator'] or 'unknown').lower(), area=labels.get(areas.get(identity), 'unknown'),
            hour_sin=math.sin(math.tau*hour/24), hour_cos=math.cos(math.tau*hour/24),
            stop_fraction=context['coverage']['stopped_seconds']/max(1.,context['coverage']['stopped_seconds']+context['coverage']['moving_seconds']),
            previous_7d_miles=rows[0].get('previous_7d_miles'), previous_28d_miles=rows[0].get('previous_28d_miles'),
            pace_no_weather=none['pace_min_mile'], air_burden=math.log(none['pace_min_mile']/air['pace_min_mile'])))
    calibration['temperature_hr'] = fit_temperature(observations)
    calibration['comparison_legacy_heat_sd'] = base['calibration']['heat']['sd']
    results = []
    for ordinal, original in enumerate(aligned['runs']):
        identity = original['activity_id']
        if identity not in contexts:
            # Keep original evidence. No old heat coefficient is applied again.
            results.append({**original, 'shared_log_speed_gradient': [*original['shared_log_speed_gradient'][:3], 0., original['shared_log_speed_gradient'][3]],
                'comparison_pipeline': dict(version=VERSION, available=False,
                reason='original_evidence_retained_without_timing_context')})
            continue
        rows, weights, _ = contexts[identity]
        value = compare_run_windows(rows, weights, config, calibration, precision_seed=20261015+ordinal,
            _prepared_features=features_by_id[identity])
        value['shared_log_speed_gradient'].append(0.)
        results.append({**original, **value, 'sampled_hr_bpm': original['sampled_hr_bpm'],
            'aligned_sampled_hr_bpm': value['sampled_hr_bpm'],
            'comparison_pipeline': dict(version=VERSION, available=True)})
    return {**aligned, 'version': VERSION, 'calibration': calibration, 'runs': results,
        'temperature_response': calibration['temperature_hr'],
        'comparison_pipeline': dict(version=VERSION, humidity_fitted=False, source_rewritten=False,
            recorded_load_changed=False, predictor_changed=False,
            basis='Shared terrain, HR timing, reference HR/time, wind and signed air-temperature comparison',
            weather='Replaces legacy heat correction; no additive humidity or cold-weather bonus',
            sampling='Original run/window set retained, including strong/weak observations; missing context explicit')}


def comparable_progress_data(prepared, report):
    """Adapt shared comparison observations, never the recorded session ledger."""
    from dataclasses import replace
    from .web.schemas import FitnessPoint
    by_id = {r['activity_id']: r for r in report['runs']}
    sessions = {r.activity_id: r for r in prepared.sessions}
    points = []; analytics = []
    old = {p.activity_id: p for p in prepared.points}
    for identity in sorted(set(old) | set(by_id)):
        r = by_id.get(identity)
        if r is None:
            p = old[identity]
            # A stale stored score must not masquerade as this pipeline's score.
            points.append(p.model_copy(update=dict(standardized_pace_min_mile=None,
                included_in_trend=False,trend_weight=0.,context_pace_min_mile=p.raw_pace_min_mile)))
            continue
        session = sessions.get(identity)
        p = FitnessPoint(activity_id=identity, start_time=datetime.fromisoformat(r['start_time']),
            raw_pace_min_mile=r['observed_pace_min_mile'], standardized_pace_min_mile=r['pace_min_mile'],
            uncertainty_95_min_mile=r['uncertainty_95_min_mile'], distance_miles=r.get('distance_miles',0.),
            zone_load=session.zone_load if session else None, workout_type=r['workout_type'],health_tag=r['health_tag'],
            included_in_trend=r['included_in_trend'],trend_weight=r['trend_weight'],
            measurement_quality='shared_comparable_evidence',pace_basis=VERSION,observed_average_hr_bpm=r['sampled_hr_bpm'])
        points.append(p)
        if p.included_in_trend:
            analytics.append(dict(start_time_utc=r['start_time'],standardized_pace=r['pace_min_mile'],
                uncertainty_95=r['uncertainty_95_min_mile'],trend_weight=r['trend_weight'],distance_miles=p.distance_miles))
    points.sort(key=lambda p:(p.start_time,p.activity_id));analytics.sort(key=lambda r:r['start_time_utc'])
    # The old narrow-minute benchmark must not continue feeding a different
    # fitness result. Keep its response field, now backed by shared evidence.
    return replace(prepared,points=points,analytics_rows=analytics,steady_points=points,steady_rows=analytics)


def comparable_coverage(items, report):
    by_id = {r['activity_id']:r for r in report['runs']}
    result = []
    for item in items:
        r=by_id.get(item.activity_id)
        if r is None:
            result.append(item.model_copy(update=dict(standardized_pace_min_mile=None,included_in_trend=False,
                trend_weight=0.,score_status='non_running' if item.score_status=='non_running' else 'context_only',
                reason='Recorded activity retained; no comparable running sections available.')))
            continue
        included=r['included_in_trend']
        result.append(item.model_copy(update=dict(standardized_pace_min_mile=r['pace_min_mile'],included_in_trend=included,
            trend_weight=r['trend_weight'],score_status=('trend_evidence' if r['trend_weight']>=1 else 'reduced_weight') if included else 'context_only',
            reason='Shared HR-timing and conditions-adjusted aerobic evidence.' if included else 'Workout context only; not used in the aerobic trend.')))
    return result


def comparable_run_summary(summary, report):
    """Run Analysis presentation only; workout/load facts remain untouched."""
    from .web.schemas import FitnessObservation, PaceValue, AdjustmentContribution, ConfidenceLevel
    from .progress import _pace_display
    r=next((v for v in report['runs'] if v['activity_id']==summary.activity_id),None)
    if r is None:
        return summary.model_copy(update={'fitness_observation':None})
    def pace(v):
        return PaceValue(minutes_per_mile=v,display=_pace_display(v))
    contributions=[AdjustmentContribution(name=k,minutes_per_mile=v/60,
        evidence='Shared comparable running evidence; conditional model estimate.',confidence=ConfidenceLevel.LOW)
        for k,v in r['contributions_seconds_per_mile'].items()]
    observation=FitnessObservation(activity_id=summary.activity_id,activity_uid=summary.activity_uid,start_time=summary.start_time,
        raw_pace_at_target_hr=pace(r['observed_pace_min_mile']),standardized_pace_at_target_hr=pace(r['pace_min_mile']),
        environmental_adjustment_min_mile=sum(r['contributions_seconds_per_mile'].get(k,0) for k in ('terrain','wind','weather'))/60,
        uncertainty_95_min_mile=r['uncertainty_95_min_mile'],comparable_window_minutes=r.get('sampling_support',{}).get('retained_minutes'),
        contributions=contributions,confidence=ConfidenceLevel.LOW,included_in_trend=r['included_in_trend'],
        trend_weight=r['trend_weight'],pace_basis=VERSION,observed_average_hr_bpm=r['sampled_hr_bpm'])
    return summary.model_copy(update={'fitness_observation':observation})
