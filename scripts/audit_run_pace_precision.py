"""Read-only sampling-precision and uncertainty-source inventory.

Block bootstrap describes precision of the sampled run sections, not accuracy
of physiological fitness or representativeness of that run for another day.
No run weights, uncertainty coefficients or estimates are changed.
"""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from unittest.mock import patch

import numpy as np

from run_analysis import pace_comparability as pc, aerobic_progress as ap, analytics
from run_analysis.config import load_config, resolve_project_path
from run_analysis.objective_modeling import _primary_window_weight
from run_analysis.quality_phases import detect_continuous_quality_phase
from run_analysis.read_cache import invariant_snapshot
from scripts.audit_small_aerobic_changes import describe
from scripts.audit_progress_responsiveness import build_from_windows, covariance, analytic


def sources(rows, weights, config, calibration):
    features=[pc._features(r,config) for r in rows]
    w=np.asarray(weights)/sum(weights);beta=np.array(calibration['parameters'])
    gradients=np.array([[-f['hr_log'],-f['time'],f['wind_log']] for f in features])
    gradient=w@gradients
    heat=float(w@np.array([f['heat'] or 0. for f in features]))
    missing=[]
    for row,f in zip(rows,features):
        gps=float(row.get('gps_complete_fraction',1.) or 0.)
        shortfall=max(0.,1.-gps/max(float(config['model']['minimum_gps_coverage']),1e-9))
        distance_sd=float(config['model'].get('device_distance_fallback_uncertainty_seconds_per_mile',30))/60/float(row['moving_pace_min_mile'])/1.96
        missing.append((.03*(1-f['grade_coverage']))**2+(.02 if not f['wind_known'] else 0.)**2
            +(.03 if f['heat'] is None else 0.)**2+(distance_sd*shortfall)**2)
    speeds=np.array([f['speed']*f['grade']*math.exp(beta[2]*f['wind_log']-beta[0]*f['hr_log']
        -beta[1]*f['time']+calibration['heat']['mean']*(f['heat'] or 0.)) for f in features])
    center=math.log(pc._robust_weighted_speed(speeds.tolist(),weights))
    dispersion=1.4826*float(np.median(abs(np.log(speeds)-center)))
    parts=dict(parameter=float(gradient@np.array(calibration['covariance'])@gradient)
        +(heat*calibration['heat']['sd'])**2,missing_conditions=float(w@np.array(missing)),
        terrain_transfer=float(w@np.array([abs(math.log(f['grade'])) for f in features])*.5)**2,
        wind_transfer=float(w@np.array([abs(f['wind_log']) for f in features])*.5)**2,
        whole_run_dispersion=dispersion**2)
    return speeds,parts


def sampling_precision(rows,weights,speeds,draws,seed):
    minutes=np.array([r['moving_minutes_into_run'] for r in rows]);weights=np.array(weights)
    def pace(indices):return pc.speed_to_pace_min_mile(pc._robust_weighted_speed(speeds[indices].tolist(),weights[indices].tolist()))
    result=[]
    for duration in (5,10):
        labels=np.floor((minutes-minutes.min())/duration).astype(int)
        blocks=[np.flatnonzero(labels==label) for label in np.unique(labels)]
        if len(blocks)<3:continue
        rng=np.random.default_rng(seed+duration)
        samples=[pace(np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])) for _ in range(draws)]
        result.append(dict(block_minutes=duration,blocks=len(blocks),sampling_sd_min_mile=float(np.std(samples,ddof=1))))
    return pace(np.arange(len(rows))),result


def interleaved_check(rows,weights,config,calibration,draws,seed):
    """Compare disjoint alternating time blocks; one result per physical run.

    The two halves share calibration, not sampled sections. This tests within-
    run repeatability only, never whether two days have identical fitness.
    """
    cov=np.zeros((4,4));cov[:3,:3]=calibration['covariance'];cov[3,3]=calibration['heat']['sd']**2
    minutes=np.array([r['moving_minutes_into_run'] for r in rows]);out=[]
    for duration in (5,10):
        labels=np.floor((minutes-minutes.min())/duration).astype(int)
        position=(minutes-minutes.min())%duration
        radius=float(config['model'].get('primary_window_seconds',120))/120
        # A window crossing a boundary would contribute the same sensor data
        # to both halves. Remove it before treating the sections as disjoint.
        safe=(position>=radius)&(position<=duration-radius)
        halves=[]
        for parity in (0,1):
            selected=np.flatnonzero((labels%2==parity)&safe)
            if len(np.unique(labels[selected]))<3:break
            rr=[rows[i] for i in selected];ww=[weights[i] for i in selected]
            speeds,parts=sources(rr,ww,config,calibration)
            center,boot=sampling_precision(rr,ww,speeds,draws,seed+duration*10+parity)
            match=[v for v in boot if v['block_minutes']==duration]
            if not match:break
            estimate=pc.compare_run_windows(rr,ww,config,calibration)
            gradient=np.array(estimate['shared_log_speed_gradient'])
            old_independent=max(0.,(estimate['uncertainty_95_min_mile']/1.96/center)**2-float(gradient@cov@gradient))
            conditional=match[0]['sampling_sd_min_mile']**2/center**2
            proposed_independent=conditional+sum(parts[k] for k in ('missing_conditions','terrain_transfer','wind_transfer'))
            halves.append(dict(pace=center,gradient=gradient,old=old_independent,proposed=proposed_independent))
        if len(halves)!=2:continue
        a,b=halves;gradient=a['gradient']-b['gradient']
        shared=max(0.,float(gradient@cov@gradient));difference=math.log(a['pace']/b['pace'])
        variances={name:a[name]+b[name]+shared for name in ('old','proposed')}
        out.append(dict(block_minutes=duration,log_pace_difference=difference,
            **{name:dict(se=math.sqrt(variance),covered_95=abs(difference)<=1.96*math.sqrt(variance),
                absolute_standardized_difference=abs(difference)/max(math.sqrt(variance),1e-15)) for name,variance in variances.items()}))
    return out


def release_check(report,config,end):
    cov=covariance(report)
    def snapshot():
        state=ap.progress_from_comparability(report,as_of=end,history_days=90,config=config)
        fits=pc._comparison_performance(analytic(report['runs']),28,end)
        return dict(aerobic=state,pace=fits)
    with patch.object(ap,'overlapping_variance',side_effect=lambda m,o:m+o), \
         patch.object(analytics,'overlapping_variance',side_effect=lambda m,o:m+o):old=snapshot()
    new=snapshot()
    assert old['aerobic']['current']['pace_min_mile']==new['aerobic']['current']['pace_min_mile']
    assert old['aerobic']['fitness_level']['value']==new['aerobic']['fitness_level']['value']
    assert old['aerobic']['change']['speed_change_percent']==new['aerobic']['change']['speed_change_percent']
    def trend_paces(state):return [(r['as_of'],r['pace_min_mile']) for r in state['aerobic']['history']]
    assert trend_paces(old)==trend_paces(new)
    assert old['pace']['current']['pace_min_mile']==new['pace']['current']['pace_min_mile']
    return dict(shipped_version=new['aerobic']['version'],point_estimates_and_history_unchanged=True,
        old_change=old['aerobic']['change'],new_change=new['aerobic']['change'],
        old_pace_headline=old['pace'],new_pace_headline=new['pace'])


def audit(database,config,end,draws=256,coverage_draws=0):
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        with invariant_snapshot(c):
            size=int(config['model'].get('window_seconds',300))
            sets,_=pc.load_model_window_sets(c,config,(size,),check_elevation=True)
            windows,_=pc.load_overlapping_model_windows(c,config,
                window_seconds=int(config['model']['primary_window_seconds']),
                stride_seconds=int(config['model']['primary_window_stride_seconds']),check_elevation=True)
            sets={k:[r for r in v if datetime.fromisoformat(r['start_time_utc'])<=end] for k,v in sets.items()}
            windows=[r for r in windows if datetime.fromisoformat(r['start_time_utc'])<=end]
            report=build_from_windows(c,config,sets,windows);before=deepcopy(report)
            by_id={r['activity_id']:r for r in report['runs']};groups=defaultdict(list)
            for row in windows:groups[row['activity_id']].append(row)
            results=[]
            for ordinal,(identity,rows) in enumerate(groups.items()):
                ref=by_id.get(identity)
                if ref is None or not ref['included_in_trend']:continue
                phase=detect_continuous_quality_phase(c,config,identity) if rows[0].get('workout_type')=='tempo_threshold' else None
                selected=pc._pre_quality_indexes(rows,list(range(len(rows))),phase)
                rows=[rows[i] for i in selected]
                weights=[_primary_window_weight(r,config) for r in rows]
                speeds,parts=sources(rows,weights,config,report['calibration'])
                center,boot=sampling_precision(rows,weights,speeds,draws,20261014+ordinal)
                assert math.isclose(center,ref['pace_min_mile'],rel_tol=1e-10)
                reconstructed=center*math.expm1(1.96*math.sqrt(sum(parts.values())))
                assert math.isclose(reconstructed,ref['uncertainty_95_min_mile'],rel_tol=1e-9,abs_tol=1e-9)
                total=sum(parts.values())
                results.append(dict(activity_id=identity,start_time=ref['start_time'],windows=len(rows),
                    uncertainty_95_min_mile=ref['uncertainty_95_min_mile'],log_variance_parts=parts,
                    variance_fractions={k:v/total if total else 0. for k,v in parts.items()},
                    sampling_precision=boot,interleaved=interleaved_check(rows,weights,config,report['calibration'],draws,20261014+ordinal)))
            release=release_check(report,config,end)
            # Offline only: retain common errors and replace whole dispersion
            # by the larger supported 5/10-minute sampling variance. Per-run
            # pace is unchanged, but aggregation weights can legitimately move.
            candidate=deepcopy(report);precision={r['activity_id']:r for r in results}
            changed=0
            for run in candidate['runs']:
                item=precision.get(run['activity_id'])
                if item is None or not item['sampling_precision']:continue
                conditional=max(v['sampling_sd_min_mile'] for v in item['sampling_precision'])/run['pace_min_mile']
                common=sum(v for k,v in item['log_variance_parts'].items() if k!='whole_run_dispersion')
                run['uncertainty_95_min_mile']=run['pace_min_mile']*math.expm1(1.96*math.sqrt(common+conditional**2))
                changed+=1
            candidate_state=ap.progress_from_comparability(candidate,as_of=end,history_days=90,config=config)
            original_state=ap.progress_from_comparability(report,as_of=end,history_days=90,config=config)
            candidate_result=dict(deployed=False,runs_with_sampling_estimate=changed,
                current_pace_before=original_state['current']['pace_min_mile'],current_pace_after=candidate_state['current']['pace_min_mile'],
                monthly_before=original_state['change'],monthly_after=candidate_state['change'],
                longer_before=original_state['change_90d'],longer_after=candidate_state['change_90d'])
            candidate_coverage=None
            if coverage_draws:
                from scripts.audit_aerobic_interval_coverage import checks
                candidate_coverage=checks(candidate['runs'],covariance(report),end,draws=coverage_draws)
            assert report==before and c.total_changes==0
    latest=max(datetime.fromisoformat(r['start_time']) for r in results)
    recent=[r for r in results if (latest-datetime.fromisoformat(r['start_time'])).days<90]
    def summarize(values):
        return dict(runs=len(values),fractions={k:describe([r['variance_fractions'][k] for r in values]) for k in parts},
            interleaved={str(duration):dict(runs=len([v for r in values for v in r['interleaved'] if v['block_minutes']==duration]),
                **{name:dict(coverage_95=float(np.mean([v[name]['covered_95'] for r in values for v in r['interleaved'] if v['block_minutes']==duration])),
                    standardized_difference=describe([v[name]['absolute_standardized_difference'] for r in values for v in r['interleaved'] if v['block_minutes']==duration]))
                    for name in ('old','proposed')}) for duration in (5,10)
                    if any(v['block_minutes']==duration for r in values for v in r['interleaved'])},
            bootstrap_sd_to_claimed_sd={str(duration):describe([b['sampling_sd_min_mile']/(r['uncertainty_95_min_mile']/1.96)
                for r in values for b in r['sampling_precision'] if b['block_minutes']==duration])
                for duration in (5,10) if any(b['block_minutes']==duration for r in values for b in r['sampling_precision'])})
    return dict(version='run-pace-precision-audit-v1',source_database_read_only=True,draws=draws,
        run_uncertainty_changed=False,release=release,candidate=candidate_result,candidate_coverage=candidate_coverage,
        all_history=summarize(results),last_90_days=summarize(recent),runs=results,
        assumptions=['Bootstrap is conditional precision of sampled sections, not cross-day fitness accuracy.',
            'Five/ten-minute blocks preserve overlapping neighboring windows; at least three blocks required.',
            'Adjustment coefficients held fixed; their uncertainty remains separately identified, never averaged away.',
            'Interleaved comparisons use disjoint alternating 5/10-minute blocks, at least three per half; durations are not independent cohorts.',
            'Boundary-crossing windows excluded from the interleaved halves; shared within-run physiology can still correlate them.',
            'Run representativeness, drift, HR dynamics and unmeasured conditions are not validated by this bootstrap.',
            'No run uncertainties, run weights, normalization fits or planner settings changed.'])


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coverage-draws',type=int,default=0)
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.now(timezone.utc))
    args=p.parse_args()
    if args.output.exists() or args.as_of.tzinfo is None:p.error('New output path and timezone-aware date required')
    if args.coverage_draws<0 or args.coverage_draws==1:p.error('Coverage draws must be zero or at least two')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    result=audit(resolve_project_path(root,config['paths']['database']),config,args.as_of,coverage_draws=args.coverage_draws)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('all_history','last_90_days')},indent=2))


if __name__=='__main__':main()
