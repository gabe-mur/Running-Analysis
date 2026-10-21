"""Read-only chronological and context-matched checks of pace adjustments.

No coefficient selection, threshold tuning or production changes. Matching is
fixed before looking at pace outcomes. Reduced disagreement is a conditional
repeatability check, not proof that latent fitness was correctly recovered.
"""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
import math
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np

from run_analysis import pace_comparability as pc
from run_analysis.analytics import _window_estimate
from run_analysis.config import load_config,resolve_project_path
from run_analysis.objective_modeling import _primary_window_weight
from run_analysis.read_cache import invariant_snapshot


FACTORS=('terrain','wind','heart_rate','duration','weather')
CALIPERS=dict(grade=.02,headwind=3.,crosswind=3.,hr=3.,minutes=5.,heat=3.)
TARGETS=dict(terrain=('grade',),wind=('headwind','crosswind'),heart_rate=('hr',),duration=('minutes',),weather=('heat',))
MIN_CONTRAST=dict(terrain=.02,wind=3.,heart_rate=4.,duration=8.,weather=3.)


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def omission(rows,weights,config,calibration,factor):
    cal=deepcopy(calibration);samples=rows
    if factor=='terrain':samples=[dict(r,grade_energy_ratio=1.) for r in rows]
    if factor=='wind':cal['parameters'][2]=0.
    if factor=='heart_rate':cal['parameters'][0]=0.
    if factor=='duration':cal['parameters'][1]=0.
    if factor=='weather':cal['heat']['mean']=0.
    return pc.compare_run_windows(samples,weights,config,cal)['pace_min_mile']


def context(rows,weights,config,meta):
    features=[pc._features(r,config) for r in rows]
    weights=np.asarray(weights)/sum(weights)
    def average(values,known):
        if sum(w for w,k in zip(weights,known) if k)<.8:return None
        return float(np.average([v for v,k in zip(values,known) if k],weights=[w for w,k in zip(weights,known) if k]))
    raw=lambda key:average([r.get(key) for r in rows],[finite(r.get(key)) for r in rows])
    return dict(hr=raw('average_hr_bpm'),minutes=raw('moving_minutes_into_run'),
        grade=average([math.log(f['grade']) for f in features],[f['grade_coverage']>=.8 for f in features]),
        headwind=raw('headwind_signed_mph'),crosswind=raw('crosswind_mph'),
        heat=average([f['heat'] for f in features],[finite(f['heat']) for f in features]),
        device=meta['device'],area=meta['area'],role=rows[0]['workout_type'])


def match_pairs(records,factor,max_gap_days):
    candidates=[];target=TARGETS.get(factor,())
    records=sorted(records,key=lambda r:r['start_time'])
    for i,a in enumerate(records):
        ca=a['context']
        for b in records[i+1:]:
            gap=(datetime.fromisoformat(b['start_time'])-datetime.fromisoformat(a['start_time'])).total_seconds()/86400
            if gap>max_gap_days:break
            cb=b['context']
            if gap<=0 or any(ca[k]!=cb[k] or ca[k] is None for k in ('device','area','role')):continue
            if any(not finite(ca.get(k)) or not finite(cb.get(k)) for k in CALIPERS):continue
            delta={k:abs(cb[k]-ca[k]) for k in CALIPERS}
            if any(delta[k]>v for k,v in CALIPERS.items() if k not in target):continue
            if target and max(delta[k] for k in target)<MIN_CONTRAST[factor]:continue
            rank=gap/max_gap_days+sum(delta[k]/v for k,v in CALIPERS.items() if k not in target)
            candidates.append((rank,a['activity_id'],b['activity_id'],a,b,gap))
    used=set();selected=[]
    for _,_,_,a,b,gap in sorted(candidates,key=lambda p:p[:3]):
        if a['activity_id'] in used or b['activity_id'] in used:continue
        used.update((a['activity_id'],b['activity_id']))
        before='observed' if factor is None else factor
        p=lambda r,key:r['full'] if key=='full' else r['observed'] if key=='observed' else r['omitted'][key]
        selected.append(dict(earlier_id=a['activity_id'],later_id=b['activity_id'],earlier=a['start_time'],later=b['start_time'],
            fold=a['fold'],gap_days=gap,before_seconds=abs(p(a,before)-p(b,before))*60,
            after_seconds=abs(a['full']-b['full'])*60,
            context_difference={k:b['context'][k]-a['context'][k] for k in CALIPERS}))
    return dict(candidate_pairs=len(candidates),pairs=selected)


def summary(pairs):
    if not pairs:return dict(pairs=0,calendar_blocks=0)
    before=np.array([p['before_seconds'] for p in pairs]);after=np.array([p['after_seconds'] for p in pairs])
    blocks=np.array([math.floor(datetime.fromisoformat(p['later']).timestamp()/86400/56) for p in pairs])
    unique=np.unique(blocks);interval=None
    if len(unique)>=3:
        rng=np.random.default_rng(20261014);differences=after-before
        grouped=[differences[blocks==b] for b in unique]
        draws=[np.concatenate([grouped[i] for i in choice]).mean() for choice in rng.integers(len(unique),size=(2000,len(unique)))]
        interval=np.quantile(draws,[.025,.975]).tolist()
    return dict(pairs=len(pairs),calendar_blocks=len(unique),before_mae_seconds=float(before.mean()),after_mae_seconds=float(after.mean()),
        after_minus_before_seconds=float((after-before).mean()),median_difference_seconds=float(np.median(after-before)),
        improved_pairs=int(np.sum(after<before)),block_resampling_range_95_seconds=interval)


def chronological_checks(windows,config,metadata,test_windows=None):
    # Calibration uses the established 5-minute windows. Validation must also
    # admit the shorter sections used by the actual display, not impose four
    # 5-minute windows as an accidental run-length requirement.
    test_windows=windows if test_windows is None else test_windows
    groups=sorted((g for g in pc._calibration_rows(test_windows) if len(g)>=2
        and metadata[g[0]['activity_id']]['area'] is not None),key=lambda g:g[0]['start_time_utc'])
    records=[];folds=[]
    for fold,(left,right) in enumerate(((.5,.65),(.65,.8),(.8,1.))):
        start,stop=int(len(groups)*left),int(len(groups)*right)
        if not start or stop<=start:continue
        first_testing=datetime.fromisoformat(groups[start][0]['start_time_utc'])
        training=[r for r in windows if datetime.fromisoformat(r['start_time_utc'])<first_testing]
        calibration=pc.fit_comparison_calibration(training,config)
        last_training=max((r['start_time_utc'] for r in training),default=None)
        folds.append(dict(fold=fold,training_runs=len({r['activity_id'] for r in training}),testing_runs=stop-start,
            last_training=last_training,first_testing=first_testing.isoformat()))
        assert last_training is None or datetime.fromisoformat(last_training)<first_testing
        for g in groups[start:stop]:
            weights=[_primary_window_weight(r,config) for r in g];identity=g[0]['activity_id']
            result=pc.compare_run_windows(g,weights,config,calibration)
            records.append(dict(activity_id=identity,start_time=g[0]['start_time_utc'],fold=fold,
                context=context(g,weights,config,metadata[identity]),observed=result['observed_pace_min_mile'],
                full=result['pace_min_mile'],omitted={f:omission(g,weights,config,calibration,f) for f in FACTORS}))
    checks=[]
    for gap in (7,14):
        for factor in (None,*FACTORS):
            pairs=[];candidates=0
            for fold in range(3):
                matched=match_pairs([r for r in records if r['fold']==fold],factor,gap)
                candidates+=matched['candidate_pairs'];pairs.extend(matched['pairs'])
            checks.append(dict(check='same_conditions' if factor is None else factor,max_gap_days=gap,
                candidate_pairs=candidates,all_history=summary(pairs),
                year_2026=summary([p for p in pairs if p['earlier'].startswith('2026')]),details=pairs))
    return dict(eligible_runs=len(groups),held_out_runs=len(records),folds=folds,matched_checks=checks,
        test_sampling='Production overlapping submaximal windows, at least two per physical run; calibration retains 5-minute windows.',
        omitted='Each factor removed alone, with other production corrections retained.',
        pairing='Same device, coarse area and role; no run reused within a check; both runs held out in the same fold.',
        calipers=CALIPERS,minimum_target_contrast=MIN_CONTRAST,
        missing='At least 80% known window weight for context; missing factors cannot qualify as matched.',
        interpretation='Negative after-minus-before means less disagreement, not established causal or physiological accuracy.')


def display_attribution(report,windows,config,end):
    grouped=defaultdict(list)
    for r in windows:grouped[r['activity_id']].append(r)
    rows=[]
    for r in report['runs']:
        if not r['included_in_trend']:continue
        g=grouped[r['activity_id']];weights=[_primary_window_weight(w,config) for w in g]
        full=pc.compare_run_windows(g,weights,config,report['calibration'])
        assert math.isclose(full['pace_min_mile'],r['pace_min_mile'],abs_tol=1e-10)
        prior=deepcopy(report['calibration']);prior['heat']['mean']=prior['heat']['prior_mean']
        rows.append(dict(r,prior_heat=pc.compare_run_windows(g,weights,config,prior)['pace_min_mile'],
            omitted={f:omission(g,weights,config,report['calibration'],f) for f in FACTORS}))
    checkpoints=[]
    for offset in (90,60,28,0):
        when=end-timedelta(days=offset);values={}
        for key in ('observed','full','population_heat_prior',*FACTORS):
            mapped=[dict(start_time_utc=r['start_time'],standardized_pace=r['observed_pace_min_mile'] if key=='observed'
                else r['pace_min_mile'] if key=='full' else r['prior_heat'] if key=='population_heat_prior' else r['omitted'][key],
                uncertainty_95=r['uncertainty_95_min_mile'],trend_weight=r['trend_weight']) for r in rows]
            estimate=_window_estimate(mapped,when,28)
            values[key]=estimate['pace_min_mile'] if estimate else None
        selected=[r for r in rows if when-timedelta(days=28)<datetime.fromisoformat(r['start_time'])<=when]
        checkpoints.append(dict(as_of=when.isoformat(),runs=len(selected),pace_min_mile=values,
            mean_contributions_seconds={f:float(np.mean([r['contributions_seconds_per_mile'][f] for r in selected])) for f in FACTORS} if selected else {}))
    return dict(checkpoints=checkpoints,
        scope='Retrospective current calibration; 28-day calendar snapshots, not held-out validation or daily readiness.',
        weighting='Identical recorded sections, context and uncertainty weights; residual Huber weights recalculate for each variant.',
        omitted_keys='Factor names mean that factor alone is omitted; full means all corrections and observed means none.')


def heat_sensitivity(windows,config):
    groups=pc._calibration_rows(windows)
    block=lambda g:math.floor(datetime.fromisoformat(g[0]['start_time_utc']).timestamp()/86400/56)
    results=[]
    for omitted in sorted({block(g) for g in groups}):
        kept=[r for g in groups if block(g)!=omitted for r in g]
        result=pc.fit_comparison_calibration(kept,config)['heat']
        results.append(dict(omitted_calendar_block=omitted,remaining_runs=result['runs'],
            coefficient=result['mean'],personal_weight=result.get('personal_weight',0.)))
    return dict(block_days=56,leave_one_block_out=results,
        scope='Retrospective coefficient sensitivity, not held-out outcome validation or evidence that summer improvement was environmental.')


def audit(database,config,end):
    started=perf_counter()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        with invariant_snapshot(c):
            size=int(config['model'].get('window_seconds',300))
            sets,_=pc.load_model_window_sets(c,config,(size,),check_elevation=True)
            windows=[r for r in sets[size] if datetime.fromisoformat(r['start_time_utc'])<=end]
            overlapping,_=pc.load_overlapping_model_windows(c,config,
                window_seconds=int(config['model']['primary_window_seconds']),stride_seconds=int(config['model']['primary_window_stride_seconds']),check_elevation=True)
            overlapping=[r for r in overlapping if datetime.fromisoformat(r['start_time_utc'])<=end]
            metadata={r['id']:dict(device=str(r['creator'] or 'unknown').casefold(),area=None) for r in c.execute('SELECT id,creator FROM activities')}
            for r in c.execute('''SELECT activity_id,latitude,longitude FROM trackpoints
                WHERE id IN (SELECT min(id) FROM trackpoints WHERE gps_valid=1 GROUP BY activity_id)'''):
                if finite(r['latitude']) and finite(r['longitude']):metadata[r['activity_id']]['area']=(math.floor(r['latitude']*10),math.floor(r['longitude']*10))
            # Reuse the complete reader with the pinned extraction, preventing
            # future observations from leaking into the audit's calibration.
            from scripts.audit_progress_responsiveness import build_from_windows
            report=build_from_windows(c,config,{size:windows},overlapping)
            result=dict(version='progress-adjustment-audit-v1',as_of=end.isoformat(),source_database_read_only=True,
                production_coefficients_changed=False,planner_called=False,
                existing_chronological_check=pc.validate_comparability(windows,config),
                matched_chronological=chronological_checks(windows,config,metadata,overlapping),
                display_attribution=display_attribution(report,overlapping,config,end),
                heat_sensitivity=heat_sensitivity(windows,config),
                calibration=report['calibration'],
                limitations=['Training and later test data are separated by run, not by overlapping sensor windows.',
                    'Pair tolerances and 7/14-day sensitivity are fixed engineering choices, not optimal physiological thresholds.',
                    'Within-run prediction cannot establish a heat effect; heat comparisons require different runs.',
                    'Training, sleep, surface, illness and other unmeasured differences can remain within nearby pairs.',
                    'Lower disagreement alone cannot distinguish removed noise from suppressed real short-term change.',
                    'No locality-specific elevation source or external route transmission; coarse areas stay in memory.',
                    'Sparse contrast pairs are inconclusive, not evidence that a correction is worthless.'])
            assert c.total_changes==0
    result['elapsed_seconds']=perf_counter()-started
    return result


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.now(timezone.utc));args=p.parse_args()
    if args.output.exists():p.error('Choose a new output path')
    if args.as_of.tzinfo is None:p.error('Timezone-aware --as-of required')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    result=audit(resolve_project_path(root,config['paths']['database']),config,args.as_of)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(held_out_runs=result['matched_chronological']['held_out_runs'],
        checks=[{k:v for k,v in c.items() if k!='details'} for c in result['matched_chronological']['matched_checks']],
        elapsed_seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
