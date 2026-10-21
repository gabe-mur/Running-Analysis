"""Small-signal sensitivity under real cadence and block-resampled noise.

Semisynthetic nulls have constant latent performance, not an assertion that the
athlete was actually constant. Conditional simulation rates are NOT validated
real-world error rates. No coefficients, labels or athlete data are changed.
"""
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
import math
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np

from run_analysis.aerobic_progress import _period,_change,build_aerobic_progress
from run_analysis.config import load_config,resolve_project_path
from run_analysis import pace_comparability as pc
from run_analysis.read_cache import invariant_snapshot
from scripts.audit_progress_responsiveness import (
    at,inject,inject_windows,estimate,response,covariance,thin_state,
    build_from_windows,current_state,analytic,speed_factor,
)


GAINS=(0.,.005,.01,.02)
DURATIONS=(56,84)
SEED=20261014


def describe(values):
    values=np.asarray(values,dtype=float)
    return dict(n=len(values),mean=float(values.mean()),minimum=float(values.min()),
        p05=float(np.quantile(values,.05)),median=float(np.median(values)),
        p95=float(np.quantile(values,.95)),maximum=float(values.max()))


def noise_library(rows,block_days,remove_linear_trend):
    """Donor blocks chosen by dates, never by whether an output is favorable."""
    rows=sorted(rows,key=at)
    times=np.array([(at(r)-at(rows[0])).total_seconds()/86400 for r in rows])
    logs=np.log([r['pace_min_mile'] for r in rows])
    design=np.column_stack([np.ones(len(rows)),times-times.mean()])
    residual=logs-design@np.linalg.lstsq(design,logs,rcond=None)[0] if remove_linear_trend else logs-logs.mean()
    blocks=[];spans=[]
    for i in range(len(rows)):
        indices=[j for j in range(i,len(rows)) if times[j]-times[i]<block_days]
        if len(indices)<2:continue
        blocks.append(residual[indices]);spans.append(times[indices[-1]]-times[i])
    if not blocks:raise ValueError('Not enough nearby source runs for persistent-noise controls')
    # Equal probability per donor block; center the implied observation pool,
    # not each block, so multi-run good/bad spells remain present.
    center=float(np.concatenate(blocks).mean())
    blocks=[b-center for b in blocks]
    return blocks,dict(source_runs=len(rows),source_first=rows[0]['start_time'],source_last=rows[-1]['start_time'],
        block_days=block_days,donor_blocks=len(blocks),donor_run_counts=describe([len(b) for b in blocks]),
        donor_span_days=describe(spans),log_noise_sd=float(np.std(np.concatenate(blocks))),
        residual_method='linear_trend_removed' if remove_linear_trend else 'all_observed_variation_as_noise')


def draw_noise(rows,blocks,rng):
    """Keep donor spells intact; reset at destination breaks over 14 days."""
    noise=[];remaining=[]
    for i,row in enumerate(rows):
        if i and (at(row)-at(rows[i-1])).total_seconds()>14*86400:remaining=[]
        if not remaining:remaining=list(blocks[int(rng.integers(len(blocks)))])
        noise.append(float(remaining.pop(0)))
    return np.array(noise)


def simulate_rows(template,noise,base_pace,onset,gain,duration):
    if len(template)!=len(noise):raise ValueError('One noise value required per physical run')
    return [dict(r,pace_min_mile=base_pace*math.exp(float(n))/speed_factor(at(r),onset,gain,duration))
            for r,n in zip(template,noise)]


def snapshot(rows,end,onset,cov):
    current=_period(rows,end,cov)
    periods=dict(monthly=(end-timedelta(days=28),28),since_onset=(onset,(end-onset).days),
                 ninety_day=(end-timedelta(days=90),90))
    result={}
    for name,(prior_end,days) in periods.items():
        prior=_period(rows,prior_end,cov)
        result[name]=_change(current,prior,cov,comparison_days=days) if prior else None
    return dict(current=current,changes=result)


def summarize_changes(values):
    known=[v for v in values if v is not None]
    if not known:return dict(n=0)
    counts=Counter(f"{v['direction']}:{v['evidence_strength']}" for v in known)
    established=[v for v in known if v['direction']!='provisional']
    n=len(known)
    positive=lambda strength:sum(v['direction']=='improving' and v['evidence_strength'] in strength for v in known)
    negative=lambda strength:sum(v['direction']=='declining' and v['evidence_strength'] in strength for v in known)
    return dict(n=n,established=len(established),change_percent=describe([v['speed_change_percent'] for v in known]),
        estimated_range_width_percent=describe([v['estimated_change_range_95_percent'][1]-v['estimated_change_range_95_percent'][0] for v in known]),
        improving_label_rate=positive(('likely','clear'))/n,clear_improving_label_rate=positive(('clear',))/n,
        improving_label_rate_among_established=positive(('likely','clear'))/len(established) if established else None,
        declining_label_rate=negative(('likely','clear'))/n,clear_declining_label_rate=negative(('clear',))/n,
        inconclusive_or_provisional_rate=sum(v['evidence_strength'] in ('inconclusive','provisional') for v in known)/n,
        label_counts=dict(counts))


def discrimination(positive,null):
    """Threshold-free separation in this simulation, not a real-world test."""
    positive=[v['speed_change_percent'] for v in positive if v is not None]
    null=[v['speed_change_percent'] for v in null if v is not None]
    if not positive or not null:return None
    delta=np.subtract.outer(positive,null)
    return float(np.mean(delta>0)+.5*np.mean(delta==0))


def simulation_checks(rows,cov,end,draws,seed):
    eligible=sorted([r for r in rows if r['included_in_trend'] and r['trend_weight']>0 and at(r)<=end],key=at)
    end=at(eligible[-1])
    donors=[r for r in eligible if at(r)>=end-timedelta(days=180)]
    # These dates suffice for all 28-day windows and the 90-day comparison.
    template=[r for r in eligible if at(r)>end-timedelta(days=118)]
    base_pace=float(np.median([r['pace_min_mile'] for r in donors]))
    cells=[];null_paths=[];libraries=[]
    for noise_kind in (False,True):
        for block_days in (7,14):
            blocks,library=noise_library(donors,block_days,noise_kind);libraries.append(library)
            for duration in DURATIONS:
                onset=end-timedelta(days=duration)
                samples={g:dict(response=[],monthly=[],since_onset=[],ninety_day=[]) for g in GAINS}
                # Restart for each duration so no-change and small-gain cases
                # share the same donor draws. No fresh luck for a larger gain.
                rng=np.random.default_rng(seed)
                null_switches=[]
                for trial in range(draws):
                    noise=draw_noise(template,blocks,rng)
                    null_rows=simulate_rows(template,noise,base_pace,onset,0,duration)
                    baseline=snapshot(null_rows,end,onset,cov)
                    for gain in GAINS:
                        trial_rows=null_rows if gain==0 else simulate_rows(template,noise,base_pace,onset,gain,duration)
                        evaluated=baseline if gain==0 else snapshot(trial_rows,end,onset,cov)
                        cell=samples[gain];cell['response'].append(response(baseline['current'],evaluated['current']))
                        for key in ('monthly','since_onset','ninety_day'):cell[key].append(evaluated['changes'][key])
                    # A fixed, separately identified subset measures chart
                    # ping-pong, not just the final endpoint. No trial selected
                    # because it happened to cross a threshold.
                    if trial<min(64,draws):
                        labels=[]
                        for when in [at(r) for r in template if onset<=at(r)<=end]:
                            change=_change(_period(null_rows,when,cov),_period(null_rows,when-timedelta(days=28),cov),cov)
                            if change is not None:labels.append(change['direction'])
                        signs=[v for v in labels if v in ('improving','declining')]
                        null_switches.append(dict(any_improving='improving' in signs,any_declining='declining' in signs,
                            both_directions='improving' in signs and 'declining' in signs,
                            directional_reversals=sum(a!=b for a,b in zip(signs,signs[1:]))))
                for gain in GAINS:
                    zero=np.zeros(len(template))
                    noiseless=snapshot(simulate_rows(template,zero,base_pace,onset,gain,duration),end,onset,cov)
                    baseline=snapshot(simulate_rows(template,zero,base_pace,onset,0,duration),end,onset,cov)
                    cells.append(dict(residual_method=library['residual_method'],block_days=block_days,ramp_days=duration,
                        injected_endpoint_gain_percent=gain*100,draws=draws,
                        paired_recovered_injection_percent=describe(samples[gain]['response']),
                        noiseless_trailing_response_percent=response(baseline['current'],noiseless['current']),
                        noiseless_comparisons={k:v['speed_change_percent'] if v else None for k,v in noiseless['changes'].items()},
                        discrimination_auc_against_null={k:discrimination(samples[gain][k],samples[0.][k])
                            for k in ('monthly','since_onset','ninety_day')},
                        **{key:summarize_changes(samples[gain][key]) for key in ('monthly','since_onset','ninety_day')}))
                null_paths.append(dict(residual_method=library['residual_method'],block_days=block_days,ramp_days=duration,
                    trials=len(null_switches),any_improving_rate=float(np.mean([v['any_improving'] for v in null_switches])),
                    any_declining_rate=float(np.mean([v['any_declining'] for v in null_switches])),
                    both_directions_rate=float(np.mean([v['both_directions'] for v in null_switches])),
                    directional_reversals=describe([v['directional_reversals'] for v in null_switches])))
    inventory={}
    for name,when in (('current',end),('previous_month',end-timedelta(days=28)),
                      ('ninety_days_earlier',end-timedelta(days=90)),
                      *[(f'onset_{d}_days',end-timedelta(days=d)) for d in DURATIONS]):
        reference=simulate_rows(template,np.zeros(len(template)),base_pace,end,0,84)
        p=_period(reference,when,cov)
        inventory[name]={k:p[k] for k in ('run_count','effective_run_count','coverage_span_days','evidence','evidence_start','evidence_end')} if p else None
    return dict(seed=seed,source_noise_lookback_days=180,base_pace_min_mile=base_pace,comparison_inventory=inventory,
        template_runs=len(template),last_run=end.isoformat(),libraries=libraries,cells=cells,null_chart_paths=null_paths,
        scope='Conditional semisynthetic rates, not calibrated athlete-level sensitivity, specificity or real-world false-positive probabilities.')


def real_history_replays(connection,config,sets,overlapping,report,end):
    rows=report['runs'];latest=max(at(r) for r in rows if r['included_in_trend'])
    base=current_state(report,end);cases=[];cadence=[]
    for duration in DURATIONS:
        onset=latest-timedelta(days=duration)
        for gain in GAINS:
            # Real-history counterfactual: this tests injected increment only;
            # its zero case retains actual unknown progress and is NOT a null.
            changed=inject(rows,onset,gain,duration)
            cadence.append(dict(ramp_days=duration,gain_percent=gain*100,
                paired_response_percent=response(estimate(rows,latest),estimate(changed,latest))))
            fixed=build_from_windows(connection,config,
                {k:inject_windows(v,onset,gain,duration) for k,v in sets.items()},
                inject_windows(overlapping,onset,gain,duration),report['calibration'])
            refit=build_from_windows(connection,config,
                {k:inject_windows(v,onset,gain,duration) for k,v in sets.items()},
                inject_windows(overlapping,onset,gain,duration))
            fs,rs=current_state(fixed,end),current_state(refit,end)
            cases.append(dict(case_kind='recent',gain_percent=gain*100,ramp_days=duration,onset=onset.isoformat(),
                frozen_calibration_response_percent=response(base['current'],fs['current']),
                refitted_calibration_response_percent=response(base['current'],rs['current']),
                baseline=thin_state(base,config),frozen=thin_state(fs,config),refitted=thin_state(rs,config),
                pace_headline_baseline=pc._comparison_performance(analytic(rows),28,end),
                pace_headline_refitted=pc._comparison_performance(analytic(refit['runs']),28,end)))
    return cadence,cases


def audit(database,config,end,draws=256,seed=SEED):
    started=perf_counter()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        with invariant_snapshot(c):
            size=int(config['model'].get('window_seconds',300))
            sets,_=pc.load_model_window_sets(c,config,(size,),check_elevation=True)
            overlapping,_=pc.load_overlapping_model_windows(c,config,
                window_seconds=int(config['model']['primary_window_seconds']),stride_seconds=int(config['model']['primary_window_stride_seconds']),check_elevation=True)
            sets={k:[r for r in v if datetime.fromisoformat(r['start_time_utc'])<=end] for k,v in sets.items()}
            overlapping=[r for r in overlapping if datetime.fromisoformat(r['start_time_utc'])<=end]
            report=build_from_windows(c,config,sets,overlapping);before=deepcopy(report)
            if not any(r['included_in_trend'] for r in report['runs']):raise ValueError('Comparable history required')
            cov=covariance(report)
            simulations=simulation_checks(report['runs'],cov,end,draws,seed)
            cadence,pipeline=real_history_replays(c,config,sets,overlapping,report,end)
            # Validate the fast snapshot against the complete shipped reader.
            eligible=sorted([r for r in report['runs'] if r['included_in_trend']],key=at)
            latest=at(eligible[-1]);onset=latest-timedelta(days=84)
            example=inject(eligible,onset,.01,84)
            fast=snapshot(example,latest,onset,cov)
            full=build_aerobic_progress(example,as_of=latest,covariance=cov,history_days=1)
            assert fast['current']==full['current'] and fast['changes']['monthly']==full['change']
            assert fast['changes']['ninety_day']==full['change_90d']
            assert report==before and c.total_changes==0
    return dict(version='small-aerobic-change-audit-v1',as_of=end.isoformat(),source_database_read_only=True,
        production_model_changed=False,simulation=simulations,real_history_paired_replays=cadence,pipeline_replays=pipeline,
        elapsed_seconds=perf_counter()-started,
        assumptions=[
            'Endpoint changes are 0%, 0.5%, 1% and 2% speed gain, log-linear over 56 or 84 days at fixed HR.',
            'Real-history paired replays retain actual unknown progress; 0% injection there is NOT evidence of no true progress.',
            'Semisynthetic nulls specify constant latent pace; donors reuse the last 180 days of adjusted-pace variation.',
            'Two donor constructions: all observed variation as noise, or residuals after one global log-linear calendar trend.',
            'Neither donor construction identifies true biological noise; seasonal fitness and calibration errors may remain.',
            '7/14-calendar-day moving blocks preserve good/bad spells; destination breaks over 14 days reset spells.',
            'No block-wise centering, rescaling to improve power, outcome-based matching or threshold tuning.',
            'Recorded dates, uncertainty, health weights and shared calibration gradients are fixed; only pace is simulated.',
            'Null endpoint and any-point chart rates are distinct; any-point rates use the first 64 fixed draws, not selected examples.',
            'Monte Carlo draws are resampled versions of one athlete history, not thousands of independent athletes.',
            'Current 28-day level, monthly change, since-onset change and 90-day change answer different questions.',
            'A provisional comparison is not a failed detection; established-only label rates are reported separately.',
            'AUC describes separation of simulated score distributions, not measured real-world diagnostic accuracy.',
            'No production changes, raw-FIT gate validation, new sensor inputs or planner calls.'])


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.now(timezone.utc));p.add_argument('--draws',type=int,default=256)
    args=p.parse_args()
    if args.output.exists():p.error('Choose a new output path')
    if args.as_of.tzinfo is None:p.error('Timezone-aware --as-of required')
    if args.draws<2:p.error('At least two draws required')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    result=audit(resolve_project_path(root,config['paths']['database']),config,args.as_of,args.draws)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cells=len(result['simulation']['cells']),draws_per_cell=args.draws,
        template_runs=result['simulation']['template_runs'],elapsed_seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
