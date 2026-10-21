"""Read-only chronological persistence/update test for consistent training.

No May value selects a method, cutoff or coefficient. Current live forecast is
unchanged. Responses retain the previous frozen, causal Progress benchmark;
current observations must reproduce those historical origin benchmarks before
testing updates. Future consistent training is conditional cohort information.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime,timedelta
import json
import math
from pathlib import Path
import sqlite3
from urllib.request import urlopen

import numpy as np

from run_analysis.aerobic_candidate import measurement_inputs
from run_analysis.aerobic_performance import LocalPaceTrend
from run_analysis.config import load_config,resolve_project_path
from run_analysis.pace_temperature import comparison_covariance
from run_analysis.privacy import private_file
from run_analysis.progress import prepare_progress_data
from run_analysis.training_response import POLICIES,classify_consistency,fit_response,regime
from scripts.audit_empirical_training_response import disjoint,score

WINDOWS=(42,56,84)
RECENT_METHODS=tuple(f'{kind}_{days}' for kind in ('run','benchmark','clean_run','clean_benchmark') for days in WINDOWS)
METHODS=('fixed_previous_stretches','expanding_responses','unchanged',*RECENT_METHODS,'selected_update')


def stretches(rows,gap_days=14):
    """Group gaps between consistent origins, never using pace or gain sign.

    Same descriptive 14-day origin-gap grouping used in the optimism diagnosis.
    Report 12/21-day sensitivity, not a newly inferred physiological boundary.
    """
    groups=[]
    for origin in sorted({r['origin'] for r in rows}):
        if not groups or (datetime.fromisoformat(origin)-datetime.fromisoformat(groups[-1][-1])).total_seconds()>gap_days*86400:
            groups.append([])
        groups[-1].append(origin)
    return groups


def recent_rate(observations,origin,days,kind,*,policy=None):
    """Observed recent slope; no taper, sign constraint, or mean reversion.

    At least eight available runs over 28 observed days, matching the existing
    historical trend fit's support policy. Benchmark anchors are serially
    dependent, so this candidate supplies point forecasts, not confidence CIs.
    """
    at=datetime.fromisoformat(origin)
    rows=[r for r in observations if r['available_at']<=at and at-timedelta(days=days)<=r['time']<=at]
    if kind.startswith('clean_'):
        rows=[r for r in rows if classify_consistency(r['past_context'],**(policy or {}))['regime']=='consistent']
    if len(rows)<8:return None
    t=np.array([(r['time']-at).total_seconds()/86400 for r in rows])
    if np.ptp(t)<28:return None
    y=np.array([r['log_pace'] if kind in {'run','clean_run'} else r['benchmark_log_pace'] for r in rows])
    w=1/np.array([r['variance'] for r in rows]);w/=w.sum()
    x=np.column_stack([np.ones(len(t)),t]);root=np.sqrt(w)
    beta=np.linalg.lstsq(root[:,None]*x,root*y,rcond=None)[0]
    return dict(rate_per_day=float(beta[1]),latest_input=max(r['available_at'] for r in rows).isoformat(),
        count=len(rows),span_days=float(np.ptp(t)))


def select_update(prior,current_methods):
    """Select history/target ONLY on completed earlier forecast misses."""
    eligible=[method for method in current_methods if len({r['origin'] for r in prior if r['method']==method})>=8]
    if not eligible:return None
    common=set.intersection(*[{(r['origin'],r['target_id']) for r in prior if r['method']==method} for method in eligible])
    if len({origin for origin,target in common})<8:return None
    losses={}
    for method in eligible:
        chosen=[r for r in prior if r['method']==method and (r['origin'],r['target_id']) in common]
        by_h=[score([r for r in chosen if r['horizon']==h]) for h in sorted({r['horizon'] for r in chosen})]
        losses[method]=float(np.mean([v['log_mae'] for v in by_h]))
    return dict(method=min(losses,key=losses.get),losses=losses,common_cases=len(common),
        latest_selection_outcome=max(r['outcome_end'] for r in prior if (r['origin'],r['target_id']) in common))


def descriptive_rate(rows):
    """Small-episode point summary, NEVER a forecast fit or significance test."""
    origins=Counter(r['origin_id'] for r in rows);outcomes=Counter(i for r in rows for i in r['outcome_ids'])
    w=np.array([sum(1/outcomes[i] for i in r['outcome_ids'])/len(r['outcome_ids'])/
        origins[r['origin_id']]/r['response_variance'] for r in rows])
    x=np.array([r['elapsed']*28 for r in rows]);y=np.array([r['change'] for r in rows])
    return float((w@(x*y))/(w@(x*x)))


def evaluate(rows,observations,gap_days=14,*,policy=None):
    predictions=[];episodes=[]
    for episode_id,group in enumerate(stretches(rows,gap_days),1):
        first=group[0]
        older=[r for r in rows if r['origin']<first and r['outcome_end']<first]
        frozen=fit_response(older,'calendar')
        episode_rows=[r for r in rows if r['origin'] in group]
        realized=descriptive_rate(episode_rows)
        episodes.append(dict(episode=episode_id,first_origin=first,last_origin=group[-1],
            examples=len(episode_rows),origins=len(group),
            previous_rate_per_day=(frozen['coefficients']['elapsed']/28 if frozen else None),
            realized_rate_per_day=realized,
            realized_faster_percent_per_28_days=100*(-math.expm1(realized*28)),
            note='Realized rate is descriptive, never used to predict its own stretch'))
        for origin in group:
            prior=[r for r in rows if r['outcome_end']<origin]
            expanding=fit_response(prior,'calendar')
            rates={'unchanged':dict(rate_per_day=0.,latest_input=None)}
            if frozen:rates['fixed_previous_stretches']=dict(rate_per_day=frozen['coefficients']['elapsed']/28,latest_input=frozen['latest_outcome'])
            if expanding:rates['expanding_responses']=dict(rate_per_day=expanding['coefficients']['elapsed']/28,latest_input=expanding['latest_outcome'])
            for kind in ('run','benchmark','clean_run','clean_benchmark'):
                for days in WINDOWS:
                    fitted=recent_rate(observations,origin,days,kind,policy=policy)
                    if fitted:rates[f'{kind}_{days}']=fitted
            known=[r for r in predictions if r['outcome_end']<origin and r['method'] in RECENT_METHODS]
            selection=select_update(known,[m for m in RECENT_METHODS if m in rates])
            if selection:rates['selected_update']={**rates[selection['method']],**selection}
            for e in (r for r in episode_rows if r['origin']==origin):
                for method,fit in rates.items():
                    forecast=e['start_pace']*math.exp(fit['rate_per_day']*e['elapsed']*28)
                    predictions.append(dict(method=method,episode=episode_id,origin=origin,target_id=e['target_id'],
                        horizon=e['horizon'],outcome_ids=e['outcome_ids'],outcome_end=e['outcome_end'],
                        actual=e['actual'],persistence=e['start_pace'],predicted=forecast,
                        log_error=math.log(forecast/e['actual']),rate_per_day=fit['rate_per_day'],
                        latest_input=fit['latest_input'],selected_method=fit.get('method'),
                        latest_selection_outcome=fit.get('latest_selection_outcome')))
    return predictions,episodes


def matched_comparison(predictions,method,baseline,*,horizon=None,episode=None):
    subset=[r for r in predictions if (horizon is None or r['horizon']==horizon) and (episode is None or r['episode']==episode)]
    lookup={(r['origin'],r['target_id']):r for r in subset if r['method']==baseline}
    chosen=[r for r in subset if r['method']==method and (r['origin'],r['target_id']) in lookup]
    prior=[lookup[r['origin'],r['target_id']] for r in chosen]
    unique=disjoint(chosen)
    return dict(candidate=score(chosen),baseline=score(prior),candidate_disjoint=score(unique),
        baseline_disjoint=score([lookup[r['origin'],r['target_id']] for r in unique]))


def report(predictions):
    if not predictions:return {}
    result={}
    for label,horizon,episode in [('all',None,None),('14_day',14,None),('28_day',28,None),('latest_stretch',None,max(r['episode'] for r in predictions))]:
        result[label]={method:{baseline:matched_comparison(predictions,method,baseline,horizon=horizon,episode=episode)
            for baseline in ('fixed_previous_stretches','expanding_responses','unchanged')}
            for method in RECENT_METHODS+('selected_update',)}
    return result


def observation_snapshot(base,config,root):
    with urlopen('http://127.0.0.1:8000/api/progress-wip?window_days=365&include_exploratory=false&include_legacy_aerobic=false',timeout=150) as f:data=json.load(f)
    comparison=data['pace_comparison'];performance=data['aerobic_performance'];hidden=set(performance['hidden_activity_ids'])
    era=datetime.fromisoformat(base['era_start']);cutoff=datetime.fromisoformat(base['as_of'])
    rows=sorted([dict(r) for r in comparison['runs'] if r['included_in_trend'] and r['trend_weight']>0
        and r['activity_id'] not in hidden and not r.get('manually_excluded')
        and r.get('workout_type') not in {'hike','bike'} and era<=datetime.fromisoformat(r['start_time'])<=cutoff],
        key=lambda r:(r['start_time'],r['activity_id']))
    database=resolve_project_path(root,config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        metadata={r['id']:dict(r) for r in c.execute('SELECT id,start_time_utc,total_elapsed_time_s FROM activities')}
        prepared=prepare_progress_data(c);training=[];health=[]
        for s in prepared.sessions:
            end=s.start_time+timedelta(seconds=metadata[s.activity_id]['total_elapsed_time_s'] or 0)
            if end>cutoff:continue
            tag=prepared.details[s.activity_id]['health_tag']
            training.append(dict(time=s.start_time,available_at=end,miles=s.distance_miles))
            if tag!='normal':health.append(dict(at=s.start_time,status=tag,activity_id=s.activity_id))
        if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_health_reports'").fetchone():
            health.extend(dict(at=datetime.fromisoformat(r['reported_at']),status=r['status'])
                for r in c.execute('SELECT reported_at,status FROM user_health_reports') if datetime.fromisoformat(r['reported_at'])<=cutoff)
        assert c.total_changes==0
    for r in rows:
        m=metadata[r['activity_id']]
        r['available_at']=datetime.fromisoformat(m['start_time_utc'])+timedelta(seconds=m['total_elapsed_time_s'] or 0)
    rows=[r for r in rows if r['available_at']<=cutoff]
    covariance=comparison_covariance(comparison['calibration']);m,g=measurement_inputs(rows,covariance)
    variance=m+np.einsum('ij,jk,ik->i',g,covariance,g)
    times=np.array([(datetime.fromisoformat(r['start_time'])-era).total_seconds()/86400 for r in rows])
    y=np.log([r['pace_min_mile'] for r in rows]);observations=[]
    for i,r in enumerate(rows):
        curve=LocalPaceTrend(y[:i+1],m[:i+1],g[:i+1],covariance,times[:i+1])
        observations.append(dict(id=r['activity_id'],time=datetime.fromisoformat(r['start_time']),
            available_at=r['available_at'],variance=float(variance[i]),log_pace=float(y[i]),
            benchmark_log_pace=curve.level(times[i])['log_pace'],
            past_context=regime(dict(origin=r['available_at'].isoformat(),
                training_by_outcome=[dict(at=r['available_at'].isoformat())]),training,health,era_start=era)))
    levels={r['id']:r['benchmark_log_pace'] for r in observations}
    error=max(abs(levels[e['origin_id']]-e['start_level']) for e in base['examples'])
    if error>1e-9:raise ValueError(f'Current corrections no longer match frozen origin targets: {error}')
    return observations,error


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('output/consistent-training-response-20261020.json'))
    parser.add_argument('--benchmark',type=Path,default=Path('output/empirical-training-response-benchmark-20261020.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    source=json.loads(args.input.read_text());base=json.loads(args.benchmark.read_text())
    observations,snapshot_error=observation_snapshot(base,load_config(root/'config.yaml'),root)
    policies={}
    for key,policy in source['policies'].items():
        rows=[r for r in policy['examples'] if r['consistency']['regime']=='consistent']
        predictions,episodes=evaluate(rows,observations,policy=POLICIES[key])
        policies[key]=dict(episodes=episodes,predictions=predictions,summary=report(predictions),
            stretch_boundary_sensitivity={str(g):stretches(rows,g) for g in (12,14,21)})
    primary=policies[source['primary_policy']]
    result=dict(version='training-rate-persistence-v1',as_of=source['as_of'],primary_policy=source['primary_policy'],
        policies=policies,origin_benchmark_max_log_error=snapshot_error,
        protocol=['Frozen prior consistent-training responses; current observations must reproduce origin benchmarks',
            'Fixed-rate baseline fitted only to outcome blocks finished before the next stretch begins',
            'Expanding rate uses only completed earlier response blocks at each origin',
            'Recent independent-run or causal-benchmark slopes use 42/56/84 days; no imposed rate decay or sign constraint',
            'Health-filtered controls use only the training/health context known by each observation completion, never its future consistency',
            'Window/target selection uses completed earlier forecast errors on common cases, never the May output',
            'No forecasts used to fit their own rate; current completed origin observation is allowed for recent updates',
            'Matched baselines and outcome-disjoint sensitivity; no independence claim'],
        limitations=['Few distinct stretches and overlapping outcome benchmarks',
            'Uninterrupted future training is conditional, not known in advance',
            'Benchmark-anchor slope candidates reuse correlated evidence; no confidence interval inferred',
            'Retrospective HR/weather/grade calibration frozen, not reconstructed historically'],
        live_forecast_changed=False,planner_changed=False,database_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    compact={label:{method:{baseline:dict(candidate_mae=v['candidate'].get('mae_seconds'),
        baseline_mae=v['baseline'].get('mae_seconds'),pairs=v['candidate']['pairs'])
        for baseline,v in comparisons.items()} for method,comparisons in section.items()}
        for label,section in primary['summary'].items()}
    print(json.dumps(dict(episodes=primary['episodes'],summary=compact),indent=2))


if __name__=='__main__':main()
