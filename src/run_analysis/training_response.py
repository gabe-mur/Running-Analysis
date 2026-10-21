"""Pure empirical response and consistency helpers shared with offline audits.

No database/files, planner calls, training-policy writes or research artifacts.
Consistency cutoffs and response forms are explicit modeling choices, not
physiological boundaries or a calibrated prediction interval.
"""
from collections import Counter
from datetime import datetime,timedelta
import math

import numpy as np

from .aerobic_performance import LocalPaceTrend

HORIZONS=(14,28,56,84,112,168)
BASE_MODELS={'calendar':('elapsed',),'miles':('miles',),'minutes':('minutes',),
    'miles_calendar':('elapsed','miles'),'log_miles':('log_miles',),
    'log_miles_calendar':('elapsed','log_miles')}
MODELS={name+('' if history is None else f'_recent{history}'):keys
    for history in (None,56,84,112) for name,keys in BASE_MODELS.items()}
HISTORY={name+('' if history is None else f'_recent{history}'):history
    for history in (None,56,84,112) for name in BASE_MODELS}
REGIMES=('consistent','health_affected','gap_affected','reduced_training','insufficient_baseline')
POLICIES={f'gap{gap}_fraction{fraction:.3f}':dict(max_gap_days=gap,minimum_fraction=fraction)
    for gap in (7,10) for fraction in (.5,2/3)}


def features(examples,training,era_start):
    """Cumulative exposure at origin; actual preceding dose at each outcome."""
    output=[]
    for e in examples:
        origin=datetime.fromisoformat(e['origin'])
        prior=sum(r['miles'] for r in training if era_start<=r['time'] and r['available_at']<=origin)
        if prior<=0:continue
        exposures=e['training_by_outcome']
        output.append({**e,'miles':e['delivered_total_miles']/100,
            'minutes':sum(v['weight']*v['minutes']*v['days']/7 for v in exposures)/1000,
            'log_miles':sum(v['weight']*math.log1p(v['total_miles']/prior) for v in exposures),
            'origin_cumulative_miles':prior})
    return output


def benchmark_levels(rows,measurement,gradients,covariance,*,bandwidth_days=28.):
    """Causal prefix anchors; changing bandwidth never includes later runs."""
    dates=[datetime.fromisoformat(r['start_time']) for r in rows]
    times=np.array([(d-dates[0]).total_seconds()/86400 for d in dates])
    y=np.log([r['pace_min_mile'] for r in rows]);levels={}
    for i,r in enumerate(rows):
        curve=LocalPaceTrend(y[:i+1],measurement[:i+1],gradients[:i+1],covariance,times[:i+1],bandwidth_days=bandwidth_days)
        levels[r['activity_id']]=curve.level(times[i])['log_pace']
    return levels


def apply_benchmark_levels(examples,levels):
    """Same physical outcomes/precision/exposure, using supplied prefix levels."""
    output=[]
    for e in examples:
        origin=levels[e['origin_id']]
        actual=sum(v['weight']*levels[i] for i,v in zip(e['outcome_ids'],e['training_by_outcome']))
        output.append({**e,'start_pace':math.exp(origin),'start_level':origin,
            'actual':math.exp(actual),'change':actual-origin})
    return output


def benchmark_examples(examples,rows,measurement,gradients,covariance,*,bandwidth_days=28.):
    """Rebuild the current Progress benchmark separately at each causal prefix.

    Never use today's retrospective curve as a past forecast target: it has
    later information. Neither this signal nor its changes measure true fitness.
    """
    return apply_benchmark_levels(examples,benchmark_levels(rows,measurement,gradients,covariance,bandwidth_days=bandwidth_days))


def fit_response(rows,model,*,as_of=None):
    """Unpenalized zero-intercept response; no forced improvement sign.

    One total vote per origin across horizons, additional repeated-outcome
    downweighting and existing response precision. Overlap is not independence.
    """
    if HISTORY[model] is not None and rows:
        cutoff=(as_of or max(datetime.fromisoformat(r['outcome_end']) for r in rows))-timedelta(days=HISTORY[model])
        rows=[r for r in rows if datetime.fromisoformat(r['origin'])>=cutoff]
    if len({r['origin_id'] for r in rows})<8:return None
    keys=MODELS[model]
    x=np.array([[r[k] for k in keys] for r in rows]);y=np.array([r['change'] for r in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite aligned examples required')
    origins=Counter(r['origin_id'] for r in rows)
    outcomes=Counter(i for r in rows for i in r['outcome_ids'])
    w=np.array([sum(1/outcomes[i] for i in r['outcome_ids'])/len(r['outcome_ids'])/
        origins[r['origin_id']]/r['response_variance'] for r in rows]);w/=w.sum()
    scale=np.sqrt(w@(x*x));scale[scale<1e-12]=1
    z=x/scale;root=np.sqrt(w)
    if np.linalg.matrix_rank(root[:,None]*z)<len(keys):return None
    beta=np.linalg.lstsq(root[:,None]*z,root*y,rcond=None)[0]/scale
    return dict(model=model,coefficients=dict(zip(keys,map(float,beta))),
        examples=len(rows),origins=len(origins),latest_outcome=max(r['outcome_end'] for r in rows),
        condition_number=float(np.linalg.cond(root[:,None]*z)))


def regime(example,training,health,*,era_start,max_gap_days=7,minimum_fraction=.5):
    """Training/health-only context label; never inspect pace or error sign.

    Personal preceding 28-day baseline, complete trailing 14-day dose/frequency
    checks daily, and actual gaps including any crossing the context boundary.
    Future uninterrupted training is a CONDITIONAL scenario, not known upfront.
    """
    origin=datetime.fromisoformat(example['origin'])
    end=max(datetime.fromisoformat(v['at']) for v in example['training_by_outcome'])
    start=origin-timedelta(days=28)
    completed=[r for r in training if r['available_at']<=origin and start<r['available_at']<=origin]
    baseline=sum(r['miles'] for r in completed)
    sufficient=start>=era_start and len(completed)>=4 and baseline>0
    events=[r for r in health if start<=r['at']<=end]
    affected=[r for r in events if r['status'] not in {'normal','little_tired',''}]
    prior_start=max((r['time'] for r in training if r['time']<=start),default=start)
    starts=sorted([prior_start,*[r['time'] for r in training if start<r['time']<=end],end])
    gap=max(((b-a).total_seconds()/86400 for a,b in zip(starts,starts[1:])),default=0.)
    checks=[]
    relevant=[r for r in training if start<r['available_at']<=end]
    for offset in range(14,math.floor((end-start).total_seconds()/86400)+1):
        at=start+timedelta(days=offset)
        window_start=at-timedelta(days=14)
        runs=[r for r in relevant if window_start<r['available_at']<=at]
        checks.append(dict(end=at.isoformat(),volume_fraction=(sum(r['miles'] for r in runs)/(baseline/2) if baseline else 0.),
            frequency_fraction=(len(runs)/(len(completed)/2) if completed else 0.)))
    low_volume=min((r['volume_fraction'] for r in checks),default=0.)
    low_frequency=min((r['frequency_fraction'] for r in checks),default=0.)
    metrics=dict(flags=[] if sufficient else ['insufficient_baseline'],context_start=start.isoformat(),training_end=end.isoformat(),
        baseline_miles_per_7=baseline/4,baseline_runs_per_7=len(completed)/4,
        maximum_gap_days=gap,minimum_volume_fraction=low_volume,minimum_frequency_fraction=low_frequency,
        health_events=[dict(at=r['at'].isoformat(),status=r['status'],activity_id=r.get('activity_id')) for r in affected])
    return classify_consistency(metrics,max_gap_days=max_gap_days,minimum_fraction=minimum_fraction)


def classify_consistency(metrics,*,max_gap_days=7,minimum_fraction=.5):
    """Reuse measured context across policy cutoffs; never rescan history."""
    flags=[]
    sufficient='insufficient_baseline' not in metrics['flags']
    if not sufficient:flags.append('insufficient_baseline')
    if metrics['health_events']:flags.append('health_affected')
    if metrics['maximum_gap_days']>max_gap_days:flags.append('gap_affected')
    if min(metrics['minimum_volume_fraction'],metrics['minimum_frequency_fraction'])<minimum_fraction:flags.append('reduced_training')
    label=next((name for name in REGIMES[1:] if name in flags),'consistent')
    if not sufficient:label='insufficient_baseline'
    return {**metrics,'regime':label,'flags':flags}


def annotate(examples,training,health,**kwargs):
    return [{**e,'consistency':regime(e,training,health,**kwargs)} for e in examples]
