"""Read-only gradual-memory trial with prefix-calibrated forward checks.

Primary candidate fixed in advance: positive marginal-precision weighting,
14-calendar-day half-life. 7/28 days are sensitivity checks, not a search.
The positive 28-day boxcar isolates reweighting from abrupt window turnover.
No app, original observation, normalization policy or scheduler is changed.
"""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from html import escape
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np

from run_analysis import pace_comparability as pc
from run_analysis.aerobic_candidate import measurement_inputs
from run_analysis.aerobic_coherent_history import CoherentHistory
from run_analysis.config import load_config, resolve_project_path
from run_analysis.objective_modeling import _primary_window_weight, _pre_quality_indexes
from run_analysis.privacy import private_file
from run_analysis.quality_phases import detect_continuous_quality_phase
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_decay_history import PrecisionDecayHistory
from scripts.audit_progress_responsiveness import build_from_windows, covariance


METHODS = ('current_28d', 'precision_boxcar_28d', 'decay_14d', 'decay_7d', 'decay_28d')


def at(row):
    return datetime.fromisoformat(row['start_time'])


def inputs(report, end=None):
    rows = sorted([r for r in report['runs'] if r['included_in_trend'] and r['trend_weight'] > 0
        and (end is None or at(r) <= end)], key=at)
    if not rows:
        raise ValueError('At least one eligible aerobic observation required')
    end = at(rows[-1]) if end is None and rows else end
    cov = covariance(report)
    m, g = measurement_inputs(rows, cov)
    return rows, dict(y=np.log([r['pace_min_mile'] for r in rows]), measurement=m, gradients=g,
        covariance=cov, times=np.array([(at(r)-end).total_seconds()/86400 for r in rows]))


def model(data, name):
    if name == 'current_28d':
        return CoherentHistory(**data)
    if name == 'precision_boxcar_28d':
        return PrecisionDecayHistory(**data, boxcar_days=28.)
    if name not in METHODS:
        raise ValueError('Unregistered experiment')
    return PrecisionDecayHistory(**data, half_life_days=float(name.split('_')[1][:-1]))


def series(report):
    rows, data = inputs(report)
    result = {}
    for name in METHODS:
        curve = model(data, name)
        result[name] = [dict(as_of=at(row).isoformat(), activity_id=row['activity_id'],
            **curve.level(float(t))) for row, t in zip(rows, data['times'])]
    return result


def summaries(cases):
    out = {}
    for name in METHODS:
        errors = np.array([c['errors_seconds_per_mile'][name] for c in cases])
        out[name] = dict(n=len(errors), mae_seconds_per_mile=float(np.mean(abs(errors))) if len(errors) else None,
            rmse_seconds_per_mile=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            bias_seconds_per_mile=float(errors.mean()) if len(errors) else None)
    return out


def paired_blocks(cases, *, draws=4000, seed=20261015):
    """Descriptive paired 28-calendar-day block resampling, not fitness truth."""
    if not cases:
        return {}
    labels = np.array([math.floor(datetime.fromisoformat(c['as_of']).timestamp()/86400/28) for c in cases])
    blocks = [np.flatnonzero(labels == key) for key in np.unique(labels)]
    rng = np.random.default_rng(seed)
    samples = [np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]) for _ in range(draws)]
    baseline = np.array([abs(c['errors_seconds_per_mile']['current_28d']) for c in cases])
    out = {}
    for name in METHODS[1:]:
        delta = np.array([abs(c['errors_seconds_per_mile'][name]) for c in cases])-baseline
        distribution = [float(delta[ix].mean()) for ix in samples]
        out[name] = dict(mae_difference_seconds_per_mile=float(delta.mean()),
            descriptive_95_block_range=np.quantile(distribution, [.025,.975]).tolist(), calendar_blocks=len(blocks), draws=draws)
    return out


def forward_check(report, primary, selected, config):
    eligible = sorted([r for r in report['runs'] if r['included_in_trend'] and r['trend_weight'] > 0], key=at)
    cases = []
    # Every eligible later run after four preceding runs. No random split,
    # future-derived pace correction, half-life optimization or target reuse.
    for index in range(4, len(eligible)):
        target = eligible[index]; when = at(target); prior = eligible[:index]
        calibration = pc.fit_comparison_calibration([r for r in primary
            if datetime.fromisoformat(r['start_time_utc']) < when], config)
        normalized = []
        for row in [*prior, target]:
            ordinal, windows = selected[row['activity_id']]
            value = pc.compare_run_windows(windows, [_primary_window_weight(w, config) for w in windows], config,
                calibration, precision_seed=20261014+ordinal)
            normalized.append(dict(row, **value))
        training = dict(calibration=calibration, runs=normalized[:-1])
        _, data = inputs(training)
        predicted = {name:model(data, name).level(0.)['pace_min_mile'] for name in METHODS}
        observed = normalized[-1]['pace_min_mile']
        cases.append(dict(as_of=when.isoformat(), activity_id=target['activity_id'], prior_runs=index,
            gap_days=(when-at(prior[-1])).total_seconds()/86400,
            actual_adjusted_pace_min_mile=observed, predicted_pace_min_mile=predicted,
            errors_seconds_per_mile={name:60*(value-observed) for name,value in predicted.items()},
            target_context=dict(terrain=target['terrain_coverage'], weather=target['weather_coverage'],
                sampled_hr_bpm=target['sampled_hr_bpm']), calibration_runs=calibration['contributing_runs']))
        if (index-3)%10 == 0:
            print(f'Prefix-only forward checks: {index-3}/{len(eligible)-4}', flush=True)
    groups = defaultdict(list)
    for case in cases:
        groups[case['as_of'][:7]].append(case)
    split = len(cases)*2//3
    latest=at(eligible[-1]) if eligible else None
    recent={str(days):[c for c in cases if datetime.fromisoformat(c['as_of'])>=latest-timedelta(days=days)] for days in (90,180,365)}
    return dict(cases=cases, summary=summaries(cases), paired_block_comparison=paired_blocks(cases),
        first_two_thirds=summaries(cases[:split]), final_third=summaries(cases[split:]),
        recent={days:dict(summary=summaries(items),paired_block_comparison=paired_blocks(items)) for days,items in recent.items()},
        monthly={key:summaries(value) for key,value in sorted(groups.items())},
        gaps_over_14_days=summaries([c for c in cases if c['gap_days']>14]),
        scope='Next normalized-run prediction, not independent latent-fitness validation; all candidates frozen before scoring')


def controls(report):
    rows, data = inputs(report)
    base = {name:model(data, name).level(0.) for name in METHODS}
    response = []
    for gain in (-.02,.02):
        for kind in ('step_28d','ramp_84d'):
            altered = deepcopy(data)
            fraction = (data['times'] > -28).astype(float) if kind == 'step_28d' else np.clip((data['times']+84)/84,0,1)
            altered['y'] -= math.log1p(gain)*fraction
            response.append(dict(kind=kind, injected_endpoint_percent=gain*100,
                response_percent={name:100*(base[name]['pace_min_mile']/model(altered,name).level(0.)['pace_min_mile']-1)
                    for name in METHODS}))
    influence = []
    for index in np.flatnonzero(data['times'] > -28):
        for change in (None,-.05,.05):
            altered = deepcopy(data)
            if change is None:
                mask = np.arange(len(rows)) != index
                for key in ('y','measurement','gradients','times'):
                    altered[key] = altered[key][mask]
            else:
                altered['y'][index] += math.log1p(change)
            influence.append(dict(activity_id=rows[index]['activity_id'], change=change,
                shift_seconds_per_mile={name:60*(model(altered,name).level(0.)['pace_min_mile']-base[name]['pace_min_mile'])
                    for name in METHODS}))
    flat = deepcopy(data); flat['y'] = np.full(len(rows),math.log(10.))
    flat_control = {name:model(flat,name).level(0.)['pace_min_mile'] for name in METHODS}
    assert all(math.isclose(value,10.,abs_tol=1e-9) for value in flat_control.values())
    return dict(response=response, single_run_influence=influence, constant_input_pace=flat_control,
        current={name:{k:v for k,v in level.items() if k not in ('weights','indices')} for name,level in base.items()})


def history_diagnostics(histories):
    out = {}
    for name, history in histories.items():
        # Changes across long gaps are not scored as sudden on consecutive days.
        steps = [dict(from_date=a['as_of'],to_date=b['as_of'],
            absolute_seconds_per_mile=60*abs(b['pace_min_mile']-a['pace_min_mile']))
            for a,b in zip(history,history[1:])
            if (datetime.fromisoformat(b['as_of'])-datetime.fromisoformat(a['as_of'])).total_seconds()/86400 <= 14]
        out[name] = dict(maximum_step=max(steps,key=lambda r:r['absolute_seconds_per_mile'],default=None),
            median_step_seconds_per_mile=float(np.median([r['absolute_seconds_per_mile'] for r in steps])) if steps else None,
            august_4_to_6=[dict(as_of=r['as_of'],pace_min_mile=r['pace_min_mile'],run_count=r['run_count'])
                for r in history if r['as_of'].startswith(('2026-08-04','2026-08-06'))])
    return out


def audit(database, config, end):
    started = perf_counter(); before_config = deepcopy(config)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            size = int(config['model'].get('window_seconds',300))
            sets,_ = pc.load_model_window_sets(connection,config,(size,),check_elevation=True)
            raw,_ = pc.load_overlapping_model_windows(connection,config,
                window_seconds=int(config['model']['primary_window_seconds']),
                stride_seconds=int(config['model']['primary_window_stride_seconds']),check_elevation=True)
            sets = {k:[r for r in rr if datetime.fromisoformat(r['start_time_utc'])<=end] for k,rr in sets.items()}
            raw = [r for r in raw if datetime.fromisoformat(r['start_time_utc'])<=end]
            report = build_from_windows(connection,config,sets,raw); before_report = deepcopy(report)
            selected = {}; grouped = defaultdict(list)
            for index,row in enumerate(raw):
                grouped[row['activity_id']].append(index)
            for ordinal,(identity,indexes) in enumerate(grouped.items()):
                phase = detect_continuous_quality_phase(connection,config,identity) if raw[indexes[0]].get('workout_type')=='tempo_threshold' else None
                selected[identity] = (ordinal,[raw[i] for i in _pre_quality_indexes(raw,indexes,phase)])
            print(f'Evidence loaded: {sum(r["included_in_trend"] for r in report["runs"])} eligible runs', flush=True)
            histories = series(report)
            print('Historical curves complete; starting forward checks', flush=True)
            forward = forward_check(report,sets[size],selected,config)
            sensitivity = controls(report)
            assert connection.total_changes == 0 and report == before_report and config == before_config
    return dict(version='aerobic-gradual-memory-trial-v1', deployed=False, source_database_read_only=True,
        as_of=end.isoformat(), elapsed_seconds=perf_counter()-started,
        normalization_sha256=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
        eligible_runs=len(histories['current_28d']), histories=histories,
        historical_diagnostics=history_diagnostics(histories), forward=forward, controls=sensitivity,
        assumptions=[
            'Primary 14-day half-life fixed in advance; 7/28 days are sensitivity checks, not optimized alternatives.',
            'Positive weights are inverse marginal measurement/calibration variance times exponential calendar recency; no day-noise/kernel refit.',
            'Boxcar precision ablation uses the identical positive weighting without decay and the original 28-day cutoff.',
            'Historical display uses current full-cohort calibration like live Progress; it is retrospective, not as-known-then.',
            'Each forward origin recalibrates HR/drift/wind/heat from earlier primary windows only, then normalizes training and held-out runs in that same basis.',
            'Current roles, health tags, terrain repair, reference HR and configuration are held fixed; not a complete historical application replay.',
            'Every eligible run after four priors is scored once; no target is used to fit its adjustments or performance estimate.',
            'Normalized level does not move without new observations; no guessed fitness decay during missing data or proof of adequate readiness.',
            'Shared covariance remains in conditional sensitivity, which omits state/day-noise uncertainty; no 95-percent fitness band is claimed.',
            'Prediction of noisy later performance and injected-response controls cannot prove physiological fitness validity in one athlete.',
            'Paired 28-day block ranges are descriptive and preserve common target cases; no winner is selected by graph smoothness.',
        ])


def plot(result, destination):
    """Dependency-free SVG with true calendar spacing and a shared pace axis."""
    if destination.suffix.lower()!='.svg':
        raise ValueError('Use an SVG figure destination')
    latest=datetime.fromisoformat(result['histories']['current_28d'][-1]['as_of'])
    start=latest-timedelta(days=365);left,right,top,bottom=90,1150,140,480
    names=[('current_28d','Current 28-day estimator','#ae90dc'),
        ('decay_14d','Gradual memory, 14-day half-life','#68cdb0')]
    visible=[r['pace_min_mile'] for name,_,_ in names for r in result['histories'][name]
        if datetime.fromisoformat(r['as_of'])>=start]
    low=math.floor((min(visible)-.1)*2)/2;high=math.ceil((max(visible)+.1)*2)/2
    x=lambda when:left+(when-start).total_seconds()/86400/365*(right-left)
    y=lambda pace:top+(pace-low)/(high-low)*(bottom-top)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="580" viewBox="0 0 1200 580">',
        '<rect width="1200" height="580" fill="#101820"/>',
        '<g font-family="Arial, sans-serif" fill="#e4edf2">',
        '<text x="36" y="40" font-size="23" font-weight="bold">Comparable aerobic pace · gradual-memory trial</text>',
        '<text x="36" y="67" font-size="15" fill="#a5b3be">Same adjusted run evidence. Faster is higher. Gaps are not interpolated.</text>']
    for i,(_,label,color) in enumerate(names):
        lx=36+i*420
        parts.append(f'<path d="M{lx},99h28" stroke="{color}" stroke-width="3"/><text x="{lx+38}" y="104" font-size="14">{escape(label)}</text>')
    for value in np.arange(low,high+.01,.5):
        total=int(round(value*60));label=f'{total//60}:{total%60:02}'
        parts.append(f'<path d="M{left},{y(value):.2f}H{right}" stroke="#2c3943"/><text x="75" y="{y(value)+5:.2f}" text-anchor="end" font-size="14">{label}</text>')
    parts.append('<text x="36" y="132" fill="#a5b3be" font-size="13">min/mi ↑</text>')
    for day in range(0,366,60):
        when=start+timedelta(days=day)
        parts.append(f'<text x="{x(when):.2f}" y="513" text-anchor="middle" font-size="13" fill="#a5b3be">{when:%b %Y}</text>')
    for name,_,color in names:
        path=[];previous=None
        for row in result['histories'][name]:
            when=datetime.fromisoformat(row['as_of'])
            if when<start:continue
            command='M' if previous is None or (when-previous).total_seconds()/86400>28 else 'L'
            path.append(f'{command}{x(when):.2f},{y(row["pace_min_mile"]):.2f}');previous=when
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="{3 if name=="decay_14d" else 2}"/>')
    parts.extend(['<text x="36" y="551" font-size="14" fill="#a5b3be">Offline comparison · no change to live Progress · not a direct measurement of fitness</text>','</g></svg>'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(parts)+'\n');private_file(destination)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path,help='Optional dependency-free SVG comparison')
    parser.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.now(timezone.utc))
    args=parser.parse_args()
    if args.output.exists() or (args.figure and (args.figure.exists() or args.figure.suffix.lower()!='.svg')) or args.as_of.tzinfo is None:
        parser.error('New output destinations, SVG figure extension and a timezone-aware date required')
    root=Path(__file__).resolve().parents[1];config=load_config(root/'config.yaml')
    result=audit(resolve_project_path(root,config['paths']['database']),config,args.as_of)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    if args.figure:
        plot(result,args.figure)
    print(json.dumps(dict(eligible=result['eligible_runs'],elapsed_seconds=result['elapsed_seconds'],forward=result['forward']['summary'],
        paired=result['forward']['paired_block_comparison']),indent=2))


if __name__=='__main__':
    main()
