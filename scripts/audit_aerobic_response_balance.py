"""Read-only full-run, symmetric/asymmetric HR-balance experiment."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from run_analysis.elevation_quality import prepare_terrain, tile_directory
from run_analysis.model_windows import _hourly_for_activity
from run_analysis.movement import classify_movement
from run_analysis.pace_comparability import build_pace_comparability
from run_analysis.privacy import private_file
from run_analysis.processing import _load_points
from run_analysis.read_cache import invariant_snapshot
from run_analysis.aerobic_response_evidence import blocks_from_intervals, candidate_pairs, losses, score, select


def evaluate(evidence, asymmetric=False):
    pairs = candidate_pairs(asymmetric)
    all_losses = {identity: losses(value['blocks'], pairs) for identity, value in evidence.items()}
    runs = []
    for identity, e in evidence.items():
        chosen = select(all_losses, identity, pairs)
        prefix = {k: v for k, v in all_losses.items() if evidence[k]['date'] < e['date']}
        earlier = select(prefix, identity, pairs)
        before = score(e['blocks']); after = score(e['blocks'], *pairs[chosen])
        phase = []
        for low, high in ((0, 5), (5, 10), (10, 20), (20, 40), (40, float('inf'))):
            selected = [b for b in e['blocks'] if low <= b.moving_minute < high]
            phase.append(dict(minutes=[low, high if np.isfinite(high) else None],
                              before=score(selected), after=score(selected, *pairs[chosen])))
        gains = []
        for gain in (.97, 1.03):
            shifted = [replace(b, effort_distance=b.effort_distance*gain) for b in e['blocks']]
            value = score(shifted, *pairs[chosen])
            gains.append(dict(planted_percent=(gain-1)*100,
                recovered_percent=(after['pace_min_mile']/value['pace_min_mile']-1)*100) if after['available'] and value['available'] else None)
        # Non-overlapping early/late evidence: no trackpoint in both halves.
        midpoint = e['coverage']['moving_seconds']/120
        early = [b for b in e['blocks'] if b.moving_minute+b.moving_seconds/120 <= midpoint]
        late = [b for b in e['blocks'] if b.moving_minute-b.moving_seconds/120 >= midpoint]
        halves = []
        for pair in ((0., 0.), pairs[chosen]):
            a, b = score(early, *pair), score(late, *pair)
            halves.append(dict(early=a, late=b, gap_seconds=60*(b['pace_min_mile']-a['pace_min_mile'])
                if a['available'] and b['available'] else None))
        runs.append(dict(activity_id=identity, date=e['date'], canonical_pace=e['canonical_pace'],
            coverage=e['coverage'], before=before, after=after, phases=phase, planted=gains,
            halves=halves,
            tau=pairs[chosen], prefix_tau=pairs[earlier], loss_before=all_losses[identity][0],
            loss_after=all_losses[identity][chosen], loss_prefix=all_losses[identity][earlier]))
    def summary(rows):
        paired = [r for r in rows if all(h['gap_seconds'] is not None for h in r['halves'])]
        return dict(runs=len(rows), available=sum(r['after']['available'] for r in rows),
            lost_moving_seconds=sum(r['coverage']['missing_moving_seconds'] for r in rows),
            retained_moving_seconds=sum(r['coverage']['retained_moving_seconds'] for r in rows),
            mean_loss_before=float(np.mean([r['loss_before'] for r in rows])),
            mean_loss_held_out=float(np.mean([r['loss_after'] for r in rows])),
            mean_loss_prefix=float(np.mean([r['loss_prefix'] for r in rows])),
            selected_taus=dict(Counter(str(r['tau']) for r in rows)),
            improved=sum(r['loss_after'] < r['loss_before'] for r in rows),
            worsened=sum(r['loss_after'] > r['loss_before'] for r in rows),
            half_checks=len(paired),
            mean_absolute_half_gap_seconds=[float(np.mean([abs(r['halves'][i]['gap_seconds']) for r in paired])) for i in (0, 1)],
            phase_median_absolute_response_percent=[float(np.median([abs(r['phases'][i]['after']['response_fraction'])*100
                for r in rows if r['phases'][i]['after']['available']]))
                if any(r['phases'][i]['after']['available'] for r in rows) else None for i in range(5)])
    return dict(summary=summary(runs), recent=summary([r for r in runs if r['date'] >= '2026-03-01']), runs=runs)


def audit(connection, config):
    original = build_pace_comparability(connection, config)
    eligible = [r for r in original['runs'] if r['included_in_trend'] and r['trend_weight'] > 0]
    resolution = {n: {} for n in (60., 120., 240.)}
    for r in eligible:
        identity = r['activity_id']
        creator = connection.execute('SELECT creator FROM activities WHERE id=?', (identity,)).fetchone()[0]
        points = _load_points(connection, identity)
        movement = classify_movement(points, config['moving_time'])
        terrain = prepare_terrain(points, movement.intervals, config, creator, tile_directory(connection, config))
        hourly = _hourly_for_activity(connection, identity)
        for n in resolution:
            blocks, coverage = blocks_from_intervals(movement.intervals, config, original['calibration'],
                hourly=hourly, terrain_quality=terrain, block_seconds=n)
            resolution[n][identity] = dict(blocks=blocks, coverage=coverage, date=r['start_time'],
                canonical_pace=r['pace_min_mile'])
    evidence = resolution[120.]
    trials = {str(n): dict(symmetric=evaluate(es), asymmetric=evaluate(es, True)) for n, es in resolution.items()}
    stability = {}
    for method in ('symmetric', 'asymmetric'):
        values = {key: {r['activity_id']: r for r in trial[method]['runs']} for key, trial in trials.items()}
        spans = [60*(max(values[key][k]['after']['pace_min_mile'] for key in values)
            -min(values[key][k]['after']['pace_min_mile'] for key in values)) for k in evidence
            if all(values[key][k]['after']['available'] for key in values)]
        stability[method] = dict(runs=len(spans), median_score_span_seconds=float(np.median(spans)),
                                maximum_score_span_seconds=max(spans))
    result = dict(version='aerobic-response-balance-v2', live_changed=False, source_read_only=True,
        original_runs=len(eligible), calibration=original['calibration'],
        symmetric=trials['120.0']['symmetric'], asymmetric=trials['120.0']['asymmetric'],
        resolutions={key: {method: trial[method]['summary'] for method in trial} for key, trial in trials.items()},
        aggregation_stability=stability,
        evidence=[dict(activity_id=k, blocks=[asdict(b) for b in e['blocks']], coverage=e['coverage']) for k, e in evidence.items()],
        assumptions=[
            'Integrated first-order balance in the existing HR-reserve effort proxy, not measured oxygen consumption.',
            'All readable moving data retained: startup, short slowdowns, restarts and HR above the former score ceiling.',
            'Unknown recording intervals are not interpolated; each readable bout uses measured HR endpoints.',
            'Stopped sections do not imply HR must decay toward sleeping resting HR.',
            'Two-minute primary integration; one/four-minute aggregation sensitivity retained.',
            'Symmetric and separate rise/fall response times selected using other runs; earlier-run-only selection also retained.',
            'Shared HR/weather/drift calibration frozen at current full-history values; not a prospective forecast.',
            'No desired pace ranking, no new fast-run penalty, no score-based trimming or stability gate.',
            'Within-run balance consistency is not independently measured fitness accuracy.',
            'Above-domain HR and early-run HR mapping are approximations tested here, not silently treated as validated.'
        ])
    return result


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists():
        p.error('Use a new report path')
    c = load_config('config.yaml'); database = resolve_project_path(Path.cwd(), c['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            result = audit(connection, c)
            assert connection.total_changes == 0
    # Create private at the outset, not only after sensitive data is written.
    import os
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    private_file(args.output)
    print(json.dumps({key: {k: v for k, v in result[key].items() if k != 'runs'} for key in ('symmetric', 'asymmetric')}, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k not in ('phases', 'planted')} for r in result['asymmetric']['runs']
                      if r['activity_id'] in (352, 353, 356, 34, 373, 390)], indent=2))


if __name__ == '__main__':
    main()
