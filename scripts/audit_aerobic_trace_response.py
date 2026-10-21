"""Read-only observed-initial-state HR-response assessment, no exclusion gate."""
from argparse import ArgumentParser
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from run_analysis.elevation_quality import prepare_terrain, tile_directory
from run_analysis.model_windows import _hourly_for_activity
from run_analysis.movement import classify_movement
from run_analysis.pace_comparability import build_pace_comparability
from run_analysis.processing import _load_points
from run_analysis.read_cache import invariant_snapshot
from run_analysis.aerobic_response_evidence import blocks_from_intervals, TAUS, BalanceBlock
from run_analysis.aerobic_hr_timing import statistics, fit, cross_half_loss


def controls(report):
    """Frozen-trace checks, with no source reread or pace-ranking selection."""
    forward = {}; backward = {}; phases = {t: [] for t in (0, 5, 10, 20)}; gains = []
    for raw in report['microblocks']:
        key = raw['activity_id']; blocks = [BalanceBlock(**b) for b in raw['blocks']]
        stats = statistics(blocks); f = fit(stats); forward[key] = f['losses']
        anchor = np.ceil((blocks[-1].start+blocks[-1].seconds)/120)*120
        reverse = [replace(b, start=float(anchor-b.start-b.seconds), first_hr_effort=b.last_hr_effort,
            last_hr_effort=b.first_hr_effort, rise=-b.fall, fall=-b.rise) for b in reversed(blocks)]
        reverse_stats = statistics(reverse, group_ids=[-int((b.start+b.seconds/2)/120) for b in reversed(blocks)])
        # Exact same observation groups in reverse order: the null must match.
        np.testing.assert_allclose(reverse_stats['observed'][::-1], stats['observed'], atol=1e-12)
        b = fit(reverse_stats); backward[key] = b['losses']
        np.testing.assert_allclose(f['losses'][0], b['losses'][0], atol=1e-12)
        by_bin = {}
        for block in blocks:
            by_bin.setdefault(int((block.start+block.seconds/2)/120), []).append(block)
        minute = np.array([sum(b.moving_minute*b.seconds for b in g)/sum(b.seconds for b in g) for g in by_bin.values()])
        for low in phases:
            value = fit(stats, minute >= low)
            if value is not None:
                phases[low].append(value['losses'])
        for multiplier in (.97, 1.03):
            shifted = fit(statistics([replace(b, effort_distance=b.effort_distance*multiplier) for b in blocks]))
            gains.append(float(np.max(abs(f['inverse_reference_speed']/shifted['inverse_reference_speed']-multiplier))))
    def held_out(values):
        result = []
        for key, loss in values.items():
            i = int(np.argmin(np.mean([v for k, v in values.items() if k != key], axis=0)))
            result.append(dict(activity_id=key, tau=TAUS[i], loss_before=float(loss[0]), loss_after=float(loss[i])))
        return dict(mean_loss_before=float(np.mean([v['loss_before'] for v in result])),
            mean_loss_after=float(np.mean([v['loss_after'] for v in result])), runs=result)
    return dict(version='aerobic-trace-response-controls-v1', live_changed=False,
        forward=held_out(forward), reverse_time_control=held_out(backward),
        after_minute_sensitivity={str(t): dict(runs=len(values),
            selected_tau=TAUS[int(np.argmin(np.mean(values, axis=0)))], mean_losses=np.mean(values, axis=0).tolist())
            for t, values in phases.items()},
        maximum_planted_gain_error_fraction=max(gains),
        interpretation='Both directions smooth speed; extra forward benefit supports timing but does not validate a unique physiological lag. Later subsets are diagnostics, not new exclusions.')


def audit(connection, config):
    base = build_pace_comparability(connection, config)
    data = {}
    for r in base['runs']:
        if not r['included_in_trend'] or r['trend_weight'] <= 0:
            continue
        k = r['activity_id']; points = _load_points(connection, k)
        creator = connection.execute('SELECT creator FROM activities WHERE id=?', (k,)).fetchone()[0]
        movement = classify_movement(points, config['moving_time'])
        terrain = prepare_terrain(points, movement.intervals, config, creator, tile_directory(connection, config))
        blocks, coverage = blocks_from_intervals(movement.intervals, config, base['calibration'],
            hourly=_hourly_for_activity(connection, k), terrain_quality=terrain, block_seconds=10.)
        stats = statistics(blocks); fitted = fit(stats)
        data[k] = dict(original=r, stats=stats, fitted=fitted, blocks=blocks, coverage=coverage)
    runs = []
    for k, e in data.items():
        others = [v['fitted']['losses'] for key, v in data.items() if key != k]
        chosen = int(np.argmin(np.mean(others, axis=0)))
        earlier = [v['fitted']['losses'] for v in data.values() if v['original']['start_time'] < e['original']['start_time']]
        prefix = int(np.argmin(np.mean(earlier, axis=0))) if len(earlier) >= 3 else 0
        f = e['fitted']
        runs.append(dict(activity_id=k, date=e['original']['start_time'], workout_type=e['original']['workout_type'],
            canonical_pace=e['original']['pace_min_mile'], coverage=e['coverage'], tau=TAUS[chosen], prefix_tau=TAUS[prefix],
            unaligned_pace=float(f['inverse_reference_speed'][0]*1609.344/60),
            response_pace=float(f['inverse_reference_speed'][chosen]*1609.344/60),
            losses=f['losses'].tolist(), loss_before=float(f['losses'][0]), loss_after=float(f['losses'][chosen]),
            loss_prefix=float(f['losses'][prefix]), valid=bool(f['valid'][chosen]),
            cross_half_before=cross_half_loss(e['stats'], 0), cross_half_after=cross_half_loss(e['stats'], chosen)))
    def summary(rows):
        paired = [r for r in rows if r['cross_half_before'] is not None and r['cross_half_after'] is not None]
        return dict(runs=len(rows), available=sum(r['valid'] for r in rows),
            mean_loss_before=float(np.mean([r['loss_before'] for r in rows])),
            mean_loss_after=float(np.mean([r['loss_after'] for r in rows])),
            mean_loss_prefix=float(np.mean([r['loss_prefix'] for r in rows])),
            improved=sum(r['loss_after'] < r['loss_before'] for r in rows),
            worsened=sum(r['loss_after'] > r['loss_before'] for r in rows),
            cross_half_runs=len(paired), cross_half_loss_before=float(np.mean([r['cross_half_before'] for r in paired])),
            cross_half_loss_after=float(np.mean([r['cross_half_after'] for r in paired])))
    return dict(version='aerobic-trace-response-v1', live_changed=False, source_read_only=True,
        summary=summary(runs), recent=summary([r for r in runs if r['date'] >= '2026-03-01']), runs=runs,
        calibration=base['calibration'],
        microblocks=[dict(activity_id=k, blocks=[asdict(b) for b in e['blocks']]) for k, e in data.items()],
        assumptions=[
            'Observed bout-start HR initializes response; no forced decay toward sleeping resting HR through pauses.',
            '10-second microblocks drive dynamics; all candidates evaluated on identical nonoverlapping 120-second observations.',
            'One run-level pace is fitted from all its readable moving HR, including startup and HR above the former scoring ceiling.',
            'Shared response time chosen without the target run; separate prefix choice retained. HR/condition calibration frozen full-history.',
            'Cross-half check predicts HR with a level estimated from the opposite half; known bout-start HR remains conditional input.',
            'No new run/transition exclusions, no preferred fitness ranking, no fast-run penalty.',
            'Conditional HR prediction and agreement are not independent validation of latent fitness or all normalization extrapolations.'
        ])


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--controls-from', type=Path); args = p.parse_args()
    if args.output.exists():
        p.error('Use a new report path')
    if args.controls_from:
        result = controls(json.loads(args.controls_from.read_text()))
    else:
        config = load_config('config.yaml'); database = resolve_project_path(Path.cwd(), config['paths']['database'])
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            with invariant_snapshot(connection):
                result = audit(connection, config)
                assert connection.total_changes == 0
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    if args.controls_from:
        print(json.dumps({k: {a: b for a, b in v.items() if a != 'runs'} if k in ('forward', 'reverse_time_control') else v
            for k, v in result.items()}, indent=2))
    else:
        print(json.dumps(dict(summary=result['summary'], recent=result['recent'],
            examples=[r for r in result['runs'] if r['activity_id'] in (352, 353, 356, 34, 373, 390)]), indent=2))


if __name__ == '__main__':
    main()
