"""Read-only coherent-curve, influence and same-parameter inference checks."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis import aerobic_progress as ap, pace_comparability as pc
from run_analysis.config import load_config, resolve_project_path
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_coherent_history import CoherentHistory
from scripts.aerobic_joint_profile import GaussianChangeProfile
from scripts.audit_aerobic_shared_history import frame
from scripts.audit_aerobic_three_layers import measurement_inputs
from scripts.audit_progress_responsiveness import at, build_from_windows, covariance
from scripts.audit_small_aerobic_changes import describe


def inference(rows, y, m, g, cov, end, anchor, days, point, *, draws, seed):
    when = end+timedelta(days=anchor)
    ix, observed, groups = frame(rows, when)
    period = 1 if days == 28 else 2
    if 0 not in observed or period not in observed or len(ix) <= len(observed):
        return dict(available=False)
    contrast = np.zeros(len(observed)); contrast[0] = -1; contrast[observed.index(period)] = 1
    times = np.array([(at(rows[i])-when).total_seconds()/86400 for i in ix])
    model = GaussianChangeProfile(y[ix], m[ix], g[ix], cov, groups,
        times=times, future_time=2., mode='nuisance_residual')
    tested = model.bootstrap_test(contrast, point, draws=draws, seed=seed, decision_only=True)
    return dict(available=True, point=point, **{k:tested[k] for k in (
        'estimate', 'covered_95', 'statistic', 'pvalue_bounds', 'evaluated_draws',
        'observed_unconverged_kernels', 'optimizer_warning_draws', 'nesting_refits')})


def audit(database, config, as_of, precision_file, *, draws, scope):
    precision = json.loads(precision_file.read_text())
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            sets, _ = pc.load_model_window_sets(connection, config,
                (int(config['model'].get('window_seconds', 300)),), check_elevation=True)
            windows, _ = pc.load_overlapping_model_windows(connection, config,
                window_seconds=int(config['model']['primary_window_seconds']),
                stride_seconds=int(config['model']['primary_window_stride_seconds']), check_elevation=True)
            sets = {k:[r for r in rr if datetime.fromisoformat(r['start_time_utc']) <= as_of] for k,rr in sets.items()}
            windows = [r for r in windows if datetime.fromisoformat(r['start_time_utc']) <= as_of]
            report = build_from_windows(connection, config, sets, windows); before = deepcopy(report)
            rows = sorted([r for r in report['runs'] if r['included_in_trend'] and r['trend_weight'] > 0 and at(r) <= as_of], key=at)
            end = at(rows[-1]); m, g, _ = measurement_inputs(report, precision, rows); cov = covariance(report)
            y = np.log([r['pace_min_mile'] for r in rows]); t = np.array([(at(r)-end).total_seconds()/86400 for r in rows])
            curve = CoherentHistory(y, m, g, cov, t)
            anchors = t[t >= -90].tolist(); references = [v-days for v in anchors for days in (28,90)]
            series = curve.series(anchors+references)
            lookup = {v['anchor']:v for v in series}
            comparisons = []; max_disagreement = 0.
            for anchor in anchors:
                for days in (28, 90):
                    value = curve.compare(anchor, days)
                    if value['available']:
                        disagreement = abs(value['estimate']-(lookup[anchor-days]['log_pace']-lookup[anchor]['log_pace']))
                        max_disagreement = max(max_disagreement, disagreement)
                    comparisons.append(value)
            assert max_disagreement == 0.
            for width in (28, 90, 180):
                visible = [v for v in series if v['anchor'] >= -width]
                assert all(v == curve.level(v['anchor']) for v in visible)
            prior = anchors[-2]; mask = t <= prior
            old_curve = CoherentHistory(y[mask], m[mask], g[mask], cov, t[mask])
            # Indices are relative to the input array; chronological prefix
            # has the same indices and must reproduce the full result exactly.
            assert old_curve.level(prior) == curve.level(prior)
            current = curve.level(0.); base_changes = {str(days):curve.compare(0., days) for days in (28,90)}
            tests = []
            inference_cases = [(dict(label='historical',anchor=a), rows,y,m,g,t,a) for a in anchors]
            if scope == 'baseline':
                inference_cases = [(dict(label='baseline',anchor=0.),rows,y,m,g,t,0.)]
            influences = []
            for index in np.flatnonzero(t > -28):
                for edit in (None, -.05, -.03, .03, .05):
                    if edit is None:
                        keep = np.arange(len(y)) != index
                        rr = [r for i,r in enumerate(rows) if keep[i]]
                        yy, mm, gg, tt = y[keep],m[keep],g[keep],t[keep]
                        label = dict(label='leave_one_out', activity_id=rows[index]['activity_id'])
                    else:
                        rr, mm, gg, tt = rows,m,g,t
                        yy = y.copy(); yy[index] += math.log1p(edit)
                        label = dict(label='ordinary_single_run', activity_id=rows[index]['activity_id'], pace_change_percent=edit*100)
                    candidate = CoherentHistory(yy,mm,gg,cov,tt)
                    candidate_current = candidate.level(0.)
                    influences.append(dict(**label, current_shift_seconds_per_mile=60*(candidate_current['pace_min_mile']-current['pace_min_mile']),
                        local_optimizer_warnings=candidate_current['unconverged_kernels']))
                    if scope == 'complete':
                        inference_cases.append((label,rr,yy,mm,gg,tt,0.))
            responsiveness = []
            for gain in (-.02, .02):
                altered = y.copy(); altered[t > -28] -= math.log1p(gain)
                changed = CoherentHistory(altered,m,g,cov,t)
                response = current['pace_min_mile']/changed.level(0.)['pace_min_mile']-1
                assert math.isclose(response,gain,abs_tol=1e-9)
                for days in (28,90):
                    assert changed.level(-days) == curve.level(-days)
                responsiveness.append(dict(kind='period_wide', injected_speed_change_percent=100*gain,
                    current_response_percent=100*response, contrast_response_log={str(days):changed.compare(0.,days)['estimate']-base_changes[str(days)]['estimate'] for days in (28,90)}))
                responsiveness[-1]['local_optimizer_warnings'] = changed.level(0.)['unconverged_kernels']
                altered = y-math.log1p(gain)*np.clip((t+84)/84,0,1)
                changed = CoherentHistory(altered,m,g,cov,t)
                responsiveness.append(dict(kind='84_day_ramp',injected_endpoint_speed_change_percent=100*gain,
                    current_response_percent=100*(current['pace_min_mile']/changed.level(0.)['pace_min_mile']-1),
                    contrast_response_percentage_points={str(days):changed.compare(0.,days)['change_percent']-base_changes[str(days)]['change_percent'] for days in (28,90)}))
                responsiveness[-1]['local_optimizer_warnings'] = changed.level(0.)['unconverged_kernels']
            # Confidence calculations receive the same mean contrast target,
            # but never supply or overwrite the curve point. Cases are fixed
            # dependent checks, not Monte Carlo coverage samples.
            for number, (label,rr,yy,mm,gg,tt,anchor) in enumerate(inference_cases):
                candidate = CoherentHistory(yy,mm,gg,cov,tt)
                for days in (28,90):
                    point = candidate.compare(anchor,days)
                    if not point['available'] or point['provisional']:
                        continue
                    tested = inference(rr,yy,mm,gg,cov,end,anchor,days,point['estimate'],draws=draws,seed=20261203+number)
                    assert candidate.compare(anchor,days) == point
                    tests.append(dict(**label,days=days,**tested))
                if (number+1)%10 == 0:
                    print(f'Checked {number+1}/{len(inference_cases)} fixed inference cases',flush=True)
            shipped = ap.progress_from_comparability(report,as_of=as_of,history_days=90,config=config)
            assert report == before and connection.total_changes == 0
    root = Path(__file__).resolve().parents[1]
    ordinary = [abs(v['current_shift_seconds_per_mile']) for v in influences if v['label'] == 'ordinary_single_run']
    deletions = [abs(v['current_shift_seconds_per_mile']) for v in influences if v['label'] == 'leave_one_out']
    return dict(version='aerobic-coherent-history-audit-v1',deployed=False,source_database_read_only=True,
        as_of=as_of.isoformat(),latest_run=rows[-1]['activity_id'],latest_anchor=end.isoformat(),eligible_runs=len(rows),
        observed_curve_anchors=len(anchors),evaluated_curve_points=len(series),current=current,changes=base_changes,
        production_current=shipped['current']['pace_min_mile'],production_monthly=shipped['change']['speed_change_percent'],
        production_ninety_day=shipped['change_90d']['speed_change_percent'],
        consistency=dict(maximum_log_disagreement=max_disagreement,viewport_invariant=True,new_run_does_not_revise_older_frozen_normalization_point=True),
        local_optimizer_warning_points=sum(bool(v.get('unconverged_kernels')) for v in series),
        local_optimizer_warning_sensitivity_cases=sum(bool(v['local_optimizer_warnings']) for v in influences+responsiveness),
        responsiveness=responsiveness,influence_summary=dict(ordinary=describe(ordinary),deletion=describe(deletions)),
        influence= influences,curve=series,comparisons=comparisons,inference=tests,
        inference_summary=dict(tested=len(tests),accepted=sum(v.get('covered_95',False) for v in tests),
            unavailable=sum(not v['available'] for v in tests),
            outer_warnings=sum(bool(v.get('observed_unconverged_kernels')) for v in tests),
            inner_warnings=sum(v.get('optimizer_warning_draws',0) for v in tests),
            rejected=[v for v in tests if v['available'] and not v['covered_95']]),
        scope=scope,bootstrap_budget=draws,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'scripts/aerobic_coherent_history.py','scripts/audit_aerobic_coherent_history.py',
            'scripts/aerobic_shared_state.py','scripts/aerobic_joint_profile.py','scripts/audit_aerobic_three_layers.py')},
        precision_sha256=hashlib.sha256(precision_file.read_bytes()).hexdigest(),
        assumptions=[
            'Each exact historical anchor uses its own trailing 28-day local Gaussian ML log mean with the existing measurement inputs, full shared calibration covariance and unchanged serial kernel grid.',
            'The displayed curve and changes use those identical cached levels; neither viewport nor comparison-specific nuisance fits select another point.',
            'Full joint nuisance-residual inference retains cross-period shared calibration and serial covariance for the same period-mean contrast parameters.',
            'Compatibility is checked, never forced; fixed dependent cases do not imply new coverage calibration or physiological truth.',
            'Single observations remain visible without an invented between-day variance; missing periods remain missing.',
            'Absolute-level variance is conditional on fitted nuisance values, not a newly calibrated 95-percent physiological interval.',
            'Uniform period-wide synthetic gains are exact response checks; ramp endpoints are not identical to trailing-period average improvements.',
            'All source normalization, precision and weights are fixed in injection and influence tests.',
            'No live Progress graph, original VO2, source evidence, correction coefficient, headline or scheduler change.'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--precision',type=Path,default=Path('output/run-pace-precision-current-20261015.json'))
    parser.add_argument('--as-of',type=datetime.fromisoformat,default=datetime.fromisoformat('2026-10-15T04:28:21+00:00'))
    parser.add_argument('--scope',choices=('baseline','complete'),default='complete')
    parser.add_argument('--draws',type=int,default=63)
    args = parser.parse_args()
    if args.output.exists() or args.as_of.tzinfo is None or args.draws < 19:
        parser.error('New output, timezone-aware date and at least 19 bootstrap draws required')
    root = Path(__file__).resolve().parents[1]; config = load_config(root/'config.yaml')
    result = audit(resolve_project_path(root,config['paths']['database']),config,args.as_of,args.precision,draws=args.draws,scope=args.scope)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('changes','consistency','responsiveness','influence_summary','inference_summary','local_optimizer_warning_points')},indent=2))


if __name__ == '__main__':
    main()
