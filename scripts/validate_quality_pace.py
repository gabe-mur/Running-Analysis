"""Frozen conditional pace comparison; no tuning or claim of HR-load validation."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import json
from math import fsum
from pathlib import Path

from run_analysis.performance_reference import geometric_mean


def summarize(rows):
    if not rows:
        return dict(n=0, model_mae_seconds_mile=None, baseline_mae_seconds_mile=None,
                    improvement_seconds_mile=None, model_bias_seconds_mile=None)
    mean = lambda key: fsum(r[key] for r in rows)/len(rows)
    return dict(n=len(rows), model_mae_seconds_mile=mean('model_absolute_error'),
        baseline_mae_seconds_mile=mean('baseline_absolute_error'),
        improvement_seconds_mile=mean('baseline_absolute_error')-mean('model_absolute_error'),
        model_bias_seconds_mile=mean('model_signed_error'))


def validate(artifact):
    cases = artifact['cases']
    by_key = {c['key']: c for c in cases}
    if len(by_key) != len(cases):
        raise ValueError('Unique case identities required')
    results, skipped = [], []
    for case in cases:
        refs = {r['requested_bout_seconds']: r for r in case['prior_references']}
        bouts = case['assessment']['bouts']
        if not bouts:
            skipped.append(dict(key=case['key'], reason='unresolved_work_alignment'))
            continue
        errors = []
        for bout in bouts:
            ref = refs.get(bout['seconds'])
            if ref is None or ref['speed_mps'] is None or bout['speed_mps'] is None:
                break
            session_means = []
            issue = datetime.fromisoformat(ref['issued_at'])
            if issue != datetime.fromisoformat(case['start']):
                raise ValueError('Expected prior reference frozen at target start')
            for key, _, _ in ref['session_estimates']:
                previous = by_key[key]['assessment']
                run = previous['run']
                if (key == case['key'] or run['source'] != 'observed'
                        or datetime.fromisoformat(run['end']) >= issue
                        or datetime.fromisoformat(run['available_at']) > issue
                        or (issue-datetime.fromisoformat(run['start'])).total_seconds() > ref['policy']['current_days']*86400):
                    raise ValueError('Future/self/stale evidence in forecast comparison')
                speeds = [b['speed_mps'] for b in previous['bouts'] if b['speed_mps'] is not None and b['speed_mps'] > 0]
                session_means.append(geometric_mean(speeds))
            baseline = geometric_mean(session_means)
            actual_pace = 1609.344/bout['speed_mps']
            model_error = 1609.344/ref['speed_mps']['estimate']-actual_pace
            baseline_error = 1609.344/baseline-actual_pace
            errors.append((bout['seconds'], model_error, baseline_error))
        else:
            total = fsum(t for t, _, _ in errors)
            results.append(dict(key=case['key'], start=case['start'], label=case['label'],
                model_absolute_error=fsum(t*abs(e) for t,e,_ in errors)/total,
                baseline_absolute_error=fsum(t*abs(e) for t,_,e in errors)/total,
                model_signed_error=fsum(t*e for t,e,_ in errors)/total))
            continue
        skipped.append(dict(key=case['key'], reason='no_supported_recent_reference_or_pace'))
    results.sort(key=lambda r: (r['start'],r['key']))
    split = len(results)//2
    groups = {label:summarize([r for r in results if r['label']==label])
              for label in sorted({c['label'] for c in cases})}
    overall = summarize(results)
    return dict(version='quality_pace_conditional_comparison_v1', input_version=artifact['version'],
        total_cases=len(cases), eligible=len(results), excluded=dict(Counter(r['reason'] for r in skipped)),
        skipped=skipped, overall=overall, groups=groups,
        chronological_halves=[summarize(results[:split]),summarize(results[split:])], cases=results,
        predeclared_directional_check_passed=(bool(results) and overall['improvement_seconds_mile'] > 0
            and all(g['n'] > 0 and g['improvement_seconds_mile'] > 0 for g in groups.values())),
        check_meaning='Directional diagnostic only: lower MAE overall AND in each quality category; not statistical significance',
        protocol=['Freeze 30-day window and existing duration-similarity rule; do not optimize them here',
            'Baseline: geometric bout means within session, then equal session geometric mean; same preceding contributors',
            'Use actual bout duration as query; pace response held out: conditional hindcast, not genuine issued-prescription forecast',
            'Absolute/signed pace error weighted by work duration within session, equal votes between sessions',
            'Unsupported sessions remain in reported denominator; no recovery, HR-load, benefit or scheduler claim'],
        planner_influence=False)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(json.loads(args.input.read_text()))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in {'cases','skipped','protocol'}},indent=2))


if __name__=='__main__':
    main()
