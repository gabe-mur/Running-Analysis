"""Attribute frozen rollover distance choices without searching or publishing.

Later prescriptions are hindsight witnesses, NEVER causal search inputs. All
variants use the common explicit dates; no newly exposed endpoint work is added.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, timedelta
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from run_analysis.adaptive.program_objective_trial import AXES
from run_analysis.adaptive.soft_horizon_trial import day_weight
from run_analysis.program_delivery import _target_total


def additive_parts(evaluation):
    """Exclusive terms: linear volume and race work already live in totals."""
    parts = evaluation.parts
    boundary = parts['continuation_boundary']
    volume = boundary['volume_cost']
    linear = parts.get('volume_benefit', {}).get('added_volume_cost', 0.)
    terms = dict(explicit_fatigue=parts['soft_horizon']['demand'],
        continuation_cross_fatigue=boundary['cross_demand_cost'],
        race_fatigue=parts.get('race_event', {}).get('cost', 0.),
        squared_rolling_volume=volume, linear_rolling_volume=linear,
        purpose_delivery=evaluation.delivery_cost-volume-linear,
        short_run=evaluation.short_run_cost,
        frequency=parts['frequency_preference']['cost'])
    if not isclose(sum(terms.values()), evaluation.total_cost, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Exclusive component decomposition does not reproduce total')
    return terms


def explicit_fatigue_rows(context, evaluation):
    span = (context.end-context.stage_start).days
    weights = context.scenario.axis_weights
    result = {r['day'].isoformat(): day_weight(context, r['day'])*7/span*context.exchange*
        sum(w*r['axis_cost'][name] for name, w in zip(AXES, weights))/sum(weights)
        for r in evaluation.parts['session_costs']}
    if len(result) != len(evaluation.parts['session_costs']):
        raise ValueError('Per-date attribution requires no doubles')
    if not isclose(sum(result.values()), evaluation.parts['soft_horizon']['demand'],
                   rel_tol=0., abs_tol=1e-12):
        raise ValueError('Dated fatigue attribution does not reproduce explicit cost')
    return result


def subtract(before, after):
    return {k: after.get(k, 0.)-before.get(k, 0.) for k in sorted(set(before)|set(after))}


def run(root, *, before, after, artifact, focus, output):
    started = perf_counter()
    if output.exists():
        raise FileExistsError(output)
    a, b = (load_decision(root, index) for index in (before, after))
    if a['state'].history != b['state'].history or a['state'].phase_history != b['state'].phase_history:
        raise ValueError('Physical and phase history must be identical')
    common = set(a['dates']) & set(b['dates'])
    for day in sorted(common):
        d = date.fromisoformat(day)
        for field in ('volume_targets', 'support_targets'):
            av, bv = (getattr(c['state'].context, field) for c in (a, b))
            if abs(_target_total(av, d, d+timedelta(days=1))-
                   _target_total(bv, d, d+timedelta(days=1))) > 1e-10:
                raise ValueError('Common dated targets differ')
    saved = json.loads(artifact.read_text())
    cases = {c['index']: c for c in saved['cases']}
    for c, index in ((a, before), (b, after)):
        if cases[index]['snapshot'] != c['state'].fingerprint:
            raise ValueError('Refined witness snapshot mismatch')
    p, q = (tuple(tuple(r) for r in cases[index]['program']) for index in (before, after))
    omitted = {name: [r for r in program if r[0] not in common]
               for name, program in (('before', p), ('after', q))}
    if any(omitted.values()):
        raise ValueError('This attribution requires identical explicit date coverage')
    if dict(p).keys() != dict(q).keys():
        raise ValueError('Fixed-calendar distance witnesses required')
    if any(a['catalog'][k].role != a['catalog'][dict(q)[d]].role for d, k in p):
        raise ValueError('Run roles changed')
    if focus not in dict(p) or a['catalog'][dict(p)[focus]].role != 'easy':
        raise ValueError('Focus must be an existing easy run')
    shift = (b['state'].context.taper_start-a['state'].context.taper_start).days
    end_only = replace(a['state'], context=replace(a['state'].context,
        end=b['state'].context.end, volume_targets=b['state'].context.volume_targets,
        support_targets=b['state'].context.support_targets))
    boundary_only = replace(a['state'], context=replace(a['state'].context,
        continuation_guard_days=a['state'].context.continuation_guard_days+shift))
    if boundary_only.context.taper_start != b['state'].context.taper_start:
        raise ValueError('Boundary ablation failed')
    later_frozen = replace(b['state'], context=replace(b['state'].context,
        continuation_boundary_references=a['state'].context.continuation_boundary_references))
    variants = {'before': (a, a['state']), 'end_extension_only': (a, end_only),
        'boundary_shift_only': (a, boundary_only), 'after_frozen_forecasts': (a, later_frozen),
        'after_actual_forecasts': (b, b['state'])}
    calls = 0

    def price(case, state, program):
        nonlocal calls
        calls += 1
        response = case['evaluate_details'](program, state_override=state)
        ev = response['evaluation']
        return dict(cost=ev.total_cost, eligible=response['decision_eligible'],
            terms=additive_parts(ev), explicit_by_date=explicit_fatigue_rows(state.context, ev),
            continuation_by_handoff=ev.parts['continuation_boundary']['scenarios'])

    scores = {}
    for name, (case, state) in variants.items():
        old, new = (price(case, state, program) for program in (p, q))
        dated = subtract(old['explicit_by_date'], new['explicit_by_date'])
        scores[name] = dict(before_allocation=old, after_allocation=new,
            delta_cost=new['cost']-old['cost'], delta_terms=subtract(old['terms'], new['terms']),
            explicit_delta_by_date=dated, normalization_days=(state.context.end-state.context.stage_start).days,
            fully_weighted_before=state.context.taper_start.isoformat())
    # Distinguish direct marginal rollover effects from an indirect change in
    # marginal volume value when the OTHER prescribed distances are different.
    scans = {}
    # A dose can affect smoothed trailing volume through window+smoothing-2
    # days later. Edits beyond that cannot directly change its volume credit.
    influence_end = (date.fromisoformat(focus)+timedelta(
        days=a['state'].context.policy.volume_window_days+
        a['state'].context.volume_smoothing_days-2)).isoformat()
    tails = dict(before_tail=p, after_tail=q)
    tails['after_edits_within_volume_influence'] = tuple((day, dict(q)[day]
        if focus < day <= influence_end else key) for day, key in p)
    tails['after_edits_beyond_volume_influence'] = tuple((day, dict(q)[day]
        if day > influence_end else key) for day, key in p)
    low, high = sorted(a['catalog'][dict(prog)[focus]].miles for prog in (p, q))
    doses = sorted((d.miles, key) for key, d in a['catalog'].items()
                   if d.role == 'easy' and low-.2-1e-9 <= d.miles <= high+.1+1e-9
                   and key in b['handoffs'])
    for name in ('before', 'after_frozen_forecasts'):
        case, state = variants[name]
        scans[name] = {}
        for label, program in tails.items():
            rows = []
            for miles, key in doses:
                candidate = tuple((day, key if day == focus else k) for day, k in program)
                value = price(case, state, candidate)
                rows.append(dict(miles=miles, cost=value['cost'], eligible=value['eligible'], terms=value['terms']))
            eligible = [r for r in rows if r['eligible']]
            scans[name][label] = dict(rows=rows,
                best_miles=min(eligible, key=lambda r: (r['cost'], r['miles']))['miles'])
    old_score, new_score = scores['before'], scores['after_frozen_forecasts']
    ratio = new_score['normalization_days']/old_score['normalization_days']
    # Compare on one denominator; otherwise a common 66->67 divisor can look
    # like a physical or boundary effect on every session.
    rerating = {day: new_score['explicit_delta_by_date'][day]*ratio-value
                for day, value in old_score['explicit_delta_by_date'].items()}
    result = dict(scope='Frozen common-calendar hindsight attribution, not search or a stability test',
        root=str(root), before=before, after=after, focus=focus,
        artifact=str(artifact), artifact_sha256=sha256(artifact.read_bytes()).hexdigest(),
        fingerprints=[a['state'].fingerprint, b['state'].fingerprint],
        physical_history_identical=True, common_dated_targets_identical=True,
        forecasts_identical=a['handoffs']==b['handoffs'],
        common_program_edits=[dict(day=d, before=k, after=dict(q)[d]) for d, k in p if dict(q)[d]!=k],
        scores=scores, marginal_scans=scans, focus_volume_influence_through=influence_end,
        explicit_rerating_on_original_denominator=rerating, full_evaluations=calls,
        seconds=perf_counter()-started,
        limitations=['Later witnesses are hindsight only; never causal candidates',
            'No newly exposed endpoint work is added',
            'Guard and end changes are attribution ablations, not policy recommendations',
            'HR forecast bounds are scenarios, not confidence intervals',
            'No live data, search or score weights changed'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False, default=str)
    print(json.dumps(dict(path=str(output), seconds=result['seconds'], evaluations=calls,
        allocation_deltas={k: v['delta_terms'] for k, v in scores.items()},
        direct_optima={k: {label: v['best_miles'] for label, v in scans[k].items()} for k in scans}), indent=2))
    return result


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--before', required=True, type=int)
    parser.add_argument('--after', required=True, type=int)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--focus', required=True)
    parser.add_argument('--output', required=True, type=Path)
    run(**vars(parser.parse_args()))
