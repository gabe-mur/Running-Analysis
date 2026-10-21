"""Reprice two archived calendars; no search, upload, publication or data edits.

Removing one receipt is an explicit diagnostic counterfactual. We verify the
fallback gives full credit and that physical fatigue is unchanged; it is not a
production reconciliation path or permission to discard completion evidence.
"""
from argparse import ArgumentParser
from dataclasses import asdict, replace
import json
from pathlib import Path

from run_analysis.config import load_config
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def audit(root, before, after, activity):
    prior = load_decision(root, before)
    current = load_decision(root, after)
    state = current['state']
    receipts = [r for r in state.quality_completions if r.activity_key == activity]
    if len(receipts) != 1:
        raise ValueError('One verified completion receipt required')
    receipt = receipts[0]
    programs = {
        'prior_calendar': tuple(tuple(p) for p in prior['event']['program']
                                if p[0] in current['dates']),
        'selected_calendar': tuple(tuple(p) for p in current['event']['program']),
    }
    full_credit = replace(state, quality_completions=tuple(
        r for r in state.quality_completions if r.activity_key != activity))
    proportional=replace(state,quality_coaching=replace(state.quality_coaching,
        cadence_contract='proportional_grace_v1'))
    rows = {}
    for label, snapshot in (('observed_credit', state),
                            ('diagnostic_full_credit', full_credit),
                            ('proportional_cadence_actual_credit',proportional)):
        rows[label] = {}
        for name, program in programs.items():
            result = current['evaluate_details'](program, state_override=snapshot)
            e = result['evaluation']
            credit = result['quality_credit'][activity]
            if label == 'diagnostic_full_credit' and (credit.low, credit.high) != (1., 1.):
                raise ValueError('Fallback is not the intended full-credit control')
            if label=='proportional_cadence_actual_credit' and credit!=receipt.credit:
                raise ValueError('Cadence revision changed measured completion credit')
            rows[label][name] = dict(
                eligible=result['decision_eligible'], cost=e.total_cost,
                fatigue=e.weighted_demand_cost, delivery=e.delivery_cost,
                short=e.short_run_cost,
                quality_timing_loss=e.parts['timing']['quality']['weighted_loss'],
                completion_credit=asdict(credit),
                prospective_volume_funded=result['progression_delivery']['funded'])
    for name in programs:
        if len({values[name]['fatigue'] for values in rows.values()})!=1:
            raise ValueError('Credit-only counterfactual changed physical fatigue')
    return dict(
        before_decision=before, after_decision=after, activity=activity,
        receipt=asdict(receipt), comparison=rows,
        selected_minus_prior={label: values['selected_calendar']['cost'] -
            values['prior_calendar']['cost'] for label, values in rows.items()},
        interpretation='Fixed-calendar rank reversal attributable to completion credit; '
            'not an exhaustive full-credit search or a physiological validation',
        physical_history_unchanged=True, live_modified=False, search_performed=False)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--before', required=True, type=int)
    parser.add_argument('--after', required=True, type=int)
    parser.add_argument('--activity', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.before, args.after, args.activity)
    args.output.mkdir(exist_ok=False)
    inputs = [args.root / 'scenario.json']
    for i in (args.before, args.after):
        event_file = args.root / f'decision-{i:04d}.json'
        event = json.loads(event_file.read_text())
        inputs.extend((event_file, args.root / event['snapshot_file']))
    result['manifest'] = freeze_sources(args.output / 'sources.zip',
        config=load_config(), entrypoint=__name__, inputs=inputs)
    with (args.output / 'report.json').open('x') as destination:
        json.dump(result, destination, indent=2, default=str, allow_nan=False)
    print(json.dumps(dict(output=str(args.output),
        selected_minus_prior=result['selected_minus_prior'])))


if __name__ == '__main__':
    main()
