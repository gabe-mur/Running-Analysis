"""Offline joined-report controls; no reconstruction of issued historical plans."""
from argparse import ArgumentParser
from collections import Counter
from datetime import timedelta
from time import perf_counter
from pathlib import Path
from datetime import datetime
import json

from scripts.audit_program_delivery import ORIGIN, ledger, regular
from scripts.audit_recency_demand import audit as recency_audit, future_events, json_ready
from run_analysis.config import load_config
from run_analysis.development_report import ProgramOption, OpportunityPair, development_report
from run_analysis.program_delivery import TargetSegment
from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy


def audit(database, config, as_of):
    started = perf_counter()
    source = recency_audit(database, config, as_of)
    history = tuple(DemandEvent('actual:'+r['key'], r['day'], tuple(
        DemandTerm(c, f"actual:{r['key']}:{c}", 1., r['quantities'][c]) for c in CHANNELS),
        r['accounting_key'], r['role'], r['structure'], 'observed') for r in source['sessions'])
    signature = history[0].accounting_key
    def option(key, motif):
        return ProgramOption(key, tuple(e for e in ledger(motif) if e.source == 'planned'),
                             tuple(future_events(motif, source['rates'], signature, horizon_days=112)))
    reference = option('reference', regular())
    # These are explicitly declared test anchors, not a newly accepted stage or
    # a claim that these targets were issued before any recorded completion.
    stage = DevelopmentStage('sandbox_current_stage', as_of, 17.3, 4.4, 8.4, 17., 7.,
        ProgressionPolicy(14, .05), 'declared_provisional_control_not_accepted_policy')
    definitions = dict(hold=regular(), extend_easy=regular(easy=5.3),
        insert_support=regular()+[(5, .9, 'easy', .9, None, None)],
        supported_long=regular(long=8.8, easy=4.9),
        compressed=[(0, 8.8, 'long', 8.8, None, None), (1, 4.9, 'easy', 4.9, None, None),
                    (2, 4.5, 'quality', 2., 'threshold', (300., 300., 300.))],
        sparse=regular(easy=3.)[:2],
        quality_funded=regular(quality=5.4, work=(300.,)*4))
    end = ORIGIN+timedelta(days=112)
    results = {}
    for name, motif in definitions.items():
        candidate = option(name, motif)
        pairs = []
        for purpose in ('long', 'easy', 'threshold'):
            # Match explicit opportunities within the first 28 days. Extra short
            # support runs remain unmatched, rather than replacing ordinary easy.
            left = [e for e in reference.demands if e.purpose == purpose and e.day < ORIGIN+timedelta(days=28)]
            right = [e for e in candidate.demands if e.purpose == purpose and e.day < ORIGIN+timedelta(days=28)
                     and not (name == 'insert_support' and (e.day-ORIGIN).days % 7 == 5)]
            pairs.extend(OpportunityPair(a.key, b.key) for a, b in zip(left, right))
        report = development_report(stage, reference, candidate, as_of=as_of,
            reference_targets=[TargetSegment(ORIGIN, end, 17.3)],
            candidate_targets=[TargetSegment(ORIGIN, end, 17.3 if name in {'hold', 'sparse'} else 18.2)],
            coverage_start=ORIGIN, coverage_end=end, evaluation_starts=[ORIGIN+timedelta(days=i) for i in range(28)],
            history=history, opportunity_pairs=pairs, coverage_complete=True,
            timezone=config['timezone_default'])
        delivery = report['program_delivery']['candidate']
        first28 = next(r for r in delivery['intervals'] if r['start'] == ORIGIN and r['days'] == 28)
        comparisons = report['demand']['comparisons']
        constraints = report['feasibility']['constraint_checks']
        feasibility = {k: v for k, v in report['feasibility'].items() if k != 'constraint_checks'}
        feasibility['constraint_checks'] = dict(status=constraints['status'],
            checked_sessions=len(constraints['sessions']),
            session_statuses=dict(Counter(r['status'] for r in constraints['sessions'])),
            projected_reference_sessions=sum(bool(r['projected_reference_keys']) for r in constraints['sessions']),
            recovery_feasibility=constraints['recovery_feasibility'])
        results[name] = dict(stage_review=report['execution_and_stage_review']['status'],
            funding=delivery['funding_evidence_status'], screen=report['program_delivery']['screen'],
            opening28={k: first28[k] for k in ('miles', 'target_miles', 'running_days', 'maximum_running_streak')},
            reference_long_miles=8.4, candidate_first_long=report['goal']['candidate']['long_prescriptions'][0],
            destination_long_miles=report['goal']['destination_long_miles'],
            training_benefit_established=report['goal']['candidate']['training_benefit_established'],
            compared_opportunities=len(comparisons),
            demand_orders=dict(Counter(r['recency_demand_order'] for r in comparisons)),
            demand_orientation='left=reference, right=candidate',
            uncompared_opportunities=len(report['demand']['unmatched_candidate_keys']),
            response=report['response'], feasibility=feasibility, next_step=report['next_step'],
            progression_accepted=report['progression_accepted'], selected_candidate=report['selected_candidate'])
    return dict(as_of=as_of, results=results, original_sessions=len(source['sessions']),
        original_unresolved=source['unresolved'], elapsed_seconds=perf_counter()-started,
        planner_influence=False, running_authorized=False, selected_candidate=None,
        limitations=['Fixed synthetic future programs and explicit sandbox anchors, not searched calendars',
            'Original exposure informs shared demand scenarios, not reconstructed historical compliance',
            'Complete future prescription ledger does not assert complete historical uploads',
            'No historical response expectations or current readiness reconstructed',
            'Only first-28-day ordinary/key opportunities matched; later and added opportunities remain visible',
            'Funding, demand comparisons and conditional stage proposals never establish feasibility'])


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.database, load_config(args.config), args.as_of),
                     default=json_ready, allow_nan=False, indent=2))
