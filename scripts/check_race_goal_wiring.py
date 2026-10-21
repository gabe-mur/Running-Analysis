"""Real-evidence isolated general/race projection. Never adopts the live runtime."""
from argparse import ArgumentParser
from datetime import date, datetime, timedelta
from dataclasses import replace, asdict
import json
from pathlib import Path

from run_analysis.config import load_config
from run_analysis.adaptive.new_model_daily_harness import DailyHarness,HarnessPolicy
from run_analysis.adaptive.stage_acceptance import StageLedger
from run_analysis.adaptive.service import AdaptivePlannerService
from run_analysis.adaptive.race_goal import RaceGoalRevision
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from scripts.run_adaptive_lifecycle import fork_seed_plan
from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries


def run(source, output, *, distance='10k', race_days=40, complete_first=None,final_winner_evaluations=None,
        coordinated_final_evaluations=None, decision_at=None,calendar_merge_evaluations=None):
    latest=json.loads(sorted(source.glob('decision-*.json'))[-1].read_text())
    case=load_decision(source,latest['index']+1)  # Exact old score/admission check.
    state=case['state']; runtime=case['event']['stage_runtime']
    config=load_config()
    reference_root=output.with_suffix('.reference')
    fork_seed_plan(source,reference_root)
    reference_harness=DailyHarness.resume(reference_root,config,as_of=state.as_of)
    try:
        past_reviews=tuple(row for row in review_entries(reference_harness,state.as_of)
            if row.completion is not None and row.completion.available_at<=state.as_of
            and row.completion.end<state.as_of)
    finally:
        reference_harness.sandbox.close()
    raw=runtime['stage']
    stage=DevelopmentStage(**dict(raw,issued_at=datetime.fromisoformat(raw['issued_at']),
        policy=ProgressionPolicy(**raw['policy'])))
    targets=tuple(StageTarget(**dict(row,available_at=datetime.fromisoformat(row['available_at']),
        review_anchor=datetime.fromisoformat(row['review_anchor']) if row.get('review_anchor') else None,
        effective_on=date.fromisoformat(row['effective_on']))) for row in runtime['target_history'])
    scenario=json.loads((source/'scenario.json').read_text())
    inherited=HarnessPolicy(**scenario['search_policy'])
    updates={k:v for k,v in dict(complete_first_search=complete_first,
        final_winner_evaluations=final_winner_evaluations,
        coordinated_final_evaluations=coordinated_final_evaluations,
        calendar_merge_evaluations=calendar_merge_evaluations).items() if v is not None}
    origin=decision_at or state.as_of
    if origin < state.as_of:
        raise ValueError('Diagnostic origin cannot precede source evidence')
    manifest=freeze_sources(output.with_suffix('.sources.zip'),config=config,entrypoint=__name__,
        inputs=(source/f"decision-{latest['index']+1:04d}.json",source/latest['snapshot_file'],source/'scenario.json'))
    h=DailyHarness(output,config,origin=origin,catalog=tuple(case['catalog'].values()),
        history=state.history,phase_history=state.phase_history,scales=state.context.scales,
        training_policy=state.context.policy,search_policy=replace(inherited,**updates),
        demand_scenario=state.context.scenario,exchange=state.context.exchange,timing=state.context.timing,
        coaching_contracts=dict(state.contracts),easy_duration_policy=state.context.easy_duration_policy,
        decision_source='Isolated real-evidence shared search wiring; original physical uncertainty preserved',
        restrictions=state.restrictions,current_context=state.current,responses=state.responses,
        stage=stage,target_history=targets,frequency_contract=state.frequency_contract,
        seed_review_entries=past_reviews,
        capacity_eligibility={r.key:r.capacity_eligible for r in state.athlete.sessions})
    h.planner_preferences=state.planner_preferences
    h.race_event_history=state.context.race_event_history
    StageLedger.for_sandbox(h.sandbox,seed=stage)
    at=origin+timedelta(seconds=1)
    if distance!='general_fitness':
        h.race_goal=RaceGoalRevision.create(available_at=at,distance=distance,
            race_date=at.astimezone(h.zone).date()+timedelta(days=race_days),
            preparation_mode='balanced',parent_key='diagnostic:'+state.fingerprint,general_long_goal_miles=stage.goal_long_miles)
    h.previous=tuple(map(tuple,case['event']['program']))
    inputs=dict(history=state.history,sessions=h.session_evidence(),phase_history=state.phase_history,reference=h.ref)
    service=AdaptivePlannerService(h,inputs)
    try:
        decision=service.refresh(at,reason='explicit diagnostic shared search objective')
        h=service.harness; event=decision['event']
        report=dict(manifest=manifest,source=str(source),live_changed=False,
            source_score_reproduced=case['archived_cost'],scope='One real-evidence projection; not daily stability',
            stage_review=decision['stage_review'],race_preparation=event.get('race_preparation'),
            **{k:event[k] for k in ('seconds','evaluations','volume','cost','program','selected','churn')},
            refinement=event.get('refinement'))
        (output/'race-wiring-summary.json').write_text(json.dumps(report,indent=2,
            default=lambda v:asdict(v) if hasattr(v,'__dataclass_fields__') else v.isoformat()))
        print(json.dumps({k:report[k] for k in ('seconds','evaluations','volume','cost','race_preparation')},default=str),flush=True)
    finally:
        h.sandbox.close()
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('data/adaptive-planner-v201-20261003'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--distance',choices=('general_fitness','5k','10k','half_marathon','marathon'),default='10k')
    parser.add_argument('--race-days',type=int,default=40)
    parser.add_argument('--complete-first',action='store_true',default=None,help='Otherwise inherit the current shared search policy')
    parser.add_argument('--final-winner-evaluations',type=int,default=None)
    parser.add_argument('--coordinated-final-evaluations',type=int,default=None,
        help='Additional full checks for exact fixed-calendar dose refinement; offline only')
    parser.add_argument('--decision-at',type=datetime.fromisoformat,default=None)
    parser.add_argument('--calendar-merge-evaluations',type=int,default=None)
    args=parser.parse_args();run(args.source,args.output,distance=args.distance,race_days=args.race_days,
        complete_first=args.complete_first,final_winner_evaluations=args.final_winner_evaluations,
        coordinated_final_evaluations=args.coordinated_final_evaluations,decision_at=args.decision_at,
        calendar_merge_evaluations=args.calendar_merge_evaluations)
