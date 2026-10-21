"""Construct the isolated adaptive service from causally available app evidence.

No process-global selection or live planner activation. The caller owns the output
root and must freeze its source/config manifest before invoking expensive search.
"""
from datetime import timedelta
from dataclasses import replace
from math import ceil
from zoneinfo import ZoneInfo

from run_analysis.stage_target_history import StageTarget
from .new_model_daily_harness import DailyHarness, HarnessPolicy
from .stage_acceptance import StageLedger
from .adaptive_candidate_pipeline import load_athlete, compile_athlete
from .service import AdaptivePlannerService

VOLUME_LINEAR_COEFFICIENT = .1  # AS80 central declared policy assumption.

def initial_target_history(candidate,as_of,timezone,*,baseline_candidate=None,lead_in_days=27):
    """A prospective starting adjustment must not create historical mileage debt."""
    baseline = baseline_candidate or candidate
    day = as_of.astimezone(ZoneInfo(timezone)).date()
    records = (StageTarget(baseline['stage'].key,as_of,day-timedelta(days=lead_in_days),
        baseline['training_policy'].weekly_miles,baseline['easy_duration_policy'].supporting_easy_miles_per_7,
        'AS49 provisional baseline lead-in; not an earlier issued prescription'),)
    if baseline_candidate is not None:
        if baseline['stage'].key == candidate['stage'].key:
            raise ValueError('Prospective starting adjustment requires distinct stage identity')
        records += (StageTarget(candidate['stage'].key,as_of,day,
            candidate['training_policy'].weekly_miles,candidate['easy_duration_policy'].supporting_easy_miles_per_7,
            'Prospective starting-volume experiment; historical reference unchanged'),)
    stage=candidate['stage']
    if stage.policy.volume_path=='gradual':
        delta=stage.policy.volume_goal_miles-stage.weekly_equivalent_miles
        if delta<0 or delta>stage.weekly_equivalent_miles*stage.policy.growth_fraction+.05+1e-8:
            raise ValueError('Gradual phase must be bounded to one declared growth increment')
        ramp=StageTarget(stage.key+':volume-path',as_of,day,stage.weekly_equivalent_miles,
            candidate['easy_duration_policy'].supporting_easy_miles_per_7,
            'Bounded prospective volume path; not measured adaptation or automatic future stages',
            stage.policy.review_days,stage.policy.volume_goal_miles,
            candidate['easy_duration_policy'].supporting_easy_miles_per_7+delta)
        records=(*records[:-1],ramp) if records[-1].effective_on==day else (*records,ramp)
    return records


def build(root,config,inputs,candidate,as_of,*,evaluations=3000,tail_days=14,steady_continuation=False,soft_horizon_taper=False,baseline_candidate=None,delivery_contract='sustained_v1',continuation_boundary=False,single_volume=False,volume_linear_coefficient=None,prospective_review_hour=None,purpose_continuation=False,purpose_continuation_phase=None,frequency_contract=None,coordinated_final_evaluations=0,continuation_distribution='linear_v1',general_progression=None):
    # New single-volume builds use the tested central curve. Old archived runs
    # resume their explicit old contract; passing zero builds a baseline control.
    if volume_linear_coefficient is None:
        volume_linear_coefficient=VOLUME_LINEAR_COEFFICIENT if single_volume else 0.
    # New enabled builds use corrected timing; archived snapshots/resume retain
    # their version. The optional override is a research comparator, not a UI preference.
    if purpose_continuation_phase is None:
        purpose_continuation_phase='known_full_phase_v1' if purpose_continuation else 'uniform_v1'
    if single_volume:
        if not continuation_boundary:
            raise ValueError('Single-volume prototype requires guarded continuation')
        from .adaptive_candidate_pipeline import supporting_easy_target
        def convert(value):
            p=value['training_policy']
            support=supporting_easy_target(value['stage'],next(d.miles for d in value['catalog'] if d.key=='quality:ordinary'),
                single_volume=True,long_per_7=p.long_per_7,quality_per_7=p.quality_per_7)
            return dict(value,training_policy=replace(p,purpose_weights=(p.purpose_weights[0],0.,*p.purpose_weights[2:])),
                easy_duration_policy=replace(value['easy_duration_policy'],supporting_easy_miles_per_7=support))
        candidate=convert(candidate)
        baseline_candidate=convert(baseline_candidate) if baseline_candidate else None
    if candidate['stage'].policy.volume_path=='gradual' and delivery_contract!='sustained_v1':
        raise ValueError('Gradual targets require the dated sustained delivery contract')
    policy=candidate['training_policy']
    smoothing=ceil(max(7/policy.long_per_7,7/policy.quality_per_7)) if delivery_contract=='sustained_v1' else 1
    harness=DailyHarness(root,config,origin=as_of,history=inputs['history'],phase_history=inputs['phase_history'],
        capacity_eligibility={r.key:r.capacity_eligible for r in inputs['sessions']},
        search_policy=HarnessPolicy(evaluations=evaluations,tail_days=tail_days,steady_continuation=steady_continuation,
            soft_horizon_taper=soft_horizon_taper,delivery_contract=delivery_contract,continuation_boundary=continuation_boundary,
            continuation_guard_days=7 if continuation_boundary else 0,
            volume_linear_coefficient=volume_linear_coefficient,prospective_review_hour=prospective_review_hour,
            purpose_continuation=purpose_continuation,purpose_continuation_phase=purpose_continuation_phase,
            coordinated_final_evaluations=coordinated_final_evaluations,
            continuation_distribution=continuation_distribution,general_progression=general_progression),
        stage=candidate['stage'],target_history=initial_target_history(candidate,as_of,
            config['timezone_default'],baseline_candidate=baseline_candidate,
            lead_in_days=(policy.volume_window_days or 1)+smoothing-2),
        frequency_contract=frequency_contract,
        **{k:v for k,v in candidate.items() if k not in {'stage','construction'}})
    StageLedger.for_sandbox(harness.sandbox,seed=candidate['stage'])
    return harness


def create_service(root,config,database,as_of,*,goal=17.,evaluations=10000,
                   tail_days=14,steady_continuation=False,soft_horizon_taper=False,continuation_boundary=False,
                   single_volume=False,volume_linear_coefficient=None,purpose_continuation=False):
    inputs=load_athlete(database,config,as_of)
    candidate=compile_athlete(inputs,as_of,goal=goal)
    harness=build(root,config,inputs,candidate,as_of,evaluations=evaluations,
        tail_days=tail_days,steady_continuation=steady_continuation,soft_horizon_taper=soft_horizon_taper,
        continuation_boundary=continuation_boundary,single_volume=single_volume,
        volume_linear_coefficient=volume_linear_coefficient,purpose_continuation=purpose_continuation)
    return AdaptivePlannerService(harness,inputs)


def resume_service(root,config,as_of):
    """Reopen committed evidence; do not bootstrap targets from today's history."""
    harness=DailyHarness.resume(root,config,as_of=as_of)
    inputs=dict(history=tuple(r for r in harness.history if r.key in harness.seed_keys),
        sessions=tuple(r for r in harness.session_evidence() if r.key in harness.seed_keys),
        phase_history=harness.phase_history,reference=harness.ref)
    return AdaptivePlannerService(harness,inputs)
