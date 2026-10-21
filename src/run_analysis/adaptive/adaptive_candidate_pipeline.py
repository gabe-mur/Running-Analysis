"""Read-only athlete evidence to real candidate contracts; no live scheduler writes."""
from dataclasses import replace
from decimal import Decimal
from datetime import datetime, timedelta
from statistics import median
import sqlite3

from run_analysis.adaptive_prescriptions import compile_prescriptions, PrescriptionPolicy
from run_analysis.athlete_projection import SessionEvidence
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.movement import classify_movement
from run_analysis.response_model import rate
from run_analysis.staged_progression import bootstrap_stage, ProgressionPolicy, MILE,tenth
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.work_stimulus_evidence import observed_work_signals
from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.evidence_history import load_history
from run_analysis.adaptive.continuous_program_adapter import with_continuous_trace
from run_analysis.adaptive.direct_quality_load import PhaseHistory, phase_observations
from run_analysis.adaptive.forecast_handoff import ForecastScope, PhaseRateScenario
from run_analysis.adaptive.new_model_daily_harness import Dose
from run_analysis.adaptive.program_evidence_adapter import QualityReference
from run_analysis.adaptive.program_objective_trial import TrialPolicy, DemandScenario, EasyDurationPolicy
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy


def resize_easy(dose,miles,contract):
    """One-decimal materialization at the SAME frozen pace/rate assumptions."""
    if (dose.role!='easy' or abs(miles*10-round(miles*10))>1e-8 or
            not contract.ordinary.miles.low<=miles<=contract.ordinary.miles.high):
        raise ValueError('Easy command outside the frozen one-decimal envelope')
    factor=miles/dose.miles
    return replace(dose,key=f'easy:{miles:.1f}',miles=miles,
        phases=tuple((k,s*factor) for k,s in dose.phases),
        phase_scenarios=tuple((lo*factor,hi*factor) for lo,hi in dose.phase_scenarios))
from run_analysis.phase_partitions import phase_layout


def supporting_easy_target(stage,quality_miles,*,single_volume=False,long_per_7=1.,quality_per_7=1.):
    """Exact arithmetic on already committed decimal commands.

    20.0 - 8.4 - 3.9 is 7.7, not a binary value just below it that rounds a
    half-share 3.85 down to 3.8 and quietly underfunds a nominal seed.
    """
    remaining=(Decimal(str(stage.weekly_equivalent_miles))-Decimal(str(stage.long_miles))*Decimal(str(long_per_7))
               -Decimal(str(quality_miles))*Decimal(str(quality_per_7)))
    if single_volume:
        if remaining<=0:
            raise ValueError('Single-volume established stage requires a positive easy remainder; revise key allocations')
        return float(remaining)
    return max(stage.ordinary_miles,float(remaining))


def load_athlete(database, config, as_of, *, since=None):
    reference = ContinuousDoseReference('sandbox upload accounting')
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        metadata = {str(k):dict(available=max(datetime.fromisoformat(c),datetime.fromisoformat(s)),
            eligible=not illness and health in (None,'','normal') and included!=0)
            for k,c,s,illness,health,included in db.execute('''SELECT a.id,a.created_at_utc,s.imported_at_utc,
                o.illness,o.health_tag,o.include_in_model FROM activities a
                JOIN activity_sources l ON l.activity_id=a.id AND l.is_primary=1
                JOIN source_files s ON s.id=l.source_file_id
                LEFT JOIN run_overrides o ON o.activity_id=COALESCE(NULLIF(a.activity_id,''),a.activity_uid)''')}
    hr = HeartRateReference(config['resting_hr'],config['max_hr'],'current frozen athlete HR configuration')
    phases,sessions,physical = [],[],[]
    def collect(run,activity):
        m = metadata[run.key]
        available = max(run.end,m['available'])
        if available>as_of:
            return run
        intervals = classify_movement(activity.trackpoints,config['moving_time']).intervals
        trace = observed_cardio_trace(activity,intervals,hr,reference.policy,include_spans=True,compute_density=False)
        if trace['load_bounds_within_reference'] is None:
            raise ValueError(f'Out-of-reference HR needs explicit reference review: {run.key}')
        updated,_ = with_continuous_trace(replace(run,available_at=available),trace['normalized_spans'],reference=reference,
            trace_key=run.key,trace_source='observed',elapsed_seconds=trace['elapsed_seconds'])
        if updated.role=='quality':
            signals = observed_work_signals(updated,activity,config,intervals,trace['normalized_spans']) if updated.bouts else None
            layout = phase_layout(signals.windows,trace['elapsed_seconds']) if signals else ()
        else:
            layout = (('warmup',0.,trace['elapsed_seconds']),)
        physical.append(updated)
        if m['eligible']:
            phases.append(PhaseHistory(updated.key,updated.start,updated.end,available,reference.accounting_key,
                phase_observations(layout,trace['normalized_spans'])))
        sessions.append(SessionEvidence(updated.key,updated.start,updated.end,available,
            'intervals' if updated.role=='quality' else updated.role,updated.miles*MILE,
            updated.demand[0].low*60,updated.demand[1],reference.accounting_key,capacity_eligible=m['eligible']))
        return updated
    old = QualityReference('threshold','work_seconds',Quantity.point(900,'discarded old adapter reference'),Quantity(290,310),
                           'original-file loader only')
    _,unresolved = load_history(database,config,as_of,old,
        since=since if since is not None else as_of-timedelta(days=85),run_transform=collect)
    if unresolved:
        raise ValueError(f'Unresolved recent timing: {unresolved}')
    return dict(history=tuple(physical),phase_history=tuple(phases),sessions=tuple(sessions),reference=reference)


def compile_easy_volume_phase(inputs,as_of,*,timezone):
    """Shared bootstrap for the AS64 prototype and daily/upload experiment."""
    from run_analysis.volume_retention import recent_volume_proposal
    baseline=compile_athlete(inputs,as_of)
    proposal=recent_volume_proposal(inputs['sessions'],as_of,baseline['stage'],timezone=timezone)
    if proposal['candidate'] is None:
        raise ValueError(proposal['status'])
    proposed=proposal['candidate']
    phase_goal=tenth(proposed.weekly_equivalent_miles*(1+proposed.policy.growth_fraction))
    proposal=dict(proposal,candidate=replace(proposed,policy=replace(proposed.policy,
        growth_focus='easy_volume',volume_goal_miles=phase_goal,volume_path='gradual')))
    return baseline,compile_athlete(inputs,as_of,stage=proposal['candidate']),proposal


def compile_athlete(inputs,as_of,*,stage=None,goal=17.,policy=PrescriptionPolicy()):
    seed = bootstrap_stage(inputs['sessions'],as_of,goal_long_miles=goal,policy=ProgressionPolicy(14,.05)) if stage is None else None
    stage = stage or seed['stage']
    if stage is None:
        raise ValueError(f"Cannot bootstrap established athlete stage: {seed['status']}")
    built = compile_prescriptions(inputs['sessions'],stage,physical=inputs['history'],policy=policy)
    if not built['prescriptions']:
        raise ValueError('No prior duration evidence for established athlete')
    ref = inputs['reference']
    # These are consciously declared model-rate scenarios, not instantaneous HR
    # targets or calibrated priors. Only absent phases use them.
    fallback = Quantity(rate(.6,ref.policy),rate(.9,ref.policy),rate(.75,ref.policy),
                        'provisional controlled aerobic work HRR-equivalent rate scenario')
    catalog,contracts = [],{}
    for p in built['prescriptions']:
        rates = (PhaseRateScenario('work',fallback,stage.issued_at,ref.accounting_key,p.intensity_key,
            'AS39 controlled introduction/recent-work gap; not inferred fatigue or target instantaneous HR'),) if p.role=='quality' else ()
        scope = ForecastScope(p.intensity_key,p.duration_source,
            'continuous running/jogging prescription; standing recovery not specified',rate_scenarios=rates)
        catalog.append(Dose(p.key,p.role,p.miles,p.phases,scope,p.phase_scenarios))
        contracts[p.key] = p.contract
    easy = [r for r in inputs['history'] if r.role=='easy' and r.end<=stage.issued_at and
            r.available_at<=stage.issued_at and r.start>=stage.issued_at-timedelta(days=28)]
    minute_scale = stage.ordinary_miles*built['nominal_minutes_per_mile']
    cardio_scale = stage.ordinary_miles*median((r.demand[1].estimate if r.demand[1].estimate is not None else
            (r.demand[1].low+r.demand[1].high)/2)/r.miles for r in easy) if easy else minute_scale
    q = next(p for p in built['prescriptions'] if p.key=='quality:ordinary')
    support = supporting_easy_target(stage,q.miles)
    ordinary=next(d for d in catalog if d.role=='easy')
    envelope=contracts[ordinary.key].ordinary.miles
    for count in range(1,7):
        miles=tenth(Decimal(str(support))/count)
        if envelope.low<=miles<=envelope.high:
            extra=resize_easy(ordinary,miles,contracts[ordinary.key])
            if extra.key not in contracts:
                catalog.append(extra)
                contracts[extra.key]=contracts[ordinary.key]
    return dict(stage=stage,catalog=tuple(catalog),coaching_contracts=contracts,
        scales=(minute_scale,cardio_scale,ref.quality_minutes),
        training_policy=TrialPolicy(weekly_miles=stage.weekly_equivalent_miles,ordinary_miles=stage.ordinary_miles,
            long_miles=stage.long_miles,quality_work_minutes=built['quality_work_minutes'],volume_window_days=28),
        demand_scenario=DemandScenario(),exchange=.01,timing=PurposeTimingPolicy(1.,grace_fraction=1.),
        easy_duration_policy=EasyDurationPolicy(policy.minimum_minutes,policy.penalty_free_minutes,.01,policy.source,support),
        decision_source='A3/P1 frozen original-evidence prescriptions; documented provisional defaults',
        construction=built)
