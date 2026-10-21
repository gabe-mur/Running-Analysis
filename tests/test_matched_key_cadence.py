from types import SimpleNamespace

from scripts.audit_matched_key_cadence import balanced_units,matched_program,same_volume_sections,conditioned_tail_control


def test_exact_decimal_balancing_and_capacity_failures():
    available={n:None for n in range(11,45)}
    assert balanced_units(103,4,available)==(26,26,26,25)
    assert balanced_units(103,4,available,reverse=True)==(25,26,26,26)
    assert balanced_units(10,1,available) is None
    assert balanced_units(45,1,available) is None
    assert balanced_units(0,0,available)==()
    assert balanced_units(1,0,available) is None


def test_match_preserves_key_dates_and_exact_review_hard_tail_sections():
    catalogue={f'e{n}':SimpleNamespace(role='easy',miles=n/10) for n in range(11,45)}
    catalogue['long']=SimpleNamespace(role='long',miles=8.9)
    program=(('2026-10-06','e22'),('2026-10-08','long'),('2026-10-15','e22'),('2026-10-27','e22'))
    result=matched_program(program,catalogue,('2026-10-14','2026-10-26','2026-11-28'),(115,30,40))
    assert result==(('2026-10-06','e26'),('2026-10-08','long'),('2026-10-15','e30'),('2026-10-27','e40'))
    miles={k:d.miles for k,d in catalogue.items()}
    assert same_volume_sections(result,result,miles,'2026-10-26')
    assert not same_volume_sections(program,result,miles,'2026-10-26')
    assert matched_program(program,catalogue,('2026-10-14','2026-10-26','2026-11-28'),(115,50,40)) is None


def test_uniform_reset_can_penalize_nominal_eight_day_phase_with_no_overdue_session():
    from datetime import date,timedelta
    from run_analysis.adaptive.program_objective_trial import TrialPolicy,TrialSession
    from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy,envelope_day
    from run_analysis.adaptive.continuation_boundary_trial import purpose_handoff_losses
    from run_analysis.workload_accounting import Quantity
    handoff=date(2026,10,20)
    context=SimpleNamespace(policy=TrialPolicy(long_per_7=.875,quality_per_7=.875),
        timing=PurposeTimingPolicy(1.,purposes=('long',),grace_fraction=1.),
        stage_start=date(2026,9,29),decision=date(2026,10,5),end=date(2026,11,28),opening_coverage=())
    row=TrialSession('known',handoff-timedelta(days=7),8.4,'long',tuple(Quantity.point(0.,'test') for _ in range(3)),0.,'planned')
    assert envelope_day(((row.day,1.),),handoff,grace=8.,memory=8.)['loss']==0.
    assert purpose_handoff_losses(context,(row,),(handoff,))['long']['loss']>0.
    control=conditioned_tail_control(context,(row,),(handoff,))['long']
    assert control['loss']==0.
    assert control['handoffs'][0]['arrival_delay_days']==1
    # A same-handoff future workout is NOT borrowed into the known prefix.
    from dataclasses import replace
    assert conditioned_tail_control(context,(replace(row,day=handoff),),(handoff,))['long']['handoffs'][0]['basis']=='uniform_unknown_partial_or_overdue_phase'
    assert conditioned_tail_control(context,(replace(row,day=handoff-timedelta(days=9)),),(handoff,))['long']['handoffs'][0]['basis']=='uniform_unknown_partial_or_overdue_phase'
    assert conditioned_tail_control(context,(replace(row,miles=4.2),),(handoff,))['long']['handoffs'][0]['basis']=='uniform_unknown_partial_or_overdue_phase'
