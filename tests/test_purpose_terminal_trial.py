from datetime import date,timedelta
from types import SimpleNamespace
import pytest

from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.program_objective_trial import TrialSession,TrialPolicy,DemandScenario
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy,interval_loss
from run_analysis.adaptive.purpose_terminal_trial import renewal_reference,terminal_value


def test_average_cost_reference_has_zero_full_cycle_relative_cost():
    r=renewal_reference(session_cost=.14,grace=7,memory=7,weight=1)
    d=r['period_days']
    assert r['status']=='finite_reference_policy'
    assert .14+interval_loss(1,max(0,d-7),7)-r['average_cost']*d==pytest.approx(0,abs=1e-12)
    assert renewal_reference(session_cost=1000,grace=7,memory=7,weight=1)['status']=='no_finite_renewal_advantage'


def test_terminal_cost_is_bounded_and_rewards_recent_purpose_not_prior_schedule():
    end=date(2026,11,1)
    q=(Quantity.point(2,'fixture'),)*3
    context=SimpleNamespace(timing=PurposeTimingPolicy(1,grace_fraction=1),policy=TrialPolicy(),
        scenario=DemandScenario(),scales=(1,1,1),exchange=.01,end=end,stage_start=end-timedelta(days=35))
    references=dict(long=q,quality=q)
    def price(age):
        # Different dates for two purposes; same per-role accounting.
        rows=(TrialSession('l',end-timedelta(days=age),8.4,'long',q),
              TrialSession('q',end-timedelta(days=age+1),4,'quality',q,quality_credit=1))
        return terminal_value(context,SimpleNamespace(conditional_ledger=rows),references)
    recent,old=price(1),price(30)
    assert 0<=recent['cost']<old['cost']
    assert old['cost']==pytest.approx(.28/35)
    assert not recent['activated_by_default']
    assert 'omitted' in recent['scope']
