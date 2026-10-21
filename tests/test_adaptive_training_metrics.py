from datetime import datetime,timedelta,timezone
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.new_model_daily_harness import accounted,point,training_metrics


def test_metrics_cover_history_boundary_and_both_rolling_windows():
    origin=datetime(2026,9,27,tzinfo=timezone.utc)
    ref=ContinuousDoseReference('metrics fixture')
    def run(day,miles,role):
        start=origin+timedelta(days=day)
        return accounted(str(day),start,miles,(('warmup',miles*600),),point(miles*10),role,
            ref,source='observed',available=start+timedelta(seconds=miles*600))
    result=training_metrics((run(0,3,'easy'),run(2,8,'long')),
        (run(-7,8,'long'),run(-1,4,'quality')),origin.date(),3,timezone.utc)
    assert result['moving_minutes']==110
    assert result['by_role']['long']['miles']==8
    assert result['rolling_7'][-1]['miles']==15
    assert result['rolling_14'][-1]['miles']==23
    assert result['boundary_gaps']['long'][-1]['days']==9
    assert result['gap_coverage']['long']['longest_bracketed_gap_including_lead_in']==9
    assert result['gap_coverage']['quality']['trailing_days_without_completion']==3
    assert result['gap_coverage']['quality']['first_in_window'] is None
    assert result['gap_coverage']['quality']['trailing_gap_right_censored']
    assert result['short_window_rate_warning']
    assert result['compression_is_diagnostic_not_automatic_failure']


def test_summary_includes_actual_on_failure_day_without_claiming_day_complete():
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness,HarnessPolicy
    from run_analysis.adaptive.program_objective_trial import TrialPolicy,DemandScenario
    from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy
    origin=datetime(2026,9,27,tzinfo=timezone.utc)
    ref=ContinuousDoseReference('metrics fixture')
    h=DailyHarness.__new__(DailyHarness)
    h.origin,h.zone=origin,timezone.utc
    h.history=[accounted('failed-day-run',origin+timedelta(days=1),8.,(('warmup',4800.),),
        point(80),'long',ref,source='observed',available=origin+timedelta(days=1,hours=2))]
    h.seed_keys,h.events,h.catalog=set(),[],{}
    h.policy,h.training_policy=HarnessPolicy(),TrialPolicy()
    h.demand_scenario,h.exchange,h.timing=DemandScenario(),.01,PurposeTimingPolicy(1.)
    report=h.summary(1,[])
    assert report['days']==1
    assert report['delivery_window_days']==2 and report['delivery_window_includes_unfinished_day']
    assert report['delivered']['miles']==8.
