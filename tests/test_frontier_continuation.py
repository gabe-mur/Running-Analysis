from types import SimpleNamespace
from datetime import date,timedelta
from run_analysis.adaptive.frontier_continuation import rebuild_frontiers


DATES=tuple((date(2026,9,27)+timedelta(days=i)).isoformat() for i in range(12))


def item(program,cost,stable=False):
    return dict(physical=tuple(program),score=SimpleNamespace(total_cost=cost),stable=stable)


def test_competing_prefixes_get_rebuilt_tails_and_equal_budgets_without_score_bonus():
    prior=((DATES[0],'easy'),(DATES[8],'long'))
    changed=((DATES[1],'easy'),(DATES[10],'long'))
    best=(prior[0],changed[1])
    seen=[]
    def assess(p,source):
        seen.append((p,source))
        return item(p,1. if p==best else 2. if p==changed else 3. if p==prior else 10.)
    result=rebuild_frontiers([item(changed,2.),item(prior,3.,True)],[prior,changed],
        dates=DATES,keys=('easy','long'),hard_end=DATES[9],prefix_days=4,budget=400,assess=assess)
    lanes=result['lanes']
    assert len(lanes)==2 and lanes[0]['proposal_budget']==lanes[1]['proposal_budget']==200
    assert lanes[0]['best_cost']==2.
    assert lanes[1]['best_cost']==1. and lanes[1]['program']==best
    for lane in lanes:
        assert tuple(p for p in lane['program'] if p[0]<DATES[4])==lane['prefix']
        assert lane['best_cost']<=lane['seed_cost']
    assert any(source=='frontier_tail_rebuild' for _,source in seen)


def test_rejected_or_worse_regrowth_never_erases_the_feasible_incumbent():
    prior=((DATES[0],'easy'),(DATES[8],'long'))
    donor=((DATES[2],'easy'),(DATES[9],'long'))
    def assess(p,source):
        return None if (DATES[9],'long') in p else item(p,20.)
    result=rebuild_frontiers([item(prior,1.,True)],[donor],dates=DATES,keys=('easy','long'),
        hard_end=DATES[9],prefix_days=4,budget=300,assess=assess)
    assert result['lanes'][0]['program']==prior
    assert result['lanes'][0]['best_cost']==1.


def test_terminal_repair_can_change_multiple_days_and_budget_is_explicit():
    prior=((DATES[0],'easy'),(DATES[8],'long'),(DATES[10],'easy'))
    donor=((DATES[0],'long'),(DATES[7],'easy'),(DATES[9],'long'),(DATES[11],'easy'))
    wanted=prior[:1]+donor[-2:]
    calls=[]
    def assess(p,source):
        calls.append((p,source))
        return item(p,0. if p==wanted else 10.)
    result=rebuild_frontiers([item(prior,5.)],[donor],dates=DATES,keys=('easy','long'),
        hard_end=DATES[6],prefix_days=4,budget=12,assess=assess)
    assert result['lanes'][0]['program']==wanted
    assert (wanted,'frontier_terminal_repair') in calls
    assert len(calls)<=12
    assert result['lanes'][0]['status']=='budget_exhausted'


def test_compound_tail_move_can_cross_a_worse_single_edit_without_changing_prefix():
    prior=((DATES[0],'easy'),(DATES[6],'long'),(DATES[9],'easy'))
    paired=(prior[0],(DATES[7],'long'),(DATES[10],'easy'))
    forbidden=((DATES[1],'easy'),*paired[1:])
    calls=[]
    def assess(p,source):
        calls.append(p)
        return item(p,0. if p==paired else -1. if p==forbidden else 10.)
    result=rebuild_frontiers([item(prior,5.)],[],dates=DATES,keys=('easy','long'),
        hard_end=DATES[8],prefix_days=4,budget=300,assess=assess,
        compound_neighbors=lambda c:(paired,forbidden))
    lane=result['lanes'][0]
    assert lane['program']==paired and lane['best_cost']==0.
    assert forbidden not in calls
    assert lane['status']=='conditional_joint_complete'
    assert any(s['phase']=='joint' and s['best_cost']<s['cost'] for s in lane['sweeps'])


def test_calendar_prefix_does_not_freeze_mileage_after_tail_rebuilding():
    prior=((DATES[0],'short'),(DATES[8],'long'))
    resized=((DATES[0],'easy'),(DATES[8],'long'))
    def assess(p,source):
        return item(p,1. if p==resized else 10.)
    result=rebuild_frontiers([item(prior,5.)],[],dates=DATES,keys=('short','easy','long'),
        roles={'short':'easy','easy':'easy','long':'long'},hard_end=DATES[8],prefix_days=4,
        budget=400,assess=assess)
    lane=result['lanes'][0]
    assert lane['program']==resized
    assert lane['prefix']==((DATES[0],'easy'),)
    assert lane['prefix_identity']=='dates_and_roles_doses_free'
