from test_compiled_calendar_doses import fixture
from run_analysis.adaptive.calendar_merge_refinement import refine_merges


def test_joint_diagnostic_rechecks_every_adoption_and_stays_bounded():
    _,evaluate,state,physical,program,details=fixture()
    calls=[]
    def captured(p):
        calls.append(p)
        return details(p)
    report=refine_merges(state,program,physical=physical,
        evaluate_details=captured,max_full_evaluations=32)
    assert len(calls)==len(set(calls))==report['full_evaluations']<=32
    assert report['cost']<=evaluate(program)['cost']
    assert evaluate(report['program'])['eligible']
    assert evaluate(report['program'])['cost']==report['cost']
    assert set(dict(report['program']))<=set(dict(program))
    assert all(physical[p].miles>0 for p in report['program'])


def test_diagnostic_is_repeatable_and_never_claims_a_calendar_certificate():
    _,evaluate,state,physical,program,details=fixture()
    first=refine_merges(state,program,physical=physical,
        evaluate_details=details,max_full_evaluations=4)
    second=refine_merges(state,program,physical=physical,
        evaluate_details=details,max_full_evaluations=4)
    assert first['program']==second['program']
    assert first['cost']==second['cost']
    assert first['full_evaluations']==second['full_evaluations']
    assert 'no global or calendar-local certificate' in first['scope']


def test_joint_refinement_does_not_bypass_caller_admission():
    _,evaluate,state,physical,program,details=fixture()
    def only_initial(p):
        return dict(details(p),decision_eligible=p==program)
    report=refine_merges(state,program,physical=physical,
        evaluate_details=only_initial,max_full_evaluations=32)
    assert report['program']==program
    assert not report['changed']
