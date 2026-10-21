from types import SimpleNamespace

import pytest

from scripts.audit_available_search_starts import available_programs,carry_easy_doses,shortlist_captured


def fixtures():
    dates=('2026-10-07','2026-10-08','2026-10-09')
    catalog={k:SimpleNamespace(role=role) for k,role in (('e:2','easy'),('e:3','easy'),('q','quality'))}
    event=dict(index=2,as_of='2026-10-07T08:00:00+00:00',
        program=[[dates[0],'e:2'],[dates[2],'q']],refinement=dict(final_joint=[
            dict(best_program=[[dates[0],'e:3'],[dates[2],'q']]),
            dict(best_program=[[dates[0],'e:2'],[dates[2],'q']])]))
    return event,catalog,dates


def test_only_recorded_programs_and_actual_predecessor_are_returned_with_provenance():
    event,catalog,dates=fixtures()
    prior=dict(index=1,as_of='2026-10-06T08:00:00+00:00',
        program=[['2026-10-06','q'],[dates[1],'e:3']])
    rows=available_programs(event,catalog,dates,prior=prior)
    assert len(rows)==3
    assert rows[0]['sources']==['archived_winner','refinement.final_joint[1].best_program']
    assert rows[-1]['program']==((dates[1],'e:3'),)
    assert all('2026-10-06' not in dict(r['program']) for r in rows)


@pytest.mark.parametrize('index,at',[(3,'2026-10-06T08:00:00+00:00'),(1,'2026-10-08T08:00:00+00:00')])
def test_later_plan_cannot_be_used_even_if_physical_inputs_are_identical(index,at):
    event,catalog,dates=fixtures()
    prior=dict(index=index,as_of=at,program=event['program'])
    with pytest.raises(ValueError,match='later decision'):
        available_programs(event,catalog,dates,prior=prior)


def test_easy_transplant_keeps_dates_types_and_non_easy_doses():
    event,catalog,dates=fixtures()
    winner=tuple(map(tuple,event['program']))
    alternative=((dates[0],'e:3'),(dates[1],'e:2'),(dates[2],'e:3'))
    mapped=carry_easy_doses(winner,alternative,catalog)
    assert mapped==((dates[0],'e:3'),(dates[2],'q'))
    assert [d for d,k in mapped]==[d for d,k in winner]


def test_duplicate_date_and_out_of_domain_trace_fail_closed():
    event,catalog,dates=fixtures()
    for bad in ([[dates[0],'e:2'],[dates[0],'q']],[['2026-10-20','e:2']]):
        event['refinement']['final_joint'][0]['best_program']=bad
        with pytest.raises(ValueError,match='domain'):
            available_programs(event,catalog,dates)


def test_captured_shortlist_is_admitted_deterministic_and_separates_calendars_from_doses():
    event,catalog,dates=fixtures()
    winner=tuple(map(tuple,event['program']))
    rows=[dict(program=winner,cost=3.,eligible=True,evaluation=1),
        dict(program=((dates[0],'e:3'),(dates[2],'q')),cost=2.,eligible=True,evaluation=2),
        dict(program=((dates[1],'e:2'),(dates[2],'q')),cost=1.,eligible=True,evaluation=3),
        dict(program=((dates[1],'e:3'),(dates[2],'q')),cost=.5,eligible=False,evaluation=4)]
    expected=shortlist_captured(rows,winner,catalog,per_group=1)
    assert expected==shortlist_captured(list(reversed(rows)),winner,catalog,per_group=1)
    assert [r['sources'] for r in expected]==[['captured_search_evaluation:2'],['captured_search_evaluation:3']]


def test_pool_mapping_prices_all_admitted_unique_doses_and_matches_full_scorer():
    from test_compiled_calendar_doses import fixture
    from scripts.audit_mapped_candidate_pool import mapped_pool
    compiled,evaluate,state,physical,winner,details=fixture()
    catalog={k:v for (d,k),v in physical.items()}
    alternate=compiled.original_levels.copy();alternate[0]+=1
    alternative=compiled.physical(alternate)
    pool=[dict(program=winner,eligible=True,evaluation=1),
        dict(program=alternative,eligible=True,evaluation=2),
        dict(program=alternative,eligible=True,evaluation=3),
        dict(program=compiled.physical(alternate+1),eligible=False,evaluation=4)]
    rows=mapped_pool(compiled,winner,pool,catalog)
    assert len(rows)==2
    assert rows[1]['source_evaluations']==[2,3]
    for row in rows:
        assert evaluate(row['program'])['cost']==pytest.approx(row['cost'],abs=1e-12)
