from types import SimpleNamespace

from run_analysis.adaptive.decimal_dose_search import calendar_representatives, decimal_pair_neighbors, decimal_merge_neighbors,quality_easy_neighbors
from run_analysis.adaptive.continuous_search_adapter import search


def domain():
    miles={f'easy:{n/10:.1f}':n/10 for n in range(26,45)}
    return miles,{k:'easy' for k in miles}


def test_native_quality_exchange_preserves_volume_dates_roles_and_pair_coverage():
    miles={'progression':4.2,'threshold':3.,'intervals':3.,'e2':2.4,'e3':3.6}
    roles={k:'easy' if k.startswith('e') and k!='intervals' else 'quality' for k in miles}
    index={str(i):i for i in range(12)}
    p=(('0','progression'),('2','e2'),('8','progression'),('9','e2'))
    rows=list(quality_easy_neighbors(p,roles,miles,index=index,radius=3))
    assert len(rows)==len(set(rows))==4
    assert rows==list(quality_easy_neighbors(reversed(p),roles,miles,index=index,radius=3))
    assert (('0','threshold'),('2','e3'),('8','progression'),('9','e2')) in rows
    assert (('0','progression'),('2','e2'),('8','intervals'),('9','e3')) in rows
    for candidate in rows:
        assert sum(round(miles[k]*10) for _,k in candidate)==132
        assert [(d,roles[k]) for d,k in candidate]==[(d,roles[k]) for d,k in p]


def test_funded_quality_swap_requires_coordinated_easy_compensation():
    miles={'progression':4.2,'threshold':3.,'easy:2.4':2.4,'easy:3.6':3.6}
    roles={k:'easy' if k.startswith('easy') else 'quality' for k in miles}
    dates=('2026-10-01','2026-10-02','2026-10-03')
    prior=((dates[0],'progression'),(dates[2],'easy:2.4'))
    wanted=((dates[0],'threshold'),(dates[2],'easy:3.6'))
    def evaluate(p):
        valid=(tuple((d,roles[k]) for d,k in p)==((dates[0],'quality'),(dates[2],'easy'))
            and sum(round(miles[k]*10) for _,k in p)==66)
        return dict(eligible=valid,cost=0. if p==wanted else 1.)
    options=dict(dates=dates,keys=tuple(miles),roles=roles,dose_miles=miles,
        exploration_keys=tuple(miles),previous=prior,hard_end=dates[-1],
        budget=200,evaluate=evaluate,seed_programs=(prior,),complete_first=True)
    assert search(**options,fair_coordinates=True)['program']==wanted


def test_publication_priority_changes_only_order_not_quality_pair_universe():
    miles={'q0':3.,'q1':3.2,'q2':3.4,'e0':4.,'e1':3.8,'e2':3.6}
    roles={k:'quality' if k.startswith('q') else 'easy' for k in miles}
    dates=tuple(f'2026-10-{n:02d}' for n in range(1,25))
    p=((dates[0],'q0'),(dates[1],'e0'),(dates[14],'q0'),(dates[15],'e0'))
    args=dict(index={d:i for i,d in enumerate(dates)})
    old=list(quality_easy_neighbors(p,roles,miles,**args))
    ordered=list(quality_easy_neighbors(p,roles,miles,**args,priority_before=dates[7]))
    assert set(old)==set(ordered) and len(ordered)==len(set(ordered))==4
    assert all(dict(r)[dates[0]]!='q0' for r in ordered[:2])
    assert all(dict(r)[dates[14]]!='q0' for r in ordered[2:])


def test_full_pair_line_exact_volume_bounds_and_determinism():
    miles,roles=domain()
    index={str(i):i for i in range(9)}
    program=(('1','easy:3.5'),('5','easy:3.5'),('8','easy:3.5'))
    rows=list(decimal_pair_neighbors(program,roles,miles,index=index))
    assert (('1','easy:3.1'),('5','easy:3.9'),('8','easy:3.5')) in rows
    assert (('1','easy:3.5'),('5','easy:3.0'),('8','easy:4.0')) in rows
    assert rows == list(decimal_pair_neighbors(reversed(program),roles,miles,index=index))
    assert len(rows)==36==len(set(rows))
    assert all(sum(round(miles[k]*10) for _,k in p)==105 for p in rows)
    assert all(dict(p)['1']=='easy:3.5' or dict(p)['8']=='easy:3.5' for p in rows)


def test_calendar_representatives_frozen_and_coarse():
    miles,roles=domain()
    catalog={k:SimpleNamespace(miles=v,role=roles[k]) for k,v in miles.items()}
    keys=calendar_representatives(catalog,ordinary=4.4,support=7.)
    assert set(keys)=={'easy:2.6','easy:3.1','easy:3.5','easy:3.6','easy:4.1','easy:4.4'}


def test_search_finds_non_menu_pair_even_when_single_edits_are_ineligible():
    miles,roles=domain()
    dates=tuple(f'2026-10-0{i}' for i in range(1,7))
    prior=((dates[0],'easy:3.5'),(dates[4],'easy:3.5'))
    calls=[]
    def evaluate(p):
        calls.append(p)
        eligible=tuple(d for d,_ in p)==(dates[0],dates[4]) and sum(round(miles[k]*10) for _,k in p)==70
        return dict(eligible=eligible,cost=sum((miles[k]-target)**2 for (_,k),target in zip(p,(3.1,3.9))))
    result=search(dates=dates,keys=tuple(miles),roles=roles,dose_miles=miles,
        exploration_keys=('easy:2.6','easy:3.5','easy:4.4'),previous=prior,
        hard_end=dates[-1],budget=1000,evaluate=evaluate)
    assert result['program']==((dates[0],'easy:3.1'),(dates[4],'easy:3.9'))
    assert len(calls)==len(set(calls))==result['evaluations']<=1000
    assert result['dose_resolution']['decimal_pair_refinement']
    assert result['continuity']['tolerance'] is None


def test_exact_merge_uses_full_tenth_catalog_not_coarse_representatives():
    miles={f'easy:{n/10:.1f}':n/10 for n in range(11,67)}
    roles={k:'easy' for k in miles}
    program=(('2026-11-03','easy:2.4'),('2026-11-04','easy:2.3'),('2026-11-10','easy:2.8'))
    index={d:i for i,d in enumerate(('2026-11-03','2026-11-04','2026-11-10'))}
    rows=list(decimal_merge_neighbors(program,roles,miles,index=index,radius=1))
    assert (('2026-11-03','easy:4.7'),('2026-11-10','easy:2.8')) in rows
    assert (('2026-11-04','easy:4.7'),('2026-11-10','easy:2.8')) in rows
    assert rows==list(decimal_merge_neighbors(reversed(program),roles,miles,index=index,radius=1))
    assert all(sum(round(miles[k]*10) for _,k in p)==75 for p in rows)
    assert not list(decimal_merge_neighbors(program,roles,miles,index=index,minimum_date='2026-11-10'))


def test_rejected_prior_has_scored_volume_preserving_tail_repair_lane():
    miles={'easy:2.3':2.3,'easy:2.4':2.4,'easy:4.7':4.7}
    roles={k:'easy' for k in miles}
    dates=tuple(f'2026-10-{i:02}' for i in range(1,7))
    prior=((dates[3],'easy:2.4'),(dates[4],'easy:2.3'))
    def evaluate(p):
        eligible=len(p)==1 and p[0][0] in dates[3:5] and abs(sum(miles[k] for _,k in p)-4.7)<1e-9
        return dict(eligible=eligible,construction_eligible=eligible,cost=1. if eligible else 2.)
    result=search(dates=dates,keys=tuple(miles),roles=roles,dose_miles=miles,
        exploration_keys=('easy:2.3','easy:2.4'),previous=prior,hard_end=dates[2],
        evaluate=evaluate,budget=200)
    assert result['program'] in (((dates[3],'easy:4.7'),),((dates[4],'easy:4.7'),))
    assert result['prior_tail_repair']['evaluations']>0
    assert result['evaluations']<=200
