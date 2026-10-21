from run_analysis.adaptive.native_quality_final_refinement import refine


def test_quality_adoption_reopens_pairs_and_never_bypasses_admission():
    dates=('2026-10-01','2026-10-02','2026-10-03')
    roles={'q3':'quality','q32':'quality','q34':'quality','e4':'easy','e38':'easy','e36':'easy'}
    miles=dict(q3=3.,q32=3.2,q34=3.4,e4=4.,e38=3.8,e36=3.6)
    original=((dates[0],'q3'),(dates[2],'e4'))
    winner=((dates[0],'q32'),(dates[2],'e38'))
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=0. if dict(p)[dates[0]]=='q34' else .5 if p==winner else 1.,
            eligible=dict(p)[dates[0]]!='q34')
    result=refine(original,dict(cost=1.,eligible=True),evaluate=evaluate,dates=dates,
        roles=roles,miles=miles,budget=20)
    assert result['program']==winner and result['selected']['eligible']
    assert len(calls)==len(set(calls))==result['evaluations']<=20
    assert result['certified_fingerprint'] is not None
    assert len(result['passes'])==2


def test_terminal_budget_cannot_certify_partial_scan():
    dates=('2026-10-01','2026-10-02')
    roles=dict(q3='quality',q32='quality',q34='quality',e4='easy',e38='easy',e36='easy')
    miles=dict(q3=3.,q32=3.2,q34=3.4,e4=4.,e38=3.8,e36=3.6)
    original=((dates[0],'q3'),(dates[1],'e4'))
    result=refine(original,dict(cost=1.,eligible=True),evaluate=lambda p:dict(cost=.5,eligible=True),
        dates=dates,roles=roles,miles=miles,budget=1)
    assert result['evaluations']==1 and result['budget_exhausted']
    assert result['certified_fingerprint'] is None


def test_disjoint_cheaper_tail_does_not_discard_scanned_visible_edit(monkeypatch):
    import run_analysis.adaptive.native_quality_final_refinement as module
    dates=('2026-10-01','2026-10-02','2026-10-03','2026-10-04')
    original=((dates[0],'q0'),(dates[1],'e0'),(dates[2],'q0'),(dates[3],'e0'))
    visible=((dates[0],'q1'),(dates[1],'e1'),(dates[2],'q0'),(dates[3],'e0'))
    tail=((dates[0],'q0'),(dates[1],'e0'),(dates[2],'q1'),(dates[3],'e1'))
    combined=((dates[0],'q1'),(dates[1],'e1'),(dates[2],'q1'),(dates[3],'e1'))
    monkeypatch.setattr(module,'quality_easy_neighbors',lambda *a,**kw:iter((visible,tail)))
    monkeypatch.setattr(module,'decimal_pair_neighbors',lambda *a,**kw:iter(()))
    costs={original:1.,visible:.9,tail:.8,combined:.7}
    result=refine(original,dict(cost=1.,eligible=True),evaluate=lambda p:dict(cost=costs[p],eligible=True),
        dates=dates,roles=dict(q0='quality',q1='quality',e0='easy',e1='easy'),
        miles=dict(q0=3.,q1=3.2,e0=4.,e1=3.8),budget=10)
    assert result['program']==combined
    assert result['promising_edit_reuse'][0]['adoptions']==1
    assert result['evaluations']==3
