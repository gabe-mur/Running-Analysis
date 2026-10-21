from dataclasses import replace
from datetime import timedelta
import pytest

from run_analysis.staged_progression import review_recent_delivery,ProgressionPolicy
from scripts.stage_acceptance import StageLedger,accept_candidate
from test_staged_progression import stage,entry,MILE,AT as REVIEW_AT
from test_decision_snapshot import freeze
from test_r3_decision_controls import AT,REF,context,programme


def ledger_entries(s,*,start=None):
    start=start or s.issued_at
    rows=[]
    # Exactly 40 credited miles over 14 days, including two long opportunities.
    for i,miles in enumerate((8.4,3.2,4.,4.4)*2):
        e=entry(s,miles,key=str(i))
        when=start+timedelta(days=i*1.6)
        target=replace(e.contract.targets[0],quantity=replace(e.contract.targets[0].quantity,
            low=(miles-.2)*MILE,high=(miles+.2)*MILE,estimate=miles*MILE))
        contract=replace(e.contract,purpose='long_endurance' if i%4==0 else 'easy_volume',targets=(target,))
        done=replace(e.completion,key=str(i),start=when,end=when+timedelta(hours=2),available_at=when+timedelta(hours=2))
        rows.append(replace(e,scheduled_at=when,contract=contract,completion=done))
    return rows


@pytest.mark.parametrize('focus',['long_and_volume','easy_volume'])
def test_whole_distance_band_is_compliant_and_excess_not_growth_bonus(focus):
    s=stage(weekly_equivalent_miles=20.,policy=ProgressionPolicy(14,.05,focus,
        21. if focus=='easy_volume' else None))
    rows=ledger_entries(s)
    def review(rs):
        return review_recent_delivery(s,rs,s.review_at,ledger_complete=True,longest_prior_30d_miles=8.4)
    exact=review(rows)
    assert exact['status']=='conditional_development_candidate'
    if focus=='easy_volume':
        assert exact['candidate'].long_miles==s.long_miles
    for sign in (-1,1):
        edges=[replace(e,completion=replace(e.completion,distance_meters=replace(e.completion.distance_meters,
            low=e.contract.targets[0].quantity.estimate+sign*.2*MILE,
            high=e.contract.targets[0].quantity.estimate+sign*.2*MILE,
            estimate=e.contract.targets[0].quantity.estimate+sign*.2*MILE))) for e in rows]
        assert review(edges)['candidate']==exact['candidate']
    excess=replace(rows[1],completion=replace(rows[1].completion,distance_meters=replace(rows[1].completion.distance_meters,
        low=6*MILE,high=6*MILE,estimate=6*MILE)))
    assert review([rows[0],excess,*rows[2:]])['candidate']==exact['candidate']


def test_miss_does_not_rebase_or_permanently_block_later_evidence():
    s=stage(weekly_equivalent_miles=20.)
    rows=ledger_entries(s)
    rows[0]=replace(rows[0],completion=None)
    early=review_recent_delivery(s,rows,s.review_at,ledger_complete=True,longest_prior_30d_miles=8.4)
    assert early['status']=='recent_delivery_hold_targets'
    assert early['stage_target_unchanged']==s
    assert early['volume_accounting']['attribution']=='execution_shortfall'
    later=ledger_entries(s,start=s.review_at)
    later=[replace(e,key='later:'+e.key,completion=replace(e.completion,key='later:'+e.completion.key)) for e in later]
    result=review_recent_delivery(s,(*rows,*later),s.review_at+timedelta(days=14),ledger_complete=True,longest_prior_30d_miles=8.4)
    assert result['status']=='conditional_development_candidate'
    assert result['candidate'].weekly_equivalent_miles==21.


def test_underprescription_is_not_athlete_noncompliance_or_automatic_growth():
    s=stage(weekly_equivalent_miles=21.)
    rows=ledger_entries(s)  # Prescribed and completed 40 against required 42.
    r=review_recent_delivery(s,rows,s.review_at,ledger_complete=True,longest_prior_30d_miles=8.4)
    assert r['status']=='recent_delivery_hold_targets'
    assert r['delivery_problems']==[]
    a=r['volume_accounting']
    assert a['prescribed_miles']==pytest.approx(40.)
    assert a['credited_miles']==pytest.approx(40.)
    assert a['required_miles']==pytest.approx(42.)
    assert a['planner_shortfall_miles']==pytest.approx(2.)
    assert a['execution_gap_miles']==pytest.approx(0.)
    assert a['attribution']=='planner_underprescription'
    assert r['stage_target_unchanged']==s
    missed=[replace(rows[0],completion=None),*rows[1:]]
    both=review_recent_delivery(s,missed,s.review_at,ledger_complete=True,longest_prior_30d_miles=8.4)
    assert both['volume_accounting']['attribution']=='planner_and_execution'


def test_gradual_dated_requirement_does_not_manufacture_delivered_exposure():
    s=stage(weekly_equivalent_miles=20.,policy=ProgressionPolicy(14,.05,'easy_volume',21.,'gradual'))
    rows=ledger_entries(s)  # 40 miles: enough for a flat stage, not the rising path.
    result=review_recent_delivery(s,rows,s.review_at,ledger_complete=True,
        longest_prior_30d_miles=8.4,required_miles=40.92857142857143)
    assert result['status']=='recent_delivery_hold_targets'
    assert result['credited_miles']==pytest.approx(40.)
    assert result['volume_accounting']['attribution']=='planner_underprescription'
    assert result['stage_target_unchanged']==s


def test_unknown_prescription_is_not_reported_as_known_underfunding():
    s=stage(weekly_equivalent_miles=20.)
    rows=ledger_entries(s)
    first=rows[0]
    unknown=replace(first.contract.targets[0],quantity=replace(first.contract.targets[0].quantity,estimate=None))
    rows[0]=replace(first,contract=replace(first.contract,targets=(unknown,)))
    result=review_recent_delivery(s,rows,s.review_at,ledger_complete=True,longest_prior_30d_miles=8.4)
    assert result['volume_accounting']['attribution']=='unknown_prescription'
    assert result['volume_accounting']['prescribed_miles'] is None


def acceptance_fixture():
    old=stage(key='old',issued_at=AT-timedelta(days=14),weekly_equivalent_miles=19.,ordinary_miles=3.5,long_miles=8.)
    candidate=replace(old,key='new',issued_at=AT,weekly_equivalent_miles=20.,long_miles=8.4)
    review=dict(status='conditional_development_candidate',candidate=candidate,stage_key=old.key)
    state=freeze(stage_key=candidate.key,context=context())
    rows=programme(easy=((2,3.5),(6,3.6)))
    return old,review,state,rows


def test_persistent_acceptance_is_idempotent_and_cannot_rebootstrap(tmp_path):
    old,review,state,rows=acceptance_fixture()
    path=tmp_path/'stages.sqlite'
    ledger=StageLedger(path,seed=old)
    first=accept_candidate(ledger,review,state,rows,tuple(r.role for r in rows),reference=REF)
    assert first['accepted']
    assert accept_candidate(ledger,review,state,rows,tuple(r.role for r in rows),reference=REF)['acceptance']==first['acceptance']
    ledger.close()
    reopened=StageLedger(path)
    assert reopened.current==review['candidate']
    reopened.close()
    with pytest.raises(ValueError,match='rebootstrap'):
        StageLedger(path,seed=old)


def test_underfunded_candidate_is_not_accepted_despite_finite_score(tmp_path):
    old,review,state,rows=acceptance_fixture()
    ledger=StageLedger(tmp_path/'stages.sqlite',seed=old)
    result=accept_candidate(ledger,review,state,rows[:4],tuple(r.role for r in rows[:4]),reference=REF)
    assert not result['accepted'] and result['reason']=='candidate_stage_underfunded'
    assert ledger.current==old
    ledger.close()


def test_stage_acceptance_can_share_atomic_sandbox_publication_transaction(tmp_path):
    from scripts.execution_upload_sandbox import ExecutionUploadSandbox
    from run_analysis.config import load_config
    old,review,state,rows=acceptance_fixture()
    sandbox=ExecutionUploadSandbox(tmp_path/'sandbox',load_config(),create=True)
    ledger=StageLedger.for_sandbox(sandbox,seed=old)
    try:
        with pytest.raises(RuntimeError,match='publication failure'):
            with sandbox.db:
                sandbox.db.execute('BEGIN IMMEDIATE')
                result=accept_candidate(ledger,review,state,rows,tuple(r.role for r in rows),reference=REF)
                assert result['accepted'] and ledger.current==review['candidate']
                raise RuntimeError('publication failure')
        assert ledger.current==old
        a=accept_candidate(ledger,review,state,rows,tuple(r.role for r in rows),reference=REF)
        b=accept_candidate(ledger,review,state,rows,tuple(r.role for r in rows),reference=REF)
        assert a['acceptance']==b['acceptance']
        ledger.close()
        assert sandbox.db.execute('SELECT 1').fetchone()[0]==1
    finally:
        sandbox.close()
