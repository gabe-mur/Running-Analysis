"""Attachment precondition controls; fixtures are NOT daily behavioral evidence."""
import json

import pytest

from scripts.release_c1_live_planner import race_closeout_proof


def proof_paths(tmp_path,*,days=21,dose_edit=.3):
    paths=[]
    for kind in ('race','omission','general'):
        root=tmp_path/kind;root.mkdir()
        event=dict(refinement=dict(calendar_merge_doses=dict(max_full_evaluations=32)))
        if kind!='general':event['race_preparation']=dict(phase='build')
        summary=dict(status='bounded_screen_complete',completed_days=days,failures=[],events=[event],
            search_policy=dict(calendar_merge_evaluations=32))
        if kind=='omission':summary['missed_runs']=[dict(status='not_completed')]
        audit=dict(status=summary['status'],completed_days=days,failures=[],decisions=1,
            score_replay_verified=True,decision_diagnostics=[dict(eligibility=True)],
            churn={'4':dict(date_edits=0,type_edits=0,largest_absolute_dose_edit=dose_edit),'7':{}},
            delivered_funding_ratio=1.)
        (root/'lifecycle-summary.json').write_text(json.dumps(summary))
        path=root/'daily-audit.json';path.write_text(json.dumps(audit));paths.append(path)
    return paths


def test_attachment_records_all_three_distinct_scopes(tmp_path):
    result=race_closeout_proof(proof_paths(tmp_path))
    assert {r['kind'] for r in result}=={'race','omission','general'}
    assert all(r['completed_days']==21 and r['audit_sha256'] and r['summary_sha256'] for r in result)


@pytest.mark.parametrize('changes',[dict(days=14),dict(dose_edit=.6)])
def test_attachment_cannot_turn_partial_or_failed_screen_into_a_pass(tmp_path,changes):
    with pytest.raises(ValueError,match='gate did not pass'):
        race_closeout_proof(proof_paths(tmp_path,**changes))


def test_three_copies_of_one_scope_do_not_close_three_release_gates(tmp_path):
    paths=proof_paths(tmp_path)
    with pytest.raises(ValueError,match='separate'):
        race_closeout_proof([paths[0]]*3)
    summary=paths[2].parent/'lifecycle-summary.json'
    value=json.loads(summary.read_text());value['events'][0]['race_preparation']=dict(phase='build')
    summary.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='not distinct'):
        race_closeout_proof(paths)
