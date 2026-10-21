from copy import deepcopy

from scripts.compare_elapsed_quality_response import diagnostic_gate, summarize


def summaries():
    phase = dict(baseline_count=4, valid_pairs=4, original_mae=.1, candidate_mae=.08)
    return {g:dict(baseline_count=4, compared=4, methods={'conditional_recorded':dict(
        invalid_hr_count=0, phases={p:deepcopy(phase) for p in ('whole','warmup','work','recovery','cooldown')})})
        for g in ('all','intervals','sustained')}


def test_conditional_gate_is_not_passed_by_whole_load_phase_cancellation():
    s = summaries()
    assert diagnostic_gate(s)['passed']
    s['intervals']['methods']['conditional_recorded']['phases']['warmup']['candidate_mae']=.11
    r = diagnostic_gate(s)
    assert not r['passed']
    assert 'intervals:warmup:higher MAE' in r['failures']


def test_lost_coverage_invalid_hr_and_work_nonimprovement_cannot_pass():
    s = summaries()
    s['all']['compared']=3
    s['intervals']['methods']['conditional_recorded']['invalid_hr_count']=1
    s['sustained']['methods']['conditional_recorded']['phases']['work']['candidate_mae']=.1
    r = diagnostic_gate(s)
    assert not r['passed']
    assert 'all:lost conditional coverage' in r['failures']
    assert 'intervals:invalid HR' in r['failures']
    assert 'sustained:work:no improvement' in r['failures']


def test_unknown_original_evidence_stays_visible_not_zero_error():
    r = summarize([{'elapsed_candidate':dict(status='no_recent_quality_pace',
        baseline_status='no_recent_quality_pace')}])
    assert r['total']==1 and r['baseline_count']==0 and r['compared']==0
    assert r['statuses']=={'no_recent_quality_pace':1}
    assert r['methods']['conditional_recorded']['phases']['whole']['candidate_mae'] is None


def test_invalid_new_load_is_retained_in_coverage_and_gate_not_silently_dropped():
    a, b = dict(fraction_error=.1), dict(fraction_error=None)
    cell = dict(original=dict(whole=a,phases={'work':a}), metrics=dict(whole=b,phases={'work':b}),
                diagnostics={'hr_outside_reference':True})
    r = summarize([{'elapsed_candidate':dict(status='compared',baseline_status='diagnosed',
        conditional_recorded=cell,conditional_prior_pace=cell)}])
    p = r['methods']['conditional_recorded']['phases']['whole']
    assert r['compared']==1 and p['baseline_count']==1 and p['valid_pairs']==0
    assert p['candidate_mae'] is None
    assert not diagnostic_gate({g:r for g in ('all','intervals','sustained')})['passed']
