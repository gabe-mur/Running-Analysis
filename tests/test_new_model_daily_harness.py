from datetime import datetime, timedelta, timezone
from itertools import product
import json

import pytest

from scripts.new_model_daily_harness import Dose, HarnessPolicy, neighbors, prototype_search as search, churn, demo


def test_search_can_insert_delete_move_change_role_and_distance():
    dates = ('2026-09-27','2026-09-28','2026-09-29')
    p = ((dates[0],'easy'),(dates[2],'long'))
    results = set(neighbors(p,dates,('easy','long')))
    assert ((dates[2],'long'),) in results
    assert ((dates[0],'long'),(dates[2],'long')) in results
    assert ((dates[1],'easy'),(dates[2],'long')) in results
    assert ((dates[0],'long'),(dates[2],'easy')) in results
    assert all(len(dict(r))==len(r) for r in results)


def test_bounded_search_is_deterministic_and_tiny_case_matches_exhaustive():
    dates = ('2026-09-27','2026-09-28','2026-09-29')
    target = {dates[0]:'easy',dates[2]:'long'}
    def evaluate(p):
        actual = dict(p)
        return dict(cost=sum(actual.get(d)!=target.get(d) for d in dates),eligible=True)
    args = dict(dates=dates,keys=('easy','long'),previous=(),evaluate=evaluate,budget=100)
    first,second = search(**args),search(**args)
    assert first['program']==second['program']==tuple(target.items())
    exhaustive = [tuple((d,k) for d,k in zip(dates,ks) if k)
                  for ks in product((None,'easy','long'),repeat=3)]
    assert first['cost']==min(evaluate(p)['cost'] for p in exhaustive)
    assert first['local_neighborhood_complete']


def test_budget_exhaustion_and_constraint_rejection_are_explicit():
    def evaluate(p):
        return dict(cost=-len(p),eligible=all(k!='forbidden' for _,k in p))
    result = search(dates=('2026-09-27','2026-09-28'),keys=('easy','forbidden'),
        previous=(),evaluate=evaluate,budget=2)
    assert result['evaluations']==2 and result['budget_exhausted']
    assert all(k!='forbidden' for _,k in result['program'])


def test_prior_is_candidate_not_a_lock_and_same_cost_prefers_it():
    date = '2026-09-27'
    args = dict(dates=(date,),keys=('easy','long'),previous=((date,'easy'),),budget=20)
    tied = search(**args,evaluate=lambda p:dict(cost=0.,eligible=True))
    assert tied['program']==args['previous']
    changed = search(**args,evaluate=lambda p:dict(cost=-sum(k=='long' for _,k in p),eligible=True))
    assert changed['program']==((date,'long'),)


def test_churn_distinguishes_dates_roles_and_distance():
    d = datetime(2026,9,27,tzinfo=timezone.utc).date()
    catalog = {x.key:x for x in (Dose('a','easy',4.,(('warmup',2400.),)),
        Dose('b','easy',4.4,(('warmup',2640.),)),Dose('l','long',8.,(('warmup',4800.),)))}
    before = ((d.isoformat(),'a'),((d+timedelta(days=2)).isoformat(),'l'))
    after = ((d.isoformat(),'b'),((d+timedelta(days=3)).isoformat(),'l'))
    edits = churn(before,after,d,catalog)['4']
    assert len(edits)==3 and edits[0]['distance_change']==pytest.approx(.4)
    assert not edits[0]['role_changed'] and sum(e['date_changed'] for e in edits)==2


@pytest.mark.parametrize('scenario',['expected','higher_cost','lower_edge','upper_edge','missed'])
def test_daily_driver_searches_uploads_and_rolls_without_legacy_planner(tmp_path,monkeypatch,scenario):
    # Fail loudly if anyone reconnects the old planner merely to get a calendar.
    import run_analysis.weekly_schedule as legacy
    def forbidden(*a,**k):
        raise AssertionError('Legacy planner used')
    monkeypatch.setattr(legacy,'build_weekly_schedule',forbidden,raising=False)
    from scripts import new_model_daily_harness as module
    calls = []
    real = module.evaluate_continuous
    def traced(*a,**k):
        calls.append((k['as_of'],len(a[1]),len(a[2])))
        return real(*a,**k)
    monkeypatch.setattr(module,'evaluate_continuous',traced)
    report = demo(tmp_path/'run',days=2,budget=500,hard_days=5,tail_days=3,scenario=scenario)
    assert len(calls)==sum(e['evaluations'] for e in report['events'])
    assert report['events'][0]['reason']=='initial'
    assert any(e['reason']=='daily_rollover' for e in report['events'])
    assert report['uploads']
    assert len(list((tmp_path/'run').glob('decision-*.json')))==len(report['events'])
    for event in report['events']:
        archived = json.loads((tmp_path/'run'/f"decision-{event['index']+1:04d}.json").read_text())
        assert archived['active'] == event['active']
        from scripts.decision_snapshot import restore, evaluate_snapshot
        from run_analysis.continuous_training_dose import ContinuousDoseReference
        state = restore((tmp_path/'run'/event['snapshot_file']).read_text(),
                        expected_fingerprint=event['snapshot_fingerprint'])
        assert [r.key for r in state.history] == sorted(event['history_keys'],
            key=lambda key: next(r.start for r in state.history if r.key == key))
        catalog = {d['key']:d for d in report['catalog']}
        reference = ContinuousDoseReference('sandbox upload accounting')
        planned = tuple(module.planned_accounted(f'candidate:{day}:{key}',
            datetime.fromisoformat(day+'T18:00:00+00:00'),
            Dose(key,catalog[key]['role'],catalog[key]['miles'],tuple(map(tuple,catalog[key]['phases']))),
            event['forecast_handoff_by_dose'][key],reference,available=state.as_of) for day,key in event['program'])
        replay = evaluate_snapshot(state,planned,tuple(key for _,key in event['program']),reference=reference)
        assert replay['evaluation'].fingerprint == event['selected']['score_fingerprint']
        assert replay['suitability'] == event['selected']['suitability']
        assert event['selected']['recovery_feasibility']=='not_established'
        assert event['selected']['suitability']['candidate_eligible']
        assert event['selected']['suitability']['status']=='provisionally_supported'
        assert not event['selected']['suitability']['selection_ready']
        assert all(d>=event['decision'] for d,_ in event['program'])
        assert len(event['history_keys'])==len(set(event['history_keys']))
    accepted = [u for u in report['uploads'] if u['status']=='consumed']
    for upload in accepted:
        c = upload['comparison']
        following = next(e for e in report['events'] if e['reason']=='post_upload'
                         and e['as_of'].startswith(upload['day']))
        observed = {k:(lo,hi) for k,lo,hi in following['history_cardio_bounds']}
        assert observed[c['actual_key']]==(c['actual_cardio']['low'],c['actual_cardio']['high'])
        assert following['decision']>upload['day']
        if scenario=='expected':
            assert c['routine_variation_without_new_replan_signal']
        if scenario=='higher_cost':
            assert not c['routine_variation_without_new_replan_signal']
        if scenario in {'lower_edge','upper_edge'}:
            assert c['distance_adherence']=='within'
            assert all(s=='within' for s in c['phase_adherence'])
    if scenario=='missed':
        assert report['uploads'][0]['status']=='missed'
    saved = json.loads((tmp_path/'run'/'report.json').read_text())
    assert saved['delivered']['runs']==len(accepted)


def test_stop_on_churn_does_not_relabel_failure_as_clean(tmp_path):
    result = demo(tmp_path/'run',days=5,budget=500,hard_days=5,tail_days=3,stop_on_churn=True)
    assert result['stop_reason']=='near_term_churn'
    assert result['days']<result['requested_days']
    assert any(result['events'][-1]['churn'].values())


def test_retried_upload_does_not_duplicate_physical_history(tmp_path,monkeypatch):
    from scripts.new_model_daily_harness import DailyHarness
    original = DailyHarness.upload
    seen = []
    def retry(self,payload,**kwargs):
        first = original(self,payload,**kwargs)
        size = len(self.history)
        second = original(self,payload,**dict(kwargs,available_at=kwargs['available_at']+timedelta(seconds=30)))
        assert second['status']=='duplicate' and len(self.history)==size
        assert json.loads(json.dumps(first['comparison']))==second['comparison']
        seen.append(first)
        return first
    monkeypatch.setattr(DailyHarness,'upload',retry)
    demo(tmp_path/'retry',days=1,budget=500,hard_days=5,tail_days=3)
    assert seen


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        HarnessPolicy(evaluations=0)
    with pytest.raises(ValueError):
        Dose('bad','easy',4.25,(('warmup',2400.),))
    with pytest.raises(ValueError,match='opening-search'):
        HarnessPolicy(complete_first_search='yes')
    for allowance in (-1,True,1.5):
        with pytest.raises(ValueError,match='final-winner'):
            HarnessPolicy(final_winner_evaluations=allowance)


def test_shared_complete_first_search_is_dispatched_and_restored_without_race(tmp_path,monkeypatch):
    from dataclasses import replace
    from run_analysis.config import load_config
    from scripts import new_model_daily_harness as module
    original_policy=module.HarnessPolicy
    monkeypatch.setattr(module,'HarnessPolicy',lambda *a,**k:replace(original_policy(*a,**k),
        complete_first_search=True,final_winner_evaluations=100))
    original_search=module.migrated_search
    dispatched=[]
    def traced(**kwargs):
        dispatched.append((kwargs['complete_first'],kwargs['final_winner_budget']))
        return original_search(**kwargs)
    monkeypatch.setattr(module,'migrated_search',traced)
    root=tmp_path/'shared'
    report=demo(root,days=1,budget=100,hard_days=3,tail_days=2)
    assert dispatched and all(row==(True,100) for row in dispatched)
    assert all(e['refinement']['opening']['strategy']=='complete_first' for e in report['events'])
    assert json.loads((root/'scenario.json').read_text())['search_policy']['complete_first_search']
    monkeypatch.setattr(module,'HarnessPolicy',original_policy)
    config=dict(load_config('config.yaml'),resting_hr=50,max_hr=200,timezone_default='UTC')
    resumed=module.DailyHarness.resume(root,config,as_of=datetime.fromisoformat(report['events'][-1]['as_of']))
    try:
        assert resumed.policy.complete_first_search
        assert resumed.policy.final_winner_evaluations==100
        assert getattr(resumed,'race_goal',None) is None
    finally:
        resumed.sandbox.close()


def test_search_cannot_bypass_decision_gate_even_for_empty_program(tmp_path, monkeypatch):
    from scripts import new_model_daily_harness as module
    original = module.evaluate_continuous
    def deny(*args, **kwargs):
        result = original(*args, **kwargs)
        # Simulate a current-context review: constraints/finite score are not enough.
        result['decision_eligible'] = False
        return result
    monkeypatch.setattr(module, 'evaluate_continuous', deny)
    with pytest.raises(ValueError, match='No revalidated feasible seed'):
        demo(tmp_path/'blocked', days=1, budget=10, hard_days=3, tail_days=2)


def test_accounted_harness_retains_between_work_recovery():
    from scripts.new_model_daily_harness import accounted, point
    from run_analysis.continuous_training_dose import ContinuousDoseReference
    at = datetime(2026,9,27,tzinfo=timezone.utc)
    row = accounted('test',at,4.,(('warmup',600.),('work',300.),('recovery',120.),
        ('work',300.),('cooldown',600.)),point(30.),'quality',ContinuousDoseReference('test'),
        source='planned',available=at)
    assert [b.recovery_after_seconds for b in row.bouts] == [120., None]


def test_resume_after_commit_before_memory_update_matches_uninterrupted_run(tmp_path,monkeypatch):
    from run_analysis.config import load_config
    from scripts.new_model_daily_harness import DailyHarness
    baseline = demo(tmp_path/'baseline',days=1,budget=500,hard_days=5,tail_days=3)
    real_sync = DailyHarness.sync_completions
    def crash_after_commit(self,as_of):
        if self.sandbox.completed_accounted(as_of):
            raise RuntimeError('simulated stop after durable completion')
        return real_sync(self,as_of)
    monkeypatch.setattr(DailyHarness,'sync_completions',crash_after_commit)
    with pytest.raises(RuntimeError,match='durable completion'):
        demo(tmp_path/'interrupted',days=1,budget=500,hard_days=5,tail_days=3)
    monkeypatch.setattr(DailyHarness,'sync_completions',real_sync)
    config = dict(load_config('config.yaml'),resting_hr=50,max_hr=200,timezone_default='UTC')
    expected = baseline['events'][-1]
    at = datetime.fromisoformat(expected['as_of'])
    resumed = DailyHarness.resume(tmp_path/'interrupted',config,as_of=at)
    try:
        assert len(resumed.history) == len(resumed.seed_keys)+1
        completed_day = baseline['uploads'][0]['day']
        assert completed_day not in dict(resumed.previous)
        assert completed_day not in resumed.active
        restored = resumed.replan(at,reason='post_upload')
        assert restored['program'] == expected['program']
        assert restored['snapshot_fingerprint'] == expected['snapshot_fingerprint']
        assert restored['selected'] == expected['selected']
        resumed.sync_completions(at)
        assert len(resumed.history) == len(resumed.seed_keys)+1
    finally:
        resumed.sandbox.close()


def test_atomic_plan_failure_rolls_back_every_slot(tmp_path,monkeypatch):
    from scripts.execution_upload_sandbox import ExecutionUploadSandbox
    from run_analysis.config import load_config
    original=ExecutionUploadSandbox.publish
    calls=0
    def fail(self,*args,**kwargs):
        nonlocal calls
        calls+=1
        result=original(self,*args,**kwargs)
        if calls==2:
            raise RuntimeError('failed second slot')
        return result
    monkeypatch.setattr(ExecutionUploadSandbox,'publish',fail)
    with pytest.raises(RuntimeError,match='second slot'):
        demo(tmp_path/'partial',days=1,budget=500,hard_days=5,tail_days=3)
    config=dict(load_config('config.yaml'),resting_hr=50,max_hr=200,timezone_default='UTC')
    sandbox=ExecutionUploadSandbox(tmp_path/'partial',config)
    try:
        assert sandbox.db.execute('SELECT count(*) FROM sandbox_issues').fetchone()[0]==0
        assert sandbox.decisions()==()
    finally:
        sandbox.close()


def test_committed_plan_recovers_missing_exports_after_crash(tmp_path,monkeypatch):
    from scripts.new_model_daily_harness import DailyHarness
    from run_analysis.config import load_config
    def crash(self):
        raise RuntimeError('crash before JSON export')
    monkeypatch.setattr(DailyHarness,'save',crash)
    with pytest.raises(RuntimeError,match='JSON export'):
        demo(tmp_path/'committed',days=1,budget=500,hard_days=5,tail_days=3)
    assert not list((tmp_path/'committed').glob('decision-*.json'))
    config=dict(load_config('config.yaml'),resting_hr=50,max_hr=200,timezone_default='UTC')
    at=datetime(2026,9,27,23,tzinfo=timezone.utc)
    resumed=DailyHarness.resume(tmp_path/'committed',config,as_of=at)
    try:
        assert len(resumed.events)==1 and resumed.active
        assert (tmp_path/'committed'/'decision-0001.json').exists()
        assert resumed.sandbox.state(at)['pending_scope']=='committed_current_calendar'
        assert len(resumed.sandbox.state(at)['pending'])==len(resumed.active)
    finally:
        resumed.sandbox.close()


def test_export_comparison_cache_checks_contents_and_preserves_json_semantics(monkeypatch):
    from run_analysis.adaptive import new_model_daily_harness as module
    module._verified_decision_exports.clear()
    module._decision_export_headers.clear()
    committed = '{"index": 0, "snapshot_file": "snapshot.json", "snapshot_fingerprint": "digest", "cache_test": "original", "nested": [1, 2]}'
    event = json.loads(committed)
    exported = json.dumps(event, indent=2)
    original_loads = json.loads
    calls = []
    def counted(value, *args, **kwargs):
        calls.append(value)
        return original_loads(value, *args, **kwargs)
    monkeypatch.setattr(module.json, 'loads', counted)
    first = module._validate_decision_export(committed, event, exported)
    second = module._validate_decision_export(committed, None, exported)
    assert calls == [exported, exported], 'Warm comparisons only decode the export, not the journal'
    assert first == second and first is not second
    # Equal byte length is not evidence of equal contents.
    with pytest.raises(ValueError, match='differs from committed journal'):
        module._validate_decision_export(committed, event, exported.replace('original', 'modified'))
    changed = committed.replace('original', 'modified')
    with pytest.raises(ValueError, match='differs from committed journal'):
        module._validate_decision_export(changed, original_loads(changed), exported)
    # An equivalent reformat is valid, but must first be checked.
    compact = json.dumps(event, separators=(',', ':'))
    module._validate_decision_export(committed, event, compact)
    assert calls[-1] == compact
    equivalent = '{"nested": [1.0, 2], "cache_test": "original", "index": 0, "snapshot_file": "snapshot.json", "snapshot_fingerprint": "digest"}'
    for _ in range(2):
        result = module._validate_decision_export(committed, None, equivalent)
        assert list(result) == ['nested', 'cache_test', 'index', 'snapshot_file', 'snapshot_fingerprint']
        assert type(result['nested'][0]) is float
    # Bound memory independently of decision size; no parsed objects are cached.
    monkeypatch.setattr(module, '_DECISION_EXPORT_CACHE_SIZE', 2)
    for i in range(3):
        data = {'index': i, 'cache_test': i, 'snapshot_file': 'snapshot.json', 'snapshot_fingerprint': 'digest'}
        text = json.dumps(data)
        module._validate_decision_export(text, data, text)
    assert len(module._verified_decision_exports) == 2
    assert len(module._decision_export_headers) == 2
    assert all(value is None for value in module._verified_decision_exports.values())


def test_resume_cache_isolated_mutations_and_same_stat_export_tampering(tmp_path):
    import os
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.config import load_config
    root = tmp_path/'cached'
    report = demo(root, days=1, budget=500, hard_days=5, tail_days=3)
    expected = [json.loads(path.read_text()) for path in sorted(root.glob('decision-*.json'))]
    config = dict(load_config('config.yaml'), resting_hr=50, max_hr=200, timezone_default='UTC')
    at = datetime.fromisoformat(report['events'][-1]['as_of'])
    first = DailyHarness.resume(root, config, as_of=at)
    try:
        first.events[0]['active'].clear()
        first.events[0]['program'].clear()
    finally:
        first.sandbox.close()
    second = DailyHarness.resume(root, config, as_of=at)
    try:
        assert second.events == expected
    finally:
        second.sandbox.close()
    path = root/'decision-0001.json'
    content, stat = path.read_text(), path.stat()
    assert 'initial' in content
    path.write_text(content.replace('initial', 'changed'))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    with pytest.raises(ValueError, match='differs from committed journal'):
        DailyHarness.resume(root, config, as_of=at)
    # Recovering a deleted export still uses the durable journal, even warm.
    path.unlink()
    recovered = DailyHarness.resume(root, config, as_of=at)
    try:
        assert recovered.events == expected
        assert json.loads(path.read_text()) == expected[0]
    finally:
        recovered.sandbox.close()
    # The committed source is also fetched afresh; never trust just the export.
    import sqlite3
    with sqlite3.connect(root/'sandbox.sqlite') as db:
        sequence, serialized = db.execute('SELECT sequence,event FROM sandbox_decisions ORDER BY sequence LIMIT 1').fetchone()
        db.execute('UPDATE sandbox_decisions SET event=? WHERE sequence=?',
                   (serialized.replace('initial', 'changed'), sequence))
    with pytest.raises(ValueError, match='differs from committed journal'):
        DailyHarness.resume(root, config, as_of=at)


@pytest.mark.parametrize('fallback',[False,True])
def test_missing_work_forecast_is_explicit_not_whole_catalogue_failure(tmp_path,monkeypatch,fallback):
    from dataclasses import replace
    from run_analysis.workload_accounting import Quantity
    from run_analysis.continuous_training_dose import ContinuousDoseReference
    from scripts.forecast_handoff import PhaseRateScenario
    from scripts.new_model_daily_harness import DailyHarness
    original = DailyHarness.__init__
    def no_quality_history(self,*args,**kwargs):
        kwargs['history'] = tuple(r for r in kwargs['history'] if r.role!='quality')
        keys = {r.key for r in kwargs['history']}
        kwargs['phase_history'] = tuple(r for r in kwargs['phase_history'] if r.key in keys)
        if fallback:
            rates = (PhaseRateScenario('work',Quantity(.8,2.,1.2,'explicit test rate'),
                kwargs['origin']-timedelta(days=1),ContinuousDoseReference('test').accounting_key,
                'controlled_threshold','explicit synthetic fallback, not athlete default'),)
            kwargs['catalog'] = tuple(replace(d,forecast_scope=replace(d.forecast_scope,rate_scenarios=rates))
                if d.role=='quality' else d for d in kwargs['catalog'])
        original(self,*args,**kwargs)
    monkeypatch.setattr(DailyHarness,'__init__',no_quality_history)
    result = demo(tmp_path/'run',days=1,budget=100,hard_days=3,tail_days=2)
    first = result['events'][0]
    assert first['forecast_coverage']==('complete_catalogue' if fallback else 'partial_catalogue')
    if not fallback:
        assert first['unresolved_forecasts']['quality-3x5']['reason']=='phase_rate_unavailable'
        assert all(k!='quality-3x5' for _,k in first['program'])
    else:
        h = first['forecast_handoff_by_dose']['quality-3x5']
        assert h['usable'] and any(p['source']=='declared_rate_scenario' for p in h['phase_diagnostics'])
    assert first['program']  # No quiet all-rest fallback.


def test_all_unsupported_forecasts_stop_explicitly_without_partial_publication(tmp_path,monkeypatch):
    from dataclasses import replace
    from scripts.new_model_daily_harness import DailyHarness
    original = DailyHarness.__init__
    def unscoped(self,*args,**kwargs):
        kwargs['catalog'] = tuple(replace(d,forecast_scope=None) for d in kwargs['catalog'])
        original(self,*args,**kwargs)
    monkeypatch.setattr(DailyHarness,'__init__',unscoped)
    with pytest.raises(ValueError,match='No supported or explicitly bounded'):
        demo(tmp_path/'run',days=1,budget=100,hard_days=3,tail_days=2)
    assert not list((tmp_path/'run').glob('snapshot-*.json'))
    assert not list((tmp_path/'run').glob('decision-*.json'))
