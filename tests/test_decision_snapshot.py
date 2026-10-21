from dataclasses import replace
from datetime import timedelta
import json
import sqlite3

import pytest

from run_analysis.athlete_projection import PaceSample, SessionEvidence, accounting_signature
from run_analysis.program_constraints import DateRestriction
from run_analysis.recovery_response import ResponseExpectation
from run_analysis.session_decision import CurrentContext, DatedResponse
from scripts.decision_snapshot import (
    build_snapshot, capture_app_inputs, evaluate_snapshot, resolve_current,
    restore, save_snapshot, serialize,
)
from test_r3_decision_controls import AT, REF, context, contracts, programme, row, q


def freeze(**updates):
    values = dict(as_of=AT, timezone='UTC', context=context(), stage_key='A1-stage',
        stage_issued_at=AT, source='explicit test stage', history=programme(cycles=range(-2, 0), source='observed'),
        phase_history=(), contracts=contracts(), reference=REF, resting_hr=50, maximum_hr=200,
        references_available_at=AT, references_source='fixture configuration',
        measurement_context=json.dumps({'resting_hr': 50, 'max_hr': 200}))
    values.update(updates)
    return build_snapshot(**values)


def evaluate(state):
    return evaluate_snapshot(state, (row(0, 3.5), row(2, 4.5, 'quality')), ('easy', 'quality'), reference=REF)


def test_cached_restore_keeps_identity_checks_and_independent_contracts(tmp_path):
    from hashlib import sha256
    from run_analysis.adaptive.decision_snapshot import save_committed_snapshot, _restore_canonical
    state=freeze()
    content=serialize(state)
    digest=sha256(content.encode()).hexdigest()
    _restore_canonical.cache_clear()
    first=restore(content,expected_fingerprint=digest)
    object.__setattr__(first, 'source', 'caller-local edit')
    second=restore(content,expected_fingerprint=digest)
    assert second.contracts==state.contracts
    assert second.source==state.source
    assert _restore_canonical.cache_info().hits==1
    with pytest.raises(ValueError,match='recorded identity'):
        restore(content,expected_fingerprint='wrong')
    path=tmp_path/'committed.json'
    save_committed_snapshot(path,content,expected_fingerprint=digest)
    assert path.read_text()==content
    save_committed_snapshot(path,content,expected_fingerprint=digest)
    path.write_text('changed')
    with pytest.raises(ValueError,match='different decision evidence'):
        save_committed_snapshot(path,content,expected_fingerprint=digest)


def test_snapshot_reload_reproduces_decision_score_and_fingerprint(tmp_path):
    state = freeze()
    path = tmp_path/'snapshot.json'
    save_snapshot(path, state)
    save_snapshot(path, state)
    loaded = restore(path.read_text(), expected_fingerprint=state.fingerprint)
    assert state == loaded
    assert state.demonstrated() == loaded.demonstrated()
    before, after = evaluate(state), evaluate(loaded)
    assert before == after
    assert before['decision_eligible']
    assert not state.aerobic_reference.sessions  # No fabricated pace knots.
    assert state.demonstrated()['windows']['28']['history_status'] == 'recorded_only'
    with pytest.raises(ValueError, match='different decision'):
        save_snapshot(path, freeze(source='different declared stage source'))


def test_historical_timing_default_requires_explicit_identity_preserving_migration():
    from hashlib import sha256
    from scripts.decision_snapshot import migrate_snapshot_defaults
    value=json.loads(serialize(freeze()))
    def strip(v):
        if isinstance(v,dict):
            if v.get('type')=='PurposeTimingPolicy':
                v['fields'].pop('grace_fraction')
            for child in v.values():
                strip(child)
        elif isinstance(v,list):
            for child in v:
                strip(child)
    strip(value)
    original=json.dumps(value,sort_keys=True,separators=(',',':'))
    digest=sha256(original.encode()).hexdigest()
    with pytest.raises(ValueError,match='Canonical'):
        restore(original,expected_fingerprint=digest)
    migrated=migrate_snapshot_defaults(original,expected_fingerprint=digest)
    assert migrated['state'].context.timing.grace_fraction==0.
    assert migrated['source_fingerprint']==digest
    assert migrated['migrated_fingerprint']!=digest
    assert migrated['changes']==[dict(type='PurposeTimingPolicy',field='grace_fraction',historical_default=0.)]


def test_future_observations_reports_and_restrictions_cannot_change_snapshot():
    base = freeze()
    future = row(2, 3.5, source='observed')
    unknown_response = DatedResponse(future.key, None, future.end,
        ResponseExpectation('no expectation'), 'future import', effort=9)
    state = freeze(history=(*base.history, future),
        contexts=(CurrentContext((AT+timedelta(days=1)).date(), AT+timedelta(days=1), 'health_concern', 'future'),),
        responses=(unknown_response,), restrictions=(DateRestriction('future', AT.date(),
            (AT+timedelta(days=2)).date(), AT+timedelta(days=1), 'reserved_rest', 'future'),))
    assert state.fingerprint == base.fingerprint
    assert evaluate(state) == evaluate(base)


def test_available_late_upload_changes_only_later_snapshot():
    base = freeze()
    late = replace(row(-20, 3.5, source='observed'), available_at=AT+timedelta(hours=1))
    assert freeze(history=(*base.history, late)).fingerprint == base.fingerprint
    later = freeze(as_of=AT+timedelta(hours=2), history=(*base.history, late))
    assert late.key in {r.key for r in later.history}
    assert len(later.history) == len(base.history)+1


def test_source_order_does_not_choose_away_conflicting_health():
    health = CurrentContext((AT-timedelta(days=2)).date(), AT-timedelta(days=2), 'health_concern', 'health source')
    usual = CurrentContext(AT.date(), AT, 'usual', 'other source')
    first, second = freeze(contexts=(health, usual)), freeze(contexts=(usual, health))
    assert first.fingerprint == second.fingerprint
    assert first.current.state == 'conflicting'
    assert not evaluate(first)['decision_eligible']
    cleared = freeze(contexts=(health, replace(usual, source=health.source)))
    assert cleared.current.state == 'usual'
    assert evaluate(cleared)['decision_eligible']
    assert resolve_current((health,), as_of=AT, timezone='UTC').state == 'health_concern'


def test_health_uses_report_date_not_future_planned_workout_date():
    tired = CurrentContext((AT-timedelta(days=1)).date(), AT-timedelta(days=1), 'little_tired', 'report')
    state = freeze(contexts=(tired,))
    assert evaluate(state)['suitability']['action'] == 'request_current_context'
    assert freeze().current is None


def test_protected_dates_go_through_snapshot_into_actual_constraint_check():
    restriction = DateRestriction('rest', AT.date(), (AT+timedelta(days=1)).date(), AT, 'reserved_rest', 'user')
    base = evaluate(freeze())
    blocked = evaluate(freeze(restrictions=(restriction,)))
    assert not blocked['decision_eligible']
    assert blocked['constraints']['status'] == 'constraint_conflict'
    assert blocked['evaluation'].total_cost == base['evaluation'].total_cost
    assert blocked['evaluation'].fingerprint != base['evaluation'].fingerprint


def test_hr_reference_changes_need_available_recompiled_measurement_context():
    with pytest.raises(ValueError, match='recompile'):
        freeze(maximum_hr=190)
    with pytest.raises(ValueError, match='Available sourced'):
        freeze(references_available_at=AT+timedelta(seconds=1))


def test_original_projection_pace_samples_survive_and_are_causally_filtered():
    physical = row(-1, 3.5, source='observed')
    evidence = SessionEvidence(physical.key, physical.start, physical.end, physical.available_at,
        'easy', physical.miles*1609.344, physical.demand[0].low*60, q(999), accounting_signature(50, 200, REF.policy),
        (PaceSample('easy', 120, 300, 'original steady window', .6),))
    state = freeze(history=(physical,), session_evidence=(evidence,))
    assert state.aerobic_reference.sessions[0][0] == physical.key
    assert state.athlete.sessions[0].cardio == physical.demand[1]  # No double/old accounting.
    with pytest.raises(ValueError, match='same physical'):
        freeze(history=(physical,), session_evidence=(replace(evidence, meters=1),))
    with pytest.raises(ValueError, match='same frozen HR'):
        freeze(history=(physical,), session_evidence=(replace(evidence, accounting_key=accounting_signature(50, 190, REF.policy)),))


def test_stage_and_collections_are_frozen_not_rebased_or_mutable():
    supplied = contracts()
    original = freeze(contracts=supplied)
    supplied.clear()
    assert len(original.contracts) == 3
    assert original.context.policy.weekly_miles == 20
    assert restore(serialize(original), expected_fingerprint=original.fingerprint) == original
    with pytest.raises(ValueError, match='recorded identity'):
        restore(serialize(original)+' ', expected_fingerprint=original.fingerprint)


@pytest.fixture
def app_db():
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.executescript('''
        CREATE TABLE app_state(key TEXT,value_json TEXT,updated_at_utc TEXT);
        CREATE TABLE activities(id INTEGER,activity_id TEXT,activity_uid TEXT,created_at_utc TEXT,updated_at_utc TEXT);
        CREATE TABLE run_overrides(activity_id TEXT,perceived_exertion REAL,health_tag TEXT,illness INTEGER,include_in_model INTEGER,workout_type TEXT);
        CREATE TABLE source_files(id INTEGER,imported_at_utc TEXT);
        CREATE TABLE activity_sources(activity_id INTEGER,source_file_id INTEGER,source_activity_index INTEGER,is_primary INTEGER);
        CREATE TABLE activity_sensor_streams(activity_id INTEGER,source_file_id INTEGER,source_activity_index INTEGER,payload_json TEXT);
    ''')
    stamp = (AT-timedelta(days=1)).isoformat()
    db.execute('INSERT INTO activities VALUES (1,?,?,?,?)', ('run', 'uid', stamp, stamp))
    # Recording is imported AFTER this fixture's prior physical run ends.
    db.execute('INSERT INTO source_files VALUES (1,?)', ((AT-timedelta(hours=6)).isoformat(),))
    db.execute('INSERT INTO activity_sources VALUES (1,1,0,1)')
    db.execute('INSERT INTO activity_sensor_streams VALUES (1,1,0,?)',
        (json.dumps({'session_fields': {'workout_rpe': 40, 'workout_feel': 75}}),))
    db.commit()
    yield db
    db.close()


def capture(db, **kwargs):
    return capture_app_inputs(db, captured_at=AT, timezone='UTC', activity_keys={1: 'observed:-1'}, **kwargs)


def test_db_fit_feedback_no_expectation_is_unknown_not_fabricated_effort(app_db):
    changes = app_db.total_changes
    inputs = capture(app_db)
    assert app_db.total_changes == changes
    assert not app_db.in_transaction
    assert not inputs.contexts
    response = inputs.responses[0]
    assert (response.effort, response.feeling, response.expected_at) == (4., 75., None)
    assert response.available_at == AT-timedelta(hours=6)
    state = freeze(history=(row(-1, 8.4, 'long', source='observed'),), app_inputs=inputs)
    result = evaluate(state)
    assert result['decision_eligible']
    assert result['suitability']['responses'][response.key]['channels']['effort']['status'] == 'unknown'
    assert evaluate(restore(serialize(state), expected_fingerprint=state.fingerprint)) == result


def test_first_fit_capture_does_not_make_old_bad_feeling_newer_than_current_report(app_db):
    from run_analysis.health_reports import record_health_report
    app_db.execute("UPDATE activity_sensor_streams SET payload_json=?",
        (json.dumps({'session_fields': {'workout_rpe':40,'workout_feel':25}}),))
    app_db.commit()
    record_health_report(app_db, 'normal', reported_at=AT,
                         source='actual current self-report')
    prior=row(-1,8.4,'long',source='observed')
    expectations={prior.key:(prior.start-timedelta(hours=1),ResponseExpectation('issued fixture',feeling=q(75)))}
    first=capture(app_db,expectations=expectations)
    assert first.responses[0].available_at==AT-timedelta(hours=6)
    assert first.contexts[-1].available_at>first.responses[0].available_at
    checked=evaluate(freeze(history=(prior,),app_inputs=first))
    assert checked['suitability']['action']!='request_current_context'
    # A first capture after process restart retains this chronology without
    # inventing an earlier capture or relying on a previous in-memory object.
    later=capture_app_inputs(app_db,captured_at=AT+timedelta(hours=1),timezone='UTC',
                            activity_keys={1:'observed:-1'},expectations=expectations)
    assert later.responses[0].available_at==first.responses[0].available_at


def test_late_conflicting_fit_report_remains_new_evidence(app_db):
    from run_analysis.health_reports import record_health_report
    record_health_report(app_db,'normal',reported_at=AT-timedelta(hours=2),source='actual report')
    app_db.execute('INSERT INTO source_files VALUES (2,?)',((AT-timedelta(hours=1)).isoformat(),))
    app_db.execute('INSERT INTO activity_sources VALUES (1,2,0,0)')
    app_db.execute('INSERT INTO activity_sensor_streams VALUES (1,2,0,?)',
        (json.dumps({'session_fields':{'workout_rpe':40,'workout_feel':25}}),))
    app_db.commit()
    captured=capture(app_db)
    assert captured.responses[0].conflict
    assert captured.responses[0].available_at==AT-timedelta(hours=1)
    checked=evaluate(freeze(history=(row(-1,8.4,'long',source='observed'),),app_inputs=captured))
    assert checked['suitability']['action']=='request_current_context'


def test_manual_feedback_preserves_provenance_and_does_not_refresh_when_unchanged(app_db):
    app_db.execute('INSERT INTO run_overrides VALUES (?,?,?,?,?,?)', ('run', 8, 'normal', 0, 1, None))
    app_db.commit()
    previous = row(-1, 8.4, 'long', source='observed')
    expectations = {previous.key: (previous.start-timedelta(hours=1), ResponseExpectation('issued', effort=q(4)))}
    first = capture(app_db, expectations=expectations)
    assert first.responses[0].effort == 8 and 'manual' in first.responses[0].source
    state = freeze(history=(previous,), app_inputs=first)
    assert evaluate(state)['suitability']['action'] == 'request_current_context'
    second = capture_app_inputs(app_db, captured_at=AT+timedelta(hours=1), timezone='UTC',
        activity_keys={1: previous.key}, expectations=expectations, previous=first)
    assert second.responses == first.responses
    app_db.execute('UPDATE run_overrides SET perceived_exertion=5')
    app_db.commit()
    changed = capture_app_inputs(app_db, captured_at=AT+timedelta(hours=1), timezone='UTC',
        activity_keys={1: previous.key}, expectations=expectations, previous=first)
    assert changed.responses[0].available_at > first.responses[0].available_at
    assert first.responses[0].effort == 8


def test_db_dated_health_and_protected_days_are_used_not_default_normal(app_db):
    stamp = AT.isoformat()
    app_db.execute('INSERT INTO app_state VALUES (?,?,?)', ('current_health', json.dumps({'health_status':'little_tired',
        'planned_at': (AT+timedelta(days=5)).isoformat()}), stamp))
    app_db.execute('INSERT INTO app_state VALUES (?,?,?)', ('weekly_forced_rest_dates', json.dumps([AT.date().isoformat()]), stamp))
    app_db.commit()
    inputs = capture(app_db)
    assert inputs.contexts[0].applies_on == AT.date()
    assert inputs.contexts[0].state == 'unverified'
    assert not inputs.health_report_time_known
    state = freeze(history=(row(-1, 8.4, 'long', source='observed'),), app_inputs=inputs)
    assert not evaluate(state)['decision_eligible']


def test_conflicting_fit_reports_remain_visible(app_db):
    app_db.execute('INSERT INTO source_files VALUES (2,?)', ((AT-timedelta(hours=1)).isoformat(),))
    app_db.execute('INSERT INTO activity_sources VALUES (1,2,0,0)')
    app_db.execute('INSERT INTO activity_sensor_streams VALUES (1,2,0,?)',
        (json.dumps({'session_fields': {'workout_rpe': 80, 'workout_feel': 75}}),))
    app_db.commit()
    inputs = capture(app_db)
    assert inputs.responses[0].conflict
    state = freeze(history=(row(-1, 8.4, 'long', source='observed'),), app_inputs=inputs)
    assert not evaluate(state)['decision_eligible']


def test_mutable_database_cannot_be_misrepresented_as_historical_snapshot(app_db):
    app_db.execute('UPDATE activities SET updated_at_utc=?', ((AT+timedelta(days=1)).isoformat(),))
    app_db.commit()
    with pytest.raises(ValueError, match='newer than'):
        capture(app_db)
    future = replace(capture_app_inputs(app_db, captured_at=AT+timedelta(days=1), timezone='UTC',
        activity_keys={1:'observed:-1'}), captured_at=AT+timedelta(days=1))
    with pytest.raises(ValueError, match='backdated'):
        freeze(app_inputs=future)


def test_no_timestamp_means_no_response_bounds():
    with pytest.raises(ValueError, match='timestamp'):
        DatedResponse('x', None, AT, ResponseExpectation('not issued', effort=q(5)), 'capture', effort=4)


def test_missing_health_row_does_not_clear_a_captured_restriction(app_db):
    app_db.execute('INSERT INTO app_state VALUES (?,?,?)', ('current_health',
        json.dumps({'health_status':'pain_or_injury_concern'}), AT.isoformat()))
    app_db.commit()
    previous = capture(app_db)
    app_db.execute('DELETE FROM app_state')
    app_db.commit()
    missing = capture(app_db, previous=previous)
    assert missing.contexts == previous.contexts
    app_db.execute('INSERT INTO app_state VALUES (?,?,?)', ('current_health',
        json.dumps({'health_status':'normal'}), AT.isoformat()))
    app_db.commit()
    uncleared = capture(app_db, previous=previous)
    assert uncleared.contexts[0].state == 'health_concern'
    # Only an independently dated current report can clear that captured source.
    explicit = CurrentContext(AT.date(), AT+timedelta(seconds=1), 'usual', 'app:current_health')
    state = freeze(as_of=AT+timedelta(seconds=1),
        history=(row(-1,8.4,'long',source='observed'),),app_inputs=uncleared,contexts=(explicit,))
    assert state.current.state == 'usual'


def test_automatic_health_timestamp_refresh_is_not_a_new_readiness_report(app_db):
    app_db.execute('INSERT INTO app_state VALUES (?,?,?)', ('current_health',
        json.dumps({'health_status':'normal'}), AT.isoformat()))
    app_db.commit()
    original = capture(app_db)
    assert original.contexts == () and original.saved_health_status == 'normal'
    app_db.execute('UPDATE app_state SET updated_at_utc=?', ((AT+timedelta(hours=1)).isoformat(),))
    app_db.commit()
    refreshed = capture_app_inputs(app_db,captured_at=AT+timedelta(hours=1),timezone='UTC',
        activity_keys={1:'observed:-1'},previous=original)
    assert refreshed.contexts == ()
    assert not refreshed.health_report_time_known


def test_run_health_exclusion_keeps_physical_load_but_not_capacity_samples(app_db):
    app_db.execute('INSERT INTO run_overrides VALUES (?,?,?,?,?,?)', ('run', None, 'illness', 1, 0, None))
    app_db.commit()
    actual = row(-1, 8.4, 'long', source='observed')
    state = freeze(history=(actual,), app_inputs=capture(app_db))
    assert state.history == (actual,)
    assert not state.athlete.sessions[0].capacity_eligible
    assert state.demonstrated()['windows']['28']['run_count'] == 1
    assert state.demonstrated()['windows']['28']['reference_eligible']['keys'] == []


def test_real_binary_fit_import_can_feed_existing_feedback_resolver(tmp_path):
    from datetime import datetime, timezone
    from run_analysis.db import connect, initialize
    from run_analysis.importer import import_files
    from fit_builder import FitWriter, UINT8, UINT32, ENUM, fit_timestamp
    writer = FitWriter()
    start = AT-timedelta(days=1)
    for index in range(3):
        writer.message(20, {253:(UINT32,fit_timestamp(start+timedelta(seconds=10*index))),
            3:(UINT8,145),5:(UINT32,2000*index)})
    writer.message(18, {2:(UINT32,fit_timestamp(start)),5:(ENUM,1),7:(UINT32,20000),
        8:(UINT32,20000),9:(UINT32,4000),192:(UINT8,75),193:(UINT8,40)})
    path = tmp_path/'run.fit'
    path.write_bytes(writer.build())
    db = connect(tmp_path/'test.sqlite')
    try:
        initialize(db)
        imported = import_files(db, tmp_path, default_timezone='UTC', paths=[path])
        assert not imported.failed_files
        activity_id = db.execute('SELECT id FROM activities').fetchone()[0]
        inputs = capture_app_inputs(db, captured_at=datetime.now(timezone.utc)+timedelta(seconds=1),
            timezone='UTC', activity_keys={activity_id:'physical-fit'})
        assert inputs.responses[0].effort == 4.
        assert inputs.responses[0].feeling == 75.
        assert inputs.responses[0].expected_at is None
        assert inputs.responses[0].source.startswith('captured app feedback; effort=fit:')
    finally:
        db.close()
