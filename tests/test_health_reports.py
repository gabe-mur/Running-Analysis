from datetime import datetime,timedelta,timezone
import json
import sqlite3

import pytest

from run_analysis.health_reports import record_health_report,read_health_reports
from scripts.decision_snapshot import capture_app_inputs,resolve_current


AT=datetime(2026,9,27,12,tzinfo=timezone.utc)


def database():
    db=sqlite3.connect(':memory:')
    db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE app_state(key TEXT PRIMARY KEY,value_json TEXT,updated_at_utc TEXT)')
    return db


def capture(db,at):
    return capture_app_inputs(db,captured_at=at,timezone='UTC',activity_keys={})


def test_refresh_timestamp_does_not_refresh_user_health_report():
    with database() as db:
        record_health_report(db,'little_tired',reported_at=AT,source='explicit form')
        db.execute('INSERT INTO app_state VALUES (?,?,?)',('current_health',json.dumps({'health_status':'little_tired'}),
            (AT+timedelta(days=1)).isoformat()))
        result=capture(db,AT+timedelta(days=1))
        assert result.health_report_time_known
        assert result.contexts[0].available_at==AT
        assert result.contexts[0].applies_on==AT.date()
        assert len(read_health_reports(db,as_of=AT+timedelta(days=2)))==1
        # A later automatic cache write is irrelevant to an earlier journal
        # cutoff. Other mutable app metadata still cannot be backdated.
        earlier=capture(db,AT)
        assert earlier.contexts[0].available_at==AT


def test_explicit_normal_can_clear_prior_concern_but_future_report_cannot():
    with database() as db:
        record_health_report(db,'sick_or_recovering',reported_at=AT,source='explicit form',report_key='one')
        record_health_report(db,'normal',reported_at=AT+timedelta(days=1),source='explicit form',report_key='two')
        early=capture(db,AT)
        late=capture(db,AT+timedelta(days=1))
        assert resolve_current(early.contexts,as_of=AT,timezone='UTC').state=='health_concern'
        assert resolve_current(late.contexts,as_of=AT+timedelta(days=1),timezone='UTC').state=='usual'


def test_idempotent_retry_and_immutable_report_identity():
    with database() as db:
        for _ in range(2):
            record_health_report(db,'normal',reported_at=AT,source='form',report_key='one')
        assert len(read_health_reports(db,as_of=AT))==1
        with pytest.raises(ValueError,match='Immutable'):
            record_health_report(db,'little_tired',reported_at=AT,source='form',report_key='one')


def test_status_uses_journal_not_stale_cache_and_exposes_local_freshness():
    from run_analysis.recommendation_service import load_current_status
    with database() as db:
        db.execute('INSERT INTO app_state VALUES (?,?,?)',
            ('current_health',json.dumps({'health_status':'normal'}),AT.isoformat()))
        missing=load_current_status(db,as_of=AT)
        assert missing.health_reported_at is None and not missing.health_report_current
        # UTC midnight is still the previous day for this athlete.
        reported=AT.replace(hour=23)
        record_health_report(db,'little_tired',reported_at=reported,source='form')
        fresh=load_current_status(db,as_of=reported+timedelta(hours=2),timezone_name='America/New_York')
        assert fresh.health_status=='little_tired' and fresh.health_report_current
        assert fresh.health_reported_at==reported
        stale=load_current_status(db,as_of=reported+timedelta(hours=7),timezone_name='America/New_York')
        assert stale.health_status=='little_tired' and not stale.health_report_current
        # Fetching status must not refresh or fabricate a check-in.
        assert len(read_health_reports(db,as_of=reported+timedelta(days=1)))==1


def test_current_status_endpoint_serializes_report_metadata(tmp_path):
    from test_web_phase1 import _write_config
    from run_analysis.db import connect,initialize
    from run_analysis.web.app import create_app
    from fastapi.testclient import TestClient
    _write_config(tmp_path)
    (tmp_path/'data').mkdir()
    with connect(tmp_path/'data/test.sqlite') as db:
        initialize(db)
        record_health_report(db,'pain_or_injury_concern',reported_at=AT,source='form')
    with TestClient(create_app(tmp_path)) as client:
        response=client.get('/api/current-status')
        assert response.status_code==200
        assert response.json()['health_status']=='pain_or_injury_concern'
        assert response.json()['health_reported_at'] is not None
        assert response.json()['health_report_current'] is False
