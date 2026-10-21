import sqlite3
from contextlib import closing

import pytest

from run_analysis.read_cache import invariant_read,invariant_snapshot


@pytest.fixture
def readers(tmp_path):
    path = tmp_path / "cache.sqlite"
    with closing(sqlite3.connect(path)) as writer, closing(sqlite3.connect(path)) as reader:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE evidence (value INTEGER)")
        writer.execute("INSERT INTO evidence VALUES (1)")
        writer.commit()
        yield path, writer, reader


def test_reuses_across_connections_but_returns_independent_objects(readers):
    path, writer, reader = readers
    calls = []
    def build():
        calls.append(1)
        return {"values": [reader.execute("SELECT value FROM evidence").fetchone()[0]]}
    first = invariant_read(reader, "history", build)
    first["values"].append(99)
    with closing(sqlite3.connect(path)) as other:
        assert invariant_read(other, "history", build) == {"values": [1]}
    assert len(calls) == 1


def test_external_commit_and_wal_checkpoint_invalidate(readers):
    path, writer, reader = readers
    calls = []
    def build():
        calls.append(1)
        return reader.execute("SELECT value FROM evidence").fetchone()[0]
    assert invariant_read(reader, "history", build) == 1
    writer.execute("UPDATE evidence SET value=2")
    writer.commit()
    assert invariant_read(reader, "history", build) == 2
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert invariant_read(reader, "history", build) == 2
    assert len(calls) == 3


def test_transaction_snapshot_and_uncommitted_writes_never_borrow_cache(readers):
    path, writer, reader = readers
    read = lambda: reader.execute("SELECT value FROM evidence").fetchone()[0]
    assert invariant_read(reader, "history", read) == 1
    reader.execute("BEGIN")
    assert read() == 1
    writer.execute("UPDATE evidence SET value=2")
    writer.commit()
    assert invariant_read(writer, "history", lambda: 2) == 2
    assert invariant_read(reader, "history", read) == 1
    reader.rollback()
    reader.execute("UPDATE evidence SET value=3")
    assert invariant_read(reader, "history", read) == 3
    reader.rollback()
    assert invariant_read(reader, "history", read) == 2


def test_concurrent_write_during_build_is_not_published(readers):
    path, writer, reader = readers
    def changing():
        old = reader.execute("SELECT value FROM evidence").fetchone()[0]
        writer.execute("UPDATE evidence SET value=2")
        writer.commit()
        return old
    assert invariant_read(reader, "history", changing) == 1
    assert invariant_read(reader, "history", lambda: 2) == 2


def test_key_changes_and_exceptions_do_not_reuse_wrong_result(readers):
    path, writer, reader = readers
    assert invariant_read(reader, ("history", "config-a"), lambda: 1) == 1
    assert invariant_read(reader, ("history", "config-b"), lambda: 2) == 2
    def fail():
        raise ValueError("incomplete")
    with pytest.raises(ValueError):
        invariant_read(reader, "failure", fail)
    assert invariant_read(reader, "failure", lambda: 3) == 3


def test_memory_databases_and_time_dependent_fallbacks_are_not_cached():
    with closing(sqlite3.connect(":memory:")) as reader:
        assert invariant_read(reader, "history", lambda: 1) == 1
        assert invariant_read(reader, "history", lambda: 2) == 2


def test_uncacheable_result_is_not_retained(readers):
    path, writer, reader = readers
    assert invariant_read(reader, "fallback", lambda: 1, cacheable=lambda result: False) == 1
    assert invariant_read(reader, "fallback", lambda: 2) == 2


def test_verified_read_snapshot_reuses_exact_version_and_concurrent_write_falls_back(readers):
    _,writer,reader=readers
    calls=[]
    def build():
        calls.append(1)
        return reader.execute('SELECT value FROM evidence').fetchone()[0]
    assert invariant_read(reader,'verified',build)==1
    with invariant_snapshot(reader):
        assert invariant_read(reader,'verified',build)==1 and len(calls)==1
        with pytest.raises(sqlite3.OperationalError):
            reader.execute('UPDATE evidence SET value=99')
        writer.execute('UPDATE evidence SET value=2');writer.commit()
        assert invariant_read(reader,'verified',build)==1 and len(calls)==2
    assert not reader.in_transaction
    assert reader.execute('PRAGMA query_only').fetchone()[0]==0
    assert invariant_read(reader,'verified',build)==2
    reader.execute('BEGIN')
    with pytest.raises(ValueError,match='fresh'):
        with invariant_snapshot(reader):pass
    reader.rollback()


def test_feedback_keys_support_yaml_dates_and_fall_back_for_unknown_values(readers, monkeypatch):
    from datetime import date
    from types import SimpleNamespace
    import run_analysis.run_feedback as feedback
    path, writer, reader = readers
    calls=[]
    def build(connection,config,activity_id):
        calls.append(config)
        return SimpleNamespace(run=SimpleNamespace(start_time=date(2026,10,11)))
    monkeypatch.setattr(feedback,'_get_run_feedback',build)
    config={'coaching':{'goal_date':date(2027,1,1)}}
    feedback.get_run_feedback(reader,config,1)
    feedback.get_run_feedback(reader,config,1)
    assert len(calls)==1
    feedback.get_run_feedback(reader,{'coaching':{'goal_date':date(2027,2,1)}},1)
    assert len(calls)==2
    feedback.get_run_feedback(reader,{'unknown':object()},1)
    assert len(calls)==3
