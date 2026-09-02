from __future__ import annotations

from pathlib import Path

from run_analysis.db import connect
from run_analysis.importer import import_files
from run_analysis.tcx import parse_tcx
from test_tcx import make_tcx


def test_duplicate_activities_share_one_canonical_record(tmp_path: Path) -> None:
    make_tcx(tmp_path / "one.tcx")
    make_tcx(tmp_path / "copy.tcx")
    with connect(tmp_path / "test.sqlite") as connection:
        summary = import_files(connection, tmp_path, "America/New_York")
        assert summary.discovered_files == 2
        assert summary.activities_added == 1
        assert summary.duplicate_activities == 1
        assert connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM activity_sources").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM trackpoints").fetchone()[0] == 2


def test_incremental_import_skips_unchanged_files(tmp_path: Path) -> None:
    make_tcx(tmp_path / "one.tcx")
    with connect(tmp_path / "test.sqlite") as connection:
        first = import_files(connection, tmp_path, "America/New_York")
        second = import_files(connection, tmp_path, "America/New_York")
        assert first.imported_files == 1
        assert second.unchanged_files == 1
        assert second.imported_files == 0


def test_changed_source_is_replaced_without_orphans(tmp_path: Path) -> None:
    source = make_tcx(tmp_path / "one.tcx")
    with connect(tmp_path / "test.sqlite") as connection:
        import_files(connection, tmp_path, "America/New_York")
        make_tcx(
            source,
            start="2024-07-02T12:00:00Z",
            end="2024-07-02T12:00:20Z",
            activity_id="2024-07-02T12:00:00Z",
        )
        summary = import_files(connection, tmp_path, "America/New_York")
        assert summary.imported_files == 1
        assert connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0] == 1
        value = connection.execute("SELECT activity_id FROM activities").fetchone()[0]
        assert value == "2024-07-02T12:00:00Z"


def test_failed_reparse_preserves_last_known_good_activity(tmp_path: Path) -> None:
    source = make_tcx(tmp_path / "one.tcx")
    with connect(tmp_path / "test.sqlite") as connection:
        import_files(connection, tmp_path, "America/New_York")
        original_source = connection.execute(
            "SELECT sha256,parse_status FROM source_files"
        ).fetchone()
        original_activity = connection.execute(
            "SELECT activity_id FROM activities"
        ).fetchone()[0]
        source.write_text("not xml", encoding="utf-8")

        summary = import_files(connection, tmp_path, "America/New_York")
        retained_source = connection.execute(
            "SELECT sha256,parse_status FROM source_files"
        ).fetchone()
        retained_activity = connection.execute(
            "SELECT activity_id FROM activities"
        ).fetchone()[0]

    assert summary.failed_files == 1
    assert dict(retained_source) == dict(original_source)
    assert retained_activity == original_activity


def test_richer_duplicate_upgrades_canonical_activity_regardless_of_order(
    tmp_path: Path, monkeypatch
) -> None:
    import run_analysis.importer as importer_module

    first = make_tcx(tmp_path / "first.tcx")
    second = make_tcx(tmp_path / "second.tcx")
    richer = parse_tcx(second, default_timezone="America/New_York")
    richer.activities[0].trackpoints[0].pause_after_s = 12

    with connect(tmp_path / "test.sqlite") as connection:
        import_files(connection, tmp_path, "America/New_York", paths=[first])
        monkeypatch.setattr(importer_module, "parse_tcx", lambda *args, **kwargs: richer)
        summary = import_files(
            connection, tmp_path, "America/New_York", paths=[second]
        )
        pause = connection.execute(
            "SELECT MAX(pause_after_s) FROM trackpoints"
        ).fetchone()[0]
        primary = connection.execute(
            """
            SELECT sf.display_path
            FROM activity_sources source
            JOIN source_files sf ON sf.id=source.source_file_id
            WHERE source.is_primary=1
            """
        ).fetchone()[0]

    assert summary.duplicate_activities == 1
    assert pause == 12
    assert primary == "second.tcx"
