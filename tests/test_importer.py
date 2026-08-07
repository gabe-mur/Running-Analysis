from __future__ import annotations

from pathlib import Path

from run_analysis.db import connect
from run_analysis.importer import import_files
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

