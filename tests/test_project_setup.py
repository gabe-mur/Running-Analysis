from __future__ import annotations

from argparse import Namespace
import os
from pathlib import Path
import shutil
import stat
import yaml

from run_analysis.cli import command_all, command_model
from run_analysis.db import connect, initialize
from run_analysis.project_setup import initialize_project
from run_analysis.weather import update_weather


REPOSITORY = Path(__file__).parents[1]


def _copy_templates(root: Path) -> None:
    shutil.copy2(REPOSITORY / "config.example.yaml", root / "config.example.yaml")
    shutil.copy2(
        REPOSITORY / "run_overrides.example.csv",
        root / "run_overrides.example.csv",
    )


def test_project_initialization_is_private_and_idempotent(tmp_path: Path) -> None:
    _copy_templates(tmp_path)
    first = initialize_project(tmp_path)
    second = initialize_project(tmp_path)
    assert "config.yaml" in first.created
    assert "run_overrides.csv" in first.created
    assert (tmp_path / "TCX").is_dir()
    assert (tmp_path / "data" / "weather_cache").is_dir()
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / "config.yaml").stat().st_mode) == 0o600
        assert stat.S_IMODE((tmp_path / "run_overrides.csv").stat().st_mode) == 0o600
        assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
    assert not second.created


def test_initialization_secures_an_existing_local_overlay(tmp_path: Path) -> None:
    _copy_templates(tmp_path)
    overlay = tmp_path / "config.local.yaml"
    overlay.write_text("target_hr: 145\n", encoding="utf-8")
    overlay.chmod(0o644)

    initialize_project(tmp_path)

    if os.name != "nt":
        assert stat.S_IMODE(overlay.stat().st_mode) == 0o600


def test_initialization_secures_existing_private_data_files(tmp_path: Path) -> None:
    _copy_templates(tmp_path)
    private_data = tmp_path / "TCX"
    private_data.mkdir()
    activity = private_data / "example.tcx"
    activity.write_text("<TrainingCenterDatabase />", encoding="utf-8")
    activity.chmod(0o644)

    initialize_project(tmp_path)

    if os.name != "nt":
        assert stat.S_IMODE(activity.stat().st_mode) == 0o600


def test_initialization_does_not_chmod_repository_root(tmp_path: Path) -> None:
    _copy_templates(tmp_path)
    config = yaml.safe_load((tmp_path / "config.example.yaml").read_text())
    config["paths"]["report"] = "report.html"
    (tmp_path / "config.example.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if os.name != "nt":
        tmp_path.chmod(0o755)

    initialize_project(tmp_path)

    if os.name != "nt":
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o755


def test_historical_weather_defaults_to_no_network(tmp_path: Path) -> None:
    _copy_templates(tmp_path)
    initialize_project(tmp_path)
    called = False

    def downloader(_url: str, _timeout: float) -> dict:
        nonlocal called
        called = True
        return {}

    with connect(tmp_path / "data" / "run_analysis.sqlite") as connection:
        initialize(connection)
        summary = update_weather(
            connection,
            yaml.safe_load((tmp_path / "config.yaml").read_text()),
            tmp_path,
            downloader=downloader,
        )
    assert summary.historical_weather_enabled is False
    assert called is False


def test_model_and_full_pipeline_defer_cleanly_for_empty_history(
    tmp_path: Path, capsys
) -> None:
    _copy_templates(tmp_path)
    initialize_project(tmp_path)
    args = Namespace(project_root=str(tmp_path), config="config.yaml")
    assert command_all(args) == 0
    assert command_model(args) == 0
    assert "Model deferred" in capsys.readouterr().out
