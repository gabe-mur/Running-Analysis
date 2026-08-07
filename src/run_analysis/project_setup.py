"""Idempotent creation of the ignored local project files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

from .config import load_config, resolve_project_path


@dataclass(slots=True)
class ProjectSetupSummary:
    created: list[str] = field(default_factory=list)
    existing: list[str] = field(default_factory=list)


def _record_directory(path: Path, root: Path, summary: ProjectSetupSummary) -> None:
    label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    if path.exists():
        summary.existing.append(label)
        return
    path.mkdir(parents=True, exist_ok=True)
    summary.created.append(label)


def initialize_project(
    project_root: str | Path = ".",
    config_path: str | Path = "config.yaml",
) -> ProjectSetupSummary:
    """Create local configuration, metadata, and runtime directories once."""

    root = Path(project_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected_config = Path(config_path)
    if not selected_config.is_absolute():
        selected_config = root / selected_config
    summary = ProjectSetupSummary()

    if selected_config.exists():
        summary.existing.append(str(selected_config.relative_to(root)))
    else:
        template = root / "config.example.yaml"
        if not template.exists():
            raise FileNotFoundError(
                f"Configuration template not found: {template}. Run initialization from the repository root."
            )
        selected_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, selected_config)
        summary.created.append(str(selected_config.relative_to(root)))

    config = load_config(selected_config)
    overrides = resolve_project_path(root, config["paths"]["overrides"])
    if overrides.exists():
        summary.existing.append(str(overrides.relative_to(root)))
    else:
        override_template = root / "run_overrides.example.csv"
        overrides.parent.mkdir(parents=True, exist_ok=True)
        if override_template.exists():
            shutil.copy2(override_template, overrides)
        else:
            overrides.write_text(
                "activity_id,include_in_model,workout_type,illness,health_tag,"
                "perceived_exertion,notes\n",
                encoding="utf-8",
            )
        summary.created.append(str(overrides.relative_to(root)))

    directories = {
        root / "TCX",
        root / "uploads",
        root / "docs" / "local",
        resolve_project_path(root, config["paths"]["database"]).parent,
        resolve_project_path(root, config["paths"]["report"]).parent,
        resolve_project_path(root, config["paths"]["weather_cache"]),
    }
    for directory in sorted(directories, key=str):
        _record_directory(directory, root, summary)
    return summary
