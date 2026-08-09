"""Command-line entry point."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path
import json
import sqlite3

from .config import load_config, resolve_project_path
from .classification import write_activity_type_review
from .audit import build_audit, write_audit
from .db import connect, initialize
from .importer import import_files
from .modeling import InsufficientModelDataError, fit_models
from .overrides import sync_overrides
from .processing import process_activities
from .project_setup import initialize_project
from .reporting import write_report
from .weather import update_weather


def _project_and_config(args: Namespace) -> tuple[Path, dict]:
    root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.exists():
        initialize_project(root, config_path)
    return root, load_config(config_path)


def command_init(args: Namespace) -> int:
    summary = initialize_project(args.project_root, args.config)
    if summary.created:
        print("Created local files/directories:")
        for path in summary.created:
            print(f"  {path}")
    else:
        print("Local project setup is already complete.")
    root = Path(args.project_root).resolve()
    selected_config = Path(args.config)
    if not selected_config.is_absolute():
        selected_config = root / selected_config
    historical_enabled = bool(
        load_config(selected_config).get("weather", {}).get("historical_enabled", False)
    )
    if historical_enabled:
        print("Historical weather retrieval is enabled in the local configuration.")
    else:
        print("Historical weather remains off until explicitly enabled in Settings.")
    return 0


def command_import(args: Namespace) -> int:
    root, config = _project_and_config(args)
    database = resolve_project_path(root, config["paths"]["database"])
    with connect(database) as connection:
        summary = import_files(
            connection,
            root,
            default_timezone=config["timezone_default"],
            force=args.force,
        )
    print(f"Database: {database}")
    for key, value in asdict(summary).items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    return 1 if summary.failed_files else 0


def command_process(args: Namespace) -> int:
    root, config = _project_and_config(args)
    database = resolve_project_path(root, config["paths"]["database"])
    if not database.exists():
        print(f"Database does not exist: {database}. Run the import command first.")
        return 1
    overrides_path = resolve_project_path(root, config["paths"]["overrides"])
    with connect(database) as connection:
        initialize(connection)
        override_count = sync_overrides(connection, overrides_path)
        summary = process_activities(connection, config, force=args.force)
    print(f"Database: {database}")
    print(f"Overrides synchronized: {override_count}")
    for key, value in asdict(summary).items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    return 0


def command_audit(args: Namespace) -> int:
    root, config = _project_and_config(args)
    database = resolve_project_path(root, config["paths"]["database"])
    if not database.exists():
        print(f"Database does not exist: {database}. Run import and process first.")
        return 1
    output = root / "output" / "data_audit.json"
    with connect(database) as connection:
        initialize(connection)
        audit = build_audit(connection)
        candidate_count = write_activity_type_review(
            connection, config, root / "output" / "activity_type_review.csv"
        )
    write_audit(audit, output)
    print(json.dumps(audit, indent=2))
    print(f"\nAudit written to: {output}")
    print(f"Activity-type review candidates: {candidate_count} (output/activity_type_review.csv)")
    return 0


def command_weather(args: Namespace) -> int:
    root, config = _project_and_config(args)
    database = resolve_project_path(root, config["paths"]["database"])
    if not database.exists():
        print(f"Database does not exist: {database}. Run import and process first.")
        return 1
    with connect(database) as connection:
        initialize(connection)
        summary = update_weather(connection, config, root, force=args.force)
    print(f"Database: {database}")
    for key, value in asdict(summary).items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    return 1 if summary.failures else 0


def command_model(args: Namespace) -> int:
    root, config = _project_and_config(args)
    database = resolve_project_path(root, config["paths"]["database"])
    if not database.exists():
        print(f"Database does not exist: {database}. Run import, process, and weather first.")
        return 1
    output = root / "output" / "model_results.json"
    with connect(database) as connection:
        initialize(connection)
        try:
            summary = fit_models(connection, config, output)
        except InsufficientModelDataError as exc:
            print(f"Model deferred: {exc}")
            print("Imported runs remain available; modeling will retry as usable history grows.")
            return 0
    for key, value in asdict(summary).items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    return 0


def command_report(args: Namespace) -> int:
    root, config = _project_and_config(args)
    database = resolve_project_path(root, config["paths"]["database"])
    if not database.exists():
        print(f"Database does not exist: {database}. Run the analysis pipeline first.")
        return 1
    output = resolve_project_path(root, config["paths"]["report"])
    with connect(database) as connection:
        initialize(connection)
        write_report(connection, config, output)
    print(f"Report written to: {output}")
    return 0


def command_serve(args: Namespace) -> int:
    """Run the local coach web application.

    Reloading is on by default. Static assets are re-read from disk on every
    request while Python modules are loaded once, so a server left running
    across an edit serves current markup against stale data — which surfaces
    as a confusing client-side error rather than as the restart it is.
    Watching the source removes that whole class of confusion.
    """

    import os

    import uvicorn

    from .web.app import CONFIG_PATH_ENV, PROJECT_ROOT_ENV, create_app

    initialize_project(args.project_root, args.config)
    if args.no_reload:
        uvicorn.run(create_app(args.project_root, args.config), host=args.host, port=args.port)
        return 0

    # The reloader re-imports in a fresh worker, so the target has to be an
    # import string and its arguments have to travel by environment.
    os.environ[PROJECT_ROOT_ENV] = str(Path(args.project_root).resolve())
    os.environ[CONFIG_PATH_ENV] = str(args.config)
    uvicorn.run(
        "run_analysis.web.app:create_app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)],
    )
    return 0


def command_all(args: Namespace) -> int:
    common = {"project_root": args.project_root, "config": args.config}
    commands = (
        (command_import, Namespace(**common, force=False)),
        (command_process, Namespace(**common, force=False)),
        (command_weather, Namespace(**common, force=False)),
        (command_model, Namespace(**common)),
        (command_audit, Namespace(**common)),
        (command_report, Namespace(**common)),
    )
    for command, command_args in commands:
        status = command(command_args)
        if status:
            return status
    return 0


def _activity_query(connection: sqlite3.Connection, identifier: str) -> sqlite3.Row | None:
    if identifier.isdigit():
        row = connection.execute("SELECT * FROM activities WHERE id = ?", (int(identifier),)).fetchone()
        if row:
            return row
    row = connection.execute("SELECT * FROM activities WHERE activity_id = ?", (identifier,)).fetchone()
    if row:
        return row
    return connection.execute(
        "SELECT * FROM activities WHERE activity_uid LIKE ? ORDER BY id LIMIT 1", (f"{identifier}%",)
    ).fetchone()


def command_inspect(args: Namespace) -> int:
    root, config = _project_and_config(args)
    database = resolve_project_path(root, config["paths"]["database"])
    if not database.exists():
        print(f"Database does not exist: {database}. Run the import command first.")
        return 1
    with connect(database) as connection:
        initialize(connection)
        row = _activity_query(connection, args.activity_id)
        if row is None:
            print(f"Activity not found: {args.activity_id}")
            return 1
        activity_id = int(row["id"])
        sources = connection.execute(
            """
            SELECT sf.display_path, sf.parse_status, sf.parse_encoding,
                   actsrc.is_primary, actsrc.duplicate_reason
            FROM activity_sources actsrc
            JOIN source_files sf ON sf.id = actsrc.source_file_id
            WHERE actsrc.activity_id = ? ORDER BY actsrc.is_primary DESC, sf.display_path
            """,
            (activity_id,),
        ).fetchall()
        point_summary = connection.execute(
            """
            SELECT COUNT(*) AS points, SUM(gps_valid) AS valid_gps,
                   SUM(heart_rate_bpm IS NOT NULL) AS hr_points,
                   SUM(altitude_m IS NOT NULL) AS elevation_points,
                   SUM(cadence IS NOT NULL) AS cadence_points,
                   SUM(speed_mps IS NOT NULL) AS speed_points,
                   MIN(timestamp_utc) AS first_point, MAX(timestamp_utc) AS last_point
            FROM trackpoints WHERE activity_id = ?
            """,
            (activity_id,),
        ).fetchone()
        metrics = connection.execute(
            "SELECT * FROM activity_metrics WHERE activity_id = ?", (activity_id,)
        ).fetchone()
        segments = connection.execute(
            """
            SELECT segment_index, distance_m, moving_time_s, elapsed_time_s,
                   stopped_time_s, moving_pace_min_mile, average_hr_bpm,
                   average_grade_percent, gps_complete_fraction,
                   route_bearing_degrees, is_pathological, flags_json
            FROM segments WHERE activity_id = ? ORDER BY segment_index
            """,
            (activity_id,),
        ).fetchall()
    printable = dict(row)
    printable["namespaces_json"] = json.loads(printable["namespaces_json"])
    printable["data_quality_json"] = json.loads(printable["data_quality_json"])
    print(json.dumps(printable, indent=2, default=str))
    print("\nTrackpoint coverage:")
    print(json.dumps(dict(point_summary), indent=2, default=str))
    print("\nSources:")
    print(json.dumps([dict(source) for source in sources], indent=2, default=str))
    if metrics:
        metric_values = dict(metrics)
        for key in ("metrics_json", "hr_zone_seconds_json", "diagnostics_json"):
            metric_values[key] = json.loads(metric_values[key])
        print("\nCalculated run metrics:")
        print(json.dumps(metric_values, indent=2, default=str))
        print("\nSegments:")
        segment_values = []
        for segment in segments:
            value = dict(segment)
            value["flags_json"] = json.loads(value["flags_json"])
            segment_values.append(value)
        print(json.dumps(segment_values, indent=2, default=str))
    return 0


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m run_analysis")
    parser.add_argument("--project-root", default=".", help="Project directory containing TCX files")
    parser.add_argument("--config", default="config.yaml", help="Configuration path relative to project root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Create ignored local configuration and runtime directories"
    )
    init_parser.set_defaults(handler=command_init)

    import_parser = subparsers.add_parser("import", help="Incrementally import new or changed TCX files")
    import_parser.add_argument("--force", action="store_true", help="Reparse all source files")
    import_parser.set_defaults(handler=command_import)

    process_parser = subparsers.add_parser(
        "process", help="Calculate moving time, run metrics, and quarter-mile segments"
    )
    process_parser.add_argument("--force", action="store_true", help="Reprocess all activities")
    process_parser.set_defaults(handler=command_process)

    audit_parser = subparsers.add_parser("audit", help="Write and print the current data-quality audit")
    audit_parser.set_defaults(handler=command_audit)

    weather_parser = subparsers.add_parser(
        "weather", help="Fetch/cache Open-Meteo history and interpolate segment weather"
    )
    weather_parser.add_argument("--force", action="store_true", help="Refresh cached weather days")
    weather_parser.set_defaults(handler=command_weather)

    model_parser = subparsers.add_parser(
        "model", help="Run grouped validation and fit standardized pace at target HR"
    )
    model_parser.set_defaults(handler=command_model)

    report_parser = subparsers.add_parser(
        "report", help="Generate the self-contained local HTML dashboard"
    )
    report_parser.set_defaults(handler=command_report)

    serve_parser = subparsers.add_parser("serve", help="Run the local Running Coach web app")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Address to listen on")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Do not watch the source for changes (reloading is on by default)",
    )
    serve_parser.set_defaults(handler=command_serve)

    all_parser = subparsers.add_parser(
        "all", help="Incrementally import, process, fetch missing weather, model, audit, and report"
    )
    all_parser.set_defaults(handler=command_all)

    inspect_parser = subparsers.add_parser("inspect", help="Show one imported activity and its quality diagnostics")
    inspect_parser.add_argument("activity_id", help="Database id, TCX Activity/Id, or activity UID prefix")
    inspect_parser.set_defaults(handler=command_inspect)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.handler(args))
