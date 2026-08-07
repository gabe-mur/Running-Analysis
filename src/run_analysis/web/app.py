"""FastAPI application factory for the local running coach."""

from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timezone
import sqlite3

from fastapi import APIRouter, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import load_config, resolve_project_path
from ..db import connect, initialize
from ..dashboard import DASHBOARD_WINDOW_DAYS, build_dashboard
from ..external_fitness import save_snapshot, summarize_external_fitness
from ..metadata_service import update_run_metadata
from ..modeling import fit_models
from ..processing import process_activities
from ..run_feedback import get_run_feedback, list_runs
from ..progress import build_progress
from ..recommendation_service import (
    current_fitness_state,
    generate_recommendation,
    generate_weekly_schedule,
    ensure_current_weekly_schedule,
    load_current_status,
    load_latest_recommendation,
)
from ..settings_service import recalculate_for_settings, save_settings_overlay, settings_response
from ..weather import save_activity_postal_code, update_weather
from .schemas import (
    FitnessState,
    ExternalFitnessSnapshot,
    ExternalFitnessSnapshotInput,
    ExternalFitnessSummary,
    DashboardResponse,
    HealthResponse,
    ProgressResponse,
    RecommendationRequest,
    RecommendationResponse,
    RunFeedback,
    RunSummary,
    RunMetadataPatch,
    SettingsPatch,
    SettingsResponse,
    UploadResponse,
    WeeklyScheduleRequest,
    WeeklyScheduleResponse,
)
from .upload_service import MAX_UPLOAD_BYTES, UploadPayload, run_upload_pipeline


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
COUNT_TABLES = (
    "source_files",
    "activities",
    "trackpoints",
    "activity_weather",
    "model_runs",
)


def _read_health(project_root: Path, config_path: Path) -> HealthResponse:
    warnings: list[str] = []
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        return HealthResponse(
            status="setup_required",
            database_path="",
            database_exists=False,
            warnings=[str(exc)],
        )

    database = resolve_project_path(project_root, config["paths"]["database"])
    if not database.exists():
        return HealthResponse(
            status="setup_required",
            database_path=str(database),
            database_exists=False,
            warnings=["Run the import pipeline to create the analysis database."],
        )

    counts: dict[str, int] = {}
    schema_version: int | None = None
    try:
        # A health check is read-only. URI mode prevents an accidental empty DB.
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            existing = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            for table in COUNT_TABLES:
                if table in existing:
                    counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                else:
                    counts[table] = 0
                    warnings.append(f"Missing database table: {table}")
            if "schema_metadata" in existing:
                row = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
                ).fetchone()
                if row:
                    schema_version = int(row[0])
        finally:
            connection.close()
    except (OSError, sqlite3.Error, ValueError) as exc:
        return HealthResponse(
            status="error",
            database_path=str(database),
            database_exists=True,
            counts=counts,
            warnings=[f"Database health check failed: {exc}"],
        )

    if not counts.get("activities"):
        warnings.append("No activities have been imported yet.")
    return HealthResponse(
        status="ready" if not warnings else "degraded",
        database_path=str(database),
        database_exists=True,
        schema_version=schema_version,
        counts=counts,
        warnings=warnings,
    )


def create_app(
    project_root: str | Path = ".",
    config_path: str | Path = "config.yaml",
) -> FastAPI:
    """Build an app bound to one local running-analysis project."""

    root = Path(project_root).resolve()
    selected_config = Path(config_path)
    if not selected_config.is_absolute():
        selected_config = root / selected_config

    app = FastAPI(
        title="Local Running Coach API",
        version="0.1.0",
        description=(
            "Local-first Garmin run analysis. Fitness observations, session "
            "difficulty, training load, and coaching decisions are separate contracts."
        ),
    )
    app.state.project_root = root
    app.state.config_path = selected_config

    api = APIRouter(prefix="/api")

    @api.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return _read_health(root, selected_config)

    @api.post("/uploads", response_model=UploadResponse, tags=["activities"])
    async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
        if not files:
            raise HTTPException(status_code=422, detail="At least one TCX file is required")
        payloads: list[UploadPayload] = []
        for uploaded in files:
            content = await uploaded.read(MAX_UPLOAD_BYTES + 1)
            payloads.append(UploadPayload(uploaded.filename or "activity.tcx", content))
        try:
            return run_upload_pipeline(root, selected_config, payloads)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @api.get("/runs", response_model=list[RunSummary], tags=["activities"])
    def runs(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        workout_type: str | None = None,
        health_tag: str | None = None,
        flag: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        sort_by: str = "date",
        sort_order: str = "desc",
    ) -> list[RunSummary]:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            return []
        with connect(database) as connection:
            initialize(connection)
            values = list_runs(connection, limit=5000, offset=0)
        if workout_type:
            values = [item for item in values if item.workout_type.value == workout_type]
        if health_tag:
            values = [item for item in values if item.health_tag.value == health_tag]
        if date_from:
            values = [item for item in values if item.start_time and item.start_time.date() >= date_from]
        if date_to:
            values = [item for item in values if item.start_time and item.start_time.date() <= date_to]
        if flag == "easy":
            values = [item for item in values if item.workout_type.value in {"easy", "recovery"}]
        elif flag == "hard":
            values = [item for item in values if item.session_difficulty and item.session_difficulty.is_quality_session]
        elif flag == "long":
            values = [item for item in values if item.session_difficulty and item.session_difficulty.is_long_run]
        elif flag == "illness":
            values = [item for item in values if item.health_tag.value != "normal"]
        elif flag == "no_gps":
            values = [item for item in values if "missing" in item.gps_quality]
        elif flag == "excluded":
            values = [item for item in values if item.model_included is False]
        key_functions = {
            "date": lambda item: item.start_time or datetime.min.replace(tzinfo=timezone.utc),
            "distance": lambda item: item.distance_miles,
            "pace": lambda item: item.moving_pace_min_mile or float("inf"),
            "heart_rate": lambda item: item.average_hr_bpm or 0,
            "standardized": lambda item: item.fitness_observation.standardized_pace_at_target_hr.minutes_per_mile if item.fitness_observation else float("inf"),
        }
        if sort_by not in key_functions or sort_order not in {"asc", "desc"}:
            raise HTTPException(status_code=422, detail="Unsupported run sort")
        values.sort(key=key_functions[sort_by], reverse=sort_order == "desc")
        return values[offset : offset + limit]

    @api.get("/runs/{activity_id}", response_model=RunFeedback, tags=["activities"])
    def run_detail(activity_id: int) -> RunFeedback:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            raise HTTPException(status_code=404, detail="Analysis database does not exist")
        with connect(database) as connection:
            initialize(connection)
            feedback = get_run_feedback(connection, config, activity_id)
        if feedback is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return feedback

    @api.patch("/runs/{activity_id}/metadata", response_model=RunFeedback, tags=["activities"])
    def patch_run_metadata(activity_id: int, patch: RunMetadataPatch) -> RunFeedback:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        overrides = resolve_project_path(root, config["paths"]["overrides"])
        with connect(database) as connection:
            initialize(connection)
            try:
                update_run_metadata(connection, overrides, activity_id, patch)
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if "postal_code" in patch.model_fields_set:
                try:
                    save_activity_postal_code(
                        connection,
                        activity_id,
                        patch.postal_code,
                        config,
                    )
                except LookupError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
            process_activities(connection, config)
            if "postal_code" in patch.model_fields_set:
                update_weather(connection, config, root)
            try:
                fit_models(connection, config, root / "output" / "model_results.json")
            except ValueError:
                pass
            current = load_current_status(connection)
            generate_weekly_schedule(
                connection,
                config,
                WeeklyScheduleRequest(
                    health_status=current.health_status,
                    notes=current.notes,
                ),
                root,
            )
            feedback = get_run_feedback(connection, config, activity_id)
        assert feedback is not None
        return feedback

    @api.get("/progress", response_model=ProgressResponse, tags=["analysis"])
    def progress(window_days: int | None = Query(default=None, ge=7, le=365)) -> ProgressResponse:
        config = load_config(selected_config)
        selected_window = window_days or int(config.get("app", {}).get("default_fitness_window", 28))
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            raise HTTPException(status_code=404, detail="Analysis database does not exist")
        with connect(database) as connection:
            initialize(connection)
            return build_progress(connection, selected_window, config=config)

    @api.get("/external-fitness", response_model=ExternalFitnessSummary, tags=["analysis"])
    def external_fitness() -> ExternalFitnessSummary:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            return summarize_external_fitness(connection, datetime.now(timezone.utc))

    @api.post("/external-fitness", response_model=ExternalFitnessSnapshot, tags=["analysis"])
    def add_external_fitness(snapshot: ExternalFitnessSnapshotInput) -> ExternalFitnessSnapshot:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            return save_snapshot(connection, snapshot)

    @api.get("/fitness-state", response_model=FitnessState, tags=["coaching"])
    def fitness_state() -> FitnessState:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            raise HTTPException(status_code=404, detail="Analysis database does not exist")
        with connect(database) as connection:
            initialize(connection)
            return current_fitness_state(connection, config)

    @api.get("/dashboard", response_model=DashboardResponse, tags=["analysis"])
    def dashboard() -> DashboardResponse:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            return build_dashboard(connection, config, DASHBOARD_WINDOW_DAYS, root)

    @api.post("/recommendation", response_model=RecommendationResponse, tags=["coaching"])
    def recommendation(request: RecommendationRequest) -> RecommendationResponse:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            try:
                _, result = generate_recommendation(connection, config, request, root)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result

    @api.get("/current-status", response_model=RecommendationRequest, tags=["coaching"])
    def current_status() -> RecommendationRequest:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            return load_current_status(connection)

    @api.get(
        "/recommendation/latest",
        response_model=RecommendationResponse | None,
        tags=["coaching"],
    )
    def latest_recommendation() -> RecommendationResponse | None:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            return load_latest_recommendation(connection)

    @api.post("/weekly-schedule", response_model=WeeklyScheduleResponse, tags=["coaching"])
    def weekly_schedule(request: WeeklyScheduleRequest) -> WeeklyScheduleResponse:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            return generate_weekly_schedule(connection, config, request, root)

    @api.get(
        "/weekly-schedule/latest",
        response_model=WeeklyScheduleResponse | None,
        tags=["coaching"],
    )
    def latest_weekly_schedule() -> WeeklyScheduleResponse | None:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            return ensure_current_weekly_schedule(connection, config, root)

    @api.get("/settings", response_model=SettingsResponse, tags=["settings"])
    def get_settings() -> SettingsResponse:
        return settings_response(load_config(selected_config))

    @api.patch("/settings", response_model=SettingsResponse, tags=["settings"])
    def patch_settings(patch: SettingsPatch) -> SettingsResponse:
        try:
            config = save_settings_overlay(selected_config, patch)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        database = resolve_project_path(root, config["paths"]["database"])
        stages = []
        if database.exists():
            with connect(database) as connection:
                initialize(connection)
                stages = recalculate_for_settings(connection, config, root, patch)
        return settings_response(config, stages)

    app.include_router(api)
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
