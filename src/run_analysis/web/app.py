"""FastAPI application factory for the local running coach."""

from __future__ import annotations

from pathlib import Path
from contextlib import asynccontextmanager,closing
from datetime import date, datetime, timedelta, timezone
import os
import json
import sqlite3
import time
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import APIRouter, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..config import load_config, resolve_project_path
from ..db import connect, initialize
from ..dashboard import DASHBOARD_WINDOW_DAYS, build_dashboard
from ..metadata_service import update_run_metadata
from ..beginner import BeginnerFeedback
from ..modeling import fit_models
from ..processing import process_activities
from ..run_feedback import get_run_feedback, list_runs
from ..onboarding import derive_zones, estimate_max_hr, recommend_comparison_hr, automatic_comparison_hr, setup_state
from dataclasses import asdict

from ..progress import build_progress
from ..read_cache import invariant_snapshot
from ..progress_wip import build_progress_wip
from ..training_overview import build_training_overview
from ..race_goals import goal_progress
from ..recommendation_service import (
    current_fitness_state,
    generate_recommendation,
    generate_weekly_schedule,
    ensure_current_weekly_schedule,
    load_forced_rest_dates,
    load_current_status,
    load_latest_recommendation,
    save_forced_rest_dates,
)
from ..settings_service import (
    recalculate_for_settings,
    reset_advanced_overlay,
    save_settings_overlay,
    settings_response,
)
from ..weather import save_activity_postal_code, update_weather
from .schemas import (
    RacePreparationSettings, RacePreparationApply, RaceEventReport,
    ComparisonHrRecommendation,
    ComparisonHrSupportPoint,
    MaxHrEstimateResponse,
    SetupStateResponse,
    SetupStepStatus,
    ZonePreviewRequest,
    ZonePreviewResponse,
    GoalProgressResponse,
    MovingTimeSettings,
    ZoneRange,
    FitnessState,
    DashboardResponse,
    HealthResponse,
    ProgressResponse,
    RecommendationRequest,
    CurrentStatusResponse,
    RecommendationResponse,
    RunFeedback,
    RunSummary,
    RunMetadataPatch,
    SettingsPatch,
    SettingsResponse,
    UploadResponse,
    UploadStage,
    WeeklyScheduleRequest,
    WeeklyScheduleResponse,
    WeeklyRestDayRequest,
    PlannerJobRequest,
    StrengthCompletionRequest,
)
from .upload_service import MAX_UPLOAD_BYTES, UploadPayload, run_locked_upload_pipeline, UploadBusyError
from starlette.concurrency import run_in_threadpool


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
COUNT_TABLES = (
    "source_files",
    "activities",
    "trackpoints",
    "activity_weather",
    "model_runs",
)


#: When this process began. Compared against source modification times so a
#: server left running across an edit can report that fact itself.
PROCESS_STARTED_AT = time.time()


def _source_is_newer_than_process() -> bool:
    """Whether any application module has been edited since startup.

    Static assets are re-read from disk on every request, but Python modules
    are loaded once. After an edit that touches both, an un-restarted server
    serves current markup against stale data, and the mismatch surfaces as a
    confusing client-side error about a missing field rather than as the
    restart it actually is.
    """

    package_root = Path(__file__).resolve().parent.parent
    try:
        newest = max(
            path.stat().st_mtime for path in package_root.rglob("*.py") if path.is_file()
        )
    except (OSError, ValueError):
        return False
    # A second of slack keeps normal filesystem timestamp jitter from flagging
    # a server that started immediately after a write.
    return newest > PROCESS_STARTED_AT + 1.0


def _read_health(project_root: Path, config_path: Path) -> HealthResponse:
    warnings: list[str] = []
    stale = _source_is_newer_than_process()
    restart_hint = (
        "This server is running code from before your most recent edit. Static files "
        "are read from disk on every request but Python is loaded once, so the page "
        "and the data can disagree. Stop the server and run `python3 start.py` again."
        if stale
        else None
    )
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        return HealthResponse(
            status="setup_required",
            database_path="",
            database_exists=False,
            warnings=[str(exc)],
            source_newer_than_process=stale,
            restart_hint=restart_hint,
        )

    database = resolve_project_path(project_root, config["paths"]["database"])
    if not database.exists():
        return HealthResponse(
            status="setup_required",
            database_path=str(database),
            database_exists=False,
            warnings=["Run the import pipeline to create the analysis database."],
            source_newer_than_process=stale,
            restart_hint=restart_hint,
        )

    counts: dict[str, int] = {}
    schema_version: int | None = None
    committed_planner_version: int | None = None
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
            if 'app_state' in existing:
                row=connection.execute("SELECT value_json FROM app_state WHERE key='weekly_schedule_internal'").fetchone()
                if row:
                    committed_planner_version=json.loads(row[0]).get('planner_version')
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
    from ..adaptive.live_bridge import enabled, ENGINE, PLANNER_VERSION
    from ..weekly_schedule import WEEKLY_PLANNER_VERSION
    return HealthResponse(
        status="ready" if not warnings else "degraded",
        database_path=str(database),
        database_exists=True,
        schema_version=schema_version,
        counts=counts,
        warnings=warnings,
        planner_engine=ENGINE if enabled(config) else 'legacy',
        planner_version=(committed_planner_version or PLANNER_VERSION) if enabled(config) else WEEKLY_PLANNER_VERSION,
        source_newer_than_process=stale,
        restart_hint=restart_hint,
    )


#: Environment variables the reload worker reads, because uvicorn's reloader
#: re-imports the application in a fresh process and cannot be handed arguments.
PROJECT_ROOT_ENV = "RUN_ANALYSIS_PROJECT_ROOT"
CONFIG_PATH_ENV = "RUN_ANALYSIS_CONFIG"


def create_app_from_env() -> FastAPI:
    """Factory used by ``uvicorn --reload``; see the environment names above."""
    return create_app(
        os.environ.get(PROJECT_ROOT_ENV, "."),
        os.environ.get(CONFIG_PATH_ENV, "config.yaml"),
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

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            planner_jobs.close()

    app = FastAPI(
        title="Local Running Coach API",
        version="0.2.0",
        lifespan=lifespan,
        description=(
            "Local-first Garmin run analysis. Fitness observations, session "
            "difficulty, training load, and coaching decisions are separate contracts."
        ),
    )
    app.state.project_root = root
    app.state.config_path = selected_config
    from ..planner_observation import PlannerProgress, finished
    planner_progress = PlannerProgress()
    app.state.planner_progress = planner_progress
    from .planner_jobs import PlannerJobs
    planner_jobs=PlannerJobs(planner_progress)
    app.state.planner_jobs=planner_jobs
    from ..adaptive.live_bridge import LivePlannerReviewRequired

    @app.exception_handler(LivePlannerReviewRequired)
    async def adaptive_review_required(request, error):
        return JSONResponse(status_code=409,content={'detail':str(error)})
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "::1", "testserver"],
    )

    @app.middleware("http")
    async def observe_planning(request, call_next):
        tracked = request.url.path in {'/api/weekly-schedule/latest',
            '/api/weekly-schedule', '/api/weekly-schedule/rest-day'}
        request_id = request.headers.get('X-Planner-Request')
        if tracked and request_id:
            try:
                request_id = str(UUID(request_id))
            except (ValueError, AttributeError):
                # Malformed display telemetry cannot change the planner call.
                request_id = None
        else:
            request_id = None
        # Upload, sync and other callers can own the same expensive live job.
        # Keep their telemetry available to a refresh queued behind that job.
        with planner_progress.provide():
            if not request_id:
                return await call_next(request)
            with planner_progress.observe(request_id):
                response = await call_next(request)
                finished(response.status_code < 400)
                return response

    @app.middleware("http")
    async def private_response_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
            "form-action 'self'; connect-src 'self'; img-src 'self' data:; "
            "script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'"
        )
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    api = APIRouter(prefix="/api")

    @api.get('/weekly-schedule/progress/{request_id}', tags=['coaching'])
    def weekly_planner_progress(request_id: UUID) -> dict:
        # No database, model evaluation, or planning side effects.
        return planner_progress.snapshot(planner_jobs.resolve(str(request_id)))

    from .intervals import connection_router
    app.include_router(connection_router(root, selected_config))

    @api.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return _read_health(root, selected_config)

    @api.post("/uploads", response_model=UploadResponse, tags=["activities"])
    async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
        if not files:
            raise HTTPException(status_code=422, detail="At least one activity file is required")
        if len(files) > 25:
            raise HTTPException(status_code=422, detail="Maximum 25 files per upload")
        payloads: list[UploadPayload] = []
        total_bytes = 0
        for uploaded in files:
            content = await uploaded.read(MAX_UPLOAD_BYTES + 1)
            total_bytes += len(content)
            if len(content) > MAX_UPLOAD_BYTES or total_bytes > 2 * MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=422, detail="Upload exceeds 50 MB per file or 100 MB total")
            payloads.append(UploadPayload(uploaded.filename or "activity.tcx", content))
        try:
            return await run_in_threadpool(run_locked_upload_pipeline, root, selected_config, payloads)
        except UploadBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
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
            # Recorded sort fields are unchanged; the adjusted-pace field now
            # uses exactly the same evidence as Progress and Run Analysis.
            from ..comparable_evidence import cached_comparable_evidence, comparable_run_summary
            comparable = cached_comparable_evidence(connection,config)
            values = [comparable_run_summary(item,comparable) for item in values]
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
            "date": lambda item: item.start_time,
            "distance": lambda item: item.distance_miles,
            "pace": lambda item: item.moving_pace_min_mile,
            "heart_rate": lambda item: item.average_hr_bpm,
            "standardized": lambda item: item.fitness_observation.standardized_pace_at_target_hr.minutes_per_mile if item.fitness_observation else None,
            "temperature": lambda item: item.temperature_f,
            "assessment": lambda item: item.assessment_label.casefold() if item.assessment_label else None,
            "data": lambda item: {"good": "Complete", "partial": "Mostly complete", "poor": "Limited", "unavailable": "Unavailable"}.get(item.data_quality.value, item.data_quality.value).casefold(),
        }
        if sort_by not in key_functions or sort_order not in {"asc", "desc"}:
            raise HTTPException(status_code=422, detail="Unsupported run sort")
        key = key_functions[sort_by]
        # Missing measurements remain at the end in either direction, and
        # sorting happens before pagination rather than just within a page.
        measured = [item for item in values if key(item) is not None]
        missing = [item for item in values if key(item) is None]
        measured.sort(key=key, reverse=sort_order == "desc")
        values = measured + missing
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
            if feedback is not None:
                from ..comparable_evidence import cached_comparable_evidence, comparable_run_summary
                feedback = feedback.model_copy(update={'run':comparable_run_summary(feedback.run,cached_comparable_evidence(connection,config))})
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
                WeeklyScheduleRequest(health_status=current.health_status),
                root,
            )
            feedback = get_run_feedback(connection, config, activity_id)
        assert feedback is not None
        # Re-read presentation evidence after the metadata/model transaction.
        with connect(database) as connection:
            from ..comparable_evidence import cached_comparable_evidence, comparable_run_summary
            feedback = feedback.model_copy(update={'run':comparable_run_summary(feedback.run,cached_comparable_evidence(connection,config))})
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
            from ..comparable_evidence import cached_comparable_evidence
            return build_progress(connection, selected_window, config=config,
                comparison_report=cached_comparable_evidence(connection,config))

    @api.get("/progress-wip", tags=["analysis"])
    def progress_wip(window_days: int = Query(default=28, ge=14, le=365),
                     include_exploratory: bool = True,
                     include_legacy_aerobic: bool = True,
                     include_goal_pace_outlook: bool = False) -> dict:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            raise HTTPException(status_code=404, detail="Analysis database does not exist")
        # Separate prototype reader: no initialization/migrations or planner calls.
        with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            try:
                with invariant_snapshot(connection):
                    from ..pace_comparability import comparison_view
                    from ..comparable_evidence import cached_comparable_evidence
                    comparable = cached_comparable_evidence(connection, config)
                    result = build_progress_wip(connection, config, root, window_days,
                                                include_exploratory=include_exploratory,comparison_report=comparable)
                    # The WIP displays the direct pace curve, not the retired
                    # Gaussian/bootstrapped exploratory cards. Keep those
                    # available to the original page without blocking WIP on
                    # their (much more expensive) cold fit.
                    if include_legacy_aerobic:
                        from ..aerobic_candidate import progress_from_comparability
                        result['aerobic_progress'] = progress_from_comparability(
                            connection, comparable, as_of=datetime.fromisoformat(result['as_of']),
                            history_days=window_days, config=config)
                    from ..aerobic_performance import progress_from_comparability as performance_trend
                    from ..progress_chart_context import cached_chart_context
                    result['aerobic_performance'] = performance_trend(
                        connection, comparable,
                        as_of=datetime.fromisoformat(result['as_of']), history_days=window_days,
                        chart_context=cached_chart_context(connection))
                    if include_goal_pace_outlook:
                        from ..pace_outlook import pace_outlook
                        result['goal_journey']['pace_outlook'] = pace_outlook(
                            connection,comparable,result['aerobic_performance'],result['goal_journey'],
                            as_of=datetime.fromisoformat(result['as_of']),timezone=result['timezone'])
                    result['pace_comparison'] = comparison_view(
                        comparable, window_days,
                        datetime.fromisoformat(result['as_of']))
                    return result
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

    @api.get("/pace-comparability", tags=["analysis"])
    def pace_comparability(window_days: int = Query(default=90, ge=14, le=365)) -> dict:
        from ..pace_comparability import comparison_view
        from ..comparable_evidence import cached_comparable_evidence
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            raise HTTPException(status_code=404, detail="Analysis database does not exist")
        with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            with invariant_snapshot(connection):
                report = cached_comparable_evidence(connection,config)
                result = comparison_view(report,window_days,datetime.now(timezone.utc))
        result['timezone'] = config.get('timezone_default','America/New_York')
        return result

    @api.get("/training-overview", tags=["analysis"])
    def training_overview() -> dict:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            raise HTTPException(status_code=404, detail="Analysis database does not exist")
        with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            with invariant_snapshot(connection):
                return build_training_overview(connection, config)

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
    def dashboard(fitness_window_days: int | None = Query(default=None, ge=14, le=365)) -> DashboardResponse:
        config = load_config(selected_config)
        from ..progress import AVAILABLE_WINDOWS
        if fitness_window_days is not None and fitness_window_days not in AVAILABLE_WINDOWS:
            raise HTTPException(status_code=422, detail="Unsupported fitness window")
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            from ..comparable_evidence import cached_comparable_evidence
            return build_dashboard(connection, config, DASHBOARD_WINDOW_DAYS, root, fitness_window_days=fitness_window_days,
                comparison_report=cached_comparable_evidence(connection,config))

    @api.post("/recommendation", response_model=RecommendationResponse, tags=["coaching"])
    def recommendation(request: RecommendationRequest) -> RecommendationResponse:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            from ..health_reports import record_health_report
            record_health_report(connection,request.health_status,source='POST /recommendation')
            try:
                _, result = generate_recommendation(connection, config, request, root)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result

    @api.get("/current-status", response_model=CurrentStatusResponse, tags=["coaching"])
    def current_status() -> CurrentStatusResponse:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            return load_current_status(connection, timezone_name=config['timezone_default'])

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
            from ..adaptive.live_bridge import enabled
            if enabled(config):
                program=ensure_current_weekly_schedule(connection,config,root)
                return next((day.recommendation for day in program.planning_days if day.recommendation),None)
            return load_latest_recommendation(connection)

    @api.post('/beginner-feedback/{activity_id}', tags=['coaching'])
    def beginner_feedback(activity_id: int, feedback: BeginnerFeedback) -> dict:
        from ..beginner import save_feedback
        from ..plan_refresh import mark_plan_refresh
        config = load_config(selected_config)
        database = resolve_project_path(root, config['paths']['database'])
        with connect(database) as connection:
            initialize(connection)
            try:
                save_feedback(connection, config, datetime.now(timezone.utc), activity_id, feedback)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            mark_plan_refresh(connection)
        # No search in a feedback write. The calendar request owns handoff.
        return {'saved': True}

    @api.post("/weekly-schedule", response_model=WeeklyScheduleResponse, tags=["coaching"])
    def weekly_schedule(request: WeeklyScheduleRequest) -> WeeklyScheduleResponse:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            from ..health_reports import record_health_report
            record_health_report(connection,request.health_status,source='POST /weekly-schedule')
            return generate_weekly_schedule(connection, config, request, root)

    @api.post(
        "/weekly-schedule/rest-day",
        response_model=WeeklyScheduleResponse,
        tags=["coaching"],
    )
    def set_weekly_rest_day(request: WeeklyRestDayRequest) -> WeeklyScheduleResponse:
        config = load_config(selected_config)
        local_today = datetime.now(timezone.utc).astimezone(
            ZoneInfo(str(config.get("timezone_default", "UTC")))
        ).date()
        if not local_today <= request.date <= local_today + timedelta(days=6):
            raise HTTPException(
                status_code=422,
                detail="Choose a date in the current seven-day plan.",
            )
        database = resolve_project_path(root, config["paths"]["database"])
        with connect(database) as connection:
            initialize(connection)
            current = ensure_current_weekly_schedule(connection, config, root)
            selected_day = next(
                (day for day in current.days if day.date == request.date),
                None,
            )
            if selected_day and selected_day.completed_activities:
                raise HTTPException(
                    status_code=422,
                    detail="A completed run cannot be replaced with a rest-day constraint.",
                )
            forced_dates = {
                value
                for value in load_forced_rest_dates(connection)
                if value >= local_today
            }
            if request.is_rest_day:
                forced_dates.add(request.date)
            else:
                forced_dates.discard(request.date)
            save_forced_rest_dates(connection, forced_dates)
            status = load_current_status(connection)
            return generate_weekly_schedule(
                connection,
                config,
                WeeklyScheduleRequest(health_status=status.health_status),
                root,
                # Changing availability invalidates the constrained calendar
                # in either direction. The normal 21-day optimizer must solve
                # the new problem instead of treating the old dates as a
                # continuity baseline.
                discard_prior_schedule=True,
            )

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

    @api.post('/weekly-schedule/jobs',status_code=202,tags=['coaching'])
    def start_planner_job(request: PlannerJobRequest) -> dict:
        # Call the SAME validated live operations, on a dedicated worker with
        # its own connection. This does not replace publication/input locks.
        def build():
            if request.operation=='refresh':return latest_weekly_schedule()
            if request.operation=='update':return weekly_schedule(request.update)
            return set_weekly_rest_day(request.rest)
        try:
            return planner_jobs.start(str(request.request_id),request.operation,
                request.model_dump(mode='json',exclude={'request_id'}),build)
        except ValueError as error:
            raise HTTPException(status_code=409,detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503,detail=str(error)) from error

    @api.get('/weekly-schedule/jobs/{job_id}',tags=['coaching'])
    def planner_job_result(job_id: UUID) -> dict:
        result=planner_jobs.snapshot(str(job_id))
        if result is None:
            raise HTTPException(status_code=404,detail='This planner job expired or the app restarted. Update the plan again.')
        return result

    @api.post('/strength-completions', tags=['coaching'])
    def strength_completion(request: StrengthCompletionRequest):
        from ..strength_completion import save_completion
        config = load_config(selected_config)
        database = resolve_project_path(root, config['paths']['database'])
        with connect(database) as connection:
            initialize(connection)
            try:
                receipt = save_completion(connection, config, request.date,
                    request.session_type, request.completed)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {'date': request.date, 'strength_completion': receipt}

    @api.get('/weekly-schedule/saved', tags=['coaching'])
    def saved_weekly_schedule() -> dict:
        from ..plan_refresh import saved_plan_state
        config = load_config(selected_config)
        database = resolve_project_path(root, config['paths']['database'])
        if not database.exists():
            return {'schedule': None, 'needs_refresh': True}
        with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            with invariant_snapshot(connection):
                return saved_plan_state(connection, config)

    def _recommendation_model(result) -> ComparisonHrRecommendation:
        return ComparisonHrRecommendation(
            recommended_bpm=result.recommended_bpm,
            run_count=result.run_count,
            segment_count=result.segment_count,
            zone_lower_bpm=result.zone_lower_bpm,
            zone_upper_bpm=result.zone_upper_bpm,
            rationale=result.rationale,
            candidates=[
                ComparisonHrSupportPoint(
                    heart_rate_bpm=item.heart_rate_bpm,
                    run_count=item.run_count,
                    segment_count=item.segment_count,
                )
                for item in result.candidates
            ],
        )

    @api.post("/settings/reset-advanced", response_model=SettingsResponse, tags=["settings"])
    def reset_advanced() -> SettingsResponse:
        """Restore the modelling parameters, and nothing that describes you."""

        try:
            config = reset_advanced_overlay(selected_config)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        database = resolve_project_path(root, config["paths"]["database"])
        stages: list = []
        if database.exists():
            with connect(database) as connection:
                initialize(connection)
                # Movement and reference conditions feed the pipeline, so the
                # stored results are stale the moment they change back.
                stages = recalculate_for_settings(
                    connection, config, root, SettingsPatch(moving_time=MovingTimeSettings.model_validate(config["moving_time"]))
                )
        return settings_response(config, stages, project_root=root)

    @api.get("/goal-progress", response_model=GoalProgressResponse, tags=["coaching"])
    def goal_progress_route() -> GoalProgressResponse:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            return GoalProgressResponse.model_validate(asdict(goal_progress(None, config)))
        with connect(database) as connection:
            initialize(connection)
            return GoalProgressResponse.model_validate(asdict(goal_progress(connection, config)))

    @api.get("/setup", response_model=SetupStateResponse, tags=["settings"])
    def get_setup() -> SetupStateResponse:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            raise HTTPException(status_code=404, detail="Analysis database does not exist")
        with connect(database) as connection:
            initialize(connection)
            state = setup_state(connection, config)
        return SetupStateResponse(
            complete=state.complete,
            steps=[
                SetupStepStatus(
                    step=item.step,
                    title=item.title,
                    complete=item.complete,
                    detail=item.detail,
                    blocking=item.blocking,
                )
                for item in state.steps
            ],
            next_step=state.next_step.step if state.next_step else None,
        )

    @api.get("/setup/max-hr", response_model=MaxHrEstimateResponse, tags=["settings"])
    def estimated_max_hr(age_years: float = Query(gt=5, le=110)) -> MaxHrEstimateResponse:
        return MaxHrEstimateResponse(
            estimated_max_hr=estimate_max_hr(age_years),
            method="Tanaka, Monahan & Seals (2001): 208 - 0.7 x age",
            caveat=(
                "An age formula, not a measurement. Individuals vary by roughly 7 bpm "
                "either side, and this single number sets every zone boundary below it."
            ),
        )

    @api.post("/setup/zone-preview", response_model=ZonePreviewResponse, tags=["settings"])
    def preview_zones(request: ZonePreviewRequest) -> ZonePreviewResponse:
        """Show the zones a method produces, and what it does to the comparison
        heart rate, before anything is saved."""

        config = load_config(selected_config)
        watch_recorded_at = None
        maximum_hr, resting_hr = request.max_hr, request.resting_hr or config['resting_hr']
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            raise HTTPException(status_code=404, detail="Analysis database does not exist")
        # Resolve the source BEFORE validating: watch ranges must be checked
        # against that same watch's max/rest HR, not stale custom form values.
        recorded = None
        if request.method != 'device' and request.max_hr is None:
            raise HTTPException(status_code=422, detail='Enter a maximum heart rate for this method.')
        if request.method == 'device':
            from ..watch_zones import watch_zone_patch
            try:
                with connect(database) as connection:
                    initialize(connection)
                    recorded = watch_zone_patch({**config,
                        'setup': {**(config.get('setup') or {}), 'zone_method': 'device'}},
                        connection, required=True)
                maximum_hr, resting_hr = recorded['max_hr'], recorded['resting_hr']
                watch_recorded_at = recorded['watch_zone_import']['recorded_at']
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            zones = derive_zones(
                request.method,
                maximum_hr=maximum_hr,
                resting_hr=resting_hr if recorded else request.resting_hr,
                boundaries=(recorded['zones'] if recorded else
                    {name: (zone.minimum_bpm, zone.maximum_bpm) for name, zone in request.boundaries.items()}
                    if request.boundaries or recorded else None),
            )
            from ..settings_service import validate_config
            validate_config({**config, 'max_hr': maximum_hr,
                'resting_hr': resting_hr,
                'target_hr': sum(zones['z2']) // 2, 'zones': zones})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with connect(database) as connection:
            initialize(connection)
            recommendation = automatic_comparison_hr(connection, zones, current=config['target_hr'])
        return ZonePreviewResponse(
            method=request.method,
            zones={name: ZoneRange(minimum_bpm=low, maximum_bpm=high) for name, (low, high) in zones.items()},
            watch_recorded_at=watch_recorded_at,
            max_hr=maximum_hr, resting_hr=resting_hr,
            comparison_hr=_recommendation_model(recommendation),
        )

    @api.get("/setup/comparison-hr", response_model=ComparisonHrRecommendation, tags=["settings"])
    def comparison_hr() -> ComparisonHrRecommendation:
        config = load_config(selected_config)
        database = resolve_project_path(root, config["paths"]["database"])
        if not database.exists():
            raise HTTPException(status_code=404, detail="Analysis database does not exist")
        with connect(database) as connection:
            initialize(connection)
            return _recommendation_model(automatic_comparison_hr(connection, config["zones"], current=config['target_hr']))

    @api.post('/settings/race-preparation/preview', tags=['settings'])
    def race_preparation_preview(draft: RacePreparationSettings):
        from ..adaptive.race_settings import preview_preparation
        try:
            return preview_preparation(load_config(selected_config), root, draft,
                                       as_of=datetime.now(timezone.utc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @api.get("/settings", response_model=SettingsResponse, tags=["settings"])
    def get_settings() -> SettingsResponse:
        return settings_response(load_config(selected_config), project_root=root)

    @api.get('/settings/race-preparation/status', tags=['settings'])
    def race_goal_status():
        from ..adaptive.race_settings import goal_status
        return goal_status(load_config(selected_config),root)

    @api.post('/settings/race-preparation/apply', tags=['settings'])
    def apply_race_goal(request: RacePreparationApply):
        from ..adaptive.race_settings import save_race_request
        try:
            return save_race_request(load_config(selected_config),root,request.goal,
                expected_revision=request.expected_revision,as_of=datetime.now(timezone.utc),
                planner_preferences=request.planner_preferences)
        except (ValueError,OSError,sqlite3.Error) as error:
            raise HTTPException(status_code=422,detail=str(error)) from error

    @api.post('/settings/race-preparation/outcome', tags=['settings'])
    def report_race_outcome(request: RaceEventReport):
        from ..adaptive.race_settings import save_race_outcome
        try:
            return save_race_outcome(load_config(selected_config),root,status=request.status,
                physical_key=request.physical_key,expected_revision=request.expected_revision,
                as_of=datetime.now(timezone.utc))
        except (ValueError,OSError,sqlite3.Error) as error:
            raise HTTPException(status_code=422,detail=str(error)) from error

    @api.patch("/settings", response_model=SettingsResponse, tags=["settings"])
    def patch_settings(patch: SettingsPatch) -> SettingsResponse:
        base = load_config(selected_config)
        database = resolve_project_path(root, base["paths"]["database"])
        stages = []
        try:
            if 'planner_preferences' in patch.model_fields_set:
                if patch.model_fields_set != {'planner_preferences'} or patch.planner_preferences is None:
                    raise ValueError('Save planner preferences separately from calibration and other settings')
                if not database.exists():
                    raise ValueError('No analysis database; no planner preferences saved')
                from datetime import datetime, timezone
                from run_analysis.adaptive.settings_view import save_planner_preferences
                with connect(database) as connection:
                    revision = save_planner_preferences(connection,base,patch.planner_preferences,
                        as_of=datetime.now(timezone.utc))
                return settings_response(base, [UploadStage(name='planner_preferences',
                    status='deferred' if revision else 'unchanged',
                    detail='Saved a dated preference request. Frequency and long-run day edits apply on the next successful plan re-evaluation; destination-only edits join the next normal replan. The issued plan is unchanged.'
                        if revision else 'Preferences are unchanged; no new request or search.')], project_root=root)
            if patch.model_fields_set & {'training_background','beginner_start'} and not database.exists():
                # A starting-level declaration is usable before the first
                # upload, but must still verify there is no adopted runtime.
                with connect(database) as connection:
                    initialize(connection)
            if database.exists():
                with connect(database) as connection:
                    initialize(connection)
                    config = save_settings_overlay(
                        selected_config, patch, connection=connection
                    )
                    stages = recalculate_for_settings(
                        connection,
                        config,
                        root,
                        patch,
                        previous_config=base,
                    )
            else:
                config = save_settings_overlay(selected_config, patch)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return settings_response(config, stages, project_root=root)

    app.include_router(api)
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
