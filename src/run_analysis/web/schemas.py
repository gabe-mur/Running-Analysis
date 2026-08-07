"""Typed contracts shared by the API, services, and recommendation engine.

These models deliberately separate a comparable fitness observation from the
full difficulty of a session.  Pace at the reference heart rate is evidence
about fitness; distance, duration, intensity-weighted load, and recent load are
context that feedback and coaching rules must consider independently.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiModel(BaseModel):
    """Base model with strict, frontend-friendly validation."""

    model_config = ConfigDict(extra="forbid")


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNAVAILABLE = "unavailable"


class FitnessTrend(StrEnum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


class WorkoutType(StrEnum):
    REST = "rest"
    EASY = "easy"
    RECOVERY = "recovery"
    LONG = "long"
    TEMPO_THRESHOLD = "tempo_threshold"
    INTERVALS = "intervals"
    RACE = "race"
    RUN_WALK = "run_walk"
    HIKE = "hike"
    BIKE = "bike"
    OTHER = "other"
    UNKNOWN = "unknown"


class QualitySessionType(StrEnum):
    SHORT_INTERVALS = "short_intervals"
    LONG_INTERVALS = "long_intervals"
    THRESHOLD = "threshold"
    PROGRESSION = "progression"
    HILL_REPEATS = "hill_repeats"


class ActivityHealthTag(StrEnum):
    NORMAL = "normal"
    ILLNESS = "illness"
    ILLNESS_RECOVERY = "illness_recovery"
    INJURY_AFFECTED = "injury_affected"
    OTHER_ABNORMAL = "other_abnormal"


class CurrentHealthStatus(StrEnum):
    NORMAL = "normal"
    LITTLE_TIRED = "little_tired"
    SICK_OR_RECOVERING = "sick_or_recovering"
    PAIN_OR_INJURY_CONCERN = "pain_or_injury_concern"


class DataQuality(StrEnum):
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"
    UNAVAILABLE = "unavailable"


class ReadinessFlag(StrEnum):
    READY = "ready"
    CAUTION = "caution"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"


class EvidenceAvailability(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


class ContextEvidence(ApiModel):
    factor: str
    availability: EvidenceAvailability
    reliability: ConfidenceLevel
    detail: str


class HealthResponse(ApiModel):
    status: str
    service: str = "running-coach"
    schema_version: int | None = None
    database_path: str
    database_exists: bool
    counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PaceValue(ApiModel):
    minutes_per_mile: float = Field(gt=0)
    display: str


class PaceChange(ApiModel):
    seconds_per_mile: float
    direction: FitnessTrend
    comparison_days: int = Field(gt=0)


class AdjustmentContribution(ApiModel):
    name: str
    minutes_per_mile: float
    evidence: str
    confidence: ConfidenceLevel
    available: bool = True


class FitnessObservation(ApiModel):
    activity_id: int
    activity_uid: str
    start_time: datetime
    raw_pace_at_target_hr: PaceValue
    environmental_adjustment_min_mile: float
    standardized_pace_at_target_hr: PaceValue
    uncertainty_95_min_mile: float = Field(ge=0)
    comparable_window_minutes: float | None = Field(default=None, ge=0)
    contributions: list[AdjustmentContribution] = Field(default_factory=list)
    confidence: ConfidenceLevel
    included_in_trend: bool
    exclusion_reasons: list[str] = Field(default_factory=list)


class ZoneBreakdown(ApiModel):
    zone_seconds: dict[str, float] = Field(default_factory=dict)
    zone_fractions: dict[str, float] = Field(default_factory=dict)
    easy_minutes: float = Field(default=0, ge=0)
    moderate_minutes: float = Field(default=0, ge=0)
    hard_minutes: float = Field(default=0, ge=0)


class SessionDifficulty(ApiModel):
    distance_miles: float = Field(ge=0)
    moving_minutes: float = Field(ge=0)
    elapsed_minutes: float = Field(ge=0)
    stopped_minutes: float = Field(ge=0)
    zone_load: float | None = Field(default=None, ge=0)
    perceived_exertion: int | None = Field(default=None, ge=1, le=10)
    session_rpe_load: float | None = Field(default=None, ge=0)
    elevation_gain_ft: float | None = Field(default=None, ge=0)
    elevation_loss_ft: float | None = Field(default=None, ge=0)
    zone_breakdown: ZoneBreakdown
    is_long_run: bool = False
    is_quality_session: bool = False
    difficulty_flags: list[str] = Field(default_factory=list)


class LoadWindow(ApiModel):
    days: int = Field(gt=0)
    distance_miles: float = Field(ge=0)
    moving_minutes: float = Field(ge=0)
    zone_load: float | None = Field(default=None, ge=0)
    hard_minutes: float = Field(default=0, ge=0)
    activity_count: int = Field(ge=0)


class LoadContext(ApiModel):
    trailing_7d: LoadWindow
    trailing_14d: LoadWindow
    trailing_28d: LoadWindow
    acute_to_prior_ratio: float | None = Field(
        default=None,
        ge=0,
        description="Trailing 7-day load divided by the prior 28-day weekly norm.",
    )
    acute_distance_to_capacity_ratio: float | None = Field(
        default=None,
        ge=0,
        description="Trailing 7-day distance divided by retained demonstrated weekly capacity.",
    )
    prior_28d_weekly_miles: float | None = Field(default=None, ge=0)
    sustained_capacity_miles: float | None = Field(default=None, ge=0)
    capacity_reference_miles: float | None = Field(default=None, ge=0)
    confidence: ConfidenceLevel
    flags: list[str] = Field(default_factory=list)


class RunMetadata(ApiModel):
    workout_type: WorkoutType = WorkoutType.UNKNOWN
    health_tag: ActivityHealthTag = ActivityHealthTag.NORMAL
    include_in_model: bool | None = None
    perceived_exertion: int | None = Field(default=None, ge=1, le=10)
    notes: str = Field(default="", max_length=4000)
    postal_code: str | None = Field(default=None, pattern=r"^\d{5}$")
    location_label: str | None = None


class RunMetadataPatch(ApiModel):
    workout_type: WorkoutType | None = None
    health_tag: ActivityHealthTag | None = None
    include_in_model: bool | None = None
    perceived_exertion: int | None = Field(default=None, ge=1, le=10)
    notes: str | None = Field(default=None, max_length=4000)
    postal_code: str | None = Field(default=None, pattern=r"^\d{5}$")

    @model_validator(mode="after")
    def require_one_change(self) -> "RunMetadataPatch":
        if not self.model_fields_set:
            raise ValueError("At least one metadata field is required")
        return self


class RunSummary(ApiModel):
    activity_id: int
    activity_uid: str
    start_time: datetime | None
    distance_miles: float = Field(ge=0)
    moving_minutes: float | None = Field(default=None, ge=0)
    moving_pace_min_mile: float | None = Field(default=None, gt=0)
    average_hr_bpm: float | None = Field(default=None, gt=0)
    maximum_hr_bpm: float | None = Field(default=None, gt=0)
    temperature_f: float | None = None
    gps_quality: str
    model_included: bool | None = None
    assessment_label: str
    workout_type: WorkoutType
    health_tag: ActivityHealthTag
    data_quality: DataQuality
    fitness_observation: FitnessObservation | None = None
    session_difficulty: SessionDifficulty | None = None


class Split(ApiModel):
    index: int = Field(gt=0)
    distance_miles: float = Field(gt=0)
    moving_minutes: float = Field(ge=0)
    pace_min_mile: float | None = Field(default=None, gt=0)
    average_hr_bpm: float | None = Field(default=None, gt=0)
    elevation_change_feet: float | None = None
    is_partial: bool = False


class WeatherSnapshot(ApiModel):
    temperature_f: float | None = None
    dewpoint_f: float | None = None
    apparent_temperature_f: float | None = None
    relative_humidity_percent: float | None = None
    wind_speed_mph: float | None = Field(default=None, ge=0)
    wind_gust_mph: float | None = Field(default=None, ge=0)
    headwind_mph: float | None = Field(default=None, ge=0)
    precipitation_in: float | None = Field(default=None, ge=0)
    quality: str | None = None


class DriftAssessment(ApiModel):
    decoupling_percent: float | None = None
    valid: bool
    confidence: ConfidenceLevel
    reason: str


class WorkoutAnalysisMetric(ApiModel):
    name: str
    value: str
    detail: str


class WorkoutAnalysisDimension(ApiModel):
    status: str
    summary: str
    confidence: ConfidenceLevel
    metrics: list[WorkoutAnalysisMetric] = Field(default_factory=list)


class IntervalRepetition(ApiModel):
    index: int = Field(gt=0)
    kind: str
    source: str
    duration_seconds: float = Field(gt=0)
    distance_miles: float = Field(ge=0)
    pace_min_mile: float | None = Field(default=None, gt=0)
    average_hr_bpm: float | None = Field(default=None, gt=0)
    end_hr_bpm: float | None = Field(default=None, gt=0)
    minimum_hr_bpm: float | None = Field(default=None, gt=0)
    maximum_hr_bpm: float | None = Field(default=None, gt=0)
    average_cadence_spm: float | None = Field(default=None, gt=0)
    recovery_after_seconds: float | None = Field(default=None, ge=0)
    recovery_start_hr_bpm: float | None = Field(default=None, gt=0)
    recovery_min_hr_bpm: float | None = Field(default=None, gt=0)
    recovery_hr_drop_bpm: float | None = Field(default=None, ge=0)
    recovery_hr_drop_percent: float | None = Field(default=None, ge=0)


class IntervalAnalysis(ApiModel):
    available: bool
    source: str
    confidence: ConfidenceLevel
    work_repetition_count: int = Field(ge=0)
    recovery_repetition_count: int = Field(ge=0)
    mean_work_pace_min_mile: float | None = Field(default=None, gt=0)
    median_work_time_seconds: float | None = Field(default=None, gt=0)
    mean_work_time_seconds: float | None = Field(default=None, gt=0)
    fastest_work_time_seconds: float | None = Field(default=None, gt=0)
    slowest_work_time_seconds: float | None = Field(default=None, gt=0)
    work_speed_cv_percent: float | None = Field(default=None, ge=0)
    fade_percent: float | None = None
    first_to_last_percent: float | None = None
    pacing_pattern: str | None = None
    final_rep_overspeed_percent: float | None = None
    recovery_time_cv_percent: float | None = Field(default=None, ge=0)
    work_minutes: float = Field(default=0, ge=0)
    work_distance_miles: float = Field(default=0, ge=0)
    work_z4_z5_minutes: float | None = Field(default=None, ge=0)
    work_recovery_speed_separation_percent: float | None = None
    explanation: str
    repetitions: list[IntervalRepetition] = Field(default_factory=list)


class HistoricalWorkoutComparison(ApiModel):
    available: bool
    activity_id: int | None = None
    date: datetime | None = None
    summary: str
    metrics: list[WorkoutAnalysisMetric] = Field(default_factory=list)


class WorkoutAnalysis(ApiModel):
    workout_type: WorkoutType
    definition: str
    execution: WorkoutAnalysisDimension
    control: WorkoutAnalysisDimension
    stimulus: WorkoutAnalysisDimension
    recovery: WorkoutAnalysisDimension
    interval_analysis: IntervalAnalysis | None = None
    historical_comparison: HistoricalWorkoutComparison | None = None
    progression_recommendation: str | None = None


class RunFeedback(ApiModel):
    run: RunSummary
    metadata: RunMetadata
    splits: list[Split] = Field(default_factory=list)
    weather: WeatherSnapshot | None = None
    cardiac_drift: DriftAssessment
    workout_analysis: WorkoutAnalysis | None = None
    load_context_before_run: LoadContext | None = None
    assessment: str
    positives: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    comparison_text: str | None = None


class PeriodSummary(ApiModel):
    start_date: date
    end_date: date
    run_count: int = Field(ge=0)
    distance_miles: float = Field(ge=0)
    moving_minutes: float = Field(ge=0)
    zone_load: float | None = Field(default=None, ge=0)
    standardized_pace_min_mile: float | None = Field(default=None, gt=0)
    longest_run_miles: float = Field(default=0, ge=0)


class PeriodComparison(ApiModel):
    current: PeriodSummary
    previous: PeriodSummary
    pace_change_seconds_per_mile: float | None = None
    distance_change_percent: float | None = None
    load_change_percent: float | None = None
    interpretation: str


class FitnessPoint(ApiModel):
    activity_id: int
    start_time: datetime
    raw_pace_min_mile: float | None = Field(default=None, gt=0)
    standardized_pace_min_mile: float = Field(gt=0)
    uncertainty_95_min_mile: float = Field(ge=0)
    distance_miles: float = Field(ge=0)
    zone_load: float | None = Field(default=None, ge=0)
    workout_type: WorkoutType
    health_tag: ActivityHealthTag = ActivityHealthTag.NORMAL
    included_in_trend: bool = True
    trend_weight: float = Field(default=1.0, ge=0, le=1)
    measurement_quality: str | None = None
    benchmark_quality: str | None = None


class FitnessTrendPoint(ApiModel):
    as_of: datetime
    pace_min_mile: float = Field(gt=0)
    uncertainty_95_min_mile: float = Field(ge=0)
    run_count: int = Field(ge=1)


class FitnessBenchmarkSummary(ApiModel):
    definition: str
    trend: FitnessTrend
    confidence: ConfidenceLevel
    current_pace: PaceValue | None = None
    uncertainty_95_min_mile: float | None = Field(default=None, ge=0)
    pace_change_seconds_per_mile: float | None = None
    eligible_run_count: int = Field(default=0, ge=0)
    strict_run_count: int = Field(default=0, ge=0)
    estimated_run_count: int = Field(default=0, ge=0)
    series: list[FitnessPoint] = Field(default_factory=list)
    trend_7d: list[FitnessTrendPoint] = Field(default_factory=list)
    trend_28d: list[FitnessTrendPoint] = Field(default_factory=list)


class FitnessCoverageItem(ApiModel):
    activity_id: int
    start_time: datetime
    distance_miles: float = Field(ge=0)
    workout_type: WorkoutType
    health_tag: ActivityHealthTag = ActivityHealthTag.NORMAL
    score_status: str
    standardized_pace_min_mile: float | None = Field(default=None, gt=0)
    included_in_trend: bool = False
    trend_weight: float = Field(default=0.0, ge=0, le=1)
    reason: str


class ExternalFitnessSnapshotInput(ApiModel):
    measured_at: date
    vo2_max: float | None = Field(default=None, gt=10, le=100)
    predicted_5k_seconds: int | None = Field(default=None, gt=600, le=10800)
    predicted_10k_seconds: int | None = Field(default=None, gt=1200, le=21600)
    predicted_half_marathon_seconds: int | None = Field(default=None, gt=2400, le=43200)
    predicted_marathon_seconds: int | None = Field(default=None, gt=4800, le=86400)
    source: str = Field(default="Garmin", max_length=100)

    @model_validator(mode="after")
    def require_metric(self) -> "ExternalFitnessSnapshotInput":
        values = (
            self.vo2_max,
            self.predicted_5k_seconds,
            self.predicted_10k_seconds,
            self.predicted_half_marathon_seconds,
            self.predicted_marathon_seconds,
        )
        if all(value is None for value in values):
            raise ValueError("Enter VO2 max or at least one race prediction")
        return self


class ExternalFitnessSnapshot(ExternalFitnessSnapshotInput):
    id: int


class ExternalFitnessSummary(ApiModel):
    snapshots: list[ExternalFitnessSnapshot] = Field(default_factory=list)
    vo2_max_trend: FitnessTrend = FitnessTrend.INSUFFICIENT_DATA
    race_prediction_trend: FitnessTrend = FitnessTrend.INSUFFICIENT_DATA
    confidence: ConfidenceLevel = ConfidenceLevel.UNAVAILABLE
    interpretation: str


class LocalVo2Estimate(ApiModel):
    value_ml_kg_min: float | None = Field(default=None, gt=0)
    uncertainty_95_ml_kg_min: float | None = Field(default=None, ge=0)
    method: str
    confidence: ConfidenceLevel
    trend: FitnessTrend
    demographic_baseline_ml_kg_min: float | None = Field(default=None, gt=0)
    demographic_uncertainty_95_ml_kg_min: float | None = Field(default=None, ge=0)
    interpretation: str
    limitations: list[str] = Field(default_factory=list)


class ConsistencySummary(ApiModel):
    running_days: int = Field(ge=0)
    runs_per_week: float = Field(ge=0)
    longest_gap_days: float | None = Field(default=None, ge=0)
    longest_run_miles: float = Field(ge=0)
    quality_sessions: int = Field(ge=0)


class IntensitySummary(ApiModel):
    easy_percent: float | None = Field(default=None, ge=0, le=100)
    moderate_percent: float | None = Field(default=None, ge=0, le=100)
    hard_percent: float | None = Field(default=None, ge=0, le=100)
    known_hr_minutes: float = Field(ge=0)
    missing_hr_minutes: float = Field(ge=0)
    confidence: ConfidenceLevel


class ProgressResponse(ApiModel):
    as_of: datetime
    window_days: int
    reference_within_run_minutes: float = Field(default=20, gt=0)
    available_windows: list[int]
    fitness_trend: FitnessTrend
    fitness_confidence: ConfidenceLevel
    current_pace: PaceValue | None = None
    uncertainty_95_min_mile: float | None = Field(default=None, ge=0)
    pace_change_seconds_per_mile: float | None = None
    pace_change_uncertainty_95_seconds_per_mile: float | None = Field(default=None, ge=0)
    definition: str
    series: list[FitnessPoint]
    trend_7d: list[FitnessTrendPoint]
    trend_28d: list[FitnessTrendPoint]
    steady_aerobic: FitnessBenchmarkSummary
    activity_coverage: list[FitnessCoverageItem] = Field(default_factory=list)
    period_comparison: PeriodComparison
    current_load: LoadContext
    consistency: ConsistencySummary
    intensity: IntensitySummary
    external_fitness: ExternalFitnessSummary
    local_vo2_estimate: LocalVo2Estimate
    blind_spots: list[str] = Field(default_factory=list)


class FitnessHorizon(ApiModel):
    label: str
    window_days: int = Field(gt=0)
    trend: FitnessTrend
    confidence: ConfidenceLevel
    pace_change_seconds_per_mile: float | None = None
    current_pace: PaceValue | None = None


class FitnessSignal(ApiModel):
    label: str
    trend: FitnessTrend
    status: str
    confidence: ConfidenceLevel
    detail: str


class FitnessInterpretation(ApiModel):
    headline: str
    summary: str
    short_term: FitnessHorizon
    long_term: FitnessHorizon
    capacity_direction: FitnessTrend
    capacity_summary: str
    signals: list[FitnessSignal] = Field(default_factory=list)
    illness_context: str | None = None
    caveats: list[str] = Field(default_factory=list)


class DashboardResponse(ApiModel):
    progress: ProgressResponse
    fitness_interpretation: FitnessInterpretation
    last_run: RunFeedback | None = None
    recommendation: RecommendationResponse
    current_status: RecommendationRequest
    weekly_schedule: "WeeklyScheduleResponse | None" = None


class FitnessState(ApiModel):
    """Compact, serializable input to the Python recommendation engine."""

    as_of: datetime
    window_days: int = Field(gt=0)
    fitness_trend: FitnessTrend
    trend_confidence: ConfidenceLevel
    standardized_pace_at_target_hr: PaceValue | None = None
    short_term_change: PaceChange | None = None
    medium_term_change: PaceChange | None = None
    recent_load: LoadContext
    days_since_last_run: float | None = Field(default=None, ge=0)
    days_since_quality_run: float | None = Field(default=None, ge=0)
    days_since_long_run: float | None = Field(default=None, ge=0)
    last_run: SessionDifficulty | None = None
    last_run_workout_type: WorkoutType | None = None
    last_run_drift_percent: float | None = None
    longest_run_30d_miles: float = Field(default=0, ge=0)
    quality_sessions_14d: int = Field(default=0, ge=0)
    completed_quality_session_count: int = Field(default=0, ge=0)
    running_days_28d: int = Field(default=0, ge=0)
    easy_fraction_14d: float | None = Field(default=None, ge=0, le=1)
    moderate_fraction_14d: float | None = Field(default=None, ge=0, le=1)
    moderate_evidence_runs_14d: int = Field(default=0, ge=0)
    hard_fraction_14d: float | None = Field(default=None, ge=0, le=1)
    recent_performance_anomaly: str = "unknown"
    recent_illness_or_recovery: bool = False
    normal_runs_since_health_event: int = Field(default=0, ge=0)
    current_health_status: CurrentHealthStatus = CurrentHealthStatus.NORMAL
    anomaly_flags: list[str] = Field(default_factory=list)
    data_quality_flags: list[str] = Field(default_factory=list)
    context_evidence: list[ContextEvidence] = Field(default_factory=list)
    known_blind_spots: list[str] = Field(default_factory=list)
    planned_weather: "PlannedWeather | None" = None


class PlannedWeather(ApiModel):
    """Best-effort forecast at an anonymized recent route centroid."""

    forecast_time: datetime
    temperature_f: float | None = None
    dewpoint_f: float | None = None
    apparent_temperature_f: float | None = None
    wind_speed_mph: float | None = None
    wind_gust_mph: float | None = None
    precipitation_probability_percent: float | None = Field(default=None, ge=0, le=100)
    precipitation_in: float | None = Field(default=None, ge=0)
    source: str = "Open-Meteo forecast"
    location_basis: str = "privacy-jittered recent route centroid"
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE


class RecommendationRequest(ApiModel):
    health_status: CurrentHealthStatus
    planned_at: datetime | None = None
    notes: str = Field(default="", max_length=2000)
    desired_distance_miles: float | None = Field(default=None, gt=0, le=50)


class WorkoutStep(ApiModel):
    instruction: str
    duration_minutes: float | None = Field(default=None, gt=0)
    distance_miles: float | None = Field(default=None, gt=0)
    target_zones: list[str] = Field(default_factory=list)


class RuleTrace(ApiModel):
    rule_id: str
    description: str
    fired: bool
    facts: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RecommendationResponse(ApiModel):
    generated_at: datetime
    fitness_state_as_of: datetime
    planned_for: datetime | None = None
    planned_weather: PlannedWeather | None = None
    workout_type: WorkoutType
    quality_session_type: QualitySessionType | None = None
    title: str
    distance_range_miles: tuple[float, float] | None = None
    duration_range_minutes: tuple[float, float] | None = None
    target_zones: list[str] = Field(default_factory=list)
    structure: list[WorkoutStep] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    modification_rules: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    readiness: ReadinessFlag
    readiness_reason: str = ""
    rule_trace: list[RuleTrace] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_workout_extent(self) -> "RecommendationResponse":
        if (
            self.workout_type != WorkoutType.REST
            and self.distance_range_miles is None
            and self.duration_range_minutes is None
        ):
            raise ValueError("A recommendation needs a distance or duration range")
        for value in (self.distance_range_miles, self.duration_range_minutes):
            if value is not None and value[0] > value[1]:
                raise ValueError("Recommendation range minimum cannot exceed maximum")
        return self


class WeeklyScheduleRequest(ApiModel):
    health_status: CurrentHealthStatus
    notes: str = Field(default="", max_length=2000)


class WeeklyScheduleDay(ApiModel):
    date: date
    planned_at: datetime | None = None
    recommendation: RecommendationResponse | None = None
    day_role: str
    rationale: str
    completed_activities: list["TrailingDayActivity"] = Field(default_factory=list)


class TrailingDayActivity(ApiModel):
    activity_id: int
    start_time: datetime
    distance_miles: float = Field(ge=0)
    workout_type: WorkoutType
    health_tag: ActivityHealthTag


class TrailingCalendarDay(ApiModel):
    date: date
    day_role: str
    total_distance_miles: float = Field(ge=0)
    activities: list[TrailingDayActivity] = Field(default_factory=list)


class WeeklyTargetEvidence(ApiModel):
    recent_7d_miles: float = Field(ge=0)
    chronic_42d_weekly_miles: float = Field(ge=0)
    best_sustained_28d_weekly_miles: float = Field(ge=0)
    peak_7d_miles: float = Field(ge=0)
    demonstrated_run_days_per_week: float = Field(ge=0, le=7)
    capacity_reference_miles: float = Field(ge=0)
    rationale: str


class WeeklyScheduleResponse(ApiModel):
    generated_at: datetime
    start_date: date
    end_date: date
    target_run_count: int = Field(ge=0, le=7)
    target_distance_range_miles: tuple[float, float]
    target_evidence: WeeklyTargetEvidence
    trailing_days: list[TrailingCalendarDay] = Field(default_factory=list)
    completed_run_count: int = Field(default=0, ge=0, le=7)
    run_count: int = Field(ge=0, le=7)
    projected_distance_range_miles: tuple[float, float]
    summary: str
    days: list[WeeklyScheduleDay]


class UploadStage(ApiModel):
    name: str
    status: str
    detail: str = ""


class UploadedFileResult(ApiModel):
    filename: str
    status: str
    activity_ids: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class UploadResponse(ApiModel):
    files: list[UploadedFileResult]
    stages: list[UploadStage]
    primary_activity_id: int | None = None


class ZoneRange(ApiModel):
    minimum_bpm: int = Field(gt=0)
    maximum_bpm: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> "ZoneRange":
        if self.minimum_bpm > self.maximum_bpm:
            raise ValueError("minimum_bpm cannot exceed maximum_bpm")
        return self


class MovingTimeSettings(ApiModel):
    minimum_running_speed_mps: float = Field(gt=0)
    stopped_speed_mps: float = Field(ge=0)
    gps_stopped_speed_mps: float = Field(ge=0)
    stopped_distance_meters: float = Field(ge=0)
    maximum_interval_seconds: float = Field(gt=0)
    minimum_stop_seconds: float = Field(ge=0)
    maximum_plausible_speed_mps: float = Field(gt=0)


class QualitySessionSettings(ApiModel):
    short_intervals: bool = True
    long_intervals: bool = True
    threshold: bool = True
    progression: bool = True
    hill_repeats: bool = False

    @model_validator(mode="after")
    def at_least_one_enabled(self) -> "QualitySessionSettings":
        if not any(self.model_dump().values()):
            raise ValueError("At least one quality-session type must remain enabled")
        return self


class CoachingSettings(ApiModel):
    training_goal: str
    long_run_progression_factor: float = Field(ge=1, le=1.5)
    high_load_ratio: float = Field(gt=1, le=3)
    moderate_intensity_leakage_fraction: float = Field(ge=0, le=1)
    minimum_days_between_quality_sessions: float = Field(ge=1, le=14)
    quality_recency_reference_days: float = Field(ge=4, le=21)
    typical_rest_days_between_runs: int = Field(ge=0, le=3)
    capacity_retention_half_life_days: float = Field(ge=14, le=120)
    capacity_retention_grace_days: int = Field(ge=0, le=90)
    minimum_running_days_28d_for_quality: int = Field(ge=1, le=28)
    long_run_recency_reference_days: float = Field(ge=5, le=30)
    reduced_volume_factor: float = Field(gt=0, le=1)
    quality_sessions: QualitySessionSettings = Field(default_factory=QualitySessionSettings)

    @model_validator(mode="after")
    def ordered_weekly_mileage(self) -> "CoachingSettings":
        return self


class ProfileSettings(ApiModel):
    birth_date: date
    sex: str = Field(pattern="^(male|female)$")
    weight_lb: float = Field(gt=50, le=1000)
    height_in: float = Field(gt=36, le=100)


class SettingsResponse(ApiModel):
    max_hr: int = Field(gt=0)
    resting_hr: int = Field(gt=0)
    target_hr: int = Field(gt=0)
    zones: dict[str, ZoneRange]
    reference_temperature_f: float
    reference_dewpoint_f: float
    reference_wind_mph: float = Field(ge=0)
    reference_grade_percent: float
    reference_within_run_minutes: float = Field(gt=0)
    weather_privacy_radius_km: float = Field(ge=0)
    historical_weather_enabled: bool = False
    forecast_weather_enabled: bool = False
    available_fitness_windows: list[int]
    default_fitness_window: int
    moving_time: MovingTimeSettings
    coaching: CoachingSettings
    profile: ProfileSettings | None = None
    recalculation: list[UploadStage] = Field(default_factory=list)


class SettingsPatch(ApiModel):
    max_hr: int | None = Field(default=None, gt=0)
    resting_hr: int | None = Field(default=None, gt=0)
    target_hr: int | None = Field(default=None, gt=0)
    zones: dict[str, ZoneRange] | None = None
    reference_temperature_f: float | None = None
    reference_dewpoint_f: float | None = None
    reference_wind_mph: float | None = Field(default=None, ge=0)
    reference_grade_percent: float | None = None
    reference_within_run_minutes: float | None = Field(default=None, gt=0)
    weather_privacy_radius_km: float | None = Field(default=None, ge=0)
    historical_weather_enabled: bool | None = None
    forecast_weather_enabled: bool | None = None
    default_fitness_window: int | None = Field(default=None, gt=0)
    moving_time: MovingTimeSettings | None = None
    coaching: CoachingSettings | None = None
    profile: ProfileSettings | None = None

    @model_validator(mode="after")
    def require_one_change(self) -> "SettingsPatch":
        if not self.model_fields_set:
            raise ValueError("At least one setting is required")
        return self
