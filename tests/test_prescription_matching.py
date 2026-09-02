from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from run_analysis.db import connect, initialize
from run_analysis.prescription_matching import (
    archive_weekly_prescriptions,
    match_activities_to_prescriptions,
)
from run_analysis.web.schemas import (
    ConfidenceLevel,
    QualitySessionType,
    ReadinessFlag,
    RecommendationResponse,
    WeeklyScheduleDay,
    WeeklyScheduleResponse,
    WeeklyTargetEvidence,
    WorkoutStep,
    WorkoutType,
)


def _threshold(planned_for: datetime) -> RecommendationResponse:
    return RecommendationResponse(
        generated_at=planned_for - timedelta(hours=1),
        fitness_state_as_of=planned_for - timedelta(hours=1),
        planned_for=planned_for,
        workout_type=WorkoutType.TEMPO_THRESHOLD,
        quality_session_type=QualitySessionType.THRESHOLD,
        title="Continuous threshold run",
        distance_range_miles=(4.5, 5.0),
        target_zones=["Z1", "Z2", "upper Z3 / low Z4"],
        structure=[
            WorkoutStep(
                instruction="Easy warm-up.",
                duration_minutes=12,
                target_zones=["Z1", "Z2"],
            ),
            WorkoutStep(
                instruction="Run 18 minutes continuously at threshold.",
                duration_minutes=18,
                target_zones=["upper Z3", "low Z4"],
            ),
            WorkoutStep(
                instruction="Easy cool-down.",
                duration_minutes=10,
                target_zones=["Z1", "Z2"],
            ),
        ],
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )


def _schedule(
    generated_at: datetime,
    recommendation: RecommendationResponse,
) -> WeeklyScheduleResponse:
    plan_date = recommendation.planned_for.date()
    evidence = WeeklyTargetEvidence(
        recent_7d_miles=12,
        chronic_42d_weekly_miles=15,
        best_sustained_28d_weekly_miles=16,
        peak_7d_miles=18,
        demonstrated_run_days_per_week=4,
        capacity_reference_miles=16,
        rationale="test",
    )
    return WeeklyScheduleResponse(
        generated_at=generated_at,
        start_date=plan_date,
        end_date=plan_date + timedelta(days=6),
        target_run_count=4,
        target_distance_range_miles=(15.5, 18.0),
        target_evidence=evidence,
        run_count=1,
        projected_distance_range_miles=(4.5, 5.0),
        summary="test",
        days=[
            WeeklyScheduleDay(
                date=plan_date,
                planned_at=recommendation.planned_for,
                recommendation=recommendation,
                day_role="quality_run",
                rationale="test",
            )
        ],
    )


def _insert_activity(connection, start: datetime) -> int:
    cursor = connection.execute(
        """
        INSERT INTO activities(
            activity_uid,activity_id,sport,start_time_utc,
            start_time_utc_epoch,total_distance_m,lap_count,
            trackpoint_count,gps_quality,hr_quality,elevation_quality,
            cadence_quality,distance_source,namespaces_json,
            data_quality_json,created_at_utc,updated_at_utc
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "matched-threshold",
            start.isoformat(),
            "Running",
            start.isoformat(),
            start.timestamp(),
            4.41 * 1609.344,
            3,
            100,
            "good",
            "good",
            "good",
            "good",
            "device",
            "{}",
            "{}",
            start.isoformat(),
            start.isoformat(),
        ),
    )
    return int(cursor.lastrowid)


def _insert_activity_metrics(connection, activity_id: int, minutes: float) -> None:
    connection.execute(
        """
        INSERT INTO activity_metrics(
            activity_id,metrics_json,calculated_at_utc,
            calculated_moving_time_s
        ) VALUES (?, '{}', 'now', ?)
        """,
        (activity_id, minutes * 60),
    )


def test_delayed_upload_matches_plan_that_existed_before_run(tmp_path) -> None:
    sunday = datetime(2026, 8, 30, 19, tzinfo=timezone.utc)
    activity_start = sunday + timedelta(minutes=36)
    with connect(tmp_path / "matching.sqlite") as connection:
        initialize(connection)
        original = _threshold(sunday)
        archive_weekly_prescriptions(
            connection,
            _schedule(sunday - timedelta(hours=6), original),
        )
        # A Monday reload may roll the missing workout forward, but it was
        # generated after the athlete had already run and cannot rewrite the
        # historical prescription.
        rolled = _threshold(sunday + timedelta(days=1))
        archive_weekly_prescriptions(
            connection,
            _schedule(sunday + timedelta(hours=14), rolled),
        )
        activity_id = _insert_activity(connection, activity_start)

        assert match_activities_to_prescriptions(
            connection, [activity_id]
        ) == [activity_id]
        matched = connection.execute(
            """
            SELECT ph.planned_for,ph.workout_type,ap.match_confidence,
                   ro.workout_type AS override_type
            FROM activity_plan_matches ap
            JOIN planned_workout_history ph ON ph.id=ap.planned_workout_id
            JOIN activities a ON a.id=ap.activity_id
            LEFT JOIN run_overrides ro ON ro.activity_id=a.activity_id
            WHERE ap.activity_id=?
            """,
            (activity_id,),
        ).fetchone()

    assert datetime.fromisoformat(matched["planned_for"]) == sunday
    assert matched["workout_type"] == "tempo_threshold"
    assert matched["override_type"] is None
    assert matched["match_confidence"] == "high"


def test_matching_uses_latest_pre_run_plan_not_best_historical_fit(tmp_path) -> None:
    planned_for = datetime(2026, 8, 30, 19, tzinfo=timezone.utc)
    activity_start = planned_for + timedelta(minutes=30)
    with connect(tmp_path / "latest-plan.sqlite") as connection:
        initialize(connection)
        older = _threshold(planned_for)
        archive_weekly_prescriptions(
            connection,
            _schedule(planned_for - timedelta(days=1), older),
        )
        latest = _threshold(planned_for + timedelta(hours=2))
        latest.title = "Latest pre-run prescription"
        latest.distance_range_miles = (7.0, 7.5)
        archive_weekly_prescriptions(
            connection,
            _schedule(planned_for - timedelta(hours=2), latest),
        )
        activity_id = _insert_activity(connection, activity_start)

        assert match_activities_to_prescriptions(connection, [activity_id]) == [activity_id]
        matched = connection.execute(
            """
            SELECT ph.title,ph.distance_low_miles
            FROM activity_plan_matches ap
            JOIN planned_workout_history ph ON ph.id=ap.planned_workout_id
            WHERE ap.activity_id=?
            """,
            (activity_id,),
        ).fetchone()

    assert matched["title"] == "Latest pre-run prescription"
    assert matched["distance_low_miles"] == 7.0


def test_latest_rest_snapshot_prevents_an_obsolete_workout_match(tmp_path) -> None:
    planned_for = datetime(2026, 8, 30, 19, tzinfo=timezone.utc)
    activity_start = planned_for + timedelta(minutes=30)
    with connect(tmp_path / "rest-plan.sqlite") as connection:
        initialize(connection)
        archive_weekly_prescriptions(
            connection,
            _schedule(planned_for - timedelta(days=1), _threshold(planned_for)),
        )
        rest_schedule = _schedule(
            planned_for - timedelta(hours=2), _threshold(planned_for)
        ).model_copy(
            update={
                "days": [
                    WeeklyScheduleDay(
                        date=planned_for.date(),
                        day_role="rest",
                        rationale="latest plan says rest",
                    )
                ]
            }
        )
        archive_weekly_prescriptions(connection, rest_schedule)
        activity_id = _insert_activity(connection, activity_start)

        matched = match_activities_to_prescriptions(connection, [activity_id])

    assert matched == []


def test_time_based_baseline_matches_by_duration_not_distance(tmp_path) -> None:
    planned_for = datetime(2026, 8, 30, 19, tzinfo=timezone.utc)
    baseline = RecommendationResponse(
        generated_at=planned_for - timedelta(hours=1),
        fitness_state_as_of=planned_for - timedelta(hours=1),
        planned_for=planned_for,
        workout_type=WorkoutType.EASY,
        title="Conversational baseline run",
        duration_range_minutes=(10.0, 30.0),
        target_zones=["Z2"],
        structure=[
            WorkoutStep(
                instruction="Run or run/walk conversationally.",
                target_zones=["Z2"],
            )
        ],
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )
    with connect(tmp_path / "duration-plan.sqlite") as connection:
        initialize(connection)
        archive_weekly_prescriptions(
            connection,
            _schedule(planned_for - timedelta(hours=2), baseline),
        )
        activity_id = _insert_activity(
            connection, planned_for + timedelta(minutes=15)
        )
        _insert_activity_metrics(connection, activity_id, 18.0)

        assert match_activities_to_prescriptions(
            connection, [activity_id]
        ) == [activity_id]
        matched = connection.execute(
            """
            SELECT ph.duration_low_minutes,ph.duration_high_minutes,
                   ap.distance_delta_miles,ap.duration_delta_minutes,
                   ap.match_confidence
            FROM activity_plan_matches ap
            JOIN planned_workout_history ph ON ph.id=ap.planned_workout_id
            WHERE ap.activity_id=?
            """,
            (activity_id,),
        ).fetchone()

    assert matched["duration_low_minutes"] == 10.0
    assert matched["duration_high_minutes"] == 30.0
    assert matched["distance_delta_miles"] == 0.0
    assert matched["duration_delta_minutes"] == 0.0
    assert matched["match_confidence"] == "high"
