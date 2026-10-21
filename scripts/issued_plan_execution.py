"""Sandbox execution contract, independent of the planner's mutable output.

Known issued-contract history is supplied to the simulator. A separate actual
production-matcher probe reports which prescription an upload would match;
that result is NOT silently fed back as the runner's original intent.
"""

import sqlite3

from run_analysis.db import initialize
from run_analysis.prescription_matching import archive_weekly_prescriptions, match_activities_to_prescriptions
from run_analysis.web.schemas import WorkoutType


class OpeningPlanExecution:
    def __init__(self):
        self.issued_at = None
        self.opening = None
        self.decisions = []
        self.executions = []
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        initialize(self.connection)

    def close(self):
        self.connection.close()

    def select(self, start, end, schedule):
        if self.opening is None:
            self.issued_at = start
            self.opening = schedule.model_copy(deep=True)
        visible_dates = {day.date for day in self.opening.days}
        if start.date() not in visible_dates:
            raise ValueError("Opening-contract experiment cannot execute beyond its original visible dates")
        # The simulator's decision timestamp, not its future target-state date,
        # determines which prescription existed before the synthetic activity.
        archive_weekly_prescriptions(self.connection, schedule.model_copy(update={"generated_at": start}))
        chosen = [day.recommendation.model_copy(deep=True) for day in self.opening.days
                  if day.recommendation and day.recommendation.workout_type != WorkoutType.REST
                  and day.recommendation.planned_for and start <= day.recommendation.planned_for < end]
        self.decisions.append({"at": start.isoformat(), "issued_at": self.issued_at.isoformat(),
                               "execution_dates": [x.planned_for.date().isoformat() for x in chosen]})
        return chosen

    def observe(self, run, latest):
        actual_id = len(self.executions) + 1
        timestamp = run.start_time.isoformat()
        cursor = self.connection.execute(
            """INSERT INTO activities(
                activity_uid,activity_id,sport,start_time_utc,start_time_utc_epoch,total_distance_m,
                lap_count,trackpoint_count,gps_quality,hr_quality,elevation_quality,cadence_quality,
                distance_source,namespaces_json,data_quality_json,created_at_utc,updated_at_utc
            ) VALUES (?,?,?,?,?,?,0,0,'unknown','unknown','unknown','unknown','synthetic','{}','{}',?,?)""",
            (f"sandbox-{actual_id}", str(actual_id), "Running", timestamp, run.start_time.timestamp(),
             run.distance_miles * 1609.344, timestamp, timestamp),
        )
        activity_id = cursor.lastrowid
        self.connection.execute(
            "INSERT INTO activity_metrics(activity_id,metrics_json,calculated_at_utc,calculated_moving_time_s) VALUES (?, '{}', ?, ?)",
            (activity_id, timestamp, run.moving_minutes * 60),
        )
        match_activities_to_prescriptions(self.connection, [activity_id])
        matched = self.connection.execute(
            """SELECT p.schedule_generated_at,p.workout_type,p.distance_low_miles,p.distance_high_miles,
                      m.distance_delta_miles,m.match_confidence
               FROM activity_plan_matches m JOIN planned_workout_history p ON p.id=m.planned_workout_id
               WHERE m.activity_id=?""", (activity_id,),
        ).fetchone()
        issued_range = (run.prescribed_low_miles, run.prescribed_high_miles)
        latest_range = latest.distance_range_miles if latest else None
        self.executions.append({
            "at": timestamp, "issued_at": self.issued_at.isoformat(),
            "contract_age_days": (run.start_time - self.issued_at).total_seconds() / 86400,
            "actual_miles": run.distance_miles, "actual_type": run.workout_type.value,
            "issued_range": issued_range,
            "within_issued_range": issued_range[0] <= run.distance_miles <= issued_range[1],
            "latest_range": latest_range,
            "within_latest_range": latest_range[0] <= run.distance_miles <= latest_range[1] if latest_range else None,
            "latest_type_matches": latest.workout_type == run.workout_type if latest else None,
            "production_match": dict(matched) if matched else None,
            "history_association": "explicit original issued contract; production match is diagnostic only",
        })
