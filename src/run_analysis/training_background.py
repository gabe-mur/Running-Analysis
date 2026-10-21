"""Self-reported starting exposure, never recorded training or measured fitness."""
from datetime import date
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingBackground(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    weekly_miles: float = Field(gt=0, le=200, strict=True)
    runs_per_week: int = Field(ge=2, le=7, strict=True)
    regular_long_miles: float = Field(gt=0, le=100, strict=True)
    maintaining_training: bool = Field(strict=True)
    last_run_date: date | None = None

    @model_validator(mode='after')
    def consistent(self):
        if self.regular_long_miles >= self.weekly_miles:
            raise ValueError('Regular long run must leave some weekly volume for your other runs')
        ordinary = (self.weekly_miles-self.regular_long_miles)/(self.runs_per_week-1)
        if ordinary > self.regular_long_miles + .05:
            raise ValueError('Regular long run cannot be shorter than the average of your other runs')
        if ordinary < .1:
            raise ValueError('Weekly volume must leave at least 0.1 miles for each other run')
        return self

    @property
    def ordinary_miles(self):
        from .staged_progression import tenth
        return tenth((self.weekly_miles-self.regular_long_miles)/(self.runs_per_week-1))


def reported_background(config):
    value = config.get('training_background')
    return TrainingBackground.model_validate(value) if value else None


def calibration_activity(connection, config, *, reported_at, as_of=None):
    """Require an actual, healthy HR/distance recording after the report.

    One usable easy run calibrates rate and duration, not weekly exposure.
    A hard session, GPS route with no HR, illness or injury is not a substitute.
    """
    rows = connection.execute('''SELECT a.*,m.calculated_moving_time_s AS moving_time_s,m.moving_average_hr_bpm,
        m.detected_workout_type,o.health_tag,o.illness,o.include_in_model,o.workout_type
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=COALESCE(NULLIF(a.activity_id,''),a.activity_uid)
        WHERE a.sport='Running' AND a.start_time_utc_epoch>=? ORDER BY a.start_time_utc_epoch''',
        (reported_at.timestamp(),)).fetchall()
    for row in rows:
        role = row['workout_type'] or row['detected_workout_type']
        if (row['health_tag'] not in (None, '', 'normal') or row['illness']
                or row['include_in_model'] == 0 or role not in ('easy', 'recovery', 'long', 'unknown', None)):
            continue
        if as_of is not None:
            from datetime import datetime
            sources = connection.execute('''SELECT s.imported_at_utc FROM source_files s
                JOIN activity_sources l ON l.source_file_id=s.id
                WHERE l.activity_id=? AND l.is_primary=1''', (row['id'],)).fetchall()
            if (row['start_time_utc_epoch'] >= as_of.timestamp()
                    or datetime.fromisoformat(row['created_at_utc']) > as_of
                    or not sources or any(datetime.fromisoformat(s[0]) > as_of for s in sources)):
                continue
        count, hrs = connection.execute('SELECT COUNT(*),SUM(heart_rate_bpm IS NOT NULL) '
            'FROM trackpoints WHERE activity_id=?', (row['id'],)).fetchone()
        if (count < 2 or (hrs or 0)/count < config['model']['minimum_hr_coverage']
                or not row['total_distance_m'] or not row['moving_time_s']
                or row['moving_time_s'] < 60*config['model'].get('minimum_run_minutes', 10)):
            continue
        easy, known = connection.execute('''SELECT SUM(heart_rate_bpm<=?),COUNT(heart_rate_bpm)
            FROM trackpoints WHERE activity_id=? AND heart_rate_bpm>? AND heart_rate_bpm<=?''',
            (config['zones']['z2'][1],row['id'],config['resting_hr'],config['max_hr'])).fetchone()
        if not known or (easy or 0)/known < .7:
            continue
        if as_of is not None:
            end = connection.execute('SELECT MAX(timestamp_utc) FROM trackpoints WHERE activity_id=?',
                (row['id'],)).fetchone()[0]
            if end is None or datetime.fromisoformat(end) > as_of:
                continue
        return dict(row)
    return None
