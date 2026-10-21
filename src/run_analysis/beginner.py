"""Measured starter sessions, separate from the established-athlete optimizer.

This is an explicit coaching policy, not an inferred physiological capacity.
Only completed, confirmed conversational sessions advance it. No synthetic runs,
quality credit, calendar-time growth or permanent optimizer weights are added.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
from statistics import median

from pydantic import BaseModel, Field, model_validator


class BeginnerFeedback(BaseModel):
    comfortable: bool
    running_minutes: float | None = Field(default=None, gt=0, le=180, allow_inf_nan=False)
    longest_bout_minutes: float | None = Field(default=None, gt=0, le=180, allow_inf_nan=False)

    @model_validator(mode='after')
    def paired_times(self):
        if (self.running_minutes is None) != (self.longest_bout_minutes is None):
            raise ValueError('Enter both running time and longest running stretch')
        if self.running_minutes is not None and self.longest_bout_minutes > self.running_minutes:
            raise ValueError('Longest stretch cannot exceed total running time')
        return self


@dataclass(frozen=True)
class StarterRecipe:
    running_s: int
    bout_s: int
    walk_s: int = 90

    @property
    def blocks(self):
        left = self.running_s
        result = []
        while left > 0:
            seconds = min(left, self.bout_s)
            result.append(('run', seconds))
            left -= seconds
            if left:
                result.append(('walk', self.walk_s))
        return tuple(result)

    @property
    def outing_s(self):
        return 600 + sum(seconds for _, seconds in self.blocks)


def rounded(seconds):
    """Downward half-minute commands; retain short demonstrated bouts."""
    return max(30, int(seconds // 30) * 30)


def advance(recipe):
    # At most 10% more running per successful pair, not 10% more distance
    # on every calendar week. Continuity is progressed separately. These are
    # transparent coaching choices, NOT validated injury-risk thresholds.
    total = min(1800, max(recipe.running_s + max(1, min(30, int(recipe.running_s*.1))), rounded(recipe.running_s * 1.10)))
    bout = total if recipe.bout_s == recipe.running_s else min(total,
        rounded(recipe.bout_s + max(30, recipe.bout_s*.25, recipe.running_s*.10)))
    return StarterRecipe(total, bout, recipe.walk_s)


def reduce(recipe):
    return StarterRecipe(rounded(recipe.running_s * .8),
                         min(rounded(recipe.bout_s * .8), rounded(recipe.running_s * .8)), recipe.walk_s)


def fingerprint(connection, row, config):
    source = connection.execute('''SELECT s.sha256 FROM source_files s JOIN activity_sources l
        ON l.source_file_id=s.id WHERE l.activity_id=? AND l.is_primary=1''', (row['id'],)).fetchone()
    # A watch-zone update changes the effort check, not what the athlete said
    # about an unchanged recording. Do not erase check-ins as zones recalibrate.
    return hashlib.sha256(json.dumps(dict(source=source[0] if source else None,
        metadata=dict(row)), sort_keys=True).encode()).hexdigest()


def sessions(connection, config, as_of):
    """Observed bout summaries with source/override/HR guards, not model eligibility.

    Reliable cadence separates running from walking, independent of pace. With
    ambiguous/missing cadence, ask for times instead of pretending a slow jog is
    a walk. Brief gaps are not bridged into a continuous running achievement.
    """
    from .movement import classify_movement
    from .processing import _load_points
    from .session_feedback import with_session_feedback
    journal = connection.execute("SELECT value_json FROM app_state WHERE key='beginner_training_v1'").fetchone()
    since = as_of-timedelta(days=90)
    if journal:
        began = json.loads(journal[0]).get('started_at')
        if began:
            since = min(since, datetime.fromisoformat(began))
    rows = connection.execute('''SELECT a.id,a.start_time_utc,a.start_time_local,a.created_at_utc,
        a.total_distance_m,m.detected_workout_type,o.workout_type,o.health_tag,o.illness,
        o.include_in_model,o.perceived_exertion FROM activities a
        JOIN activity_metrics m ON m.activity_id=a.id LEFT JOIN run_overrides o
        ON o.activity_id=COALESCE(NULLIF(a.activity_id,''),a.activity_uid)
        WHERE a.sport='Running' AND a.start_time_utc>=? AND a.start_time_utc<?
        ORDER BY a.start_time_utc,a.id''',
        (since.isoformat(), as_of.isoformat())).fetchall()
    result = []
    for row in with_session_feedback(connection, rows):
        available = connection.execute('''SELECT MAX(s.imported_at_utc) FROM source_files s
            JOIN activity_sources l ON l.source_file_id=s.id WHERE l.activity_id=? AND l.is_primary=1''', (row['id'],)).fetchone()[0]
        if not available or max(datetime.fromisoformat(available), datetime.fromisoformat(row['created_at_utc'])) > as_of:
            continue
        points = _load_points(connection, row['id'])
        end = max((p.timestamp_utc for p in points if p.timestamp_utc is not None), default=None)
        if end is None or end > as_of:
            continue
        intervals = classify_movement(points, config['moving_time']).intervals
        moving = known = easy = running = longest = stretch = unknown = walking = running_m = run_known = run_easy = 0.
        walks = []
        walk_stretch = 0.
        motion_stretch = longest_motion = 0.
        previous_end = None
        run_windows = []
        origin = datetime.fromisoformat(row['start_time_utc'])
        def add_window(interval):
            a = (interval.start.timestamp_utc-origin).total_seconds()
            b = (interval.end.timestamp_utc-origin).total_seconds()
            if run_windows and abs(run_windows[-1][1]-a)<1e-6:
                run_windows[-1]=(run_windows[-1][0],b)
            else:
                run_windows.append((a,b))
        for interval in intervals:
            if previous_end is not None and interval.start.timestamp_utc != previous_end:
                stretch = walk_stretch = motion_stretch = 0
            previous_end = interval.end.timestamp_utc
            seconds = interval.moving_time_s
            if (seconds <= 0 or interval.elapsed_s > config['moving_time']['maximum_interval_seconds']
                    or 'implausible_distance_speed' in interval.flags):
                stretch = walk_stretch = motion_stretch = 0
                continue
            motion_stretch += seconds
            longest_motion = max(longest_motion,motion_stretch)
            moving += seconds
            hr = interval.end.heart_rate_bpm
            if hr is not None and config['resting_hr'] < hr <= config['max_hr']:
                known += seconds
                easy += seconds * (hr <= config['zones']['z2'][1])
            cadence = interval.end.cadence_spm
            if cadence is not None and cadence >= config['activity_classification']['very_low_cadence_max_spm']:
                running += seconds
                running_m += interval.distance_m
                if hr is not None and config['resting_hr'] < hr <= config['max_hr']:
                    run_known += seconds
                    run_easy += seconds*(hr<=config['zones']['z2'][1])
                add_window(interval)
                stretch += seconds
                longest = max(longest, stretch)
                if walk_stretch:
                    walks.append(walk_stretch)
                walk_stretch = 0
            else:
                stretch = 0
                if cadence is not None and cadence <= config['activity_classification']['high_confidence_walk_cadence_max_spm']:
                    walking += seconds
                    walk_stretch += seconds
                else:
                    unknown += seconds
        token = fingerprint(connection, {k: row[k] for k in (
            'id','workout_type','detected_workout_type','health_tag','illness','include_in_model','perceived_exertion')}, config)
        saved = connection.execute('SELECT value_json FROM app_state WHERE key=?', ('beginner_feedback:'+str(row['id']),)).fetchone()
        feedback = json.loads(saved[0]) if saved else None
        if feedback and (feedback['fingerprint'] != token or datetime.fromisoformat(feedback['reported_at'])>as_of):
            feedback = None
        if feedback and feedback.get('longest_bout_minutes') is not None and feedback['longest_bout_minutes']*60>longest_motion+1:
            feedback = None
        if feedback and feedback.get('running_minutes') is not None:
            running = feedback['running_minutes']*60
            longest = feedback['longest_bout_minutes']*60
            unknown = 0
            if longest >= .95*running and running >= .95*moving:
                run_windows = []
                for interval in intervals:
                    if (interval.moving_time_s>0 and interval.elapsed_s<=config['moving_time']['maximum_interval_seconds']
                            and 'implausible_distance_speed' not in interval.flags):
                        add_window(interval)
                running_m = sum(i.distance_m for i in intervals if i.moving_time_s>0
                    and i.elapsed_s <= config['moving_time']['maximum_interval_seconds']
                    and 'implausible_distance_speed' not in i.flags)
        role = row['workout_type'] or row['detected_workout_type']
        healthy = (not row['illness'] and row['health_tag'] in (None, '', 'normal') and row['include_in_model'] != 0)
        eligible = (healthy and role in (None,'unknown','easy','recovery','long','run_walk') and moving >= 300
            and known/max(moving,1) >= config['model']['minimum_hr_coverage'])
        # Conversation confirmation cannot override a clearly hard recording.
        # Short HR excursions are allowed; warm-up lag is NOT extra run credit.
        hard = (known > 0 and easy/known < .7 or run_known > 0 and run_easy/run_known < .7
                or (row['perceived_exertion'] is not None and row['perceived_exertion'] > 4))
        recording_issue = ('This recording has too little usable heart-rate data. Check your sensor and record the next easy outing.'
            if known/max(moving,1) < config['model']['minimum_hr_coverage'] else
            'This recording is too short to establish a repeatable session. Try another short, easy outing when comfortable.'
            if moving < 300 else None)
        result.append(dict(id=row['id'], start=row['start_time_utc'], date=row['start_time_local'][:10],
            workout_type=role or 'unknown', health_tag=row['health_tag'] or 'normal',
            moving_s=moving, running_s=running, running_miles=running_m/1609.344, run_windows=run_windows,
            continuous_motion_s=longest_motion,
            bout_s=longest, unknown_s=unknown,
            walk_s=median(walks) if walks else 90, miles=(row['total_distance_m'] or 0)/1609.344,
            eligible=eligible, hard=hard, healthy=healthy, fingerprint=token, feedback=feedback,
            recording_issue=recording_issue,
            confirmed=bool(feedback and feedback['comfortable'] and not hard and eligible and unknown <= .1*max(moving,1)),
            needs_times=unknown > .1*max(moving,1) or running < 60))
    return result


def save_feedback(connection, config, as_of, activity_id, feedback):
    if not config.get('beginner_start') or connection.execute("SELECT 1 FROM app_state WHERE key='adaptive_runtime'").fetchone():
        raise ValueError('Starter feedback is available only during introductory training')
    observed = next((s for s in sessions(connection, config, as_of) if s['id'] == activity_id), None)
    if observed is None:
        raise ValueError('Choose a completed running recording')
    if feedback.running_minutes is not None and feedback.running_minutes*60 > observed['moving_s'] + 1:
        raise ValueError('Running time cannot exceed the recorded moving time')
    if feedback.longest_bout_minutes is not None and feedback.longest_bout_minutes*60 > observed['continuous_motion_s']+1:
        raise ValueError('Longest running stretch cannot cross a recorded stop or gap')
    payload = dict(feedback.model_dump(), fingerprint=observed['fingerprint'], reported_at=as_of.isoformat())
    connection.execute('''INSERT INTO app_state VALUES (?,?,?) ON CONFLICT(key) DO UPDATE
        SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc''',
        ('beginner_feedback:'+str(activity_id), json.dumps(payload), as_of.isoformat()))
    connection.commit()


def progress(observed, issues, as_of):
    """Deterministic replay of confirmed *issued* sessions, idempotent on refresh.

    The first comfortable recording seeds the dose. Later recordings use a
    command issued BEFORE that recording, including runs on a different day.
    An imported archive cannot manufacture prescription adherence.
    Missed sessions earn nothing, and more than a week between successes resets
    the pair. A two-week absence reduces the return prescription, not history.
    """
    # Re-establish today's starting session after a month without recordings;
    # historical work stays in the physical ledger, not a permanent capacity claim.
    cut = 0
    for i in range(1,len(observed)):
        if (datetime.fromisoformat(observed[i]['start'])-datetime.fromisoformat(observed[i-1]['start'])).days >= 28:
            cut=i
    observed=observed[cut:]
    if observed and (as_of-datetime.fromisoformat(observed[-1]['start'])).days>=28:
        observed=[]
    seed = next((s for s in observed if s['confirmed'] and s['running_s'] >= 60), None)
    if seed is None:
        return dict(recipe=None, successes=0, completed_ids=[], handoff=None)
    recipe = StarterRecipe(min(1800, rounded(seed['running_s'])), min(1800, rounded(seed['bout_s'])),
        max(30, min(180, rounded(seed['walk_s']))))
    successes = 0
    completed = [seed]
    last = datetime.fromisoformat(seed['start'])
    for session in observed:
        at = datetime.fromisoformat(session['start'])
        if at <= last:
            continue
        issued = issues.get(session['date'])
        if not issued or not issued.get('recipe'):
            issued = max((v for v in issues.values() if v.get('recipe')
                and datetime.fromisoformat(v['issued_at'])<at),
                key=lambda v:v['issued_at'],default=None)
        if not issued or not issued.get('recipe') or datetime.fromisoformat(issued['issued_at']) >= at:
            continue
        command = StarterRecipe(**issued['recipe'])
        if (at-last).total_seconds() > 7*86400:
            successes = 0
        matched = (session['confirmed'] and session['running_s'] >= .9*command.running_s
                   and session['bout_s'] >= .9*command.bout_s)
        if matched:
            successes += 1
            completed.append(session)
            recipe = command
            if successes >= 2:
                recipe = advance(recipe)
                successes = 0
        else:
            successes = 0
            recipe = reduce(command) if (session['feedback'] and not session['feedback']['comfortable']) or session['hard'] else command
        last = at
    if (as_of-last).total_seconds() >= 14*86400:
        recipe = reduce(recipe)
        successes = 0
    ready = [s for s in completed if s['bout_s'] >= 1800 and s['bout_s'] >= .95*s['running_s']
             and s['running_miles'] > 0 and not s['needs_times']]
    recent = [s for s in ready if datetime.fromisoformat(s['start']) > as_of-timedelta(days=7)]
    # Repeated continuous half-hours across at least a week, not a single good
    # test. Normal catalogue compilation/feasibility is a separate final gate.
    handoff = None
    if (len(recent) >= 3 and ready and (last-datetime.fromisoformat(ready[0]['start'])).days >= 7
            and (as_of-last).days <= 3 and all((datetime.fromisoformat(b['start'])-datetime.fromisoformat(a['start'])).days >= 1
                for a,b in zip(recent, recent[1:]))):
        chosen = recent[-3:]
        handoff = dict(weekly_miles=sum(s['running_miles'] for s in chosen), runs_per_week=3,
            ordinary_miles=median(s['running_miles'] for s in chosen), regular_long_miles=max(s['running_miles'] for s in chosen),
            source_ids=[s['id'] for s in chosen])
    return dict(recipe=asdict(recipe), successes=successes, completed_ids=[s['id'] for s in completed], handoff=handoff)
