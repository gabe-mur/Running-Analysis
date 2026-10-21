"""Replay strength annotations on saved daily/upload run calendars only.

No running search, winner repricing, live API call, activity import, or physical
lifting completion. Archived inputs stay read-only; only the audit output writes.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import date, datetime, time, timedelta
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.strength_schedule import (
    TAXING_RUN_TYPES, add_strength_suggestions, strength_rotation_payload,
)
from run_analysis.web.schemas import (
    ConfidenceLevel, ReadinessFlag, RecommendationResponse, StrengthSessionType,
    WeeklyScheduleDay, WorkoutType,
)


ROLE_TYPES = {'easy': WorkoutType.EASY, 'long': WorkoutType.LONG,
              'quality': WorkoutType.INTERVALS, 'race': WorkoutType.RACE}


def saved_input(root, event, scenario):
    """Read exact saved roles/doses and as-of physical presence; never optimize."""
    snapshot_path = root / event['snapshot_file']
    snapshot = json.loads(snapshot_path.read_text())['state']['fields']
    now = datetime.fromisoformat(event['as_of'])
    zone = ZoneInfo(snapshot['timezone'])
    catalog = {d['key']: d for d in event.get('stage_runtime', {}).get('catalog', scenario['catalog'])}
    program = dict(event['program'])
    start, end = date.fromisoformat(event['decision']), date.fromisoformat(event['hard_end'])
    physical = []
    completed = {}
    for encoded in snapshot['history']['value']:
        fields = encoded['fields']
        at = datetime.fromisoformat(fields['start']['value'])
        available = datetime.fromisoformat(fields['available_at']['value'])
        if available > now:
            continue
        physical.append((at, fields['role']))
        completed.setdefault(at.astimezone(zone).date(), set()).add(ROLE_TYPES.get(fields['role'], WorkoutType.EASY))
    days = []
    race_date = event.get('race_preparation', {}).get('race_date')
    for i in range((end - start).days):
        day = start + timedelta(days=i)
        dose = catalog.get(program.get(day.isoformat()))
        result = None
        if dose:
            result = RecommendationResponse(
                generated_at=now, fitness_state_as_of=now,
                planned_for=datetime.combine(day, time(18), zone),
                workout_type=ROLE_TYPES[dose['role']], title=dose['role'],
                distance_target_miles=dose['miles'],
                distance_range_miles=(dose['miles'], dose['miles']),
                confidence=ConfidenceLevel.LOW, readiness=ReadinessFlag.UNKNOWN,
            )
        days.append(WeeklyScheduleDay(date=day, planned_at=result.planned_for if result else None,
            recommendation=result, day_role='race_event_day' if day.isoformat() == race_date
            else dose['role'] + '_run_day' if dose else 'rest_recovery_day', rationale='Archived role/dose.'))
    def since(role):
        starts = [at for at, kind in physical if kind == role]
        return (now - max(starts)).total_seconds() / 86400 if starts else None
    return days, completed, since('quality'), since('long'), snapshot_path


def signature(day):
    return day.strength_suggestion.session_type.value if day.strength_suggestion else None


def run_signature(day):
    r = day.recommendation
    return (day.day_role, r.workout_type.value if r else None, r.distance_target_miles if r else None)


def compare(previous, current, completed, window, event):
    prior = {d.date: d for d in previous[:window]}
    changes = []
    for day in current[:window]:
        old = prior.get(day.date)
        if old is None or signature(old) == signature(day):
            continue
        physical = completed.get(day.date, set())
        old_run, new_run = run_signature(old), run_signature(day)
        if physical & TAXING_RUN_TYPES:
            evidence = 'completed demanding run'
        elif old_run[:2] != new_run[:2] and not (physical and new_run[1] is None):
            evidence = 'run date/role changed'
        else:
            # A dose edit or removal of the completed easy run is not by itself
            # evidence that a previously assigned strength session must change.
            evidence = 'no new placement restriction'
        changes.append(dict(event=event['index'], reason=event['reason'], day=day.date.isoformat(),
                            before=signature(old), after=signature(day), evidence=evidence,
                            previous_run=old_run, current_run=new_run))
    return changes


def audit(root, config):
    started = perf_counter()
    root = root.resolve()
    reports = sorted(root.glob('resume-*.summary.json'))
    summary_path = reports[-1] if reports else root / 'lifecycle-summary.json'
    summary = json.loads(summary_path.read_text())
    scenario_path = root / 'scenario.json'
    scenario = json.loads(scenario_path.read_text())
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in (summary_path, scenario_path)}
    prior, previous, last = {}, None, None
    changes = {4: [], 7: []}
    times, guards, records = [], [], []
    committed, issued_first = {}, {}
    preserved = 0
    for event in summary['events']:
        days, completed, quality_since, long_since, snapshot = saved_input(root, event, scenario)
        hashes[str(snapshot)] = sha256(snapshot.read_bytes()).hexdigest()
        original = [d.model_dump(exclude={'strength_suggestion'}) for d in days]
        at = perf_counter()
        result = add_strength_suggestions(days, config, prior_plan=prior,
            completed_run_types=completed, days_since_quality_run=quality_since, days_since_long_run=long_since)
        times.append(perf_counter() - at)
        assert [d.model_dump(exclude={'strength_suggestion'}) for d in result] == original
        # Identical saved inputs plus retained guidance must be repeatable.
        payload = json.loads(strength_rotation_payload(result, prior, config))
        memory = {date.fromisoformat(d): StrengthSessionType(kind) for d, kind in payload['suggestions'].items()}
        repeat = add_strength_suggestions(days, config, prior_plan=memory,
            completed_run_types=completed, days_since_quality_run=quality_since, days_since_long_run=long_since)
        assert [signature(d) for d in repeat] == [signature(d) for d in result]
        if previous:
            for window in changes:
                changes[window].extend(compare(previous, result, completed, window, event))
        for i, day in enumerate(result):
            kind = signature(day)
            if kind == 'legs' and (day.recommendation or completed.get(day.date)
                    or i + 1 >= len(result) or result[i + 1].day_role == 'race_event_day'
                    or result[i + 1].recommendation and result[i + 1].recommendation.workout_type in TAXING_RUN_TYPES):
                guards.append(dict(event=event['index'], day=str(day.date), violation='leg placement'))
            if kind in {'push', 'pull', 'full_upper'} and (day.day_role == 'race_event_day'
                    or day.recommendation and day.recommendation.workout_type not in {WorkoutType.EASY, WorkoutType.RECOVERY}
                    or completed.get(day.date, set()) & TAXING_RUN_TYPES):
                guards.append(dict(event=event['index'], day=str(day.date), violation='upper placement'))
        if previous:
            for day in result[:7]:
                old = next((d for d in previous[:7] if d.date == day.date), None)
                if old and completed.get(day.date) and old.recommendation and old.recommendation.workout_type == WorkoutType.EASY and signature(old):
                    preserved += int(signature(old) == signature(day))
        for day in result[:7]:
            issued_first.setdefault(str(day.date), signature(day))
        # The final decision on a local date is the lifting guidance that
        # survives that day's upload. This does NOT record a completed lift.
        committed[str(result[0].date)] = signature(result[0])
        records.append(dict(event=event['index'], reason=event['reason'], as_of=event['as_of'],
                            visible=[dict(day=str(d.date), session=signature(d)) for d in result[:7]]))
        prior, previous, last = memory, result, result
    for path, digest in hashes.items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest
    interval_start = min(committed)
    interval_end = (date.fromisoformat(interval_start) + timedelta(days=summary['completed_days'])).isoformat()
    executed = {d: kind for d, kind in committed.items() if interval_start <= d < interval_end}
    cadence = {}
    for lane in ('push', 'pull', 'legs'):
        dates = [date.fromisoformat(d) for d, kind in sorted(executed.items())
                 if kind == lane or lane != 'legs' and kind == 'full_upper']
        cadence[lane] = dict(sessions=len(dates), dates=[str(d) for d in dates],
            maximum_internal_gap_days=max(((b-a).days for a,b in zip(dates,dates[1:])), default=None))
    return dict(root=str(root), scope='Strength-only replay of saved daily/upload run calendars; no running optimization or physiological lifting validation',
        completed_days=summary['completed_days'], saved_decisions=len(records),
        four_day_changes=changes[4], seven_day_changes=changes[7],
        unexplained_seven_day_changes=[r for r in changes[7] if r['evidence']=='no new placement restriction'],
        guard_violations=guards, issued_session_counts=dict(Counter(kind for kind in executed.values() if kind)),
        cadence=cadence, existing_easy_pairings_preserved_on_upload=preserved,
        maximum_strength_annotation_seconds=max(times), total_seconds=perf_counter()-started,
        input_hashes=hashes, timeline=records,
        first_visible_assignments=issued_first, final_visible=records[-1]['visible'] if last else [])


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [audit(root, load_config()) for root in args.roots]
    report = dict(running_search_calls=0, live_server_called=False,
                  lifting_completion_inferred=False, cases=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    for case in results:
        print(json.dumps({k:v for k,v in case.items() if k not in {'timeline','input_hashes','first_visible_assignments'}}))
