from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import sqlite3
import pytest

from run_analysis.strength_schedule import (
    add_strength_suggestions, load_strength_rotation, strength_rotation_payload,
)
from run_analysis.web.schemas import (
    ConfidenceLevel,
    ReadinessFlag,
    RecommendationResponse,
    StrengthSessionType,
    WeeklyScheduleDay,
    WorkoutType,
)


CONFIG = {
    "coaching": {
        "strength_training": {
            "enabled": True,
            "leg_frequency_days": 8,
        }
    }
}
START = date(2026, 9, 17)


def _day(offset: int, workout_type: WorkoutType | None = None) -> WeeklyScheduleDay:
    moment = datetime(2026, 9, 17, 12, tzinfo=timezone.utc) + timedelta(days=offset)
    recommendation = None
    if workout_type is not None:
        recommendation = RecommendationResponse(
            generated_at=moment,
            fitness_state_as_of=moment,
            planned_for=moment,
            workout_type=workout_type,
            title=workout_type.value,
            distance_range_miles=(3.0, 4.0) if workout_type != WorkoutType.REST else None,
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.READY,
        )
    return WeeklyScheduleDay(
        date=START + timedelta(days=offset),
        planned_at=moment if recommendation is not None else None,
        recommendation=recommendation,
        day_role="run" if recommendation is not None else "rest_day",
        rationale="Fixture.",
    )


def test_strength_layer_preserves_run_plan_and_covers_each_eight_day_span() -> None:
    run_types = {
        0: WorkoutType.EASY,
        2: WorkoutType.LONG,
        4: WorkoutType.EASY,
        6: WorkoutType.INTERVALS,
        8: WorkoutType.EASY,
        10: WorkoutType.EASY,
        12: WorkoutType.LONG,
        14: WorkoutType.EASY,
        16: WorkoutType.TEMPO_THRESHOLD,
        18: WorkoutType.EASY,
        20: WorkoutType.EASY,
    }
    original = [_day(index, run_types.get(index)) for index in range(21)]
    annotated = add_strength_suggestions(original, CONFIG)

    assert [day.recommendation for day in annotated] == [
        day.recommendation for day in original
    ]
    leg_days = [
        index
        for index, day in enumerate(annotated)
        if day.strength_suggestion
        and day.strength_suggestion.session_type == StrengthSessionType.LEGS
    ]
    assert leg_days[0] < 8
    assert 20 - leg_days[-1] < 8
    assert all(right - left <= 8 for left, right in zip(leg_days, leg_days[1:]))
    assert all(original[index].recommendation is None for index in leg_days)


def test_leg_day_prefers_easy_neighbors_over_long_or_quality_neighbors() -> None:
    days = [
        _day(0, WorkoutType.LONG),
        _day(1),
        _day(2, WorkoutType.EASY),
        _day(3),
        _day(4, WorkoutType.EASY),
        _day(5),
        _day(6, WorkoutType.INTERVALS),
        _day(7),
    ]
    annotated = add_strength_suggestions(days, CONFIG)
    leg_days = [
        index
        for index, day in enumerate(annotated)
        if day.strength_suggestion
        and day.strength_suggestion.session_type == StrengthSessionType.LEGS
    ]

    assert leg_days == [3]


def test_declared_race_day_is_not_a_lifting_opportunity_and_protects_neighbors():
    from run_analysis.strength_schedule import _eligible_off_day,_leg_candidate_cost
    days=[_day(i) for i in range(7)]
    days[3]=days[3].model_copy(update={'day_role':'race_event_day'})
    assert not _eligible_off_day(days[3])
    assert _leg_candidate_cost(2,days,days_since_quality_run=None,days_since_long_run=None)>=1000
    assert _leg_candidate_cost(4,days,days_since_quality_run=None,days_since_long_run=None)>=1000
    result=add_strength_suggestions(days,CONFIG)
    assert result[3].strength_suggestion is None


def test_leg_day_never_uses_the_day_before_a_long_or_quality_run() -> None:
    days = [
        _day(0, WorkoutType.EASY),
        _day(1),
        _day(2, WorkoutType.LONG),
        _day(3, WorkoutType.EASY),
        _day(4),
        _day(5, WorkoutType.INTERVALS),
        _day(6, WorkoutType.EASY),
        _day(7),
        _day(8, WorkoutType.EASY),
    ]

    annotated = add_strength_suggestions(days, CONFIG)
    leg_days = [
        index
        for index, day in enumerate(annotated)
        if day.strength_suggestion
        and day.strength_suggestion.session_type == StrengthSessionType.LEGS
    ]

    assert leg_days == [7]


def test_isolated_off_days_use_push_pull_instead_of_automatic_full_upper() -> None:
    run_types = {
        0: WorkoutType.LONG,
        2: WorkoutType.EASY,
        4: WorkoutType.EASY,
        7: WorkoutType.INTERVALS,
    }
    days = [_day(index, run_types.get(index)) for index in range(8)]
    annotated = add_strength_suggestions(days, CONFIG)
    types = {
        index: day.strength_suggestion.session_type
        for index, day in enumerate(annotated)
        if day.strength_suggestion
    }

    assert types[1] == StrengthSessionType.PUSH
    assert types[3] == StrengthSessionType.LEGS
    assert types[5] == StrengthSessionType.PULL
    assert StrengthSessionType.FULL_UPPER not in types.values()
    assert annotated[1].strength_suggestion.muscle_groups == [
        "upper pecs",
        "lower pecs",
        "side delts",
        "triceps",
        "core/trunk",
    ]
    assert annotated[5].strength_suggestion.muscle_groups == [
        "lats",
        "upper back",
        "rear delts",
        "biceps",
    ]


def test_guardrail_rest_stays_empty_and_easy_run_can_close_upper_cadence() -> None:
    days = [
        _day(0, WorkoutType.REST),
        _day(1, WorkoutType.EASY),
        _day(2),
        _day(3, WorkoutType.EASY),
    ]
    annotated = add_strength_suggestions(days, CONFIG)

    assert annotated[0].strength_suggestion is None
    assert (
        annotated[1].strength_suggestion.session_type
        == StrengthSessionType.PUSH
    )
    assert "share an easy-run day" in annotated[1].strength_suggestion.rationale
    assert annotated[2].strength_suggestion is not None
    assert annotated[3].strength_suggestion.session_type == StrengthSessionType.PULL
    assert annotated[2].strength_suggestion.muscle_groups == [
        "quads",
        "hamstrings",
        "glutes",
        "calves",
    ]
    assert all(
        "lb" not in group and "dumbbell" not in group
        for group in annotated[2].strength_suggestion.muscle_groups
    )


def test_dense_run_plan_uses_easy_days_to_keep_push_and_pull_inside_eight_days() -> None:
    days = [
        _day(index, None if index in {7, 15} else WorkoutType.EASY)
        for index in range(21)
    ]
    annotated = add_strength_suggestions(days, CONFIG)

    def exposures(session_type: StrengthSessionType) -> list[int]:
        return [
            index
            for index, day in enumerate(annotated)
            if day.strength_suggestion
            and day.strength_suggestion.session_type
            in {session_type, StrengthSessionType.FULL_UPPER}
        ]

    for session_type in (StrengthSessionType.PUSH, StrengthSessionType.PULL):
        values = exposures(session_type)
        assert values  # Coverage comes from dated gaps, not ceil(horizon/8).
        assert values[0] < 8
        assert 20 - values[-1] < 8
        assert all(right - left <= 8 for left, right in zip(values, values[1:]))
    assert any(
        day.recommendation is not None and day.strength_suggestion is not None
        for day in annotated
    )
    assert all(
        day.recommendation is None
        or day.recommendation.workout_type == WorkoutType.EASY
        or day.strength_suggestion is None
        for day in annotated
    )


def test_full_upper_is_only_used_when_there_is_not_room_for_separate_sessions():
    days = [_day(0, WorkoutType.REST), _day(1, WorkoutType.EASY), _day(2, WorkoutType.INTERVALS)]
    result = add_strength_suggestions(days, CONFIG)
    assert result[1].strength_suggestion.session_type == StrengthSessionType.FULL_UPPER
    assert result[0].strength_suggestion is None and result[2].strength_suggestion is None


def test_completed_run_is_not_a_fresh_lifting_opportunity_after_upload():
    days = [_day(i) for i in range(7)]
    result = add_strength_suggestions(days, CONFIG,
        completed_run_types={days[0].date: {WorkoutType.LONG}},
        prior_plan={days[0].date: StrengthSessionType.LEGS})
    assert result[0].strength_suggestion is None
    assert [day.recommendation for day in result] == [day.recommendation for day in days]


def test_compliant_easy_upload_does_not_cancel_its_existing_upper_session():
    days = [_day(i, WorkoutType.EASY) for i in range(7)]
    prior = {days[0].date: StrengthSessionType.PUSH}
    days[0] = days[0].model_copy(update={'recommendation': None, 'planned_at': None, 'day_role': 'rest_day'})
    result = add_strength_suggestions(days, CONFIG, prior_plan=prior,
        completed_run_types={days[0].date: {WorkoutType.EASY}})
    assert result[0].strength_suggestion.session_type == StrengthSessionType.PUSH
    assert 'share an easy-run day' in result[0].strength_suggestion.rationale


@pytest.mark.parametrize('run_types', [
    {1: WorkoutType.INTERVALS, 2: WorkoutType.EASY, 4: WorkoutType.LONG, 6: WorkoutType.EASY},
    {i: WorkoutType.EASY for i in range(7)},
])
def test_strength_stays_stable_through_daily_rollover_and_same_day_upload(run_types):
    prior, previous, completed = {}, None, {}
    paired_uploads = 0
    def kinds(days):
        return {d.date: d.strength_suggestion.session_type if d.strength_suggestion else None for d in days[:7]}
    def memory(days, prior):
        payload = json.loads(strength_rotation_payload(days, prior, CONFIG))
        return {date.fromisoformat(d): StrengthSessionType(kind) for d, kind in payload['suggestions'].items()}
    for decision in range(21):
        days = [_day(i, run_types.get(i % 7)) for i in range(decision, decision + 21)]
        morning = add_strength_suggestions(days, CONFIG, prior_plan=prior, completed_run_types=completed)
        if previous:
            before = kinds(previous)
            assert all(kind == before[d] for d, kind in kinds(morning).items() if d in before)
        prior = memory(morning, prior)
        current = morning[0]
        if current.recommendation:
            completed[current.date] = {current.recommendation.workout_type}
            paired_uploads += int(current.recommendation.workout_type == WorkoutType.EASY and current.strength_suggestion is not None)
            after_upload = [days[0].model_copy(update={'recommendation': None, 'planned_at': None, 'day_role': 'rest_day'}), *days[1:]]
            result = add_strength_suggestions(after_upload, CONFIG, prior_plan=prior, completed_run_types=completed)
            assert kinds(result) == kinds(morning)
        else:
            result = morning
        previous, prior = result, memory(result, prior)
    if len(run_types) == 7:
        assert paired_uploads >= 4  # Exercise real pairs, not a vacuous no-lift case.


def test_new_key_run_invalidates_previously_suggested_leg_day():
    days = [_day(i, WorkoutType.LONG if i == 4 else None) for i in range(7)]
    result = add_strength_suggestions(days, CONFIG, prior_plan={days[3].date: StrengthSessionType.LEGS})
    assert result[3].strength_suggestion is None or result[3].strength_suggestion.session_type != StrengthSessionType.LEGS


def test_unknown_following_day_is_not_leg_clearance():
    days = [_day(0, WorkoutType.EASY), _day(1)]
    result = add_strength_suggestions(days, CONFIG)
    assert all(not day.strength_suggestion or day.strength_suggestion.session_type != StrengthSessionType.LEGS for day in result)


def test_disabled_strength_removes_stale_annotations():
    days = add_strength_suggestions([_day(i) for i in range(7)], CONFIG)
    assert any(day.strength_suggestion for day in days)
    disabled = add_strength_suggestions(days, {'coaching': {'strength_training': {'enabled': False}}})
    assert all(day.strength_suggestion is None for day in disabled)


@pytest.mark.parametrize('run_types', [
    {1: WorkoutType.INTERVALS, 4: WorkoutType.LONG, 6: WorkoutType.EASY},
    {1: WorkoutType.INTERVALS, 2: WorkoutType.EASY, 4: WorkoutType.LONG, 6: WorkoutType.EASY},
    {1: WorkoutType.INTERVALS, 2: WorkoutType.EASY, 3: WorkoutType.EASY, 4: WorkoutType.LONG, 6: WorkoutType.EASY},
    {i: WorkoutType.EASY for i in range(7)},
])
def test_90_daily_strength_rollovers_preserve_visible_rotation_and_running(run_types):
    # Isolates the lifting layer. No new running optimization or simulated HR.
    prior = {}
    previous = None
    delivered = []
    for decision in range(90):
        days = [_day(i, run_types.get(i % 7)) for i in range(decision, decision + 21)]
        original = [day.model_dump(exclude={'strength_suggestion'}) for day in days]
        result = add_strength_suggestions(days, CONFIG, prior_plan=prior)
        assert [day.model_dump(exclude={'strength_suggestion'}) for day in result] == original
        if previous:
            before = {day.date: day.strength_suggestion for day in previous[:7]}
            for day in result[:7]:
                if day.date in before:
                    assert day.strength_suggestion == before[day.date]
        for i, day in enumerate(result):
            if day.strength_suggestion and day.strength_suggestion.session_type == StrengthSessionType.LEGS:
                assert day.recommendation is None and i + 1 < len(result)
                assert result[i + 1].recommendation is None or result[i + 1].recommendation.workout_type not in {WorkoutType.LONG, WorkoutType.INTERVALS}
        delivered.append(result[0])
        payload = json.loads(strength_rotation_payload(result, prior, CONFIG))
        assert payload['provenance'] == 'issued_suggestions_not_completed_lifts'
        prior = {date.fromisoformat(d): StrengthSessionType(kind) for d, kind in payload['suggestions'].items()}
        previous = result
    for lane in (StrengthSessionType.PUSH, StrengthSessionType.PULL):
        dates = [day.date for day in delivered if day.strength_suggestion and day.strength_suggestion.session_type in {lane, StrengthSessionType.FULL_UPPER}]
        assert dates and (dates[0] - START).days < 7
        assert all((right - left).days <= 8 for left, right in zip(dates, dates[1:]))
    combined = [day for day in delivered if day.strength_suggestion
                and day.strength_suggestion.session_type == StrengthSessionType.FULL_UPPER]
    if len(run_types) == 5:
        # Two no-run days cannot fit leg + separate push + pull. Prefer the
        # combined off-day fallback over adding lifting to an easy run.
        assert all(day.recommendation is None for day in combined)
    else:
        assert not combined


def test_rotation_memory_roundtrips_without_recording_lifting_completions():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE app_state(key TEXT PRIMARY KEY, value_json TEXT, updated_at_utc TEXT)')
    days = add_strength_suggestions([_day(i) for i in range(7)], CONFIG)
    db.execute('INSERT INTO app_state VALUES (?,?,?)', ('strength_rotation', strength_rotation_payload(days, {}, CONFIG), 'fixture'))
    prior = load_strength_rotation(db)
    assert prior == {day.date: day.strength_suggestion.session_type for day in days if day.strength_suggestion}
    assert add_strength_suggestions(days, CONFIG, prior_plan=prior) == days
    assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [('app_state',)]
    db.close()


def test_new_available_days_replace_combined_fallback_with_split_sessions():
    # Three off days fit legs AND separate push/pull. Easy-run days alone do
    # not justify moving an otherwise feasible combined off-day fallback.
    days = [_day(0), _day(1), _day(2), _day(3, WorkoutType.EASY), _day(4, WorkoutType.EASY)]
    result = add_strength_suggestions(days, CONFIG, prior_plan={days[0].date: StrengthSessionType.FULL_UPPER})
    assert not any(day.strength_suggestion and day.strength_suggestion.session_type == StrengthSessionType.FULL_UPPER for day in result)
    assert {day.strength_suggestion.session_type for day in result if day.strength_suggestion} >= {StrengthSessionType.PUSH, StrengthSessionType.PULL, StrengthSessionType.LEGS}


def test_combined_off_day_precedes_unnecessary_easy_day_pairing():
    days = [_day(0), _day(1), _day(2, WorkoutType.EASY), _day(3, WorkoutType.EASY)]
    result = add_strength_suggestions(days, CONFIG)
    assert {day.strength_suggestion.session_type for day in result if day.strength_suggestion} == {
        StrengthSessionType.LEGS, StrengthSessionType.FULL_UPPER}
    assert all(day.recommendation is None for day in result if day.strength_suggestion)


def test_run_move_rehomes_old_pairs_to_available_off_days_without_moving_runs():
    roles = {1: WorkoutType.EASY, 3: WorkoutType.LONG, 5: WorkoutType.EASY,
             6: WorkoutType.INTERVALS, 7: WorkoutType.EASY, 9: WorkoutType.LONG,
             11: WorkoutType.EASY, 13: WorkoutType.INTERVALS, 14: WorkoutType.EASY,
             16: WorkoutType.LONG, 18: WorkoutType.EASY, 20: WorkoutType.INTERVALS}
    days = [_day(i, roles.get(i)) for i in range(21)]
    prior = {START + timedelta(days=i): kind for i, kind in (
        (-6, StrengthSessionType.PULL), (-4, StrengthSessionType.LEGS),
        (-2, StrengthSessionType.PUSH), (1, StrengthSessionType.PULL),
        (4, StrengthSessionType.LEGS), (5, StrengthSessionType.PUSH),
        (8, StrengthSessionType.PULL), (10, StrengthSessionType.LEGS),
        (12, StrengthSessionType.PUSH), (15, StrengthSessionType.PULL),
        (17, StrengthSessionType.LEGS), (19, StrengthSessionType.PUSH))}
    result = add_strength_suggestions(days, CONFIG, prior_plan=prior,
        days_since_quality_run=1., days_since_long_run=5.)
    assert [d.model_dump(exclude={'strength_suggestion'}) for d in result] == [
        d.model_dump(exclude={'strength_suggestion'}) for d in days]
    assert result[0].strength_suggestion.session_type == StrengthSessionType.PULL
    assert result[2].strength_suggestion.session_type == StrengthSessionType.PUSH
    assert result[4].strength_suggestion.session_type == StrengthSessionType.LEGS
    assert result[1].strength_suggestion is None and result[5].strength_suggestion is None
    assert all(d.recommendation is None for d in result if d.strength_suggestion)
    memory = json.loads(strength_rotation_payload(result, prior, CONFIG))
    new_prior = {date.fromisoformat(d): StrengthSessionType(kind)
                 for d, kind in memory['suggestions'].items()}
    assert add_strength_suggestions(days, CONFIG, prior_plan=new_prior,
        days_since_quality_run=1., days_since_long_run=5.) == result


def test_cadence_change_discards_future_rotation_but_not_past_issued_guidance():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE app_state(key TEXT PRIMARY KEY, value_json TEXT, updated_at_utc TEXT)')
    days = add_strength_suggestions([_day(i) for i in range(21)], CONFIG)
    db.execute('INSERT INTO app_state VALUES (?,?,?)', ('strength_rotation', strength_rotation_payload(days, {}, CONFIG), 'fixture'))
    changed = {'coaching': {'strength_training': {'enabled': True, 'leg_frequency_days': 14, 'push_pull_frequency_days': 14}}}
    start = START + timedelta(days=7)
    prior = load_strength_rotation(db, changed, start_date=start)
    assert prior and all(d < start for d in prior)
    assert any(d >= start for d in load_strength_rotation(db, CONFIG, start_date=start))
    db.close()
