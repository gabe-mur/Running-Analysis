from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.adherence_projection import (
    DeterministicAdherenceProfile, DeterministicAdherenceScenario,
    simulate_adherence,
)
from scripts.adaptation_audit import adaptation_transitions
from run_analysis.recovery import athlete_relative_session_load
from run_analysis.web.schemas import LoadWindow, ZoneBreakdown
from test_adherence_projection import _repeating_range_schedule
from test_projection_gate import _replan, _session, OPENING
from test_recommendation import CONFIG, _state, _difficulty


@pytest.mark.parametrize('scenario,fraction,expected_miles,harder', [
    ('above_range_easy', 0., 4.1, False),
    ('above_range_easy', .5, 6., False),
    ('harder_same_distance', .5, 3.5, True),
    ('above_range_or_harder', .5, 6., True),
])
def test_distance_and_effort_are_independent_single_deviations(
    monkeypatch, scenario, fraction, expected_miles, harder,
):
    template = _state(as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
                      typical_easy_run_miles=4.)
    monkeypatch.setattr('run_analysis.adherence_projection.build_weekly_schedule',
                        _repeating_range_schedule())
    completed = []
    weeks = simulate_adherence(template, [], CONFIG, template.as_of,
        weeks=1, simulation_days=4,
        execution_observer=lambda run, _: completed.append(run),
        deterministic_profile=DeterministicAdherenceProfile(
            scenario=DeterministicAdherenceScenario(scenario),
            above_range_easy_fraction=fraction))
    assert len(completed) >= 2
    first, later = completed[:2]
    assert first.distance_miles == pytest.approx(expected_miles)
    assert (first.difficulty.zone_load is not None) == harder
    assert ('more intense' in first.adherence_note) == harder
    assert later.distance_miles == 3.5
    assert later.difficulty.zone_load is None
    assert later.adherence_note == 'within prescribed margin'
    assert len(weeks[0].adherence_event_records) == 1
    upload = next(x for x in weeks[0].replan_snapshots if x.trigger == 'post_upload')
    assert upload.material_evidence_reasons
    # These are scenario-generation checks, not recovery calibration. In this
    # empty-history fixture the references themselves change after the run;
    # the prescription load band can also absorb this effort perturbation.
    assert upload.recovery_surprise_units is not None
    assert (first.difficulty.zone_breakdown.hard_minutes > 0) == harder


def test_adaptation_audit_keeps_exempt_edits_visible_and_no_evidence_separate():
    before = _replan(_session(2), _session(3))
    upload = _replan(_session(2, midpoint_miles=3.9), _session(4),
        generated_at=OPENING + timedelta(hours=1), trigger='post_upload',
        material_evidence_reasons=('deviation: above prescribed range',),
        recovery_surprise_units=.01)
    refresh = _replan(_session(2, midpoint_miles=3.8), _session(3),
        generated_at=OPENING + timedelta(days=1))
    rows = adaptation_transitions([before, upload, refresh])
    assert rows[0]['interpretation'] == 'adaptation_requires_review'
    assert len(rows[0]['windows']['4']['date_changes']) == 2
    assert rows[0]['windows']['4']['distance_changes'][0]['movement_miles'] == pytest.approx(.1)
    assert rows[1]['interpretation'] == 'change_without_new_material_evidence'
    assert rows[1]['material_evidence_reasons'] == []


def test_adaptation_audit_does_not_invent_changes_from_evidence():
    before = _replan(_session(2))
    after = _replan(_session(2), material_evidence_reasons=('deviation',))
    assert adaptation_transitions([before, after])[0]['interpretation'] == 'unchanged'


def test_extra_easy_distance_never_lowers_load_against_fixed_reference():
    typical = LoadWindow(days=28, activity_count=10, distance_miles=40., moving_minutes=440.)
    loads = []
    for miles in (3.3, 3.5, 3.6, 4., 5.5):
        session = _difficulty(miles=miles).model_copy(update={
            'moving_minutes': miles * 11, 'elapsed_minutes': miles * 11,
            'zone_load': None, 'zone_breakdown': ZoneBreakdown()})
        load, _ = athlete_relative_session_load(session, typical,
            prescribed_distance_range_miles=(3., 3.5))
        loads.append(load)
    assert loads == sorted(loads)
    assert loads[0] == loads[1]  # entire prescribed range remains compliant
    assert loads[-1] > loads[0]  # material extra work cannot vanish in the band
