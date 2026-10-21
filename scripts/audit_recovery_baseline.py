"""Small falsification grid, not physiological calibration or calendar validation.

Run: .venv/bin/python -m scripts.audit_recovery_baseline
All doses and labels below are predeclared engineering/coaching hypotheses for
an established runner familiar with a 90-minute easy run, NOT cold-start defaults.
Sources and limitations: docs/sourced_recovery_baseline_proposal.md.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product
import json
from time import perf_counter

from run_analysis.recovery_assessment import AssessmentPolicy, SessionEntry, sensitivity_panel
from run_analysis.workload_accounting import Quantity, Workload

NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)


def dose(minutes, cardio, work=0):
    q = lambda value: Quantity.point(value, 'predeclared scenario, not measured physiology')
    return Workload('projected', None, q(minutes), q(cardio), {}), q(work)


EASY = dose(30, 60)
LONG = dose(90, 180)
QUALITY = dose(45, 150, 15)


def history(past, end_gaps):
    workload, work = past
    return tuple(SessionEntry(str(i), NOW-timedelta(hours=gap, minutes=workload.running_minutes.high),
                              NOW-timedelta(hours=gap), workload, work)
                 for i, gap in enumerate(end_gaps))


@dataclass(frozen=True)
class Scenario:
    name: str
    past: tuple
    proposed: tuple
    required: str


SCENARIOS = (
    Scenario('familiar_long_when_fresh', (), LONG, 'model_supported'),
    Scenario('easy_after_familiar_long', history(LONG, (24,)), EASY, 'model_supported'),
    Scenario('easy_after_controlled_work', history(QUALITY, (24,)), EASY, 'model_supported'),
    Scenario('third_consecutive_short_easy', history(EASY, (24, 48)), EASY, 'model_supported'),
    Scenario('fourth_consecutive_large_dose', history(LONG, (24, 48, 72)), LONG, 'model_contradicted'),
    Scenario('fourth_consecutive_quality_dose', history(QUALITY, (24, 48, 72)), QUALITY, 'model_contradicted'),
)


def policies(structured):
    # Shared half-life first, to avoid granting the larger model extra decay knobs.
    for half, running, cardio, work in product(
        (12, 24, 36), range(90, 181, 10), range(180, 401, 20),
        range(16, 41, 4) if structured else (None,),
    ):
        yield AssessmentPolicy(f'h{half}/t{running}/c{cardio}/w{work}', running, cardio,
                               half, half, work, half if work is not None else None)


def select_policies(structured):
    """Return candidates matching only the original, explicitly declared anchors."""
    accepted, total = [], 0
    failures = {scenario.name: 0 for scenario in SCENARIOS}
    for policy in policies(structured):
        total += 1
        passed = True
        for scenario in SCENARIOS:
            workload, work = scenario.proposed
            result = sensitivity_panel(scenario.past, workload, NOW, (policy,),
                                       proposed_work_minutes=work)
            if result['status'] != scenario.required:
                failures[scenario.name] += 1
                passed = False
        if passed:
            accepted.append(policy)
    return accepted, total, failures


def audit():
    started = perf_counter()
    groups = {}
    for structured in (False, True):
        accepted, total, failures = select_policies(structured)
        groups['three_channels' if structured else 'two_channels'] = dict(
            tested=total, accepted=len(accepted), scenario_rejections=failures,
            accepted_by_half_life={str(h): sum(p.running_half_life_hours == h for p in accepted)
                                  for h in (12, 24, 36)},
            accepted_parameter_ranges={name: [min(values), max(values)]
                for name in ('running_minutes_envelope', 'cardio_points_envelope', 'work_minutes_envelope')
                if (values := [getattr(p, name) for p in accepted if getattr(p, name) is not None])},
        )
    return dict(groups=groups, elapsed_seconds=perf_counter()-started,
                scenarios=[dict(name=s.name, required=s.required) for s in SCENARIOS],
                timing_offsets_hours=[-6, 0, 6], planner_influence=False,
                conclusions=[
                    'Acceptance only means consistency with these six declared examples',
                    'A larger raw feasible count is not evidence that three channels are better',
                    'No selected defaults, personal calibration, cold-start capacity or stability claim',
                    'No magnitude or timetable for adapting to subjective feedback is selected',
                ])


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
