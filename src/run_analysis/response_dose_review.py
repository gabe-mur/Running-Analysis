"""Bounded next-dose review prototype; no live scheduler consumer.

RPE triggers review, not invented load points or recovery hours. A numerical
reduction requires an independent, robust demand assessment and a caller-named
adjustment boundary. Current health/readiness rules remain authoritative.
"""
from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from .prescription_distance import rounded_distance_target
from .recovery_assessment import sensitivity_panel
from .recovery_response import ResponseExpectation, assess_response
from .workload_accounting import Quantity, Workload


@dataclass(frozen=True)
class DoseOption:
    key: str
    session_key: str
    planned_for: datetime
    workout_kind: str
    target_miles: float
    workload: Workload
    work_minutes: Quantity | None

    def __post_init__(self):
        if not self.key or not self.session_key or not self.workout_kind:
            raise ValueError('Option, session and workout identifiers required')
        if self.planned_for.tzinfo is None or self.planned_for.utcoffset() is None:
            raise ValueError('Timezone-aware execution reference required')
        if not isfinite(self.target_miles) or self.target_miles <= 0:
            raise ValueError('Positive finite distance required')
        if abs(rounded_distance_target(self.target_miles)-self.target_miles) > 1e-8:
            raise ValueError('Commit to a single-decimal distance before reviewing')
        if self.workload.source != 'projected':
            raise ValueError('Next-dose options must be projections')
        distance = self.workload.distance_meters
        if distance is None or not distance.low-1e-8 <= self.target_miles*1609.344 <= distance.high+1e-8:
            raise ValueError('Projected distance must contain the declared target')


def _not_increased(candidate, issued):
    if issued is None or candidate is None:
        return candidate == issued
    return candidate.low <= issued.low+1e-8 and candidate.high <= issued.high+1e-8


def review_next_dose(*, issued: DoseOption, alternatives, history, policies,
                     expected_response: ResponseExpectation, report_activity_key: str,
                     minimum_target_miles: float, adjustment_source: str,
                     expected_history=None,
                     effort=None, feeling=None, readiness='unknown', feedback_conflict=False,
                     offsets_hours=(-6., 0., 6.)):
    """Return a proposal relative to the immutable issued dose, never mutate it.

    The caller supplies only already-eligible same-session alternatives and the
    adjustment floor; this module cannot waive progression/novelty guards. It
    cannot change dates/roles, add rest, raise load or redefine long-term capacity.
    Missing or surprising effort alone cannot mechanically shrink a workout.
    """
    history, policies, alternatives = tuple(history), tuple(policies), tuple(alternatives)
    if not report_activity_key or report_activity_key not in {entry.key for entry in history}:
        raise ValueError('The report must refer to a session in the completed ledger')
    if not any(entry.key == report_activity_key and entry.workload.source == 'observed' for entry in history):
        raise ValueError('A post-run report cannot be attributed to an uncompleted projection')
    latest = max((entry for entry in history if entry.workload.source == 'observed'), key=lambda entry: entry.end)
    if latest.key != report_activity_key or any(entry.key == issued.session_key for entry in history):
        raise ValueError('This response review expires once another run has been completed')
    if expected_history is not None:
        expected_history = tuple(expected_history)
        if {entry.key for entry in history} != {entry.key for entry in expected_history}:
            raise ValueError('Expected and actual ledgers must refer to the same sessions')
        prior = {entry.key: entry for entry in expected_history}
        if prior[report_activity_key].workload.source != 'projected':
            raise ValueError('The issued snapshot must retain the expected completion')
        if any(entry != prior[entry.key] for entry in history if entry.key != report_activity_key):
            raise ValueError('Unrelated historical changes need a separate adaptation review')
    if readiness not in {'unknown', 'normal', 'concern'}:
        raise ValueError('Unknown readiness state')
    if not adjustment_source.strip() or not isfinite(minimum_target_miles) or not 0 < minimum_target_miles <= issued.target_miles:
        raise ValueError('Explicit named adjustment floor required')
    if len({option.key for option in (issued, *alternatives)}) != len(alternatives)+1:
        raise ValueError('Duplicate options')
    for option in alternatives:
        if (option.session_key != issued.session_key or option.planned_for != issued.planned_for
                or option.workout_kind != issued.workout_kind):
            raise ValueError('Response review cannot change session identity, timing or workout kind')
        if not minimum_target_miles-1e-8 <= option.target_miles <= issued.target_miles:
            raise ValueError('Alternative exceeds the declared dose adjustment boundary')
        for name in ('distance_meters', 'running_minutes', 'cardio_points'):
            if not _not_increased(getattr(option.workload, name), getattr(issued.workload, name)):
                raise ValueError('A shorter alternative cannot increase another measured demand')
        if not _not_increased(option.work_minutes, issued.work_minutes):
            raise ValueError('Alternative increases or changes unknown work evidence')
    response = assess_response(expected_response, effort=effort, feeling=feeling,
                               feedback_conflict=feedback_conflict, readiness_concern=readiness == 'concern')
    def assess(option, ledger=history):
        return sensitivity_panel(ledger, option.workload, option.planned_for, policies,
                                 offsets_hours, proposed_work_minutes=option.work_minutes)
    baseline = assess(issued)
    before = assess(issued, expected_history) if expected_history is not None else None
    result = dict(action='retain_issued', selected_option_key=issued.key, target_miles=issued.target_miles,
        session_key=issued.session_key, report_activity_key=report_activity_key,
        original_target_miles=issued.target_miles, minimum_target_miles=minimum_target_miles,
        adjustment_source=adjustment_source, response=response, baseline_status=baseline['status'],
        issued_expected_status=before['status'] if before is not None else None,
        option_statuses={issued.key: baseline['status']}, planner_influence=False,
        automatic_progression=False, scope='this_issued_next_session_only',
        reason='No model-supported reason for a dose change')
    if readiness == 'concern':
        result.update(action='defer_to_existing_readiness_rules', selected_option_key=None,
                      target_miles=None, reason='Current readiness concerns override this limited review')
    elif feedback_conflict:
        result.update(action='review_conflicting_feedback', selected_option_key=None, target_miles=None,
                      reason='Do not turn conflicting reports into an automatic dose change')
    elif response['review_needed'] and readiness == 'unknown':
        result.update(action='request_readiness_check', selected_option_key=None, target_miles=None,
                      reason='Unexpected response needs current context, not an invented recovery multiplier')
    elif baseline['status'] == 'uncertain':
        result.update(action='review_model_uncertainty', selected_option_key=None, target_miles=None,
                      reason='Parameter or timing disagreement is not evidence for a forced reduction')
    elif baseline['status'] == 'model_contradicted':
        if before is None or before['status'] != 'model_supported':
            result.update(action='review_issue_baseline', selected_option_key=None, target_miles=None,
                          reason='Cannot attribute a pre-existing or undocumented demand conflict to this upload')
            return result
        # Largest supported supplied target is the smallest distance edit. The
        # stable key only breaks exact ties; no random selection or recursive cut.
        for option in sorted(alternatives, key=lambda o: (-o.target_miles, o.key)):
            checked = assess(option)
            result['option_statuses'][option.key] = checked['status']
            if checked['status'] == 'model_supported':
                result.update(action='propose_bounded_reduction', selected_option_key=option.key,
                              target_miles=option.target_miles,
                              reason='Independent demand conflict resolved by the smallest supplied distance edit')
                break
        else:
            result.update(action='review_beyond_adjustment_scope', selected_option_key=None, target_miles=None,
                          reason='No robustly supported option within the bound; do not delete or move the run here')
    elif response['channels']['effort']['status'] == 'less_demanding_than_expected':
        result['reason'] = 'Record easier response for progression review; one report does not authorize an increase'
    elif response['review_needed']:
        result['reason'] = 'Retain with response discrepancy visible: current readiness normal and dose model-supported'
    return result
