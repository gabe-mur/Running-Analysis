"""Offline comparisons replacing horizon-aggregate ordinary-aerobic support.

Preferred trial: bounded carry credit using the existing one easy-equivalent
per seven days and one-session allowance. The overlapping-window comparator is
retained as evidence: repeatedly granting its allowance dilutes the floor.
Neither is a physiological threshold or mandatory frequency.
Long-run aerobic mileage remains in the separate total-volume accounting. This
term retains the existing narrower meaning of ordinary easy support; it does
not purport to measure the complete training benefit of every workout type.
"""

from collections import defaultdict, deque
from datetime import timedelta
from math import isfinite


def per_run_shortfall_components(days, states, request):
    """Exact legacy charges, separate from distribution and other shape terms.

    This is diagnostic accounting, not an enabled replacement objective. Match
    the detailed scorer's date-local reference and medium-long exclusion.
    """
    from run_analysis import weekly_schedule as planner

    if len(days) != len(states):
        raise ValueError('Require aligned days and states')
    missing = sum(planner._aerobic_shortfall_fraction(planner._midpoint(day.recommendation),
                  planner.typical_easy_distance(states[i])[0])
        for i, day in enumerate(days) if day.recommendation
        and day.recommendation.workout_type == planner.WorkoutType.EASY
        and day.day_role != 'medium_long_run')
    support = planner.PROGRAM_FIT_UNIT * planner.AEROBIC_SUPPORT_FIT_FRACTION * missing
    fragmentation = (planner.PROGRAM_FIT_UNIT * missing
                     if request.health_status == planner.CurrentHealthStatus.NORMAL else 0.)
    return {'missing_equivalents': missing, 'support_fee': support,
            'fragmentation_fee': fragmentation, 'total_fee': support + fragmentation}


def bounded_support(dates, sessions, *, ordinary_miles):
    """Carry at most one equivalent; consume the existing one-per-seven rate.

    Unlike overlapping windows, do not regrant the one-session allowance in
    every window. Credit can cover at most seven days. No negative balance is
    carried forward: missed exposure is scored when it happens, never debt to
    be repaid by a later run. This is an accounting policy, not modeled fitness.
    """
    dates = tuple(dates)
    if not dates or any(b != a + timedelta(days=1) for a, b in zip(dates, dates[1:])):
        raise ValueError('Require nonempty consecutive scoring dates')
    if not isfinite(ordinary_miles) or ordinary_miles <= 0:
        raise ValueError('Require a positive finite observed ordinary reference')
    by_date = defaultdict(float)
    for day, miles in sessions:
        if not isfinite(miles) or miles < 0:
            raise ValueError('Support distance must be finite and nonnegative')
        if day > dates[-1]:
            raise ValueError('Support session lies beyond the scored sequence')
        by_date[day] += miles
    # Reconstruct credit from all supplied history, not a finite lookback:
    # ongoing small contributions can preserve a balance established earlier.
    # Advance gaps directly. Internally use days of credit so ordinary whole
    # days subtract exactly; only the reported units are session equivalents.
    balance, previous = 0., None
    for day in sorted(d for d in by_date if d < dates[0]):
        if previous is not None:
            balance = max(0., balance - (day - previous).days)
        balance = min(7., balance + min(7., 7 * by_date[day] / ordinary_miles))
        previous = day
    if previous is not None:
        balance = max(0., balance - (dates[0] - previous).days)
    cursor, trace = dates[0], []
    while cursor <= dates[-1]:
        added = min(7., 7 * by_date.get(cursor, 0.) / ordinary_miles)
        available = min(7., balance + added)
        missing = max(0., 1 - available)
        balance = max(0., available - 1)
        trace.append({'date': cursor.isoformat(), 'added_equivalents': added / 7,
                      'remaining_equivalents': balance / 7, 'missing_equivalents': missing / 7})
        cursor += timedelta(days=1)
    return {'violation': sum(row['missing_equivalents'] for row in trace),
            'maximum_credit_equivalents': 1., 'equivalents_per_day': 1 / 7,
            'ordinary_reference_miles': ordinary_miles, 'daily': trace}


def distributed_support(dates, sessions, *, ordinary_miles, window_days=14):
    """Score only future dates, using actual history plus the proposed sequence.

    Sessions are (date, ordinary-easy miles), not one credit per run. Aggregate
    on each date before capping at one ordinary equivalent, so splitting a run
    into files does not increase support. Short runs get proportional credit;
    excess early mileage cannot accumulate a bank of future support credits.
    The additive deficit / seven-day cadence preserves session-equivalent units.
    """
    dates = tuple(dates)
    if not dates or any(b != a + timedelta(days=1) for a, b in zip(dates, dates[1:])):
        raise ValueError('Require nonempty consecutive scoring dates')
    if not isinstance(window_days, int) or isinstance(window_days, bool) or window_days <= 7:
        raise ValueError('Support window must be an integer greater than seven days')
    if not isfinite(ordinary_miles) or ordinary_miles <= 0:
        raise ValueError('Require a positive finite observed ordinary reference')
    by_date = defaultdict(float)
    for day, miles in sessions:
        if not isfinite(miles) or miles < 0:
            raise ValueError('Support distance must be finite and nonnegative')
        if day > dates[-1]:
            raise ValueError('Support session lies beyond the scored sequence')
        by_date[day] += miles
    required = window_days / 7 - 1
    active = deque()
    supplied = 0.
    trace = []
    cursor = dates[0] - timedelta(days=window_days - 1)
    while cursor <= dates[-1]:
        credit = min(1., by_date.get(cursor, 0.) / ordinary_miles)
        active.append(credit)
        supplied += credit
        if len(active) > window_days:
            supplied -= active.popleft()
        if cursor >= dates[0]:
            deficit = max(0., required - supplied)
            trace.append({'date': cursor.isoformat(), 'supplied_equivalents': max(0., supplied),
                          'missing_equivalents': deficit})
        cursor += timedelta(days=1)
    return {'violation': sum(row['missing_equivalents'] for row in trace) / 7,
            'window_days': window_days, 'required_equivalents': required,
            'ordinary_reference_miles': ordinary_miles, 'daily': trace}


def continuation_support(days, states, completed_runs, *, window_days=14, policy='local',
                         history_source='observed'):
    """Explicit history required; never synthesize it from a last-run average."""
    from run_analysis import weekly_schedule as planner

    if completed_runs is None:
        raise ValueError('Local support requires explicit completed history')
    if not days or len(days) != len(states):
        raise ValueError('Require nonempty aligned days and states')
    origin = states[0].as_of
    evidence = tuple(completed_runs)
    if history_source not in {'observed', 'realized_simulation'}:
        raise ValueError('Unknown completed-history source')
    if any((run.projected and history_source != 'realized_simulation') or run.start_time >= origin for run in evidence):
        raise ValueError('Completed support history must precede the decision; projected runs require an explicit realized simulation ledger')
    # One observed reference across the physical sequence, never a reference
    # inferred from the candidate's own shorter or longer prescriptions.
    ordinary = planner.typical_easy_distance(states[0])[0]
    sessions = [(run.start_time.astimezone(origin.tzinfo).date(), run.distance_miles)
                for run in evidence if run.workout_type == planner.WorkoutType.EASY]
    sessions += [(day.date, planner._midpoint(day.recommendation)) for day in days
                 if day.recommendation and day.recommendation.workout_type == planner.WorkoutType.EASY]
    if policy == 'bounded':
        return bounded_support([day.date for day in days], sessions, ordinary_miles=ordinary)
    if policy != 'local':
        raise ValueError('Unknown local support policy')
    return distributed_support([day.date for day in days], sessions,
                               ordinary_miles=ordinary, window_days=window_days)
