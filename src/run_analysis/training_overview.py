"""Read-only load display: the strip describes exactly the plotted values.

This is a load comparison, not a fitness/recovery classification. Underlying
HR-load accounting and the planner are untouched. Display smoothing is causal.
"""
from datetime import datetime, timedelta, timezone
from statistics import mean
from zoneinfo import ZoneInfo

from .progress import prepare_progress_data
from .health_reports import read_health_reports
from .progress_training_load import recorded_cardio, training_load_history


LOAD_DEADBAND = .05  # Display tolerance around equality, NOT an optimal range.
DISPLAY_SMOOTHING_DAYS = 7
REPORT_FRESHNESS_HOURS = 24
# Provisional coaching reference, not a fitted tolerance or injury threshold.
# Runalyze publishes 0.8–1.3 for ATL/CTL. Here it is applied to our smoothed
# HR-reserve load curves; this transfer is an explicit display assumption.
RECOMMENDED_LOAD_RATIO = (.8, 1.3)
LOW_LOAD_PERSISTENCE_DAYS = 7
LABELS = {'above': 'More load than usual', 'near': 'About your usual load',
    'below': 'Less load than usual', 'unknown': 'Unclear comparison'}


def classify_load(recent, baseline):
    """The same bounded, smoothed values drive the lines, number and strip."""
    if not recent['complete'] or not baseline['complete'] or baseline['low'] <= 0:
        return dict(status='unknown', ratio=None, ratio_bounds=None)
    ratio = recent['value']/baseline['value']
    bounds = dict(low=recent['low']/baseline['high'], high=recent['high']/baseline['low'])
    status = 'above' if bounds['low'] > 1+LOAD_DEADBAND else (
        'below' if bounds['high'] < 1-LOAD_DEADBAND else (
        'near' if bounds['low'] >= 1-LOAD_DEADBAND and bounds['high'] <= 1+LOAD_DEADBAND else 'unknown'))
    return dict(status=status, ratio=ratio, ratio_bounds=bounds)


def smooth_load_history(load, timezone_name='UTC'):
    """Trailing seven-day means of both existing curves, with complete coverage.

    Smooth the values/bounds themselves, not the SVG or a hidden ratio. Callers
    supply six extra dates so the first visible day is independent of the chart
    window. Missing dates/HR remain gaps, not zero or interpolated evidence.
    """
    daily = []
    zone = ZoneInfo(timezone_name)
    for index, row in enumerate(load['daily']):
        week = load['daily'][max(0, index-DISPLAY_SMOOTHING_DAYS+1):index+1]
        dates = [datetime.fromisoformat(item['as_of']).astimezone(zone).date() for item in week]
        continuous = len(week)==DISPLAY_SMOOTHING_DAYS and all(
            b-a == timedelta(days=1) for a,b in zip(dates, dates[1:]))
        def smooth(key):
            complete = continuous and all(item[key]['complete'] for item in week)
            return dict(complete=complete, **{field:mean(item[key][field] for item in week)
                if complete else None for field in ('value', 'low', 'high')})
        recent, baseline = smooth('recent'), smooth('baseline')
        daily.append(dict(as_of=row['as_of'], recent=recent, baseline=baseline,
            **classify_load(recent, baseline)))
    current = daily[-1]
    return dict(daily=daily, current=current, ratio=current['ratio'],
        ratio_bounds=current['ratio_bounds'], smoothing_days=DISPLAY_SMOOTHING_DAYS,
        equality_tolerance=LOAD_DEADBAND, hr_coverage=load.get('hr_coverage'),
        unit=load['unit'], source_memory_days=dict(recent=7, baseline=42))


def load_status(row):
    ratio, status = row['ratio'], row['status']
    detail = 'More recorded HR history is needed for the load comparison.' if ratio is None else (
        'Load uncertainty overlaps the boundary between comparison states.' if status=='unknown' else
        'Recent training load is close to your six-week training trend.' if status=='near' else
        f"Recent training load is {abs(ratio-1)*100:.0f}% {'higher' if ratio>1 else 'lower'} than your six-week training trend.")
    return dict(as_of=row['as_of'], status=status, label=LABELS[status], detail=detail,
        ratio=ratio, ratio_bounds=row['ratio_bounds'],
        confidence='limited' if status=='unknown' else 'descriptive')


def recommended_load_history(display, timezone_name='UTC'):
    """Causal, athlete-relative guidance, separate from planner acceptance.

    The absolute band follows the slow historical curve, not today's dose or
    all-time maximum. Accounting bounds soften the advice, not the load state.
    Low load is not automatically undertraining or a debt to make up.
    """
    zone = ZoneInfo(timezone_name)
    daily, low_days, previous = [], 0, None
    for row in display['daily']:
        day = datetime.fromisoformat(row['as_of']).astimezone(zone).date()
        consecutive = previous is not None and day-previous == timedelta(days=1)
        ratio = row['ratio']
        band = None
        if ratio is None:
            status, label, low_days = 'unavailable', 'Load history incomplete', 0
            action = 'Keep using the issued plan.'
            detail = 'There is not enough complete HR-load history for a personal range.'
            overlap = False
        else:
            low, high = RECOMMENDED_LOAD_RATIO
            baseline = row['baseline']['value']
            band = dict(low=baseline*low, high=baseline*high)
            status = 'below' if ratio < low else 'above' if ratio > high else 'within'
            label = {'below':'Below recommended load', 'within':'Within recommended load',
                     'above':'Above recommended load'}[status]
            low_days = min(LOW_LOAD_PERSISTENCE_DAYS, low_days+1 if consecutive else 1) if status == 'below' else 0
            bounds = row['ratio_bounds']
            overlap = any(bounds['low'] < edge < bounds['high'] for edge in (low, high))
            if status == 'above':
                action = 'Hold off on extra training.'
                detail = 'Recent cardiovascular load exceeds the recommended range. Avoid additional increases; review demanding sessions if recovery feels worse.'
            elif status == 'below' and low_days >= LOW_LOAD_PERSISTENCE_DAYS and bounds['high'] < low:
                action = 'Rebuild consistent training.'
                detail = 'Load has stayed below the range for at least seven days. Resume consistent easy running through the plan; do not cram in missed mileage.'
            elif status == 'below':
                action = 'Keep to your planned recovery and training.'
                detail = 'This is a lighter training patch, not a reason to add a run today. Watch for a sustained reduction rather than chasing daily load.'
            else:
                action = 'Continue your planned training.'
                detail = 'Your cardiovascular load is within the recommended range for your recent training level. No load-based reduction is suggested.'
        daily.append(dict(as_of=row['as_of'], status=status, label=label,
            action=action, detail=detail, recommended_range=band, ratio=ratio,
            boundary_overlap=overlap, consecutive_low_days=low_days))
        previous = day
    return dict(daily=daily, current=daily[-1], ratio_reference=dict(
        low=RECOMMENDED_LOAD_RATIO[0], high=RECOMMENDED_LOAD_RATIO[1]),
        low_load_persistence_days=LOW_LOAD_PERSISTENCE_DAYS,
        scope='Provisional cardiovascular-load guidance, not recovery clearance, injury risk or planner policy',
        reference_url='https://runalyze.com/glossary/ac-ratio')


def build_training_overview(connection, config, *, as_of=None, window_days=90):
    end = as_of or datetime.now(timezone.utc)
    if end.tzinfo is None:
        raise ValueError('Timezone-aware evaluation time required')
    if window_days < 1:
        raise ValueError('Positive display window required')
    prepared = prepare_progress_data(connection)
    timezone_name = config['timezone_default']
    cardio = recorded_cardio(connection, config, end)
    completed = {r['activity_id'] for r in cardio}
    sessions = [s for s in prepared.sessions if s.activity_id in completed]
    points = {p.activity_id:p for p in prepared.points if p.activity_id in completed}
    load = training_load_history(cardio, sessions, points, prepared.details, end,
        window_days+DISPLAY_SMOOTHING_DAYS-1+LOW_LOAD_PERSISTENCE_DAYS-1, timezone_name)
    display = smooth_load_history(load, timezone_name)
    guidance = recommended_load_history(display, timezone_name)
    load['daily'] = load['daily'][-window_days-1:]
    display['daily'] = display['daily'][-window_days-1:]
    guidance['daily'] = guidance['daily'][-window_days-1:]
    statuses = [load_status(row) for row in display['daily']]
    # Keep fresh adverse reports visible separately; they cannot recolor a
    # historical load comparison or become evidence of load/pace adaptation.
    reports = read_health_reports(connection, as_of=end)
    report = reports[-1] if reports and end-reports[-1]['reported_at'] <= timedelta(hours=REPORT_FRESHNESS_HOURS) else None
    check_in = dict(status=report['status'], reported_at=report['reported_at'].isoformat()) if report else None
    # Preserve raw HR-load values; do not ship the removed exploratory scatter.
    for key in ('runs', 'comparison'):
        load.pop(key, None)
    return dict(version='dashboard_load_balance_v2', as_of=end.isoformat(), timezone=timezone_name,
        window_days=window_days, window_start=display['daily'][0]['as_of'],
        training_load=load, load_display=display, status_history=statuses, current_status=statuses[-1],
        load_guidance=guidance,
        current_check_in=check_in,
        activities=[dict(activity_id=s.activity_id, start_time=s.start_time.isoformat(),
            distance_miles=s.distance_miles, role=prepared.details[s.activity_id]['workout'].value)
            for s in sessions if end-timedelta(days=window_days) < s.start_time <= end],
        assumptions=dict(display_smoothing_days=DISPLAY_SMOOTHING_DAYS,
            load_equality_tolerance=LOAD_DEADBAND,
            health_context_freshness_hours=REPORT_FRESHNESS_HOURS,
            scope='Read-only load comparison, not a readiness score, optimal load zone or planner rule',
            history='Retrospective using current stored run analyses and HR settings'))
