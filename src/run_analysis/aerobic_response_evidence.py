"""Data-retaining HR response evidence and offline balance controls.

For a first-order response h, integral(demand) = integral(h) + tau * delta(h).
Unlike averaging selected pace/HR ratios, the boundary term charges an effort
whose HR is still rising and credits recovery whose HR is still falling. The
on/off extension is an empirical test, not an established physiological law.
No run is removed based on its pace, score, HR zone or changing effort.
"""
from dataclasses import dataclass
import math

import numpy as np


TAUS = (0., 15., 30., 60., 90., 120., 180., 240., 300.)


@dataclass(frozen=True)
class BalanceBlock:
    start: float
    seconds: float
    moving_minute: float
    effort_distance: float
    hr_area: float
    rise: float
    fall: float
    moving_seconds: float
    high_hr_seconds: float = 0.
    first_hr_effort: float | None = None
    last_hr_effort: float | None = None

    def __post_init__(self):
        if (not all(v is None or math.isfinite(v) for v in self.__dict__.values())
                or self.seconds <= 0 or self.effort_distance < 0 or self.hr_area <= 0
                or self.rise < 0 or self.fall > 0
                or not 0 <= self.moving_seconds <= self.seconds+1e-6):
            raise ValueError('Finite observed balance block required')


def score(blocks, rise_tau=0., fall_tau=0.):
    """Ratio of integrals; all supplied blocks retain their complete duration."""
    if min(rise_tau, fall_tau) < 0 or not all(math.isfinite(v) for v in (rise_tau, fall_tau)):
        raise ValueError('Finite nonnegative response times required')
    distance = sum(b.effort_distance for b in blocks)
    area = sum(b.hr_area for b in blocks)
    boundary = rise_tau*sum(b.rise for b in blocks)+fall_tau*sum(b.fall for b in blocks)
    if not blocks or distance <= 0 or area+boundary <= 0:
        return dict(available=False, reason='Nonpositive measured response balance', blocks=len(blocks))
    return dict(available=True, pace_min_mile=1609.344/60*(area+boundary)/distance,
                unaligned_pace_min_mile=1609.344/60*area/distance,
                response_fraction=boundary/area, blocks=len(blocks),
                moving_minutes=sum(b.moving_seconds for b in blocks)/60,
                high_hr_minutes=sum(b.high_hr_seconds for b in blocks)/60)


def candidate_pairs(asymmetric=False):
    return tuple((a, b) for a in TAUS for b in TAUS if asymmetric or a == b)


def losses(blocks, pairs):
    """One duration-normalized loss per run, after fitting its nuisance level.

    HR/effort calibration is fixed outside this experiment. This compares
    measured within-run response balance, not independent fitness accuracy.
    """
    if not blocks:
        return None
    dt = np.array([b.seconds for b in blocks])
    v = np.array([b.effort_distance for b in blocks])/dt
    h = np.array([b.hr_area for b in blocks])/dt
    rise = np.array([b.rise for b in blocks])/dt
    fall = np.array([b.fall for b in blocks])/dt
    if not np.any(v > 0):
        return None
    w = dt/dt.sum()
    costs = h[:, None]+rise[:, None]*np.array([p[0] for p in pairs])+fall[:, None]*np.array([p[1] for p in pairs])
    level = np.sum(w[:, None]*v[:, None]*costs, axis=0)/np.sum(w*v*v)
    residual = costs-v[:, None]*level
    return np.average(residual**2, axis=0, weights=w).tolist()


def select(loss_by_run, identity, pairs):
    others = [v for k, v in loss_by_run.items() if k != identity and v is not None]
    if len(others) < 3:
        return pairs.index((0., 0.))
    return int(np.argmin(np.mean(others, axis=0)))


def blocks_from_intervals(intervals, config, calibration, *, hourly=None, terrain_quality=None, block_seconds=120.):
    """Keep every readable moving interval, including warmup, Z4 and restarts.

    Missing/long recording intervals are recorded, never filled with invented
    pace or HR. The next readable section starts at its measured HR. Stops are
    not modeled as running at sleeping resting HR. Their measured endpoints
    initialize the surrounding running bouts. No post-stop suppression is used.
    Grade uses the same prepared deltas and micro-distance scale as production.
    """
    from run_analysis.pace_comparability import _features
    from run_analysis.physiology import grade_energy_ratio
    from run_analysis.weather import interpolate_hourly, wind_components

    if not math.isfinite(block_seconds) or block_seconds <= 0:
        raise ValueError('Positive aggregation duration required')
    if not intervals:
        return [], dict(moving_seconds=0., retained_moving_seconds=0., missing_moving_seconds=0., stopped_seconds=0.)
    beta = calibration['parameters']
    origin = intervals[0].start.timestamp_utc.timestamp()
    max_gap = float(config['moving_time']['maximum_interval_seconds'])
    maximum_grade = float(config['elevation']['maximum_plausible_grade_percent'])
    micro = float(config['elevation']['grade_cost_window_meters'])
    # Use distance-sized groups, not noisy instantaneous gradients. Preserve
    # each group's integrated terrain cost when allocating to time blocks.
    grade_ratios = np.ones(len(intervals))
    indexes = []; distance = delta = 0.
    def flush():
        nonlocal indexes, distance, delta
        if distance >= min(20., micro/3.) and abs(delta/distance*100) <= maximum_grade:
            ratio = grade_energy_ratio(delta/distance*100, maximum_grade)
            for ix in indexes:
                grade_ratios[ix] = ratio
        indexes = []; distance = delta = 0.
    usable_grade = (terrain_quality or {}).get('usable') is not False
    for ix, interval in enumerate(intervals):
        if interval.stopped_time_s > 0 or interval.elapsed_s > max_gap:
            flush()
        if usable_grade and interval.elevation_delta_m is not None and interval.distance_m > 0:
            indexes.append(ix); distance += interval.distance_m; delta += interval.elevation_delta_m
            if distance >= micro:
                flush()
    flush()
    blocks = []; current = []; moving = retained = missing = stopped = 0.
    def finish():
        if not current:
            return
        start = current[0]['start']; end = current[-1]['end']
        h0 = current[0]['h0']; h1 = current[-1]['h1']
        change = h1-h0
        blocks.append(BalanceBlock(start=start, seconds=end-start,
            moving_minute=sum(r['minute']*r['seconds'] for r in current)/(end-start),
            effort_distance=sum(r['distance'] for r in current), hr_area=sum(r['area'] for r in current),
            rise=max(0., change), fall=min(0., change),
            moving_seconds=sum(r['moving'] for r in current), high_hr_seconds=sum(r['high_hr'] for r in current),
            first_hr_effort=h0, last_hr_effort=h1))
        current.clear()
    for ix, interval in enumerate(intervals):
        dt = interval.elapsed_s
        if dt <= 0:
            continue
        before = moving; moving += interval.moving_time_s; stopped += interval.stopped_time_s
        values = (interval.start.heart_rate_bpm, interval.end.heart_rate_bpm)
        readable = (dt <= max_gap and interval.distance_source != 'unknown'
            and not ({'distance_reset', 'implausible_distance_speed'} & set(getattr(interval, 'flags', ())))
            and all(v is not None and math.isfinite(v) and v > config['resting_hr'] for v in values))
        if not readable or interval.moving_time_s <= 0:
            finish(); missing += interval.moving_time_s
            continue
        start = interval.start.timestamp_utc.timestamp()-origin
        if current and (start-current[-1]['end'] > 1e-6 or start-current[0]['start'] >= block_seconds):
            finish()
        hr = sum(values)/2; minute = (before+interval.moving_time_s/2)/60
        speed = interval.distance_m/dt
        if speed <= 0:
            finish(); missing += interval.moving_time_s
            continue
        weather = interpolate_hourly(hourly or {}, interval.start.timestamp_utc)
        head, tail, cross = wind_components(weather['wind_speed_mph'], weather['wind_direction_degrees'], interval.bearing_degrees)
        f = _features(dict(moving_pace_min_mile=1609.344/60/speed, average_hr_bpm=hr,
            moving_minutes_into_run=minute, grade_energy_ratio=grade_ratios[ix], grade_complete_fraction=1.,
            headwind_signed_mph=(float(head or 0)-float(tail or 0)) if head is not None or tail is not None else None,
            crosswind_mph=cross, **weather), config)
        effort = speed*grade_ratios[ix]*math.exp(beta[2]*f['wind_log']-beta[1]*f['time']
            +calibration['heat']['mean']*(f['heat'] or 0.))
        h0, h1 = [((v-config['resting_hr'])/(config['target_hr']-config['resting_hr']))**beta[0] for v in values]
        # Integrate the existing HR-reserve transformation before averaging.
        # A two-point Gaussian rule avoids taking powers of an averaged HR.
        hs = [((hr+(values[1]-values[0])*q-config['resting_hr'])/(config['target_hr']-config['resting_hr']))**beta[0]
              for q in (-1/(2*math.sqrt(3)), 1/(2*math.sqrt(3)))]
        current.append(dict(start=start, end=start+dt, seconds=dt, minute=minute,
            distance=effort*dt, area=sum(hs)/2*dt, h0=h0, h1=h1, moving=interval.moving_time_s,
            high_hr=dt if hr > config['model']['maximum_reliable_hr_bpm'] else 0.))
        retained += interval.moving_time_s
    finish()
    return blocks, dict(moving_seconds=moving, retained_moving_seconds=retained,
        missing_moving_seconds=missing, stopped_seconds=stopped, block_seconds=block_seconds,
        blocks=len(blocks), score_based_exclusions=0)
