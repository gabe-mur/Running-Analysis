"""Direct retrospective performance trend on shared comparison evidence.

The run/window set is retained with an explicit HR timing correction.
This is a descriptive trend in comparable pace,
not a measured fitness state, future forecast or another VO2 conversion.
"""
from datetime import datetime, timedelta
import json
import math

import numpy as np

from .aerobic_candidate import measurement_inputs
from .read_cache import invariant_read


VERSION = 'direct-aerobic-performance-v9-headline-amount'
BANDWIDTH_DAYS = 28.
GAP_DAYS = 28.
COMPARISON_BANDWIDTH_DAYS = 14.
PROGRESS_ASSESSMENT_DAYS = 84.
PROGRESS_ESTABLISHED_DAYS = 56.


class LocalPaceTrend:
    """The audited local log-pace regression, without research imports."""
    def __init__(self, y, measurement, gradients, covariance, times, *, bandwidth_days=28., gap_days=28.):
        self.y, self.m, self.g, self.c, self.t = [np.array(v, dtype=float, copy=True)
            for v in (y, measurement, gradients, covariance, times)]
        n = len(self.y)
        if (not n or self.y.shape != (n,) or self.m.shape != (n,) or self.t.shape != (n,)
            or self.g.ndim != 2 or self.g.shape[0] != n or self.c.shape != (self.g.shape[1], self.g.shape[1])
            or not all(np.isfinite(v).all() for v in (self.y, self.m, self.g, self.c, self.t))
            or np.any(self.m <= 0) or np.any(np.diff(self.t) <= 0)
            or not np.allclose(self.c, self.c.T) or np.linalg.eigvalsh(self.c).min(initial=0.) < -1e-10):
            raise ValueError('Finite chronological aligned observations, positive precision and PSD covariance required')
        if not math.isfinite(bandwidth_days) or bandwidth_days <= 0 or not math.isfinite(gap_days) or gap_days <= 0:
            raise ValueError('Positive finite bandwidth and gap required')
        self.bandwidth = float(bandwidth_days)
        self.segments = np.split(np.arange(n), np.flatnonzero(np.diff(self.t) > gap_days)+1)
        for value in (self.y, self.m, self.g, self.c, self.t):
            value.setflags(write=False)

    def level(self, anchor):
        anchor = float(anchor)
        if not math.isfinite(anchor):
            raise ValueError('Finite anchor required')
        ix = next((s for s in self.segments if len(s) and self.t[s[0]] <= anchor <= self.t[s[-1]]), None)
        if ix is None:
            return dict(available=False, anchor=anchor)
        g = self.g[ix]
        variance = self.m[ix]+np.einsum('ij,jk,ik->i', g, self.c, g)
        dt = (self.t[ix]-anchor)/self.bandwidth
        log_w = -.5*dt**2-np.log(variance)
        w = np.exp(log_w-log_w.max())
        x = np.column_stack([np.ones(len(ix)), dt])
        root = np.sqrt(w)
        beta = np.linalg.lstsq(root[:, None]*x, root*self.y[ix], rcond=None)[0]
        influence = np.linalg.pinv(root[:, None]*x)[0]*root
        shared = influence@g
        return dict(available=True, anchor=anchor, log_pace=float(beta[0]), pace_min_mile=math.exp(float(beta[0])),
            run_count=len(ix), conditional_variance=float((influence*influence)@self.m[ix]+shared@self.c@shared),
            future_observation_weight_present=bool(np.any(self.t[ix] > anchor)),
            uncertainty_kind='Conditional measurement/calibration sensitivity; not a fitness confidence interval')

    def progress_trend(self):
        """Assess one broad trajectory, leaving the responsive curve unchanged.

        Fit independent run observations over up to 12 weeks in the latest
        continuous segment, using the model's existing marginal precisions.
        Daily fitted curve anchors are not independent extra observations.
        Shared correction uncertainty is contrasted jointly; empirical residual
        variability remains a floor. This is not a physiological confidence CI.
        """
        from .analytics import _directional_evidence
        segment=self.segments[-1];end=self.t[segment[-1]]
        ix=segment[self.t[segment]>=end-PROGRESS_ASSESSMENT_DAYS];t=self.t[ix]
        span=float(end-t[0])
        context=dict(assessment_days=PROGRESS_ASSESSMENT_DAYS,span_days=span,run_count=len(ix),
            early_estimate=span<PROGRESS_ESTABLISHED_DAYS,
            basis='Single pace trajectory over up to 12 weeks; independent of displayed history',
            scope='Model-based pace trend, not proof of physiological change')
        if len(ix)<3 or span<14:
            return dict(direction='insufficient_data',evidence='provisional',**context)
        g=self.g[ix];m=self.m[ix];values=self.y[ix]
        variance=m+np.einsum('ij,jk,ik->i',g,self.c,g)
        w=1/variance;w/=w.sum()
        effective=float(1/(w@w))
        if effective<3:
            return dict(direction='insufficient_data',evidence='provisional',effective_run_count=effective,**context)
        x=np.column_stack([np.ones(len(ix)),(t-end)/PROGRESS_ASSESSMENT_DAYS]);root=np.sqrt(w)
        beta=np.linalg.lstsq(root[:,None]*x,root*values,rcond=None)[0]
        influence=np.linalg.pinv(root[:,None]*x)[1]*root/PROGRESS_ASSESSMENT_DAYS
        log_change=float(influence@values)
        shared=influence@g
        measurement=float((influence*influence)@m+shared@self.c@shared)
        residual=values-x@beta
        residual_variance=float(w@(residual*residual))*effective/(effective-2.)
        empirical=residual_variance*float(influence@influence)
        error=math.sqrt(max(measurement,empirical,1e-24))
        evidence=_directional_evidence(log_change,error)
        direction=evidence['directional_interpretation']
        return dict(direction='maintaining' if direction=='stable_or_uncertain' else direction,
            evidence=evidence['evidence_strength'],effective_run_count=effective,
            pace_change_seconds_per_mile=60*math.exp(float(beta[0]))*(-math.expm1(-log_change*span)),
            log_pace_change_per_day=log_change,standard_error_log_pace_change_per_day=error,
            **context)


def _fit(report):
    context=report.get('chart_context') or {}
    hidden=set(context.get('hidden_activity_ids',[]))
    display_context=dict(hidden_activity_ids=sorted(hidden),
        manually_excluded_activity_ids=context.get('manually_excluded_activity_ids',[]),
        quality_sessions=context.get('quality_sessions',[]))
    rows = sorted([r for r in report.get('runs', []) if r.get('included_in_trend') and r.get('trend_weight', 0) > 0
        and r['activity_id'] not in hidden and not r.get('manually_excluded') and r.get('workout_type') not in {'hike','bike'}],
                  key=lambda r: (r['start_time'], r['activity_id']))
    if not rows:
        return dict(version=VERSION, available=False, reason='No comparable aerobic observations',observations=[],**display_context)
    if len({r['activity_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate physical activity in aerobic evidence')
    if any(not r.get('comparison_precision') for r in rows):
        return dict(version=VERSION, available=False, reason='Comparable run precision is unavailable',observations=[],**display_context)
    calibration = report['calibration']
    from .pace_temperature import comparison_covariance
    cov = comparison_covariance(calibration)
    dates = [datetime.fromisoformat(r['start_time']) for r in rows]
    end = dates[-1]
    times = np.array([(date-end).total_seconds()/86400 for date in dates])
    m, g = measurement_inputs(rows, cov)
    curve = LocalPaceTrend(np.log([r['pace_min_mile'] for r in rows]), m, g, cov, times,
                           bandwidth_days=BANDWIDTH_DAYS, gap_days=GAP_DAYS)
    comparison = LocalPaceTrend(np.log([r['pace_min_mile'] for r in rows]), m, g, cov, times,
        bandwidth_days=COMPARISON_BANDWIDTH_DAYS, gap_days=GAP_DAYS)
    # Fit once to full history. Viewports only crop these daily/run anchors.
    anchors = sorted(set(np.arange(times[0], 0., 1.).tolist()+times.tolist()))
    history = [dict(as_of=(end+timedelta(days=float(t))).isoformat(), **curve.level(t)) for t in anchors]
    comparison_history = [dict(as_of=(end+timedelta(days=float(t))).isoformat(), **comparison.level(t)) for t in anchors]
    # Paired display control: observed pace from the same retained sections,
    # with identical date/precision weighting. No normalization is applied to y.
    raw_ix=np.array([i for i,r in enumerate(rows) if isinstance(r.get('observed_pace_min_mile'),(int,float))
        and math.isfinite(r['observed_pace_min_mile']) and r['observed_pace_min_mile']>0],dtype=int)
    raw_curve=LocalPaceTrend(np.log([rows[i]['observed_pace_min_mile'] for i in raw_ix]),
        m[raw_ix],g[raw_ix],cov,times[raw_ix],bandwidth_days=BANDWIDTH_DAYS,gap_days=GAP_DAYS) if len(raw_ix) else None
    def raw_level(t):
        level=raw_curve.level(t) if raw_curve is not None else dict(available=False)
        # The corrected model's conditional uncertainty is not a raw-pace CI.
        return {key:level[key] for key in ('available','pace_min_mile','run_count') if key in level}
    raw_current=raw_level(times[raw_ix[-1]]) if len(raw_ix) else dict(available=False)
    unadjusted=dict(available=raw_current['available'],current=raw_current,
        progress_trend=raw_curve.progress_trend() if raw_curve is not None else dict(direction='insufficient_data',evidence='provisional'),
        last_observation=dates[raw_ix[-1]].isoformat() if len(raw_ix) else None,
        history=[dict(as_of=(end+timedelta(days=float(t))).isoformat(),**raw_level(t)) for t in anchors],
        definition='Recorded pace from the same aerobic sections; same date and precision weights, no pace corrections')
    observations = [dict(as_of=date.isoformat(), activity_id=r['activity_id'],
        pace_min_mile=r['pace_min_mile'], observed_pace_min_mile=r['observed_pace_min_mile'],
        trend_pace_min_mile=curve.level(t)['pace_min_mile'],
        unadjusted_trend_pace_min_mile=raw_level(t).get('pace_min_mile'),
        comparison_trend_pace_min_mile=comparison.level(t)['pace_min_mile'], distance_miles=r.get('distance_miles'),
        trend_weight=r['trend_weight'], sampling_support=r.get('sampling_support'),
        workout_type=r.get('workout_type'),health_tag=r.get('health_tag','normal'),
        measurement_quality=('device_distance_fallback' if r.get('gps_coverage',1.)<=0 else
            'partial_gps_device_distance' if r.get('gps_coverage',1.)<.9 else 'full_sensor'),
        continuity=r.get('continuity'),section_screen=r.get('section_screen'),hr_response=r.get('hr_response'),
        temperature_response=r.get('temperature_response'),comparison_pipeline=r.get('comparison_pipeline')) for r, date, t in zip(rows, dates, times, strict=True)]
    return dict(version=VERSION, available=True, target_hr_bpm=calibration['target_hr_bpm'],
        last_observation=end.isoformat(), current=curve.level(0.), history=history, observations=observations,
        progress_trend=curve.progress_trend(),
        unadjusted=unadjusted,
        smoothing_comparison=dict(bandwidth_days=COMPARISON_BANDWIDTH_DAYS,
            baseline_bandwidth_days=BANDWIDTH_DAYS,current=comparison.level(0.),history=comparison_history,
            scope='Display comparison only; same corrected observations, precision, gap rules and reference conditions'),
        section_screen=report.get('section_screen'),
        hr_response=report.get('hr_response'),
        temperature_response=report.get('temperature_response'),comparison_pipeline=report.get('comparison_pipeline'),
        **display_context,
        definition='Comparable aerobic performance in pace units, not measured VO2max',
        assumptions=dict(bandwidth_days=BANDWIDTH_DAYS, gap_days=GAP_DAYS,
            headline='Single precision-weighted trajectory of independent run observations over up to 84 days in the latest continuous segment; no previous-block comparison and no view-dependent refit',
            headline_history='Use available history with at least 3 effective runs and 14 observed days; mark estimates early until 56 days of observed span. These are display support policies, not biological adaptation cutoffs.',
            headline_evidence='Existing 80% likely / 95% clear model-direction thresholds; joint correction sensitivity with empirical residual variability as a floor; not calibrated physiological confidence',
            trend='Direct local linear fit of log pace; Gaussian calendar weighting and marginal precision',
            retrospective='Later runs can revise earlier trend values; normalization harmonizes all history',
            precision='Shared correction covariance retained; no physiological confidence band claimed',
            sampling='Existing reference-HR/time, stability, movement and health rules unchanged; no new settled-section exclusions',
            continuity='Existing window set retained; measured bout-start HR and full recorded response inform timing; missing context keeps original evidence',
            hr_timing='Observed HR plus modeled instantaneous-minus-delayed HR effort; residual HR cost retained, not replaced by the fitted curve',
            outliers='Strong and weak observations retained; no trimming to make the graph smoother',
            unmeasured='No new sleep, medication, fatigue or time-of-day pace corrections'))


def progress_from_comparability(connection, report, *, as_of, history_days=90, chart_context=None):
    if as_of.tzinfo is None or history_days < 1:
        raise ValueError('Timezone-aware date and positive history required')
    selected = {**report, 'runs': [r for r in report.get('runs', []) if datetime.fromisoformat(r['start_time']) <= as_of]}
    context=chart_context if chart_context is not None else report.get('chart_context')
    if context is not None:
        selected['chart_context']={**context,'quality_sessions':[r for r in context.get('quality_sessions',[])
            if datetime.fromisoformat(r['as_of'])<=as_of]}
    result = invariant_read(connection, (VERSION, 'chart-context-v1','smoothing-comparison',COMPARISON_BANDWIDTH_DAYS,
        json.dumps(selected, sort_keys=True)), lambda: _fit(selected))
    result['as_of'] = as_of.isoformat()
    start = as_of-timedelta(days=history_days)
    result['quality_sessions']=[r for r in result['quality_sessions'] if start<=datetime.fromisoformat(r['as_of'])<=as_of]
    if not result['available']:
        return result
    before = [r for r in result['history'] if datetime.fromisoformat(r['as_of']) < start]
    result['history'] = before[-1:]+[r for r in result['history'] if datetime.fromisoformat(r['as_of']) >= start]
    raw=result['unadjusted']
    raw_before=[r for r in raw['history'] if datetime.fromisoformat(r['as_of'])<start]
    raw['history']=raw_before[-1:]+[r for r in raw['history'] if datetime.fromisoformat(r['as_of'])>=start]
    comparison = result['smoothing_comparison']
    predecessor = [r for r in comparison['history'] if datetime.fromisoformat(r['as_of']) < start]
    comparison['history'] = predecessor[-1:]+[r for r in comparison['history'] if datetime.fromisoformat(r['as_of']) >= start]
    result['observations'] = [r for r in result['observations'] if datetime.fromisoformat(r['as_of']) >= start]
    if result.get('section_screen'):
        result['section_screen']['excluded_runs']=[r for r in result['section_screen']['excluded_runs']
            if start<=datetime.fromisoformat(r['start_time'])<=as_of]
    result['days_since_observation'] = (as_of-datetime.fromisoformat(result['last_observation'])).total_seconds()/86400
    return result
