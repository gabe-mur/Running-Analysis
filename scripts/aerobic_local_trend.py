"""Retrospective trend preview fitted directly to comparable run observations.

Local linear regression of log pace with a fixed 28-calendar-day Gaussian
bandwidth and inverse marginal measurement/calibration variance. This is not
a new live fitness model or a forward-tested estimator. Historical points
use observations on both sides; they can revise after subsequent uploads.
"""
from argparse import ArgumentParser
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from run_analysis.aerobic_candidate import measurement_inputs
from run_analysis.privacy import private_file
from scripts.aerobic_decay_history import PrecisionDecayHistory


class LocalPaceTrend(PrecisionDecayHistory):
    def __init__(self, *args, bandwidth_days=28., gap_days=28., **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(bandwidth_days) or bandwidth_days <= 0 or not math.isfinite(gap_days) or gap_days <= 0:
            raise ValueError('Positive finite bandwidth and gap required')
        self.bandwidth = float(bandwidth_days)
        self.segments = np.split(np.arange(len(self.t)), np.flatnonzero(np.diff(self.t)>gap_days)+1)

    def level(self, anchor):
        anchor=float(anchor)
        if not math.isfinite(anchor):raise ValueError('Finite anchor required')
        ix=next((s for s in self.segments if len(s) and self.t[s[0]]<=anchor<=self.t[s[-1]]),None)
        if ix is None:return dict(available=False,anchor=anchor)
        g=self.g[ix];v=self.m[ix]+np.einsum('ij,jk,ik->i',g,self.c,g)
        dt=(self.t[ix]-anchor)/self.bandwidth
        log_w=-.5*dt**2-np.log(v);w=np.exp(log_w-log_w.max())
        x=np.column_stack([np.ones(len(ix)),dt]);root=np.sqrt(w)
        # The intercept is the trend at the requested calendar date, not an
        # average of rolling means or a decorative interpolation spline.
        beta=np.linalg.lstsq(root[:,None]*x,root*self.y[ix],rcond=None)[0]
        influence=np.linalg.pinv(root[:,None]*x)[0]*root
        shared=influence@g
        return dict(available=True,anchor=anchor,log_pace=float(beta[0]),pace_min_mile=math.exp(float(beta[0])),
            run_count=len(ix),conditional_variance=float((influence*influence)@self.m[ix]+shared@self.c@shared),
            future_observation_weight_present=bool(np.any(self.t[ix]>anchor)),
            uncertainty_kind='Conditional measurement/calibration sensitivity; not a fitness confidence interval')


def preview(payload, *, history_days=180, bandwidth_days=28.):
    report=payload['pace_comparison']
    rows=sorted([r for r in report['runs'] if r['included_in_trend'] and r['trend_weight']>0],key=lambda r:r['start_time'])
    if not rows:raise ValueError('Comparable run observations required')
    dates=[datetime.fromisoformat(r['start_time']) for r in rows];end=dates[-1]
    cov=np.zeros((4,4));cov[:3,:3]=report['calibration']['covariance'];cov[3,3]=report['calibration']['heat']['sd']**2
    m,g=measurement_inputs(rows,cov)
    times=np.array([(d-end).total_seconds()/86400 for d in dates])
    curve=LocalPaceTrend(np.log([r['pace_min_mile'] for r in rows]),m,g,cov,times,bandwidth_days=bandwidth_days)
    start=max(end-timedelta(days=history_days),dates[0]);anchors=np.arange((start-end).total_seconds()/86400,0.,1.).tolist()+[0.]
    line=[]
    for anchor in anchors:
        value=curve.level(anchor)
        line.append(dict(date=(end+timedelta(days=anchor)).isoformat(),pace=value.get('pace_min_mile')))
    points=[dict(date=d.isoformat(),id=r['activity_id'],pace=r['pace_min_mile'],
        range=r.get('estimated_range_95_min_mile')) for r,d in zip(rows,dates) if d>=start]
    return dict(version='direct-local-pace-trend-preview-v1',deployed=False,
        target_hr_bpm=payload['target_hr'],start=start.isoformat(),end=end.isoformat(),points=points,line=line,
        current=curve.level(0.)['pace_min_mile'],bandwidth_days=bandwidth_days,
        method='Direct local linear fit of comparable log pace; Gaussian calendar weighting and marginal precision',
        scope='Retrospective descriptive trend; uses later runs to estimate earlier dates; not forward validated',
        uncertainty='No confidence band claimed; dots preserve all selected comparable run observations')


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('New output path required')
    with urlopen('http://127.0.0.1:8000/api/progress-wip?window_days=365&include_exploratory=false',timeout=120) as response:
        payload=json.load(response)
    result=preview(payload)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    print(json.dumps(dict(runs=len(result['points']),current_pace=result['current'],bandwidth_days=result['bandwidth_days'])))


if __name__=='__main__':main()
