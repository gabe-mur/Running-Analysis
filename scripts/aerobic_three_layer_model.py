"""Offline three-layer model. Not imported by app, scheduler or live scoring.

Log pace = period mean + random day effect + shared adjustment error + run error.
Paule-Mandel estimates additional day variance, with a modified Hartung-Knapp
scale and Student-t intervals. This covariance-aware extension is experimental.
"""
from functools import lru_cache
import math

import numpy as np


@lru_cache(maxsize=1)
def quadrature():
    return np.polynomial.legendre.leggauss(128)


def student_cdf(x, df):
    if df<1:raise ValueError('At least one residual degree of freedom required')
    nodes,weights=quadrature();length=abs(x)
    points=(nodes+1)*length/2
    log_constant=math.lgamma((df+1)/2)-math.lgamma(df/2)-.5*math.log(df*math.pi)
    area=float(weights@np.exp(log_constant-(df+1)/2*np.log1p(points*points/df)))*length/2
    return min(1.,max(0.,.5+math.copysign(area,x)))


@lru_cache(maxsize=512)
def student_critical(df, probability=.975):
    if not .5<probability<1:raise ValueError('Upper-tail probability required')
    lo,hi=0.,1.
    while student_cdf(hi,df)<probability:hi*=2
    for _ in range(45):
        mid=(lo+hi)/2
        if student_cdf(mid,df)<probability:lo=mid
        else:hi=mid
    return (lo+hi)/2


def fit_layers(y, measurement_variance, gradients, covariance, groups, *, day_correlation=None, contrasts=None):
    """Fit period means and one additional day variance; do not label causes.

    Measurement variance excludes identified shared calibration. Day effects
    are independent unless an explicit correlation matrix is supplied. Means
    are constant within each comparison period; no training-load input exists.
    """
    y=np.asarray(y,float);m=np.asarray(measurement_variance,float)
    g=np.asarray(gradients,float);c=np.asarray(covariance,float);groups=np.asarray(groups,int)
    n=len(y);p=int(groups.max())+1 if n else 0
    if (n<=p or m.shape!=(n,) or g.ndim!=2 or g.shape[0]!=n or c.shape!=(g.shape[1],g.shape[1])
        or not np.isfinite(y).all() or not np.isfinite(m).all() or np.any(m<=0)
        or not np.isfinite(g).all() or not np.isfinite(c).all()
        or not np.allclose(c,c.T) or np.linalg.eigvalsh(c).min() < -1e-10
        or set(groups)!=set(range(p))):raise ValueError('Finite aligned data, positive measurement variance and residual degrees of freedom required')
    r=np.eye(n) if day_correlation is None else np.asarray(day_correlation,float)
    if r.shape!=(n,n) or not np.isfinite(r).all() or not np.allclose(r,r.T) or np.linalg.eigvalsh(r).min() < -1e-10:
        raise ValueError('Positive semidefinite aligned day correlation required')
    if not np.allclose(np.diag(r),1):raise ValueError('Day correlation must have unit diagonal')
    x=np.eye(p)[groups];known=np.diag(m)+g@c@g.T;df=n-p
    def solve(tau):
        v=known+tau*r
        solved=np.linalg.solve(v,np.column_stack([x,y]))
        inv_x=solved[:,:p];inv_y=solved[:,p]
        beta_cov=np.linalg.inv(x.T@inv_x)
        a=beta_cov@inv_x.T;beta=a@y
        residual=y-x@beta
        q=float(residual@(inv_y-inv_x@beta))
        return beta,beta_cov,a,q
    beta,beta_cov,a,q=solve(0.)
    tau=0.
    if q>df:
        lo,hi=0.,max(float(np.var(y))*2,float(m.max()),1e-8)
        while solve(hi)[3]>df:
            hi*=2
            if hi>100:raise ValueError('Day variance not bounded by the model')
        for _ in range(26):
            mid=(lo+hi)/2
            if solve(mid)[3]>df:lo=mid
            else:hi=mid
        tau=(lo+hi)/2;beta,beta_cov,a,q=solve(tau)
    scale=max(1.,q/df)
    critical=student_critical(df)
    def uncertainty(contrast):
        contrast=np.asarray(contrast,float);w=contrast@a
        measurement=float(np.sum(w*w*m));day=float(tau*w@r@w)
        shared_gradient=w@g;shared=max(0.,float(shared_gradient@c@shared_gradient))
        total=(measurement+day+shared)*scale
        estimate=float(contrast@beta)
        return dict(estimate=estimate,measurement_variance=measurement,day_variance=day,
            shared_variance=shared,small_sample_scale=scale,se=math.sqrt(total),
            range_95=[estimate-critical*math.sqrt(total),estimate+critical*math.sqrt(total)],
            probability_positive=student_cdf(estimate/max(math.sqrt(total),1e-15),df))
    levels=[uncertainty(np.eye(p)[i]) for i in range(p)]
    result=dict(periods=levels,extra_day_variance=tau,extra_day_log_sd=math.sqrt(tau),df=df,
        critical_95=critical,residual_q=q,small_sample_scale=scale,run_counts=np.bincount(groups,minlength=p).tolist())
    if p==2:result['change']=uncertainty([-1.,1.]) # Group 0=current; positive means faster than prior.
    if contrasts is not None:
        result['contrasts']={name:uncertainty(vector) for name,vector in contrasts.items()}
    # Prediction concerns true next-day standardized performance at unchanged
    # underlying level, not raw outdoor pace or a new sensor reading. It retains
    # uncertainty in that level and adds an independent next-day deviation.
    level=levels[0];predictive_se=math.sqrt(level['se']**2+tau)
    result['next_day_true_performance']=dict(estimate=level['estimate'],se=predictive_se,
        range_95=[level['estimate']-critical*predictive_se,level['estimate']+critical*predictive_se],
        assumption='Unchanged period level and independent next-day effect; not a prescribed pace or readiness forecast')
    w=a[0];new_m=float(np.median(m[groups==0]));new_g=g[groups==0].mean(axis=0)
    difference_gradient=new_g-w@g
    prediction_variance=(float(np.sum(w*w*m))+new_m+tau*(float(w@r@w)+1.)
        +max(0.,float(difference_gradient@c@difference_gradient)))*scale
    se=math.sqrt(prediction_variance)
    result['next_comparable_measured_run']=dict(estimate=level['estimate'],se=se,
        range_95=[level['estimate']-critical*se,level['estimate']+critical*se],
        assumed_measurement_variance=new_m,assumed_shared_gradient=new_g.tolist(),
        assumption='Unchanged level, independent next-day effect, current median run precision and average adjustment exposure')
    return result


def fit_period_layers(y, measurement_variance, gradients, covariance, groups, *, contrasts=None):
    """Period-specific extra variation; finite-variance uncertainty propagated.

    Normal moment covariance for estimated day variances is obtained from the
    residual projection matrices. Student-t degrees of freedom use the delta
    approximation (Welch-Satterthwaite), bounded by total residual df. This is
    still an experimental independent-day model, not a serial-noise solution.
    """
    y=np.asarray(y,float);m=np.asarray(measurement_variance,float)
    g=np.asarray(gradients,float);c=np.asarray(covariance,float);groups=np.asarray(groups,int)
    # Reuse full input validation before fitting individual periods.
    baseline=fit_layers(y,m,g,c,groups)
    p=len(baseline['periods']);n=len(y);x=np.eye(p)[groups]
    indices=[np.flatnonzero(groups==i) for i in range(p)]
    if any(len(ix)<2 for ix in indices):raise ValueError('At least two observations per period to estimate its day variation')
    tau=[];projections=[];traces=[]
    for ix in indices:
        fit=fit_layers(y[ix],m[ix],g[ix],c,np.zeros(len(ix),int))
        tau.append(fit['extra_day_variance'])
        v=np.diag(m[ix]+tau[-1])+g[ix]@c@g[ix].T
        inv=np.linalg.inv(v);one=np.ones(len(ix));wi=inv@one
        projection=inv-np.outer(wi,wi)/float(one@wi)
        projections.append(projection);traces.append(float(np.trace(projection)))
    tau=np.array(tau);v=np.diag(m+tau[groups])+g@c@g.T
    solved=np.linalg.solve(v,np.column_stack([x,y]));inv_x=solved[:,:p];inv_y=solved[:,p]
    beta_cov=np.linalg.inv(x.T@inv_x);a=beta_cov@inv_x.T;beta=a@y
    residual=y-x@beta;q=float(residual@(inv_y-inv_x@beta));scale=max(1.,q/(n-p))
    tau_cov=np.zeros((p,p))
    for i,ix in enumerate(indices):
        for j,jx in enumerate(indices):
            cross=v[np.ix_(ix,jx)]
            tau_cov[i,j]=2*float(np.trace(projections[i]@cross@projections[j]@cross.T))/(traces[i]*traces[j])
    def uncertainty(contrast,predict=False,measured=False):
        contrast=np.asarray(contrast,float);w=contrast@a
        measurement=float(np.sum(w*w*m));day=float(np.sum(w*w*tau[groups]))
        shared_gradient=w@g
        if measured:
            measurement+=float(np.median(m[groups==0]))
            shared_gradient=g[groups==0].mean(axis=0)-shared_gradient
        shared=max(0.,float(shared_gradient@c@shared_gradient))
        total=(measurement+day+shared)*scale
        derivative=np.array([float(np.sum(w[ix]**2)) for ix in indices])*scale
        if predict:
            addition=tau[0]*scale if measured else tau[0]
            total+=addition;day+=addition/scale;derivative[0]+=scale if measured else 1.
        variance_of_variance=max(0.,float(derivative@tau_cov@derivative))
        df=min(n-p,max(1,int(math.floor(2*total**2/max(variance_of_variance,1e-30)))))
        se=math.sqrt(total);critical=student_critical(df);estimate=float(contrast@beta)
        return dict(estimate=estimate,measurement_variance=measurement,day_variance=day,
            shared_variance=shared,small_sample_scale=scale,se=se,df=df,critical_95=critical,
            range_95=[estimate-critical*se,estimate+critical*se],
            probability_positive=student_cdf(estimate/max(se,1e-15),df))
    result=dict(periods=[uncertainty(np.eye(p)[i]) for i in range(p)],
        extra_day_variance=float(tau[0]),extra_day_log_sd=math.sqrt(tau[0]),
        period_day_log_sd=np.sqrt(tau).tolist(),df=n-p,residual_q=q,small_sample_scale=scale,
        run_counts=np.bincount(groups,minlength=p).tolist(),variance_model='period-specific day variance; independent days')
    if p==2:result['change']=uncertainty([-1.,1.])
    if contrasts is not None:result['contrasts']={k:uncertainty(v) for k,v in contrasts.items()}
    result['next_day_true_performance']=dict(uncertainty(np.eye(p)[0],predict=True),
        assumption='Unchanged current period mean and independent next-day effect; not raw measured pace or readiness')
    result['next_comparable_measured_run']=dict(uncertainty(np.eye(p)[0],predict=True,measured=True),
        assumed_measurement_variance=float(np.median(m[groups==0])),assumed_shared_gradient=g[groups==0].mean(axis=0).tolist(),
        assumption='Unchanged level, independent next-day effect, current median run precision and average adjustment exposure')
    return result
