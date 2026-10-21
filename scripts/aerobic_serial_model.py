"""Offline dependence experiment; never imported by the application.

Fixed-kernel period-specific Paule-Mandel estimates separate measurement,
shared calibration and correlated day effects. An all-kernel sensitivity
envelope is NOT labelled an exactly calibrated 95% confidence interval.
"""
import math

import numpy as np

from scripts.aerobic_three_layer_model import student_cdf, student_critical


def _period_variance(y,known,r):
    """One-intercept PM root, using a whitened eigenbasis for cheap solves."""
    whitening=np.linalg.inv(np.linalg.cholesky(known))
    eigenvectors_values,eigenvectors=np.linalg.eigh(whitening@r@whitening.T)
    eigenvalues=np.maximum(eigenvectors_values,0.)
    transformed_y=eigenvectors.T@whitening@y
    transformed_one=eigenvectors.T@whitening@np.ones(len(y))
    def q(tau):
        inverse=1/(1+tau*eigenvalues)
        mean=float(np.sum(inverse*transformed_one*transformed_y)/np.sum(inverse*transformed_one**2))
        return float(np.sum(inverse*(transformed_y-mean*transformed_one)**2))
    df=len(y)-1
    if q(0.)<=df:return 0.
    lo,hi=0.,max(float(np.var(y))*2,float(np.diag(known).max()),1e-8)
    while q(hi)>df:
        hi*=2
        if hi>100:raise ValueError('Day variance not bounded by model')
    for _ in range(26):
        mid=(lo+hi)/2
        if q(mid)>df:lo=mid
        else:hi=mid
    return (lo+hi)/2


def correlation_matrix(times, *, half_life_days=None, run_rho=None):
    times=np.asarray(times,float)
    if times.ndim!=1 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Finite strictly increasing run dates required')
    if (half_life_days is None)==(run_rho is None):
        raise ValueError('Specify exactly one calendar or run-index kernel')
    if run_rho is not None:
        if not 0<=run_rho<1:raise ValueError('Run correlation must be in [0,1)')
        return run_rho**np.abs(np.arange(len(times))[:,None]-np.arange(len(times)))
    if not math.isfinite(half_life_days) or half_life_days<0:raise ValueError('Nonnegative finite half life required')
    return np.eye(len(times)) if half_life_days==0 else np.exp2(-np.abs(times[:,None]-times)/half_life_days)


def fit_serial_layers(y,m,g,c,groups,*,correlation,future_correlation=None,contrasts=None):
    """Known correlation kernel; variance amplitudes remain learned per period.

    Period amplitudes are estimated from within-period residuals. Cross-period
    covariance is retained in the joint GLS fit and in variance-of-variance.
    Prediction is unconditional (not kriging): the current mean is its point
    prediction, with covariance between that estimator and the future day.
    """
    y=np.asarray(y,float);m=np.asarray(m,float);g=np.asarray(g,float);c=np.asarray(c,float)
    groups=np.asarray(groups,int);r=np.asarray(correlation,float);n=len(y)
    p=int(groups.max())+1 if n else 0
    if (y.shape!=(n,) or n<=p or groups.shape!=(n,) or set(groups)!=set(range(p))
        or m.shape!=(n,) or g.ndim!=2 or g.shape[0]!=n or c.shape!=(g.shape[1],g.shape[1])
        or not all(np.isfinite(v).all() for v in (y,m,g,c)) or np.any(m<=0)
        or not np.allclose(c,c.T) or np.linalg.eigvalsh(c).min() < -1e-10):
        raise ValueError('Finite aligned observations, positive measurement variance, PSD calibration covariance and residual df required')
    if (r.shape!=(n,n) or not np.isfinite(r).all() or not np.allclose(r,r.T)
        or not np.allclose(np.diag(r),1) or np.linalg.eigvalsh(r).min() < -1e-10):
        raise ValueError('PSD unit-diagonal aligned correlation required')
    x=np.eye(p)[groups];indices=[np.flatnonzero(groups==i) for i in range(p)]
    if any(len(ix)<2 for ix in indices):raise ValueError('At least two observations per period required')
    future=np.zeros(n) if future_correlation is None else np.asarray(future_correlation,float)
    if future.shape!=(n,) or not np.isfinite(future).all():raise ValueError('Aligned finite future correlation required')
    augmented_r=np.block([[r,future[:,None]],[future[None,:],np.ones((1,1))]])
    if np.linalg.eigvalsh(augmented_r).min() < -1e-10:
        raise ValueError('Future correlation must extend the kernel to a PSD matrix')
    tau=[];projections=[];traces=[]
    for ix in indices:
        local_r=r[np.ix_(ix,ix)]
        tau.append(_period_variance(y[ix],np.diag(m[ix])+g[ix]@c@g[ix].T,local_r))
        v=np.diag(m[ix])+g[ix]@c@g[ix].T+tau[-1]*local_r
        inv=np.linalg.inv(v);wi=inv@np.ones(len(ix))
        projection=inv-np.outer(wi,wi)/float(wi.sum())
        projections.append(projection);traces.append(float(np.trace(projection@local_r)))
    tau=np.array(tau);sd=np.sqrt(tau[groups]);day_cov=sd[:,None]*r*sd
    v=np.diag(m)+g@c@g.T+day_cov
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
        measurement=float(np.sum(w*w*m));shared_gradient=w@g
        used_groups=groups;weights=w;used_r=r
        if predict:
            weights=np.r_[-w,1.];used_groups=np.r_[groups,0];used_r=augmented_r
            if measured:
                measurement+=float(np.median(m[groups==0]))
                shared_gradient=g[groups==0].mean(axis=0)-shared_gradient
        weighted_sd=weights*np.sqrt(tau[used_groups])
        day=max(0.,float(weighted_sd@used_r@weighted_sd))
        shared=max(0.,float(shared_gradient@c@shared_gradient));total=(measurement+day+shared)*scale
        derivative=[]
        for k in range(p):
            select=used_groups==k
            if tau[k]>1e-15:
                derivative.append(float(np.sum(weights[select]*(used_r@weighted_sd)[select]))/math.sqrt(tau[k])*scale)
            else:
                # Boundary approximation holds other zero-amplitude cross
                # components at zero. Boundary behavior is tested, not exact.
                derivative.append(float(weights[select]@used_r[np.ix_(select,select)]@weights[select])*scale)
        derivative=np.array(derivative)
        vv=max(0.,float(derivative@tau_cov@derivative))
        df=min(n-p,max(1,int(math.floor(2*total**2/max(vv,1e-30)))))
        se=math.sqrt(total);critical=student_critical(df);estimate=float(contrast@beta)
        return dict(estimate=estimate,se=se,df=df,critical_95=critical,
            measurement_variance=measurement,day_variance=day,shared_variance=shared,small_sample_scale=scale,
            range_95=[estimate-critical*se,estimate+critical*se],
            probability_positive=student_cdf(estimate/max(se,1e-15),df))

    result=dict(periods=[uncertainty(np.eye(p)[i]) for i in range(p)],
        extra_day_variance=float(tau[0]),extra_day_log_sd=math.sqrt(tau[0]),period_day_log_sd=np.sqrt(tau).tolist(),
        run_counts=np.bincount(groups,minlength=p).tolist(),small_sample_scale=scale,residual_q=q,df=n-p,
        restricted_deviance=float(np.linalg.slogdet(v)[1]+np.linalg.slogdet(x.T@inv_x)[1]+q),
        variance_model='period-specific amplitudes; fixed correlated day kernel')
    if p==2:result['change']=uncertainty([-1,1])
    if contrasts is not None:result['contrasts']={k:uncertainty(v) for k,v in contrasts.items()}
    result['next_day_true_performance']=dict(uncertainty(np.eye(p)[0],predict=True),
        assumption='Unchanged mean; future day covariance retained; unconditional prediction, not readiness')
    result['next_comparable_measured_run']=dict(uncertainty(np.eye(p)[0],predict=True,measured=True),
        assumption='Unchanged mean, current median measurement variance and mean adjustment exposure; future covariance retained')
    return result


def kernel_grid(times,future_time):
    """Predeclared sensitivity range, not fitted physiology or a recovery timer.

    Both date-based and run-index kernels are included because the historical
    adverse control was generated by run index, including across a long gap.
    """
    augmented=np.r_[times,future_time]
    kernels={f'calendar_half_life_{h}d':correlation_matrix(augmented,half_life_days=h) for h in (0,1,3,7,14)}
    kernels.update({f'run_rho_{rho}':correlation_matrix(augmented,run_rho=rho) for rho in (.3,.5,.7,.9)})
    return kernels


def fit_serial_sensitivity(y,m,g,c,groups,*,times,future_time,contrasts=None,support_deviance=None):
    """Return blind point selection and all-kernel sensitivity bands separately.

    Selection minimizes restricted deviance evaluated at moment amplitudes,
    NOT a maximized REML profile. Its conditional bands ignore kernel-selection
    uncertainty. The envelope retains all nine assumptions rather than
    presenting that plug-in uncertainty as calibrated.
    """
    fits={name:fit_serial_layers(y,m,g,c,groups,correlation=r[:-1,:-1],future_correlation=r[:-1,-1],contrasts=contrasts)
        for name,r in kernel_grid(times,future_time).items()}
    selected=min(fits,key=lambda name:fits[name]['restricted_deviance'])
    result=dict(fits[selected]);result['selected_kernel']=selected
    def envelope(values,point):
        return dict(point,range_95=[min(v['range_95'][0] for v in values),max(v['range_95'][1] for v in values)],
            interval_kind='Sensitivity envelope of conditional 95% intervals; not an exact 95% confidence interval')
    result['periods']=[envelope([f['periods'][i] for f in fits.values()],point) for i,point in enumerate(result['periods'])]
    if contrasts is not None:result['contrasts']={k:envelope([f['contrasts'][k] for f in fits.values()],v) for k,v in result['contrasts'].items()}
    if 'change' in result:result['change']=envelope([f['change'] for f in fits.values()],result['change'])
    for key in ('next_day_true_performance','next_comparable_measured_run'):
        result[key]=envelope([f[key] for f in fits.values()],result[key])
    result['conditional_selected_fit']=fits[selected]
    result['kernel_diagnostics']={name:dict(restricted_deviance=f['restricted_deviance'],period_day_log_sd=f['period_day_log_sd'],
        monthly=f.get('contrasts',{}).get('monthly'),ninety_day=f.get('contrasts',{}).get('ninety_day')) for name,f in fits.items()}
    if support_deviance is not None:
        if not math.isfinite(support_deviance) or support_deviance<0:
            raise ValueError('Finite nonnegative support-deviance screen required')
        minimum=fits[selected]['restricted_deviance']
        supported={k:f for k,f in fits.items() if f['restricted_deviance']-minimum<=support_deviance}
        def supported_interval(values,point):
            # Do not attach one selected kernel's SE/probability to an interval
            # spanning several incompatible covariance assumptions.
            return dict(estimate=point['estimate'],range_95=[min(v['range_95'][0] for v in values),max(v['range_95'][1] for v in values)],
                interval_kind='Deviance-screened sensitivity envelope, not an exact 95% confidence interval')
        screened=dict(fits[selected],selected_kernel=selected,supported_kernels=list(supported),support_deviance=support_deviance)
        screened['periods']=[supported_interval([f['periods'][i] for f in supported.values()],v) for i,v in enumerate(screened['periods'])]
        if contrasts is not None:screened['contrasts']={k:supported_interval([f['contrasts'][k] for f in supported.values()],v) for k,v in screened['contrasts'].items()}
        if 'change' in screened:screened['change']=supported_interval([f['change'] for f in supported.values()],screened['change'])
        for key in ('next_day_true_performance','next_comparable_measured_run'):
            screened[key]=supported_interval([f[key] for f in supported.values()],screened[key])
        result['supported_kernel_fit']=screened
    return result
