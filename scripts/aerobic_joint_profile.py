"""Offline Gaussian change profiles, not imported by production.

Each proposed change refits all period day amplitudes and searches the same
fixed correlation grid. Measurement and shared calibration covariance stay
explicit. Both ordinary ML and a likelihood of nuisance-mean residuals are
available; neither has an assumed finite-sample 95% coverage guarantee.
"""
import math
from statistics import NormalDist

import numpy as np

from scripts.aerobic_serial_model import kernel_grid


LR_95=NormalDist().inv_cdf(.975)**2


def minimize_positive(function,start,*,max_iterations=120,tolerance=2e-5):
    """Projected BFGS with analytic gradients and monotone Armijo search.

    Coordinates are scaled SDs, not log variances, so zero is a real boundary.
    Failed convergence is returned explicitly, never silently certified.
    """
    x=np.maximum(np.asarray(start,float),0.);f,g=function(x);h=np.eye(len(x))*.1;calls=1
    for iteration in range(max_iterations):
        projected=g.copy();projected[(x<=1e-10)&(g>0)]=0.
        if np.max(np.abs(projected))<=tolerance:
            return dict(x=x,value=f,gradient=g,converged=True,iterations=iteration,calls=calls)
        direction=-h@projected;direction[(x<=1e-10)&(direction<0)]=0.
        if float(g@direction)>=-1e-12:
            h=np.eye(len(x))*.1;direction=-projected*.1
        accepted=False;step=1.
        for _ in range(30):
            candidate=np.maximum(x+step*direction,0.);delta=candidate-x
            nf,ng=function(candidate);calls+=1
            if np.isfinite(nf) and nf<=f+1e-4*float(g@delta):accepted=True;break
            step*=.5
        if not accepted:break
        s=candidate-x;z=ng-g;sz=float(s@z)
        if sz>1e-12*max(1.,float(np.linalg.norm(s)*np.linalg.norm(z))):
            rho=1/sz;identity=np.eye(len(x));left=identity-rho*np.outer(s,z)
            h=left@h@left.T+rho*np.outer(s,s)
        else:h=np.eye(len(x))*.1
        x,f,g=candidate,nf,ng
    projected=g.copy();projected[(x<=1e-10)&(g>0)]=0.
    return dict(x=x,value=f,gradient=g,converged=bool(np.max(np.abs(projected))<=tolerance),
        iterations=iteration+1,calls=calls)


def polish_coordinates(function,start):
    """Rare numerical fallback after BFGS stalls at a near-zero amplitude."""
    x=np.asarray(start,float).copy();calls=0;gold=(math.sqrt(5)-1)/2
    for _ in range(8):
        initial=function(x)[0];calls+=1
        for i in range(len(x)):
            def scalar(value):
                nonlocal calls
                candidate=x.copy();candidate[i]=value;calls+=1
                return function(candidate)[0]
            lo=0.;hi=max(2.,2*x[i]);previous=scalar(hi/2);current=scalar(hi)
            for _ in range(16):
                if current>=previous:break
                hi*=2;previous,current=current,scalar(hi)
            else:raise ValueError('Coordinate optimum not bounded')
            left=hi-gold*(hi-lo);right=lo+gold*(hi-lo);fl=scalar(left);fr=scalar(right)
            for _ in range(35):
                if fl<fr:
                    hi,right,fr=right,left,fl;left=hi-gold*(hi-lo);fl=scalar(left)
                else:
                    lo,left,fl=left,right,fr;right=lo+gold*(hi-lo);fr=scalar(right)
            choices=[(x[i],scalar(x[i])),(0.,scalar(0.)),(left,fl),(right,fr)]
            x[i]=min(choices,key=lambda pair:pair[1])[0]
        if initial-function(x)[0]<1e-10:break
    result=minimize_positive(function,x,max_iterations=480,tolerance=1e-4)
    result['calls']+=calls
    return result


class GaussianChangeProfile:
    def __init__(self,y,m,g,c,groups,*,times,future_time,mode='ml'):
        y=np.asarray(y,float);m=np.asarray(m,float);g=np.asarray(g,float);c=np.asarray(c,float)
        groups=np.asarray(groups,int);n=len(y);p=int(groups.max())+1 if n else 0
        if (y.shape!=(n,) or m.shape!=(n,) or groups.shape!=(n,) or n<=p or set(groups)!=set(range(p))
            or g.ndim!=2 or g.shape[0]!=n or c.shape!=(g.shape[1],g.shape[1])
            or not all(np.isfinite(v).all() for v in (y,m,g,c)) or np.any(m<=0)
            or not np.allclose(c,c.T) or np.linalg.eigvalsh(c).min() < -1e-10):
            raise ValueError('Finite aligned data, PSD calibration covariance, positive measurement variance and residual df required')
        if mode not in ('ml','nuisance_residual'):raise ValueError('Unknown profile mode')
        self.center=float(y.mean());self.y=y-self.center;self.groups=groups;self.x=np.eye(p)[groups]
        self.known=np.diag(m)+g@c@g.T;self.mode=mode;self.p=p;self.n=n
        self.template=dict(m=m.copy(),g=g.copy(),c=c.copy(),groups=groups.copy(),times=np.asarray(times,float).copy(),future_time=future_time)
        self.kernels={k:r[:-1,:-1] for k,r in kernel_grid(times,future_time).items()}
        if any(r.shape!=(n,n) for r in self.kernels.values()):raise ValueError('Dates must align with observations')
        residual=self.y-self.x@np.linalg.lstsq(self.x,self.y,rcond=None)[0]
        self.scale=math.sqrt(float(np.mean(residual**2)+np.median(m)))
        self.initial=np.array([math.sqrt(max(float(np.mean(residual[groups==i]**2)-np.median(m[groups==i])),float(np.median(m[groups==i]))*.1))/self.scale for i in range(p)])

    def geometry(self,contrast):
        contrast=np.asarray(contrast,float)
        if contrast.shape!=(self.p,) or not np.isfinite(contrast).all() or not np.any(contrast):
            raise ValueError('Finite nonzero aligned contrast required')
        # Fixed, covariance-independent nuisance basis: projecting it away
        # yields a proper Gaussian residual likelihood retaining the contrast.
        null=np.linalg.svd(contrast[None,:],full_matrices=True)[2][1:].T
        return contrast,self.x@null

    def objective(self,kernel,contrast,*,value=None):
        contrast,z=self.geometry(contrast);r=self.kernels[kernel]
        offset=float(contrast.sum())*self.center
        target=None if value is None else float(value)-offset
        if target is not None and not math.isfinite(target):raise ValueError('Finite proposed change required')
        def evaluate(theta,details=False):
            sd=np.asarray(theta)*self.scale;observed_sd=sd[self.groups]
            v=self.known+observed_sd[:,None]*r*observed_sd
            sign,logdet=np.linalg.slogdet(v)
            if sign<=0:raise ValueError('Nonpositive fitted covariance')
            inv=np.linalg.inv(v);inv_x=inv@self.x;b=np.linalg.inv(self.x.T@inv_x)
            beta=b@inv_x.T@self.y
            if target is not None:
                direction=b@contrast;beta=beta+direction*(target-float(contrast@beta))/float(contrast@direction)
            residual=self.y-self.x@beta;u=inv@residual;trace_precision=inv
            objective=float(logdet+residual@u)
            if self.mode=='nuisance_residual' and z.shape[1]:
                inv_z=inv@z;metric=z.T@inv_z
                objective+=float(np.linalg.slogdet(metric)[1])
                trace_precision=inv-inv_z@np.linalg.solve(metric,inv_z.T)
            local=2*((trace_precision*r)@observed_sd-u*(r@(observed_sd*u)))
            gradient=np.bincount(self.groups,weights=local,minlength=self.p)*self.scale
            if details:return dict(objective=objective,beta=(beta+self.center).tolist(),
                estimate=float(contrast@beta)+offset,period_day_log_sd=sd.tolist(),
                conditional_se=math.sqrt(float(contrast@b@contrast)),gradient=gradient)
            return objective,gradient
        return evaluate

    def fit_kernel(self,kernel,contrast,*,value=None,warm=None):
        function=self.objective(kernel,contrast,value=value)
        starts=[self.initial,np.ones(self.p)] if warm is None else [np.asarray(warm)/self.scale,self.initial]
        # Flat-at-zero SD gradients cannot rule out a better interior mode.
        candidates=[minimize_positive(function,start) for start in starts]
        candidates.append(dict(x=np.zeros(self.p),value=function(np.zeros(self.p))[0],converged=True,calls=1,iterations=0))
        best=min(candidates,key=lambda fit:fit['value'])
        if not best['converged']:
            # A second numerical solve is independent of outcomes/coverage.
            # Near-zero amplitudes can stall the first inverse-Hessian history.
            refined=minimize_positive(function,best['x'],max_iterations=480,tolerance=1e-4)
            candidates.append(refined)
            if refined['value']<=best['value']+1e-9:best=refined
        if not best['converged']:
            refined=polish_coordinates(function,best['x']);candidates.append(refined)
            if refined['value']<=best['value']+1e-9:best=refined
        details=function(best['x'],details=True)
        return dict({k:v for k,v in details.items() if k!='gradient'},kernel=kernel,
            converged=best['converged'],calls=sum(f['calls'] for f in candidates),iterations=best['iterations'])

    def fit(self,contrast,*,value=None,warm=None):
        fits={k:self.fit_kernel(k,contrast,value=value,warm=None if warm is None else warm[k]['period_day_log_sd']) for k in self.kernels}
        selected=min(fits,key=lambda k:fits[k]['objective'])
        return dict(fits[selected],kernel_fits=fits,unconverged_kernels=[k for k,f in fits.items() if not f['converged']])

    def likelihood_ratio(self,contrast,value,*,free=None,return_fits=False):
        free=self.fit(contrast) if free is None else free
        constrained=self.fit(contrast,value=value,warm=free['kernel_fits'])
        nesting_refits=0
        # A constrained optimum is a valid start for the larger model. Check
        # nesting per kernel, not merely the two selected best objective values.
        # Keep every draw; do not clip away failed fits or negative LR samples.
        for _ in range(4):
            repair=[k for k in self.kernels if constrained['kernel_fits'][k]['objective']<free['kernel_fits'][k]['objective']-1e-7]
            if not repair:break
            fits=dict(free['kernel_fits'])
            for k in repair:
                candidate=self.fit_kernel(k,contrast,warm=constrained['kernel_fits'][k]['period_day_log_sd'])
                if candidate['objective']<fits[k]['objective']:fits[k]=candidate
                nesting_refits+=1
            selected=min(fits,key=lambda k:fits[k]['objective'])
            free.clear();free.update(dict(fits[selected],kernel_fits=fits,
                unconverged_kernels=[k for k,f in fits.items() if not f['converged']]))
            constrained=self.fit(contrast,value=value,warm=fits)
        statistic=constrained['objective']-free['objective']
        if statistic < -1e-5:raise ValueError('Constrained likelihood improved on free optimum; optimizer audit failed')
        result=dict(statistic=max(0.,statistic),covered_95=statistic<=LR_95,
            estimate=free['estimate'],free_kernel=free['kernel'],constrained_kernel=constrained['kernel'],
            unconverged_kernels=sorted(set(free['unconverged_kernels']+constrained['unconverged_kernels'])),
            constrained_day_log_sd=constrained['period_day_log_sd'],nesting_refits=nesting_refits)
        if return_fits:result.update(null_fit=constrained,free_fit=free)
        return result

    def bootstrap_test(self,contrast,value,*,draws=127,seed=20261103,free=None,decision_only=False):
        """Plug-in parametric-bootstrap LR test at one proposed change.

        Simulate under the constrained fitted covariance and mean, then refit
        both hypotheses, including all kernels, in every replicate. The +1
        Monte Carlo p-value avoids zero p-values. This remains conditional on
        a fitted Gaussian null; coverage is not guaranteed under misspecification.
        """
        if draws<19:raise ValueError('At least 19 bootstrap draws required for a 5% Monte Carlo test')
        observed=self.likelihood_ratio(contrast,value,free=free,return_fits=True)
        null=observed['null_fit'];sd=np.array(null['period_day_log_sd'])[self.groups]
        covariance=self.known+sd[:,None]*self.kernels[null['kernel']]*sd
        mean=self.x@np.array(null['beta']);factor=np.linalg.cholesky(covariance)
        rng=np.random.default_rng(seed);statistics=[];warnings=0;warning_examples=[];nesting_refits=observed['nesting_refits'];exceedances=0
        for _ in range(draws):
            y=mean+factor@rng.normal(size=self.n)
            model=GaussianChangeProfile(y,mode=self.mode,**self.template)
            model.kernels=self.kernels
            result=model.likelihood_ratio(contrast,value)
            statistics.append(result['statistic']);warnings+=bool(result['unconverged_kernels'])
            if result['unconverged_kernels'] and len(warning_examples)<1:
                warning_examples.append(dict(draw=len(statistics)-1,y=y.tolist(),result=result))
            nesting_refits+=result['nesting_refits']
            exceedances+=result['statistic']>=observed['statistic']
            lower=(1+exceedances)/(draws+1);upper=(1+exceedances+draws-len(statistics))/(draws+1)
            if decision_only and (lower>.05 or upper<=.05):break
        complete=len(statistics)==draws
        pvalue=(1+exceedances)/(draws+1) if complete else None
        # This quantile is descriptive; accept/reject uses the exact stated
        # finite Monte Carlo rank rule above, not quantile interpolation.
        return dict(estimate=observed['estimate'],statistic=observed['statistic'],pvalue=pvalue,
            covered_95=lower>.05,bootstrap_draws=draws,bootstrap_seed=seed,evaluated_draws=len(statistics),
            pvalue_bounds=[lower,upper],decision_only=decision_only,
            bootstrap_statistic_95=float(np.quantile(statistics,.95)) if complete else None,
            optimizer_warning_draws=warnings,observed_unconverged_kernels=observed['unconverged_kernels'],
            null_kernel=null['kernel'],null_day_log_sd=null['period_day_log_sd'],mode=self.mode,nesting_refits=nesting_refits,
            numerical_warning_examples=warning_examples)

    def interval(self,contrast,*,free=None,iterations=20):
        free=self.fit(contrast) if free is None else free;evaluations=[]
        class RestartAnchor(Exception):pass
        for restart in range(5):
            center=free['estimate'];reference=free['objective']
            def difference(value):
                result=self.likelihood_ratio(contrast,value,free=free);evaluations.append(dict(value=value,**result))
                if result['unconverged_kernels']:raise ValueError('Profile optimizer did not converge')
                # A nesting repair can improve the free optimum during root
                # finding. Earlier endpoints then used an obsolete LR baseline,
                # even if its point happened not to move. Recompute both sides.
                if free['objective']<reference-1e-7:raise RestartAnchor
                return result['statistic']-LR_95
            try:
                bounds=[]
                for sign in (-1,1):
                    inside=center;step=max(free['conditional_se']*2,.001);outside=center+sign*step
                    for _ in range(20):
                        if difference(outside)>0:break
                        step*=2;outside=center+sign*step
                    else:raise ValueError('Profile endpoint not bracketed; no finite interval claimed')
                    for _ in range(iterations):
                        middle=(inside+outside)/2
                        if difference(middle)>0:outside=middle
                        else:inside=middle
                    bounds.append((inside+outside)/2)
            except RestartAnchor:
                continue
            return dict(estimate=center,range_95=bounds,mode=self.mode,critical=LR_95,
                interval_kind='Gaussian profile likelihood, asymptotic chi-square reference; finite-sample coverage tested separately',
                selected_kernel=free['kernel'],period_day_log_sd=free['period_day_log_sd'],
                anchor_restarts=restart,evaluations=evaluations)
        raise ValueError('Free profile optimum did not settle; no interval claimed')
