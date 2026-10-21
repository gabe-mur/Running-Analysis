"""Audited conditional time-block sampling, not cross-day fitness accuracy."""
import numpy as np
from . import pace_comparability as pc

def robust_speed_samples(values, weights, samples):
    """Batch the existing eight-step Huber estimator, not a new estimator.

    Preserve every resampled index (including repetitions), its original
    weight, MAD floor and iteration count. Padding has zero weight and is
    excluded from medians. MAD is translation invariant, so it need only be
    calculated once per sample instead of at each iteration.
    """
    length=max(map(len,samples))
    indices=np.zeros((len(samples),length),dtype=int)
    valid=np.zeros_like(indices,dtype=bool)
    for i,sample in enumerate(samples):
        indices[i,:len(sample)]=sample
        valid[i,:len(sample)]=True
    values=np.asarray(values)[indices]
    weights=np.where(valid,np.asarray(weights)[indices],0.)
    median_values=np.where(valid,values,np.nan)
    center=np.nanmedian(median_values,axis=1,keepdims=True)
    scale=np.maximum(1e-4,1.4826*np.nanmedian(abs(median_values-center),axis=1,keepdims=True))
    cutoff=1.345*scale
    estimate=np.sum(values*weights,axis=1,keepdims=True)/weights.sum(axis=1,keepdims=True)
    for _ in range(8):
        robust=weights*np.minimum(1.,cutoff/np.maximum(abs(values-estimate),1e-12))
        estimate=np.sum(values*robust,axis=1,keepdims=True)/robust.sum(axis=1,keepdims=True)
    return estimate[:,0]


def sampling_precision(rows,weights,speeds,draws,seed,*,estimator=None,batch_estimator=None):
    minutes=np.array([r['moving_minutes_into_run'] for r in rows]);weights=np.array(weights)
    def pace(indices):
        if estimator is not None:return estimator(indices)
        return pc.speed_to_pace_min_mile(pc._robust_weighted_speed(speeds[indices].tolist(),weights[indices].tolist()))
    result=[]
    for duration in (5,10):
        labels=np.floor((minutes-minutes.min())/duration).astype(int)
        blocks=[np.flatnonzero(labels==label) for label in np.unique(labels)]
        if len(blocks)<3:continue
        rng=np.random.default_rng(seed+duration)
        indices=[np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]) for _ in range(draws)]
        if batch_estimator is not None:
            samples=batch_estimator(indices)
        elif estimator is None:
            samples=1609.344/(60*np.maximum(.1,robust_speed_samples(speeds,weights,indices)))
        else:
            samples=[pace(ix) for ix in indices]
        result.append(dict(block_minutes=duration,blocks=len(blocks),sampling_sd_min_mile=float(np.std(samples,ddof=1))))
    return pace(np.arange(len(rows))),result
