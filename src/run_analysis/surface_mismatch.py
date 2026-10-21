"""Signal-only candidates for a running surface absent from a ground DEM.

These are provisional source-selection heuristics, not bridge identification or
surveyed elevation. No coordinates, bridge catalogues or network calls occur.
"""
from dataclasses import replace
from math import isfinite

import numpy as np

from .movement import smooth_elevation


def _finite(value):
    return isinstance(value, (int, float)) and isfinite(value)


def surface_candidates(points, intervals, mapped, config):
    """Find bounded raised profiles and ground dips beneath smoother recordings.

    Uniform-distance samples keep stationary GPS fixes from dominating. Constant
    offsets are removed for comparison only, never applied to source heights.
    Missing GPS/map/altitude breaks the comparison; it cannot become an anchor.
    """
    n = len(points)
    mask = [False]*n
    if len(mapped) != n:
        raise ValueError('Each source point requires a map sample')
    tolerance = 6.  # Existing recorded-elevation engineering tolerance, metres.
    minimum_span = float(config['elevation']['minimum_grade_distance_meters'])
    smoothing = float(config['elevation']['smoothing_window_meters'])
    spacing = min(20., minimum_span/5.)
    differences = [p.altitude_m-h for p,h in zip(points,mapped)
                   if p.gps_valid and _finite(p.altitude_m) and _finite(h)]
    # Lossless rejection for this detector: a profile whose entire difference
    # range fits inside the significance tolerance cannot contain a qualifying
    # raised excursion or the required deeper map valley. Avoid smoothing it.
    if len(differences)<3 or max(differences)-min(differences)<tolerance:
        return dict(mask=mask,regions=[],assumptions=dict(vertical_tolerance_m=tolerance,
            minimum_span_m=minimum_span,sample_spacing_m=spacing,
            status='Provisional surface-mismatch candidates, not confirmed bridges'))
    by_index = {i.index: i for i in intervals}
    runs, start = [], 0
    while start < n:
        p = points[start]
        if not (p.gps_valid and _finite(p.altitude_m) and _finite(mapped[start])):
            start += 1
            continue
        stop = start+1
        while (stop < n and stop-1 in by_index and points[stop].gps_valid
               and _finite(points[stop].altitude_m) and _finite(mapped[stop])):
            stop += 1
        local = [replace(by_index[i], index=i-start) for i in range(start, stop-1)]
        distance = np.r_[0., np.cumsum([i.distance_m for i in local])]
        if distance[-1] < minimum_span:
            start = stop
            continue
        observed = smooth_elevation(points[start:stop], local, smoothing)
        ground = smooth_elevation([replace(p, altitude_m=h) for p,h in zip(points[start:stop], mapped[start:stop])], local, smoothing)
        # Aggregate repeated distance positions, rather than duplicate stopped
        # samples or interpolation through a zero-distance altitude jump.
        unique, first, counts = np.unique(distance, return_index=True, return_counts=True)
        observed, ground = np.asarray(observed), np.asarray(ground)
        raw, dem = observed[first].copy(), ground[first].copy()
        for i in np.flatnonzero(counts>1):
            raw[i] = np.median(observed[first[i]:first[i]+counts[i]])
            dem[i] = np.median(ground[first[i]:first[i]+counts[i]])
        grid = np.arange(0., unique[-1]+1e-8, spacing)
        raw = np.interp(grid, unique, raw); dem = np.interp(grid, unique, dem)
        # Do not interpolate evidence across sparsely sampled or skipped travel.
        left = np.clip(np.searchsorted(unique,grid,side='right')-1,0,len(unique)-1)
        right = np.clip(np.searchsorted(unique,grid,side='left'),0,len(unique)-1)
        valid = (unique[right]-unique[left]) <= 2*smoothing
        residual = raw-dem
        offset = float(np.median(residual[valid])) if valid.any() else 0.
        residual -= offset
        chosen = np.zeros(len(grid), dtype=bool)
        candidates = []

        # A coherent raised excursion bounded by ordinary-ground agreement.
        # Broad GPS drift is not evidence that the DEM omitted a bridge.
        elevated = valid & (residual > tolerance/2)
        # A map can include a building/tower spike next to its riverbed dip.
        # Keep the whole bounded disagreement together instead of treating the
        # zero-crossing between two DEM errors as an ordinary-ground anchor.
        excursion = valid & (abs(residual) > tolerance/2)
        edges = np.flatnonzero(np.diff(np.r_[False,excursion,False]))
        groups = list(zip(edges[::2], edges[1::2]))
        outside = valid & ~elevated
        background_ok = (outside.sum() >= 2*minimum_span/spacing
                         and np.percentile(abs(residual[outside]),90) <= tolerance)
        for a,b in groups:
            if not background_ok or a==0 or b==len(grid):
                continue
            lo,hi = a-1,b
            if grid[hi]-grid[lo] < minimum_span or max(residual[a:b]) < tolerance:
                continue
            flank = max(1, int(np.ceil(minimum_span/spacing)))
            before, after = residual[max(0,a-flank):a], residual[b:min(len(grid),b+flank)]
            if (len(before)<flank or len(after)<flank
                    or not valid[max(0,a-flank):min(len(grid),b+flank)].all()
                    or abs(np.median(before))>tolerance/2 or abs(np.median(after))>tolerance/2):
                continue
            chosen[lo:hi+1] = True
            candidates.append(dict(kind='bounded_raised_profile', start_m=float(grid[lo]), end_m=float(grid[hi])))

        # A local map valley with a substantially smoother recorded surface.
        # This can work after a missing start, without a whole-run quality veto.
        half = max(2, int(np.ceil(minimum_span/spacing)))
        for center in range(half, len(grid)-half):
            a,b = center-half,center+half+1
            if not valid[a:b].all():
                continue
            drop, rise = dem[a]-dem[center], dem[b-1]-dem[center]
            if (min(drop,rise) >= tolerance and np.ptp(raw[a:b]) <= tolerance
                    and np.ptp(dem[a:b]) >= 2*tolerance):
                chosen[a:b] = True
                candidates.append(dict(kind='ground_dip_below_smoother_recording', start_m=float(grid[a]), end_m=float(grid[b-1])))
        # A recorded candidate must itself have plausible grades at the same
        # distance scale, not merely have a smaller range than a broken DEM.
        step = max(1,int(np.ceil(smoothing/spacing)))
        plausible = np.ones(len(grid),dtype=bool)
        for i in range(len(grid)):
            a,b = max(0,i-step),min(len(grid)-1,i+step)
            if grid[b]>grid[a] and abs(raw[b]-raw[a])*100/(grid[b]-grid[a]) > float(config['elevation']['maximum_plausible_grade_percent']):
                plausible[i] = False
        chosen &= plausible & valid
        nearest = np.clip(np.searchsorted(grid,distance,side='left'),0,len(grid)-1)
        for i, take in enumerate(chosen[nearest]):
            mask[start+i] = bool(take)
        runs.append(dict(start_index=start,end_index=stop-1,comparison_offset_m=offset,
                         candidates=candidates,selected_points=int(sum(mask[start:stop]))))
        start = stop
    return dict(mask=mask, regions=runs, assumptions=dict(vertical_tolerance_m=tolerance,
        minimum_span_m=minimum_span,sample_spacing_m=spacing,
        status='Provisional surface-mismatch candidates, not confirmed bridges'))
