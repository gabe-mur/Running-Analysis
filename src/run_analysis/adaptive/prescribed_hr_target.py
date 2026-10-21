"""Offline prescription semantics, independent of measured execution or errors.

These are authored interpretation conventions, not learned physiology. Existing
full-envelope exposure sampling and live forecast APIs are deliberately unchanged.
"""
from dataclasses import dataclass
import re

from run_analysis.training_signals import HeartRateReference

VERSION = 'prescribed_hr_target_v1'
_ZONE = re.compile(r'(?:(low(?:er)?|upper|high|bottom|top|mid(?:dle)?)\s+)?\bZ([1-5])\b', re.I)


@dataclass(frozen=True)
class PrescribedHRTarget:
    low: float
    nominal: float
    high: float
    bpm: tuple[float, float, float]
    cues: tuple[str, ...]
    basis: str


def prescribed_hr_target(cues, config):
    """Unqualified range center; qualified half center or literal edge.

    Low/lower and upper/high narrow a zone to its lower or upper half. Bottom
    and top request the literal edge, with the same half as a sensitivity band.
    Adjacent upper Zk / low Z(k+1) requests their boundary (mid-gap if integer
    zone settings leave a one-BPM gap). Plain multi-zone cues keep their envelope;
    they do not imply endpoint order. Progression order belongs to Effort.
    """
    cues = tuple(cues)
    parsed = set()
    for cue in cues:
        matches = list(_ZONE.finditer(cue))
        if not matches:
            return None  # Do not silently drop an unresolved cue.
        parsed.update(((m.group(1) or '').lower(), int(m.group(2))) for m in matches)
    if not parsed:
        return None
    bands, targets = [], []
    for qualifier, zone in sorted(parsed):
        lo, hi = map(float, config['zones']['z'+str(zone)])
        mid = (lo+hi)/2
        if qualifier in {'low', 'lower', 'bottom'}:
            band = lo, mid
            target = lo if qualifier == 'bottom' else (lo+mid)/2
        elif qualifier in {'upper', 'high', 'top'}:
            band = mid, hi
            target = hi if qualifier == 'top' else (mid+hi)/2
        else:
            band = (mid, mid) if qualifier in {'mid', 'middle'} else (lo, hi)
            target = mid
        bands.append(band)
        targets.append(target)
    low, high = min(b[0] for b in bands), max(b[1] for b in bands)
    boundary = None
    if len(parsed) == 2:
        ordered = sorted(parsed, key=lambda p:p[1])
        (qa, za), (qb, zb) = ordered
        if qa in {'upper', 'high', 'top'} and qb in {'low', 'lower', 'bottom'} and zb == za+1:
            boundary = (config['zones']['z'+str(za)][1]+config['zones']['z'+str(zb)][0])/2
    nominal = boundary if boundary is not None else targets[0] if len(parsed) == 1 else (low+high)/2
    if not low <= nominal <= high:
        raise ValueError('Prescription target must lie within its semantic band')
    reference = HeartRateReference(config['resting_hr'], config['max_hr'], VERSION)
    normalized = tuple(reference.reserve(v) for v in (low, nominal, high))
    if any(v is None for v in normalized):
        raise ValueError('Prescribed HR target must fit athlete normalization')
    return PrescribedHRTarget(*normalized, (low, nominal, high), cues,
        'adjacent upper/low zone boundary' if boundary is not None else
        'literal zone edge' if len(parsed) == 1 and next(iter(parsed))[0] in {'bottom','top'} else
        'midpoint of prescribed zone or qualified range')
