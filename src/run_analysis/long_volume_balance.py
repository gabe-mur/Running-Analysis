"""Explicit coaching balance prior, not a fatigue score or physiological limit.

Inputs are sustained, accepted miles per seven days and a single long-run dose.
No calendar, rolling-window mileage, run count, quality dose or athlete identity
is used. The curve is a progression guide; it cannot shrink an issued long run.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from math import isfinite


@dataclass(frozen=True)
class LongVolumeBalancePolicy:
    low_volume_share: float = .45
    high_volume_share: float = .20
    anchor_volume: float = 80.
    anchor_share: float = .22
    shape: float = 4.
    source: str = 'AS118 provisional coaching curve; not fitted athlete outcomes'

    def __post_init__(self):
        values = (self.low_volume_share, self.high_volume_share,
                  self.anchor_volume, self.anchor_share, self.shape)
        if (any(not isfinite(v) for v in values)
                or not 0 < self.high_volume_share < self.anchor_share < self.low_volume_share < 1
                or self.anchor_volume <= 0 or not 1 <= self.shape <= 8 or not self.source):
            raise ValueError('Ordered finite shares, positive anchor and explicit curve provenance required')

    @property
    def transition_volume(self):
        p = self.shape
        low, high, anchor = (v**p for v in
                             (self.low_volume_share, self.high_volume_share, self.anchor_share))
        return self.anchor_volume*((anchor-high)/(low-anchor))**(1/p)

    def share(self, volume):
        if not isfinite(volume) or volume < 0:
            raise ValueError('Finite nonnegative sustained volume required')
        p = self.shape
        # Stable reciprocal form for large volumes; no unbounded V**p.
        ratio = volume/self.transition_volume
        fraction = (1/(1+ratio**p) if ratio <= 1 else
                    (1/ratio)**p/(1+(1/ratio)**p))
        return (self.high_volume_share**p+
                (self.low_volume_share**p-self.high_volume_share**p)*fraction)**(1/p)

    def long_at_volume(self, volume):
        """Smooth strictly increasing dose even though its share decreases.

        L**p = high**p * V**p + (low**p-high**p)*V**p/(1+(V/k)**p).
        Both terms increase: interpolation cannot create a shrinking long cap.
        """
        return volume*self.share(volume)

    def committed_long_at_volume(self, volume):
        return float(Decimal(str(self.long_at_volume(volume)+1e-10)).quantize(
            Decimal('.1'), rounding=ROUND_FLOOR))

    def volume_for_long(self, long_miles):
        """Small deterministic inverse, rounded UP to a supported decimal target."""
        if not isfinite(long_miles) or long_miles <= 0:
            raise ValueError('Finite positive long-run dose required')
        left, right = long_miles/self.low_volume_share, long_miles/self.high_volume_share
        for _ in range(64):
            middle = (left+right)/2
            if self.long_at_volume(middle) < long_miles:
                left = middle
            else:
                right = middle
        return float(Decimal(str(right-1e-10)).quantize(Decimal('.1'), rounding=ROUND_CEILING))


def propose_balance_phase(stage, *, balance=None):
    """Configure the next OFFLINE phase without accepting or resetting a stage.

    Existing bounded easy growth can reach the inverse-curve endpoint gradually.
    After it is funded, the existing reviewed coordinated-dose branch can use
    this same curve. That branch remains a step proposal, NOT a new gradual
    long-run ramp. Calendar acceptance, target history and restart still belong
    to the existing lifecycle; this function neither writes nor authorizes runs.
    """
    from dataclasses import replace
    balance = balance or LongVolumeBalancePolicy()
    needed = balance.volume_for_long(stage.long_miles)
    hold = stage.weekly_equivalent_miles+1e-8 < needed
    policy = replace(stage.policy, balance=balance,
        growth_focus='easy_volume' if hold else 'long_and_volume',
        volume_goal_miles=needed if hold else None,
        volume_path='gradual' if hold else 'step')
    return dict(status='supporting_volume_rebalance' if hold else 'coordinated_balance_development',
                policy=policy, required_volume_per_7=needed,
                stage_target_unchanged=stage, long_target_unchanged=stage.long_miles,
                accepted=False, requires_funded_calendar_and_recovery=True)
