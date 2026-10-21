"""Explicit causal history windows shared by load forecasts and audits."""
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class QualityHistoryPolicy:
    current_days: float = 30.
    historical_days: float = 84.
    source: str = 'provisional engineering policy; frozen before predictive evaluation'

    def __post_init__(self):
        if (not self.source or not all(isfinite(x) for x in (self.current_days, self.historical_days))
                or not 0 < self.current_days <= self.historical_days):
            raise ValueError('Explicit ordered positive history windows required')
