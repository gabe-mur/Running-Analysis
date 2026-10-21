"""Experimental numerical response mapping; not a production recovery predictor.

Shared by synthetic checks and the recorded-signal adapter. Coefficients remain
explicit hypotheses; see docs/numerical_workload_response_mapping.md.
"""
from dataclasses import asdict, dataclass
from math import exp, isfinite, log


def nonnegative(*values):
    if any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in values):
        raise ValueError('Finite nonnegative values required')


@dataclass(frozen=True)
class Policy:
    intensity_exponent: float = 1.92
    reserve_reference: float = 0.60
    density_memory_minutes: float = 3.0
    density_weight: float = 0.25
    fast_reference_excess: float = 0.25
    cardio_half_life_hours: float = 24.0
    fast_half_life_hours: float = 36.0
    exposure_weight: float = 0.5

    def __post_init__(self):
        nonnegative(*asdict(self).values())
        if any(v <= 0 for v in (
            self.reserve_reference, self.density_memory_minutes,
            self.fast_reference_excess, self.cardio_half_life_hours,
            self.fast_half_life_hours,
        )) or self.reserve_reference > 1:
            raise ValueError('Positive scales and reserve reference <= 1 required')


@dataclass(frozen=True)
class Phase:
    minutes: float
    hr_reserve: float
    speed_over_easy: float

    def __post_init__(self):
        nonnegative(self.minutes, self.hr_reserve, self.speed_over_easy)
        if self.hr_reserve > 1:
            raise ValueError('HR above the assumed maximum requires reference review')


@dataclass(frozen=True)
class Dose:
    minutes: float
    cardio: float
    fast: float
    density_peak: float
    response_cardio: float


@dataclass(frozen=True)
class Reference:
    daily_response_cardio: float
    daily_fast: float
    ordinary_cardio: float
    ordinary_fast: float
    ordinary_minutes: float
    cardio_residue_reference_days: float = 1/log(2)
    fast_residue_reference_days: float = 1.5/log(2)

    def __post_init__(self):
        nonnegative(*asdict(self).values())
        if min(self.ordinary_cardio, self.ordinary_fast, self.ordinary_minutes,
               self.cardio_residue_reference_days, self.fast_residue_reference_days) <= 0:
            raise ValueError('Named positive references required; unknown is not zero')


def rate(reserve, policy=Policy()):
    nonnegative(reserve)
    if reserve > 1:
        raise ValueError('Review the HR reference')
    ref = policy.reserve_reference
    return reserve/ref * exp(policy.intensity_exponent*(reserve-ref))


def dose(phases, policy=Policy()):
    """Exact piecewise-constant integral and density recurrence, in minutes.

    A real trace needs interval integration and coverage handling. This probe
    does NOT imply constant HR is a good forecast for short prescribed reps.
    """
    cardio = fast = density = peak = minutes = 0.0
    for phase in phases:
        if phase.minutes == 0:
            continue
        intensity = rate(phase.hr_reserve, policy)
        drive = max(0., intensity-1.)
        density = drive+(density-drive)*exp(-phase.minutes/policy.density_memory_minutes)
        peak = max(peak, density)
        cardio += phase.minutes*intensity
        fast += phase.minutes*(max(0., phase.speed_over_easy-1.)/policy.fast_reference_excess)**2
        minutes += phase.minutes
    return Dose(minutes, cardio, fast, peak, cardio*(1.+policy.density_weight*peak))


def ewma(previous, load_per_day, elapsed_days, tau_days):
    """Constant load-rate segment, not an impulse/session-total convention."""
    nonnegative(previous, load_per_day, elapsed_days, tau_days)
    if tau_days == 0:
        raise ValueError('Positive time constant required')
    decay = exp(-elapsed_days/tau_days)
    return previous*decay+load_per_day*(1-decay)


def balance(atl, ctl):
    nonnegative(atl, ctl)
    return dict(atl=atl, ctl=ctl, tsb=ctl-atl, ratio=atl/ctl if ctl > 0 else None)


def easy_load(atl, ctl):
    """Next-day uniform load giving zero balance: diagnostic, never a cap."""
    nonnegative(atl, ctl)
    a, b = exp(-1/7), exp(-1/42)
    return (b*ctl-a*atl)/(b-a)


def pressure(history, proposed, reference, policy=Policy(), *, current_leg_rating=None):
    """Soft index only: no eligibility, safety, readiness percentage or rest order.

    history contains (Dose, hours since completion). Caller must enforce unique
    actual/projected ledger identity and coherent timestamps before integration.
    """
    rc = rf = 0.
    for prior, age_hours in history:
        nonnegative(age_hours)
        rc += prior.response_cardio*2**(-age_hours/policy.cardio_half_life_hours)
        rf += prior.fast*2**(-age_hours/policy.fast_half_life_hours)
    scale_c = (reference.daily_response_cardio*reference.cardio_residue_reference_days
               + reference.ordinary_cardio)
    scale_f = (reference.daily_fast*reference.fast_residue_reference_days
               + reference.ordinary_fast)
    pc = max(0., (rc+proposed.response_cardio)/scale_c-1.)**2
    pf = max(0., (rf+proposed.fast)/scale_f-1.)**2
    leg = None
    if current_leg_rating is not None:
        nonnegative(current_leg_rating)
        if current_leg_rating > 10:
            raise ValueError('Leg rating must be in [0, 10]')
        leg = (current_leg_rating/10)**2 * (
            proposed.minutes/reference.ordinary_minutes+proposed.fast/reference.ordinary_fast)
    exposure = max(pf, leg) if leg is not None else pf
    return dict(cardio_pressure=pc, fast_exposure_pressure=pf,
                reported_leg_pressure=leg, feedback_known=leg is not None,
                total=pc+policy.exposure_weight*exposure,
                cardio_reference=scale_c, fast_reference=scale_f,
                physiological_calibration=False, planner_influence=False)
