"""Offline bridge from frozen native prescriptions to the existing load predictor.

No recipe-specific multipliers, purpose credit, admission or calendar selection.
Pace resolves distance-command duration; authored HR cues resolve work exposure.
The raw-speed response predictor is deliberately not used here.
"""
from math import fsum

from .quality_block_contract import BlockPrescription, block_scenario
from .quality_prescription_contract import QualityPrescription
from .requested_quality_forecast import command_scenario
from .unsupported_effort_transfer import forecast_extrapolated_work
from .work_exposure_matching import forecast_work_for_ranking, requested_work_bands


POINTS = ('lower_speed', 'nominal', 'higher_speed')


def prescription_query(prescription, config, *, point='nominal'):
    """Complete contiguous clock and per-phase cues, never inferred from family."""
    if isinstance(prescription, BlockPrescription):
        phases = block_scenario(prescription, point=point)
    elif isinstance(prescription, QualityPrescription):
        phases = command_scenario(prescription, point=point)
    else:
        raise TypeError('Frozen v2 or block prescription required')
    clock = 0.
    layout = []
    for p, compiled in zip(phases, prescription.phases):
        # Nominal is already committed by the compiler. Reconstructing it via
        # reciprocal speed introduces tiny floating changes (notably portions)
        # and needlessly alters supported forecast identities.
        end = clock+(compiled.seconds.estimate if point=='nominal' else p.seconds)
        layout.append((p.kind, clock, end))
        clock = end
    return tuple(layout), requested_work_bands(prescription.phases, config)


def forecast_prescription_point(prescription, history, profiles, config, *, reference,
                                anchors=(), response=None, point='nominal'):
    layout, requested = prescription_query(prescription, config, point=point)
    kwargs = dict(as_of=prescription.request.issued_at, accounting_key=reference.accounting_key)
    # Missing extrapolation assumptions must not turn a diagnostic timing proxy
    # into a usable score, nor prevent an already-supported forecast.
    if response is None or not anchors:
        return forecast_work_for_ranking(layout, history, profiles, requested, **kwargs)
    return forecast_extrapolated_work(layout, history, profiles, requested,
        anchors=anchors, response=response, reference=reference, **kwargs)


def forecast_prescription(prescription, history, profiles, config, *, reference,
                          anchors=(), response=None):
    history, anchors = tuple(history), tuple(anchors)
    scenarios = {p:forecast_prescription_point(prescription, history, profiles, config,
        reference=reference, anchors=anchors, response=response, point=p) for p in POINTS}
    complete = all(f['status']=='predicted' for f in scenarios.values())
    costs = [f['predicted_load'] for f in scenarios.values()] if complete else []
    for f in scenarios.values():
        if f['status']=='predicted' and abs(f['predicted_load']-fsum(p['load'] for p in f['phases']))>1e-8:
            raise ValueError('Whole-session forecast must equal all compiled phase costs')
    return dict(version='quality_catalogue_forecast_v1', prescription_key=prescription.key,
        issued_at=prescription.request.issued_at, nominal=scenarios['nominal'], scenarios=scenarios,
        all_scenarios_predicted=complete,
        scenario_envelope=dict(low=min(costs), high=max(costs), nominal=costs[1],
            basis='Three execution scenarios only; not confidence or exhaustive bounds') if complete else None,
        uncertainty_calibrated=False, planner_influence=False, purpose_credit_assigned=False)
