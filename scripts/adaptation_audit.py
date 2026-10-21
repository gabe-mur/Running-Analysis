"""Report adaptation separately from churn exemptions; never certify necessity."""

from run_analysis.projection_gate import _future_schedule_changes, _future_distance_changes


def adaptation_transitions(snapshots):
    rows = []
    for previous, current in zip(snapshots, snapshots[1:]):
        windows = {}
        for horizon in (4, 7):
            dates, types = _future_schedule_changes(previous, current, horizon_days=horizon)
            distances = _future_distance_changes(previous, current, horizon_days=horizon, tolerance_miles=0.)
            windows[str(horizon)] = dict(date_changes=dates, type_changes=types,
                                         distance_changes=distances)
        changed = any(any(x.values()) for x in windows.values())
        rows.append(dict(at=current.generated_at.isoformat(), trigger=current.trigger,
            material_evidence_reasons=list(current.material_evidence_reasons),
            recovery_surprise_units=current.recovery_surprise_units,
            continuous_distance_surprise_miles=current.continuous_distance_surprise_miles,
            short_term_distance_surprise_miles=current.short_term_distance_surprise_miles,
            windows=windows,
            interpretation=('adaptation_requires_review' if changed and current.material_evidence_reasons else
                            'change_without_new_material_evidence' if changed else 'unchanged'),
            limitation='Evidence permits adaptation; this audit does not prove each edit necessary or proportional.'))
    return rows
