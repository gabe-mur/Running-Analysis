"""Typed past issued/completed evidence carried into a fresh diagnostic runtime.

This preserves delivery credit, never adds training demand or fabricates a
completion. Uncompleted opportunities are not copied into a new experiment.
"""
from datetime import datetime

from run_analysis.delivered_training import DoseContract,DoseTarget,DeliveredDose,PurposeEvidence
from run_analysis.staged_progression import ReviewEntry
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout


def validate_seed(entries, history, origin):
    entries=tuple(entries)
    keys={r.key for r in history}
    if (len({r.key for r in entries})!=len(entries)
            or len({r.completion.key for r in entries if r.completion})!=len(entries)):
        raise ValueError('Duplicate seeded issued/completed review identity')
    for row in entries:
        done=row.completion
        if done is None or done.available_at>origin or done.end>=origin or done.key not in keys:
            raise ValueError('Seeded review completion must belong to available physical history')
        if row.purpose_evidence and row.purpose_evidence.available_at>origin:
            raise ValueError('Seeded purpose evidence must already be available')
    return entries


def completed_seed(harness, origin):
    """Carry both inherited and locally completed issue evidence into a fork.

    Copying only ``seed_review_entries`` drops uploads owned by the source
    journal. Physical history alone cannot reconstruct their issued targets.
    Uncompleted opportunities belong to the old experiment, not the new seed.
    """
    from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
    entries=tuple(row for row in review_entries(harness,origin)
        if row.completion is not None)
    return validate_seed(entries,harness.history,origin)


def phase_seed(harness, origin):
    """Keep the source journal's completed phase evidence as well as its seeds.

    A fresh database's numeric IDs must not collide with an inherited physical
    key, and current forecasts must retain already observed completion phases.
    """
    by_key={}
    for row in (*harness.phase_history,*harness.sandbox.completed_history(origin)):
        if row.available_at>origin:
            raise ValueError('Future phase evidence cannot enter a fresh experiment')
        old=by_key.get(row.key)
        if old is not None and old!=row:
            raise ValueError('Conflicting physical phase seed identity')
        by_key[row.key]=row
    completed={row.completion.key for row in completed_seed(harness,origin)}
    if not completed<=set(by_key):
        raise ValueError('Completed review needs its original phase evidence')
    return tuple(by_key[key] for key in sorted(by_key))


def read_entry(raw):
    contract=raw['contract']
    quantity=lambda value:Quantity(**value) if value is not None else None
    issued=DoseContract(**dict(contract,issued_at=datetime.fromisoformat(contract['issued_at']),
        targets=tuple(DoseTarget(r['metric'],quantity(r['quantity'])) for r in contract['targets']),
        bout_seconds=quantity(contract['bout_seconds']),bout_meters=quantity(contract['bout_meters'])))
    done=raw['completion']
    completed=DeliveredDose(**dict(done,
        **{k:datetime.fromisoformat(done[k]) for k in ('start','end','available_at')},
        distance_meters=quantity(done['distance_meters']),moving_seconds=quantity(done['moving_seconds']),
        bouts=tuple(WorkBout(**dict(b,target_zones=tuple(b['target_zones']),
            duration_range_seconds=tuple(b['duration_range_seconds']) if b['duration_range_seconds'] else None))
            for b in done['bouts']))) if done else None
    evidence=raw.get('purpose_evidence')
    purpose=PurposeEvidence(**dict(evidence,available_at=datetime.fromisoformat(evidence['available_at']))) if evidence else None
    return ReviewEntry(raw['key'],datetime.fromisoformat(raw['scheduled_at']),issued,completed,purpose)
