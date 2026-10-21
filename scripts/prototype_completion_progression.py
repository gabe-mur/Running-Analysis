"""Falsification-first completion-credit experiment; NOT a runtime policy.

An extra mile relative to a prescription is adherence surplus, not necessarily
an increase above the runner's previous total training. A genuine growth claim
needs an independently frozen, conserved no-growth allocation. This experiment
keeps provisional credit separate from irreversible progression authorization.
It deliberately has no planner, state publisher, medical model or time window.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path
import json
import sqlite3

from run_analysis.athlete_projection import aware
from scripts.research_manifest import freeze_sources


@dataclass(frozen=True)
class Allocation:
    key: str
    frozen_at: datetime
    baseline_miles: float
    proposed_miles: float
    baseline_source: str

    def __post_init__(self):
        aware(self.frozen_at)
        if (not self.key or not self.baseline_source
                or any(not isfinite(v) or v < 0 for v in
                       (self.baseline_miles, self.proposed_miles))):
            raise ValueError('Known frozen nonnegative allocation and baseline provenance required')


@dataclass(frozen=True)
class Resolution:
    allocation_key: str
    known_at: datetime
    physical_key: str | None
    actual_low: float
    actual_high: float

    def __post_init__(self):
        aware(self.known_at)
        if (not self.allocation_key
                or not all(isfinite(v) for v in (self.actual_low, self.actual_high))
                or not 0 <= self.actual_low <= self.actual_high
                or self.physical_key is None and self.actual_high != 0
                or self.physical_key == ''):
            raise ValueError('Known actual bounds, or explicit zero-distance omission, required')


def reconcile(allocations, resolutions, *, as_of):
    """Conserve signed volume; never clip each run or turn a forecast into credit.

    `settled` concerns the supplied allocation identities, not days elapsed.
    Waiting for an entire arbitrary allocation group would STILL be a gate;
    therefore neither settled nor provisional credit authorizes a live target.
    """
    aware(as_of)
    allocations=tuple(allocations); resolutions=tuple(resolutions)
    keys=[a.key for a in allocations]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate allocation identity')
    if len({r.allocation_key for r in resolutions})!=len(resolutions):
        raise ValueError('Resolve each allocation once; corrections require reconciliation')
    physical=[r.physical_key for r in resolutions if r.physical_key is not None]
    if len(physical)!=len(set(physical)):
        raise ValueError('Physical completion counted twice')
    if any(r.allocation_key not in keys for r in resolutions):
        raise ValueError('Unknown frozen allocation')
    all_allocations={a.key:a for a in allocations}
    if any(r.known_at<all_allocations[r.allocation_key].frozen_at for r in resolutions):
        raise ValueError('A retrospective allocation is not original growth intent')
    known={r.allocation_key:r for r in resolutions if r.known_at<=as_of}
    available=[a for a in allocations if a.frozen_at<=as_of]
    rows=[]
    for a in available:
        r=known.get(a.key)
        if r is not None:
            rows.append(dict(key=a.key, baseline=a.baseline_miles, proposed=a.proposed_miles,
                actual_low=r.actual_low, actual_high=r.actual_high,
                signed_increment_low=r.actual_low-a.baseline_miles,
                signed_increment_high=r.actual_high-a.baseline_miles,
                physical_key=r.physical_key))
    baseline=sum(a.baseline_miles for a in available)
    proposed=sum(a.proposed_miles for a in available)
    low=sum(r['signed_increment_low'] for r in rows)
    high=sum(r['signed_increment_high'] for r in rows)
    requested=max(0.,proposed-baseline)
    settled=bool(available) and len(rows)==len(available)
    return dict(status='settled_accounting' if settled else 'provisional_accounting',
        baseline_miles=baseline, proposed_miles=proposed, requested_increment_miles=requested,
        actual_miles=sum(r['actual_low'] for r in rows), signed_increment_low=low,
        signed_increment_high=high, eligible_increment_low=min(requested,max(0.,low)),
        unobserved_allocations=[a.key for a in available if a.key not in known],
        all_allocations_settled=settled,
        distance_increment_delivered=bool(settled and requested>0 and low+1e-8>=requested),
        positive_only_credit=sum(max(0.,r['signed_increment_low']) for r in rows),
        rows=rows, authorized_target_change=0., future_physical_credit=0.,
        limitation='Volume attribution only; no adaptation estimate, growth-rate conversion or automatic target authorization')


def course(baseline, proposed, actual, *, scale=1., known=3):
    from datetime import timedelta, timezone
    issued=datetime(2026,10,1,tzinfo=timezone.utc)
    allocations=tuple(Allocation(str(i),issued,b*scale,p*scale,'explicit synthetic no-growth allocation')
        for i,(b,p) in enumerate(zip(baseline,proposed,strict=True)))
    receipts=tuple(Resolution(str(i),issued+timedelta(days=i+1),
        str(i) if a is not None else None,(a or 0)*scale,(a or 0)*scale)
        for i,a in enumerate(actual))
    return reconcile(allocations,receipts,as_of=issued+timedelta(days=known))


def controls():
    cases={}
    for volume in (5.,20.,60.):
        scale=volume/20
        cases[f'{volume:g}_perfect_increase']=course([8,6,6],[8,6.4,6.4],[8,6.4,6.4],scale=scale)
        cases[f'{volume:g}_flat_actual']=course([8,6,6],[8,6.4,6.4],[8,6,6],scale=scale)
        cases[f'{volume:g}_redistribution']=course([8,6,6],[10,4,6],[10,4,6],scale=scale)
    cases['early_surplus_open']=course([8,6,6],[8,6.4,6.4],[10,4,6],known=1)
    cases['early_surplus_then_flat']=course([8,6,6],[8,6.4,6.4],[10,4,6])
    cases['overshoot_offsets_omission']=course([8,6,6],[8,6.4,6.4],[8,12.8,None])
    # Splitting reference and completion into more runs must not add volume.
    cases['six_runs_same_volume']=course([4,4,3,3,3,3],[4,4,3.2,3.2,3.2,3.2],
        [4,4,3,3,3,3],known=6)
    return cases


def real_adherence(source):
    """Original issued identities for the previously discussed descriptive audit.

    Oct 5–18 selects this report, NOT an activation/progression window. Existing
    issues have no frozen no-growth allocations: do not fabricate them now.
    """
    source=Path(source)
    from zoneinfo import ZoneInfo
    zone=ZoneInfo('America/New_York')
    db=sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        rows=[]
        for row in db.execute('SELECT c.session_key,c.accounted,i.serialized FROM sandbox_completions c '
                              'JOIN sandbox_issues i ON i.fingerprint=c.fingerprint'):
            actual=json.loads(row['accounted']); intent=json.loads(row['serialized'])['intent']
            day=datetime.fromisoformat(actual['start']).astimezone(zone).date().isoformat()
            if '2026-10-05'<=day<='2026-10-18':
                rows.append(dict(day=day,key=actual['key'],actual=actual['miles'],
                    prescribed=intent['distance_miles']['target']))
        previous=json.loads((source/'decision-0032.json').read_text())
        missed=previous['active']['2026-10-18']
        saved=db.execute('SELECT serialized FROM sandbox_issues WHERE fingerprint=?',
            (missed['fingerprint'],)).fetchone()
        omitted=json.loads(saved[0])['intent']['distance_miles']['target']
        actual=sum(r['actual'] for r in rows)
        prescribed=sum(r['prescribed'] for r in rows)+omitted
        from scripts.trial_smoothed_progression import target
        from run_analysis.stage_target_history import target_segments, target_miles_between
        start=datetime(2026,10,5,tzinfo=zone)
        end=datetime(2026,10,19,tzinfo=zone)
        records=tuple(map(target,previous['stage_runtime']['target_history']))
        segments,_=target_segments(records,as_of=datetime.fromisoformat(previous['as_of']),
            start=start.date(),end=end.date())
        dated=target_miles_between(segments,start,end,zone)
        return dict(rows=sorted(rows,key=lambda r:r['day']),actual_miles=actual,
            prescribed_miles_including_omission=prescribed,omitted_miles=omitted,
            adherence_surplus=actual-prescribed,actual_growth_credit=None,
            dated_volume_target_miles=dated,nominal_prescription_gap=dated-prescribed,
            actual_minus_dated_target=actual-dated,
            reference_decision=previous['index'],reference_available_at=previous['as_of'],
            status='original_growth_baseline_not_recorded',
            limitation='Prescribed-vs-actual distance is known; a no-growth allocation cannot be retroactively invented')
    finally:
        db.close()


def run(source, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cases=controls()
    report=dict(status='accounting_prototype_not_accepted_controller',cases=cases,
        real=real_adherence(source),live_changed=False,planner_searches=0,
        findings=[
            'Immediate irreversible extra-mile rewards fail: early surplus can become mere redistribution.',
            'Signed total-volume accounting conserves credit and preserves overshoots.',
            'Waiting for all allocations to settle would reintroduce a completion-group gate.',
            'Allocation grouping cannot be disguised as a windowless growth policy.',
            'Old issued receipts lack frozen no-growth allocations; adherence surplus is not established growth.',
        ])
    assert cases['early_surplus_open']['eligible_increment_low']>0
    assert cases['early_surplus_then_flat']['eligible_increment_low']==0
    assert cases['early_surplus_then_flat']['positive_only_credit']>0
    assert isclose(report['real']['adherence_surplus'],.15396037143085,abs_tol=1e-8)
    report['manifest']=freeze_sources(output/'sources.zip',config={},entrypoint=__name__,
        inputs=(Path(source)/'sandbox.sqlite',Path(source)/'decision-0032.json'))
    with (output/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(dict(status=report['status'],cases=len(cases),real=report['real'],
        findings=report['findings'],planner_searches=0,live_changed=False)),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.output)
