"""Opt-in sandbox calendar exposure journal; no counterfactual dose ratios.

Original dates and span membership are frozen with each prescription. Only
actual completed mileage supplies measured mass; future plans supply timing,
not physical credit. The previous fixed-dose allocator remains in source
archives, not in this replacement measurement's runtime.
"""
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, time
from math import isfinite
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.program_delivery import TargetSegment
from run_analysis.general_progression import VOLUME_GROWTH_PER_7
from .cadence_mass_growth import CalendarDose, estimate

POLICY = 'continuous_general_v7'


def initialize(h, as_of):
    previous = getattr(h, 'general_progression_state', None) or {}
    if previous.get('realized_reference'):
        config=previous['realized_reference']
        if config['version']!='calendar_mass_reference_v2':
            raise ValueError('Old ratio reference needs explicit calendar-mass migration')
        return config
    path = [p for p in h.target_history if p.available_at <= as_of][-1]
    day = h.decision(as_of)
    level = previous.get('volume', path.rate(day, 'volume_per_7'))
    support = previous.get('support', path.rate(day, 'supporting_easy_per_7'))
    if not isfinite(level) or level<=0 or not isfinite(support) or support<0:
        raise ValueError('Positive volume normalization and nonnegative support required')
    return dict(version='calendar_mass_reference_v2',baseline_key='calendar-mass:'+as_of.isoformat(),
        available_at=as_of.isoformat(),effective_on=day.isoformat(),volume=level,support=support,
        exposure_anchor='original_issued_long_dates')


def fixed_state(state, config):
    """Legacy counterfactual diagnostic only; NOT used by mass measurement."""
    ctx = state.context
    anchor = date.fromisoformat(config['effective_on'])
    def flat(segments, value):
        old = tuple(TargetSegment(r.start,min(r.end,anchor),r.miles_per_7_days)
            for r in segments if r.start < anchor)
        return (*old,TargetSegment(anchor,max(ctx.end,anchor+timedelta(days=1)),value))
    context = replace(ctx,
        policy=replace(ctx.policy,weekly_miles=config['volume']),
        easy_duration_policy=replace(ctx.easy_duration_policy,supporting_easy_miles_per_7=config['support']),
        volume_targets=flat(ctx.volume_targets,config['volume']),
        support_targets=flat(ctx.support_targets,config['support']))
    return replace(state,context=context,prospective_volume_review=None)


def allocate(state, program, physical, config, *, reference, max_full_evaluations=64, prior_events=()):
    """Freeze the selected calendar's exposure, without another dose search.

    The old function name is retained for the atomic publisher interface. No
    alternative program or candidate distance enters the volume measurement.
    Long dates delimit naturally variable calendar spans, not review gates.
    Leading/trailing incomplete coverage is declared, never padded with data.
    """
    started = perf_counter()
    if config['version']!='calendar_mass_reference_v2':
        raise ValueError('Explicit calendar-mass reference required')
    program=tuple(program); zone=ZoneInfo(state.timezone)
    anchors=[date.fromisoformat(d) for d,k in program if physical[d,k].role=='long']
    prior={r['opportunity']:r for event in prior_events
        if datetime.fromisoformat(event['as_of'])<=state.as_of
        for r in event.get('realized_reference_allocation',{}).get('allocations',())
        if r.get('basis')=='issued_calendar_mass_v2' and r.get('span_start') is not None}
    rows = []
    for day,key in program:
        if day >= state.context.hard_end.isoformat():
            continue
        run = physical[day,key]
        pair=next(((a,b) for a,b in zip(anchors,anchors[1:]) if a.isoformat()<=day<b.isoformat()),None)
        # A selectable-tail anchor supplies only an exposure boundary. Every
        # run INSIDE a measured span must have been in the hard issued horizon.
        if pair and pair[1].isoformat()>state.context.hard_end.isoformat():pair=None
        a,b=pair if pair else (None,None)
        members=tuple('slot:'+d for d,_ in program if a and a.isoformat()<=d<b.isoformat())
        # After a long is uploaded it leaves the candidate calendar. Retain
        # already declared exposure/membership, NOT a new truncated span.
        old=prior.get('slot:'+day)
        if pair is None and old and date.fromisoformat(day)<date.fromisoformat(old['span_end'][:10]):
            a=date.fromisoformat(old['span_start'][:10]); b=date.fromisoformat(old['span_end'][:10])
            members=tuple(old['span_slots'])
        row=asdict(CalendarDose('slot:'+day,run.start,state.as_of,run.miles,
            config['baseline_key'],'Original issued calendar; actual-mileage mass only',
            datetime.combine(a,time.min,zone) if a else None,
            datetime.combine(b,time.min,zone) if b else None,members))
        for field in ('scheduled_at','available_at','span_start','span_end'):
            if row[field] is not None:row[field]=row[field].isoformat()
        row['span_slots']=list(row['span_slots'])
        rows.append(row)
    return dict(version='calendar_mass_publication_v2',config=config,allocations=rows,
        source_snapshot=state.fingerprint,program=[list(p) for p in program],full_evaluations=0,
        compiled_evaluations=0,neighborhood_complete=True,budget_exhausted=False,
        seconds=perf_counter()-started,calendar_searches=0,physical_credit_added=0.,
        reference_scope='Frozen calendar exposure; no optimized no-growth doses or future physical credit')


def signal(h, entries, config, as_of, *, maximum_growth_per_7=VOLUME_GROWTH_PER_7):
    """Resolve each ORIGINAL issued fingerprint, not a moving latest target."""
    by_source = {}
    for event in h.events:
        if datetime.fromisoformat(event['as_of']) > as_of:
            continue
        block = event.get('realized_reference_allocation')
        if block is None:
            continue
        if block['config'] != config:
            raise ValueError('Frozen reference identity changed across publications')
        for raw in block['allocations']:
            source = raw['issued_fingerprint']
            value = {k:v for k,v in raw.items() if k!='issued_fingerprint'}
            for name in ('scheduled_at','available_at','span_start','span_end'):
                if isinstance(value[name],str): value[name]=datetime.fromisoformat(value[name])
            value['span_slots']=tuple(value['span_slots'])
            reference = CalendarDose(**value)
            if source in by_source and by_source[source] != reference:
                raise ValueError('Original reference allocation was repriced')
            by_source[source] = reference
    epoch = datetime.fromisoformat(config['available_at'])
    rows = tuple(e for e in entries if e.contract.issued_at >= epoch)
    refs = tuple(by_source[e.contract.source] for e in rows if e.contract.source in by_source)
    return estimate(rows,refs,as_of=as_of,zone=h.zone,baseline_volume=config['volume'],
        memory_days=h.stage.policy.review_days,maximum_growth_per_7=maximum_growth_per_7)
