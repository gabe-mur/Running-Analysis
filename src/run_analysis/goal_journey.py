"""Read-only goal presentation from an already verified planner decision.

The dotted path is arithmetic under unchanged coaching rates and future
adherence, not another search, a fitness prediction, or earned training credit.
"""
from datetime import datetime, timedelta
from math import floor, isfinite
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from .staged_progression import DevelopmentStage, ProgressionPolicy


def conditional_path(day, *, volume, long_miles, long_budget, target_volume,
                     target_long, volume_rate, long_fraction, long_days,
                     long_share, taper_start=None):
    """A bounded future-adherence scenario, NEVER earned planner credit.

    Re-evaluate long-run support against the projected volume each day. Today's
    smoothed delivery/headroom fraction belongs to the actual controller, not
    every future day: freezing it here turns normal inter-run fluctuations into
    a permanent growth penalty even when this scenario assumes future delivery.
    The coaching long-run rate applies only after projected volume supports it.
    """
    if volume_rate <= 0 or long_fraction <= 0 or long_days <= 0:
        return dict(points=[],target_date=None,support_date=None,taper=None)
    points=[];support_date=None;target_date=None;taper=None
    budget=max(long_miles,long_budget)
    for offset in range(366):
        at=day+timedelta(days=offset)
        supported=volume*long_share+1e-8 >= long_miles
        if supported and support_date is None:
            support_date=at.isoformat()
        if offset:
            volume=min(target_volume,volume*(1+volume_rate)**(1/7))
            if supported:
                budget*=(1+long_fraction)**(1/long_days)
            long_miles=max(long_miles,min(target_long,floor((min(budget,volume*long_share)+1e-8)*10)/10))
        reached=volume>=target_volume-1e-8 and long_miles>=target_long-1e-8
        if offset%7==0 or reached or at==taper_start:
            points.append(dict(date=at.isoformat(),volume_per_7=volume,long_miles=long_miles))
        if at==taper_start:
            taper=points[-1]
            if reached:target_date=at.isoformat()
            break  # No continued build ramp through a scheduled taper.
        if reached:
            target_date=at.isoformat();break
    return dict(points=points,target_date=target_date,support_date=support_date,taper=taper)


def goal_journey(state,event,*,as_of,completed,needs_refresh=False,review_required=False,requested=None):
    """All facts are sourced separately: recorded / issued / conditional."""
    raw=event.get('stage_runtime',{}).get('stage')
    progress=event.get('stage_runtime',{}).get('general_progression_state')
    unavailable=dict(available=False,reason='Create or refresh your Weekly Plan to see goal progression.')
    if not raw or not progress or 'growth_per_7' not in progress or state.as_of>as_of:
        return unavailable
    stage=DevelopmentStage(**dict(raw,issued_at=datetime.fromisoformat(raw['issued_at']),
        policy=ProgressionPolicy(**raw['policy'])))
    zone=ZoneInfo(state.timezone);today=as_of.astimezone(zone).date()
    owner=SimpleNamespace(stage=stage,planner_preferences=state.planner_preferences,
        race_goal=state.race_goal,race_event=state.context.race_event,
        race_event_history=state.context.race_event_history,decision=lambda at:today)
    from .adaptive.progression_policy import resolve
    from .adaptive.race_event import preparation_active
    policy=resolve(owner,state.as_of)
    race=state.race_goal if preparation_active(owner,today) else None
    target=policy.target
    if not all(isfinite(x) and x>0 for x in (progress['volume'],stage.long_miles,
            target.volume_per_7,target.long_miles,policy.long_share)):
        return unavailable
    current=progress['volume'];actual=progress.get('observed_volume_per_7')
    rate=progress.get('growth_per_7',0.) if progress.get('growth_allowed') else 0.
    rate=min(rate,policy.volume_growth_per_7)
    stale=needs_refresh or state.as_of.astimezone(zone).date()<today
    current_context=state.current
    health_hold=bool(current_context and current_context.state in {'health_concern','conflicting','unverified'})
    phase=race.phase(today) if race else policy.phase
    required=stage.long_miles/policy.long_share
    supporting_hold=(not progress.get('long_growth_allowed') and
        min(current,actual if actual is not None else current)+1e-8<required)
    if health_hold or review_required:
        status='review'
    elif stale:
        status='update_pending'
    elif phase in {'taper','race','post_race'}:
        status=phase
    elif stage.long_miles>=target.long_miles-1e-8 and current>=target.volume_per_7-1e-8:
        status='at_target'
    elif rate<=0:
        status='holding'
    elif supporting_hold:
        status='supporting_volume'
    elif not progress.get('long_growth_allowed'):
        status='long_hold'
    else:
        status='progressing'
    valid=[r for r in completed if r['date']<=today.isoformat()]
    recent=[r for r in valid if r['date']>=(today-timedelta(days=30)).isoformat()]
    recent_long=max(recent,key=lambda r:r['miles'],default=None)
    history=[r for r in valid if r['date']>=(today-timedelta(days=84)).isoformat()
        and (r.get('role')=='long' or r is recent_long)]
    catalog={r['key']:r for r in event.get('stage_runtime',{}).get('catalog',())}
    issued=[]
    for day,key in event.get('program',()):
        dose=catalog.get(key)
        if today.isoformat()<=day<event['hard_end'] and dose and dose['role']=='long':
            issued.append(dict(date=day,miles=dose['miles']))
    project=status in {'supporting_volume','progressing'}
    path=conditional_path(today,volume=current,long_miles=stage.long_miles,
        long_budget=progress.get('long_budget',stage.long_miles),
        target_volume=target.volume_per_7,target_long=target.long_miles,
        volume_rate=rate,long_fraction=policy.long_growth_fraction,long_days=policy.long_growth_days,
        long_share=policy.long_share,
        taper_start=race.taper_start if race else None) if project else dict(points=[],target_date=None,support_date=None,taper=None)
    # The actual current hold remains visible. Future growth is conditional on
    # delivered training supporting the projected volume, not credit earned now.
    return dict(available=True,as_of=state.as_of.isoformat(),today=today.isoformat(),status=status,phase=phase,
        goal_kind=race.distance if race else 'general_fitness',race_date=race.race_date.isoformat() if race else None,
        taper_start=race.taper_start.isoformat() if race else None,
        target_long_miles=target.long_miles,target_volume_per_7=target.volume_per_7,
        recent_long_run=recent_long,current_long_miles=stage.long_miles,
        volume_per_7=current,volume_growth_per_7=rate,observed_volume_per_7=actual,
        long_growth_allowed=bool(progress.get('long_growth_allowed')),
        supporting_volume_for_long=required,
        rate_provisional=bool(progress.get('reference_provisional')),
        goal_change_pending=bool(requested and requested.goal_long_miles != (
            state.planner_preferences.goal_long_miles if state.planner_preferences else stage.goal_long_miles)),
        recorded=history,scheduled=issued,projection=path,
        race_projection_meets_target=(bool(path['target_date']) if race and project else None),
        projection_basis='Current coaching rates with future adherence; projected volume gates long-run growth. Not earned credit, an issued schedule or a fitness forecast')
