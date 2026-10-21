"""Preference filtering and numeric menu installation before ordinary search.

The original equal-work-clock prototype remains a reproducible comparator;
installation uses the reviewed native recipes and existing direct predictor.
"""
from dataclasses import replace
from hashlib import sha256
import json
from math import ceil

from run_analysis.recommendation import _quality_prescription
from run_analysis.session_decision import CoachingContract, DoseLimits
from run_analysis.staged_progression import tenth
from run_analysis.web.schemas import QualitySessionSettings, QualitySessionType, WorkoutStep
from run_analysis.workload_accounting import Quantity

VERSION = 'quality_direct_calendar_v1'
LEGACY_VERSION = 'quality_choices_v1'


def quality_focus(config):
    focus = config.get('coaching', {}).get('quality_focus', 'variety')
    if focus not in {'variety', 'vo2max'}:
        raise ValueError('Unknown quality focus')
    return focus


def enabled_types(config):
    return tuple(k for k,v in QualitySessionSettings.model_validate(
        config.get('coaching',{}).get('quality_sessions') or {}).model_dump().items() if v)


def dose_type(dose,contract):
    if dose.workout_definition:
        return dose.workout_definition['family']
    # Compatibility with already issued generic intervals. Do not relabel
    # them as a new recipe merely because preferences or history changed.
    return ('long_intervals' if contract.ordinary.bout_seconds.high>=180 else 'short_intervals')


def allowed(dose,contract,config):
    if dose.workout_definition and dose.workout_definition.get('version')==LEGACY_VERSION:
        # Keep issued prototype bytes readable, but do not offer those superseded
        # recipes alongside the complete numeric commands on future searches.
        return False
    return dose.role!='quality' or dose_type(dose,contract) in enabled_types(config)


def _time(seconds):
    minutes,remaining=divmod(round(seconds),60)
    if not minutes:
        return f'{remaining} seconds'
    return f'{minutes}:{remaining:02d}' if remaining else f'{minutes} min'


def compile_choices(anchor,contract):
    """Append immutable alternatives without rebinding an archived dose key."""
    # Preserve the original prototype as a comparator and old identity reader.
    VERSION=LEGACY_VERSION
    from .new_model_daily_harness import Dose
    budget=sum(s for k,s in anchor.phases if k=='work')
    pace=sum(s for _,s in anchor.phases)/anchor.miles
    if budget<=0:
        raise ValueError('Quality selection needs a frozen positive work budget')
    doses,contracts=[],{}
    for kind in QualitySessionType:
        original=_quality_prescription(kind)
        pair=[]
        for smaller in (False,True):
            work=budget*(.8 if smaller else 1.)
            warm,cool=600.,600.
            main=next(s for s in original['structure'] if s.phase=='work')
            if kind==QualitySessionType.PROGRESSION:
                work=max(30.,round(work/30)*30.)
                half=max(600.,work)
                phases=[('warmup',half)]
                if half>work:
                    phases.append(('warmup',half-work))
                phases.append(('work',work))
                steps=[WorkoutStep(instruction='Run the first half relaxed in Z1/Z2.',
                    phase='warmup',prescription_basis='portion',duration_minutes=half/60,target_zones=['Z1','Z2']),
                    main.model_copy(update=dict(duration_minutes=half/60,work_duration_minutes=work/60,
                        prescription_basis='portion'))]
            else:
                if kind==QualitySessionType.THRESHOLD:
                    count=1; rest=0.; unit='time'
                elif kind==QualitySessionType.LONG_INTERVALS:
                    count=max(1,min(3,ceil(budget/300))); rest=120.; unit='time'
                elif kind==QualitySessionType.SHORT_INTERVALS:
                    # Preserve the established number of short efforts. Their
                    # distance estimates use the original bout pace at render.
                    count=max(3,len([s for k,s in anchor.phases if k=='work']),ceil(budget/120)); rest=90.; unit='distance'
                elif kind==QualitySessionType.FARTLEK:
                    count=max(3,ceil(budget/90)); rest=90.; unit='time'
                else:
                    count=max(3,ceil(budget/45)); rest=90.; unit='time'
                seconds=work/count
                if unit=='time':
                    # Commit readable timings before forecasts/contracts are
                    # built, rather than displaying a different work dose.
                    increment=(5 if kind==QualitySessionType.HILL_REPEATS else
                        15 if kind==QualitySessionType.FARTLEK else 30)
                    seconds=max(float(increment),round(seconds/increment)*float(increment))
                    work=seconds*count
                phases=[('warmup',warm)]
                for i in range(count):
                    phases.append(('work',seconds))
                    if i<count-1:
                        phases.append(('recovery',rest))
                phases.append(('cooldown',cool))
                if kind==QualitySessionType.THRESHOLD:
                    instruction=f'Run {_time(work)} continuously at controlled threshold effort. Start conservatively; do not sprint.'
                elif kind==QualitySessionType.HILL_REPEATS:
                    instruction=f'Run {count} × {_time(seconds)} uphill at strong controlled effort. Jog gently downhill for about {_time(rest)} between efforts; stop early if form changes.'
                elif kind==QualitySessionType.FARTLEK:
                    instruction=f'Add {count} surges of about {_time(seconds)} using landmarks. Run easily for about {_time(rest)} between surges; stay controlled.'
                elif kind==QualitySessionType.LONG_INTERVALS:
                    instruction=f'Run {count} × {_time(seconds)} at controlled hard effort with {_time(rest)} of easy jogging between efforts. Finish as smoothly as you start.'
                else:
                    # The distance-based interval body is generated from these
                    # exact phases by the shared compact formatter, not guessed
                    # by converting the duration of a different named workout.
                    instruction=None
                if instruction is None:
                    steps=None
                else:
                    steps=[WorkoutStep(instruction='Easy warm-up.',phase='warmup',prescription_basis='time',
                        duration_minutes=warm/60,target_zones=['Z1','Z2']),
                        main.model_copy(update=dict(instruction=instruction,prescription_basis=unit,
                            duration_minutes=(work+(count-1)*rest)/60,repetitions=count if count>1 else None,
                            work_duration_minutes=seconds/60,work_duration_range_minutes=None,
                            recovery_duration_minutes=rest/60 if rest else None)),
                        WorkoutStep(instruction='Easy cool-down.',phase='cooldown',prescription_basis='time',
                            duration_minutes=cool/60,target_zones=['Z1','Z2'])]
            definition=dict(version=VERSION,family=kind.value,title=original['title'],
                workout_type=original['workout_type'].value,zones=original['zones'],
                structure=[s.model_dump(mode='json') for s in steps] if steps else None)
            miles=tenth(sum(s for _,s in phases)/pace)
            key='quality:'+kind.value+':'+sha256(json.dumps(dict(definition=definition,phases=phases,
                miles=miles,source=contract.source),sort_keys=True).encode()).hexdigest()[:16]
            scope=replace(anchor.forecast_scope,duration_source=VERSION+'; accepted work budget') if anchor.forecast_scope else None
            dose=Dose(key,'quality',miles,tuple(phases),scope,tuple((s*.9,s*1.1) for _,s in phases),definition)
            bouts=[s for k,s in phases if k=='work']
            ratios=[phases[i+1][1]/s for i,(k,s) in enumerate(phases[:-1])
                if k=='work' and phases[i+1][0]=='recovery'] or [0.]
            limits=DoseLimits(Quantity.point(miles,VERSION),Quantity.point(sum(bouts),VERSION),
                Quantity(min(bouts),max(bouts)),Quantity(min(ratios),max(ratios)))
            pair.append((dose,limits))
        full,small=pair
        shared=CoachingContract(contract.issued_at,contract.source+';'+VERSION,
            contract.purpose,contract.intensity_key,full[1],small[1],full[1] if contract.trial else None,
            full[1].work_seconds.estimate if contract.trial else None)
        for dose,_ in pair:
            doses.append(dose);contracts[dose.key]=shared
    return tuple(doses),contracts


def install(h):
    if getattr(h,'quality_runtime',None) is not None:
        # An explicitly frozen offline catalogue owns its accepted doses. Do
        # not append the old generic-credit menu during a lifecycle refresh.
        return
    if any(d.workout_definition and d.workout_definition.get('calendar_integration')==VERSION
           for d in h.catalog.values()):
        return
    anchor=h.catalog.get('quality:ordinary')
    if anchor is None:
        return
    from .quality_calendar_integration import compile_calendar_choices
    doses,contracts=compile_calendar_choices(h,anchor,h.coaching_contracts[anchor.key])
    h.catalog={**h.catalog,**{d.key:d for d in doses}}
    h.coaching_contracts={**h.coaching_contracts,**contracts}
