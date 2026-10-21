"""Forecast purpose exposure from the load predictor's SAME donor mixture.

Do not invert mean TRIMP into mean HR or assign full credit from a workout name.
Sustained exposure integrates raw donor HR before averaging. Faster exposure
uses the separately declared command speed scenario, not a claimed observation.
This predicts descriptors, not physiological equivalence or recovery clearance.
"""
from math import fsum

from run_analysis.work_stimulus import _integrated_suitability
from run_analysis.workload_accounting import Quantity
from .direct_quality_load import similarity
from .quality_block_contract import BlockPrescription, block_scenario
from .quality_catalogue_forecast import prescription_query
from .quality_purpose_contract import _planned_windows
from .requested_quality_forecast import command_scenario


def forecast_purposes(prescription, forecast, history, profiles, config, *, contract, point='nominal'):
    if contract.issued_at>prescription.request.issued_at:
        raise ValueError('Purpose target must be frozen before the command')
    layout,_=prescription_query(prescription,config,point=point)
    if tuple((p['kind'],p['start'],p['end']) for p in forecast['phases'])!=layout:
        raise ValueError('Purpose and load forecast must share the exact complete clock')
    by_key={h.key:h for h in history}
    speeds=(block_scenario if isinstance(prescription,BlockPrescription) else command_scenario)(prescription,point=point)
    reports={};credits={}
    for req in contract.requirements:
        nominal_windows,unknown=_planned_windows(prescription,req.purpose,req.minimum_bout_seconds)
        nominal_clock=0.;authorized=[]
        for p in prescription.phases:
            authorized.append((nominal_clock,nominal_clock+p.seconds.estimate) in nominal_windows)
            nominal_clock+=p.seconds.estimate
        exposure=0.;unresolved=bool(unknown);contributors=[];previous=None
        for allowed,phase,speed in zip(authorized,forecast['phases'],speeds):
            seconds=phase['end']-phase['start']
            if allowed:
                if req.target.channel=='speed_mps':
                    exposure+=seconds*_integrated_suitability(speed.first_speed,speed.last_speed,req.target)
                elif req.target.channel=='hrr':
                    matching=phase.get('work_exposure',{})
                    if not matching.get('supported'):
                        unresolved=True
                    else:
                        sessions=[]
                        for donor in matching['sources']:
                            h=by_key[donor['key']];trace=profiles[h.key]
                            work=[(o,t) for o,t in zip(h.phases,trace) if o.kind=='work' and o.load is not None]
                            fractions=donor['match_fractions']
                            if len(work)!=len(fractions):
                                raise ValueError('Load-source and purpose-source phases must align')
                            weights=[similarity(o,seconds,previous,True)*f for (o,_),f in zip(work,fractions)]
                            total=fsum(weights)
                            if not total:
                                raise ValueError('Supported donor needs positive matching weights')
                            # Integrate the actual nonlinear exposure function,
                            # not suitability(mean HR) or suitability(inverse TRIMP).
                            rates=[fsum((s.end-s.start)*_integrated_suitability(s.first,s.last,req.target)
                                for s in t.spans)/o.known_seconds for o,t in work]
                            rate=fsum(w*v for w,v in zip(weights,rates))/total
                            sessions.append((donor['relevance'],rate))
                            contributors.append(dict(phase=phase['start'],key=h.key,relevance=donor['relevance'],rate=rate))
                        total=fsum(w for w,_ in sessions)
                        exposure+=seconds*fsum(w*v for w,v in sessions)/total
                else:
                    raise ValueError('Current purpose forecast needs explicit HRR or absolute speed target')
            previous=seconds
        value=min(1.,exposure/60/req.target.minutes)
        credits[req.purpose]=(Quantity(0.,1.,basis='unresolved forecast purpose exposure') if unresolved
            else Quantity.point(value,'same matched raw-HR donor mixture or explicit command-speed scenario'))
        reports[req.purpose]=dict(exposure_minutes=exposure/60,unresolved=unresolved,contributors=contributors,
            minimum_bout_seconds=req.minimum_bout_seconds,target=req.target,
            speed_is_command_scenario=req.target.channel=='speed_mps')
    opportunity=Quantity(max(q.low for q in credits.values()),max(q.high for q in credits.values()),
        basis='one shared quality opportunity; not sum of adaptation')
    return dict(contract_key=contract.key,prescription_key=prescription.key,credits=credits,
        opportunity=opportunity,reports=reports,planner_influence=False,uncertainty_calibrated=False,
        physiology_equivalence_established=False,
        limitation='Known-time donor rates extrapolate across the command clock; gaps are not observed zero exposure')
