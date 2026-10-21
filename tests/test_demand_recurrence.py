from datetime import date,timedelta
from math import exp
import numpy as np
import pytest
from run_analysis.adaptive.program_objective_trial import (
    TrialSession,TrialPolicy,DemandScenario,components,demand_value,
)
from run_analysis.workload_accounting import Quantity


@pytest.mark.parametrize('memory',[.3,2.,7.])
@pytest.mark.parametrize('compression',[0.,.25,.99])
@pytest.mark.parametrize('fraction',[0.,.5,1.])
def test_recursive_demand_matches_explicit_all_pairs(memory,compression,fraction):
    origin=date(2026,9,27)
    offsets=(-84,-30,-8,-3,-1,0,1,4,8,14,21,34,48)
    rows=tuple(TrialSession(str(i),origin+timedelta(days=d),3.,'easy',
        tuple(Quantity(10+j+i,30+j+i,17+j+i) for j in range(3)),
        source='observed' if d<0 else 'planned') for i,d in enumerate(offsets))
    scenario=DemandScenario(memory,fraction,compression)
    scales=(44.,61.,15.)
    q=np.array([[demand_value(v,fraction) for v in r.demand] for r in rows])/scales
    total=np.zeros(3)
    for i,r in enumerate(rows):
        if r.day<origin:
            continue
        preceding=sum((q[j]*exp(-max(0.,(r.day-rows[j].day).days-compression)/memory)
                       for j in range(i)),np.zeros(3))
        total+=.5*q[i]**2+q[i]*preceding
    actual=components(tuple(r for r in rows if r.source=='observed'),
        tuple(r for r in rows if r.source=='planned'),stage_start=origin,decision=origin,
        end=origin+timedelta(days=49),scales=scales,policy=TrialPolicy(),scenario=scenario)
    assert tuple(actual['demand_by_axis'].values())==pytest.approx(total*7/49,rel=1e-12,abs=1e-12)
