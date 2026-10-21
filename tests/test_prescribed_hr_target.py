import pytest

from run_analysis.adaptive.prescribed_hr_target import prescribed_hr_target
from run_analysis.adaptive.quality_prescription_contract import Effort
from scripts.compare_direct_effort_load import requested_targets
from scripts.compare_requested_quality_response import saved_effort

CFG=dict(resting_hr=50,max_hr=200,zones=dict(z1=[100,120],z2=[121,140],
    z3=[141,160],z4=[161,180],z5=[181,200]))


@pytest.mark.parametrize('cues,bpm',[
    (('Z4 effort',),(161,170.5,180)),
    (('low Z4',),(161,165.75,170.5)),
    (('upper Z3',),(150.5,155.25,160)),
    (('bottom Z3',),(141,141,150.5)),
    (('top Z3',),(150.5,160,160)),
    (('middle Z3',),(150.5,150.5,150.5)),
    (('upper Z3','low Z4'),(150.5,160.5,170.5)),
    (('upper Z3 / low Z4',),(150.5,160.5,170.5)),
    (('Z1','Z2'),(100,120,140)),
    (('Z3','Z4 effort'),(141,160.5,180)),
])
def test_targets_follow_prescription_not_observations(cues,bpm):
    target=prescribed_hr_target(cues,CFG)
    assert target.bpm==pytest.approx(bpm)
    assert (target.low,target.nominal,target.high)==pytest.approx([(v-50)/150 for v in bpm])


def test_mapping_is_order_case_whitespace_and_duplicate_invariant():
    a=prescribed_hr_target(('upper Z3','low Z4'),CFG)
    b=prescribed_hr_target((' LOW z4 ','Upper z3','Upper z3'),CFG)
    assert a.bpm==b.bpm and a.basis==b.basis


def test_unknown_cues_are_not_silently_dropped():
    assert prescribed_hr_target(('threshold',),CFG) is None
    assert prescribed_hr_target(('Z4','threshold'),CFG) is None


def test_scaling_uses_athletes_hr_reference_not_fixed_reserve_values():
    other=dict(CFG,resting_hr=40,max_hr=210)
    a,b=(prescribed_hr_target(('Z4',),c) for c in (CFG,other))
    assert a.bpm==b.bpm and a.nominal!=b.nominal


def test_progression_order_survives_mapping_and_nonwork_is_not_work():
    payload=dict(structure=[dict(phase=None,target_zones=['Z1','Z2']),
        dict(phase='work',target_zones=['Z2','Z3'],
            instruction='Gradually increase, finishing controlled in Z3')])
    request=saved_effort(payload,'work')
    start,finish=requested_targets(request,CFG,'prescription')
    assert start.bpm[1]==130.5 and finish.bpm[1]==150.5
    nonwork=requested_targets(saved_effort(payload,'cooldown'),CFG,'prescription')
    assert nonwork[0].bpm[1]==120


def test_original_full_envelope_comparator_is_unchanged():
    effort=Effort(('upper Z3','low Z4'),('upper Z3','low Z4'),'purposeful')
    old=requested_targets(effort,CFG,'full-envelope')
    assert old[0]==dict(low=(141-50)/150,nominal=(160.5-50)/150,high=(180-50)/150)
    assert requested_targets(effort,CFG,'prescription')[0].low>old[0]['low']


def test_bad_zone_configuration_remains_explicit():
    with pytest.raises(ValueError,match='normalization'):
        prescribed_hr_target(('Z4',),dict(CFG,max_hr=170))
