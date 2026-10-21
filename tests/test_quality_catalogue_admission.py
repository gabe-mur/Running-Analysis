from dataclasses import replace

from run_analysis.adaptive.quality_catalogue_admission import CatalogueAdmission, admit_candidate, compile_admitted_catalogue
from run_analysis.adaptive.quality_dose_policy import QualityDoseStage
from test_quality_recipe_catalogue import recipes
from test_quality_dose_policy import AT


def admission(work=15):
    return CatalogueAdmission(QualityDoseStage('accepted',AT,4.4,work,7,'explicit test accepted stage'),
        4.4,3600,20,'accepted prior full-session envelope')


def test_purpose_targets_are_not_physical_dose_ceilings():
    a=admission()
    assert a.window('sustained').maximum_work_seconds==a.window('faster').maximum_work_seconds==900
    assert admission(10.5).window('faster').maximum_work_seconds==630


def test_stage_envelope_rejects_offered_fartlek_range_not_uncertain_distance_speed():
    from run_analysis.adaptive.quality_recipe_catalogue import compile_recipe_options, DoseWindow
    rs={r.key:r for r in recipes()}
    generous=DoseWindow(1,900,10,7200,20,'diagnostic generation envelope')
    a=admission()
    f=next(c for c in compile_recipe_options(rs['fartlek_surges'],generous) if c.repetitions==12)
    assert 'offered_work_range_exceeds_stage' in admit_candidate(f,a)['reasons']
    d=next(c for c in compile_recipe_options(rs['pickups_distance'],generous)
        if c.repetitions==2 and c.prescription.request.blocks[1].primary.amount.estimate==400)
    assert admit_candidate(d,a)['accepted']
    assert not admit_candidate(d,replace(a,maximum_session_miles=1.))['accepted']


def test_generation_keeps_multiple_native_doses_without_self_authorizing_broader_stage():
    rs=recipes();paces={k:rs[0].base.blocks[0].pace for k in ('easy_pace',)}
    paces.update(threshold_pace=rs[0].base.blocks[1].pace,aerobic_pace=rs[2].base.blocks[1].pace,
        fast_pace=rs[4].base.blocks[1].pace,hill_pace=rs[-1].base.blocks[1].pace,open_recovery_seconds=90)
    result=compile_admitted_catalogue(admission(),issued_at=AT,**paces)
    assert len(result['candidates'])>20
    assert all(admit_candidate(c,admission())['accepted'] for c in result['candidates'])
    assert result['whole_program_admission_required'] and not result['planner_influence']
