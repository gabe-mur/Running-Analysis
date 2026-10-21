from run_analysis.calendar_diversity import retain_diverse
from scripts.continuous_search_adapter import search


def test_half_score_half_shape_and_stable_fill():
    rows=[(i, i//3) for i in range(12)]
    result=retain_diverse(rows,6,rank=lambda r:r,identity=lambda r:r[0],signature=lambda r:r[1])
    assert result==[(0,0),(1,0),(2,0),(3,1),(6,2),(9,3)]
    assert retain_diverse(list(reversed(rows)),6,rank=lambda r:r,identity=lambda r:r[0],signature=lambda r:r[1])==result


def test_beam_reaches_whole_domain_with_explicit_budget_and_role_diversity():
    dates=tuple(f'2026-10-{i:02d}' for i in range(1,15))
    def evaluate(p):
        counts={k:sum(v==k for _,v in p) for k in ('e','l','q')}
        return dict(cost=sum((counts[k]-n)**2 for k,n in (('e',4),('l',2),('q',2))),eligible=True)
    result=search(dates=dates,keys=('e','l','q'),previous=(),evaluate=evaluate,budget=1500,
        hard_end=dates[10],roles={'e':'easy','l':'long','q':'quality'})
    assert result['calendar_beam']['complete']
    assert result['calendar_beam']['completed_dates']==14
    assert max(len(p) for p in result['calendar_beam']['retained'])>4
    assert result['cost']==0.
    assert result['evaluations']<=1500


def test_incomplete_beam_never_claims_full_horizon_exploration():
    result=search(dates=('2026-10-01','2026-10-02','2026-10-03'),keys=('e','l'),previous=(),
        evaluate=lambda p:dict(cost=-len(p),eligible=True),budget=8,hard_end='2026-10-03',roles={'e':'easy','l':'long'})
    assert not result['calendar_beam']['complete']


def test_complete_stage_seeds_span_role_phases_and_easy_counts_without_fixed_dates():
    from collections import Counter
    from scripts.stage_calendar_seeds import complete_stage_seeds
    from scripts.new_model_daily_harness import Dose
    from scripts.program_objective_trial import TrialPolicy
    from datetime import date,timedelta
    dates=tuple((date(2026,10,1)+timedelta(days=i)).isoformat() for i in range(35))
    doses=[Dose(f'easy:{m:.1f}','easy',m,(('warmup',m*600),)) for m in (1.1,1.5,2.2,4.4)]
    doses.extend((Dose('long','long',8.4,(('warmup',5040),)),
                  Dose('quality','quality',3.9,(('warmup',600),('work',600),('cooldown',600)))))
    catalog={d.key:d for d in doses}
    result=complete_stage_seeds(dates,catalog,TrialPolicy(weekly_miles=16.5),supporting_miles=4.4)
    assert len(result)==168
    phases={(next(dates.index(d) for d,k in p if k=='long'),
             next(dates.index(d) for d,k in p if k=='quality')) for p in result}
    assert len(phases)==42
    counts={sum(catalog[k].role=='easy' for _,k in p) for p in result}
    assert {5,10,15}<=counts
    assert all(len(dict(p))==len(p) and p[-1][0]>=dates[-7] for p in result)
    assert complete_stage_seeds(dates,catalog,TrialPolicy(weekly_miles=16.5),supporting_miles=4.4)==result


def test_capped_seed_shapes_do_not_rotate_with_decision_origin():
    from datetime import date,timedelta
    from scripts.stage_calendar_seeds import complete_stage_seeds
    from scripts.new_model_daily_harness import Dose
    from scripts.program_objective_trial import TrialPolicy
    catalog={d.key:d for d in [Dose('easy','easy',2.5,(('warmup',1500),)),
        Dose('long','long',8.9,(('warmup',5340),)),
        Dose('quality','quality',3.9,(('work',600),))]}
    def library(shift,limit):
        dates=tuple((date(2026,10,12)+timedelta(days=shift+i)).isoformat() for i in range(54))
        return complete_stage_seeds(dates,catalog,TrialPolicy(weekly_miles=20.3),
                                    supporting_miles=7.5,limit=limit)
    # Clip to shared dates, but preserve membership/order of representatives.
    for limit in (1,41,42,168,2000):
        before=library(0,limit)
        for shift in range(1,8):
            after=library(shift,limit)
            low=(date(2026,10,12)+timedelta(days=shift)).isoformat()
            high=(date(2026,10,12)+timedelta(days=54)).isoformat()
            clip=lambda programs:tuple(tuple(p for p in plan if low<=p[0]<high) for plan in programs)
            assert clip(before)==clip(after)


def test_complete_first_library_covers_key_phases_and_easy_counts_across_volumes():
    from datetime import date,timedelta
    from dataclasses import replace
    from scripts.stage_calendar_seeds import complete_stage_seeds
    from scripts.new_model_daily_harness import Dose
    from scripts.program_objective_trial import TrialPolicy
    dates=tuple((date(2026,10,12)+timedelta(days=i)).isoformat() for i in range(54))
    # Candidate-library coverage, not a claim these toy doses are prescribed
    # training for any athlete. The same bounded search policy covers each.
    for volume in (5.,20.,60.):
        catalog={d.key:d for d in [Dose(f'easy:{m:.1f}','easy',m,(('warmup',m*600),))
            for m in (round(volume/20,1),round(volume/5,1),round(volume/3,1))]}
        catalog.update({d.key:d for d in (Dose('long','long',round(volume*.4,1),(('warmup',600),)),
            Dose('quality','quality',round(volume*.2,1),(('work',600),)))})
        for cadence in (7,8):
            policy=replace(TrialPolicy(weekly_miles=volume),long_per_7=7/cadence,quality_per_7=7/cadence)
            seeds=complete_stage_seeds(dates,catalog,policy,supporting_miles=volume*.4,limit=500)
            phases={(next(date.fromisoformat(d).toordinal()%cadence for d,k in p if k=='long'),
                     next(date.fromisoformat(d).toordinal()%cadence for d,k in p if k=='quality')) for p in seeds}
            assert len(phases)==cadence*(cadence-1)
            assert len({sum(catalog[k].role=='easy' for _,k in p) for p in seeds})>=3
