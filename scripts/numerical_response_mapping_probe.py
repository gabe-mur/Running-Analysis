"""Offline arithmetic checks of the shared experimental response model."""
from dataclasses import asdict, replace
from math import isclose
import json

from run_analysis.response_model import (
    Policy, Phase, Dose, Reference, rate, dose, ewma, balance, easy_load, pressure,
)


def scenarios():
    easy = Phase(15, .6, 1)
    hard = Phase(2.5, .88, 1.35)
    split = [easy]
    for i in range(6):
        split.append(hard)
        if i < 5:
            split.append(Phase(2, .45, .8))
    split.append(easy)
    return {
        'easy40': [Phase(40, .6, 1)],
        'long90': [Phase(90, .6, 1)],
        'continuous15': [easy, Phase(15, .88, 1.35), Phase(10, .45, .8), easy],
        'three_by_five': [easy, Phase(5, .88, 1.35), Phase(5, .45, .8),
                          Phase(5, .88, 1.35), Phase(5, .45, .8), Phase(5, .88, 1.35), easy],
        'six_by_two_half': split,
    }


def checks():
    """Algebra/behavior checks, explicitly not validation of physiological truth."""
    checks_passed = []
    def check(name, result):
        if not result:
            raise AssertionError(name)
        checks_passed.append(name)

    policy = Policy()
    profiles = scenarios()
    values = {name: dose(phases, policy) for name, phases in profiles.items()}
    ref = Reference(20, 5, 40, 35, 40)
    check('unit reference', isclose(rate(.6), 1))
    check('zero reserve', rate(0) == 0)
    check('nonlinear intensity', rate(.9) > rate(.6)*1.5)
    check('no fast exposure from constant easy pace', values['easy40'].fast == 0)
    check('equal duration effort, different absolute distance',
          dose([Phase(40, .6, 1)]) == values['easy40'] and 40/5 == 8 and 40/20 == 2)
    check('equal five miles, different duration',
          isclose(dose([Phase(100, .6, 1)]).cardio/dose([Phase(25, .6, 1)]).cardio, 4))
    for name in ('three_by_five', 'six_by_two_half'):
        check(name+' equal base cardio', isclose(values[name].cardio, values['continuous15'].cardio))
        check(name+' equal fast dose', isclose(values[name].fast, values['continuous15'].fast))
    check('specified synthetic structure ordering',
          values['continuous15'].response_cardio > values['three_by_five'].response_cardio
          > values['six_by_two_half'].response_cardio)
    split_phase = dose([Phase(2, .88, 1.35), Phase(3, .88, 1.35)])
    full_phase = dose([Phase(5, .88, 1.35)])
    check('invariant to arbitrary phase splitting', all(isclose(a,b) for a,b in
          zip(asdict(split_phase).values(), asdict(full_phase).values())))
    check('fast movement remains visible with low HR', dose([Phase(2, .5, 1.5)]).fast > 0)
    h = [(values['long90'], 24)]
    p_easy = pressure(h, values['easy40'], ref)
    p_hard = pressure(h, values['three_by_five'], ref)
    check('same history different next dose', p_hard['total'] > p_easy['total'])
    check('more elapsed time reduces modeled pressure',
          pressure([(values['long90'], 48)], values['three_by_five'], ref)['total'] < p_hard['total'])
    check('no fabricated missing feedback', p_easy['reported_leg_pressure'] is None)
    check('current leg report affects easy dose',
          pressure(h, values['easy40'], ref, current_leg_rating=8)['total'] > p_easy['total'])
    check('same profile through either observed/projected caller',
          dose(list(profiles['easy40'])) == dose(tuple(profiles['easy40'])))
    check('larger established exposure changes normalization',
          pressure(h, values['three_by_five'], Reference(40, 10, 40, 35, 40))['total'] < p_hard['total'])
    check('zero density weight removes artificial structure uplift',
          isclose(dose(profiles['three_by_five'], replace(policy, density_weight=0)).response_cardio,
                  dose(profiles['six_by_two_half'], replace(policy, density_weight=0)).response_cardio))
    check('longer modeled recovery cannot enlarge reference capacity',
          pressure(h, values['three_by_five'], ref,
                   replace(policy, cardio_half_life_hours=36))['total'] > p_hard['total'])
    check('ewma constant input equilibrium', isclose(ewma(20,20,1,7),20))
    check('ewma time partition invariance',
          isclose(ewma(ewma(10,20,.4,7),20,.6,7),ewma(10,20,1,7)))
    check('ratio uses absolute loads', isclose(balance(32,33)['ratio'],32/33))
    check('missing positive CTL has no infinite ratio', balance(5,0)['ratio'] is None)
    next_load = easy_load(32,33)
    check('easy-load equation reaches balance',
          isclose(ewma(32,next_load,1,7),ewma(33,next_load,1,42)))
    return checks_passed, values, ref


if __name__ == '__main__':
    passed, values, reference = checks()
    sensitivities = []
    for h in (12., 24., 36.):
        for density_weight in (0., .25, .5):
            policy = replace(Policy(), cardio_half_life_hours=h, density_weight=density_weight)
            profiles = scenarios()
            prior = dose(profiles['long90'], policy)
            nxt = dose(profiles['three_by_five'], policy)
            sensitivities.append(dict(cardio_half_life_hours=h, density_weight=density_weight,
                pressure=pressure([(prior,24)], nxt, reference, policy)['total']))
    comparisons = {name: pressure([(values['long90'],24)], value, reference)
                   for name, value in values.items()}
    print(json.dumps(dict(status='arithmetic_only_not_calibrated',
        checks_passed=passed, doses={k:asdict(v) for k,v in values.items()},
        after_long_24h=comparisons, sensitivity=sensitivities,
        illustrative_absolute_load_context=balance(32,33),
        illustrative_easy_load=easy_load(32,33)), indent=2))
