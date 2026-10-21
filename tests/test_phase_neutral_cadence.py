from types import SimpleNamespace

from scripts.audit_phase_neutral_cadence import adjacent_pair_moves
from scripts.audit_matched_key_cadence import matched_program


def test_adjacent_pairs_preserve_dates_roles_total_and_use_distant_dates():
    catalog={f'e{n}':SimpleNamespace(role='easy',miles=n/10) for n in range(11,45)}
    catalog['long']=SimpleNamespace(role='long',miles=8.9)
    p=(('2026-10-06','e11'),('2026-10-08','long'),('2026-11-25','e44'))
    assert list(adjacent_pair_moves(p,catalog))==[
        (('2026-10-06','e12'),('2026-10-08','long'),('2026-11-25','e43'))]


def test_two_section_matching_does_not_fix_arbitrary_21_day_total():
    catalog={f'e{n}':SimpleNamespace(role='easy',miles=n/10) for n in range(11,45)}
    catalog['long']=SimpleNamespace(role='long',miles=8.9)
    p=(('2026-10-06','e22'),('2026-10-15','e22'),('2026-10-20','long'),('2026-11-25','e22'))
    result=matched_program(p,catalog,('2026-10-14','2026-11-28'),(23,149))
    assert result==(('2026-10-06','e23'),('2026-10-15','e30'),('2026-10-20','long'),('2026-11-25','e30'))
    assert sum(round(catalog[k].miles*10) for _,k in result)==172
