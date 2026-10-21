"""Diagnostic contracts; no dependence on local athlete files or search."""
import pytest

from scripts.audit_progression_rollover_instability import splice_program, fixed_taper_guard


def test_splice_is_exact_unprivileged_prefix_and_suffix():
    prior = (('2026-10-13', 'easy:2.7'), ('2026-10-15', 'quality:ordinary'),
             ('2026-11-22', 'easy:2.2'))
    alternative = (('2026-10-13', 'quality:ordinary'), ('2026-10-15', 'easy:2.3'),
                   ('2026-11-22', 'long:8.9'))
    assert splice_program(prior, alternative, '2026-11-21') == (
        ('2026-10-13', 'easy:2.7'), ('2026-10-15', 'quality:ordinary'),
        ('2026-11-22', 'long:8.9'))
    assert prior[0][1] == 'easy:2.7' and alternative[0][1] == 'quality:ordinary'


def test_splice_half_open_boundary_has_one_owner():
    assert splice_program((('2026-10-20', 'easy:2.7'),),
        (('2026-10-20', 'quality:ordinary'),), '2026-10-20') == (
        ('2026-10-20', 'quality:ordinary'),)
    with pytest.raises(ValueError, match='One session'):
        splice_program((('2026-10-13', 'a'), ('2026-10-13', 'b')), (), '2026-10-20')


def test_fixed_taper_control_offsets_only_elapsed_days():
    assert fixed_taper_guard(7, 1) == 6
    assert fixed_taper_guard(7, 0) == 7
    for elapsed in (-1, 7, 1.5):
        with pytest.raises(ValueError):
            fixed_taper_guard(7, elapsed)
