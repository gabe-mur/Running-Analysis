from scripts.audit_search_witness_path import splice, substitutions, translated_segments
from collections import Counter
from run_analysis.adaptive.continuous_search_adapter import interleave_proposals


def test_witness_splice_preserves_untouched_dates_and_handles_rest_and_dose():
    before = (('a', 'easy'), ('b', 'long'), ('d', 'short'))
    after = (('a', 'long'), ('c', 'easy'), ('d', 'short'))
    assert substitutions(before, after) == ('a', 'b', 'c')
    assert splice(before, after, ('b', 'c')) == (('a', 'easy'), ('c', 'easy'), ('d', 'short'))
    assert splice(before, after, substitutions(before, after)) == after
    assert splice(before, after, ()) == before
    assert substitutions(after, after) == ()


def test_splice_order_cannot_duplicate_a_physical_date():
    before = (('a', 'easy'),)
    after = (('a', 'long'),)
    assert splice(before, after, ('a', 'a')) == after


def test_nearby_family_cannot_be_hidden_behind_coordinate_stream_and_duplicates():
    coordinate = [(((str(i), 'easy'),), 'coordinate') for i in range(1000)]
    swap = ((('a', 'quality'), ('b', 'easy')), 'nearby')
    rows = list(interleave_proposals(([swap], coordinate, [*coordinate, swap])))
    assert rows[0] == swap
    assert len(rows) == 1001
    assert len({p for p, _ in rows}) == len(rows)
    assert rows == list(interleave_proposals(([swap], coordinate, [*coordinate, swap])))


def test_round_robin_does_not_starve_other_families():
    streams = [[(((f'{family}{i}', 'easy'),), family) for i in range(20)] for family in 'abcd']
    rows = list(interleave_proposals(streams))
    assert [s for _, s in rows[:8]] == list('abcdabcd')
    assert len(rows) == 80


def test_segment_shift_preserves_every_dose_and_rejects_collision_or_boundary_loss():
    p = (('a', 'long'), ('c', 'easy'), ('d', 'quality'), ('f', 'easy'))
    rows = list(translated_segments(p, tuple('abcdefg')))
    assert (('a', 'long'), ('d', 'easy'), ('e', 'quality'), ('f', 'easy')) in rows
    assert len(set(rows)) == len(rows)
    for row in rows:
        assert len(dict(row)) == len(row) == len(p)
        assert set(dict(row)) <= set('abcdefg')
        assert Counter(k for _, k in row) == Counter(k for _, k in p)
