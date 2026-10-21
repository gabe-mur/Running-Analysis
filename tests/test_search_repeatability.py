from scripts.check_search_repeatability import normalized


def test_normalization_removes_only_compute_times_not_training_time_or_order():
    original = dict(seconds=2., planned_for='2026-09-24T07:00:00Z',
        duration_seconds=1200, zone_seconds={'Z2':1200},
        rows=[{'cost':1.,'seconds':4.}, {'cost':2.}])
    assert normalized(original) == dict(planned_for='2026-09-24T07:00:00Z',
        duration_seconds=1200, zone_seconds={'Z2':1200}, rows=[{'cost':1.},{'cost':2.}])
    assert normalized(original) != normalized({**original, 'rows':list(reversed(original['rows']))})
    assert original['seconds']==2.


def test_score_changes_and_generation_timestamps_remain_visible():
    assert normalized({'score':1.}) != normalized({'score':1.00000000001})
    assert normalized({'generated_at':'first'}) != normalized({'generated_at':'second'})
