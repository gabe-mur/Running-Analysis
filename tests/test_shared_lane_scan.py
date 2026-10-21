from run_analysis.adaptive.shared_lane_scan import scan_lanes


def test_large_first_lane_cannot_starve_later_lanes():
    calls=[]
    streams=[[(f'{lane}:{i}',(lane,i)) for i in range(10)] for lane in range(3)]
    scans=scan_lanes(streams,lambda p:calls.append(p) or p,cache={},budget=8)
    assert calls==[(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2)]
    assert [s['evaluations'] for s in scans]==[3,3,2]
    assert not any(s['complete'] for s in scans)


def test_exact_budget_completion_cache_rejections_and_empty_lanes():
    calls=[]; cache={}
    scans=scan_lanes([[(i,i) for i in range(3)],[(i,i) for i in range(3)],[]],
        lambda p:calls.append(p),cache=cache,budget=3)
    assert calls==[0,1,2]
    assert all(s['complete'] for s in scans)
    assert sum(s['cache_hits'] for s in scans)==3
    assert all(not s['results'] for s in scans)


def test_completed_short_lane_leaves_budget_for_long_lane():
    scans=scan_lanes([[(0,0)],[(i,i) for i in range(1,5)]],lambda p:p,cache={},budget=5)
    assert all(s['complete'] for s in scans)
    assert [s['evaluations'] for s in scans]==[1,4]
