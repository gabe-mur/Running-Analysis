from copy import deepcopy
from datetime import datetime,timedelta,timezone

import numpy as np
import pytest

from scripts.audit_small_aerobic_changes import noise_library,draw_noise,simulate_rows,snapshot,summarize_changes,discrimination
from run_analysis.aerobic_progress import build_aerobic_progress


def rows():
    origin=datetime(2026,1,1,tzinfo=timezone.utc)
    return [dict(activity_id=i,start_time=(origin+timedelta(days=i*3)).isoformat(),
        pace_min_mile=10*(1+.02*np.sin(i/3)),uncertainty_95_min_mile=.5,trend_weight=1.,
        included_in_trend=True,shared_log_speed_gradient=[0.]) for i in range(70)]


def test_null_is_constant_latent_level_not_the_original_unknown_progress():
    source=rows();before=deepcopy(source)
    end=datetime.fromisoformat(source[-1]['start_time']);onset=end-timedelta(days=84)
    constant=simulate_rows(source,np.zeros(len(source)),10,onset,0,84)
    value=snapshot(constant,end,onset,np.zeros((1,1)))
    assert value['current']['pace_min_mile']==pytest.approx(10)
    assert all(abs(v['speed_change_percent'])<1e-10 for v in value['changes'].values())
    assert source==before


def test_small_gain_is_preserved_but_trailing_level_and_monthly_change_are_not_endpoint_gain():
    source=rows();end=datetime.fromisoformat(source[-1]['start_time']);onset=end-timedelta(days=84)
    for gain in (.005,.01,.02):
        changed=simulate_rows(source,np.zeros(len(source)),10,onset,gain,84)
        value=snapshot(changed,end,onset,np.zeros((1,1)))
        assert 0<value['changes']['monthly']['speed_change_percent']<value['changes']['since_onset']['speed_change_percent']<gain*100
        oracle=build_aerobic_progress(changed,as_of=end,covariance=[[0]],history_days=1)
        assert value['current']==oracle['current']
        assert value['changes']['monthly']==oracle['change']
        assert value['changes']['ninety_day']==oracle['change_90d']


def test_blocks_preserve_good_bad_spells_without_centering_each_spell():
    source=rows();blocks,info=noise_library(source,14,False)
    assert info['donor_blocks']>20
    assert abs(np.concatenate(blocks).mean())<1e-12
    assert max(abs(b.mean()) for b in blocks)>.005
    a=draw_noise(source,blocks,np.random.default_rng(1));b=draw_noise(source,blocks,np.random.default_rng(1))
    np.testing.assert_equal(a,b)
    assert len(a)==len(source)


def test_false_improvement_counts_follow_actual_evidence_not_positive_numbers():
    def change(direction,strength,value):
        return dict(direction=direction,evidence_strength=strength,speed_change_percent=value,
            estimated_change_range_95_percent=[-3,4])
    result=summarize_changes([change('stable_or_uncertain','inconclusive',1),
        change('improving','likely',2),change('declining','clear',-3),change('provisional','provisional',4)])
    assert result['improving_label_rate']==.25
    assert result['clear_improving_label_rate']==0
    assert result['declining_label_rate']==.25
    assert result['inconclusive_or_provisional_rate']==.5
    assert result['improving_label_rate_among_established']==pytest.approx(1/3)
    assert summarize_changes([])=={'n':0}


def test_no_established_history_is_not_reported_as_zero_detection_power():
    p=dict(direction='provisional',evidence_strength='provisional',speed_change_percent=1,
        estimated_change_range_95_percent=[-2,4])
    assert summarize_changes([p])['improving_label_rate_among_established'] is None
    assert discrimination([p],[p])==.5
    assert discrimination([dict(p,speed_change_percent=3)],[p])==1
    assert discrimination([],[]) is None
