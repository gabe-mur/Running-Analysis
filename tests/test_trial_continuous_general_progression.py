"""Driver wiring only; no claim of optimizer stability or physical adherence."""
import json
from pathlib import Path

import pytest

import scripts.trial_continuous_general_progression as trial


@pytest.mark.parametrize('resume,scenario,label,expected',[
    (True,'prescribed',None,'daily'),
    (False,'imperfect',None,'imperfect'),
    (False,'imperfect','missed-easy','missed-easy'),
])
def test_extended_screen_reuses_clock_reference_and_existing_driver(tmp_path,monkeypatch,resume,scenario,label,expected):
    seed=tmp_path/'opening';seed.mkdir()
    (seed/'decision-0035.json').write_text(json.dumps(dict(as_of='2026-10-19T00:00:39+00:00')))
    config=dict(timezone_default='America/New_York',max_hr=194,resting_hr=46)
    monkeypatch.setattr(trial,'load_config',lambda:dict(max_hr=195,resting_hr=45))
    seen={}
    def frozen(current,*,root):
        assert root==Path('original') and current['max_hr']==195
        return config
    monkeypatch.setattr(trial,'frozen_live_config',frozen)
    def run(root,database,**kwargs):
        from scripts.run_adaptive_lifecycle import load_config
        assert load_config()==config
        seen.update(kwargs,root=root,database=database)
        return dict(status='fixture_only',completed_days=0,failures=[])
    monkeypatch.setattr(trial,'run',run)
    trial.daily(Path('original'),tmp_path,Path('activities.sqlite'),21,
        resume=resume,scenario=scenario,label=label,checkpoint_days=7,missed_run_offsets=(2,))
    assert seen==dict(root=tmp_path/expected,database=Path('activities.sqlite'),days=21,
        scenario=scenario,resume=resume,seed_plan=None if resume else seed,
        screen_start='2026-10-19',stability_window=4,checkpoint_days=7,missed_run_offsets=(2,))
