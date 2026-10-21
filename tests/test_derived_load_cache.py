import sqlite3

from run_analysis.derived_load_cache import cache_path,read_traces,store_traces,trace_key


def test_exact_summaries_survive_restart_and_corruption_is_a_cache_miss(tmp_path):
    database=tmp_path/'runs.sqlite'
    value=dict(elapsed_seconds=1200.,covered_seconds=600.,load_bounds_within_reference=None,unused='not persisted')
    key=trace_key(('run-sha',194,46))
    assert read_traces(database)=={}
    store_traces(database,{key:value})
    assert cache_path(database).stat().st_mode&0o777==0o600
    assert read_traces(database)=={key:{k:v for k,v in value.items() if k!='unused'}}
    with sqlite3.connect(cache_path(database)) as db:
        db.execute("UPDATE traces SET payload='{}'")
    assert read_traces(database)=={}


def test_calculation_version_changes_and_unavailable_cache_fall_back(tmp_path,monkeypatch):
    import run_analysis.derived_load_cache as module
    monkeypatch.setattr(module,'implementation_key',lambda:'first')
    first=trace_key(('same source',194))
    monkeypatch.setattr(module,'implementation_key',lambda:'second')
    assert trace_key(('same source',194))!=first
    monkeypatch.setattr(module,'implementation_key',lambda:None)
    assert trace_key(('same source',194)) is None
    def unavailable(*args,**kwargs):raise sqlite3.OperationalError('cache unavailable')
    monkeypatch.setattr(module.sqlite3,'connect',unavailable)
    assert read_traces(tmp_path/'runs.sqlite')=={}
    store_traces(tmp_path/'runs.sqlite',{first:dict(elapsed_seconds=1.,covered_seconds=1.,load_bounds_within_reference=None)})
