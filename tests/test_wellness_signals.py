from datetime import date, timedelta
from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

from run_analysis import intervals_connection as remote
from run_analysis.wellness_signals import clean_rows,read_wellness,refresh_wellness


def test_clean_data_keeps_missing_distinct_from_zero_and_does_not_infer_recovery():
    rows=clean_rows([dict(id='2026-10-14',hrv=50,sleepSecs=0,sleepScore=0,restingHR=45,vo2max=40.5,secret='no'),
        dict(id='2026-10-13',hrv=float('nan'),restingHR=0)])
    assert rows==[dict(date='2026-10-14',hrv_ms=50,sleep_seconds=0,sleep_score=0,
                       resting_hr_bpm=45,reported_vo2max_ml_kg_min=40.5)]
    assert 'readiness' not in rows[0]


def test_wellness_reader_is_get_only_bounded_and_sends_no_key_in_url(tmp_path,monkeypatch):
    remote.save_key(tmp_path,'test-wellness-secret')
    calls=[];today=date.today().isoformat()
    class Opener:
        def open(self,request,timeout):
            calls.append(request);return BytesIO(json.dumps([dict(id=today,hrv=50)]).encode())
    monkeypatch.setattr(remote,'build_opener',lambda *args:Opener())
    assert remote.list_wellness(tmp_path)[0]['hrv']==50
    assert calls[0].get_method()=='GET'
    assert calls[0].full_url.startswith('https://intervals.icu/api/v1/athlete/0/wellness?')
    assert 'secret' not in calls[0].full_url
    assert not list(tmp_path.rglob('*.sqlite'))


@pytest.mark.parametrize('content',[b'{}',b'[null]',b'[{}]',b'not json',
    b'[{"id":"2099-01-01"}]',b'x'*(remote.MAX_RESPONSE_BYTES+1)])
def test_bad_remote_wellness_rejected_without_import(tmp_path,monkeypatch,content):
    remote.save_key(tmp_path,'test-wellness-secret')
    class Opener:
        def open(self,*args,**kwargs):return BytesIO(content)
    monkeypatch.setattr(remote,'build_opener',lambda *args:Opener())
    with pytest.raises(remote.ConnectionProblem):remote.list_wellness(tmp_path)


def test_remote_error_does_not_echo_body_or_credentials(tmp_path,monkeypatch):
    remote.save_key(tmp_path,'test-wellness-secret')
    class Opener:
        def open(self,r,**kwargs):raise HTTPError(r.full_url,403,'test-wellness-secret',{},BytesIO(b'secret'))
    monkeypatch.setattr(remote,'build_opener',lambda *args:Opener())
    with pytest.raises(remote.ConnectionProblem) as err:remote.list_wellness(tmp_path)
    assert 'secret' not in str(err.value)


def test_explicit_refresh_saves_private_context_without_fitness_or_hr_changes(tmp_path,monkeypatch):
    today=date.today().isoformat()
    monkeypatch.setattr(remote,'list_wellness',lambda *args,**kwargs:[dict(id=today,hrv=50,vo2max=40)])
    result=refresh_wellness(tmp_path)
    assert result['coverage']['hrv_ms']==1
    cached=read_wellness(tmp_path)
    assert cached['series'][0]['reported_vo2max_ml_kg_min']==40
    assert not cached['planner_influence'] and not cached['fitness_influence']
    assert cached['upstream_device']=='not specified by this endpoint'
    assert (tmp_path/'data/intervals-wellness.json').stat().st_mode&0o777==0o600
    assert not list(tmp_path.rglob('*.sqlite')) and not list(tmp_path.glob('*yaml'))
    before=(tmp_path/'data/intervals-wellness.json').read_bytes()
    def fail(*args,**kwargs):raise remote.ConnectionProblem('Unavailable')
    monkeypatch.setattr(remote,'list_wellness',fail)
    with pytest.raises(remote.ConnectionProblem):refresh_wellness(tmp_path)
    assert (tmp_path/'data/intervals-wellness.json').read_bytes()==before


def test_cache_symlink_is_not_followed(tmp_path,monkeypatch):
    (tmp_path/'elsewhere').mkdir();(tmp_path/'data').symlink_to(tmp_path/'elsewhere',target_is_directory=True)
    monkeypatch.setattr(remote,'list_wellness',lambda *args,**kwargs:[])
    with pytest.raises(remote.ConnectionProblem):refresh_wellness(tmp_path)
    assert list((tmp_path/'elsewhere').iterdir())==[]


def test_malformed_cache_is_unavailable_instead_of_crashing_sync(tmp_path):
    (tmp_path/'data').mkdir();target=tmp_path/'data/intervals-wellness.json'
    target.write_text(json.dumps(dict(version=1,series=[None])));target.chmod(0o600)
    assert read_wellness(tmp_path) is None
