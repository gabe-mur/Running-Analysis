"""Real FIT extraction, recency, mode isolation and frozen-planner controls."""
from datetime import datetime, timedelta, timezone
from copy import deepcopy
from io import BytesIO
import json
import struct
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from fit_builder import FitWriter, UINT8, UINT16, UINT32, ENUM, fit_timestamp
from run_analysis.config import load_config
from run_analysis.db import connect, initialize
from run_analysis.fit import parse_fit
from run_analysis.importer import import_files
from run_analysis.settings_service import save_settings_overlay, refresh_watch_zones, _effective_changed_fields
from run_analysis.watch_zones import extract_watch_hr, latest_watch_zones
from run_analysis.web.app import create_app
from run_analysis.web.schemas import SettingsPatch
from run_analysis.web.upload_service import UploadPayload, run_upload_pipeline, complete_source_for_digest
from test_web_phase1 import _write_config
from test_upload_noop import stub_downstream

AT=datetime(2025,10,10,14,tzinfo=timezone.utc)
EDGES=(126,139,153,166,181,194)
ZONES={'z1':[127,139],'z2':[140,153],'z3':[154,166],'z4':[167,181],'z5':[182,194]}


def recorded_report(edges=EDGES):
    return dict(reference_index=0,hr_zone_high_boundary=edges,hr_calc_type='percent_hrr',
        max_heart_rate=194,resting_heart_rate=45,threshold_heart_rate=175)


def make_watch_fit(path, *, at=AT, edges=EDGES, reference=18, sport=1, reports=1, maximum=194, resting=45):
    w=FitWriter()
    for i in range(3):
        w.message(20,{253:(UINT32,fit_timestamp(at+timedelta(seconds=10*i))),
            3:(UINT8,145),5:(UINT32,3000*i)})
    w.message(18,{2:(UINT32,fit_timestamp(at)),5:(ENUM,sport),
        7:(UINT32,20000),8:(UINT32,20000),9:(UINT32,6000)})
    for index in range(reports):
        # Global 216 includes a real uint8 ARRAY, including FIT invalid values.
        fields=[(0,2,UINT16),(1,2,UINT16),(6,len(edges),UINT8),
            (10,1,ENUM),(11,1,UINT8),(12,1,UINT8),(13,1,UINT8)]
        w._data += bytes([0x4f])+struct.pack('<BBHB',0,0,216,len(fields))
        for f in fields:w._data += bytes(f)
        w._data += bytes([0x0f])+struct.pack('<HH',reference,0)
        w._data += bytes(255 if e is None else e for e in edges)+bytes([2,maximum,resting,175])
    path.write_bytes(w.build())
    return path


def ingest(db, root, path):
    summary=import_files(db,root,'UTC',paths=[path])
    assert not summary.failed_files
    return summary


def test_real_session_array_includes_below_z1_boundary_and_raw_provenance(tmp_path):
    activity=parse_fit(make_watch_fit(tmp_path/'watch.fit')).activities[0]
    watch=activity.sensor_stream.watch_hr
    assert watch['status']=='recorded' and watch['zones']==ZONES
    assert watch['maximum_hr']==194 and watch['resting_hr']==45
    assert watch['calculation']=='percent_hrr'
    assert watch['reports'][0]['hr_zone_high_boundary']==EDGES
    assert activity.sensor_stream.extraction_version==3


@pytest.mark.parametrize('edges',[(139,153,166,181,194),(126,139,None,166,181,194),
    (126,139,139,166,181,194),(126,139,153,166,181,251)])
def test_incomplete_or_invalid_boundaries_do_not_infer_default_zones(edges):
    assert extract_watch_hr([recorded_report(edges)],sport='running',session_count=1)['status']!='recorded'


def test_calc_method_alone_is_not_zone_evidence():
    report=recorded_report();report.pop('hr_zone_high_boundary')
    assert extract_watch_hr([report],sport='running',session_count=1)['status']=='incomplete_boundaries'


def test_report_must_belong_to_the_single_supported_session():
    report=recorded_report();report['reference_index']=1
    assert extract_watch_hr([report],sport='running',session_count=1)['status']=='unsupported_session_index'


@pytest.mark.parametrize('reference,sport',[(19,1),(18,2)])
def test_lap_reports_and_cycling_do_not_change_running_zones(tmp_path,reference,sport):
    stream=parse_fit(make_watch_fit(tmp_path/'other.fit',reference=reference,sport=sport)).activities[0].sensor_stream
    assert stream.watch_hr['status']!='recorded'


def test_conflicting_or_multiple_session_settings_are_not_selected_by_order():
    a=recorded_report();b=recorded_report((127,140,154,167,182,194))
    assert extract_watch_hr([a,b],sport='running',session_count=1)['status']=='conflicting_reports'
    assert extract_watch_hr([a],sport='running',session_count=2)['status']=='unsupported_activity'
    assert extract_watch_hr([a,a],sport='running',session_count=1)['status']=='recorded'


def test_newest_run_wins_old_bulk_imports_missing_reports_and_future_do_not(tmp_path):
    with connect(tmp_path/'data.sqlite') as db:
        ingest(db,tmp_path,make_watch_fit(tmp_path/'new.fit'))
        ingest(db,tmp_path,make_watch_fit(tmp_path/'old.fit',at=AT-timedelta(days=4),edges=(125,138,152,165,180,193)))
        ingest(db,tmp_path,make_watch_fit(tmp_path/'missing.fit',at=AT+timedelta(days=1),edges=(None,)*6))
        ingest(db,tmp_path,make_watch_fit(tmp_path/'future.fit',at=AT+timedelta(days=3),edges=(127,140,154,167,182,194)))
        result=latest_watch_zones(db,as_of=AT+timedelta(days=2))
        assert result['zones']==ZONES and datetime.fromisoformat(result['recorded_at'])==AT


def test_same_time_conflicting_fit_sources_cannot_win_by_filename(tmp_path):
    with connect(tmp_path/'data.sqlite') as db:
        ingest(db,tmp_path,make_watch_fit(tmp_path/'a.fit'))
        ingest(db,tmp_path,make_watch_fit(tmp_path/'b.fit',edges=(127,140,154,167,182,194)))
        assert db.execute('SELECT count(*) FROM activities').fetchone()[0]==1
        with pytest.raises(ValueError,match='Conflicting watch'):
            latest_watch_zones(db,as_of=AT+timedelta(days=1))


def test_version_one_backfill_preserves_canonical_activity_source_and_import_time(tmp_path):
    path=make_watch_fit(tmp_path/'watch.fit')
    with connect(tmp_path/'data.sqlite') as db:
        ingest(db,tmp_path,path)
        before=dict(db.execute('SELECT * FROM activities').fetchone())
        source=dict(db.execute('SELECT * FROM source_files').fetchone())
        old=json.loads(db.execute('SELECT payload_json FROM activity_sensor_streams').fetchone()[0])
        old.pop('watch_hr');old['extraction_version']=1
        db.execute('UPDATE activity_sensor_streams SET extraction_version=1,payload_json=?',(json.dumps(old),));db.commit()
        assert complete_source_for_digest(db,source['sha256'],fit=True) is None
        state=tuple(db.iterdump())
        assert latest_watch_zones(db)['zones']==ZONES  # Read-only existing-file fallback.
        assert tuple(db.iterdump())==state
        result=ingest(db,tmp_path,path)
        assert result.unchanged_files==1 and result.evidence_changed_activity_ids==[before['id']]
        assert dict(db.execute('SELECT * FROM activities').fetchone())==before
        assert dict(db.execute('SELECT * FROM source_files').fetchone())==source
        assert latest_watch_zones(db)['zones']==ZONES
        assert complete_source_for_digest(db,source['sha256'],fit=True)==path.resolve()
        again=ingest(db,tmp_path,path)
        assert not again.evidence_changed_activity_ids


@pytest.mark.parametrize('mode',['custom','percent_max','heart_rate_reserve'])
def test_non_watch_modes_keep_user_ranges(tmp_path,mode):
    _write_config(tmp_path)
    with connect(tmp_path/'data/test.sqlite') as db:
        ingest(db,tmp_path,make_watch_fit(tmp_path/'watch.fit'))
        config=save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(setup={'zone_method':mode}),connection=db)
        before=(tmp_path/'config.local.yaml').read_bytes()
        updated,changed=refresh_watch_zones(tmp_path/'config.yaml',db)
        assert not changed and updated['zones']==config['zones']
        assert (tmp_path/'config.local.yaml').read_bytes()==before


def test_watch_save_and_refresh_own_the_entire_current_hr_bundle(tmp_path):
    _write_config(tmp_path)
    with connect(tmp_path/'data/test.sqlite') as db:
        ingest(db,tmp_path,make_watch_fit(tmp_path/'watch.fit'))
        before=load_config(tmp_path/'config.yaml')
        patch=SettingsPatch(setup={'zone_method':'device','comparison_hr_mode':'automatic'})
        config=save_settings_overlay(tmp_path/'config.yaml',patch,connection=db)
        assert config['zones']==ZONES
        assert config['target_hr']==146
        assert config['max_hr']==194 and config['resting_hr']==45
        assert not config.get('profile')  # Import must not fabricate a partial personal profile.
        assert config['watch_zone_import']['resting_hr']==45
        assert _effective_changed_fields(before,config,patch.model_fields_set)>={'zones','target_hr','max_hr','resting_hr'}
        bytes_before=(tmp_path/'config.local.yaml').read_bytes()
        updated,changed=refresh_watch_zones(tmp_path/'config.yaml',db)
        assert not changed and updated==config
        assert (tmp_path/'config.local.yaml').read_bytes()==bytes_before


def test_persisted_recency_prevents_old_restored_sources_from_rolling_back_zones(tmp_path):
    _write_config(tmp_path)
    with connect(tmp_path/'data/test.sqlite') as db:
        ingest(db,tmp_path,make_watch_fit(tmp_path/'new.fit'))
        config=save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(setup={'zone_method':'device'}),connection=db)
        # Simulate a restored DB without its newest original source/linkage.
        db.execute('DELETE FROM activity_sensor_streams');db.execute('DELETE FROM activity_sources');db.commit()
        ingest(db,tmp_path,make_watch_fit(tmp_path/'old.fit',at=AT-timedelta(days=1),edges=(125,138,152,165,180,193)))
        updated,changed=refresh_watch_zones(tmp_path/'config.yaml',db)
        assert not changed and updated['zones']==config['zones']
        assert updated['watch_zone_import']==config['watch_zone_import']


def test_unverified_active_reference_cannot_be_rebased_by_watch_settings(tmp_path):
    _write_config(tmp_path)
    with connect(tmp_path/'data/test.sqlite') as db:
        initialize(db)
        zones=deepcopy(load_config(tmp_path/'config.yaml')['zones']);zones['z5'][1]=190
        config=save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(max_hr=190,
            zones={k:dict(minimum_bpm=v[0],maximum_bpm=v[1]) for k,v in zones.items()},
            setup={'zone_method':'custom'}),connection=db)
        with (tmp_path/'config.local.yaml').open('a') as stream:
            stream.write('adaptive_planner:\n  engine: adaptive_frequency_v3\n')
        ingest(db,tmp_path,make_watch_fit(tmp_path/'watch.fit'))
        config=load_config(tmp_path/'config.yaml')
        before=(tmp_path/'config.local.yaml').read_bytes()
        with pytest.raises(ValueError,match='No settings were saved'):
            save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(setup={'zone_method':'device'}),connection=db)
        assert (tmp_path/'config.local.yaml').read_bytes()==before
        assert load_config(tmp_path/'config.yaml')==config


def test_watch_preview_agrees_with_save_and_get_settings_is_read_only(tmp_path):
    _write_config(tmp_path)
    with connect(tmp_path/'data/test.sqlite') as db:
        ingest(db,tmp_path,make_watch_fit(tmp_path/'watch.fit'))
    client=TestClient(create_app(tmp_path))
    old=client.get('/api/settings').json()
    before=tuple(p.name for p in tmp_path.glob('config.local*'))
    preview=client.post('/api/setup/zone-preview',json=dict(method='device',max_hr=old['max_hr'],
        resting_hr=old['resting_hr'],boundaries=old['zones']))
    assert preview.status_code==200,preview.text
    assert preview.json()['zones']['z2']=={'minimum_bpm':140,'maximum_bpm':153}
    assert preview.json()['max_hr']==194 and preview.json()['resting_hr']==45
    assert tuple(p.name for p in tmp_path.glob('config.local*'))==before
    assert client.get('/api/settings').json()['zones']==old['zones']


def test_watch_selection_requires_real_evidence_not_manual_fallback(tmp_path):
    _write_config(tmp_path)
    with connect(tmp_path/'data/test.sqlite') as db:
        initialize(db)
        with pytest.raises(ValueError,match='Import a FIT'):
            save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(setup={'zone_method':'device'}),connection=db)
    assert not (tmp_path/'config.local.yaml').exists()
    client=TestClient(create_app(tmp_path));old=client.get('/api/settings').json()
    response=client.post('/api/setup/zone-preview',json=dict(method='device',
        max_hr=old['max_hr'],resting_hr=old['resting_hr'],boundaries=old['zones']))
    assert response.status_code==422 and 'Import a FIT' in response.text


def test_watch_preview_does_not_validate_against_stale_custom_maximum(tmp_path):
    _write_config(tmp_path)
    with connect(tmp_path/'data/test.sqlite') as db:ingest(db,tmp_path,make_watch_fit(tmp_path/'watch.fit'))
    response=TestClient(create_app(tmp_path)).post('/api/setup/zone-preview',json=dict(
        method='device',max_hr=170,resting_hr=90))
    assert response.status_code==200,response.text
    assert response.json()['max_hr']==194 and response.json()['resting_hr']==45
    watch_only=TestClient(create_app(tmp_path)).post('/api/setup/zone-preview',json=dict(method='device'))
    assert watch_only.status_code==200,watch_only.text
    assert watch_only.json()['max_hr']==194 and watch_only.json()['resting_hr']==45


def test_zip_upload_updates_before_processing_and_is_noop_on_retry(tmp_path,monkeypatch):
    _write_config(tmp_path)
    # Fixture awaits its first FIT; the UI cannot confirm watch calibration
    # before actual recorded evidence is available.
    (tmp_path/'config.local.yaml').write_text('setup:\n  zone_method: device\n  comparison_hr_mode: automatic\n')
    content=make_watch_fit(tmp_path/'fixture.fit').read_bytes()
    buffer=BytesIO()
    with ZipFile(buffer,'w') as archive:archive.writestr('garmin/run.fit',content)
    calls=stub_downstream(monkeypatch)
    seen=[]
    from test_upload_noop import Summary
    def process(db,config,**kwargs):
        seen.append((deepcopy(config['zones']),kwargs));return Summary()
    monkeypatch.setattr('run_analysis.web.upload_service.process_activities',process)
    payload=UploadPayload('run.zip',buffer.getvalue())
    result=run_upload_pipeline(tmp_path,'config.yaml',[payload])
    assert next(s.status for s in result.stages if s.name=='watch_zones')=='complete'
    assert seen==[(ZONES,{'force':True})]
    assert sum(name=='generate_weekly_schedule' for name,_ in calls)==1
    calls.clear();seen.clear()
    result=run_upload_pipeline(tmp_path,'config.yaml',[payload])
    assert result.files[0].status=='unchanged'
    assert not seen and not calls
