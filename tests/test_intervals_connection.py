from io import BytesIO
import base64
import gzip
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError

from fastapi.testclient import TestClient
import pytest

from run_analysis import intervals_connection as service
from run_analysis.web.app import create_app


KEY = "test-secret-intervals-key"
HEADERS = {"X-Run-Analysis-Local": "1"}
URL = "/api/connections/intervals"


def test_storage_private_atomic_and_project_local(tmp_path):
    assert service.connection_status(tmp_path)["configured"] is False
    assert not (tmp_path / "data").exists()
    response = service.save_key(tmp_path, KEY)
    path = tmp_path / "data/credentials/intervals.json"
    assert response["configured"] and KEY not in json.dumps(response)
    assert json.loads(path.read_text()) == {"api_key": KEY}
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    assert service.connection_status(tmp_path / "other")["configured"] is False
    service.save_key(tmp_path, "replacement-secret")
    assert KEY not in path.read_text()
    assert list(path.parent.iterdir()) == [path]
    assert not service.remove_key(tmp_path)["configured"]
    assert not path.exists()
    assert not service.remove_key(tmp_path)["configured"]


@pytest.mark.parametrize("value", ["", "tiny", "a b cdefgh", "a\nbcdefgh", "secret\x00bad", "a" * 513])
def test_invalid_keys_do_not_write_or_echo(tmp_path, value):
    with pytest.raises(service.ConnectionProblem) as error:
        service.save_key(tmp_path, value)
    if value:
        assert value not in str(error.value)
    assert not (tmp_path / "data").exists()


def test_symlink_and_insecure_file_refused(tmp_path):
    real = tmp_path / "other"
    real.mkdir()
    (tmp_path / "data").symlink_to(real, target_is_directory=True)
    with pytest.raises(service.ConnectionProblem):
        service.save_key(tmp_path, KEY)
    assert list(real.iterdir()) == []
    (tmp_path / "data").unlink()
    service.save_key(tmp_path, KEY)
    path = tmp_path / "data/credentials/intervals.json"
    if os.name == "posix":
        path.chmod(0o644)
        assert not service.connection_status(tmp_path)["available"]
        service.save_key(tmp_path, KEY)
        assert path.stat().st_mode & 0o777 == 0o600


def _remote(monkeypatch, content):
    requests = []
    class Opener:
        def open(self, request, timeout):
            requests.append((request, timeout))
            return BytesIO(content)
    monkeypatch.setattr(service, "build_opener", lambda *args: Opener())
    return requests


@pytest.mark.parametrize("activities", [[], [{"id": "i123", "type": "Run"}]])
def test_check_accepts_empty_history_and_only_returns_counts(tmp_path, monkeypatch, activities):
    service.save_key(tmp_path, KEY)
    requests = _remote(monkeypatch, json.dumps(activities).encode())
    result = service.check_connection(tmp_path)
    assert result["connected"] and result["recent_activity_count"] == len(activities)
    assert result["automatic_import"] is False
    assert "activities" not in result and KEY not in json.dumps(result)
    request, timeout = requests[0]
    assert request.get_method() == "GET" and timeout == 15
    assert request.full_url.startswith("https://intervals.icu/api/v1/athlete/0/activities?")
    assert KEY not in request.full_url
    assert request.headers["Authorization"].startswith("Basic ")
    assert base64.b64decode(request.headers["Authorization"].split(" ", 1)[1]).decode() == f"API_KEY:{KEY}"
    assert request.headers["User-agent"] == service.API_USER_AGENT
    assert not (tmp_path / "uploads").exists()


def _fit_bytes(body=b'recorded-data'):
    return bytes([14, 16])+b'\0\0'+len(body).to_bytes(4,'little')+b'.FIT\0\0'+body+b'\0\0'


@pytest.mark.parametrize('compressed',[False,True])
def test_original_fit_download_is_lossless_and_does_not_import(tmp_path,monkeypatch,compressed):
    service.save_key(tmp_path,KEY)
    original=_fit_bytes()
    requests=_remote(monkeypatch,gzip.compress(original) if compressed else original)
    assert service.download_original_fit(tmp_path,'i123')==original
    request,timeout=requests[0]
    assert request.full_url=='https://intervals.icu/api/v1/activity/i123/file'
    assert request.get_method()=='GET' and timeout==20
    assert request.headers['User-agent']==service.API_USER_AGENT
    assert not (tmp_path/'uploads').exists()
    assert not list(tmp_path.rglob('*.sqlite'))


@pytest.mark.parametrize('content',[b'',b'<html>Not FIT</html>',b'<?xml tcx?>',
    _fit_bytes()[:-1],_fit_bytes()+b'extra',b'\x1f\x8btruncated'])
def test_original_download_rejects_invalid_content(tmp_path,monkeypatch,content):
    service.save_key(tmp_path,KEY); _remote(monkeypatch,content)
    with pytest.raises(service.ConnectionProblem):
        service.download_original_fit(tmp_path,'i123')


@pytest.mark.parametrize('compressed',[False,True])
def test_original_download_bounds_both_transport_and_expansion(tmp_path,monkeypatch,compressed):
    service.save_key(tmp_path,KEY)
    monkeypatch.setattr(service,'MAX_FIT_BYTES',128)
    original=_fit_bytes(b'x'*200)
    _remote(monkeypatch,gzip.compress(original) if compressed else original)
    with pytest.raises(service.ConnectionProblem,match='exceeds'):
        service.download_original_fit(tmp_path,'i123')


@pytest.mark.parametrize('activity_id',['../../elsewhere','i123?key=secret','https://other/','i123\r\nX:evil',None])
def test_original_download_rejects_path_injection_before_network(tmp_path,monkeypatch,activity_id):
    requests=_remote(monkeypatch,b'')
    with pytest.raises(service.ConnectionProblem,match='activity ID'):
        service.download_original_fit(tmp_path,activity_id)
    assert requests==[]


@pytest.mark.parametrize('code',[401,403,404,429,500,302])
def test_original_download_errors_do_not_expose_remote_secrets(tmp_path,monkeypatch,code):
    service.save_key(tmp_path,KEY)
    class Opener:
        def open(self,request,timeout):
            raise HTTPError(request.full_url,code,KEY,{'secret':KEY},BytesIO(KEY.encode()))
    monkeypatch.setattr(service,'build_opener',lambda *args:Opener())
    with pytest.raises(service.ConnectionProblem) as error:
        service.download_original_fit(tmp_path,'i123')
    assert KEY not in str(error.value)


@pytest.mark.parametrize("code", [401, 403, 429, 500, 302])
def test_remote_errors_do_not_leak_headers_body_or_key(tmp_path, monkeypatch, code):
    service.save_key(tmp_path, KEY)
    class Opener:
        def open(self, request, timeout):
            raise HTTPError(request.full_url, code, KEY, {"secret": KEY}, BytesIO(KEY.encode()))
    monkeypatch.setattr(service, "build_opener", lambda *args: Opener())
    with pytest.raises(service.ConnectionProblem) as error:
        service.check_connection(tmp_path)
    assert KEY not in str(error.value)
    if code == 403:
        assert "does not necessarily mean" in str(error.value)
    if code == 401:
        assert "authentication" in str(error.value)


def test_documented_user_agent_avoids_default_python_client_rejection(tmp_path, monkeypatch):
    service.save_key(tmp_path, KEY)
    class Opener:
        def open(self, request, timeout):
            if request.headers.get("User-agent") != service.API_USER_AGENT:
                raise HTTPError(request.full_url, 403, "Forbidden", {}, BytesIO(b"{}"))
            return BytesIO(b"[]")
    monkeypatch.setattr(service, "build_opener", lambda *args: Opener())
    assert service.check_connection(tmp_path)["connected"] is True


@pytest.mark.parametrize("content", [b"not JSON", b"{}", b"[1]", b"[null]", b"x" * (service.MAX_RESPONSE_BYTES + 1)])
def test_check_rejects_invalid_or_oversized_responses(tmp_path, monkeypatch, content):
    service.save_key(tmp_path, KEY)
    _remote(monkeypatch, content)
    with pytest.raises(service.ConnectionProblem):
        service.check_connection(tmp_path)


def test_offline_and_missing_key(tmp_path, monkeypatch):
    with pytest.raises(service.ConnectionProblem, match="Save"):
        service.check_connection(tmp_path)
    service.save_key(tmp_path, KEY)
    class Opener:
        def open(self, request, timeout):
            raise URLError(KEY)
    monkeypatch.setattr(service, "build_opener", lambda *args: Opener())
    with pytest.raises(service.ConnectionProblem, match="securely"):
        service.check_connection(tmp_path)
    assert service._NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere/") is None


def test_routes_save_without_network_config_or_planner(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Saving connection credentials must not contact Intervals or replan")
    monkeypatch.setattr(service, "build_opener", forbidden)
    client = TestClient(create_app(tmp_path))  # No config/database needed.
    response = client.put(URL, json={"api_key": KEY}, headers=HEADERS)
    assert response.status_code == 200 and response.json()["configured"]
    assert KEY not in response.text
    assert KEY not in client.get(URL).text
    assert response.headers["Cache-Control"] == "no-store"
    assert not (tmp_path / "config.local.yaml").exists()
    assert not (tmp_path / "uploads").exists()
    assert not list(tmp_path.rglob("*.sqlite"))
    assert client.delete(URL, headers=HEADERS).status_code == 200


@pytest.mark.parametrize("payload", [{"api_key": {"secret": KEY}}, {"api_key": KEY, "extra": KEY}, [KEY], {"api_key": None}])
def test_route_validation_never_echoes_secret(tmp_path, payload):
    response = TestClient(create_app(tmp_path)).put(URL, json=payload, headers=HEADERS)
    assert response.status_code == 422 and KEY not in response.text
    assert not (tmp_path / "data").exists()


def test_cross_site_management_and_oversized_payloads_blocked(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.put(URL, json={"api_key": KEY}).status_code == 403
    foreign = {**HEADERS, "Origin": "https://evil.example"}
    assert client.put(URL, json={"api_key": KEY}, headers=foreign).status_code == 403
    assert client.get(URL, headers=foreign).status_code == 403
    assert client.post(URL + "/check", headers=foreign).status_code == 403
    assert client.delete(URL, headers=foreign).status_code == 403
    assert client.put(URL, json={"api_key": KEY}, headers={**HEADERS, "Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.put(URL, content="api_key=" + KEY, headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"}).status_code == 415
    assert client.put(URL, content='{"api_key":"' + KEY, headers={**HEADERS, "Content-Type": "application/json"}).status_code == 422
    assert client.put(URL, json={"api_key": "x" * 5000}, headers=HEADERS).status_code == 413
    assert not (tmp_path / "data").exists()
    assert client.put(URL, json={"api_key": KEY}, headers={**HEADERS, "Origin": "http://testserver"}).status_code == 200


def test_route_check_empty_account_is_success(tmp_path, monkeypatch):
    service.save_key(tmp_path, KEY)
    _remote(monkeypatch, b"[]")
    response = TestClient(create_app(tmp_path)).post(URL + "/check", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["connected"] and response.json()["recent_activity_count"] == 0
    assert KEY not in response.text
