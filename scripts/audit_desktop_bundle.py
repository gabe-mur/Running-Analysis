"""Audit the distributable and exercise the frozen engine without Python on PATH.

Uses synthetic FIT data only. Temporary workspaces are retained for inspection.
No credentials, live activities or external-service requests are required.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def audit(app):
    resources = app / 'Contents/Resources'
    internal = resources / 'backend/_internal'
    assert (resources / 'RunningCoach.icns').is_file()
    assert (resources / 'COPYRIGHT').is_file()
    for path in (ROOT / 'src/run_analysis').rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.js', '.css', '.html'):
            bundled = internal / path.relative_to(ROOT / 'src')
            assert bundled.is_file(), f'Runtime resource missing: {path.relative_to(ROOT)}'
            assert bundled.read_bytes() == path.read_bytes(), f'Runtime resource differs: {path.relative_to(ROOT)}'
    assert (internal / 'certifi/cacert.pem').stat().st_size > 100_000
    assert (resources / 'licenses/certifi-source/core.py').exists()
    forbidden = {'.fit', '.tcx', '.gpx', '.sqlite', '.db', '.hgt', '.xlsx'}
    for path in app.rglob('*'):
        if not path.is_file():
            continue
        assert path.suffix.lower() not in forbidden, f'Private data extension: {path.name}'
        assert path.name not in ('config.yaml', 'config.local.yaml', 'intervals.json', 'run_overrides.csv')
        assert 'credentials' not in path.relative_to(app).parts
    subprocess.run(['/usr/bin/codesign', '--verify', '--deep', '--strict', str(app)], check=True)
    plist = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
    assert plist['LSMinimumSystemVersion'] == '14.0'
    assert plist['CFBundleIdentifier'] == 'com.gabemurray.runningcoach'
    # Validate load commands, not just whether dependencies happen to exist on
    # this development computer. All non-system dylibs must be bundled.
    native = []
    for path in (resources / 'backend').rglob('*'):
        if path.is_file() and path.read_bytes()[:4] in (b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe'):
            native.append(str(path))
    native.append(str(app / 'Contents/MacOS/RunningCoach'))
    commands = subprocess.run(['/usr/bin/otool', '-L', *native], text=True, capture_output=True, check=True).stdout
    dependencies = [line.strip().split(' (')[0] for line in commands.splitlines() if line.startswith('\t')]
    assert all(x.startswith(('@', '/usr/lib/', '/System/Library/')) for x in dependencies), 'Unbundled native dependency'
    versions='\n'.join(subprocess.run(['/usr/bin/vtool','-show-build',path],
        text=True,capture_output=True,check=True).stdout for path in native)
    minimums=re.findall(r'^\s+minos ([\d.]+)',versions,re.M)
    assert minimums and all(tuple(map(int,v.split('.'))) <= (14,0) for v in minimums), 'Native minimum macOS exceeds 14'
    return dict(runtime_resources_match=True, private_files_absent=True, signature_valid=True,
                native_dependencies_portable=True, minimum_macos_checked=True, native_objects=len(native))


def request(base, path, *, content=None, content_type=None):
    headers = {'Content-Type': content_type} if content_type else {}
    with urlopen(Request(base + path, data=content, headers=headers), timeout=120) as response:
        data = response.read()
        assert response.status == 200, (path, response.status)
        return json.loads(data) if 'json' in response.headers.get('Content-Type', '') else data


def start(app, workspace):
    ready = workspace.parent / (workspace.name + '-ready.json')
    ready.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'DYLD_'))}
    env['PATH'] = '/usr/bin:/bin'
    log = (workspace.parent / (workspace.name + '-server.log')).open('ab')
    began = time.monotonic()
    process = subprocess.Popen([str(app / 'Contents/Resources/backend/RunningCoachServer'),
        '--data-dir', str(workspace), '--ready-file', str(ready)],
        cwd=workspace.parent, env=env, stdout=log, stderr=log)
    try:
        while time.monotonic() - began < 60:
            if process.poll() is not None:
                raise RuntimeError(f'Frozen engine exited; inspect {log.name}')
            if ready.exists():
                value = json.loads(ready.read_text())
                base = value['url'].split('#')[0].rstrip('/')
                assert request(base, '/api/desktop-health')['version'] == '0.2.0'
                return process, log, base, value['url'], round(time.monotonic() - began, 2)
            time.sleep(.1)
        raise RuntimeError(f'Frozen startup timeout; inspect {log.name}')
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        log.close()
        raise


def stop(process, log):
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=60)
    log.close()
    # Uvicorn restores and re-raises SIGTERM after graceful application shutdown.
    assert process.returncode in (0, -signal.SIGTERM)


def smoke(app):
    workspace_root = Path(tempfile.mkdtemp(prefix='Running Coach packaged test '))
    workspace = workspace_root / 'first'
    process, log, base, url, startup = start(app, workspace)
    try:
        assert url.endswith('#setup')
        assert request(base, '/api/runs') == []
        for path in ('/', '/assets/app/main.js', '/api/setup', '/api/settings', '/api/dashboard', '/api/progress-wip', '/api/weekly-schedule/saved'):
            request(base, path)
        sys.path.insert(0, str(ROOT / 'tests'))
        from fit_builder import build_run
        fit = build_run(workspace_root / 'synthetic.fit', start=datetime(2026, 10, 20, 12, tzinfo=timezone.utc), points=121)
        boundary = 'RunningCoachSyntheticUpload'
        content = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="synthetic.fit"\r\n'
                   'Content-Type: application/octet-stream\r\n\r\n').encode() + fit + f'\r\n--{boundary}--\r\n'.encode()
        response = request(base, '/api/uploads', content=content, content_type='multipart/form-data; boundary=' + boundary)
        assert not [s for s in response['stages'] if s['status'] == 'failed'], response['stages']
        assert next(s['status'] for s in response['stages'] if s['name'] == 'schedule') == 'deferred'
        runs = request(base, '/api/runs')
        assert len(runs) == 1
        request(base, f"/api/runs/{runs[0]['activity_id']}")
        duplicate = request(base, '/api/uploads', content=content, content_type='multipart/form-data; boundary=' + boundary)
        assert duplicate['files'][0]['status'] == 'unchanged'
        assert len(request(base, '/api/runs')) == 1
    finally:
        stop(process, log)
    process, log, base, url, restart = start(app, workspace)
    try:
        assert '#setup' not in url
        assert len(request(base, '/api/runs')) == 1
    finally:
        stop(process, log)
    process, log, base, url, _ = start(app, workspace_root / 'separate clean')
    try:
        assert request(base, '/api/runs') == []
    finally:
        stop(process, log)
    return dict(workspaces=str(workspace_root), startup_seconds=startup, restart_seconds=restart,
                synthetic_fit_upload=True, analysis=True, duplicate_idempotent=True,
                planner_deferred=True, restart_retains_history=True, profiles_isolated=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('app', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    result = audit(args.app.resolve())
    if args.smoke:
        result['smoke'] = smoke(args.app.resolve())
    print(json.dumps(result, indent=2))
