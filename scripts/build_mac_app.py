"""Build an empty, reproducible Apple Silicon app; never copy a checkout wholesale."""
from argparse import ArgumentParser
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import sysconfig


ROOT=Path(__file__).resolve().parents[1]
VERSION='0.2.0'


def run(command, **kwargs):
    subprocess.run([str(x) for x in command],check=True,**kwargs)


def notices(target, runtime_notices):
    target.mkdir(parents=True,exist_ok=True)
    records=[]
    names=[line.split('==')[0] for line in (ROOT/'requirements-app.txt').read_text().splitlines()
           if line and not line.startswith('#')]+['pyinstaller']
    for name in names:
        package=distribution(name)
        dest=target/name
        found=[]
        for item in package.files or []:
            path=Path(str(item))
            if ('.dist-info' not in str(item) and name!='pyinstaller'):
                continue
            if any(label in path.name.lower() for label in ('license','licence','copying','notice','copyright')):
                source=Path(package.locate_file(item))
                if source.is_file():
                    start=next((i+1 for i,part in enumerate(path.parts) if part.endswith('.dist-info')),0)
                    relative=Path(*path.parts[start:])
                    out=dest/relative
                    out.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source,out)
                    found.append(relative.as_posix())
        if not found:
            raise ValueError(f'Missing installed license: {name}; review before distributing')
        records.append(dict(name=package.metadata['Name'],version=package.version,
            license=package.metadata.get('License-Expression') or package.metadata.get('License'),files=sorted(set(found))))
    # The exact unmodified MPL-covered source is available inside every app,
    # including the CA bundle. Users need no source-download service/account.
    certifi=distribution('certifi')
    shutil.copytree(certifi.locate_file('certifi'),target/'certifi-source',dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(next(p for p in (target/'certifi').rglob('*') if p.is_file() and p.name=='LICENSE'),target/'certifi-source'/'LICENSE')
    metadata=json.loads((runtime_notices/'PYTHON.json').read_text())
    if metadata['python_version']!='.'.join(map(str,sys.version_info[:3])):
        raise ValueError('Runtime notices must match the Python interpreter being bundled')
    if metadata['apple_sdk_deployment_target']!='11.0' or metadata['target_triple']!='aarch64-apple-darwin':
        raise ValueError('Use the reviewed portable Apple Silicon Python distribution')
    shutil.copytree(runtime_notices/'licenses',target/'python-runtime')
    shutil.copyfile(runtime_notices/'PYTHON.json',target/'python-runtime/PYTHON.json')
    (target/'THIRD-PARTY-NOTICES.txt').write_text(
        'Running Coach 0.2.0 — bundled third-party software\n\n'
        'Each component retains its own copyright and license. Full installed license texts are in this folder.\n'
        'The unmodified certifi source and CA bundle are in certifi-source (MPL-2.0).\n'
        'PyInstaller uses GPL-2.0-or-later WITH Bootloader-exception; this does not change the application code license.\n'
        'Apple frameworks are supplied by macOS and are not redistributed. No exercise-plan sources or personal data are bundled.\n')
    (target/'components.json').write_text(json.dumps(records,indent=2))
    return records


def build(output, runtime_notices):
    output=output.resolve()
    if output.exists() and (output/'dist/Running Coach.app').exists():
        raise ValueError('Choose a fresh build output directory; existing apps are never silently overwritten')
    output.mkdir(parents=True,exist_ok=True)
    components=notices(output/'licenses',runtime_notices)
    files=sorted((ROOT/'src/run_analysis').rglob('*.py'))+sorted((ROOT/'src/run_analysis/web/static').rglob('*.js'))
    files += [ROOT/'src/run_analysis/web/static/index.html',ROOT/'src/run_analysis/web/static/styles.css',ROOT/'config.example.yaml']
    files += sorted((ROOT/'desktop').glob('*'))+[ROOT/'pyproject.toml',ROOT/'requirements-app.txt',ROOT/'COPYRIGHT',Path(__file__)]
    digest=hashlib.sha256()
    for path in files:
        if not path.is_file():continue
        digest.update(path.relative_to(ROOT).as_posix().encode()+b'\0'+path.read_bytes())
    info=dict(version=VERSION,architecture='arm64',source_sha256=digest.hexdigest(),
        python=sys.version,components=components,contains_personal_data=False)
    (output/'build-info.json').write_text(json.dumps(info,indent=2))
    env=dict(os.environ,RUNNING_COACH_BUILD=str(output),PYINSTALLER_CONFIG_DIR=str(output/'cache'),
        PYTHONPATH=str(ROOT/'src'))
    run([sys.executable,'-m','PyInstaller','--noconfirm','--distpath',output/'backend-dist',
         '--workpath',output/'pyinstaller',ROOT/'desktop/RunningCoachServer.spec'],cwd=ROOT,env=env)
    app=output/'dist/Running Coach.app'
    macos=app/'Contents/MacOS';resources=app/'Contents/Resources'
    macos.mkdir(parents=True);resources.mkdir(parents=True)
    shutil.copytree(output/'backend-dist/RunningCoachServer',resources/'backend')
    shutil.copytree(output/'licenses',resources/'licenses')
    shutil.copyfile(output/'build-info.json',resources/'build-info.json')
    shutil.copyfile(ROOT/'COPYRIGHT',resources/'COPYRIGHT')
    # Reject an accidental Homebrew/system build whose native deployment target
    # is newer than the application's supported macOS version.
    native=[]
    for path in (resources/'backend').rglob('*'):
        if path.is_file() and path.read_bytes()[:4] in (b'\xcf\xfa\xed\xfe',b'\xca\xfe\xba\xbe'):
            native.append(str(path))
    commands='\n'.join(subprocess.run(['/usr/bin/vtool','-show-build',path],
        text=True,capture_output=True,check=True).stdout for path in native)
    minimums=re.findall(r'^\s+minos ([\d.]+)',commands,re.M)
    if not minimums or any(tuple(map(int,v.split('.')))>(14,0) for v in minimums):
        raise ValueError('A native dependency requires a newer macOS than 14; use the portable runtime')
    module_cache=output/'swift-module-cache';module_cache.mkdir()
    run(['/usr/bin/swiftc','-swift-version','5','-O','-target','arm64-apple-macosx14.0',
        '-module-cache-path',module_cache,'-framework','AppKit','-framework','WebKit',
        ROOT/'desktop/RunningCoach.swift','-o',macos/'RunningCoach'])
    run(['/usr/bin/swift','-module-cache-path',module_cache,ROOT/'desktop/Icon.swift',output/'icon.png'])
    icons=output/'RunningCoach.iconset';icons.mkdir()
    for size in (16,32,128,256,512):
        for scale in (1,2):
            name=f'icon_{size}x{size}'+('@2x' if scale==2 else '')+'.png'
            run(['/usr/bin/sips','-z',size*scale,size*scale,output/'icon.png','--out',icons/name],stdout=subprocess.DEVNULL)
    run(['/usr/bin/iconutil','-c','icns',icons,'-o',resources/'RunningCoach.icns'])
    plist=dict(CFBundleExecutable='RunningCoach',CFBundleIdentifier='com.gabemurray.runningcoach',
        CFBundleName='Running Coach',CFBundleDisplayName='Running Coach',CFBundlePackageType='APPL',
        CFBundleShortVersionString=VERSION,CFBundleVersion='2',LSMinimumSystemVersion='14.0',
        NSHighResolutionCapable=True,NSPrincipalClass='NSApplication',CFBundleIconFile='RunningCoach.icns',
        NSAppTransportSecurity={'NSAllowsLocalNetworking':True},
        NSLocalNetworkUsageDescription='Running Coach connects only to its private training engine on this Mac.',
        RunningCoachWorkspace='Running Coach')
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps(plist))
    # All bundled native dependencies are individually ad-hoc signed by
    # PyInstaller; sign the outer native launcher/bundle last. No notarization.
    run(['/usr/bin/codesign','--force','--deep','--sign','-',app])
    run(['/usr/bin/codesign','--verify','--deep','--strict',app])
    archive=output/f'Running-Coach-{VERSION}-macOS-Apple-Silicon.zip'
    run(['/usr/bin/ditto','-c','-k','--sequesterRsrc','--keepParent',app,archive])
    (output/'SHA256SUMS.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    print(json.dumps(dict(app=str(app),zip=str(archive),source_sha256=info['source_sha256']),indent=2))


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-notices',type=Path,required=True)
    args=parser.parse_args()
    build(args.output,args.runtime_notices)
