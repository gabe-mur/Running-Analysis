"""Put an audited app and separate empty-profile demo in a fresh output folder."""
from argparse import ArgumentParser
from pathlib import Path
import plistlib
import shutil
import subprocess


def prepare(build, destination):
    if destination.exists():
        raise ValueError('Choose a fresh destination; existing apps will not be overwritten')
    source = build / 'dist/Running Coach.app'
    assert source.is_dir()
    destination.mkdir(parents=True)
    shutil.copytree(source, destination / source.name, symlinks=True)
    clean = destination / 'Running Coach Clean.app'
    shutil.copytree(source, clean, symlinks=True)
    path = clean / 'Contents/Info.plist'
    plist = plistlib.loads(path.read_bytes())
    plist.update(CFBundleIdentifier='com.gabemurray.runningcoach.clean',
        CFBundleName='Running Coach Clean', CFBundleDisplayName='Running Coach Clean',
        RunningCoachWorkspace='Running Coach Clean')
    path.write_bytes(plistlib.dumps(plist))
    subprocess.run(['/usr/bin/codesign', '--force', '--deep', '--sign', '-', str(clean)], check=True)
    subprocess.run(['/usr/bin/codesign', '--verify', '--deep', '--strict', str(clean)], check=True)
    for pattern in ('*.zip', 'SHA256SUMS.txt', 'build-info.json'):
        for item in build.glob(pattern):
            shutil.copyfile(item, destination / item.name)
    return destination


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('build', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(prepare(args.build.resolve(), args.destination.resolve()))
