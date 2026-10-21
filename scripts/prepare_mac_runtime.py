"""Download the pinned portable runtime and notices into a local build folder.

Developer tool only, never launched or downloaded by an installed app. The
published SHA-256 is checked before extracting or executing the interpreter.
"""
from argparse import ArgumentParser
import hashlib
from pathlib import Path
import platform
import subprocess


ARCHIVE = 'cpython-3.14.6+20260804-aarch64-apple-darwin-pgo+lto-full.tar.zst'
URL = 'https://github.com/astral-sh/python-build-standalone/releases/download/20260804/' + ARCHIVE.replace('+', '%2B')
SHA256 = '1abf4d2c56f94537dc178c7e7156f7bee370c2d0779b0b79308b935018303ac0'


def prepare(output):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Build on an Apple Silicon Mac')
    if output.exists():
        raise ValueError('Choose a fresh runtime folder; no existing environment is replaced')
    output.mkdir(parents=True)
    archive = output / ARCHIVE
    subprocess.run(['/usr/bin/curl', '--fail', '--location', '--proto', '=https', '--output', str(archive), URL], check=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Runtime archive digest differs from the reviewed published release')
    subprocess.run(['/usr/bin/tar', '-xf', str(archive), '-C', str(output),
        'python/install', 'python/licenses', 'python/PYTHON.json'], check=True)
    environment = output / 'env'
    subprocess.run([str(output / 'python/install/bin/python3.14'), '-m', 'venv', str(environment)], check=True)
    repository = Path(__file__).resolve().parents[1]
    subprocess.run([str(environment / 'bin/python'), '-m', 'pip', 'install', '-r',
        str(repository / 'requirements-app.txt'), 'pyinstaller==6.22.3'], check=True)
    print('Runtime prepared:', environment)
    print('Build with:', environment / 'bin/python', 'scripts/build_mac_app.py --runtime-notices', output / 'python', '--output build/mac-release')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
