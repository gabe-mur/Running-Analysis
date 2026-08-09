"""Cross-platform setup and launcher for the local application.

Written to be double-clicked as much as typed. Someone who has never opened a
terminal should be able to get from a downloaded folder to a working app
without knowing that a virtual environment, a package install, or a localhost
URL exist, so this creates all three and then opens the browser itself.
"""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import closing
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import webbrowser

#: Below this the application will not import, and the failure would otherwise
#: surface as a stack trace about syntax rather than a version.
MINIMUM_PYTHON = (3, 11)


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _fail(message: str) -> None:
    """Stop with something a non-technical reader can act on."""
    print(f"\n{message}\n")
    if not sys.stdin.isatty():  # double-clicked: the window would vanish
        input("Press Return to close this window. ")
    raise SystemExit(1)


def _check_python() -> None:
    if sys.version_info < MINIMUM_PYTHON:
        current = ".".join(str(part) for part in sys.version_info[:3])
        _fail(
            f"This app needs Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer, "
            f"but it is running on Python {current}.\n"
            "Install a current version from https://www.python.org/downloads/ "
            "and start the app again."
        )


def _free_port(preferred: int) -> int:
    """Return `preferred`, or the next free port if something already has it.

    A port collision is the most likely failure on a second launch, and
    "address already in use" is not a message anyone should have to act on.
    """

    for candidate in range(preferred, preferred + 20):
        with closing(socket.socket()) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    return preferred


def _open_when_ready(port: int, timeout: float = 90.0) -> None:
    """Open the browser once the server actually answers, not before."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with closing(socket.socket()) as probe:
            probe.settimeout(0.4)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                webbrowser.open(f"http://127.0.0.1:{port}")
                return
        time.sleep(0.4)


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _ensure_environment(include_dev: bool) -> Path:
    python = _venv_python()
    if not python.exists():
        print("Creating the local Python environment…")
        _run([sys.executable, "-m", "venv", str(VENV)])
    imports = "import run_analysis, pytest" if include_dev else "import run_analysis"
    probe = subprocess.run(
        [str(python), "-c", imports],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode:
        print("Installing the application and its dependencies…")
        target = ".[dev]" if include_dev else "."
        _run([str(python), "-m", "pip", "install", "-e", target])
    return python


def main() -> None:
    parser = ArgumentParser(description="Set up and start the local Running Coach")
    parser.add_argument("--setup-only", action="store_true", help="Prepare the app without starting it")
    parser.add_argument("--dev", action="store_true", help="Install test/development dependencies")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    args = parser.parse_args()

    _check_python()
    python = _ensure_environment(args.dev)
    _run([str(python), "-m", "run_analysis", "--project-root", str(ROOT), "init"])
    if args.setup_only:
        print("Setup complete. Run `python3 start.py` when you want to launch the app.")
        return
    port = _free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} was busy, so the app is starting on {port} instead.")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(port,), daemon=True).start()
    print(f"\nRunning Coach is starting at http://127.0.0.1:{port}")
    print("A browser window will open by itself. Close this window to stop the app.\n")
    try:
        _run(
            [
                str(python),
                "-m",
                "run_analysis",
                "--project-root",
                str(ROOT),
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
        )
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as error:
        _fail(f"The app stopped unexpectedly (exit code {error.returncode}).")


if __name__ == "__main__":
    main()
