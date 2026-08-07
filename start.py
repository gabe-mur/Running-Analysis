"""Cross-platform one-command setup and launcher for the local application."""

from __future__ import annotations

from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    python = _ensure_environment(args.dev)
    _run([str(python), "-m", "run_analysis", "--project-root", str(ROOT), "init"])
    if args.setup_only:
        print("Setup complete. Run `python3 start.py` when you want to launch the app.")
        return
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
                args.host,
                "--port",
                str(args.port),
            ]
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
