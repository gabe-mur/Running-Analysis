#!/bin/bash
# Double-clickable launcher for macOS. Finder runs this from the user's home
# directory, not from where it lives, so the first job is to find the app.
cd "$(dirname "$0")" || exit 1

# Finder gives a .command no useful PATH, and python3 on a stock Mac is a stub
# that opens the Xcode installer rather than running anything. Look in the
# usual install locations before giving up.
find_python() {
  for candidate in \
    "$(command -v python3.13)" "$(command -v python3.12)" "$(command -v python3.11)" \
    /opt/homebrew/bin/python3 /usr/local/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
    "$(command -v python3)"
  do
    [ -x "$candidate" ] || continue
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON="$(find_python)"
if [ -z "$PYTHON" ]; then
  cat <<'MESSAGE'

Running Coach needs Python 3.11 or newer, and none was found on this Mac.

Install it from https://www.python.org/downloads/ -- the big yellow
"Download Python" button is the right one -- then double-click this file again.

MESSAGE
  read -r -p "Press Return to close this window. " _
  exit 1
fi

exec "$PYTHON" start.py "$@"
