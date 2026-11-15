#!/usr/bin/env bash
# One-shot start: installs deps, runs preflight checks, kicks off the endless loop.
#
# Prereqs for a fresh machine:
#   1. Clone this repo
#   2. Grant Terminal: System Settings -> Privacy & Security ->
#        * Accessibility
#        * Screen Recording
#   3. Launch the Fahren Lernen app and leave it on the home screen
#   4. ./start.sh

set -u

cd "$(dirname "$0")"

echo "=== Fahren Lernen test harness ==="
echo

# --- uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is not installed."
    if command -v brew >/dev/null 2>&1; then
        echo "  Install with:  brew install uv"
    else
        echo "  Install from: https://docs.astral.sh/uv/getting-started/installation/"
    fi
    exit 1
fi
echo "uv: $(uv --version)"

# --- .env -------------------------------------------------------------------
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo
        echo "ERROR: Created .env from .env.example. Edit it and set GEMINI_API_KEY, then rerun ./start.sh"
        exit 1
    fi
    echo "ERROR: .env is missing and no .env.example to copy from."
    exit 1
fi

# --- deps -------------------------------------------------------------------
echo
echo "Syncing dependencies..."
if ! uv sync; then
    echo "ERROR: 'uv sync' failed. See output above."
    exit 1
fi

# --- preflight --------------------------------------------------------------
echo
if ! uv run python scripts/preflight.py; then
    exit 1
fi

# --- run --------------------------------------------------------------------
echo
echo "Starting endless unattended run. Press Ctrl+C to stop."
echo "Session artifacts and logs: runs/session-<timestamp>/"
echo "Live log also mirrored to: runs/endless-run.log"
echo
mkdir -p runs
exec uv run python scripts/run_loop.py 2>&1 | tee runs/endless-run.log
