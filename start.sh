#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if command -v python3 >/dev/null 2>&1; then
    exec python3 -m videodub
fi
if command -v python >/dev/null 2>&1; then
    exec python -m videodub
fi

echo "Python 3 was not found. Please install Python 3 with Tk support." >&2
exit 1
