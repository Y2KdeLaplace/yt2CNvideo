#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if command -v uv >/dev/null 2>&1; then
    exec uv run python -m videodub
fi

echo "uv was not found. Install it from https://docs.astral.sh/uv/." >&2
exit 1
