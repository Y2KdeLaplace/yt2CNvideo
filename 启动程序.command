#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if command -v uv >/dev/null 2>&1; then
    exec uv run python -m videodub
fi

echo "没有找到 uv。请先安装：https://docs.astral.sh/uv/"
read -r _
exit 1
