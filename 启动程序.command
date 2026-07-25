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

echo "没有找到 Python 3。请先安装带 Tk 支持的 Python 3。"
read -r _
exit 1
