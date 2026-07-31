@echo off
cd /d "%~dp0"
uv run python -m videodub
if errorlevel 1 pause
