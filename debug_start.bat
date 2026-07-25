@echo off
cd /d "%~dp0"
if exist "D:\software\miniconda\python.exe" (
    "D:\software\miniconda\python.exe" -m videodub
) else (
    py -3 -m videodub
)
if errorlevel 1 pause
