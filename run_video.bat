@echo off
setlocal
cd /d "%~dp0"
python scripts\run_demo.py --source video %*
exit /b %ERRORLEVEL%
