@echo off
setlocal
cd /d "%~dp0"
python scripts\run_demo.py %*
exit /b %ERRORLEVEL%
