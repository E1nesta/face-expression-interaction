@echo off
setlocal
cd /d "%~dp0"
python scripts\bootstrap.py %*
exit /b %ERRORLEVEL%
