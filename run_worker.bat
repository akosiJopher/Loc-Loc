@echo off
REM ============================================================
REM  LOC-LOC camera worker (supervised)
REM  Restarts the worker automatically if it crashes, so an
REM  overnight Wi-Fi drop or transient error can't take the
REM  count offline until someone notices.
REM ============================================================
title LOC-LOC Camera Worker
cd /d "%~dp0"
:loop
py camera_worker.py
if %errorlevel%==2 (
    echo Worker refused to start: missing camera credentials.
    echo Fix .streamlit\secrets.toml, then close this window and rerun.
    pause
    exit /b 2
)
echo Worker exited with code %errorlevel%. Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
