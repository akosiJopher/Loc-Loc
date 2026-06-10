@echo off
REM ============================================================
REM  LOC-LOC v2.2.0 launcher (Windows)
REM  Opens the supervised camera worker in its own window, then
REM  the dashboard.
REM ============================================================
title LOC-LOC Dashboard

echo Checking dependencies...
py -m pip install -r requirements.txt --quiet

echo Starting supervised camera worker in a new window...
start "LOC-LOC Camera Worker" cmd /c run_worker.bat

echo Starting dashboard (open the Network URL on your phone, same Wi-Fi)...
py -m streamlit run app.py --server.address 0.0.0.0
