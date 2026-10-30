@echo off
title Strata deepseek-v4.1-flash-IST-DAS-2.25bpw
cd /d "E:\temp\Strata"
"E:\temp\Strata\.venv\Scripts\python.exe" "E:\temp\Strata\serve\server.py" "--engine" "strata" "--config" "E:\temp\Strata\strata-deepseek-v4.1-flash-ist-das-2.25bpw.json" "--port" "8086" "--open"
if errorlevel 1 pause
