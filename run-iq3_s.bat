@echo off
title Strata IQ3_S
cd /d "E:\temp\Strata"
"E:\temp\Strata\.venv\Scripts\python.exe" "E:\temp\Strata\serve\server.py" "--engine" "strata" "--config" "E:\temp\Strata\strata-iq3_s.json" "--port" "8086" "--open"
if errorlevel 1 pause
