@echo off
title Strata IQ3_S
cd /d "D:\temp\Strata"
"D:\temp\Strata\.venv\Scripts\python.exe" "D:\temp\Strata\serve\server.py" "--engine" "strata" "--config" "D:\temp\Strata\strata-iq3_s.json" "--port" "8086" "--open"
if errorlevel 1 pause
