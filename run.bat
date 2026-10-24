@echo off
rem Single Strata servers, 3x Titan V (ports 8086), sharing one copy of the experts in RAM.
rem   run.bat              sets the model up once if needed, tunes once if this PC has no tuning yet, starts the three
rem   run.bat --calibrate  tunes again, then starts
rem Nothing asks a question: setup and tuning run once in this window, before the three servers start.
setlocal
cd /d "%~dp0"
SET STRATA_EXPERIMENTAL_SM60=1
SET STRATA_PIN_BY_NEED=1
SET STRATA_PCIE_MIN1=1
SET STRATA_ARENA_PIN_GIB=0
SET STRATA_CONVERSATION_CACHE_MIB=8192
SET STRATA_CONVERSATION_CACHE_SLOTS=4
SET STRATA_CONVERSATION_CACHE_MIN_FREE_MIB=2560
set "SETUP=--cuda 12 --gpus all --family qwen --model IQ3_S --context 262144 --kv-streaming on --kv int8 --vision cpu --experimental-speed-projection off --layer-split auto --parallel 1"
set "SERVE=--host 0.0.0.0 --port 8086 --api-key STRATA_ABIGAIL"
set "PY=.venv\Scripts\python.exe"
set "TUNE="
if /i "%~1"=="--calibrate" set "TUNE=1"

if not exist strata-iq3_s.json call START-HERE.bat %SETUP% %SERVE% --no-start --yes
if not exist strata-iq3_s.json exit /b 1

if not defined TUNE %PY% -c "import json, setup, sys; sys.exit(0 if setup.saved_calibration(json.load(open('strata-iq3_s.json', encoding='utf-8-sig'))) else 1)" >nul 2>nul || set "TUNE=1"
if defined TUNE call START-HERE.bat --calibrate --no-start --yes %SETUP% %SERVE%

cmd.exe /k %PY% serve\server.py --engine strata --config strata-iq3_s.json %SERVE%
