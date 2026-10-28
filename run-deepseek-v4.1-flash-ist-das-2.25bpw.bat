@echo on
rem Single Strata servers, 3x Titan V (ports 8086), sharing one copy of the experts in RAM.
rem   run.bat              sets the model up once if needed, tunes once if this PC has no tuning yet, starts the three
rem Nothing asks a question: setup and tuning run once in this window, before the three servers start.
cd /d "%~dp0"
SET STRATA_EXPERIMENTAL_SM60=1
SET STRATA_PIN_BY_NEED=1
SET STRATA_PCIE_MIN1=1
SET STRATA_ARENA_PIN_GIB=24
SET STRATA_CONVERSATION_CACHE_MIB=8192
SET STRATA_CONVERSATION_CACHE_SLOTS=4
SET STRATA_CONVERSATION_CACHE_MIN_FREE_MIB=2560
SET STRATA_DSV41_EXE=engine-dsv41\strata-dsv41.exe
set "SETUP=--cuda 12 --family deepseek41 --model IST-DAS-2.25bpw --gpus all --context 262144 --kv-streaming on --kv int8 --vision cpu --experimental-speed-projection off --layer-split auto --parallel 1"
set "SERVE=--host 0.0.0.0 --port 8086 --api-key STRATA_ABIGAIL"

setlocal
set "config=strata-deepseek-v4.1-flash-ist-das-2.25bpw.json"
if not exist "%config%" call START-HERE.bat %SETUP% %SERVE% --no-start --yes
if not exist "%config%" (
    echo %config% not found.
    goto :eof
)
".venv\Scripts\python.exe" serve\server.py --engine strata --config "%config%" %SERVE%
