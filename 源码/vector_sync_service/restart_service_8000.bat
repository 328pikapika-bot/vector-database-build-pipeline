@echo off
setlocal

cd /d "%~dp0"

set "PORT=8000"
set "START_SCRIPT=%~dp0start_service_8000.bat"
set "PID="

if not exist "%START_SCRIPT%" (
    echo [ERROR] Start script not found: %START_SCRIPT%
    exit /b 1
)

echo Checking for an existing service on http://127.0.0.1:%PORT% ...

for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$listenPid = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess; if ($listenPid) { Write-Output $listenPid }"`) do (
    set "PID=%%P"
)

if defined PID (
    echo Stopping existing process on port %PORT% ^(PID %PID%^)^...
    taskkill /PID %PID% /F >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Failed to stop PID %PID%.
        exit /b 1
    )

    for /L %%I in (1,1,15) do (
        powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
        if not errorlevel 1 goto :port_ready
        timeout /t 1 /nobreak >nul
    )

    echo [ERROR] Port %PORT% is still in use after waiting 15 seconds.
    exit /b 1
) else (
    echo No running service was found on port %PORT%.
)

:port_ready
echo Starting the service with the project virtual environment...
if "%RUN_LIVE_KAFKA10_AFTER_RESTART%"=="1" (
    start "vector-sync-service-8000" cmd /c "set KAFKA_CONSUMER_ENABLED=false&& call ""%START_SCRIPT%"""
    echo Waiting for http://127.0.0.1:%PORT% ...
    for /L %%I in (1,1,30) do (
        powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%PORT%/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
        if not errorlevel 1 goto :service_ready
        timeout /t 1 /nobreak >nul
    )
    echo [ERROR] Service did not become healthy within 30 seconds.
    exit /b 1

    :service_ready
    echo Running controlled live Kafka smoke after restart...
    "%~dp0.venv\Scripts\python.exe" "%~dp0scripts\live_kafka_10_to_231.py" --limit 10 --batch-size 5 --commit-offsets
) else (
    call "%START_SCRIPT%"
)
