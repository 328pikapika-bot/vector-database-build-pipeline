@echo off
setlocal

cd /d "%~dp0"

set "SERVICE_DIR=%~dp0vector_sync_service"
set "TARGET_SCRIPT=%SERVICE_DIR%\restart_service_8000.bat"

if not exist "%TARGET_SCRIPT%" (
    echo [ERROR] Target script not found: %TARGET_SCRIPT%
    exit /b 1
)

call "%TARGET_SCRIPT%"
