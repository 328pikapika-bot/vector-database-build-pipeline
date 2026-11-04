@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python virtual environment not found: %PYTHON_EXE%
    echo [ERROR] Create or restore the virtual environment before starting the service.
    exit /b 1
)

echo Starting NSTL Vector Sync Service on http://127.0.0.1:8000
echo Using Python: %PYTHON_EXE%

"%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
