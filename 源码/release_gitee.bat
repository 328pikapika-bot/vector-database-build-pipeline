@echo off
setlocal

title Vector Library Sync Gitee Auto Submit
color 0A

set "REPO_URL=git@gitee.com:cs_shuang/vector_library_sync.git"
set "BRANCH=master"
set "DEFAULT_COMMIT_MSG=Auto submit vector_library_sync"

cd /d "%~dp0"

echo ==========================================
echo Vector Library Sync Gitee Auto Submit
echo ==========================================
echo Project: %cd%
echo Remote : %REPO_URL%
echo Branch : %BRANCH%
echo.

where git > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git was not found. Please install Git for Windows first.
    pause
    exit /b 1
)

if not exist ".git" (
    echo [ERROR] This folder is not a Git repository.
    pause
    exit /b 1
)

git rev-parse --is-inside-work-tree > nul 2>&1
if errorlevel 1 (
    echo [ERROR] This folder is not a valid Git work tree.
    pause
    exit /b 1
)

echo [1/6] Configure remote repository...
git remote get-url origin > nul 2>&1
if errorlevel 1 (
    git remote add origin "%REPO_URL%"
) else (
    git remote set-url origin "%REPO_URL%"
)
if errorlevel 1 (
    echo [ERROR] Failed to configure remote repository.
    pause
    exit /b 1
)

echo [2/6] Check branch...
for /f "tokens=*" %%i in ('git branch --show-current') do set "CURRENT_BRANCH=%%i"
if "%CURRENT_BRANCH%"=="" (
    echo [ERROR] Detached HEAD is not supported. Please checkout a branch first.
    pause
    exit /b 1
)
if not "%CURRENT_BRANCH%"=="%BRANCH%" (
    echo Current branch is "%CURRENT_BRANCH%". Switching to "%BRANCH%"...
    git checkout "%BRANCH%"
    if errorlevel 1 (
        echo [ERROR] Failed to switch to %BRANCH%. Please fix branch state manually.
        pause
        exit /b 1
    )
)

echo [3/6] Read commit message...
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set "STAMP=%%i"
if "%STAMP%"=="" set "STAMP=%date%_%time%"
set "DEFAULT_COMMIT_MSG=%DEFAULT_COMMIT_MSG% %STAMP%"
set "COMMIT_MSG="
set /p "COMMIT_MSG=Commit message, press Enter for default: "
if "%COMMIT_MSG%"=="" set "COMMIT_MSG=%DEFAULT_COMMIT_MSG%"

echo [4/6] Pull remote changes with rebase...
git ls-remote --exit-code --heads origin "%BRANCH%" > nul 2>&1
if errorlevel 1 (
    echo Remote branch "%BRANCH%" does not exist yet. Pull skipped.
) else (
    git pull --rebase --autostash origin "%BRANCH%"
    if errorlevel 1 (
        echo.
        echo [ERROR] Pull failed. There may be conflicts.
        echo Resolve conflicts, then run this script again.
        pause
        exit /b 1
    )
)

echo [5/6] Stage and commit changes...
git ls-files --error-unmatch "vector_sync_service/.env" > nul 2>&1
if not errorlevel 1 (
    echo [WARN] vector_sync_service/.env is tracked. Review secrets before pushing to shared repositories.
)

git add .
if errorlevel 1 (
    echo [ERROR] git add failed.
    pause
    exit /b 1
)

git diff --cached --quiet
if errorlevel 1 (
    git commit -m "%COMMIT_MSG%"
    if errorlevel 1 (
        echo [ERROR] git commit failed.
        pause
        exit /b 1
    )
) else (
    echo No staged changes. Creating an empty timestamp commit...
    git commit --allow-empty -m "%COMMIT_MSG%"
    if errorlevel 1 (
        echo [ERROR] Empty commit failed.
        pause
        exit /b 1
    )
)

echo [6/6] Push to Gitee...
git push -u origin "%BRANCH%"
if errorlevel 1 (
    echo.
    echo [ERROR] Push failed.
    echo Check your Gitee SSH key and repository write permission.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Sync completed.
echo Remote : %REPO_URL%
echo Branch : %BRANCH%
echo Message: %COMMIT_MSG%
echo ==========================================
pause
