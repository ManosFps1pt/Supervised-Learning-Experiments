@echo off
setlocal

pushd "%~dp0.." >nul
set "REPO_ROOT=%CD%"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8888"

set "JUPYTER_LAB=%REPO_ROOT%\.venv\Scripts\jupyter-lab.exe"

if not exist "%JUPYTER_LAB%" (
  echo Could not find JupyterLab at:
  echo   %JUPYTER_LAB%
  echo.
  echo Install it into the repo venv first.
  popd >nul
  exit /b 1
)

echo Starting JupyterLab from:
echo   %REPO_ROOT%
echo.
echo If port %PORT% is already busy, run:
echo   scripts\start_jupyterlab.bat 8889
echo.

"%JUPYTER_LAB%" --ServerApp.root_dir="%REPO_ROOT%" --port="%PORT%"

popd >nul
