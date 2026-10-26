@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)
echo.
echo DataPrep Lab - open http://127.0.0.1:5080 in your browser.
echo Keep this window open. Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Could not start DataPrep Lab. See the error above.
pause
exit /b 1
