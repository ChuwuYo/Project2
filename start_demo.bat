@echo off
rem Start the facial expression demo (Gradio web page) and open the browser. Close this window to stop.
cd /d "%~dp0"
call scripts\setup_env.bat || goto fail

echo Starting demo page ...
python app.py --inbrowser
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo Start failed, see the error above.
pause
exit /b 1
