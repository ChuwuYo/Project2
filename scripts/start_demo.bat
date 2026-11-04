@echo off
rem Start the facial expression demo (Gradio web page) and open the browser. Close this window to stop.
rem 1 = model average (runs/final_runs.txt), 2 = best single model (runs/best_single.txt).
cd /d "%~dp0.."
call "%~dp0setup_env.bat" || goto fail
echo.
echo Which model?
echo   1 = average of all final models (runs/final_runs.txt)
echo   2 = best single model (runs/best_single.txt)
set "CHOICE=1"
set /p "CHOICE=Enter 1 or 2 [default 1]: "
set "MODEL_ARG="
if "%CHOICE%"=="2" set "MODEL_ARG=--single"

echo Starting demo page ...
python app.py --inbrowser %MODEL_ARG%
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo Start failed, see the error above.
pause
exit /b 1
