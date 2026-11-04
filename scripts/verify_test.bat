@echo off
rem Evaluate on PrivateTest: 1 = model average (final_runs.txt), 2 = best single model (best_single.txt).
rem Prints accuracy and per-class report, and writes results to reports\.
cd /d "%~dp0.."
call "%~dp0setup_env.bat" || goto fail
echo.
echo Which model?
echo   1 = average of all final models (final_runs.txt)
echo   2 = best single model (best_single.txt)
set "CHOICE=1"
set /p "CHOICE=Enter 1 or 2 [default 1]: "
set "MODEL_ARG="
if "%CHOICE%"=="2" set "MODEL_ARG=--single"

python evaluate.py %MODEL_ARG%
if errorlevel 1 goto fail
echo.
echo Done. Results are saved in reports\
pause
exit /b 0

:fail
echo.
echo Evaluation failed, see the error above.
pause
exit /b 1
