@echo off
rem Evaluate the final model(s) listed in final_runs.txt on PrivateTest.
rem Prints accuracy and per-class report, and writes results to reports\.
cd /d "%~dp0.."
call "%~dp0setup_env.bat" || goto fail

python evaluate.py
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
