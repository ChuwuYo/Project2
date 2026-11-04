@echo off
rem Train one model on this computer. Results go to runs_local\<exp>_s<seed>\ (never touches the final models in runs\).
rem Usage: double-click = A1, seed 0.  Or from cmd: scripts\train.bat A2 1
rem Run it again after an interruption: --resume continues from last.pt.
cd /d "%~dp0.."
call "%~dp0setup_env.bat" || goto fail

set EXP=%1
set SEED=%2
if "%EXP%"=="" set EXP=A1
if "%SEED%"=="" set SEED=0

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not found, training on CPU will be very slow')"
echo Training %EXP% seed %SEED%, output: runs_local\%EXP%_s%SEED%
python train.py --exp %EXP% --seed %SEED% --out runs_local --resume
if errorlevel 1 goto fail
echo.
echo Done. Check runs_local\%EXP%_s%SEED%\curves.png and summary.json
pause
exit /b 0

:fail
echo.
echo Training stopped, see the message above.
pause
exit /b 1
