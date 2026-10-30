@echo off
rem Prepare the Python environment: create .venv and install packages on first run, reuse it afterwards.
rem Called by scripts\start_demo.bat and scripts\verify_test.bat.
rem Keep every .bat file ASCII-only: cmd misreads UTF-8 batch files after a code page change.
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" goto update

echo [1/2] Creating virtual environment .venv ...
where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
if not exist ".venv\Scripts\python.exe" (
    echo Failed: please install Python 3.10+ and tick "Add python.exe to PATH".
    exit /b 1
)
echo [2/2] Installing packages, this takes a few minutes the first time ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
rem NVIDIA GPU found: install the CUDA build of PyTorch first (cu126 needs driver CUDA 12.6+).
where nvidia-smi >nul 2>nul && (
    echo NVIDIA GPU found, installing GPU build of PyTorch ...
    ".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Install failed, check the network and run again.
    rmdir /s /q .venv
    exit /b 1
)
copy /y requirements.txt ".venv\requirements.installed.txt" >nul
goto activate

:update
rem .venv exists: reinstall only when requirements.txt changed since the last install.
fc /b requirements.txt ".venv\requirements.installed.txt" >nul 2>nul && goto activate
echo requirements.txt changed, updating packages ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
copy /y requirements.txt ".venv\requirements.installed.txt" >nul

:activate
call ".venv\Scripts\activate.bat"
exit /b 0
