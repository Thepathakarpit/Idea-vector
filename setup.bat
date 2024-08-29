@echo off
setlocal

REM Check if Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed. Please install Python from https://www.python.org/downloads/ and rerun this script.
    exit /b 1
)

REM Check if pip is installed
where pip >nul 2>nul
if errorlevel 1 (
    echo Pip is not installed. Installing pip...
    python -m ensurepip --upgrade
)

REM Install Python dependencies
pip install google-generativeai numpy scikit-learn matplotlib

echo All dependencies installed successfully!

endlocal
