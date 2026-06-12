@echo off
REM Quick Start Script for AI-Accelerator (Windows)
REM This script starts both the backend and frontend servers

echo.
echo ========================================
echo AI-Accelerator Quick Start (Windows)
echo ========================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Please install Node.js 16+
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)

echo Checking requirements...
echo Python version:
python --version

echo Node.js version:
node --version

echo.
echo Starting AI-Accelerator...
echo.

REM Start backend in a new window
echo [1/2] Starting Backend Server (FastAPI)...
start "Backend - AI-Accelerator" cmd /k "cd /d %SCRIPT_DIR%backend_api && python -m uvicorn main:app --reload --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak

REM Start frontend in a new window
echo [2/2] Starting Frontend Server (React + Vite)...
start "Frontend - AI-Accelerator" cmd /k "cd /d %SCRIPT_DIR%frontend && npm run dev"

echo.
echo ========================================
echo Both servers are starting!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Keep both terminal windows open.
echo To stop: Close the terminal windows or press Ctrl+C
echo.
echo Opening browser in 5 seconds...
timeout /t 5 /nobreak

REM Try to open the frontend in the default browser
start "" "http://localhost:5173"

echo.
echo Setup complete! The application should open in your browser.
echo.
