#!/bin/bash

# Quick Start Script for AI-Accelerator (macOS/Linux)
# This script starts both the backend and frontend servers

echo ""
echo "========================================"
echo "AI-Accelerator Quick Start (macOS/Linux)"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8+"
    echo "Install with: brew install python3"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found. Please install Node.js 16+"
    echo "Install with: brew install node"
    exit 1
fi

echo "Checking requirements..."
echo "Python version:"
python3 --version

echo "Node.js version:"
node --version

echo ""
echo "Starting AI-Accelerator..."
echo ""

# Start backend in background
echo "[1/2] Starting Backend Server (FastAPI)..."
cd "$SCRIPT_DIR/backend_api"
python3 -m uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 3

# Start frontend in a new terminal or background
echo "[2/2] Starting Frontend Server (React + Vite)..."
cd "$SCRIPT_DIR/frontend"

# Try to use new terminal windows if available
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - open in new Terminal windows
    osascript -e 'tell app "Terminal" to do script "cd '"$SCRIPT_DIR/frontend"' && npm run dev"'
else
    # Linux - try to open in new terminal or background
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR/frontend' && npm run dev; bash"
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$SCRIPT_DIR/frontend' && npm run dev" &
    else
        npm run dev &
    fi
fi

echo ""
echo "========================================"
echo "Both servers are starting!"
echo "========================================"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "To stop:"
echo "  Backend:  kill $BACKEND_PID"
echo "  Frontend: kill %% (in frontend terminal)"
echo ""
echo "Opening browser in 5 seconds..."
sleep 5

# Try to open the frontend in the default browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:5173"
elif command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:5173"
else
    echo "Please open: http://localhost:5173"
fi

echo ""
echo "Setup complete! The application should open in your browser."
echo ""

# Wait for backend process
wait $BACKEND_PID
