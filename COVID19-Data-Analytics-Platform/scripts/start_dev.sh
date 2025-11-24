#!/bin/bash

# COVID-19 Platform - Local Development Startup Script
# Author: IBADAT ALI
# Date: November 2025

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================"
echo "🚀 Starting COVID-19 Platform (Development Mode)"
echo "================================================"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check Python
if ! command_exists python3; then
    echo -e "${YELLOW}[ERROR]${NC} Python 3 is not installed"
    exit 1
fi

# Check Node.js
if ! command_exists node; then
    echo -e "${YELLOW}[ERROR]${NC} Node.js is not installed"
    exit 1
fi

# Check MySQL
if ! command_exists mysql; then
    echo -e "${YELLOW}[WARNING]${NC} MySQL client not found. Make sure MySQL server is running."
fi

echo -e "${BLUE}[INFO]${NC} Starting Backend API..."
cd api

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}[INFO]${NC} Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
source venv/bin/activate
pip install -q -r requirements.txt

# Start backend in background
echo -e "${GREEN}[SUCCESS]${NC} Starting FastAPI server on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

cd .. # Return to root directory

# --- Frontend Setup ---
echo -e "${BLUE}[INFO]${NC} Setting up Frontend..."
cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}[INFO]${NC} Installing npm dependencies..."
    npm install
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "VITE_API_URL=http://localhost:8000" > .env
fi

# Start frontend
echo -e "${GREEN}[SUCCESS]${NC} Starting Vite dev server on port 3000..."
npm run dev &
FRONTEND_PID=$!

cd ..

echo ""
echo "================================================"
echo "✅ Development servers started!"
echo "================================================"
echo ""
echo "🌐 Frontend:    http://localhost:3000"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs:    http://localhost:8000/api/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

