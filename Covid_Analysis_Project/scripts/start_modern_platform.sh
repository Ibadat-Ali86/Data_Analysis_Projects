#!/bin/bash

# COVID-19 Modern Platform Startup Script
# Author: IBADAT ALI
# Date: November 2025

set -e

echo "================================================"
echo "🌍 COVID-19 Analysis Platform - Startup Script"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Docker and Docker Compose are installed"

# Check if .env file exists for frontend
if [ ! -f "frontend/.env" ]; then
    print_warning ".env file not found for frontend. Creating from example..."
    cp frontend/.env.example frontend/.env 2>/dev/null || echo "VITE_API_URL=http://localhost:8000" > frontend/.env
    print_success "Created frontend/.env file"
fi

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose down 2>/dev/null || true

# Build and start containers
print_status "Building and starting containers..."
docker-compose up -d --build

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 10

# Check if MySQL is ready
print_status "Checking MySQL connection..."
for i in {1..30}; do
    if docker-compose exec -T mysql mysqladmin ping -h localhost -u root -proot &> /dev/null; then
        print_success "MySQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "MySQL failed to start"
        exit 1
    fi
    sleep 2
done

# Check if backend is ready
print_status "Checking backend API..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health &> /dev/null; then
        print_success "Backend API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Backend API failed to start"
        exit 1
    fi
    sleep 2
done

# Check if frontend is ready
print_status "Checking frontend..."
for i in {1..30}; do
    if curl -s http://localhost:3000 &> /dev/null; then
        print_success "Frontend is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_warning "Frontend may take longer to start. Please wait..."
    fi
    sleep 2
done

echo ""
echo "================================================"
echo "✅ Platform is running!"
echo "================================================"
echo ""
echo "🌐 Frontend:    http://localhost:3000"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs:    http://localhost:8000/api/docs"
echo "💾 Database:    localhost:3306"
echo ""
echo "To view logs:        docker-compose logs -f"
echo "To stop platform:    docker-compose down"
echo "To restart:          docker-compose restart"
echo ""
echo "================================================"

