#!/bin/bash

echo "=================================================="
echo "COVID-19 Analysis Platform - Live Demo Starter"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if MySQL is running
echo -n "Checking MySQL service... "
if systemctl is-active --quiet mysql || systemctl is-active --quiet mariadb; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
    echo "Starting MySQL..."
    sudo systemctl start mysql || sudo systemctl start mariadb
    sleep 2
fi

# Check if database exists
echo -n "Checking covid_db database... "
if mysql -u ibadat -pibadat -e "USE covid_db;" 2>/dev/null; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${YELLOW}⚠ Not found${NC}"
    echo "Creating database and loading data..."
    python3 data_engineering/db_creation_script.py
    python3 data_engineering/load_summary_csv.py
fi

# Check if data is loaded
echo -n "Checking data in final_covid table... "
RECORD_COUNT=$(mysql -u ibadat -pibadat -se "SELECT COUNT(*) FROM covid_db.final_covid;" 2>/dev/null)
if [ "$RECORD_COUNT" -gt 0 ] 2>/dev/null; then
    echo -e "${GREEN}✓ Found $RECORD_COUNT records${NC}"
else
    echo -e "${YELLOW}⚠ No data found${NC}"
    echo "Loading data..."
    python3 data_engineering/load_summary_csv.py
fi

# Stop any existing containers
echo ""
echo "Stopping any existing containers..."
docker rm -f covid_backend covid_frontend 2>/dev/null || true

# Start Docker containers
echo ""
echo "Starting Docker containers..."
docker-compose up -d --build

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
sleep 5

# Check if backend is running
echo -n "Checking backend API... "
if curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Running on http://localhost:8000${NC}"
else
    echo -e "${RED}✗ Failed to start${NC}"
    echo "Check logs with: docker logs covid_backend"
    exit 1
fi

# Check if frontend is running
echo -n "Checking frontend... "
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Running on http://localhost:3000${NC}"
else
    echo -e "${RED}✗ Failed to start${NC}"
    echo "Check logs with: docker logs covid_frontend"
    exit 1
fi

echo ""
echo "=================================================="
echo -e "${GREEN}✓ COVID-19 Platform is running!${NC}"
echo "=================================================="
echo ""
echo "📊 Dashboard: http://localhost:3000"
echo "🔌 API Docs:  http://localhost:8000/api/docs"
echo ""
echo "To expose to the internet (for live demo):"
echo "  Option 1: npx localtunnel --port 3000"
echo "  Option 2: ngrok http 3000 (requires ngrok installation)"
echo ""
echo "To stop: docker-compose down"
echo "To view logs: docker-compose logs -f"
echo ""
