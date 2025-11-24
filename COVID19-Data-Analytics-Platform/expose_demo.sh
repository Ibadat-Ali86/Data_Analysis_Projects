#!/bin/bash

echo "=================================================="
echo "COVID-19 Platform - Public URL Exposer"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if app is running
if ! curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo -e "${RED}✗ Application is not running!${NC}"
    echo "Please run ./start_live_demo.sh first"
    exit 1
fi

echo -e "${GREEN}✓ Application is running locally${NC}"
echo ""
echo "Choose a method to expose your demo:"
echo "  1) LocalTunnel (No installation needed, free)"
echo "  2) ngrok (Requires installation, more stable)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Starting LocalTunnel...${NC}"
        echo ""
        echo "IMPORTANT: Copy the URL below and share it as your live demo link!"
        echo "==================================================================="
        npx localtunnel --port 3000
        ;;
    2)
        if ! command -v ngrok &> /dev/null; then
            echo -e "${RED}✗ ngrok is not installed${NC}"
            echo ""
            echo "To install ngrok:"
            echo "  1. Download from: https://ngrok.com/download"
            echo "  2. Extract and move to /usr/local/bin/"
            echo "  3. Sign up at ngrok.com and run: ngrok authtoken YOUR_TOKEN"
            exit 1
        fi
        
        echo ""
        echo -e "${BLUE}Starting ngrok...${NC}"
        echo ""
        echo "IMPORTANT: Copy the 'Forwarding' URL below (the https one)!"
        echo "==================================================================="
        ngrok http 3000
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
