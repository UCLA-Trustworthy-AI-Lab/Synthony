#!/bin/bash

################################################################################
# Synthony API Server Stop Script
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping Synthony API Server...${NC}"
echo ""

# Check if PID file exists
if [ -f "logs/api_server.pid" ]; then
    PID=$(cat logs/api_server.pid)

    if ps -p $PID > /dev/null 2>&1; then
        echo "  Found server process (PID: $PID)"
        kill $PID

        # Wait for process to stop
        sleep 2

        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${YELLOW}  Process still running, forcing stop...${NC}"
            kill -9 $PID
        fi

        echo -e "${GREEN}✓ API server stopped${NC}"
        rm logs/api_server.pid
    else
        echo -e "${YELLOW}⚠  Process not running (PID: $PID)${NC}"
        rm logs/api_server.pid
    fi
else
    # Try to find and kill by port
    PORT=${API_PORT:-8000}
    echo "  Looking for process on port $PORT..."

    if command -v lsof &> /dev/null; then
        PID=$(lsof -ti:$PORT)

        if [ -n "$PID" ]; then
            echo "  Found process on port $PORT (PID: $PID)"
            kill $PID
            
            # Wait for process to stop
            sleep 2
            
            # Check if still running and force kill if needed
            if ps -p $PID > /dev/null 2>&1; then
                echo -e "${YELLOW}  Process still running, forcing stop...${NC}"
                kill -9 $PID
                sleep 1
            fi
            
            echo -e "${GREEN}✓ API server stopped${NC}"
        else
            echo -e "${YELLOW}⚠  No API server running on port $PORT${NC}"
        fi
    else
        echo -e "${YELLOW}⚠  No PID file found and lsof not available${NC}"
        echo "  Manually kill the server process if it's still running"
    fi
fi

echo ""
