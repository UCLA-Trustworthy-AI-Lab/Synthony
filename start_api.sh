#!/bin/bash

################################################################################
# Synthony API Server Startup Script
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Synthony - Data Analysis & Model Recommendation API${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Check if Python environment is activated
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}⚠️  Warning: No virtual environment detected${NC}"
    echo -e "${YELLOW}   Consider activating a virtual environment first${NC}"
    echo ""
fi

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ Loading environment variables from .env${NC}"
    export $(grep -v '^#' .env | xargs)
else
    echo -e "${YELLOW}ℹ  No .env file found (optional)${NC}"
    echo -e "${YELLOW}   Create .env file to configure VLLM/OpenAI settings${NC}"
    echo ""
fi

# Check if package is installed
if ! python -c "import synthony" 2>/dev/null; then
    echo -e "${RED}✗ Error: synthony package not installed${NC}"
    echo -e "${YELLOW}  Installing package...${NC}"
    pip install -e ".[api,llm]"
    echo -e "${GREEN}✓ Package installed${NC}"
    echo ""
fi

# Display configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Host: ${API_HOST:-0.0.0.0}"
echo "  Port: ${API_PORT:-9090}"
echo "  Reload: ${API_RELOAD:-true}"
echo "  Workers: ${API_WORKERS:-1}"
echo ""

# Display LLM configuration if available
if [ -n "$VLLM_URL" ]; then
    echo -e "${GREEN}✓ VLLM configured:${NC}"
    echo "  URL: $VLLM_URL"
    echo "  Model: ${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
    echo "  API Key: ${VLLM_API_KEY:+***configured***}"
    echo ""
elif [ -n "$OPENAI_API_KEY" ]; then
    echo -e "${GREEN}✓ OpenAI configured:${NC}"
    echo "  URL: ${OPENAI_URL:-https://api.openai.com/v1}"
    echo "  Model: ${OPENAI_MODEL:-gpt-4o}"
    echo "  API Key: ${OPENAI_API_KEY:+***configured***}"
    echo ""
else
    echo -e "${YELLOW}⚠️  No LLM configured (rule-based mode only)${NC}"
    echo "  To enable LLM mode, set VLLM_URL or OPENAI_API_KEY"
    echo ""
fi

# Check SystemPrompt
if [ -n "$SYNTHONY_SYSTEM_PROMPT" ]; then
    if [ -f "$SYNTHONY_SYSTEM_PROMPT" ]; then
        echo -e "${GREEN}✓ Custom SystemPrompt: $SYNTHONY_SYSTEM_PROMPT${NC}"
    else
        echo -e "${RED}✗ Custom SystemPrompt not found: $SYNTHONY_SYSTEM_PROMPT${NC}"
    fi
elif [ -f "docs/SystemPrompt_v3.md" ]; then
    echo -e "${GREEN}✓ Using default SystemPrompt: docs/SystemPrompt_v3.md${NC}"
else
    echo -e "${YELLOW}⚠️  SystemPrompt file not found${NC}"
fi
echo ""

# Parse command line arguments
MODE="dev"
if [ "$1" == "prod" ] || [ "$1" == "production" ]; then
    MODE="prod"
    echo -e "${BLUE}Mode: Production${NC}"
elif [ "$1" == "background" ] || [ "$1" == "bg" ]; then
    MODE="background"
    echo -e "${BLUE}Mode: Background${NC}"
else
    echo -e "${BLUE}Mode: Development (with auto-reload)${NC}"
    echo -e "${YELLOW}  Use: ./start_api.sh prod      for production mode${NC}"
    echo -e "${YELLOW}  Use: ./start_api.sh background for background mode${NC}"
fi
echo ""

# Display URLs
echo -e "${GREEN}API will be available at:${NC}"
echo "  Main API:      http://localhost:${API_PORT:-9090}"
echo "  Swagger Docs:  http://localhost:${API_PORT:-9090}/docs"
echo "  ReDoc:         http://localhost:${API_PORT:-9090}/redoc"
echo ""

# Create logs directory
mkdir -p logs

# Start server based on mode
case $MODE in
    prod)
        echo -e "${GREEN}Starting API server in production mode...${NC}"
        echo ""
        python -m uvicorn synthony.api.server:app \
            --host ${API_HOST:-0.0.0.0} \
            --port ${API_PORT:-9090} \
            --workers ${API_WORKERS:-4} \
            --log-level info
        ;;

    background)
        LOG_FILE="logs/api_server_$(date +%Y%m%d_%H%M%S).log"
        echo -e "${GREEN}Starting API server in background mode...${NC}"
        echo "  Log file: $LOG_FILE"
        echo ""

        python -m uvicorn synthony.api.server:app \
            --host ${API_HOST:-0.0.0.0} \
            --port ${API_PORT:-9090} \
            --reload \
            > "$LOG_FILE" 2>&1 &

        PID=$!
        echo $PID > logs/api_server.pid

        sleep 2

        if ps -p $PID > /dev/null; then
            echo -e "${GREEN}✓ API server started successfully (PID: $PID)${NC}"
            echo ""
            echo "To view logs:"
            echo "  tail -f $LOG_FILE"
            echo ""
            echo "To stop server:"
            echo "  ./stop_api.sh"
            echo "  or: kill $PID"
        else
            echo -e "${RED}✗ Failed to start API server${NC}"
            echo "Check logs: $LOG_FILE"
            exit 1
        fi
        ;;

    *)
        echo -e "${GREEN}Starting API server in development mode...${NC}"
        echo -e "${YELLOW}Press CTRL+C to stop${NC}"
        echo ""
        sleep 1

        python -m uvicorn synthony.api.server:app \
            --host ${API_HOST:-0.0.0.0} \
            --port ${API_PORT:-9090} \
            --reload \
            --log-level info
        ;;
esac
