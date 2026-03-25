#!/bin/bash
#
# Sample cURL commands for Synthony API
#
# Usage:
#   ./sample_curl_request.sh                    # Run all examples
#   ./sample_curl_request.sh path/to/data.csv   # Analyze your file
#

BASE_URL="http://localhost:8000"
DEFAULT_FILE="dataset/input_data/Titanic.csv"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Synthony API - cURL Examples${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}1. Health Check${NC}"
echo "curl $BASE_URL/health"
echo ""
curl -s "$BASE_URL/health" | python3 -m json.tool 2>/dev/null || echo "Server not running!"
echo ""

# List models
echo -e "${YELLOW}2. List CPU-only Models${NC}"
echo "curl '$BASE_URL/models?cpu_only=true'"
echo ""
curl -s "$BASE_URL/models?cpu_only=true" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Total: {data['total_models']}, CPU-compatible: {data['filtered_models']}\")
print(f\"Models: {', '.join(list(data['models'].keys())[:5])}...\")
" 2>/dev/null || echo "Failed to list models"
echo ""

# Get model info
echo -e "${YELLOW}3. Get Model Info (ARF)${NC}"
echo "curl $BASE_URL/models/ARF"
echo ""
curl -s "$BASE_URL/models/ARF" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Model: {data['model_name']}\")
print(f\"Type: {data['type']}\")
print(f\"Strengths: {data['strengths'][0]}\")
" 2>/dev/null || echo "Failed to get model info"
echo ""

# Analyze and recommend
FILE="${1:-$DEFAULT_FILE}"
echo -e "${YELLOW}4. Analyze & Recommend${NC}"
echo "File: $FILE"
echo ""
echo "curl -X POST '$BASE_URL/analyze-and-recommend?method=hybrid&cpu_only=true' \\"
echo "  -F 'file=@$FILE'"
echo ""

if [ -f "$FILE" ]; then
    RESULT=$(curl -s -X POST "$BASE_URL/analyze-and-recommend?method=hybrid&cpu_only=true&top_n=3" \
        -F "file=@$FILE")

    echo "$RESULT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    profile = data['analysis']['dataset_profile']
    rec = data['recommendation']['recommended_model']

    print(f\"Dataset: {profile['row_count']} rows × {profile['column_count']} columns\")

    # Stress factors
    stress = [k for k, v in profile['stress_factors'].items() if v]
    if stress:
        print(f\"Stress factors: {', '.join(stress)}\")

    print(f\"\n🏆 Recommended: {rec['model_name']} ({rec['confidence_score']:.0%} confidence)\")
    print(f\"   Type: {rec['model_info']['type']}\")

    alts = data['recommendation']['alternative_models']
    if alts:
        print(f\"   Alternatives: {', '.join([m['model_name'] for m in alts])}\")
except Exception as e:
    print(f'Error parsing response: {e}')
    print(sys.stdin.read())
" 2>/dev/null
else
    echo "File not found: $FILE"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}More examples:${NC}"
echo ""
echo "# Rule-based only (no LLM):"
echo "curl -X POST '$BASE_URL/analyze-and-recommend?method=rule_based' -F 'file=@data.csv'"
echo ""
echo "# With differential privacy requirement:"
echo "curl -X POST '$BASE_URL/analyze-and-recommend?strict_dp=true' -F 'file=@data.csv'"
echo ""
echo "# Full JSON output:"
echo "curl -X POST '$BASE_URL/analyze-and-recommend' -F 'file=@data.csv' | python3 -m json.tool"
echo ""
echo -e "${GREEN}API Docs: $BASE_URL/docs${NC}"
echo -e "${GREEN}========================================${NC}"
