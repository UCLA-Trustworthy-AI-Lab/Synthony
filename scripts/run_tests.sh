#!/bin/bash
# =============================================================================
# Synthony Test Runner
# =============================================================================
# Run tests selectively by category or run all tests. Mirrors
# .github/workflows/cicd.yml's job 'test' -- if you add a suite/ignore here,
# add it there too (or vice versa), they're meant to stay in sync.
#
# Usage:
#   ./scripts/run_tests.sh [options]
#
# Options:
#   -a, --all         Run all tests (default if no option specified) --
#                      unit, functional, regression, integration, mcp.
#                      Does NOT include --slow (see below); that's opt-in.
#   -u, --unit        Run unit tests only
#   -f, --functional  Run functional tests only (excludes live-API-key/
#                      live-server scripts, same ignore list CI uses --
#                      see FUNCTIONAL_IGNORES below)
#   -r, --regression  Run regression tests only
#   -i, --integration Run integration tests only
#   -m, --mcp         Run mcp_server/ tests only (a separate pytest root --
#                      NOT part of tests/, so --all wouldn't reach these
#                      without this option)
#   --slow            Also run the SYNTHONY_RUN_SLOW_TESTS-gated real-
#                      synthcity subprocess tests (real training + real
#                      cancellation, tens of seconds). Off by default,
#                      like CI.
#   -v, --verbose     Verbose output
#   -h, --help        Show this help message
#
# Examples:
#   ./scripts/run_tests.sh --unit
#   ./scripts/run_tests.sh -u -f
#   ./scripts/run_tests.sh --all --verbose
#   ./scripts/run_tests.sh --mcp
#   ./scripts/run_tests.sh --all --slow
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUN_UNIT=false
RUN_FUNCTIONAL=false
RUN_REGRESSION=false
RUN_INTEGRATION=false
RUN_MCP=false
RUN_SLOW=false
RUN_ALL=false
VERBOSE=""
TESTS_DIR="./tests"

# Functional tests excluded from the default run: live-server/API-key
# scripts meant for manual verification, not routine automated runs.
# Keep this in sync with .github/workflows/cicd.yml's functional step.
FUNCTIONAL_IGNORES=(
    "--ignore=tests/functional/test_api_advanced.py"
    "--ignore=tests/functional/test_llm_with_systemprompt.py"
    "--ignore=tests/functional/test_api_simple.py"
    "--ignore=tests/functional/test_openai_connection.py"
    "--ignore=tests/functional/test_recommender_benchmark.py"
)

# Help function
show_help() {
    echo "Synthony Test Runner"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -a, --all         Run all tests (default if no option specified) --"
    echo "                     unit, functional, regression, integration, mcp"
    echo "  -u, --unit        Run unit tests only"
    echo "  -f, --functional  Run functional tests only (excludes live-API-key/"
    echo "                     live-server scripts, same ignore list CI uses)"
    echo "  -r, --regression  Run regression tests only"
    echo "  -i, --integration Run integration tests only"
    echo "  -m, --mcp         Run mcp_server/ tests only (separate pytest root)"
    echo "  --slow            Also run SYNTHONY_RUN_SLOW_TESTS-gated real-synthcity"
    echo "                     tests (real training + cancellation, tens of seconds)"
    echo "  -v, --verbose     Verbose output"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --unit                    # Run unit tests only"
    echo "  $0 -u -f                     # Run unit and functional tests"
    echo "  $0 --all --verbose           # Run all tests with verbose output"
    echo "  $0                           # Run all tests (default)"
    echo "  $0 --mcp                     # Run MCP server tests only"
    echo "  $0 --all --slow              # Everything, incl. real synthcity training"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -u|--unit)
            RUN_UNIT=true
            shift
            ;;
        -f|--functional)
            RUN_FUNCTIONAL=true
            shift
            ;;
        -r|--regression)
            RUN_REGRESSION=true
            shift
            ;;
        -i|--integration)
            RUN_INTEGRATION=true
            shift
            ;;
        -m|--mcp)
            RUN_MCP=true
            shift
            ;;
        --slow)
            RUN_SLOW=true
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# If no specific test type selected, run all
if ! $RUN_UNIT && ! $RUN_FUNCTIONAL && ! $RUN_REGRESSION && ! $RUN_INTEGRATION && ! $RUN_MCP && ! $RUN_ALL; then
    RUN_ALL=true
fi

# If --all is specified, enable all test types (--slow is opt-in even
# under --all, same as CI never passing SYNTHONY_RUN_SLOW_TESTS)
if $RUN_ALL; then
    RUN_UNIT=true
    RUN_FUNCTIONAL=true
    RUN_REGRESSION=true
    RUN_INTEGRATION=true
    RUN_MCP=true
fi

if $RUN_SLOW; then
    export SYNTHONY_RUN_SLOW_TESTS=1
fi

# Track results
TOTAL_PASSED=0
TOTAL_FAILED=0
FAILED_SUITES=()

# Function to run tests
run_test_suite() {
    local suite_name=$1
    local suite_path=$2
    shift 2
    local extra_args=("$@")

    if [ ! -d "$suite_path" ]; then
        echo -e "${YELLOW}⚠ Skipping $suite_name: directory not found${NC}"
        return
    fi

    # Count test files
    local test_files=$(find "$suite_path" -name "test_*.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$test_files" -eq 0 ]; then
        echo -e "${YELLOW}⚠ Skipping $suite_name: no test files found${NC}"
        return
    fi

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Running $suite_name Tests ($test_files files)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if python -m pytest "$suite_path" $VERBOSE --tb=short "${extra_args[@]}"; then
        echo -e "${GREEN}✓ $suite_name tests passed${NC}"
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
    else
        echo -e "${RED}✗ $suite_name tests failed${NC}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        FAILED_SUITES+=("$suite_name")
    fi
}

# MCP server tests live under mcp_server/, a separate pytest root from
# tests/ -- run from within that directory, same as CI does.
run_mcp_tests() {
    if [ ! -d "mcp_server" ]; then
        echo -e "${YELLOW}⚠ Skipping MCP: mcp_server/ directory not found${NC}"
        return
    fi

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Running MCP Server Tests${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if (cd mcp_server && python -m pytest test_server.py test_protocol.py test_agent_tools.py $VERBOSE --tb=short); then
        echo -e "${GREEN}✓ MCP tests passed${NC}"
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
    else
        echo -e "${RED}✗ MCP tests failed${NC}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        FAILED_SUITES+=("MCP")
    fi
}

# Header
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Synthony Test Runner                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"

# Show what will be run
echo ""
echo -e "${YELLOW}Test suites to run:${NC}"
$RUN_UNIT && echo "  • Unit tests"
$RUN_FUNCTIONAL && echo "  • Functional tests"
$RUN_REGRESSION && echo "  • Regression tests"
$RUN_INTEGRATION && echo "  • Integration tests"
$RUN_MCP && echo "  • MCP server tests"
$RUN_SLOW && echo "  • (+ SYNTHONY_RUN_SLOW_TESTS: real synthcity subprocess tests)"

# Run selected test suites
$RUN_UNIT && run_test_suite "Unit" "$TESTS_DIR/unit"
$RUN_FUNCTIONAL && run_test_suite "Functional" "$TESTS_DIR/functional" "${FUNCTIONAL_IGNORES[@]}"
$RUN_REGRESSION && run_test_suite "Regression" "$TESTS_DIR/regression"
$RUN_INTEGRATION && run_test_suite "Integration" "$TESTS_DIR/integration"
$RUN_MCP && run_mcp_tests

# Summary
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All $TOTAL_PASSED test suite(s) passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ $TOTAL_FAILED test suite(s) failed: ${FAILED_SUITES[*]}${NC}"
    echo -e "${GREEN}✓ $TOTAL_PASSED test suite(s) passed${NC}"
    exit 1
fi
