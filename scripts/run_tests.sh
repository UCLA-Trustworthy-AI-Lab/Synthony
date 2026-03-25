#!/bin/bash
# =============================================================================
# Synthony Test Runner
# =============================================================================
# Run tests selectively by category or run all tests.
# 
# Usage:
#   ./scripts/run_tests.sh [options]
#
# Options:
#   -a, --all         Run all tests (default if no option specified)
#   -u, --unit        Run unit tests only
#   -f, --functional  Run functional tests only
#   -r, --regression  Run regression tests only
#   -i, --integration Run integration tests only
#   -v, --verbose     Verbose output
#   -h, --help        Show this help message
#
# Examples:
#   ./scripts/run_tests.sh --unit
#   ./scripts/run_tests.sh -u -f
#   ./scripts/run_tests.sh --all --verbose
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
RUN_ALL=false
VERBOSE=""
TESTS_DIR="./tests"

# Help function
show_help() {
    echo "Synthony Test Runner"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -a, --all         Run all tests (default if no option specified)"
    echo "  -u, --unit        Run unit tests only"
    echo "  -f, --functional  Run functional tests only"
    echo "  -r, --regression  Run regression tests only"
    echo "  -i, --integration Run integration tests only"
    echo "  -v, --verbose     Verbose output"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --unit                    # Run unit tests only"
    echo "  $0 -u -f                     # Run unit and functional tests"
    echo "  $0 --all --verbose           # Run all tests with verbose output"
    echo "  $0                           # Run all tests (default)"
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
if ! $RUN_UNIT && ! $RUN_FUNCTIONAL && ! $RUN_REGRESSION && ! $RUN_INTEGRATION && ! $RUN_ALL; then
    RUN_ALL=true
fi

# If --all is specified, enable all test types
if $RUN_ALL; then
    RUN_UNIT=true
    RUN_FUNCTIONAL=true
    RUN_REGRESSION=true
    RUN_INTEGRATION=true
fi

# Track results
TOTAL_PASSED=0
TOTAL_FAILED=0
FAILED_SUITES=()

# Function to run tests
run_test_suite() {
    local suite_name=$1
    local suite_path=$2
    
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
    
    if python -m pytest "$suite_path" $VERBOSE --tb=short; then
        echo -e "${GREEN}✓ $suite_name tests passed${NC}"
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
    else
        echo -e "${RED}✗ $suite_name tests failed${NC}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        FAILED_SUITES+=("$suite_name")
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

# Run selected test suites
$RUN_UNIT && run_test_suite "Unit" "$TESTS_DIR/unit"
$RUN_FUNCTIONAL && run_test_suite "Functional" "$TESTS_DIR/functional"
$RUN_REGRESSION && run_test_suite "Regression" "$TESTS_DIR/regression"
$RUN_INTEGRATION && run_test_suite "Integration" "$TESTS_DIR/integration"

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
