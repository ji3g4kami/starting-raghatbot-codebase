#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔍 Running Code Quality Checks..."
echo "=================================="

# Track if any check fails
FAILED=0

# Run Black formatter check
echo -e "\n📝 Checking code formatting with Black..."
if uv run black --check .; then
    echo -e "${GREEN}✓ Black formatting check passed${NC}"
else
    echo -e "${RED}✗ Black formatting issues found${NC}"
    echo -e "${YELLOW}  Run 'uv run black .' to fix formatting${NC}"
    FAILED=1
fi

# Run Ruff linter
echo -e "\n🔧 Running Ruff linter..."
if uv run ruff check .; then
    echo -e "${GREEN}✓ Ruff linting passed${NC}"
else
    echo -e "${RED}✗ Ruff found linting issues${NC}"
    echo -e "${YELLOW}  Run 'uv run ruff check --fix .' to auto-fix issues${NC}"
    FAILED=1
fi

# Run MyPy type checker
echo -e "\n🔬 Running MyPy type checker..."
if uv run mypy .; then
    echo -e "${GREEN}✓ MyPy type checking passed${NC}"
else
    echo -e "${RED}✗ MyPy found type issues${NC}"
    FAILED=1
fi

# Run tests
echo -e "\n🧪 Running tests..."
if uv run pytest tests/ -q; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    FAILED=1
fi

# Summary
echo -e "\n=================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All quality checks passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some quality checks failed. Please fix the issues above.${NC}"
    exit 1
fi