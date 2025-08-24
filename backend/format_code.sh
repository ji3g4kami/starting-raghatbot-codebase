#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🎨 Formatting Code..."
echo "====================="

# Format with Black
echo -e "\n📝 Running Black formatter..."
uv run black .
echo -e "${GREEN}✓ Code formatted with Black${NC}"

# Sort imports with Ruff
echo -e "\n📦 Sorting imports with Ruff..."
uv run ruff check --select I --fix .
echo -e "${GREEN}✓ Imports sorted${NC}"

# Auto-fix other Ruff issues
echo -e "\n🔧 Auto-fixing linting issues with Ruff..."
uv run ruff check --fix .
echo -e "${GREEN}✓ Auto-fixable issues resolved${NC}"

echo -e "\n====================="
echo -e "${GREEN}✅ Code formatting complete!${NC}"
echo -e "${YELLOW}Note: Run './check_quality.sh' to verify all checks pass${NC}"