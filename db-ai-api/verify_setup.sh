#!/bin/bash

# Verification script for DB AI API setup

echo "======================================"
echo "  DB AI API - Setup Verification"
echo "======================================"
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

# Function to check something
check() {
    local name="$1"
    local command="$2"

    echo -n "Checking $name... "

    if eval "$command" &>/dev/null; then
        echo "✓ PASS"
        ((CHECKS_PASSED++))
        return 0
    else
        echo "✗ FAIL"
        ((CHECKS_FAILED++))
        return 1
    fi
}

# Check Python
check "Python 3.11+" "python3 --version | grep -E 'Python 3\.(11|12)'"

# Check pip
check "pip" "pip3 --version"

# Check Ollama
check "Ollama installed" "command -v ollama"

# Check Ollama running
check "Ollama service" "curl -s http://localhost:11434/api/tags"

# Check SQLCoder model
check "SQLCoder model" "ollama list | grep sqlcoder"

# Check ODBC driver
check "ODBC Driver 18" "odbcinst -q -d | grep 'ODBC Driver 18'"

# Check project files
echo ""
echo "Checking project files..."

FILES=(
    "api.py"
    "config.py"
    "schema_extractor.py"
    "table_selector.py"
    "sql_agent.py"
    "main.py"
    "requirements.txt"
    "Dockerfile"
    "docker-compose.yml"
    ".env.example"
    "README.md"
    "QUICK_START.md"
)

for file in "${FILES[@]}"; do
    check "  $file" "test -f $file"
done

# Check if virtual environment exists
echo ""
check "Virtual environment" "test -d venv"

# Check if .env exists
check ".env file" "test -f .env"

# Summary
echo ""
echo "======================================"
echo "  Verification Summary"
echo "======================================"
echo "Passed: $CHECKS_PASSED"
echo "Failed: $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✅ All checks passed! You're ready to go!"
    echo ""
    echo "Next steps:"
    echo "  1. Edit .env with your database credentials"
    echo "  2. source venv/bin/activate"
    echo "  3. python main.py"
    exit 0
else
    echo "⚠ Some checks failed. Please review the output above."
    echo ""
    echo "Common fixes:"
    echo "  - Missing Ollama: brew install ollama"
    echo "  - Missing model: ollama pull sqlcoder:7b"
    echo "  - Missing venv: ./setup.sh"
    echo "  - Missing .env: cp .env.example .env"
    exit 1
fi
