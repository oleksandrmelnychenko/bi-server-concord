#!/bin/bash

# DB AI API Setup Script

set -e

echo "======================================"
echo "  DB AI API - Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Ollama is installed
echo ""
echo "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "✓ Ollama is running"
    else
        echo "⚠ Ollama is not running. Starting Ollama..."
        brew services start ollama
        sleep 3
    fi

    # Check if sqlcoder model is available
    echo ""
    echo "Checking for SQLCoder model..."
    if ollama list | grep -q "sqlcoder"; then
        echo "✓ SQLCoder model is available"
    else
        echo "⚠ SQLCoder model not found. Pulling model..."
        echo "This may take a few minutes (4GB download)..."
        ollama pull sqlcoder:7b
    fi
else
    echo "✗ Ollama is not installed"
    echo ""
    echo "Please install Ollama:"
    echo "  brew install ollama"
    echo "  ollama pull sqlcoder:7b"
    exit 1
fi

# Create virtual environment
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create .env file if it doesn't exist
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠ IMPORTANT: Please edit .env file with your database credentials:"
    echo "  - DB_HOST"
    echo "  - DB_NAME"
    echo "  - DB_USER"
    echo "  - DB_PASSWORD"
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs
mkdir -p vector_db
echo "✓ Directories created"

# Check ODBC driver
echo ""
echo "Checking ODBC Driver for SQL Server..."
if odbcinst -q -d | grep -q "ODBC Driver 18 for SQL Server"; then
    echo "✓ ODBC Driver 18 for SQL Server is installed"
else
    echo "⚠ ODBC Driver 18 for SQL Server not found"
    echo ""
    echo "Please install Microsoft ODBC Driver:"
    echo "  https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server"
fi

echo ""
echo "======================================"
echo "  Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your database credentials"
echo "  2. Activate virtual environment: source venv/bin/activate"
echo "  3. Start the API: python main.py"
echo "  4. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "To test the API:"
echo "  python test_api.py"
echo ""
