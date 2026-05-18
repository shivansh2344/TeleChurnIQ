#!/bin/bash
# Quick Setup Script for TeleChurnIQ
# Run this script to automatically set up the entire project

set -e  # Exit on error

echo "🚀 TeleChurnIQ Setup Script"
echo "=============================="

# Check Python version
echo "✓ Checking Python version..."
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python version: $PYTHON_VERSION"

# Check Node.js version
echo "✓ Checking Node.js version..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

NODE_VERSION=$(node --version)
echo "  Node.js version: $NODE_VERSION"

# Create Python virtual environment
echo ""
echo "✓ Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python -m venv .venv
    echo "  Virtual environment created"
else
    echo "  Virtual environment already exists"
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
echo "  Virtual environment activated"

# Install Python dependencies
echo ""
echo "✓ Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "  Python dependencies installed"

# Setup Backend
echo ""
echo "✓ Setting up Backend..."
cd backend
npm install
echo "  Backend dependencies installed"
cd ..

# Setup Frontend
echo ""
echo "✓ Setting up Frontend..."
cd frontend
npm install
echo "  Frontend dependencies installed"
cd ..

# Create .env file if it doesn't exist
echo ""
echo "✓ Checking configuration..."
if [ ! -f ".env" ]; then
    echo "  Creating .env file from .env.example..."
    cp .env.example .env
    echo "  ⚠️  Please update .env with your actual values"
else
    echo "  .env file already exists"
fi

echo ""
echo "✅ Setup Complete!"
echo "=============================="
echo ""
echo "Next steps:"
echo "1. Update .env file with your configuration"
echo "2. Run 'npm run dev' in frontend/ directory"
echo "3. Run 'npm start' in backend/ directory"
echo "4. Open http://localhost:5173 in your browser"
echo ""
echo "For detailed instructions, see README.md"
