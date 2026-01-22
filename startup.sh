#!/bin/bash
# Startup script for MNIST application

set -e

echo "╔════════════════════════════════════════════════╗"
echo "║     MNIST Digit Recognition - Startup         ║"
echo "╚════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Python
echo -e "${BLUE}[1/5]${NC} Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check/Install dependencies
echo -e "${BLUE}[2/5]${NC} Installing dependencies..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

echo "Installing packages..."
pip install -q -r requirements.txt 2>/dev/null || {
    echo -e "${YELLOW}Some packages failed. Trying with --no-cache-dir...${NC}"
    pip install --no-cache-dir -q -r requirements.txt
}
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check model
echo -e "${BLUE}[3/5]${NC} Checking model..."
if [ ! -f "models/mnist_model.h5" ]; then
    echo -e "${YELLOW}Model not found. Training...${NC}"
    python3 train_model.py
    echo -e "${GREEN}✓ Model trained${NC}"
else
    echo -e "${GREEN}✓ Model found${NC}"
fi

# Create directories
echo -e "${BLUE}[4/5]${NC} Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p visualizations
mkdir -p prediction_data
mkdir -p api_logs
echo -e "${GREEN}✓ Directories ready${NC}"

# Start application
echo -e "${BLUE}[5/5]${NC} Starting application..."
echo -e "${GREEN}✓ Starting Flask server...${NC}"
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Application running on http://localhost:5000 ║"
echo "║  Press Ctrl+C to stop                          ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

python3 app.py
