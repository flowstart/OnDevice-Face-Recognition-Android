#!/bin/bash
#
# Start the Face Recognition Offloading Server
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Face Recognition Offloading Server${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Using Python: $($PYTHON_CMD --version)${NC}"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt

# Create models directory
mkdir -p models

# Check for model files
if [ ! -f "models/facenet_512.tflite" ]; then
    echo -e "${YELLOW}Copying model files from Android assets...${NC}"
    if [ -f "../app/src/main/assets/facenet_512.tflite" ]; then
        cp ../app/src/main/assets/facenet_512.tflite models/
        cp ../app/src/main/assets/blaze_face_short_range.tflite models/
        echo -e "${GREEN}Model files copied successfully${NC}"
    else
        echo -e "${YELLOW}Warning: Model files not found in Android assets${NC}"
        echo -e "${YELLOW}Server will start with limited functionality${NC}"
    fi
fi

# Get local IP address
LOCAL_IP=$(ifconfig 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' || hostname -I 2>/dev/null | awk '{print $1}')

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting server...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Local URL: ${GREEN}http://localhost:8000${NC}"
echo -e "Network URL: ${GREEN}http://${LOCAL_IP}:8000${NC}"
echo ""
echo -e "API Documentation: ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000

