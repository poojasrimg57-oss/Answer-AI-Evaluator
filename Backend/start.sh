#!/bin/bash
# Quick Start Script for AnswerAI Evaluator Backend (Linux/Mac)

echo "========================================"
echo "  AnswerAI Evaluator - Quick Start"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created successfully!"
    echo ""
else
    echo "[1/5] Virtual environment already exists"
    echo ""
fi

# Activate virtual environment
echo "[2/5] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi
echo ""

# Install dependencies
echo "[3/5] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi
echo "Dependencies installed successfully!"
echo ""

# Download NLP models
echo "[4/5] Downloading NLP models..."
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "[!] Warning: .env file not found"
    echo "    Please copy .env.example to .env and configure it"
    echo "    Then run: python main.py"
    exit 0
fi

# Start server
echo "[5/5] Starting FastAPI server..."
echo ""
echo "========================================"
echo "  Server starting on http://localhost:8000"
echo "  Press Ctrl+C to stop"
echo "========================================"
echo ""

python main.py
