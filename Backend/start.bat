@echo off
REM Quick Start Script for AnswerAI Evaluator Backend (Windows)

echo ========================================
echo   AnswerAI Evaluator - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
    echo.
) else (
    echo [1/5] Virtual environment already exists
    echo.
)

REM Activate virtual environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

REM Install dependencies
echo [3/5] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

REM Download NLP models
echo [4/5] Downloading NLP models...
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
echo.

REM Check if .env exists
if not exist ".env" (
    echo [!] Warning: .env file not found
    echo     Please copy .env.example to .env and configure it
    echo     Then run: python main.py
    pause
    exit /b 0
)

REM Start server
echo [5/5] Starting FastAPI server...
echo.
echo ========================================
echo   Server starting on http://localhost:8000
echo   Press Ctrl+C to stop
echo ========================================
echo.

python main.py
