"""
AnswerAI Evaluator - FastAPI Application Entry Point
AI-Powered Answer Sheet Evaluation System with OCR, NLP, and Semantic Analysis
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import yaml

from api_routes import router as api_router

# Load environment variables
load_dotenv()

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="AnswerAI Evaluator API",
    description="AI-Powered Answer Sheet Evaluation with OCR, SBERT, and NLI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8080")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:8080", "http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
Path("logs").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Configure logging
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level=os.getenv("LOG_LEVEL", "INFO")
)

# Include API routes
app.include_router(api_router, prefix="/api", tags=["evaluation"])

# Serve static files (optional frontend)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Model warmup on startup
@app.on_event("startup")
async def warmup_models():
    """
    Preload models on application startup for faster first request.
    Only warmup if configured in config.yaml
    """
    try:
        models_config = config.get('models', {})
        warmup_enabled = models_config.get('warmup_on_startup', True)
        
        if not warmup_enabled:
            logger.info("Model warmup disabled in config")
            return
        
        logger.info("Starting model warmup...")
        
        # Initialize Azure OCR configuration
        ocr_config = config.get('ocr', {})
        if ocr_config.get('provider') in ['azure', 'auto']:
            azure_endpoint = os.getenv('AZURE_OCR_ENDPOINT')
            azure_key = os.getenv('AZURE_OCR_KEY')
            azure_region = os.getenv('AZURE_OCR_REGION')
            
            if azure_endpoint and azure_key:
                logger.info("Azure OCR initialized")
                logger.info(f"  Endpoint: {azure_endpoint}")
                logger.info(f"  Region: {azure_region or 'not specified'}")
            else:
                logger.warning("Azure OCR credentials not configured in .env")
        
        # Warmup TrOCR
        if ocr_config.get('provider') in ['local_trocr', 'auto']:
            try:
                from modules.models.trocr_loader import get_trocr_model, get_trocr_processor
                
                logger.info("Warming up TrOCR model...")
                processor = get_trocr_processor()
                model = get_trocr_model(device=models_config.get('device', 'cpu'))
                logger.success("✓ TrOCR model warmed up")
                
            except Exception as e:
                logger.warning(f"TrOCR warmup failed (will use fallback): {str(e)}")
        
        # Warmup SBERT
        embeddings_config = config.get('embeddings', {})
        if embeddings_config.get('use_local', False):
            try:
                from modules.models.sbert_loader import get_sbert_model
                
                logger.info("Warming up SBERT model...")
                model = get_sbert_model(device=models_config.get('device', 'cpu'))
                logger.success("✓ SBERT model warmed up")
                
            except Exception as e:
                logger.warning(f"SBERT warmup failed (will use fallback): {str(e)}")
        
        logger.success("Model warmup completed successfully!")
        
    except Exception as e:
        logger.error(f"Model warmup failed: {str(e)}")
        logger.warning("Application will continue, but first requests may be slower")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AnswerAI Evaluator API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "evaluate": "/api/evaluate",
            "health": "/api/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AnswerAI Evaluator"
    }


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG_MODE", "True").lower() == "true"
    
    logger.info(f"Starting AnswerAI Evaluator API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
