"""
API Routes for AnswerAI Evaluator
Handles evaluation requests and pipeline execution
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import uuid
from pathlib import Path
from loguru import logger
import yaml

# Import processing modules
from modules.image_preprocessing import preprocess_image
from modules.ocr_module import extract_text_from_image
from modules.text_preprocessing import clean_text, preprocess_text
from modules.embeddings import generate_embeddings, compute_similarity
from modules.relevance_checker import check_relevance
from modules.semantic_analysis import analyze_logic_flow
from modules.nli_contradiction import detect_contradictions
from modules.scoring import calculate_final_score

router = APIRouter()

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class EvaluationResponse(BaseModel):
    question_id: str
    student_raw_text: str
    cleaned_text: str
    similarity_score: float
    logic_flow_score: float
    contradiction_score: float
    final_score: float
    relevance_flag: str
    details: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    config_loaded: bool


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint with model status"""
    try:
        # Check if models are accessible
        from modules.embeddings import get_embedding_model
        from modules.nli_contradiction import get_nli_model
        
        embedding_model = get_embedding_model()
        nli_model = get_nli_model()
        
        models_loaded = embedding_model is not None and nli_model is not None
        
        return {
            "status": "healthy",
            "models_loaded": models_loaded,
            "config_loaded": config is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "models_loaded": False,
            "config_loaded": config is not None
        }


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_answer(
    question: str = Form(...),
    reference_answer: str = Form(...),
    answer_image: UploadFile = File(...)
):
    """
    Main evaluation endpoint - processes answer sheet through complete pipeline
    
    Pipeline Steps:
    1. Image Upload & Validation
    2. Image Preprocessing (grayscale, blur, threshold)
    3. OCR Text Extraction (Google Vision API)
    4. NLP Text Preprocessing (tokenization, lemmatization)
    5. SBERT Embedding Generation
    6. Relevance Check (cosine similarity)
    7. Semantic Analysis (logic flow)
    8. NLI Contradiction Detection
    9. Final Score Calculation
    """
    
    question_id = f"Q-{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting evaluation for {question_id}")
    
    try:
        # Step 1: Validate file
        if not answer_image.content_type.startswith("image/"):
            if answer_image.content_type != "application/pdf":
                raise HTTPException(
                    status_code=400,
                    detail="File must be an image (JPG, PNG) or PDF"
                )
        
        # Save uploaded file temporarily
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{question_id}_{answer_image.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await answer_image.read()
            buffer.write(content)
        
        logger.info(f"[{question_id}] Step 1: Image uploaded - {answer_image.filename}")
        
        # Step 2: Image Preprocessing
        preprocessed_image = preprocess_image(str(file_path))
        logger.info(f"[{question_id}] Step 2: Image preprocessed")
        
        # Step 3: OCR Extraction
        raw_text = extract_text_from_image(preprocessed_image)
        if not raw_text or len(raw_text.strip()) < 10:
            raise HTTPException(
                status_code=422,
                detail="Could not extract sufficient text from image. Please ensure the image is clear."
            )
        logger.info(f"[{question_id}] Step 3: OCR completed - {len(raw_text)} characters")
        
        # Step 4: Text Preprocessing
        cleaned_text = clean_text(raw_text)
        student_tokens = preprocess_text(cleaned_text)
        reference_tokens = preprocess_text(reference_answer)
        logger.info(f"[{question_id}] Step 4: Text preprocessing completed")
        
        # Step 5: Generate Embeddings
        student_embedding = generate_embeddings(cleaned_text)
        reference_embedding = generate_embeddings(reference_answer)
        question_embedding = generate_embeddings(question)
        logger.info(f"[{question_id}] Step 5: Embeddings generated")
        
        # Step 6: Relevance Check
        relevance_result = check_relevance(
            student_embedding,
            question_embedding,
            reference_embedding,
            threshold=config['scoring']['thresholds']['relevance']
        )
        similarity_score = relevance_result['similarity_to_reference']
        relevance_flag = relevance_result['relevance_flag']
        relevance_score = relevance_result.get('similarity_to_question', 1.0)
        logger.info(f"[{question_id}] Step 6: Relevance check - {relevance_flag}")
        
        # Step 7: Semantic Analysis (Logic Flow)
        logic_flow_score = analyze_logic_flow(
            cleaned_text,
            reference_answer,
            student_embedding,
            reference_embedding
        )
        logger.info(f"[{question_id}] Step 7: Logic flow analyzed - {logic_flow_score:.2f}")
        
        # Step 8: NLI Contradiction Detection
        contradiction_result = detect_contradictions(
            cleaned_text,
            reference_answer
        )
        contradiction_score = contradiction_result['contradiction_score']
        contradictions_found = contradiction_result['contradictions']
        logger.info(f"[{question_id}] Step 8: Contradiction check - {len(contradictions_found)} found")
        
        # Step 9: Calculate Final Score (with ML model support)
        # Prepare features for ML model
        features = {
            'essay_embedding': student_embedding,
            'cosine_similarity': similarity_score,
            'prompt_similarity': relevance_result.get('similarity_to_question', 0.0),
            'logic_flow_score': logic_flow_score,
            'contradiction_score': contradiction_score,
            'num_contradictions': len(contradictions_found),
            'word_count': len(cleaned_text.split()),
            'sentence_count': len(cleaned_text.split('.')),
            'keyword_overlap': relevance_result.get('keyword_overlap', 0.0)
        }
        
        scoring_result = calculate_final_score(
            similarity_score,
            logic_flow_score,
            contradiction_score,
            weights=config['scoring']['weights'],
            features=features,
            use_ml_model=True,
            question=question,
            reference_answer=reference_answer,
            student_answer=cleaned_text,
            relevance_score=relevance_result.get('similarity_to_question', 1.0)
        )
        
        # Extract all scores from result (handles dict or float for backward compatibility)
        if isinstance(scoring_result, dict):
            # Update all scores if enhanced model provided them
            if 'relevance_score' in scoring_result:
                relevance_score = scoring_result['relevance_score']
                # Update flag based on score
                relevance_flag = "RELEVANT" if relevance_score >= 0.5 else "NOT_RELEVANT"
            similarity_score = scoring_result.get('similarity_score', similarity_score)
            logic_flow_score = scoring_result.get('logic_flow_score', logic_flow_score)
            contradiction_score = scoring_result.get('contradiction_score', contradiction_score)
            final_score = scoring_result.get('final_score', 0.0)
            logger.info(f"[{question_id}] Using refined scores from enhanced model: relevance={relevance_score:.2f}, sim={similarity_score:.2f}, logic={logic_flow_score:.2f}, contra={contradiction_score:.2f}")
        else:
            final_score = scoring_result
            
        logger.info(f"[{question_id}] Step 9: Final score calculated - {final_score:.2f}")
        
        # Cleanup uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        # Prepare response
        response = {
            "question_id": question_id,
            "student_raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "similarity_score": round(similarity_score, 4),
            "logic_flow_score": round(logic_flow_score, 4),
            "contradiction_score": round(contradiction_score, 4),
            "final_score": round(final_score, 4),
            "relevance_flag": relevance_flag,
            "details": {
                "contradictions": contradictions_found,
                "relevance_to_question": round(relevance_score, 4),
                "token_count": len(student_tokens) if student_tokens else 0
            }
        }
        
        logger.info(f"[{question_id}] Evaluation complete ✓")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{question_id}] Evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


class TrainingResponse(BaseModel):
    status: str
    message: str
    metrics: Optional[dict] = None
    model_path: Optional[str] = None


@router.post("/train-model", response_model=TrainingResponse)
async def train_model():
    """
    Train ML scoring model on ASAP-SAS dataset
    
    This endpoint:
    1. Loads ASAP-SAS dataset
    2. Extracts features using existing modules
    3. Trains XGBoost/RandomForest regression model
    4. Saves model to models/scoring_model.pkl
    5. Returns training metrics (RMSE, MAE, QWK)
    """
    try:
        logger.info("=" * 70)
        logger.info("Model Training Request Received")
        logger.info("=" * 70)
        
        # Import training module
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from model_training.train_regression_model import train_scoring_model
        
        # Run training pipeline
        logger.info("Starting training pipeline...")
        result = train_scoring_model(use_cv=True)
        
        # Extract metrics
        metrics = result['metrics']
        val_metrics = metrics.get('val', {})
        
        response = {
            "status": "success",
            "message": f"Model trained successfully! Validation QWK: {val_metrics.get('qwk', 0):.4f}",
            "metrics": {
                "validation": val_metrics,
                "train": metrics.get('train', {}),
                "cross_validation": metrics.get('cv', {})
            },
            "model_path": "models/scoring_model.pkl"
        }
        
        logger.info("✅ Training complete!")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"Dataset files not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail="ASAP-SAS dataset not found. Please ensure dataset files are in datasets/asap/ directory."
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {str(e)}"
        )


class ModelInfoResponse(BaseModel):
    model_exists: bool
    model_path: Optional[str] = None
    model_size_mb: Optional[float] = None
    metrics: Optional[dict] = None
    last_trained: Optional[str] = None


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the trained ML model
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from model_training.save_load_model import get_model_info
        
        info = get_model_info()
        
        if info is None:
            return {
                "model_exists": False,
                "model_path": None
            }
        
        return {
            "model_exists": info['exists'],
            "model_path": info.get('path'),
            "model_size_mb": round(info.get('size_mb', 0), 2),
            "metrics": info.get('metrics', {}).get('val', {}),
            "last_trained": None  # Could add timestamp if needed
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return {
            "model_exists": False,
            "model_path": None
        }


@router.post("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "API is working!", "status": "success"}


@router.post("/ocr/test-azure")
async def test_azure_ocr(file: UploadFile = File(...)):
    """
    Test Azure OCR with a sample image.
    Upload an image to test Azure Computer Vision OCR functionality.
    
    Args:
        file: Image file (JPG, PNG)
        
    Returns:
        OCR result with extracted text and metadata
    """
    try:
        import time
        from modules.azure_ocr import azure_extract_text
        
        logger.info(f"Testing Azure OCR with file: {file.filename}")
        
        # Read image bytes
        image_bytes = await file.read()
        
        start_time = time.time()
        
        # Perform OCR
        text, confidence = azure_extract_text(image_bytes)
        
        elapsed = time.time() - start_time
        
        result = {
            "status": "success",
            "filename": file.filename,
            "extracted_text": text,
            "confidence": confidence,
            "text_length": len(text),
            "word_count": len(text.split()),
            "processing_time": f"{elapsed:.2f}s",
            "ocr_provider": "Azure Computer Vision"
        }
        
        logger.info(f"Azure OCR test completed: {len(text)} chars in {elapsed:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Azure OCR test failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Azure OCR test failed"
        }


@router.get("/models/health")
async def models_health():
    """
    Detailed health check for all loaded models.
    Returns model status, device, versions, and performance metrics.
    """
    try:
        import time
        import torch
        from modules.models.trocr_loader import get_model_info as get_trocr_info
        from modules.models.sbert_loader import get_model_info as get_sbert_info
        
        health_data = {
            "timestamp": time.time(),
            "status": "healthy",
            "models": {}
        }
        
        # Check TrOCR model
        try:
            trocr_info = get_trocr_info()
            health_data["models"]["trocr"] = {
                **trocr_info,
                "provider": config.get('ocr', {}).get('provider', 'unknown'),
                "status": "loaded" if trocr_info.get('model_loaded') else "not_loaded"
            }
        except Exception as e:
            health_data["models"]["trocr"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check SBERT model
        try:
            sbert_info = get_sbert_info()
            health_data["models"]["sbert"] = {
                **sbert_info,
                "use_local": config.get('embeddings', {}).get('use_local', False),
                "status": "loaded" if sbert_info.get('model_loaded') else "not_loaded"
            }
        except Exception as e:
            health_data["models"]["sbert"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check NLI model
        try:
            from modules.nli_contradiction import get_nli_model
            nli_model = get_nli_model()
            health_data["models"]["nli"] = {
                "status": "loaded" if nli_model is not None else "not_loaded",
                "model_name": config.get('nli', {}).get('model_name', 'unknown')
            }
        except Exception as e:
            health_data["models"]["nli"] = {
                "status": "error",
                "error": str(e)
            }
        
        # System information
        health_data["system"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "configured_device": config.get('models', {}).get('device', 'cpu')
        }
        
        # Check if any critical models failed to load
        critical_models = ["trocr", "sbert"]
        all_healthy = all(
            health_data["models"].get(model, {}).get("status") in ["loaded", "not_loaded"]
            for model in critical_models
        )
        
        health_data["status"] = "healthy" if all_healthy else "degraded"
        
        return health_data
        
    except Exception as e:
        logger.error(f"Models health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to retrieve model health information"
        }
