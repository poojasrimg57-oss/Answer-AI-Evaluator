# AnswerAI Evaluator - Python Backend

Complete Python FastAPI backend for the handwritten answer sheet evaluation system.

## 📁 Project Structure

```
answer-evaluator-backend/
├── main.py                    # FastAPI application entry point
├── api_routes.py              # API endpoint definitions
├── config.yaml                # Configuration file (weights, thresholds)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # Setup instructions
├── setup_instructions.md      # Step-by-step guide
│
├── modules/
│   ├── __init__.py
│   ├── image_preprocessing.py # Step 2: Image preprocessing
│   ├── ocr_module.py          # Step 3: OCR extraction
│   ├── text_preprocessing.py  # Step 4: NLP text cleaning
│   ├── embeddings.py          # Step 5: SBERT embeddings
│   ├── relevance_checker.py   # Step 6: Cosine similarity
│   ├── semantic_analysis.py   # Step 7: Logic flow analysis
│   ├── nli_contradiction.py   # Step 8: Contradiction detection
│   └── scoring.py             # Step 9: Final score calculation
│
├── tests/
│   ├── __init__.py
│   ├── test_text_preprocessing.py
│   ├── test_relevance_checker.py
│   └── test_scoring.py
│
├── static/                    # Frontend files (optional)
│   └── index.html
│
└── demo_data/
    ├── sample_question.txt
    ├── sample_reference.txt
    └── sample_answer_sheet.jpg
```

---

## 📄 File Contents

### `requirements.txt`

```txt
# Core
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
python-dotenv==1.0.0
pyyaml==6.0.1

# Image Processing
opencv-python==4.9.0.80
Pillow==10.2.0

# OCR
google-cloud-vision==3.5.0
pytesseract==0.3.10

# NLP & ML
torch==2.1.2
transformers==4.36.2
sentence-transformers==2.2.2
scikit-learn==1.4.0
numpy==1.26.3
nltk==3.8.1
spacy==3.7.2

# Testing
pytest==7.4.4
httpx==0.26.0
```

---

### `config.yaml`

```yaml
# Scoring weights (must sum to 1.0)
scoring:
  w_similarity: 0.5
  w_logic: 0.3
  w_correctness: 0.2

# Thresholds
thresholds:
  relevance_threshold: 0.4
  contradiction_threshold: 0.5

# Models
models:
  sbert_model: "all-MiniLM-L6-v2"
  nli_model: "facebook/bart-large-mnli"

# OCR settings
ocr:
  default_method: "vision_api"  # or "tesseract"
  tesseract_config: "--oem 3 --psm 6"

# Image preprocessing
preprocessing:
  target_width: 1024
  blur_kernel: 5
  threshold_block_size: 11
```

---

### `.env.example`

```env
# Google Cloud Vision API
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json

# Optional: Tesseract path (Windows)
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

---

### `main.py`

```python
"""
AnswerAI Evaluator - FastAPI Application
Main entry point for the handwritten answer evaluation system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import yaml
import os

from api_routes import router

# Load environment variables
load_dotenv()

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="AnswerAI Evaluator",
    description="AI-based system for evaluating handwritten answer sheets",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend (optional)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(router, prefix="/api")

# Store config in app state for access in routes
app.state.config = config


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AnswerAI Evaluator"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "config_loaded": True,
        "scoring_weights": config.get("scoring", {})
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )
```

---

### `api_routes.py`

```python
"""
API Routes for the evaluation system.
POST /evaluate_answer - Main evaluation endpoint.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import uuid

from modules.image_preprocessing import preprocess_image
from modules.ocr_module import extract_text
from modules.text_preprocessing import preprocess_answer_text
from modules.embeddings import get_sentence_embeddings, EmbeddingModel
from modules.relevance_checker import compute_cosine_similarity, is_answer_relevant
from modules.semantic_analysis import evaluate_logic_flow
from modules.nli_contradiction import detect_contradictions, NLIModel
from modules.scoring import calculate_final_score

router = APIRouter()

# Initialize models (loaded once at startup)
embedding_model: Optional[EmbeddingModel] = None
nli_model: Optional[NLIModel] = None


def get_models(config: dict):
    """Lazy load ML models."""
    global embedding_model, nli_model
    
    if embedding_model is None:
        embedding_model = EmbeddingModel(config["models"]["sbert_model"])
    if nli_model is None:
        nli_model = NLIModel(config["models"]["nli_model"])
    
    return embedding_model, nli_model


class EvaluationResult(BaseModel):
    """Response model for evaluation results."""
    question_id: str
    student_raw_text: str
    cleaned_text: str
    similarity_score: float
    logic_flow_score: float
    contradiction_score: float
    final_score: float
    relevance_flag: str  # "relevant" or "irrelevant"


@router.post("/evaluate_answer", response_model=EvaluationResult)
async def evaluate_answer(
    request: Request,
    question: str = Form(..., description="The exam question"),
    reference_answer: str = Form(..., description="The model/reference answer"),
    answer_image: UploadFile = File(..., description="Scanned answer sheet image")
):
    """
    Main evaluation endpoint.
    
    Pipeline:
    1. Receive answer sheet image
    2. Preprocess image (grayscale, blur, threshold)
    3. Run OCR (Vision API or Tesseract)
    4. Preprocess text (NLP: tokenize, lemmatize)
    5. Generate SBERT embeddings
    6. Check relevance via cosine similarity
    7. Evaluate semantic logic flow
    8. Detect contradictions via NLI
    9. Calculate weighted final score
    """
    config = request.app.state.config
    emb_model, nli_mod = get_models(config)
    
    question_id = f"Q-{uuid.uuid4().hex[:8]}"
    
    try:
        # Step 1: Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await answer_image.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Step 2: Preprocess image (Gaussian blur, etc.)
        preprocessed_image = preprocess_image(
            tmp_path,
            target_width=config["preprocessing"]["target_width"],
            blur_kernel=config["preprocessing"]["blur_kernel"],
            threshold_block_size=config["preprocessing"]["threshold_block_size"]
        )
        
        # Step 3: Run OCR
        ocr_method = config["ocr"]["default_method"]
        student_raw_text = extract_text(preprocessed_image, method=ocr_method)
        
        # Step 4: Text preprocessing (NLP)
        cleaned_text = preprocess_answer_text(student_raw_text)
        
        # Step 5: Generate SBERT embeddings
        student_embedding = emb_model.get_embedding(cleaned_text)
        reference_embedding = emb_model.get_embedding(reference_answer)
        
        # Step 6: Relevance check via cosine similarity
        similarity_score = compute_cosine_similarity(student_embedding, reference_embedding)
        threshold = config["thresholds"]["relevance_threshold"]
        relevance_flag = "relevant" if is_answer_relevant(
            student_embedding, reference_embedding, threshold
        ) else "irrelevant"
        
        # Step 7: Semantic analysis (logic flow)
        if relevance_flag == "relevant":
            logic_flow_score = evaluate_logic_flow(
                cleaned_text, reference_answer, emb_model
            )
        else:
            logic_flow_score = 0.0  # Skip if irrelevant
        
        # Step 8: Contradiction detection (NLI)
        contradiction_score = detect_contradictions(
            cleaned_text, reference_answer, nli_mod
        )
        
        # Step 9: Calculate final score
        weights = config["scoring"]
        final_score = calculate_final_score(
            similarity_score=similarity_score,
            logic_score=logic_flow_score,
            correctness_score=contradiction_score,
            w_sim=weights["w_similarity"],
            w_log=weights["w_logic"],
            w_cor=weights["w_correctness"]
        )
        
        # Cleanup temp file
        os.unlink(tmp_path)
        
        return EvaluationResult(
            question_id=question_id,
            student_raw_text=student_raw_text,
            cleaned_text=cleaned_text,
            similarity_score=round(similarity_score, 4),
            logic_flow_score=round(logic_flow_score, 4),
            contradiction_score=round(contradiction_score, 4),
            final_score=round(final_score, 4),
            relevance_flag=relevance_flag
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
```

---

### `modules/image_preprocessing.py`

```python
"""
Step 2: Image Preprocessing
Prepares scanned answer sheets for OCR.
"""

import cv2
import numpy as np
from typing import Union


def preprocess_image(
    image_path: str,
    target_width: int = 1024,
    blur_kernel: int = 5,
    threshold_block_size: int = 11
) -> np.ndarray:
    """
    Preprocess image for OCR.
    
    Pipeline (matching flowchart):
    - Load image
    - Convert to grayscale
    - Resize (maintain aspect ratio)
    - Apply Gaussian Blur
    - Apply adaptive thresholding / binarization
    
    Args:
        image_path: Path to the input image
        target_width: Target width for resizing
        blur_kernel: Kernel size for Gaussian blur (must be odd)
        threshold_block_size: Block size for adaptive threshold
    
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Step 2a: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2b: Resize maintaining aspect ratio
    h, w = gray.shape
    if w > target_width:
        ratio = target_width / w
        new_h = int(h * ratio)
        gray = cv2.resize(gray, (target_width, new_h), interpolation=cv2.INTER_AREA)
    
    # Step 2c: Apply Gaussian Blur (reduces noise for better OCR)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # Step 2d: Apply adaptive thresholding (binarization)
    # This helps separate text from background
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        threshold_block_size,
        2
    )
    
    return binary


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Optional: Correct skew in scanned documents.
    Useful for poorly scanned answer sheets.
    """
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated
```

---

### `modules/ocr_module.py`

```python
"""
Step 3: OCR Module
Extracts text from preprocessed images using Vision API or Tesseract.
"""

import os
from typing import Union
import numpy as np


def run_vision_api_ocr(image: np.ndarray) -> str:
    """
    Extract text using Google Cloud Vision API.
    Optimized for handwriting recognition.
    
    Requires:
        - GOOGLE_APPLICATION_CREDENTIALS env var set
        - google-cloud-vision package installed
    """
    from google.cloud import vision
    import cv2
    
    # Encode image for API
    _, encoded = cv2.imencode('.jpg', image)
    content = encoded.tobytes()
    
    # Initialize Vision client
    client = vision.ImageAnnotatorClient()
    vision_image = vision.Image(content=content)
    
    # Use document_text_detection for better handwriting support
    response = client.document_text_detection(image=vision_image)
    
    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")
    
    # Extract full text
    full_text = response.full_text_annotation.text if response.full_text_annotation else ""
    
    return full_text


def run_tesseract_ocr(image: np.ndarray, config: str = "--oem 3 --psm 6") -> str:
    """
    Extract text using Tesseract OCR (local fallback).
    
    Args:
        image: Preprocessed image
        config: Tesseract configuration string
            --oem 3: Use LSTM engine
            --psm 6: Assume uniform block of text
    
    Requires:
        - Tesseract installed on system
        - pytesseract package
    """
    import pytesseract
    
    # Set Tesseract path if configured
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    text = pytesseract.image_to_string(image, config=config)
    return text


def extract_text(image: np.ndarray, method: str = "vision_api") -> str:
    """
    Unified OCR function.
    
    Args:
        image: Preprocessed image array
        method: "vision_api" or "tesseract"
    
    Returns:
        Extracted text string
    """
    if method == "vision_api":
        return run_vision_api_ocr(image)
    elif method == "tesseract":
        return run_tesseract_ocr(image)
    else:
        raise ValueError(f"Unknown OCR method: {method}")
```

---

### `modules/text_preprocessing.py`

```python
"""
Step 4: Text Preprocessing (NLP)
Cleans and normalizes OCR output for embedding generation.
"""

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def clean_ocr_artifacts(text: str) -> str:
    """
    Remove common OCR artifacts and normalize text.
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common OCR noise characters
    text = re.sub(r'[^\w\s.,;:?!\'"-]', '', text)
    
    # Fix common OCR mistakes
    text = text.replace('|', 'l')  # pipe -> l
    text = text.replace('0', 'o')  # zero in words (be careful with this)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return word_tokenize(text.lower())


def remove_stopwords(tokens: List[str], language: str = 'english') -> List[str]:
    """Remove stopwords from token list."""
    stop_words = set(stopwords.words(language))
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize tokens to base forms."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_answer_text(raw_text: str) -> str:
    """
    Full NLP preprocessing pipeline.
    
    Steps:
    1. Clean OCR artifacts
    2. Tokenization
    3. Stopword removal
    4. Lemmatization
    5. Reconstruct cleaned text
    
    Args:
        raw_text: Raw OCR output
    
    Returns:
        Cleaned, normalized text ready for embedding
    """
    # Step 4a: Clean OCR artifacts
    cleaned = clean_ocr_artifacts(raw_text)
    
    # Step 4b: Tokenize
    tokens = tokenize(cleaned)
    
    # Step 4c: Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # Step 4d: Lemmatize
    tokens = lemmatize(tokens)
    
    # Reconstruct text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text


def get_sentences(text: str) -> List[str]:
    """Split text into sentences (useful for embedding)."""
    return sent_tokenize(text)
```

---

### `modules/embeddings.py`

```python
"""
Step 5: Sentence Embeddings (SBERT)
Generates vector representations for semantic comparison.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union


class EmbeddingModel:
    """
    Wrapper for Sentence-BERT model.
    Caches model for efficient reuse.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SBERT model.
        
        Args:
            model_name: HuggingFace model identifier
                - "all-MiniLM-L6-v2": Fast, good quality (default)
                - "all-mpnet-base-v2": Higher quality, slower
                - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text (can be sentence or paragraph)
        
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
        
        Returns:
            2D numpy array of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def get_sentence_embeddings(self, text: str) -> List[np.ndarray]:
        """
        Split text into sentences and embed each separately.
        Useful for fine-grained analysis.
        """
        from modules.text_preprocessing import get_sentences
        sentences = get_sentences(text)
        return [self.get_embedding(s) for s in sentences]


def get_sentence_embeddings(text: str, model: EmbeddingModel = None) -> np.ndarray:
    """
    Convenience function for single text embedding.
    Creates model if not provided (less efficient for multiple calls).
    """
    if model is None:
        model = EmbeddingModel()
    return model.get_embedding(text)
```

---

### `modules/relevance_checker.py`

```python
"""
Step 6: Relevance Check via Cosine Similarity
Determines if student answer is relevant to the question/reference.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Similarity score between 0 and 1
    """
    # Reshape for sklearn
    v1 = vec1.reshape(1, -1)
    v2 = vec2.reshape(1, -1)
    
    similarity = cosine_similarity(v1, v2)[0][0]
    
    # Ensure score is between 0 and 1
    return float(max(0, min(1, similarity)))


def is_answer_relevant(
    student_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    threshold: float = 0.4
) -> bool:
    """
    Check if student answer is relevant based on similarity threshold.
    
    This implements the decision diamond in the flowchart:
    - If similarity >= threshold: relevant (proceed to semantic analysis)
    - If similarity < threshold: irrelevant (flag and give low score)
    
    Args:
        student_embedding: SBERT embedding of student answer
        reference_embedding: SBERT embedding of reference answer
        threshold: Minimum similarity for relevance (default 0.4)
    
    Returns:
        True if relevant, False otherwise
    """
    similarity = compute_cosine_similarity(student_embedding, reference_embedding)
    return similarity >= threshold


def compute_query_relevance(
    student_embedding: np.ndarray,
    question_embedding: np.ndarray,
    threshold: float = 0.3
) -> bool:
    """
    Optional: Check if answer is relevant to the question itself.
    Lower threshold since question-answer similarity is naturally lower.
    """
    similarity = compute_cosine_similarity(student_embedding, question_embedding)
    return similarity >= threshold
```

---

### `modules/semantic_analysis.py`

```python
"""
Step 7: Semantic Analysis - Logic Flow Evaluation
Analyzes structural coherence and concept coverage.
"""

import numpy as np
from typing import List
from modules.embeddings import EmbeddingModel
from modules.text_preprocessing import get_sentences


def evaluate_logic_flow(
    student_text: str,
    reference_text: str,
    model: EmbeddingModel
) -> float:
    """
    Evaluate semantic logic flow and coherence.
    
    This checks:
    1. Sentence ordering similarity
    2. Key concept coverage
    3. Internal coherence
    
    Args:
        student_text: Preprocessed student answer
        reference_text: Reference/model answer
        model: SBERT embedding model
    
    Returns:
        Logic flow score between 0 and 1
    """
    # Get sentence-level embeddings
    student_sentences = get_sentences(student_text)
    reference_sentences = get_sentences(reference_text)
    
    if len(student_sentences) == 0:
        return 0.0
    
    # Embed all sentences
    student_embeddings = [model.get_embedding(s) for s in student_sentences]
    reference_embeddings = [model.get_embedding(s) for s in reference_sentences]
    
    # 1. Concept coverage score
    coverage_score = compute_concept_coverage(
        student_embeddings, reference_embeddings
    )
    
    # 2. Coherence score (how well sentences flow together)
    coherence_score = compute_coherence(student_embeddings)
    
    # 3. Order similarity (rough structural alignment)
    order_score = compute_order_similarity(
        student_embeddings, reference_embeddings
    )
    
    # Combine scores
    logic_flow_score = (
        0.4 * coverage_score +
        0.3 * coherence_score +
        0.3 * order_score
    )
    
    return float(max(0, min(1, logic_flow_score)))


def compute_concept_coverage(
    student_embs: List[np.ndarray],
    reference_embs: List[np.ndarray]
) -> float:
    """
    How many reference concepts are covered by student answer.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    if len(reference_embs) == 0 or len(student_embs) == 0:
        return 0.0
    
    # For each reference sentence, find max similarity to any student sentence
    coverage_scores = []
    for ref_emb in reference_embs:
        max_sim = max(
            cosine_similarity(ref_emb.reshape(1, -1), s.reshape(1, -1))[0][0]
            for s in student_embs
        )
        coverage_scores.append(max_sim)
    
    return float(np.mean(coverage_scores))


def compute_coherence(embeddings: List[np.ndarray]) -> float:
    """
    Measure internal coherence via consecutive sentence similarity.
    Higher coherence = sentences flow logically together.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    if len(embeddings) < 2:
        return 1.0  # Single sentence is coherent by default
    
    consecutive_sims = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        consecutive_sims.append(sim)
    
    return float(np.mean(consecutive_sims))


def compute_order_similarity(
    student_embs: List[np.ndarray],
    reference_embs: List[np.ndarray]
) -> float:
    """
    Check if student follows similar ordering to reference.
    Uses position-weighted matching.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    if len(student_embs) == 0 or len(reference_embs) == 0:
        return 0.0
    
    # Simplified: compare position-matched similarities
    min_len = min(len(student_embs), len(reference_embs))
    position_sims = []
    
    for i in range(min_len):
        sim = cosine_similarity(
            student_embs[i].reshape(1, -1),
            reference_embs[i].reshape(1, -1)
        )[0][0]
        position_sims.append(sim)
    
    return float(np.mean(position_sims))
```

---

### `modules/nli_contradiction.py`

```python
"""
Step 8: Contradiction Detection using NLI Model
Detects factual contradictions using Natural Language Inference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple
import numpy as np


class NLIModel:
    """
    Natural Language Inference model for contradiction detection.
    Uses entailment/neutral/contradiction classification.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize NLI model.
        
        Args:
            model_name: HuggingFace model identifier
                - "facebook/bart-large-mnli": High quality
                - "roberta-large-mnli": Alternative
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Label mapping (model-specific)
        self.labels = ["contradiction", "neutral", "entailment"]
    
    def predict(self, premise: str, hypothesis: str) -> Tuple[str, np.ndarray]:
        """
        Predict NLI label for a premise-hypothesis pair.
        
        Args:
            premise: The reference/ground truth text
            hypothesis: The student answer (claim to verify)
        
        Returns:
            Tuple of (predicted_label, probability_distribution)
        """
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
        
        predicted_idx = np.argmax(probs)
        predicted_label = self.labels[predicted_idx]
        
        return predicted_label, probs


def detect_contradictions(
    student_text: str,
    reference_text: str,
    model: NLIModel
) -> float:
    """
    Detect contradictions between student answer and reference.
    
    Returns a score where:
    - Higher score = fewer contradictions (more correct)
    - Lower score = more contradictions detected
    
    Args:
        student_text: The student's answer
        reference_text: The reference/correct answer
        model: NLI model instance
    
    Returns:
        Correctness score between 0 and 1
    """
    from modules.text_preprocessing import get_sentences
    
    student_sentences = get_sentences(student_text)
    
    if len(student_sentences) == 0:
        return 0.0
    
    # For each student sentence, check for contradictions
    contradiction_scores = []
    
    for sentence in student_sentences:
        # Use reference as premise, student sentence as hypothesis
        _, probs = model.predict(reference_text, sentence)
        
        # probs: [contradiction, neutral, entailment]
        contradiction_prob = probs[0]
        entailment_prob = probs[2]
        
        # Score: prefer entailment, penalize contradiction
        sentence_score = (1 - contradiction_prob) * 0.5 + entailment_prob * 0.5
        contradiction_scores.append(sentence_score)
    
    # Average across all sentences
    final_score = float(np.mean(contradiction_scores))
    
    return max(0, min(1, final_score))
```

---

### `modules/scoring.py`

```python
"""
Step 9: Final Scoring Algorithm
Combines all component scores into a weighted final score.
"""


def calculate_final_score(
    similarity_score: float,
    logic_score: float,
    correctness_score: float,
    w_sim: float = 0.5,
    w_log: float = 0.3,
    w_cor: float = 0.2
) -> float:
    """
    Calculate weighted final score.
    
    Formula:
        final_score = w_sim * similarity_score + 
                      w_log * logic_score + 
                      w_cor * correctness_score
    
    Args:
        similarity_score: Cosine similarity to reference (0-1)
        logic_score: Logic flow / coherence score (0-1)
        correctness_score: NLI-based correctness (0-1)
        w_sim: Weight for similarity (default 0.5)
        w_log: Weight for logic (default 0.3)
        w_cor: Weight for correctness (default 0.2)
    
    Returns:
        Final weighted score between 0 and 1
    
    Note:
        Weights should sum to 1.0 for normalized scoring.
        Adjust in config.yaml based on grading priorities.
    """
    # Validate inputs
    scores = [similarity_score, logic_score, correctness_score]
    for s in scores:
        if not 0 <= s <= 1:
            raise ValueError(f"Scores must be between 0 and 1, got {s}")
    
    weights = [w_sim, w_log, w_cor]
    weight_sum = sum(weights)
    
    if abs(weight_sum - 1.0) > 0.01:
        # Normalize weights if they don't sum to 1
        weights = [w / weight_sum for w in weights]
    
    # Calculate weighted sum
    final_score = (
        weights[0] * similarity_score +
        weights[1] * logic_score +
        weights[2] * correctness_score
    )
    
    return float(max(0, min(1, final_score)))


def get_verdict(score: float) -> str:
    """
    Convert numeric score to human-readable verdict.
    
    Args:
        score: Final score between 0 and 1
    
    Returns:
        Verdict string
    """
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.70:
        return "Good"
    elif score >= 0.50:
        return "Needs Improvement"
    else:
        return "Poor"


def get_score_breakdown(
    similarity_score: float,
    logic_score: float,
    correctness_score: float,
    w_sim: float = 0.5,
    w_log: float = 0.3,
    w_cor: float = 0.2
) -> dict:
    """
    Get detailed score breakdown for transparency.
    """
    final = calculate_final_score(
        similarity_score, logic_score, correctness_score,
        w_sim, w_log, w_cor
    )
    
    return {
        "components": {
            "similarity": {
                "score": similarity_score,
                "weight": w_sim,
                "contribution": similarity_score * w_sim
            },
            "logic_flow": {
                "score": logic_score,
                "weight": w_log,
                "contribution": logic_score * w_log
            },
            "correctness": {
                "score": correctness_score,
                "weight": w_cor,
                "contribution": correctness_score * w_cor
            }
        },
        "final_score": final,
        "verdict": get_verdict(final)
    }
```

---

### `tests/test_text_preprocessing.py`

```python
"""Unit tests for text preprocessing module."""

import pytest
from modules.text_preprocessing import (
    clean_ocr_artifacts,
    tokenize,
    remove_stopwords,
    lemmatize,
    preprocess_answer_text
)


class TestCleanOCRArtifacts:
    def test_removes_extra_whitespace(self):
        text = "Hello    world   test"
        result = clean_ocr_artifacts(text)
        assert result == "Hello world test"
    
    def test_removes_special_characters(self):
        text = "Hello@#$world"
        result = clean_ocr_artifacts(text)
        assert "@" not in result
        assert "#" not in result


class TestTokenize:
    def test_basic_tokenization(self):
        text = "Hello world"
        tokens = tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens
    
    def test_lowercase(self):
        text = "HELLO World"
        tokens = tokenize(text)
        assert all(t.islower() for t in tokens)


class TestRemoveStopwords:
    def test_removes_common_stopwords(self):
        tokens = ["the", "cat", "is", "on", "the", "mat"]
        result = remove_stopwords(tokens)
        assert "the" not in result
        assert "is" not in result
        assert "cat" in result


class TestLemmatize:
    def test_lemmatizes_plurals(self):
        tokens = ["cats", "dogs", "running"]
        result = lemmatize(tokens)
        assert "cat" in result
        assert "dog" in result


class TestFullPreprocessing:
    def test_full_pipeline(self):
        raw = "  The cats   are running around the house!  "
        result = preprocess_answer_text(raw)
        assert isinstance(result, str)
        assert len(result) > 0
```

---

### `tests/test_relevance_checker.py`

```python
"""Unit tests for relevance checker module."""

import pytest
import numpy as np
from modules.relevance_checker import (
    compute_cosine_similarity,
    is_answer_relevant
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = np.array([1.0, 2.0, 3.0])
        similarity = compute_cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.001
    
    def test_orthogonal_vectors(self):
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.001
    
    def test_returns_float(self):
        vec1 = np.random.rand(384)
        vec2 = np.random.rand(384)
        result = compute_cosine_similarity(vec1, vec2)
        assert isinstance(result, float)


class TestIsAnswerRelevant:
    def test_similar_vectors_are_relevant(self):
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.1, 2.1, 3.1])  # Very similar
        assert is_answer_relevant(vec1, vec2, threshold=0.9)
    
    def test_dissimilar_vectors_not_relevant(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])  # Orthogonal
        assert not is_answer_relevant(vec1, vec2, threshold=0.5)
    
    def test_threshold_boundary(self):
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])  # Identical
        assert is_answer_relevant(vec1, vec2, threshold=1.0)
```

---

### `tests/test_scoring.py`

```python
"""Unit tests for scoring module."""

import pytest
from modules.scoring import (
    calculate_final_score,
    get_verdict,
    get_score_breakdown
)


class TestCalculateFinalScore:
    def test_perfect_scores(self):
        result = calculate_final_score(1.0, 1.0, 1.0)
        assert result == 1.0
    
    def test_zero_scores(self):
        result = calculate_final_score(0.0, 0.0, 0.0)
        assert result == 0.0
    
    def test_weighted_calculation(self):
        result = calculate_final_score(
            similarity_score=0.8,
            logic_score=0.6,
            correctness_score=0.9,
            w_sim=0.5,
            w_log=0.3,
            w_cor=0.2
        )
        expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.9
        assert abs(result - expected) < 0.001
    
    def test_score_bounds(self):
        result = calculate_final_score(0.5, 0.5, 0.5)
        assert 0 <= result <= 1


class TestGetVerdict:
    def test_excellent(self):
        assert get_verdict(0.9) == "Excellent"
    
    def test_good(self):
        assert get_verdict(0.75) == "Good"
    
    def test_needs_improvement(self):
        assert get_verdict(0.55) == "Needs Improvement"
    
    def test_poor(self):
        assert get_verdict(0.3) == "Poor"


class TestScoreBreakdown:
    def test_returns_all_components(self):
        breakdown = get_score_breakdown(0.8, 0.7, 0.9)
        assert "components" in breakdown
        assert "final_score" in breakdown
        assert "verdict" in breakdown
    
    def test_contributions_sum_to_final(self):
        breakdown = get_score_breakdown(0.8, 0.7, 0.9)
        total_contribution = sum(
            c["contribution"] for c in breakdown["components"].values()
        )
        assert abs(total_contribution - breakdown["final_score"]) < 0.001
```

---

### `README.md` (Backend)

```markdown
# AnswerAI Evaluator - Python Backend

AI-based system for evaluating handwritten answer sheets using OCR, SBERT embeddings, 
semantic analysis, and NLI contradiction detection.

## Features

- **Image Preprocessing**: Grayscale, Gaussian blur, adaptive thresholding
- **OCR**: Google Cloud Vision API (primary) or Tesseract (fallback)
- **NLP**: Tokenization, stopword removal, lemmatization
- **Embeddings**: SBERT sentence vectors for semantic comparison
- **Relevance Check**: Cosine similarity with configurable threshold
- **Semantic Analysis**: Logic flow and coherence evaluation
- **Contradiction Detection**: NLI model (BART-MNLI) for factual verification
- **Weighted Scoring**: Configurable weights for final grade calculation

## Quick Start

```bash
# Clone and setup
git clone <your-repo>
cd answer-evaluator-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Configure environment
cp .env.example .env
# Edit .env with your Google Vision API credentials

# Run server
uvicorn main:app --reload
```

## API Usage

```bash
curl -X POST "http://localhost:8000/api/evaluate_answer" \
  -F "question=Explain photosynthesis" \
  -F "reference_answer=Photosynthesis is the process..." \
  -F "answer_image=@student_answer.jpg"
```

## Configuration

Edit `config.yaml` to adjust:
- Scoring weights (similarity, logic, correctness)
- Relevance thresholds
- Model selections
- OCR settings

## Testing

```bash
pytest tests/ -v
```

## License

MIT
```

---

### `setup_instructions.md`

```markdown
# Setup Instructions

## Prerequisites

1. Python 3.9+ installed
2. Google Cloud account (for Vision API) OR Tesseract installed locally
3. ~2GB disk space for ML models

## Step-by-Step Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 4. Configure Google Vision API

1. Go to Google Cloud Console
2. Create a new project or select existing
3. Enable "Cloud Vision API"
4. Create a service account and download JSON key
5. Set environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/key.json
```

### 5. (Alternative) Configure Tesseract

If using Tesseract instead of Vision API:

```bash
# Ubuntu
sudo apt install tesseract-ocr

# Mac
brew install tesseract

# Windows: Download installer from GitHub
```

Update `.env`:
```
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

And in `config.yaml`:
```yaml
ocr:
  default_method: "tesseract"
```

### 6. Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Test the API

```bash
curl http://localhost:8000/health
```

### 8. Connect Frontend

Update your React frontend to point to:
```
http://localhost:8000/api/evaluate_answer
```
```
