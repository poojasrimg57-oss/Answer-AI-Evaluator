# AnswerAI Evaluator - Backend API

AI-Powered Answer Sheet Evaluation System using OCR, SBERT Embeddings, Semantic Analysis, Natural Language Inference, and **Machine Learning**.

## 🚀 Features

- **Local AI Models**: ⭐ NEW - Offline OCR (TrOCR) and embeddings (SBERT)
- **Advanced OCR**: Local TrOCR handwriting model with Vision API fallback
- **Semantic Analysis**: Local SBERT embeddings for meaning comparison
- **Logic Flow Analysis**: Evaluates coherence and argument structure
- **Contradiction Detection**: NLI model identifies factual errors
- **ML Model Training**: Train models on ASAP-SAS dataset
- **Intelligent Scoring**: ML model or weighted scoring (fallback)
- **REST API**: FastAPI with automatic documentation
- **Production Ready**: Model warmup, health checks, caching

## 📋 System Architecture

```
Evaluation Pipeline:
1. Image Upload → 2. Preprocessing → 3. Local TrOCR Extraction → 
4. NLP Cleaning → 5. Local SBERT Embeddings → 6. Relevance Check →
7. Semantic Analysis → 8. NLI Contradiction → 9. ML Model Scoring

Training Pipeline:
ASAP-SAS Dataset → Feature Extraction → XGBoost Training → 
Model Evaluation (QWK) → Model Deployment

Local Models & Cloud Services: ⭐ NEW
├── Azure Computer Vision (Cloud OCR) ⭐ NEW
├── TrOCR (Local OCR): models/trocr-small-handwritten/
├── SBERT (Embeddings): models/all-MiniLM-L6-v2/
└── Fallback chain: TrOCR → Azure → Tesseract
```

## 🛠️ Tech Stack ⭐ UPDATED

- **Framework**: FastAPI
- **OCR**: ⭐ UPDATED
  - Azure Computer Vision (Cloud) ⭐ NEW - Primary OCR
  - TrOCR (Microsoft) - Local handwriting OCR
  - Tesseract - Universal fallback
- **Local Models**:
  - TrOCR (microsoft/trocr-small-handwritten)
  - SBERT (all-MiniLM-L6-v2)
- **Machine Learning**: XGBoost, Scikit-learn
- **NLP**: NLTK, spaCy
- **Embeddings**: Sentence-Transformers (SBERT)
- **NLI**: Hugging Face Transformers (BART-MNLI)
- **Image Processing**: OpenCV, Pillow
- **Deep Learning**: PyTorch, Transformers

## 📦 Installation

### Prerequisites

- Python 3.10 or 3.11 (3.13 not yet supported)
- pip or conda
- Azure account with Computer Vision service (for cloud OCR)

### Quick Start

1. **Clone the repository**
```bash
cd Backend
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLP models**
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

5. **Configure environment variables**
```bash
# Copy example env file
copy .env.example .env
# Edit .env with your settings
```

6. **Set up Azure Computer Vision (for cloud OCR)**

Create `.env` file in Backend directory with your Azure credentials:

```env
AZURE_OCR_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OCR_KEY=your_azure_subscription_key_here
AZURE_OCR_REGION=your_region
```

**To get Azure credentials:**
- Go to [Azure Portal](https://portal.azure.com)
- Create a Computer Vision resource
- Copy the Endpoint and Key from "Keys and Endpoint" section
- Add them to `.env` file

**Note**: Azure OCR is used by default. You can switch to local TrOCR by changing `ocr.provider` in `config.yaml`

7. **Run the server**
```bash
python main.py
```

Server will start on `http://localhost:8000`

---

## 🤖 Local Models Setup ⭐ NEW

The system now uses **local AI models** for offline operation and improved performance!

### Models Included

The following models are already downloaded in the `models/` directory:

1. **TrOCR** (`models/trocr-small-handwritten/`)
   - Microsoft's handwriting OCR model
   - Used for extracting text from answer sheets
   - Faster and more accurate than cloud APIs

2. **SBERT** (`models/all-MiniLM-L6-v2/`)
   - Sentence-BERT embedding model (384 dimensions)
   - Used for semantic similarity computation
   - Optimized for speed and accuracy

### Configuration

Edit `config.yaml` to configure model usage:

```yaml
models:
  trocr_local_path: "models/trocr-small-handwritten"
  sbert_local_path: "models/all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda" for GPU acceleration
  warmup_on_startup: true

ocr:
  provider: "azure"  # Options: azure, local_trocr, tesseract, auto
  min_confidence: 0.3
  
  # Azure Computer Vision settings (from .env)
  azure_timeout: 30        # Max time to wait for OCR result
  azure_poll_interval: 1   # Polling interval

embeddings:
  use_local: true  # Use local SBERT model
  batch_size: 32
```

### OCR Provider Options ⭐ NEW

**OCR Provider** (`ocr.provider`):
- `azure` - **Azure Computer Vision (Recommended for cloud)** ⭐ NEW
  - High accuracy for handwriting and printed text
  - Fast processing (~2-5 seconds)
  - Requires Azure credentials in `.env`
  
- `local_trocr` - Local TrOCR model (offline)
  - Good accuracy for handwriting
  - No API costs
  - Slower on CPU (~500ms), faster on GPU (~150ms)
  
- `tesseract` - Tesseract OCR (fallback)
  - Basic OCR capability
  - Fast but lower accuracy for handwriting
  
- `auto` - Intelligent fallback chain
  - Tries: `local_trocr` → `azure` → `tesseract`
  - Best of both worlds (offline + cloud backup)

**Embeddings** (`embeddings.use_local`):
- `true` - Use local SBERT model (recommended)
- `false` - Download from Hugging Face on first use

### GPU Acceleration (Optional)

If you have an NVIDIA GPU with CUDA installed:

```yaml
models:
  device: "cuda"  # Use GPU for faster inference
```

Performance improvement with GPU:
- TrOCR: 3-5x faster
- SBERT: 2-3x faster

### Model Health Check

Check model status via API:

```bash
curl http://localhost:8000/api/models/health
```

Response includes:
- Model loading status
- Device information (CPU/CUDA)
- Model versions
- Memory usage

### Re-downloading Models

If models are missing or corrupted:

```bash
# TrOCR
git clone https://huggingface.co/microsoft/trocr-small-handwritten models/trocr-small-handwritten

# SBERT
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2
```

Or use Python:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer

# Download TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
processor.save_pretrained("models/trocr-small-handwritten")
model.save_pretrained("models/trocr-small-handwritten")

# Download SBERT
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sbert.save("models/all-MiniLM-L6-v2")
```

### Testing OCR and Local Models ⭐ UPDATED

Run test suites to verify model functionality:

```bash
# Test Azure OCR (requires credentials in .env) ⭐ NEW
pytest tests/test_azure_ocr.py -v

# Test Azure OCR via API endpoint ⭐ NEW
curl -X POST http://localhost:8000/api/ocr/test-azure \
  -F "file=@path/to/image.jpg"

# Test TrOCR
python tests/test_trocr_loader.py

# Test SBERT
python tests/test_sbert_loader.py

# Test full pipeline
python tests/test_ocr_and_embed_pipeline.py

# Run demos
python tests/test_trocr_loader.py demo
python tests/test_sbert_loader.py demo
python tests/test_ocr_and_embed_pipeline.py benchmark
```

### Performance Benchmarks ⭐ UPDATED

**OCR Performance:**
- **Azure Computer Vision**: ~2-5 seconds (cloud API, network dependent) ⭐ NEW
- **TrOCR (CPU)**: ~500ms per image
- **TrOCR (GPU)**: ~150ms per image
- **Tesseract**: ~100ms per image (lower accuracy)

**Embeddings Performance:**
- **SBERT (CPU)**: ~20ms per text
- **SBERT (GPU)**: ~10ms per text

**Total Pipeline:**
- **With Azure OCR**: ~3-6 seconds per evaluation
- **With Local TrOCR (CPU)**: ~600ms per evaluation
- **With Local TrOCR (GPU)**: ~200ms per evaluation
- **Total Pipeline**: ~200ms per evaluation

---

## 📖 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoint

**POST** `/api/evaluate`

Evaluate an answer sheet image.

**Request** (multipart/form-data):
- `question` (string): The exam question
- `reference_answer` (string): Model answer for comparison
- `answer_image` (file): Student's answer sheet (JPG/PNG/PDF)

**Response**:
```json
{
  "question_id": "Q-abc123",
  "student_raw_text": "Extracted text from OCR...",
  "cleaned_text": "Preprocessed text...",
  "similarity_score": 0.82,
  "logic_flow_score": 0.75,
  "contradiction_score": 0.90,
  "final_score": 0.81,
  "relevance_flag": "relevant",
  "details": {
    "contradictions_detected": 1,
    "text_length": 250,
    "word_count": 45
  }
}
```

---

### ⭐ NEW: Model Training Endpoints

**POST** `/api/train-model`

Train ML scoring model on ASAP-SAS dataset.

**Response**:
```json
{
  "status": "success",
  "message": "Model trained successfully! Validation QWK: 0.7821",
  "metrics": {
    "validation": {
      "rmse": 0.452,
      "mae": 0.328,
      "qwk": 0.782
    }
  },
  "model_path": "models/scoring_model.pkl"
}
```

**GET** `/api/model-info`

Get information about trained ML model.

**Response**:
```json
{
  "model_exists": true,
  "model_path": "models/scoring_model.pkl",
  "model_size_mb": 1.23,
  "metrics": {
    "rmse": 0.452,
    "mae": 0.328,
    "qwk": 0.782
  }
}
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Scoring Weights
```yaml
scoring:
  weights:
    similarity: 0.5      # Semantic similarity
    logic_flow: 0.3      # Coherence
    contradiction: 0.2   # Factual accuracy
  # Note: If ML model is trained, these are only used as fallback
```

### Models
```yaml
embeddings:
  model_name: "all-MiniLM-L6-v2"  # Fast, lightweight
  # Or: "all-mpnet-base-v2" for better accuracy

nli:
  model_name: "facebook/bart-large-mnli"
  # Or: "roberta-large-mnli"
```

### ML Training Configuration ⭐ NEW

Edit `config_training.yaml`:

```yaml
training:
  model_type: "regression"
  algorithm: "xgboost"  # or "random_forest"
  test_split: 0.15

features:
  use_sbert_embedding: true
  use_cosine_similarity: true
  use_logic_flow: true
  use_contradiction: true

output:
  model_path: "models/scoring_model.pkl"
```

### Thresholds
```yaml
scoring:
  thresholds:
    relevance: 0.5        # Min similarity for "relevant"
    contradiction: 0.7    # Min score for no contradiction
    pass_score: 0.6       # Passing threshold
```

## 🧪 Testing

Run tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_scoring.py -v
```

## 📁 Project Structure

```
backend/
├── main.py                       # FastAPI app entry point
├── api_routes.py                 # API endpoints
├── config.yaml                   # Runtime configuration
├── config_training.yaml          # ⭐ ML training config
├── requirements.txt              # Dependencies
├── .env.example                  # Environment template
├── TRAINING_GUIDE.md             # ⭐ Training documentation
│
├── modules/                      # Processing modules
│   ├── azure_ocr.py              # ⭐ Azure Computer Vision OCR
│   ├── image_preprocessing.py
│   ├── ocr_module.py             # Unified OCR interface (Azure/TrOCR/Tesseract)
│   ├── text_preprocessing.py
│   ├── embeddings.py
│   ├── relevance_checker.py
│   ├── semantic_analysis.py
│   ├── nli_contradiction.py
│   └── scoring.py                # ⭐ Now supports ML models
│
├── model_training/               # ⭐ NEW: ML training pipeline
│   ├── data_loader.py           # Load ASAP-SAS dataset
│   ├── feature_engineering.py   # Extract features
│   ├── train_regression_model.py # Train XGBoost/RF
│   ├── evaluate_model.py        # QWK, RMSE, MAE metrics
│   └── save_load_model.py       # Model persistence
│
├── models/                       # ⭐ Trained models (created after training)
│   ├── scoring_model.pkl
│   ├── feature_scaler.pkl
│   └── training_metrics.json
│
├── datasets/asap/                # ASAP-SAS dataset
│   ├── train.tsv                # Main training data
│   ├── train_rel_2.tsv          # Additional training data
│   └── test.csv                 # Test data
│
├── tests/                        # Unit tests
│   ├── test_text_preprocessing.py
│   ├── test_relevance_checker.py
│   └── test_scoring.py
│
└── demo_data/                    # Sample files
```

## 🎓 ML Model Training ⭐ NEW

### Quick Start

1. **Verify setup**:
```bash
python check_training_setup.py
```

2. **Train model**:
```bash
# Option 1: Direct Python
python -m model_training.train_regression_model

# Option 2: Via API
curl -X POST http://localhost:8000/api/train-model
```

3. **Check results**:
```bash
curl http://localhost:8000/api/model-info
```

### What Gets Trained?

- **Algorithm**: XGBoost regression (or Random Forest)
- **Features**: SBERT embeddings, similarity scores, logic flow, contradictions
- **Dataset**: ASAP-SAS (4000+ samples)
- **Metric**: Quadratic Weighted Kappa (QWK) - target > 0.6
- **Output**: `models/scoring_model.pkl`

### Integration

Once trained, the model is **automatically used** by the `/api/evaluate` endpoint.
Falls back to weighted scoring if model not found.

📖 **Full Guide**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

## 🔧 Troubleshooting

### Azure OCR Issues ⭐ NEW

**Error**: `Missing Azure credentials`
- Verify `.env` has `AZURE_OCR_ENDPOINT`, `AZURE_OCR_KEY`, `AZURE_OCR_REGION`
- Get credentials from Azure Portal → Computer Vision → Keys and Endpoint

**Error**: `Operation timed out`
- Increase `azure_timeout` in `config.yaml` (default: 30s)
- Check network connectivity to Azure
- Verify Azure service region is correct

**Low confidence scores**:
- Ensure image is clear and well-lit
- Try image preprocessing: `image_preprocessing.enhance_image()`
- For handwritten text, consider switching to `local_trocr`

**Fallback**: If Azure unavailable, system uses local TrOCR or Tesseract
- Install Tesseract: https://github.com/tesseract-ocr/tesseract

### Model Download Issues

**Error**: `spaCy model not found`
```bash
python -m spacy download en_core_web_sm
```

**Error**: SBERT model download fails
- Check internet connection
- Models auto-download on first use
- Cached in `~/.cache/huggingface/`

### ML Training Issues ⭐

**Error**: `Dataset files not found`
- Ensure files in `datasets/asap/` directory
- Files: train.tsv, train_rel_2.tsv, test.csv

**Error**: `XGBoost not installed`
```bash
pip install xgboost pandas
```

**Low QWK score** (< 0.5)
- Check dataset quality
- Verify feature extraction working
- Tune hyperparameters in `config_training.yaml`

### Azure OCR Issues ⭐ NEW

**"Missing Azure credentials"**:
- Verify `.env` file has `AZURE_OCR_ENDPOINT`, `AZURE_OCR_KEY`, `AZURE_OCR_REGION`
- Check credentials in Azure Portal → Computer Vision → Keys and Endpoint

**"Operation timed out"**:
- Increase `azure_timeout` in `config.yaml` (default: 30 seconds)
- Check network connectivity to Azure
- Verify Azure service is running in your region

**Low confidence scores**:
- Ensure image is clear and well-lit
- Try preprocessing: `image_preprocessing.enhance_image()`
- Switch to `local_trocr` for handwritten text

**API quota exceeded**:
- Check Azure usage in portal
- Upgrade to higher pricing tier
- Switch to `local_trocr` for development

### CORS Issues

Frontend can't connect to backend:
- Update `FRONTEND_URL` in `.env`
- Check `api_routes.py` CORS settings

## 🚀 Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Environment Variables (Production)
```bash
DEBUG_MODE=False
LOG_LEVEL=WARNING
API_HOST=0.0.0.0
API_PORT=8000
```

## 📊 Performance ⭐ UPDATED

**Evaluation Times:**
- **With Azure OCR + ML model**: 3-6 seconds ⭐ NEW
- **With Local TrOCR + ML model**: 1-2 seconds
- **Without ML model**: 3-5 seconds (rule-based scoring)

**Resource Usage:**
- **Model memory**: ~2-3GB RAM (SBERT + NLI + XGBoost loaded)
- **ML Model size**: ~1-2MB (very efficient!)
- **Azure OCR**: Network bandwidth dependent

### Optimization Tips ⭐ UPDATED
1. **For offline/low-latency**: Use `local_trocr` provider (500ms OCR)
2. **For best accuracy**: Use `azure` provider (2-5s OCR)
3. **For reliability**: Use `auto` provider (smart fallback)
4. Use GPU for faster NLI inference
5. Batch multiple evaluations
6. Cache embeddings for repeated questions
4. Use lighter models for high-volume scenarios
5. Train custom model on domain-specific data

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

## 📄 License

MIT License

## 🙋 Support

For issues or questions:
- Create GitHub issue
- Check documentation: `/docs`
- Review logs: `logs/app.log`

## 🔄 API Integration

### Frontend Integration

```typescript
const evaluateAnswer = async (formData: FormData) => {
  const response = await fetch('http://localhost:8000/api/evaluate', {
    method: 'POST',
    body: formData
  });
  return await response.json();
};
```

### Python Client Example

```python
import requests

files = {'answer_image': open('answer.jpg', 'rb')}
data = {
    'question': 'What is photosynthesis?',
    'reference_answer': 'Photosynthesis is...'
}

response = requests.post(
    'http://localhost:8000/api/evaluate',
    files=files,
    data=data
)

result = response.json()
print(f"Final Score: {result['final_score']}")
```

## 📈 Roadmap

- [ ] Support for multiple languages
- [ ] PDF multi-page support
- [ ] Batch evaluation endpoint
- [ ] Historical analytics dashboard
- [ ] Model fine-tuning capabilities
- [ ] Real-time evaluation streaming

---

**Built with ❤️ using FastAPI, SBERT, and Transformers**
