# AnswerAI Evaluator - Complete Setup Instructions

Comprehensive guide for setting up the AnswerAI Evaluator backend with local AI models.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Environment Setup](#python-environment-setup)
3. [Dependencies Installation](#dependencies-installation)
4. [Local Models Setup](#local-models-setup)
5. [Configuration](#configuration)
6. [Running the Application](#running-the-application)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## 1. System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.10 or 3.11 (3.13 not yet supported by all ML libraries)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 5 GB free space for models and dependencies
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)

### Optional (for GPU acceleration)

- **GPU**: NVIDIA GPU with CUDA 11.8+ support
- **VRAM**: 4 GB minimum (for local models)
- **CUDA Toolkit**: 11.8 or 12.1
- **cuDNN**: Compatible version with CUDA

### Internet Connection

- Required for initial setup (downloading dependencies and models)
- Not required for runtime after setup (fully offline capable)

---

## 2. Python Environment Setup

### Step 2.1: Verify Python Version

```bash
# Check Python version
python --version

# Should show Python 3.10.x or 3.11.x
# If you have Python 3.13, use Python 3.10 instead:
# Download from: https://www.python.org/downloads/
```

**Important**: Python 3.13 is not yet fully supported by NumPy and some ML libraries. Use Python 3.10 or 3.11.

### Step 2.2: Navigate to Backend Directory

```bash
cd "C:\Users\DEEPAK BUSA\Downloads\AnswerAI Evaluator\Backend"
# Or your actual path
```

### Step 2.3: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 2.4: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 3. Dependencies Installation

### Step 3.1: Install Core Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **FastAPI** (0.115.5) - Web framework
- **PyTorch** (2.5.1) - Deep learning framework (~200MB)
- **Transformers** (4.47.0) - Hugging Face transformers
- **Sentence-Transformers** (3.3.1) - SBERT embeddings
- **XGBoost** (2.1.3) - ML training
- **spaCy** (3.8.2) - NLP
- **And 20+ other packages**

**Expected time**: 5-10 minutes depending on internet speed.

### Step 3.2: Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### Step 3.3: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 3.4: Verify Installation

```bash
python -c "import torch; import transformers; import sentence_transformers; print('✓ All imports successful!')"
```

---

## 4. Local Models Setup

### Option A: Models Already Present (Recommended)

If models are already in `models/` directory:

```bash
# Verify models exist
ls models/
# Should show:
#   trocr-small-handwritten/
#   all-MiniLM-L6-v2/
```

**Skip to Section 5 (Configuration)** if models are present.

### Option B: Download Models (If Missing)

#### Method 1: Using Git LFS (Recommended)

Install Git LFS first:
- **Windows**: Download from https://git-lfs.github.com/
- **Linux**: `sudo apt install git-lfs`
- **Mac**: `brew install git-lfs`

```bash
# Initialize Git LFS
git lfs install

# Download TrOCR model (~200MB)
git clone https://huggingface.co/microsoft/trocr-small-handwritten models/trocr-small-handwritten

# Download SBERT model (~90MB)
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2
```

#### Method 2: Using Python Script

Create `download_models.py`:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer
import os

os.makedirs("models", exist_ok=True)

print("Downloading TrOCR model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
processor.save_pretrained("models/trocr-small-handwritten")
model.save_pretrained("models/trocr-small-handwritten")
print("✓ TrOCR downloaded")

print("Downloading SBERT model...")
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sbert.save("models/all-MiniLM-L6-v2")
print("✓ SBERT downloaded")

print("\n✓ All models downloaded successfully!")
```

Run:
```bash
python download_models.py
```

### Step 4.1: Verify Model Files

**TrOCR** (`models/trocr-small-handwritten/`) should contain:
- `config.json`
- `pytorch_model.bin` (~140MB)
- `preprocessor_config.json`
- `generation_config.json`
- `tokenizer_config.json`
- `sentencepiece.bpe.model`

**SBERT** (`models/all-MiniLM-L6-v2/`) should contain:
- `config.json`
- `pytorch_model.bin` (~90MB)
- `tokenizer.json`
- `vocab.txt`

---

## 5. Configuration

### Step 5.1: Copy Environment File

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

### Step 5.2: Edit .env File

Open `.env` in a text editor and set:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=True
FRONTEND_URL=http://localhost:5173

# Google Cloud Vision API (optional - only needed for fallback)
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json

# Logging
LOG_LEVEL=INFO
```

**Note**: Google Cloud Vision API is optional if using local TrOCR model.

### Step 5.3: Configure config.yaml

The `config.yaml` file is already configured for local models:

```yaml
models:
  trocr_local_path: "models/trocr-small-handwritten"
  sbert_local_path: "models/all-MiniLM-L6-v2"
  device: "cpu"  # Change to "cuda" if you have GPU
  warmup_on_startup: true

ocr:
  provider: "local_trocr"  # Use local model by default

embeddings:
  use_local: true  # Use local SBERT model
```

**For GPU acceleration**, change:
```yaml
models:
  device: "cuda"
```

### Step 5.4: Test Configuration

```bash
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
```

Should print configuration without errors.

---

## 6. Running the Application

### Step 6.1: Start the Server

```bash
python main.py
```

You should see:
```
INFO:     Starting AnswerAI Evaluator API on 0.0.0.0:8000
INFO:     Starting model warmup...
INFO:     Loading TrOCR model from: models/trocr-small-handwritten on device: cpu
INFO:     ✓ TrOCR model loaded successfully
INFO:     Loading SBERT model from: models/all-MiniLM-L6-v2 on device: cpu
INFO:     ✓ SBERT model loaded successfully
INFO:     Model warmup completed successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6.2: Verify API is Running

Open browser and navigate to:
- **API Root**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model Health**: http://localhost:8000/api/models/health

### Step 6.3: Test Model Health Endpoint

```bash
curl http://localhost:8000/api/models/health
```

Expected response:
```json
{
  "status": "healthy",
  "models": {
    "trocr": {
      "status": "loaded",
      "device": "cpu",
      "model_name": "microsoft/trocr-small-handwritten"
    },
    "sbert": {
      "status": "loaded",
      "device": "cpu",
      "embedding_dimension": 384
    }
  },
  "system": {
    "cuda_available": false,
    "configured_device": "cpu"
  }
}
```

---

## 7. Testing

### Step 7.1: Run Unit Tests

```bash
# Test TrOCR loader
python tests/test_trocr_loader.py

# Test SBERT loader
python tests/test_sbert_loader.py

# Test full pipeline
python tests/test_ocr_and_embed_pipeline.py
```

### Step 7.2: Run Demos

```bash
# TrOCR demo
python tests/test_trocr_loader.py demo

# SBERT semantic search demo
python tests/test_sbert_loader.py demo

# Performance benchmark
python tests/test_ocr_and_embed_pipeline.py benchmark
```

### Step 7.3: Test API Endpoint

Create `test_api.py`:

```python
import requests
import json

url = "http://localhost:8000/api/evaluate"

# Test data
data = {
    "question": "What is photosynthesis?",
    "reference_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy"
}

# If testing with image:
# files = {"answer_image": open("test_image.jpg", "rb")}
# response = requests.post(url, data=data, files=files)

# For text-only test:
response = requests.post(url, json=data)

print(json.dumps(response.json(), indent=2))
```

Run:
```bash
python test_api.py
```

---

## 8. Troubleshooting

### Issue 1: Python Version Error

**Error**: `ERROR: Could not find a version that satisfies the requirement numpy`

**Solution**: You're using Python 3.13. Switch to Python 3.10:
```bash
# Download Python 3.10 from python.org
# Create new venv with Python 3.10
python310 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Issue 2: Model Not Found

**Error**: `FileNotFoundError: TrOCR model not found at models/trocr-small-handwritten`

**Solution**: Download models (see Section 4).

### Issue 3: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Switch to CPU in `config.yaml`:
```yaml
models:
  device: "cpu"
```

### Issue 4: Import Error

**Error**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
# Activate venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 5: Port Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**: Change port in `.env`:
```env
API_PORT=8001
```

### Issue 6: Slow Performance

**Symptoms**: OCR takes >3 seconds per image

**Solutions**:
1. Use GPU acceleration (if available)
2. Reduce batch size in `config.yaml`
3. Disable warmup: `warmup_on_startup: false`

### Issue 7: Vision API Errors (Optional)

If you want to use Vision API fallback:

**Error**: `google.auth.exceptions.DefaultCredentialsError`

**Solution**: Set up Google Cloud credentials:
1. Create service account in Google Cloud Console
2. Download JSON key file
3. Set environment variable:
```bash
# Windows
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\key.json

# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

---

## Next Steps

1. ✅ **Test the API**: Use Swagger UI at http://localhost:8000/docs
2. ✅ **Run Training**: Try `/api/train-model` endpoint to train ML scoring model
3. ✅ **Integrate Frontend**: Connect React frontend from `../Frontend`
4. ✅ **Production Deployment**: See deployment guide for cloud hosting

---

## Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Model Health**: http://localhost:8000/api/models/health
- **TrOCR Model**: https://huggingface.co/microsoft/trocr-small-handwritten
- **SBERT Model**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Training Guide**: See `TRAINING_GUIDE.md` for ML model training

---

## Support

For issues or questions:
1. Check logs in `logs/app.log`
2. Verify model files exist and are complete
3. Test with provided test scripts
4. Check API health endpoints

---

**Congratulations!** 🎉 Your AnswerAI Evaluator backend with local AI models is now fully set up!
