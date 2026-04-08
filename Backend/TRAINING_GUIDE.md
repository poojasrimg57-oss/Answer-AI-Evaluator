# AnswerAI Model Training Guide

## Overview

This guide explains how to train and integrate the ML scoring model with your AnswerAI backend.

## 📁 Dataset Setup

You have already placed the ASAP-SAS dataset files in:
```
Backend/datasets/asap/
    ├── train.tsv
    ├── train_rel_2.tsv
    └── test.csv
```

✅ Dataset ready!

## 🚀 Quick Start Training

### Option 1: Via API Endpoint (Recommended)

1. Start the backend:
```bash
cd Backend
python main.py
```

2. Call the training endpoint:
```bash
curl -X POST http://localhost:8000/api/train-model
```

Or use Postman/frontend to trigger training.

### Option 2: Direct Python Script

```bash
cd Backend
python -m model_training.train_regression_model
```

## 📊 Training Pipeline

The training pipeline automatically:

1. **Loads Dataset** - Reads and merges train.tsv and train_rel_2.tsv
2. **Extracts Features** - Uses your existing modules:
   - SBERT embeddings (384 dimensions)
   - Cosine similarity with reference
   - Logic flow score
   - Contradiction score
   - Keyword overlap
   - Text length features
   - Readability score

3. **Trains Model** - XGBoost regression model (default)
   - 200 estimators
   - Max depth: 6
   - Learning rate: 0.1

4. **Evaluates Performance**:
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² (R-squared)
   - **QWK (Quadratic Weighted Kappa)** ← Primary metric for ASAP-SAS

5. **Saves Artifacts**:
   - `models/scoring_model.pkl`
   - `models/feature_scaler.pkl`
   - `models/feature_names.json`
   - `models/training_metrics.json`

## 🎯 Expected Performance

Good QWK scores for ASAP-SAS:
- QWK > 0.6: Good agreement
- QWK > 0.7: Strong agreement
- QWK > 0.8: Excellent agreement

## 🔄 Backend Integration

### Automatic Loading

The backend automatically loads the trained model at startup:

```python
# In scoring.py
final_score = calculate_final_score(
    similarity_score,
    logic_flow_score,
    contradiction_score,
    features=features,  # Full feature dict
    use_ml_model=True   # Enable ML model
)
```

### Fallback Behavior

If no model is found, the system uses **weighted scoring** as fallback:
```
Final Score = 0.5×similarity + 0.3×logic_flow + 0.2×contradiction
```

## 📝 Configuration

Edit `config_training.yaml` to customize:

```yaml
training:
  model_type: "regression"
  algorithm: "xgboost"  # or "random_forest"
  test_split: 0.15
  random_seed: 42

features:
  use_sbert_embedding: true
  use_cosine_similarity: true
  use_logic_flow: true
  use_contradiction: true
  # ... more features
```

## 🔍 Check Model Status

```bash
curl http://localhost:8000/api/model-info
```

Response:
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

## 🧪 Testing the Model

After training, test evaluation with ML model:

```bash
curl -X POST http://localhost:8000/api/evaluate \
  -F "question=Explain photosynthesis" \
  -F "reference_answer=Photosynthesis is..." \
  -F "answer_image=@sample_answer.jpg"
```

The response will use the ML model prediction if available.

## 📊 Feature Importance

After training completes, check logs for top features:

```
Top 10 Most Important Features:
1. cosine_similarity: 0.3421
2. logic_flow_score: 0.2134
3. emb_0: 0.0512
4. contradiction_score: 0.0498
...
```

## 🔧 Troubleshooting

### Dataset Not Found
```
Error: ASAP-SAS dataset not found
```
**Solution**: Ensure files are in `Backend/datasets/asap/`

### Model Loading Failed
```
Warning: Failed to load ML model
```
**Solution**: Train model first using `/api/train-model`

### Low QWK Score
```
Validation QWK: 0.35
```
**Solution**: 
- Check dataset quality
- Increase training samples
- Tune hyperparameters in `config_training.yaml`

## 🚀 Production Deployment

1. **Train model** on full dataset
2. **Verify metrics** (QWK > 0.6)
3. **Commit model files** to repository (or use model registry)
4. **Deploy backend** with model files included

## 📚 Advanced Options

### Fine-tune SBERT (Optional)

Placeholder file exists: `train_sbert_finetune.py`

Future enhancement: Fine-tune Sentence-BERT directly for scoring.

### Train Transformer (Optional)

Placeholder file exists: `train_transformer_model.py`

Future enhancement: Train RoBERTa/DeBERTa for direct score prediction.

## 🎓 Understanding the Pipeline

```
Student Answer Image
        ↓
    [OCR + Preprocessing]
        ↓
    [Feature Extraction]
        ├─ SBERT embedding (384-dim)
        ├─ Cosine similarity
        ├─ Logic flow score
        ├─ Contradiction score
        ├─ Text statistics
        └─ Readability
        ↓
    [ML Model]
        ↓
    Predicted Score (0-1)
```

## 📞 Support

For issues or questions:
1. Check logs in `Backend/logs/`
2. Verify dataset files exist
3. Ensure all dependencies installed: `pip install -r requirements.txt`

---

**Ready to train?** Run: `python -m model_training.train_regression_model`
