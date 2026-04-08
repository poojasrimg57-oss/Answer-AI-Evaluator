"""
Test script to verify model training pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("AnswerAI Model Training - System Check")
print("=" * 70)

# Check 1: Dataset files
print("\n[1/5] Checking dataset files...")
dataset_dir = Path("datasets/asap")
required_files = ["train.tsv", "train_rel_2.tsv", "test.csv"]

for file in required_files:
    file_path = dataset_dir / file
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {file} found ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {file} NOT FOUND")

# Check 2: Import dependencies
print("\n[2/5] Checking dependencies...")
try:
    import pandas
    print(f"  ✓ pandas {pandas.__version__}")
except ImportError:
    print("  ✗ pandas not installed")

try:
    import xgboost
    print(f"  ✓ xgboost {xgboost.__version__}")
except ImportError:
    print("  ✗ xgboost not installed (pip install xgboost)")

try:
    import sklearn
    print(f"  ✓ scikit-learn {sklearn.__version__}")
except ImportError:
    print("  ✗ scikit-learn not installed")

# Check 3: Backend modules
print("\n[3/5] Checking backend modules...")
try:
    from modules.embeddings import generate_embeddings
    print("  ✓ embeddings module")
except ImportError as e:
    print(f"  ✗ embeddings module: {str(e)}")

try:
    from modules.semantic_analysis import analyze_logic_flow
    print("  ✓ semantic_analysis module")
except ImportError as e:
    print(f"  ✗ semantic_analysis module: {str(e)}")

try:
    from modules.nli_contradiction import detect_contradictions
    print("  ✓ nli_contradiction module")
except ImportError as e:
    print(f"  ✗ nli_contradiction module: {str(e)}")

# Check 4: Configuration
print("\n[4/5] Checking configuration...")
config_path = Path("config_training.yaml")
if config_path.exists():
    print(f"  ✓ config_training.yaml found")
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"  ✓ Algorithm: {config['training']['algorithm']}")
    print(f"  ✓ Test split: {config['training']['test_split']}")
else:
    print("  ✗ config_training.yaml NOT FOUND")

# Check 5: Output directory
print("\n[5/5] Checking output directory...")
models_dir = Path("models")
if models_dir.exists():
    print(f"  ✓ models/ directory exists")
    
    # Check for existing model
    model_path = models_dir / "scoring_model.pkl"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ℹ Trained model found: scoring_model.pkl ({size_mb:.2f} MB)")
    else:
        print("  ℹ No trained model found (will be created during training)")
else:
    print("  ✗ models/ directory not found")

print("\n" + "=" * 70)
print("System Check Complete!")
print("=" * 70)
print("\nTo start training, run:")
print("  python -m model_training.train_regression_model")
print("\nOr via API:")
print("  curl -X POST http://localhost:8000/api/train-model")
