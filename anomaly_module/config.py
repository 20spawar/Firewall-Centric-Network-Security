from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data locations
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "samples"

# Model output
MODEL_DIR = PROJECT_ROOT / "models"
ANOMALY_MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"

# Random seed for reproducibility
RANDOM_STATE = 42

# Isolation Forest settings
CONTAMINATION = "auto"
N_ESTIMATORS = 200

# Create required directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)