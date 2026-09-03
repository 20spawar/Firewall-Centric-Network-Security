from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "anomaly_module" / "outputs"

MODEL_PATH = MODEL_DIR / "anomaly_model.joblib"
FEATURE_OUTPUT_PATH = OUTPUT_DIR / "http_user_features.csv"
ANOMALY_OUTPUT_PATH = OUTPUT_DIR / "anomaly_scores.csv"

# ============================================================
# CERT HTTP DATASET
# ============================================================

HTTP_DATA_PATH = Path(
    r"D:\MAJOR PROJECT DATA\cert r4.2\r4.2\r4.2\http_sample.csv"
)

# Read the huge HTTP file in chunks.
CHUNK_SIZE = 100_000

# ============================================================
# ISOLATION FOREST
# ============================================================

N_ESTIMATORS = 200
CONTAMINATION = "auto"
RANDOM_STATE = 42

# ============================================================
# CERT HTTP COLUMNS
# ============================================================

ID_COLUMN = "id"
DATE_COLUMN = "date"
USER_COLUMN = "user"
PC_COLUMN = "pc"
URL_COLUMN = "url"
CONTENT_COLUMN = "content"