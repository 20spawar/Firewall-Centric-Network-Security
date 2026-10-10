from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INSIDER_OUTPUT = (
    PROJECT_ROOT
    / "insider_module"
    / "outputs"
    / "insider_scores.csv"
)

ANOMALY_OUTPUT = (
    PROJECT_ROOT
    / "anomaly_module"
    / "outputs"
    / "anomaly_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "fusion"
    / "outputs"
)

FUSION_OUTPUT = (
    OUTPUT_DIR
    / "fusion_results.csv"
)


# ============================================================
# FUSION WEIGHTS
# ============================================================

INSIDER_WEIGHT = 0.50
ANOMALY_WEIGHT = 0.50


# ============================================================
# DECISION THRESHOLDS
# ============================================================

LOW_RISK_THRESHOLD = 30
MEDIUM_RISK_THRESHOLD = 60
HIGH_RISK_THRESHOLD = 80