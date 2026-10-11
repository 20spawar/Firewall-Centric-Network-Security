from pathlib import Path

import numpy as np
import pandas as pd
import shap
import lime.lime_tabular

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = (
    PROJECT_ROOT
    / "insider_module"
    / "outputs"
    / "user_features.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "insider_module"
    / "outputs"
    / "user_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "xai" / "outputs"

SHAP_OUTPUT = OUTPUT_DIR / "insider_shap_values.csv"
LIME_OUTPUT = OUTPUT_DIR / "insider_lime_explanations.csv"


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    "logon_count",
    "logon_unique_hours",
    "logon_after_hours",
    "logon_weekend",
    "device_connect_count",
    "usb_connects",
    "usb_disconnects",
    "file_operations_count",
    "unique_files_accessed",
    "file_exe_access",
    "file_doc_access",
    "total_emails",
    "emails_with_attachment",
    "unique_recipients",
    "avg_email_size",
]


# ============================================================
# LOAD DATA
# ============================================================

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Missing user_features.csv:\n{FEATURES_PATH}"
    )

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Missing user_predictions.csv:\n{PREDICTIONS_PATH}"
    )

features_df = pd.read_csv(FEATURES_PATH)
predictions_df = pd.read_csv(PREDICTIONS_PATH)


# ============================================================
# VALIDATION
# ============================================================

missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in features_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required feature columns: {missing_features}"
    )

if "user" not in features_df.columns:
    raise ValueError("user column missing from user_features.csv")

if "is_malicious" not in features_df.columns:
    raise ValueError(
        "is_malicious column missing from user_features.csv"
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

X = features_df[FEATURE_COLUMNS].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# RECONSTRUCT INSIDER ISOLATION FOREST
# ============================================================

malicious_users = features_df["is_malicious"].sum()
total_users = len(features_df)

contamination = malicious_users / total_users

iso_forest = IsolationForest(
    contamination=contamination,
    random_state=42,
    n_estimators=100
)

iso_forest.fit(X_scaled)


# ============================================================
# SHAP EXPLANATION
# ============================================================

print("Generating SHAP explanations...")

explainer = shap.TreeExplainer(iso_forest)

shap_values = explainer.shap_values(X_scaled)


# ============================================================
# HANDLE SHAP OUTPUT FORMAT
# ============================================================

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]


# ============================================================
# SAVE SHAP VALUES
# ============================================================

shap_df = pd.DataFrame(
    shap_values,
    columns=FEATURE_COLUMNS
)

shap_df.insert(
    0,
    "user",
    features_df["user"].values
)

shap_df.to_csv(
    SHAP_OUTPUT,
    index=False
)

print("SHAP explanation generated successfully.")
print(f"Saved to: {SHAP_OUTPUT}")


# ============================================================
# LIME EXPLANATION
# ============================================================

print("Generating LIME explanations...")

def anomaly_probability(X_input):
    """
    Convert Isolation Forest decision scores into
    a probability-like anomaly score for LIME.

    This is not a calibrated probability.
    """

    X_input = np.asarray(X_input)

    scores = iso_forest.decision_function(X_input)

    # Lower Isolation Forest decision score = more anomalous.
    # Convert it to an increasing anomaly score.
    anomaly_score = -scores

    # Sigmoid transformation for LIME probability format.
    probability = 1 / (1 + np.exp(-anomaly_score))

    return np.column_stack([
        1 - probability,
        probability
    ])


lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_scaled,
    feature_names=FEATURE_COLUMNS,
    class_names=["normal", "anomalous"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)


lime_rows = []

for index in range(len(X_scaled)):

    explanation = lime_explainer.explain_instance(
        X_scaled[index],
        anomaly_probability,
        num_features=len(FEATURE_COLUMNS)
    )

    explanation_list = explanation.as_list()

    for rank, (feature_rule, contribution) in enumerate(
        explanation_list,
        start=1
    ):

        lime_rows.append({
            "user": features_df.iloc[index]["user"],
            "rank": rank,
            "feature_rule": feature_rule,
            "contribution": contribution
        })


# ============================================================
# SAVE LIME OUTPUT
# ============================================================

lime_df = pd.DataFrame(lime_rows)

lime_df.to_csv(
    LIME_OUTPUT,
    index=False
)

print("LIME explanation generated successfully.")
print(f"Saved to: {LIME_OUTPUT}")

print()
print("============================================")
print("INSIDER XAI COMPLETED")
print("============================================")
print(f"Users explained: {len(features_df)}")
print(f"SHAP output: {SHAP_OUTPUT}")
print(f"LIME output: {LIME_OUTPUT}")