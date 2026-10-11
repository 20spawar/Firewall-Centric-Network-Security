from pathlib import Path

import joblib
import pandas as pd
import shap
import numpy as np

from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "anomaly_model.joblib"
)

ANOMALY_OUTPUT = (
    PROJECT_ROOT
    / "anomaly_module"
    / "outputs"
    / "anomaly_scores.csv"
)

FEATURE_OUTPUT = (
    PROJECT_ROOT
    / "anomaly_module"
    / "outputs"
    / "http_user_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "xai"
    / "outputs"
)

SHAP_OUTPUT = (
    OUTPUT_DIR
    / "anomaly_shap_values.csv"
)

LIME_OUTPUT = (
    OUTPUT_DIR
    / "anomaly_lime_explanations.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_anomaly_model():

    if not MODEL_PATH.exists():
        print("Anomaly model not found.")
        print(f"Expected model path: {MODEL_PATH}")
        return None

    package = joblib.load(MODEL_PATH)

    return package


# ============================================================
# ANOMALY XAI
# ============================================================

def explain_anomaly():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    package = load_anomaly_model()

    if package is None:
        return

    # --------------------------------------------------------
    # Check feature file
    # --------------------------------------------------------

    if not FEATURE_OUTPUT.exists():
        print("Anomaly feature file not found.")
        print(f"Expected feature file: {FEATURE_OUTPUT}")
        return

    feature_df = pd.read_csv(
        FEATURE_OUTPUT
    )

    if feature_df.empty:
        print("Anomaly feature file is empty.")
        return

    # --------------------------------------------------------
    # Load anomaly scores
    # --------------------------------------------------------

    if not ANOMALY_OUTPUT.exists():
        print("Anomaly score file not found.")
        print(f"Expected output: {ANOMALY_OUTPUT}")
        return

    anomaly_df = pd.read_csv(
        ANOMALY_OUTPUT
    )

    # --------------------------------------------------------
    # Extract saved model components
    # --------------------------------------------------------

    imputer = package["imputer"]
    scaler = package["scaler"]
    model = package["model"]
    feature_columns = package["feature_columns"]

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in feature_df.columns
    ]

    if missing_features:
        print(
            "Required anomaly features are missing "
            "from the feature file."
        )
        print(
            f"Missing features: {missing_features}"
        )
        return

    # --------------------------------------------------------
    # Prepare model features
    # --------------------------------------------------------

    X_raw = feature_df[
        feature_columns
    ].copy()

    X = imputer.transform(
        X_raw
    )

    X = scaler.transform(
        X
    )

    # ========================================================
    # SHAP
    # ========================================================

    try:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            X
        )

    except Exception as error:

        print(
            "Unable to generate SHAP explanation."
        )

        print(
            f"Reason: {error}"
        )

        return

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_columns
    )

    if "user" in feature_df.columns:

        shap_df.insert(
            0,
            "user",
            feature_df["user"].values
        )

    shap_df.to_csv(
        SHAP_OUTPUT,
        index=False
    )

    print(
        "SHAP explanation generated successfully."
    )

    print(
        f"Saved to: {SHAP_OUTPUT}"
    )

    # ========================================================
    # LIME
    # ========================================================

    try:

        def predict_for_lime(data):

            decision_scores = model.decision_function(
                data
            )

            anomaly_probability = 1 / (
                1 + np.exp(decision_scores)
            )

            normal_probability = (
                1 - anomaly_probability
            )

            return np.column_stack(
                [
                    normal_probability,
                    anomaly_probability
                ]
            )

        lime_explainer = LimeTabularExplainer(
            X,
            feature_names=feature_columns,
            class_names=[
                "Normal",
                "Anomaly"
            ],
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )

        lime_rows = []

        for index in range(len(X)):

            explanation = lime_explainer.explain_instance(
                X[index],
                predict_for_lime,
                num_features=len(feature_columns)
            )

            explanation_list = explanation.as_list()

            user_id = (
                feature_df.iloc[index]["user"]
                if "user" in feature_df.columns
                else index
            )

            for rank, (feature_rule, contribution) in enumerate(
                explanation_list,
                start=1
            ):

                lime_rows.append(
                    {
                        "user": user_id,
                        "rank": rank,
                        "feature_rule": feature_rule,
                        "contribution": contribution
                    }
                )

        lime_df = pd.DataFrame(
            lime_rows
        )

        lime_df.to_csv(
            LIME_OUTPUT,
            index=False
        )

        print(
            "LIME explanation generated successfully."
        )

        print(
            f"Saved to: {LIME_OUTPUT}"
        )

        print(
            f"Users explained: {len(feature_df)}"
        )

        print(
            f"Explanation rows: {len(lime_df)}"
        )

    except Exception as error:

        print(
            "Unable to generate LIME explanation."
        )

        print(
            f"Reason: {error}"
        )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    explain_anomaly()