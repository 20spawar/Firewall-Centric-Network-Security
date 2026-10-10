"""
Insider Threat Detection - ML Pipeline

CERT r4.2
Behaviour Analysis using:
1. Isolation Forest - unknown/anomalous behaviour
2. SMOTEENN + XGBoost - known malicious behaviour
3. SHAP - explainable AI
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import shap

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    roc_curve
)

from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================================================
# PATHS
# =========================================================

SAMPLES_DIR = r"D:\MAJOR PROJECT DATA\CERT R4.2\r4.2\r4.2\samples"

OUTPUT_DIR = r"D:\MAJOR PROJECT DATA\CERT R4.2\r4.2\r4.2\ml_results"

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 70)
print("INSIDER THREAT DETECTION - XGBOOST + SMOTEENN + ISOLATION FOREST")
print("=" * 70)


# =========================================================
# STEP 1: LOAD DATA
# =========================================================

print("\n[STEP 1/8] Loading sampled CERT data...")

insiders_df = pd.read_csv(
    os.path.join(SAMPLES_DIR, "insiders_sample.csv")
)

logon_df = pd.read_csv(
    os.path.join(SAMPLES_DIR, "logon_sample.csv"),
    parse_dates=["date"]
)

device_df = pd.read_csv(
    os.path.join(SAMPLES_DIR, "device_sample.csv"),
    parse_dates=["date"]
)

file_df = pd.read_csv(
    os.path.join(SAMPLES_DIR, "file_sample.csv"),
    parse_dates=["date"]
)

email_df = pd.read_csv(
    os.path.join(SAMPLES_DIR, "email_sample.csv"),
    parse_dates=["date"]
)


print(f"✓ Insiders: {len(insiders_df)} users")
print(f"✓ Logon records: {len(logon_df)}")
print(f"✓ Device records: {len(device_df)}")
print(f"✓ File records: {len(file_df)}")
print(f"✓ Email records: {len(email_df)}")


# =========================================================
# MALICIOUS USERS
# =========================================================

malicious_users = set(
    insiders_df["user"].unique()
)

print(
    f"✓ Malicious users: {len(malicious_users)}"
)


# =========================================================
# STEP 2: FEATURE ENGINEERING
# =========================================================

print("\n[STEP 2/8] Feature Engineering...")

all_users = (
    set(logon_df["user"].unique())
    |
    set(device_df["user"].unique())
    |
    set(file_df["user"].unique())
    |
    set(email_df["user"].unique())
)

print(
    f"✓ Total unique users: {len(all_users)}"
)


features_list = []


for user in sorted(all_users):

    user_features = {
        "user": user
    }


    # -----------------------------------------------------
    # LOGON FEATURES
    # -----------------------------------------------------

    user_logon = logon_df[
        logon_df["user"] == user
    ]

    if len(user_logon) > 0:

        user_features["logon_count"] = len(
            user_logon
        )

        user_features["logon_unique_hours"] = (
            user_logon["date"]
            .dt.hour
            .nunique()
        )

        user_features["logon_after_hours"] = len(
            user_logon[
                (user_logon["date"].dt.hour < 7)
                |
                (user_logon["date"].dt.hour > 18)
            ]
        )

        user_features["logon_weekend"] = len(
            user_logon[
                user_logon["date"].dt.weekday >= 5
            ]
        )

    else:

        user_features["logon_count"] = 0
        user_features["logon_unique_hours"] = 0
        user_features["logon_after_hours"] = 0
        user_features["logon_weekend"] = 0


    # -----------------------------------------------------
    # DEVICE / USB FEATURES
    # -----------------------------------------------------

    user_device = device_df[
        device_df["user"] == user
    ]

    if len(user_device) > 0:

        user_features["device_connect_count"] = len(
            user_device
        )

        user_features["usb_connects"] = len(
            user_device[
                user_device["activity"] == "Connect"
            ]
        )

        user_features["usb_disconnects"] = len(
            user_device[
                user_device["activity"] == "Disconnect"
            ]
        )

    else:

        user_features["device_connect_count"] = 0
        user_features["usb_connects"] = 0
        user_features["usb_disconnects"] = 0


    # -----------------------------------------------------
    # FILE FEATURES
    # -----------------------------------------------------

    user_file = file_df[
        file_df["user"] == user
    ]

    if len(user_file) > 0:

        user_features["file_operations_count"] = len(
            user_file
        )

        user_features["unique_files_accessed"] = (
            user_file["filename"]
            .nunique()
        )

        user_features["file_exe_access"] = len(
            user_file[
                user_file["filename"].str.contains(
                    ".exe",
                    case=False,
                    na=False
                )
            ]
        )

        user_features["file_doc_access"] = len(
            user_file[
                user_file["filename"].str.contains(
                    r"\.(doc|docx|pdf|txt)",
                    case=False,
                    na=False
                )
            ]
        )

    else:

        user_features["file_operations_count"] = 0
        user_features["unique_files_accessed"] = 0
        user_features["file_exe_access"] = 0
        user_features["file_doc_access"] = 0


    # -----------------------------------------------------
    # EMAIL FEATURES
    # -----------------------------------------------------

    user_email = email_df[
        email_df["user"] == user
    ]

    if len(user_email) > 0:

        user_features["total_emails"] = len(
            user_email
        )

        user_features["emails_with_attachment"] = len(
            user_email[
                user_email["attachments"] > 0
            ]
        )

        user_features["unique_recipients"] = (
            user_email["to"]
            .nunique()
        )

        user_features["avg_email_size"] = (
            user_email["size"]
            .mean()
        )

    else:

        user_features["total_emails"] = 0
        user_features["emails_with_attachment"] = 0
        user_features["unique_recipients"] = 0
        user_features["avg_email_size"] = 0


    # -----------------------------------------------------
    # LABEL
    # -----------------------------------------------------

    user_features["is_malicious"] = (
        1
        if user in malicious_users
        else 0
    )


    features_list.append(
        user_features
    )


features_df = pd.DataFrame(
    features_list
)


print(
    f"✓ Created {len(features_df.columns)} columns "
    f"for {len(features_df)} users"
)


# Save feature dataset

features_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "user_features.csv"
    ),
    index=False
)

print("✓ Saved user_features.csv")


# =========================================================
# STEP 3: DATA PREPARATION
# =========================================================

print("\n[STEP 3/8] Preparing data...")


feature_cols = [
    col
    for col in features_df.columns
    if col not in [
        "user",
        "is_malicious"
    ]
]


X = features_df[
    feature_cols
].fillna(0)


y = features_df[
    "is_malicious"
]


print(
    f"✓ Number of features: {X.shape[1]}"
)

print(
    f"✓ Number of users: {X.shape[0]}"
)

print(
    f"✓ Malicious users: {y.sum()}"
)

print(
    f"✓ Normal users: {(1 - y).sum()}"
)


# =========================================================
# TRAIN / TEST SPLIT
# IMPORTANT:
# SMOTEENN is applied ONLY to training data
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nOriginal training distribution:")

print(
    y_train.value_counts()
)


# =========================================================
# STEP 4: SMOTEENN
# =========================================================

print("\n[STEP 4/8] Applying SMOTEENN...")


smote_enn = SMOTEENN(
    random_state=42
)


X_train_balanced, y_train_balanced = (
    smote_enn.fit_resample(
        X_train,
        y_train
    )
)


print(
    "✓ SMOTEENN completed"
)


print(
    "\nTraining distribution after SMOTEENN:"
)

print(
    pd.Series(
        y_train_balanced
    ).value_counts()
)


# =========================================================
# SCALE FEATURES
# =========================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_balanced
)

X_test_scaled = scaler.transform(
    X_test
)

X_all_scaled = scaler.transform(
    X
)


# Save scaler

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "insider_scaler.pkl"
    )
)


# =========================================================
# STEP 5: XGBOOST
# =========================================================

print("\n[STEP 5/8] Training XGBoost...")


xgb_model = XGBClassifier(

    n_estimators=200,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1

)


xgb_model.fit(
    X_train_scaled,
    y_train_balanced
)


print(
    "✓ XGBoost model trained"
)


# =========================================================
# XGBOOST PREDICTIONS
# =========================================================

xgb_pred = xgb_model.predict(
    X_test_scaled
)


xgb_probability = xgb_model.predict_proba(
    X_test_scaled
)[:, 1]


# =========================================================
# EVALUATION
# =========================================================

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)


xgb_f1 = f1_score(
    y_test,
    xgb_pred,
    zero_division=0
)


xgb_auc = roc_auc_score(
    y_test,
    xgb_probability
)


print("\n" + "=" * 60)

print("XGBOOST RESULTS")

print("=" * 60)

print(
    f"Accuracy : {xgb_accuracy:.4f}"
)

print(
    f"F1 Score : {xgb_f1:.4f}"
)

print(
    f"ROC-AUC  : {xgb_auc:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        xgb_pred,
        target_names=[
            "Normal",
            "Malicious"
        ],
        zero_division=0
    )
)


# Save XGBoost model

joblib.dump(
    xgb_model,
    os.path.join(
        MODEL_DIR,
        "insider_xgb_model.pkl"
    )
)


print(
    "\n✓ Saved insider_xgb_model.pkl"
)


# =========================================================
# STEP 6: ISOLATION FOREST
# =========================================================

print("\n[STEP 6/8] Training Isolation Forest...")


contamination = (
    len(malicious_users)
    /
    len(features_df)
)


# Safety limit

contamination = min(
    max(contamination, 0.01),
    0.49
)


iso_forest = IsolationForest(

    contamination=contamination,

    random_state=42,

    n_estimators=100

)


iso_pred = iso_forest.fit_predict(
    X_all_scaled
)


iso_raw_score = iso_forest.score_samples(
    X_all_scaled
)


# Convert Isolation Forest score
# into 0-100 anomaly risk

score_range = (
    iso_raw_score.max()
    -
    iso_raw_score.min()
)


if score_range == 0:

    iso_risk_score = np.zeros(
        len(iso_raw_score)
    )

else:

    iso_risk_score = (
        1
        -
        (
            (
                iso_raw_score
                -
                iso_raw_score.min()
            )
            /
            score_range
        )
    ) * 100


iso_prediction = (
    iso_pred == -1
).astype(int)


print(
    "✓ Isolation Forest trained"
)

print(
    f"✓ Predicted anomalies: "
    f"{iso_prediction.sum()}"
)


# =========================================================
# STEP 7: COMBINE XGBOOST + ISOLATION FOREST
# =========================================================

print(
    "\n[STEP 7/8] Creating combined behaviour risk score..."
)


# XGBoost probability for ALL users

xgb_all_probability = (
    xgb_model.predict_proba(
        X_all_scaled
    )[:, 1]
)


# Convert XGBoost probability to 0-100

xgb_risk_score = (
    xgb_all_probability
    * 100
)


# ---------------------------------------------------------
# HYBRID BEHAVIOUR SCORE
# ---------------------------------------------------------
#
# 60% XGBoost
# 40% Isolation Forest
#
# XGBoost = known malicious behaviour
# Isolation Forest = unusual behaviour
#

behaviour_score = (
    0.60 * xgb_risk_score
    +
    0.40 * iso_risk_score
)


behaviour_score = np.clip(
    behaviour_score,
    0,
    100
)


# Add results

features_df[
    "xgb_probability"
] = xgb_all_probability


features_df[
    "xgb_risk_score"
] = xgb_risk_score


features_df[
    "anomaly_score"
] = iso_raw_score


features_df[
    "isolation_forest_score"
] = iso_risk_score


features_df[
    "predicted_anomaly"
] = iso_prediction


features_df[
    "behaviour_score"
] = behaviour_score


features_df[
    "predicted_insider"
] = (
    behaviour_score >= 50
).astype(int)


# =========================================================
# STEP 8: SHAP EXPLANATIONS
# =========================================================

print(
    "\n[STEP 8/8] Generating SHAP explanations..."
)


explainer = shap.TreeExplainer(
    xgb_model
)


# Use limited samples for speed

sample_size = min(
    100,
    len(X_all_scaled)
)


np.random.seed(42)


sample_indices = np.random.choice(
    len(X_all_scaled),
    sample_size,
    replace=False
)


X_shap = X_all_scaled[
    sample_indices
]


shap_values = explainer.shap_values(
    X_shap
)


print(
    f"✓ SHAP calculated for "
    f"{sample_size} users"
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

mean_shap = np.abs(
    shap_values
).mean(
    axis=0
)


top_features = np.argsort(
    mean_shap
)[-10:]


print(
    "\nTop behavioural features:"
)


for index in reversed(
    top_features
):

    print(
        f"{feature_cols[index]} : "
        f"{mean_shap[index]:.4f}"
    )


# =========================================================
# SHAP SUMMARY CSV
# =========================================================

shap_summary = pd.DataFrame({

    "feature": feature_cols,

    "mean_abs_shap": mean_shap

})


shap_summary = shap_summary.sort_values(
    "mean_abs_shap",
    ascending=False
)


shap_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_feature_importance.csv"
    ),
    index=False
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

combined_prediction = (
    behaviour_score >= 50
).astype(int)


combined_cm = confusion_matrix(
    y,
    combined_prediction
)


print(
    "\nCombined Behaviour Detection:"
)


print(
    classification_report(
        y,
        combined_prediction,
        target_names=[
            "Normal",
            "Malicious"
        ],
        zero_division=0
    )
)


# =========================================================
# SAVE USER PREDICTIONS
# =========================================================

features_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "user_predictions.csv"
    ),
    index=False
)


print(
    "\n✓ Saved user_predictions.csv"
)


# =========================================================
# FUSION OUTPUT
# =========================================================

print(
    "\nPreparing Insider Module output for Fusion..."
)


insider_output = features_df[
    [
        "user",
        "behaviour_score",
        "predicted_insider",
        "xgb_risk_score",
        "isolation_forest_score"
    ]
].copy()


insider_output.rename(
    columns={

        "behaviour_score":
            "insider_score",

        "predicted_insider":
            "is_insider",

        "xgb_risk_score":
            "xgb_score",

        "isolation_forest_score":
            "anomaly_score"

    },
    inplace=True
)


INSIDER_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "outputs",
    "insider_scores.csv"
)


os.makedirs(
    os.path.dirname(
        INSIDER_OUTPUT_PATH
    ),
    exist_ok=True
)


insider_output.to_csv(
    INSIDER_OUTPUT_PATH,
    index=False
)


print(
    f"✓ Insider Fusion output saved to:"
)

print(
    INSIDER_OUTPUT_PATH
)


print(
    "\nInsider output preview:"
)

print(
    insider_output.head()
)


# =========================================================
# VISUALIZATION
# =========================================================

print(
    "\nGenerating visualizations..."
)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


# ---------------------------------------------------------
# RISK DISTRIBUTION
# ---------------------------------------------------------

axes[0, 0].hist(
    features_df[
        features_df["is_malicious"] == 0
    ]["behaviour_score"],
    bins=30,
    alpha=0.7,
    label="Normal",
    color="green"
)


axes[0, 0].hist(
    features_df[
        features_df["is_malicious"] == 1
    ]["behaviour_score"],
    bins=30,
    alpha=0.7,
    label="Malicious",
    color="red"
)


axes[0, 0].set_xlabel(
    "Behaviour Risk Score"
)

axes[0, 0].set_ylabel(
    "Frequency"
)

axes[0, 0].set_title(
    "Behaviour Risk Score Distribution"
)

axes[0, 0].legend()


# ---------------------------------------------------------
# SHAP FEATURE IMPORTANCE
# ---------------------------------------------------------

axes[0, 1].barh(
    range(len(top_features)),
    mean_shap[top_features]
)


axes[0, 1].set_yticks(
    range(len(top_features))
)


axes[0, 1].set_yticklabels(
    [
        feature_cols[i]
        for i in top_features
    ]
)


axes[0, 1].set_xlabel(
    "Mean |SHAP value|"
)


axes[0, 1].set_title(
    "Top 10 Behaviour Features"
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

sns.heatmap(
    combined_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1, 0],
    cbar=False
)


axes[1, 0].set_xlabel(
    "Predicted"
)

axes[1, 0].set_ylabel(
    "Actual"
)

axes[1, 0].set_title(
    "Combined Behaviour Detection"
)

axes[1, 0].set_xticklabels(
    ["Normal", "Malicious"]
)

axes[1, 0].set_yticklabels(
    ["Normal", "Malicious"]
)


# ---------------------------------------------------------
# ROC CURVE - XGBOOST
# ---------------------------------------------------------

fpr, tpr, _ = roc_curve(
    y_test,
    xgb_probability
)


axes[1, 1].plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"XGBoost AUC={xgb_auc:.3f}"
)


axes[1, 1].plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Classifier"
)


axes[1, 1].set_xlabel(
    "False Positive Rate"
)

axes[1, 1].set_ylabel(
    "True Positive Rate"
)

axes[1, 1].set_title(
    "XGBoost ROC Curve"
)

axes[1, 1].legend()

axes[1, 1].grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_evaluation.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("INSIDER THREAT PIPELINE COMPLETE ✓")
print("=" * 70)


print("\nModels used:")

print("✓ SMOTEENN")
print("✓ XGBoost")
print("✓ Isolation Forest")
print("✓ SHAP")


print("\nXGBoost Metrics:")

print(
    f"Accuracy : {xgb_accuracy:.4f}"
)

print(
    f"F1 Score : {xgb_f1:.4f}"
)

print(
    f"ROC-AUC  : {xgb_auc:.4f}"
)


print("\nBehaviour score formula:")

print(
    "Behaviour Score = "
    "0.60 × XGBoost Risk + "
    "0.40 × Isolation Forest Risk"
)


print("\nFiles created:")

print("• user_features.csv")
print("• user_predictions.csv")
print("• shap_feature_importance.csv")
print("• model_evaluation.png")
print("• insider_xgb_model.pkl")
print("• insider_scaler.pkl")


print("\n✓ INSIDER THREAT DETECTION COMPLETE!")