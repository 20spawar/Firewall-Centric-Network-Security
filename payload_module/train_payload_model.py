
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(r"D:\MAJOR PROJECT DATA\payload_generated")

PRIMARY_CSV = DATA_DIR / "FADE_like.csv"
INJECTIONS_CSV = DATA_DIR / "injections.csv"

MODULE_DIR = Path(__file__).resolve().parent
MODEL_DIR = MODULE_DIR / "models"
OUTPUT_DIR = MODULE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "payload_text_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================



def infer_protocol(class_value):
    """Classify a sample using its actual dataset category."""

    value = str(class_value).strip().upper()

    if not value or value in {"NAN", "NONE", "NULL"}:
        return "UNCLASSIFIED"

    # Application and transport protocols
    protocols = [
        "HTTPS", "HTTP", "FTP", "SMTP",
        "ARP", "TCP", "UDP", "ICMP", "DNS", "DHCP",
        "SSH", "TELNET", "POP3", "IMAP", "TLS", "SSL",
        "SFTP", "TFTP", "SNMP", "LDAP", "NTP",
        "IP", "IPV4", "IPV6"
    ]

    # Match specific names before shorter overlapping names.
    for protocol in sorted(protocols, key=len, reverse=True):
        if protocol in value:
            if protocol in {"TLS", "SSL"}:
                return "TLS/SSL"
            return protocol

    # Preserve other explicitly provided category names.
    if value not in {"OTHER", "UNKNOWN", "UNCLASSIFIED"}:
        return value

    return "UNCLASSIFIED"



def normalize_label(value):
    """Convert common benign/malicious labels into 0/1."""

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    benign_values = {
        "0", "0.0", "benign", "normal", "legitimate",
        "safe", "non-malicious", "non_malicious"
    }

    malicious_values = {
        "1", "1.0", "malicious", "attack", "anomaly",
        "malware", "injection", "intrusion"
    }

    if text in benign_values:
        return 0

    if text in malicious_values:
        return 1

    raise ValueError(
        f"Unrecognized label: {value!r}. "
        "Check the dataset's label column before training."
    )


def load_primary_dataset():
    """Load and validate the primary labeled dataset."""

    if not PRIMARY_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {PRIMARY_CSV}\n"
            "Check the folder path and CSV filename."
        )

    df = pd.read_csv(PRIMARY_CSV)

    required = {"sample", "label", "class"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"FADE_like.csv is missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[["sample", "label", "class"]].copy()
    df["sample"] = df["sample"].fillna("").astype(str).str.strip()
    df["target"] = df["label"].apply(normalize_label)

    df = df.dropna(subset=["target"])
    df = df[df["sample"] != ""].copy()
    df["target"] = df["target"].astype(int)
    df["protocol"] = df["class"].apply(infer_protocol)

    # Remove exact duplicate samples with conflicting labels.
    conflicts = df.groupby("sample")["target"].nunique()
    conflicting_samples = set(conflicts[conflicts > 1].index)

    if conflicting_samples:
        print(
            f"Warning: removing {len(conflicting_samples)} "
            "samples that have conflicting labels."
        )
        df = df[~df["sample"].isin(conflicting_samples)]

    df = df.drop_duplicates(subset=["sample", "target"])

    if df["target"].nunique() < 2:
        raise ValueError(
            "Training requires both benign (0) and malicious (1) examples."
        )

    return df

def score_dataset(model, df):
    """Generate payload threat scores for a dataset."""

    probabilities = model.predict_proba(df["sample"])[:, 1]

    result = pd.DataFrame({
        "sample": df["sample"].values,
        "class": df["class"].values,
        "protocol": df["protocol"].values,
        "payload_probability": probabilities,
        "payload_score": np.round(probabilities * 100, 2),
        "is_payload": (probabilities >= 0.5).astype(int),
        "decision": np.where(
            probabilities >= 0.5, "MALICIOUS", "BENIGN"
        ),
    })

    return result


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("FADE - PAYLOAD DETECTION MODEL")
    print("=" * 60)

    df = load_primary_dataset()

    print(f"\nDataset: {PRIMARY_CSV}")
    print(f"Usable samples: {len(df)}")
    print("\nLabel distribution:")
    print(df["target"].value_counts().sort_index().rename(
        index={0: "Benign", 1: "Malicious"}
    ))

    print("\nProtocol/category distribution:")
    print(df["protocol"].value_counts())

    X = df["sample"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=250000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    print("\nEvaluation results:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malicious"],
        zero_division=0,
    ))

    # Save evaluation results.
    with open(OUTPUT_DIR / "payload_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # Train the final model on all labeled primary data.
    print("\nTraining final model on all primary data...")
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")

    # Score primary dataset for later analysis.
    primary_scores = score_dataset(model, df)
    primary_scores["label"] = df["target"].values

    primary_output = OUTPUT_DIR / "payload_scores.csv"
    primary_scores.to_csv(primary_output, index=False)
    print(f"Primary scores saved to: {primary_output}")

    # Score supplementary injection examples.
    if INJECTIONS_CSV.exists():
        extra = pd.read_csv(INJECTIONS_CSV)

        if "sample" not in extra.columns:
            print(
                "Warning: injections.csv has no 'sample' column; "
                "skipping supplementary scoring."
            )
        else:
            extra = extra.copy()
            extra["sample"] = (
                extra["sample"].fillna("").astype(str).str.strip()
            )
            extra = extra[extra["sample"] != ""].copy()

            if "class" not in extra.columns:
                extra["class"] = "UNKNOWN"

            extra["protocol"] = extra["class"].apply(infer_protocol)

            extra_scores = score_dataset(model, extra)

            extra_output = OUTPUT_DIR / "injection_test_scores.csv"
            extra_scores.to_csv(extra_output, index=False)

            print(f"Injection scores saved to: {extra_output}")
            print(
                "Note: these are supplementary attack examples, "
                "not an independent balanced benchmark."
            )
    else:
        print(f"Supplementary file not found: {INJECTIONS_CSV}")

    print("\nPayload training and scoring completed.")


if __name__ == "__main__":
    main()
