
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODULE_DIR / "models" / "payload_text_model.joblib"
OUTPUT_DIR = MODULE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def infer_protocol(class_value):
    value = str(class_value).strip().upper()

    if "FTP" in value:
        return "FTP"
    if "SMTP" in value or "MAIL" in value:
        return "SMTP"
    if "HTTP" in value or "WEB" in value:
        return "HTTP"

    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(
        description="Score payload samples with the FADE model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a CSV file containing a sample column.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_payload_model.py first."
        )

    df = pd.read_csv(input_path)

    if "sample" not in df.columns:
        raise ValueError(
            f"Input CSV must contain a 'sample' column. "
            f"Found: {list(df.columns)}"
        )

    df["sample"] = df["sample"].fillna("").astype(str)

    if "class" not in df.columns:
        df["class"] = "UNKNOWN"

    df["protocol"] = df["class"].apply(infer_protocol)

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(df["sample"])[:, 1]

    df["payload_probability"] = probabilities
    df["payload_score"] = np.round(probabilities * 100, 2)
    df["is_payload"] = (probabilities >= 0.5).astype(int)
    df["decision"] = np.where(
        probabilities >= 0.5, "MALICIOUS", "BENIGN"
    )

    output_path = OUTPUT_DIR / f"{input_path.stem}_scored.csv"
    df.to_csv(output_path, index=False)

    print(f"Scored {len(df)} samples.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
