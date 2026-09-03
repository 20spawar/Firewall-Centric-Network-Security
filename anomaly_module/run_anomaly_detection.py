import argparse

from .preprocessing import preprocess
from .feature_extraction import extract_features
from .anomaly_detector import AnomalyDetector
from .config import ANOMALY_MODEL_PATH


def main():

    parser = argparse.ArgumentParser(
        description="Run anomaly detection using Isolation Forest."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save anomaly results."
    )

    args = parser.parse_args()

    # Load and preprocess
    df = preprocess(args.input)

    # Preserve identifiers separately
    identifiers = df[
        [
            column
            for column in ["timestamp", "flow_id"]
            if column in df.columns
        ]
    ].reset_index(drop=True)

    # Generate model features
    X = extract_features(
        df,
        exclude_columns=[
            "timestamp",
            "flow_id",
            "label",
            "attack"
        ]
    )

    # Train detector
    detector = AnomalyDetector()

    detector.fit(X)

    # Predict
    results = detector.predict(X)

    # Combine identifiers and predictions
    results = results.reset_index(drop=True)

    final_results = identifiers.join(results)

    # Save results
    final_results.to_csv(
        args.output,
        index=False
    )

    # Save model
    detector.save(ANOMALY_MODEL_PATH)

    print("Anomaly detection completed.")
    print(f"Results saved to: {args.output}")
    print(f"Model saved to: {ANOMALY_MODEL_PATH}")


if __name__ == "__main__":
    main()