from pathlib import Path

from .anomaly_detector import AnomalyDetector

from .config import (
    ANOMALY_OUTPUT_PATH,
    FEATURE_OUTPUT_PATH,
    HTTP_DATA_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
)

from .http_features import (
    build_user_features
)


def main():

    print("=" * 70)
    print("CERT r4.2 HTTP ANOMALY DETECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Step 1: Feature extraction
    # --------------------------------------------------------

    print("\n[1/4] Building user-level HTTP features...")

    user_features = build_user_features(
        HTTP_DATA_PATH
    )

    print(
        f"\n✓ Feature extraction complete."
    )

    print(
        f"Users: {len(user_features):,}"
    )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    user_features.to_csv(
        FEATURE_OUTPUT_PATH,
        index=False
    )

    print(
        f"\n✓ User features saved to:"
    )

    print(
        FEATURE_OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Step 2: Train Isolation Forest
    # --------------------------------------------------------

    print(
        "\n[2/4] Training Isolation Forest..."
    )

    detector = AnomalyDetector()

    detector.fit(
        user_features
    )

    print(
        "✓ Isolation Forest training complete."
    )

    # --------------------------------------------------------
    # Step 3: Generate predictions
    # --------------------------------------------------------

    print(
        "\n[3/4] Generating anomaly scores..."
    )

    predictions = detector.predict(
        user_features
    )

    print(
        "✓ Predictions generated."
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions.to_csv(
        ANOMALY_OUTPUT_PATH,
        index=False
    )

    print(
        "\n✓ Anomaly output saved to:"
    )

    print(
        ANOMALY_OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Step 4: Save model
    # --------------------------------------------------------

    print(
        "\n[4/4] Saving model..."
    )

    detector.save(
        MODEL_PATH
    )

    print(
        "✓ Model saved to:"
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ANOMALY DETECTION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nOutput columns:"
    )

    print(
        predictions.columns.tolist()
    )

    print(
        "\nFirst 10 users:"
    )

    print(
        predictions.head(10).to_string(
            index=False
        )
    )

    print(
        "\nAnomaly distribution:"
    )

    print(
        predictions[
            "is_anomaly"
        ].value_counts()
    )

    print(
        "\nAverage anomaly score:"
    )

    print(
        round(
            predictions[
                "anomaly_score"
            ].mean(),
            4
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANOMALY MODULE COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()