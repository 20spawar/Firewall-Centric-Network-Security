
import pandas as pd

from .config import (
    ANOMALY_OUTPUT,
    FUSION_OUTPUT,
    INSIDER_OUTPUT,
    PAYLOAD_OUTPUT,
    OUTPUT_DIR,
)

from .fusion_engine import FusionEngine


def load_csv(path, module_name):
    print(f"\nLoading {module_name} output...")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{module_name} output not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{module_name} output is empty: {path}")

    print(f"Loaded {len(df):,} rows.")
    return df


def main():

    print("=" * 70)
    print("FADE - THREE-MODULE FUSION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    insider_df = load_csv(INSIDER_OUTPUT, "Insider")
    anomaly_df = load_csv(ANOMALY_OUTPUT, "Anomaly")
    payload_df = load_csv(PAYLOAD_OUTPUT, "Payload")

    # Payload scores must be linked to actual users.
    if "user" not in payload_df.columns:
        raise ValueError(
            "\nPayload fusion cannot proceed yet.\n"
            "payload_scores.csv currently contains individual payload "
            "samples, not user identifiers.\n"
            "Add a genuine user/session mapping to the payload data "
            "and regenerate the scores before combining them with "
            "user-level Insider and Anomaly scores."
        )

    print("\nPerforming three-module fusion...")

    engine = FusionEngine()

    results = engine.fuse(
        insider_df=insider_df,
        anomaly_df=anomaly_df,
        payload_df=payload_df,
    )

    results.to_csv(FUSION_OUTPUT, index=False)

    print(f"\nFusion results saved to:\n{FUSION_OUTPUT}")

    print("\nTop 10 highest-risk users:")
    print(results.head(10).to_string(index=False))

    print("\nDecision distribution:")
    print(results["decision"].value_counts().to_string())

    print(
        "\nAverage fusion score:",
        round(results["final_score"].mean(), 4),
    )

    print("\nThree-module fusion completed.")


if __name__ == "__main__":
    main()
