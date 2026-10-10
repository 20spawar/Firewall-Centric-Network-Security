import pandas as pd

from .config import (
    ANOMALY_OUTPUT,
    FUSION_OUTPUT,
    INSIDER_OUTPUT,
    OUTPUT_DIR,
)

from .fusion_engine import FusionEngine


def main():

    print("=" * 70)
    print("FUSION MODULE")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load Insider output
    # --------------------------------------------------------

    print("\n[1/4] Loading Insider Module output...")

    print(INSIDER_OUTPUT)

    insider_df = pd.read_csv(
        INSIDER_OUTPUT
    )

    print(
        f"✓ Insider users: "
        f"{len(insider_df):,}"
    )

    # --------------------------------------------------------
    # Load Anomaly output
    # --------------------------------------------------------

    print("\n[2/4] Loading Anomaly Module output...")

    print(ANOMALY_OUTPUT)

    anomaly_df = pd.read_csv(
        ANOMALY_OUTPUT
    )

    print(
        f"✓ Anomaly users: "
        f"{len(anomaly_df):,}"
    )

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    print("\n[3/4] Performing score fusion...")

    engine = FusionEngine()

    results = engine.fuse(
        insider_df,
        anomaly_df,
    )

    print(
        "✓ Fusion completed."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        FUSION_OUTPUT,
        index=False,
    )

    print("\n[4/4] Saving results...")

    print(
        f"✓ Fusion output saved to:"
    )

    print(
        FUSION_OUTPUT
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FUSION RESULTS"
    )

    print(
        "=" * 70
    )

    print("\nColumns:")

    print(
        results.columns.tolist()
    )

    print("\nTop 10 highest-risk users:")

    print(
        results.head(10)
        .to_string(index=False)
    )

    print("\nDecision distribution:")

    print(
        results["decision"]
        .value_counts()
    )

    print("\nAverage final score:")

    print(
        round(
            results["final_score"].mean(),
            4,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FUSION MODULE COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()