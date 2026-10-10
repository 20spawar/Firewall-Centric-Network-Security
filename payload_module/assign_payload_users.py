
import shutil
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = Path(
    r"D:\MAJOR PROJECT DATA\payload_generated\FADE_like.csv"
)

INSIDER_PATH = (
    PROJECT_ROOT / "insider_module" / "outputs"
    / "insider_scores.csv"
)

ANOMALY_PATH = (
    PROJECT_ROOT / "anomaly_module" / "outputs"
    / "anomaly_scores.csv"
)

PAYLOAD_OUTPUT_DIR = PROJECT_ROOT / "payload_module" / "outputs"

SCORES_PATH = PAYLOAD_OUTPUT_DIR / "payload_scores.csv"
BACKUP_PATH = PAYLOAD_OUTPUT_DIR / "payload_scores_before_mapping.csv"

MAPPED_DATA_PATH = DATA_PATH.parent / "FADE_like_with_users.csv"


def normalize_users(series):
    return series.dropna().astype(str).str.strip()


def main():
    # Check required files
    required_files = [
        DATA_PATH,
        INSIDER_PATH,
        ANOMALY_PATH,
        SCORES_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load datasets
    payload_df = pd.read_csv(DATA_PATH)
    insider_df = pd.read_csv(INSIDER_PATH)
    anomaly_df = pd.read_csv(ANOMALY_PATH)
    scores_df = pd.read_csv(SCORES_PATH)

    # Validate user columns
    for name, df in [
        ("Insider", insider_df),
        ("Anomaly", anomaly_df),
    ]:
        if "user" not in df.columns:
            raise ValueError(f"{name} scores must contain a 'user' column.")

    # Both modules must cover the same users
    insider_users = set(normalize_users(insider_df["user"]))
    anomaly_users = set(normalize_users(anomaly_df["user"]))

    if not insider_users:
        raise ValueError("No user IDs found in Insider scores.")

    if insider_users != anomaly_users:
        raise ValueError(
            "Insider and Anomaly outputs contain different user IDs. "
            "Resolve the user lists before assigning payload IDs."
        )

    # Validate payload identifiers
    if "sample" not in payload_df.columns:
        raise ValueError("FADE_like.csv must contain a 'sample' column.")

    if "sample" not in scores_df.columns:
        raise ValueError("payload_scores.csv must contain a 'sample' column.")

    if payload_df["sample"].isna().any():
        raise ValueError("Payload dataset contains missing sample IDs.")

    if payload_df["sample"].astype(str).duplicated().any():
        raise ValueError(
            "Payload sample IDs are not unique. "
            "A reliable sample-to-score mapping needs unique identifiers."
        )

    if scores_df["sample"].astype(str).duplicated().any():
        raise ValueError(
            "Payload score sample IDs are not unique."
        )

    # Check that every scored sample exists in the source dataset
    source_samples = set(payload_df["sample"].astype(str))
    scored_samples = set(scores_df["sample"].astype(str))

    if source_samples != scored_samples:
        raise ValueError(
            "The source dataset and payload scores contain different sample IDs. "
            "Regenerate payload scores from FADE_like.csv before continuing."
        )

    # Assign existing user IDs consistently in a reproducible,
    # explicitly synthetic round-robin mapping.
    users = sorted(insider_users)

    payload_df = payload_df.copy()
    payload_df["user"] = [
        users[i % len(users)] for i in range(len(payload_df))
    ]
    payload_df["user_mapping_type"] = "SYNTHETIC_ROUND_ROBIN"

    # Create a sample-to-user mapping
    mapping = payload_df[["sample", "user", "user_mapping_type"]].copy()
    mapping["sample"] = mapping["sample"].astype(str)

    scores_df = scores_df.copy()
    scores_df["sample"] = scores_df["sample"].astype(str)

    # Attach exactly the same user ID to each scored sample
    scores_df = scores_df.drop(
        columns=["user", "user_mapping_type"],
        errors="ignore",
    )

    scores_df = scores_df.merge(
        mapping,
        on="sample",
        how="left",
        validate="one_to_one",
    )

    if scores_df["user"].isna().any():
        raise ValueError("Some payload scores could not be mapped to users.")

    # Preserve a copy of the original scores before updating the file
    PAYLOAD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BACKUP_PATH.exists():
        shutil.copy2(SCORES_PATH, BACKUP_PATH)

    # Save outputs
    payload_df.to_csv(MAPPED_DATA_PATH, index=False)
    scores_df.to_csv(SCORES_PATH, index=False)

    print("\nSynthetic payload-to-user mapping completed.")
    print(f"Mapped payload dataset: {MAPPED_DATA_PATH}")
    print(f"Updated payload scores: {SCORES_PATH}")
    print(f"Original scores backup: {BACKUP_PATH}")
    print(f"Number of users: {len(users)}")
    print(f"Number of payload samples: {len(payload_df)}")

    print("\nPayload samples per user:")
    print(payload_df["user"].value_counts().sort_index())


if __name__ == "__main__":
    main()
