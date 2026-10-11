from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FUSION_INPUT = (
    PROJECT_ROOT
    / "fusion"
    / "outputs"
    / "fusion_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "xai" / "outputs"

FUSION_XAI_OUTPUT = (
    OUTPUT_DIR / "fusion_explanations.csv"
)


# ============================================================
# LOAD FUSION RESULTS
# ============================================================

if not FUSION_INPUT.exists():
    raise FileNotFoundError(
        f"Fusion results file not found:\n{FUSION_INPUT}"
    )

df = pd.read_csv(FUSION_INPUT)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "user",
    "insider_score",
    "anomaly_score",
    "final_score",
    "decision"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

df["insider_score"] = pd.to_numeric(
    df["insider_score"],
    errors="coerce"
).fillna(0)

df["anomaly_score"] = pd.to_numeric(
    df["anomaly_score"],
    errors="coerce"
).fillna(0)

df["final_score"] = pd.to_numeric(
    df["final_score"],
    errors="coerce"
).fillna(0)


# ============================================================
# FUSION CONTRIBUTIONS
# ============================================================

INSIDER_WEIGHT = 0.50
ANOMALY_WEIGHT = 0.50

df["insider_contribution"] = (
    df["insider_score"] * INSIDER_WEIGHT
)

df["anomaly_contribution"] = (
    df["anomaly_score"] * ANOMALY_WEIGHT
)


# ============================================================
# DOMINANT CONTRIBUTOR
# ============================================================

def determine_dominant(row):

    if row["insider_contribution"] > row["anomaly_contribution"]:
        return "Insider Detection"

    elif row["anomaly_contribution"] > row["insider_contribution"]:
        return "Anomaly Detection"

    return "Equal Contribution"


df["dominant_contributor"] = df.apply(
    determine_dominant,
    axis=1
)


# ============================================================
# CONTRIBUTION PERCENTAGES
# ============================================================

total_contribution = (
    df["insider_contribution"]
    + df["anomaly_contribution"]
)

df["insider_contribution_percent"] = (
    df["insider_contribution"]
    / total_contribution.replace(0, 1)
    * 100
)

df["anomaly_contribution_percent"] = (
    df["anomaly_contribution"]
    / total_contribution.replace(0, 1)
    * 100
)


# ============================================================
# HUMAN-READABLE EXPLANATION
# ============================================================

def generate_explanation(row):

    insider = row["insider_contribution"]
    anomaly = row["anomaly_contribution"]
    final_score = row["final_score"]

    if insider > anomaly:

        return (
            f"Insider Detection contributed more to the "
            f"final risk score ({final_score:.2f}) than "
            f"Anomaly Detection. "
            f"Insider contribution: {insider:.2f}; "
            f"Anomaly contribution: {anomaly:.2f}."
        )

    elif anomaly > insider:

        return (
            f"Anomaly Detection contributed more to the "
            f"final risk score ({final_score:.2f}) than "
            f"Insider Detection. "
            f"Anomaly contribution: {anomaly:.2f}; "
            f"Insider contribution: {insider:.2f}."
        )

    return (
        f"Insider Detection and Anomaly Detection "
        f"contributed equally to the final risk score "
        f"({final_score:.2f})."
    )


df["explanation"] = df.apply(
    generate_explanation,
    axis=1
)


# ============================================================
# SAVE XAI OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_columns = [
    "user",
    "insider_score",
    "anomaly_score",
    "insider_contribution",
    "anomaly_contribution",
    "insider_contribution_percent",
    "anomaly_contribution_percent",
    "final_score",
    "decision",
    "dominant_contributor",
    "explanation"
]

xai_df = df[output_columns]

xai_df.to_csv(
    FUSION_XAI_OUTPUT,
    index=False
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("Fusion XAI explanation generated successfully.")
print(f"Saved to: {FUSION_XAI_OUTPUT}")
print(f"Users explained: {len(xai_df)}")
print()
print(xai_df.head())