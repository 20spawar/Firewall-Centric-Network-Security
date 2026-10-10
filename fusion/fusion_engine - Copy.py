import pandas as pd

from .config import (
    ANOMALY_WEIGHT,
    HIGH_RISK_THRESHOLD,
    INSIDER_WEIGHT,
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
)


class FusionEngine:

    def __init__(
        self,
        insider_weight=INSIDER_WEIGHT,
        anomaly_weight=ANOMALY_WEIGHT,
    ):

        total_weight = (
            insider_weight
            + anomaly_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "Fusion weights must be greater than zero."
            )

        # Normalize weights so they always sum to 1.
        self.insider_weight = (
            insider_weight / total_weight
        )

        self.anomaly_weight = (
            anomaly_weight / total_weight
        )

    # ========================================================
    # LOAD AND VALIDATE
    # ========================================================

    @staticmethod
    def validate_input(
        df,
        required_columns,
        dataset_name,
    ):

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset_name} is missing columns: "
                f"{missing}"
            )

        if df.empty:
            raise ValueError(
                f"{dataset_name} is empty."
            )

    # ========================================================
    # PREPARE INPUTS
    # ========================================================

    def prepare_inputs(
        self,
        insider_df,
        anomaly_df,
    ):

        self.validate_input(
            insider_df,
            [
                "user",
                "insider_score",
                "is_insider",
            ],
            "Insider output",
        )

        self.validate_input(
            anomaly_df,
            [
                "user",
                "anomaly_score",
                "is_anomaly",
            ],
            "Anomaly output",
        )

        insider = insider_df.copy()
        anomaly = anomaly_df.copy()

        # ----------------------------------------------------
        # Clean user IDs
        # ----------------------------------------------------

        insider["user"] = (
            insider["user"]
            .astype(str)
            .str.strip()
        )

        anomaly["user"] = (
            anomaly["user"]
            .astype(str)
            .str.strip()
        )

        # ----------------------------------------------------
        # Convert scores to numeric
        # ----------------------------------------------------

        insider["insider_score"] = pd.to_numeric(
            insider["insider_score"],
            errors="coerce",
        )

        anomaly["anomaly_score"] = pd.to_numeric(
            anomaly["anomaly_score"],
            errors="coerce",
        )

        # ----------------------------------------------------
        # Keep only required columns
        # ----------------------------------------------------

        insider = insider[
            [
                "user",
                "insider_score",
                "is_insider",
            ]
        ]

        anomaly = anomaly[
            [
                "user",
                "anomaly_score",
                "is_anomaly",
            ]
        ]

        # ----------------------------------------------------
        # Remove duplicate users
        # ----------------------------------------------------

        insider = (
            insider
            .drop_duplicates(
                subset=["user"],
                keep="first",
            )
        )

        anomaly = (
            anomaly
            .drop_duplicates(
                subset=["user"],
                keep="first",
            )
        )

        # ----------------------------------------------------
        # Merge
        # ----------------------------------------------------

        merged = pd.merge(
            insider,
            anomaly,
            on="user",
            how="outer",
        )

        # ----------------------------------------------------
        # Missing module scores = 0
        # ----------------------------------------------------

        merged["insider_score"] = (
            merged["insider_score"]
            .fillna(0)
            .clip(0, 100)
        )

        merged["anomaly_score"] = (
            merged["anomaly_score"]
            .fillna(0)
            .clip(0, 100)
        )

        merged["is_insider"] = (
            merged["is_insider"]
            .fillna(0)
            .astype(int)
        )

        merged["is_anomaly"] = (
            merged["is_anomaly"]
            .fillna(0)
            .astype(int)
        )

        return merged

    # ========================================================
    # CALCULATE FINAL SCORE
    # ========================================================

    def calculate_score(
        self,
        merged_df,
    ):

        result = merged_df.copy()

        result["final_score"] = (
            self.insider_weight
            * result["insider_score"]
            +
            self.anomaly_weight
            * result["anomaly_score"]
        )

        result["final_score"] = (
            result["final_score"]
            .clip(0, 100)
            .round(4)
        )

        return result

    # ========================================================
    # RISK CATEGORY
    # ========================================================

    @staticmethod
    def assign_decision(score):

        if score >= HIGH_RISK_THRESHOLD:
            return "HIGH"

        if score >= MEDIUM_RISK_THRESHOLD:
            return "MEDIUM"

        if score >= LOW_RISK_THRESHOLD:
            return "LOW"

        return "NORMAL"

    # ========================================================
    # ADD DECISION
    # ========================================================

    def add_decisions(
        self,
        df,
    ):

        result = df.copy()

        result["decision"] = (
            result["final_score"]
            .apply(
                self.assign_decision
            )
        )

        return result

    # ========================================================
    # COMPLETE FUSION
    # ========================================================

    def fuse(
        self,
        insider_df,
        anomaly_df,
    ):

        merged = self.prepare_inputs(
            insider_df,
            anomaly_df,
        )

        scored = self.calculate_score(
            merged
        )

        result = self.add_decisions(
            scored
        )

        # Sort highest risk first.
        result = result.sort_values(
            by="final_score",
            ascending=False,
        ).reset_index(drop=True)

        return result