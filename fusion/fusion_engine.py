
import pandas as pd

from .config import (
    ANOMALY_WEIGHT,
    HIGH_RISK_THRESHOLD,
    INSIDER_WEIGHT,
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    PAYLOAD_WEIGHT,
)


class FusionEngine:

    def __init__(
        self,
        payload_weight=PAYLOAD_WEIGHT,
        insider_weight=INSIDER_WEIGHT,
        anomaly_weight=ANOMALY_WEIGHT,
    ):

        weights = {
            "payload": payload_weight,
            "insider": insider_weight,
            "anomaly": anomaly_weight,
        }

        if any(weight < 0 for weight in weights.values()):
            raise ValueError("Fusion weights cannot be negative.")

        total_weight = sum(weights.values())

        if total_weight <= 0:
            raise ValueError("At least one fusion weight must be positive.")

        self.payload_weight = payload_weight / total_weight
        self.insider_weight = insider_weight / total_weight
        self.anomaly_weight = anomaly_weight / total_weight

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def validate_input(df, required_columns, dataset_name):

        missing = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset_name} is missing columns: {missing}"
            )

        if df.empty:
            raise ValueError(f"{dataset_name} is empty.")

    # ========================================================
    # CLEAN USER-LEVEL INPUTS
    # ========================================================

    @staticmethod
    def prepare_user_scores(
        df,
        score_column,
        flag_column,
        dataset_name,
    ):

        FusionEngine.validate_input(
            df,
            ["user", score_column, flag_column],
            dataset_name,
        )

        result = df[["user", score_column, flag_column]].copy()

        result["user"] = result["user"].astype(str).str.strip()

        if result["user"].eq("").any():
            raise ValueError(
                f"{dataset_name} contains empty user identifiers."
            )

        result[score_column] = pd.to_numeric(
            result[score_column], errors="coerce"
        )

        if result[score_column].isna().any():
            raise ValueError(
                f"{dataset_name} contains missing or invalid scores."
            )

        result[score_column] = result[score_column].clip(0, 100)

        result[flag_column] = pd.to_numeric(
            result[flag_column], errors="coerce"
        ).fillna(0).astype(int)

        # Duplicate user rows require an explicit aggregation policy.
        if result["user"].duplicated().any():
            raise ValueError(
                f"{dataset_name} contains duplicate users. "
                "Aggregate scores per user before fusion."
            )

        return result

    # ========================================================
    # PREPARE AND MERGE MODULE OUTPUTS
    # ========================================================

    def prepare_inputs(self, insider_df, anomaly_df, payload_df):

        insider = self.prepare_user_scores(
            insider_df,
            "insider_score",
            "is_insider",
            "Insider output",
        )

        anomaly = self.prepare_user_scores(
            anomaly_df,
            "anomaly_score",
            "is_anomaly",
            "Anomaly output",
        )

        # Payload scores must be associated with actual users.
        self.validate_input(
            payload_df,
            ["user", "payload_score"],
            "Payload output",
        )

        payload = payload_df[["user", "payload_score"]].copy()
        payload["user"] = payload["user"].astype(str).str.strip()

        if payload["user"].eq("").any():
            raise ValueError(
                "Payload output contains empty user identifiers."
            )

        payload["payload_score"] = pd.to_numeric(
            payload["payload_score"], errors="coerce"
        )

        if payload["payload_score"].isna().any():
            raise ValueError(
                "Payload output contains missing or invalid scores."
            )

        payload["payload_score"] = payload["payload_score"].clip(0, 100)

        # Aggregate multiple payloads belonging to the same user.
        # Mean score is used; this policy can be changed after validation.
        payload = (
            payload.groupby("user", as_index=False)
            ["payload_score"].mean()
        )

        merged = insider.merge(
            anomaly,
            on="user",
            how="outer",
            validate="one_to_one",
        )

        merged = merged.merge(
            payload,
            on="user",
            how="outer",
            validate="one_to_one",
        )

        # Do not silently replace missing module scores with zero.
        score_columns = [
            "payload_score",
            "insider_score",
            "anomaly_score",
        ]

        missing_scores = merged[score_columns].isna()

        if missing_scores.any().any():
            affected = merged.loc[
                missing_scores.any(axis=1), "user"
            ].tolist()

            raise ValueError(
                "Some users lack scores from one or more modules. "
                "Resolve the missing user-to-module mappings before "
                f"fusion. Example affected users: {affected[:10]}"
            )

        merged["is_insider"] = merged["is_insider"].fillna(0).astype(int)
        merged["is_anomaly"] = merged["is_anomaly"].fillna(0).astype(int)

        return merged

    # ========================================================
    # CALCULATE THREE-MODULE FUSION SCORE
    # ========================================================

    def calculate_score(self, merged_df):

        result = merged_df.copy()

        result["final_score"] = (
            self.payload_weight * result["payload_score"]
            + self.insider_weight * result["insider_score"]
            + self.anomaly_weight * result["anomaly_score"]
        ).clip(0, 100).round(4)

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
    # COMPLETE FUSION
    # ========================================================

    def fuse(self, insider_df, anomaly_df, payload_df):

        merged = self.prepare_inputs(
            insider_df,
            anomaly_df,
            payload_df,
        )

        merged["decision"] = (
            self.calculate_score(merged)["final_score"]
            .apply(self.assign_decision)
        )

        result = self.calculate_score(merged)

        result["decision"] = result["final_score"].apply(
            self.assign_decision
        )

        result = result.sort_values(
            by="final_score",
            ascending=False,
        ).reset_index(drop=True)

        return result
