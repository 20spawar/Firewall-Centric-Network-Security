from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import (
    CONTAMINATION,
    MODEL_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
)

from .isolation_forest import IsolationForestDetector


class AnomalyDetector:

    def __init__(
        self,
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    ):

        self.imputer = SimpleImputer(
            strategy="median"
        )

        self.scaler = StandardScaler()

        self.model = IsolationForestDetector(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

        self.feature_columns = None

        self.is_fitted = False

    # ========================================================
    # FIT
    # ========================================================

    def fit(self, df):

        if "user" not in df.columns:
            raise ValueError(
                "Input data must contain a 'user' column."
            )

        self.feature_columns = [
            column
            for column in df.columns
            if column != "user"
        ]

        X = df[self.feature_columns].copy()

        X = self.imputer.fit_transform(X)

        X = self.scaler.fit_transform(X)

        self.model.fit(X)

        self.is_fitted = True

        return self

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, df):

        if not self.is_fitted:

            raise RuntimeError(
                "Model has not been fitted. "
                "Call fit() first."
            )

        X = df[
            self.feature_columns
        ].copy()

        X = self.imputer.transform(X)

        X = self.scaler.transform(X)

        predictions = self.model.predict(X)

        decision_scores = (
            self.model
            .decision_function(X)
        )

        anomaly_scores = (
            self._convert_to_risk_score(
                decision_scores
            )
        )

        result = pd.DataFrame()

        result["user"] = (
            df["user"].values
        )

        result["anomaly_score"] = (
            anomaly_scores
        )

        result["is_anomaly"] = (
            predictions == -1
        ).astype(int)

        return result

    # ========================================================
    # FIT + PREDICT
    # ========================================================

    def fit_predict(self, df):

        self.fit(df)

        return self.predict(df)

    # ========================================================
    # SCORE CONVERSION
    # ========================================================

    @staticmethod
    def _convert_to_risk_score(
        decision_scores
    ):

        scores = np.asarray(
            decision_scores,
            dtype=float
        )

        minimum = scores.min()
        maximum = scores.max()

        if np.isclose(
            minimum,
            maximum
        ):

            return np.zeros(
                len(scores)
            )

        normalized = (
            scores - minimum
        ) / (
            maximum - minimum
        )

        # Isolation Forest:
        # lower decision score = more anomalous
        risk_scores = (
            1 - normalized
        ) * 100

        return np.round(
            risk_scores,
            4
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path=MODEL_PATH
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        package = {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "model": self.model.model,
            "feature_columns": self.feature_columns,
        }

        joblib.dump(
            package,
            path
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        path=MODEL_PATH
    ):

        package = joblib.load(path)

        self.imputer = (
            package["imputer"]
        )

        self.scaler = (
            package["scaler"]
        )

        self.model.model = (
            package["model"]
        )

        self.feature_columns = (
            package["feature_columns"]
        )

        self.is_fitted = True

        return self