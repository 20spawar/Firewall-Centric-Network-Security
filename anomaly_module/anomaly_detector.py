import pandas as pd

from .isolation_forest import IsolationForestModel
from .config import (
    N_ESTIMATORS,
    CONTAMINATION,
    RANDOM_STATE
)


class AnomalyDetector:

    def __init__(
        self,
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE
    ):

        self.model = IsolationForestModel(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X):
        """
        Train the anomaly detection model.
        """
        self.model.fit(X)

    def predict(self, X):
        """
        Generate anomaly predictions.
        """

        predictions = self.model.predict(X)
        scores = self.model.score(X)

        results = pd.DataFrame({
            "anomaly_score": scores,
            "is_anomaly": predictions == -1
        })

        return results

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)