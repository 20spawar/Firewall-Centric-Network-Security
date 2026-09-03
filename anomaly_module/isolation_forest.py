from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest


class IsolationForestModel:

    def __init__(
        self,
        n_estimators=200,
        contamination="auto",
        random_state=42
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X):
        """
        Train the Isolation Forest.
        """
        self.model.fit(X)
        return self

    def predict(self, X):
        """
        Return Isolation Forest predictions.

        1  = normal
        -1 = anomaly
        """
        return self.model.predict(X)

    def score(self, X):
        """
        Return anomaly decision scores.
        """
        return self.model.decision_function(X)

    def save(self, path):
        """
        Save trained model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, path)

    def load(self, path):
        """
        Load a previously trained model.
        """
        self.model = joblib.load(path)
        return self