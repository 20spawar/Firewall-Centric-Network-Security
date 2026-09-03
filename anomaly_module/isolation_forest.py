import joblib

from sklearn.ensemble import IsolationForest


class IsolationForestDetector:

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

        self.model.fit(X)

        return self

    def predict(self, X):

        return self.model.predict(X)

    def decision_function(self, X):

        return self.model.decision_function(X)

    def save(self, path):

        joblib.dump(
            self.model,
            path
        )

    def load(self, path):

        self.model = joblib.load(path)

        return self