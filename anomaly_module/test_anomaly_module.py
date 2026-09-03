import numpy as np

from anomaly_module.anomaly_detector import AnomalyDetector


def test_anomaly_detector():

    X = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [100, 100]
    ])

    detector = AnomalyDetector(
        n_estimators=100,
        random_state=42
    )

    detector.fit(X)

    results = detector.predict(X)

    assert "anomaly_score" in results.columns
    assert "is_anomaly" in results.columns

    assert len(results) == len(X)