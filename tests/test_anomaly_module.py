import pandas as pd

from anomaly_module.anomaly_detector import AnomalyDetector


def test_anomaly_detector():
    data = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 100],
        "feature_2": [10, 20, 30, 40, 50, 1000],
    })

    detector = AnomalyDetector()

    detector.fit(data)

    result = detector.predict(data)

    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns
    assert len(result) == len(data)