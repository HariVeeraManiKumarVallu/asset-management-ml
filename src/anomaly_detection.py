import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    frame = df.copy()

    feature_cols = [
        "price_change",
        "volume_change",
        "volatility_7",
        "momentum_5",
        "market_cap_change",
    ]
    model = IsolationForest(contamination=contamination, random_state=42)
    frame["anomaly_label"] = model.fit_predict(frame[feature_cols])
    frame["anomaly_score"] = model.decision_function(frame[feature_cols])
    frame["is_anomaly"] = (frame["anomaly_label"] == -1).astype(int)

    return frame
