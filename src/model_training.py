import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


FEATURE_COLUMNS = [
    "volume",
    "market_cap",
    "moving_avg_3",
    "moving_avg_7",
    "volatility_7",
    "momentum_5",
    "volume_change",
    "market_cap_change",
    "is_anomaly",
]
TARGET_COLUMN = "target_next_price"


@dataclass
class ModelRun:
    name: str
    model: object
    y_true: np.ndarray
    y_pred: np.ndarray


def _time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].quantile(train_ratio)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    if test.empty:
        split_idx = int(len(df) * train_ratio)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
    return train, test


def train_models(df: pd.DataFrame, random_state: int = 42) -> Dict[str, ModelRun]:
    frame = df.copy().dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    train_df, test_df = _time_split(frame)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    runs: Dict[str, ModelRun] = {}

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    runs["baseline_dummy"] = ModelRun(
        name="baseline_dummy",
        model=baseline,
        y_true=y_test.to_numpy(),
        y_pred=baseline.predict(X_test),
    )

    rf = RandomForestRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2, 4],
        },
        n_iter=12,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=random_state,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    runs["optimized_random_forest"] = ModelRun(
        name="optimized_random_forest",
        model=best_rf,
        y_true=y_test.to_numpy(),
        y_pred=best_rf.predict(X_test),
    )

    if tf is not None:
        tf.random.set_seed(random_state)
        nn = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        nn.compile(optimizer="adam", loss="mse")
        nn.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        y_pred_nn = nn.predict(X_test, verbose=0).reshape(-1)
        runs["tensorflow_regressor"] = ModelRun(
            name="tensorflow_regressor",
            model=nn,
            y_true=y_test.to_numpy(),
            y_pred=y_pred_nn,
        )

    return runs


def save_model(model: object, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if tf is not None and isinstance(model, tf.keras.Model):
        model.save(path.replace(".pkl", ".keras"))
        return

    with open(path, "wb") as f:
        pickle.dump(model, f)
