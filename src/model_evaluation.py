from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def compare_model_runs(model_runs: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, run in model_runs.items():
        metrics = evaluate_model(run.y_true, run.y_pred)
        rows.append({"model": name, **metrics})

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    baseline_rmse = results.loc[results["model"] == "baseline_dummy", "rmse"].iloc[0]
    results["rmse_improvement_vs_baseline_pct"] = (
        (baseline_rmse - results["rmse"]) / baseline_rmse * 100
    )
    return results
