from typing import Dict, Tuple

import numpy as np
import pandas as pd


def optimize_portfolio(
    df: pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.02,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    returns = df.pivot(index="date", columns="asset", values="price_change").dropna(how="any")
    if returns.empty or returns.shape[1] < 2:
        single_asset = pd.DataFrame({"asset": [df["asset"].iloc[0]], "weight": [1.0]})
        return single_asset, {"expected_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}

    mean_returns = returns.mean() * 252
    cov = returns.cov() * 252

    rng = np.random.default_rng(seed)
    n_assets = len(mean_returns)

    best_sharpe = -np.inf
    best_weights = None
    best_stats = None

    for _ in range(n_portfolios):
        w = rng.random(n_assets)
        w = w / w.sum()

        port_return = float(np.dot(w, mean_returns.values))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov.values, w))))
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else -np.inf

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = w
            best_stats = {
                "expected_return": port_return,
                "volatility": port_vol,
                "sharpe_ratio": float(sharpe),
            }

    weights_df = pd.DataFrame({"asset": mean_returns.index, "weight": best_weights})
    weights_df = weights_df.sort_values("weight", ascending=False).reset_index(drop=True)

    return weights_df, best_stats
