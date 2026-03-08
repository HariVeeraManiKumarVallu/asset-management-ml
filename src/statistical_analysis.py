from typing import Dict

import numpy as np
import pandas as pd


def compute_statistics(df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, float]:
    returns = df["price_change"].dropna()

    if returns.empty:
        return {
            "mean_daily_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "value_at_risk_95": 0.0,
            "cvar_95": 0.0,
            "max_drawdown": 0.0,
        }

    mean_daily = float(returns.mean())
    annualized_return = float((1 + mean_daily) ** 252 - 1)
    annualized_volatility = float(returns.std() * np.sqrt(252))
    sharpe = 0.0
    if annualized_volatility > 0:
        sharpe = float((annualized_return - risk_free_rate) / annualized_volatility)

    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean())

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1

    return {
        "mean_daily_return": mean_daily,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "value_at_risk_95": var_95,
        "cvar_95": cvar_95,
        "max_drawdown": float(drawdown.min()),
    }


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index="date", columns="asset", values="price_change").dropna(how="all")
    if pivot.empty:
        return pd.DataFrame()
    return pivot.corr().fillna(0.0)
