import sqlite3
from typing import Optional

import numpy as np
import pandas as pd


def load_market_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def expand_simulated_universe(
    df: pd.DataFrame,
    asset_count: int = 6,
    periods: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Expand a small seed dataset into a larger simulated multi-asset universe."""
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    if frame.empty:
        return frame

    base_price = float(frame["price"].iloc[0])
    base_volume = float(frame["volume"].median())
    base_market_cap = float(frame["market_cap"].median())

    base_returns = frame["price"].pct_change().dropna()
    mu = float(base_returns.mean()) if not base_returns.empty else 0.0003
    sigma = float(base_returns.std()) if not base_returns.empty else 0.018
    mu = float(np.clip(mu, -0.001, 0.001))
    sigma = float(np.clip(sigma, 0.01, 0.03))

    assets = [
        "Apple",
        "Microsoft",
        "Google",
        "Amazon",
        "Nvidia",
        "Tesla",
        "Meta",
        "Netflix",
    ][: max(1, asset_count)]

    dates = pd.date_range(frame["date"].min(), periods=periods, freq="D")
    seq = pd.Series(range(len(dates)))
    rng = np.random.default_rng(seed)

    expanded = []
    for i, asset in enumerate(assets):
        asset_mu = mu + (i - (len(assets) / 2)) * 0.0002
        asset_sigma = max(0.006, sigma + (i * 0.0015))

        returns = rng.normal(loc=asset_mu, scale=asset_sigma, size=len(dates))
        prices = pd.Series((1.0 + returns).cumprod() * (base_price * (1 + i * 0.08)))
        volumes = (base_volume * (1 + 0.1 * i)) * (1.0 + (seq % 14) / 100.0)
        market_cap = (base_market_cap * (1 + 0.15 * i)) * (prices / prices.iloc[0])

        asset_df = pd.DataFrame(
            {
                "date": dates,
                "asset": asset,
                "price": prices.round(2),
                "volume": volumes.round().astype(int),
                "market_cap": market_cap.round(0).astype(int),
            }
        )
        expanded.append(asset_df)

    return pd.concat(expanded, ignore_index=True)


def store_in_database(df: pd.DataFrame, db_path: str, table_name: str = "market_data") -> None:
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
