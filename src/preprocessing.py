from pathlib import Path

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "asset", "price", "volume", "market_cap"])

    frame["asset"] = frame["asset"].astype(str)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")

    frame = frame.dropna().drop_duplicates(subset=["date", "asset"]) 
    frame["volume"] = frame["volume"].astype(int)
    frame["market_cap"] = frame["market_cap"].astype(int)

    return frame.sort_values(["asset", "date"]).reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
