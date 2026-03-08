import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy().sort_values(["asset", "date"])

    grouped = frame.groupby("asset", group_keys=False)
    frame["price_change"] = grouped["price"].pct_change()
    frame["log_return"] = grouped["price"].transform(lambda s: np.log(s / s.shift(1)))

    frame["moving_avg_3"] = grouped["price"].transform(lambda s: s.rolling(window=3).mean())
    frame["moving_avg_7"] = grouped["price"].transform(lambda s: s.rolling(window=7).mean())
    frame["volatility_7"] = grouped["price_change"].transform(lambda s: s.rolling(window=7).std())
    frame["momentum_5"] = grouped["price"].pct_change(periods=5)
    frame["volume_change"] = grouped["volume"].pct_change()
    frame["market_cap_change"] = grouped["market_cap"].pct_change()
    frame["target_next_price"] = grouped["price"].shift(-1)

    frame = frame.dropna().reset_index(drop=True)
    return frame
