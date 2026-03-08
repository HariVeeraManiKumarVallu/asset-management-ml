import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_price(df: pd.DataFrame, output_path: str = "reports/price_trend.png") -> None:
    plt.figure(figsize=(10, 5))
    for asset, sub in df.groupby("asset"):
        plt.plot(sub["date"], sub["price"], label=asset, linewidth=1)

    plt.title("Asset Price Trends")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_comparison(results: pd.DataFrame, output_path: str = "reports/model_comparison.png") -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(results["model"], results["rmse"], color=["#888888", "#1f77b4", "#2ca02c"][: len(results)])
    plt.title("Model RMSE Comparison")
    plt.ylabel("RMSE")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
