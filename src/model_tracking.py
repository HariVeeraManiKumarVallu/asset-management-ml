import datetime
from pathlib import Path

import pandas as pd


def log_model_performance(
    report_path: str,
    statistics: dict,
    model_results: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    anomaly_rate: float,
) -> None:
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    best = model_results.iloc[0]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Model Evaluation & Strategy Report\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")

        f.write("Advanced Statistical Analysis\n")
        for key, value in statistics.items():
            f.write(f"- {key}: {value:.6f}\n")

        f.write(f"\nAnomaly Rate: {anomaly_rate:.2%}\n\n")

        f.write("Model Comparison\n")
        f.write(model_results.to_string(index=False))

        f.write("\n\nBest Model\n")
        f.write(f"- Name: {best['model']}\n")
        f.write(f"- RMSE: {best['rmse']:.6f}\n")
        f.write(f"- MAE: {best['mae']:.6f}\n")
        f.write(f"- R2: {best['r2']:.6f}\n")
        f.write(f"- MAPE: {best['mape']:.2f}%\n")
        f.write(
            "- Improvement vs Baseline: "
            f"{best['rmse_improvement_vs_baseline_pct']:.2f}%\n"
        )

        f.write("\nPortfolio Strategy (Long-only, Max Sharpe)\n")
        f.write(portfolio_weights.to_string(index=False))
