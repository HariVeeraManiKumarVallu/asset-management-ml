# Asset Management ML

An end-to-end machine learning pipeline for simulated asset-management research, covering ingestion, feature engineering, anomaly detection, predictive modeling, portfolio optimization, and reporting.

## Project Goals

This project is designed to demonstrate:
- Predictive modeling for asset price forecasting
- Market anomaly detection for risk signals
- Portfolio strategy generation using risk-return optimization
- Reproducible experiment flow with SQL and report artifacts

## Tech Stack

- Python 3.10+
- scikit-learn
- TensorFlow (benchmark model)
- SQLite (SQL persistence)
- pandas, numpy
- matplotlib
- pyyaml

## Repository Structure

```text
asset-management-ml/
+-- config/
¦   +-- config.yaml
+-- data/
¦   +-- raw/
¦   +-- processed/
+-- database/
¦   +-- market_data.db
+-- models/
+-- reports/
+-- src/
¦   +-- ingestion.py
¦   +-- preprocessing.py
¦   +-- feature_engineering.py
¦   +-- statistical_analysis.py
¦   +-- anomaly_detection.py
¦   +-- model_training.py
¦   +-- model_evaluation.py
¦   +-- model_tracking.py
¦   +-- portfolio_strategy.py
¦   +-- visualization.py
+-- main.py
+-- requirements.txt
```

## Pipeline Overview

`main.py` executes the following flow:
1. Load config and input data
2. Expand to multi-asset simulated universe (optional)
3. Clean data and generate engineered features
4. Detect anomalies with IsolationForest
5. Compute advanced statistics (risk and return metrics)
6. Train models:
   - Baseline (`DummyRegressor`)
   - Optimized `RandomForestRegressor` (RandomizedSearchCV)
   - TensorFlow dense regressor benchmark
7. Compare model metrics (RMSE, MAE, R2, MAPE)
8. Optimize long-only portfolio (max Sharpe via Monte Carlo)
9. Save artifacts to SQLite, report, plots, and model file

## How To Run

```bash
python -m pip install -r requirements.txt
python main.py
```

## Key Outputs

After running the pipeline, the following artifacts are produced:
- `database/market_data.db`
- `models/trained_model.pkl` (or `.keras` for TensorFlow save path)
- `reports/performance_report.txt`
- `reports/price_trend.png`
- `reports/model_comparison.png`

## Config

All major settings are managed in `config/config.yaml`, including:
- simulation controls (asset count, periods, seed)
- anomaly contamination
- model random seed
- portfolio optimization parameters
- report output paths

## Notes

- The project is intended as a prototype/research workflow.
- Current CI runs syntax checks and executes the pipeline on push/PR.
- TensorFlow may show platform-specific warnings depending on local setup.
