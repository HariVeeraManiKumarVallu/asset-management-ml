from pathlib import Path

import yaml

from src.anomaly_detection import detect_anomalies
from src.feature_engineering import create_features
from src.ingestion import expand_simulated_universe, load_market_data, store_in_database
from src.model_evaluation import compare_model_runs
from src.model_tracking import log_model_performance
from src.model_training import save_model, train_models
from src.portfolio_strategy import optimize_portfolio
from src.preprocessing import clean_data, save_processed_data
from src.statistical_analysis import compute_statistics, correlation_analysis
from src.visualization import plot_model_comparison, plot_price


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(paths: list[str]) -> None:
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)


def run_pipeline() -> None:
    cfg = load_config()

    raw_path = cfg["data"]["raw_data"]
    processed_path = cfg["data"]["processed_data"]
    db_path = cfg["database"]["db_path"]
    model_path = cfg["model"]["model_path"]
    report_path = cfg["reports"]["performance_report"]

    ensure_directories([processed_path, db_path, model_path, report_path])

    df_raw = load_market_data(raw_path)

    sim_cfg = cfg.get("simulation", {})
    if sim_cfg.get("enabled", True):
        df_raw = expand_simulated_universe(
            df_raw,
            asset_count=int(sim_cfg.get("asset_count", 6)),
            periods=int(sim_cfg.get("periods", 365)),
            seed=int(sim_cfg.get("seed", 42)),
        )

    store_in_database(df_raw, db_path, table_name="market_data_raw")

    df_clean = clean_data(df_raw)
    save_processed_data(df_clean, processed_path)

    df_feat = create_features(df_clean)
    df_feat = detect_anomalies(df_feat, contamination=float(cfg["anomaly"]["contamination"]))

    stats = compute_statistics(df_feat, risk_free_rate=float(cfg["portfolio"]["risk_free_rate"]))
    corr = correlation_analysis(df_feat)

    model_runs = train_models(df_feat, random_state=int(cfg["model"]["random_state"]))
    model_results = compare_model_runs(model_runs)

    best_name = model_results.iloc[0]["model"]
    best_model = model_runs[best_name].model
    save_model(best_model, model_path)

    portfolio_weights, portfolio_stats = optimize_portfolio(
        df_feat,
        n_portfolios=int(cfg["portfolio"]["n_portfolios"]),
        risk_free_rate=float(cfg["portfolio"]["risk_free_rate"]),
        seed=int(cfg["portfolio"]["seed"]),
    )
    portfolio_stats = {
        "portfolio_expected_return": portfolio_stats["expected_return"],
        "portfolio_volatility": portfolio_stats["volatility"],
        "portfolio_sharpe_ratio": portfolio_stats["sharpe_ratio"],
    }

    anomaly_rate = float(df_feat["is_anomaly"].mean())

    log_model_performance(
        report_path=report_path,
        statistics={**stats, **portfolio_stats},
        model_results=model_results,
        portfolio_weights=portfolio_weights,
        anomaly_rate=anomaly_rate,
    )

    plot_price(df_feat, output_path=cfg["reports"]["price_plot"])
    plot_model_comparison(model_results, output_path=cfg["reports"]["model_plot"])

    store_in_database(df_feat, db_path, table_name="market_data_features")
    store_in_database(model_results, db_path, table_name="model_results")
    store_in_database(portfolio_weights, db_path, table_name="portfolio_weights")
    if not corr.empty:
        store_in_database(corr.reset_index(), db_path, table_name="asset_correlation")

    print("Advanced Statistical Analysis:", stats)
    print("Model Comparison:\n", model_results.to_string(index=False))
    print("Portfolio Strategy:\n", portfolio_weights.to_string(index=False))
    print("Pipeline Completed")


if __name__ == "__main__":
    run_pipeline()
