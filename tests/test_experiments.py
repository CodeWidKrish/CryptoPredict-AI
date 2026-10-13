"""
Unit Tests for the 7-Experiment Academic Suite in CryptoPredict AI.
Validates baseline estimators, walk-forward regime boundaries, conformal intervals,
and friction-aware backtesting logic.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import EXPERIMENTS_DIR
from src.experiments import (
    ZeroReturnRegressor,
    PreviousCloseRegressor,
    run_backtest_experiment
)


def test_zero_return_regressor():
    """Verifies that ZeroReturnRegressor predicts exactly 0.0 for any input matrix."""
    reg = ZeroReturnRegressor()
    X = np.random.randn(25, 10)
    reg.fit(X)
    preds = reg.predict(X)
    assert len(preds) == 25
    assert (preds == 0.0).all()


def test_previous_close_regressor():
    """Verifies that PreviousCloseRegressor extracts the current Close column without alteration."""
    reg = PreviousCloseRegressor()
    df = pd.DataFrame({
        "Close": [50000.0, 51000.0, 49500.0],
        "Feature_1": [1.0, 2.0, 3.0]
    })
    preds = reg.predict(df)
    np.testing.assert_array_equal(preds, df["Close"].values)


def test_backtest_friction_logic():
    """
    Verifies that trading fee and slippage reduce portfolio capital upon position turnover.
    If price remains completely flat and strategy alternates Long and Cash, equity MUST decrease strictly.
    """
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    flat_price = 100.0
    prices = pd.Series([flat_price] * 10)
    # Alternating signals: 1, 0, 1, 0, 1, 0, ...
    signals = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    test_df = pd.DataFrame({"Date": dates, "Close": prices})
    pred_df = pd.DataFrame({"Predicted_Direction": signals})

    res = run_backtest_experiment(
        test_df=test_df,
        test_predictions_df=pred_df,
        initial_capital=10000.0,
        fee_pct=0.0010,
        slippage_pct=0.0005
    )

    strat_perf = res["strategy_performance"]
    # Due to repeated switching at 0.15% per trade on a flat asset, final equity must be strictly less than $10,000
    assert strat_perf["final_equity_usd"] < 10000.0
    assert strat_perf["total_trades"] > 0
    # Buy and hold on flat asset must have exactly 0% return
    bench_perf = res["benchmark_buy_and_hold"]
    assert np.isclose(bench_perf["cumulative_return_pct"], 0.0)


def test_conformal_interval_ordering():
    """Verifies that conformal quantiles obey monotonic ordering: q95 >= q90 >= q80 >= 0."""
    json_path = EXPERIMENTS_DIR / "prediction_intervals.json"
    assert json_path.exists(), f"Missing prediction intervals artifact: {json_path}"
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    q80 = data["quantiles_usd"]["q80_usd"]
    q90 = data["quantiles_usd"]["q90_usd"]
    q95 = data["quantiles_usd"]["q95_usd"]

    assert q95 >= q90 >= q80 >= 0.0
    assert data["empirical_test_coverage"]["actual_95_pct"] > 70.0


def test_all_experiment_artifacts_exist():
    """Verifies that all 7 experiment artifact files exist on disk with valid data."""
    required_files = [
        "return_prediction_experiment.csv",
        "return_prediction_experiment.json",
        "ablation_study.csv",
        "ablation_study.json",
        "walk_forward_validation.csv",
        "walk_forward_validation.json",
        "threshold_analysis.csv",
        "threshold_analysis.json",
        "shap_importance.csv",
        "shap_importance.json",
        "prediction_intervals.json",
        "backtest_results.json",
        "backtest_equity_curve.csv",
        "master_experiments_summary.json"
    ]
    for fname in required_files:
        p = EXPERIMENTS_DIR / fname
        assert p.exists(), f"Required experiment artifact missing: {fname}"
        assert p.stat().st_size > 0, f"Artifact file is empty: {fname}"
