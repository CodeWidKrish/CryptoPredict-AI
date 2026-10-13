"""
Tests for Feature Engineering, Indicator Calculations, and Zero-Leakage Invariance.
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

from src.features import build_feature_pipeline, compute_rsi
from config.config import REGRESSION_TARGET, CLASSIFICATION_TARGET


@pytest.fixture
def dummy_ohlcv_df():
    """Generates a controlled synthetic 150-day price series for indicator math tests."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=150, freq="D")
    base_price = 20000.0
    returns = np.random.normal(0.001, 0.02, 150)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Date": dates,
        "Open": prices * 0.99,
        "High": prices * 1.02,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(10000, 50000, 150)
    })
    return df


def test_rsi_bounds(dummy_ohlcv_df):
    """Verifies RSI stays strictly between 0 and 100."""
    rsi = compute_rsi(dummy_ohlcv_df["Close"], period=14)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()


def test_bollinger_bands_geometry(dummy_ohlcv_df):
    """Verifies upper band is always strictly greater than or equal to lower band."""
    df_feat, _, _ = build_feature_pipeline(dummy_ohlcv_df)
    valid = df_feat.dropna(subset=["Bollinger_Upper", "Bollinger_Lower"])
    assert (valid["Bollinger_Upper"] >= valid["Bollinger_Lower"]).all()


def test_target_alignment(dummy_ohlcv_df):
    """Verifies that regression and classification targets are shifted strictly by 1 day forward."""
    df_feat, _, _ = build_feature_pipeline(dummy_ohlcv_df)

    # Check alignment for first 10 rows
    for i in range(len(df_feat) - 1):
        expected_next_close = df_feat["Close"].iloc[i + 1]
        actual_target_close = df_feat[REGRESSION_TARGET].iloc[i]
        assert np.isclose(expected_next_close, actual_target_close), f"Row {i} target does not match Close[t+1]"

        current_close = df_feat["Close"].iloc[i]
        expected_dir = 1 if expected_next_close > current_close else 0
        actual_dir = df_feat[CLASSIFICATION_TARGET].iloc[i]
        assert expected_dir == actual_dir, f"Row {i} direction mismatch"

    # Last row target must be NaN
    assert np.isnan(df_feat[REGRESSION_TARGET].iloc[-1])
    assert np.isnan(df_feat[CLASSIFICATION_TARGET].iloc[-1])


def test_strict_zero_lookahead_leakage(dummy_ohlcv_df):
    """
    Formally tests temporal causality:
    Modifying future rows (t >= 50) MUST NOT alter engineered features at prior rows (t < 50).
    """
    df_original, feat_cols, _ = build_feature_pipeline(dummy_ohlcv_df)

    # Mutate future data dramatically starting at row index 50
    mutated_df = dummy_ohlcv_df.copy()
    mutated_df.loc[50:, "Close"] = mutated_df.loc[50:, "Close"] * 100.0
    mutated_df.loc[50:, "High"] = mutated_df.loc[50:, "High"] * 100.0
    mutated_df.loc[50:, "Low"] = mutated_df.loc[50:, "Low"] * 100.0

    df_after_mutation, _, _ = build_feature_pipeline(mutated_df)

    # Check rows 0 through 49: Every single feature must be 100% identical
    for col in feat_cols:
        orig_slice = df_original.loc[:49, col].dropna()
        mutated_slice = df_after_mutation.loc[:49, col].dropna()
        np.testing.assert_allclose(
            orig_slice.values,
            mutated_slice.values,
            err_msg=f"Future lookahead leakage detected in feature: {col}!"
        )
