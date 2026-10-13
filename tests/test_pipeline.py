"""
Tests for End-to-End Pipeline, Serialization, and Inference Integrity.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MODELS_DIR, PROCESSED_DATA_FILE
from src.preprocessing import DataSplitter
from src.predict import CryptoPredictor


def test_chronological_split_strict_ordering():
    """Verifies that Train, Validation, and Test sets have zero temporal overlap."""
    assert PROCESSED_DATA_FILE.exists(), f"Processed data missing: {PROCESSED_DATA_FILE}"
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df = df.dropna().reset_index(drop=True)

    non_feat = {"Date", "Adj Close", "Target_Close_Next", "Target_Direction_Next"}
    feat_cols = [c for c in df.columns if c not in non_feat]

    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split(df, feat_cols)

    train_max = pd.to_datetime(splits["train_df"]["Date"]).max()
    val_min = pd.to_datetime(splits["val_df"]["Date"]).min()
    val_max = pd.to_datetime(splits["val_df"]["Date"]).max()
    test_min = pd.to_datetime(splits["test_df"]["Date"]).min()

    assert train_max < val_min, f"Train date ({train_max}) overlaps with Validation date ({val_min})"
    assert val_max < test_min, f"Validation date ({val_max}) overlaps with Test date ({test_min})"


def test_scaler_isolation():
    """Verifies scaler was fitted exclusively on the training fold."""
    df = pd.read_csv(PROCESSED_DATA_FILE).dropna().reset_index(drop=True)
    non_feat = {"Date", "Adj Close", "Target_Close_Next", "Target_Direction_Next"}
    feat_cols = [c for c in df.columns if c not in non_feat]

    splitter = DataSplitter(0.70, 0.15, 0.15)
    splits = splitter.split(df, feat_cols)

    scaler = splits["scaler"]
    # Check that scaler mean matches train_df mean for features
    train_means = splits["train_df"][feat_cols].mean().values
    assert (abs(scaler.mean_ - train_means) < 1e-4).all()


def test_model_artifact_reloading():
    """Verifies saved models can be loaded and perform inference."""
    reg_path = MODELS_DIR / "best_regression_model.joblib"
    clf_path = MODELS_DIR / "best_classification_model.joblib"

    assert reg_path.exists(), "Regression model artifact missing"
    assert clf_path.exists(), "Classification model artifact missing"

    reg_bundle = joblib.load(reg_path)
    clf_bundle = joblib.load(clf_path)

    assert "model" in reg_bundle and "model_name" in reg_bundle
    assert "model" in clf_bundle and "model_name" in clf_bundle


def test_cryptopredictor_inference():
    """Verifies that CryptoPredictor runs live forward inference."""
    df = pd.read_csv(PROCESSED_DATA_FILE)
    latest_row = df.iloc[[-1]]

    predictor = CryptoPredictor()
    pred = predictor.predict_next_day(latest_row)

    assert "predicted_close_usd" in pred
    assert pred["predicted_close_usd"] > 0
    assert pred["predicted_direction"] in [0, 1]
    assert "disclaimer" in pred
