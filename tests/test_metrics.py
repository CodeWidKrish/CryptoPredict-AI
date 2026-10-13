"""
Tests for Evaluation Metric Formulas and Edge Cases.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import compute_regression_metrics, compute_classification_metrics


def test_regression_metrics_known_values():
    """Tests regression metrics against manually computed ground truth."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 315.0, 380.0])
    # Errors: +10, -10, +15, -20
    # Absolute errors: 10, 10, 15, 20 -> MAE = 55 / 4 = 13.75
    # Squared errors: 100, 100, 225, 400 -> MSE = 825 / 4 = 206.25
    # RMSE = sqrt(206.25) = 14.3614
    metrics = compute_regression_metrics(y_true, y_pred)

    assert np.isclose(metrics["mae"], 13.75)
    assert np.isclose(metrics["mse"], 206.25)
    assert np.isclose(metrics["rmse"], np.sqrt(206.25))
    assert metrics["r2_score"] > 0.95


def test_regression_metrics_zero_mape_protection():
    """Tests that MAPE handles true values at or near zero safely without ZeroDivisionError."""
    y_true = np.array([0.0, 0.0, 100.0])
    y_pred = np.array([10.0, 5.0, 100.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert not np.isnan(metrics["mape_pct"])
    assert not np.isinf(metrics["mape_pct"])


def test_classification_metrics_exact_formulas():
    """
    Tests exact classification metrics with predetermined confusion matrix counts:
    TN=40, FP=10, FN=20, TP=30 -> Total=100
    Accuracy = (30+40)/100 = 70%
    Precision = 30 / (30+10) = 75%
    Sensitivity = 30 / (30+20) = 60%
    Specificity = 40 / (40+10) = 80%
    F1 = 2 * (0.75 * 0.60) / (0.75 + 0.60) = 2 * 0.45 / 1.35 = 66.67%
    Error Rate = (10+20)/100 = 30%
    """
    y_true = np.array([0]*50 + [1]*50)
    # 0s: 40 predicted 0 (TN), 10 predicted 1 (FP)
    # 1s: 20 predicted 0 (FN), 30 predicted 1 (TP)
    y_pred = np.array([0]*40 + [1]*10 + [0]*20 + [1]*30)

    res = compute_classification_metrics(y_true, y_pred)
    cm = res["confusion_matrix"]
    assert cm["TN"] == 40
    assert cm["FP"] == 10
    assert cm["FN"] == 20
    assert cm["TP"] == 30

    assert np.isclose(res["raw"]["accuracy"], 0.70)
    assert np.isclose(res["raw"]["precision"], 0.75)
    assert np.isclose(res["raw"]["sensitivity"], 0.60)
    assert np.isclose(res["raw"]["specificity"], 0.80)
    assert np.isclose(res["raw"]["f1_score"], 2 * 0.75 * 0.60 / (0.75 + 0.60))
    assert np.isclose(res["raw"]["error_rate"], 0.30)
    assert res["percentages"]["accuracy_pct"] == 70.0


def test_classification_zero_division_safety():
    """Tests edge cases where precision or recall has zero denominator."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])  # TP=0, FP=0 -> Precision denominator is 0
    res = compute_classification_metrics(y_true, y_pred)
    assert res["raw"]["precision"] == 0.0
    assert res["raw"]["sensitivity"] == 0.0
    assert res["raw"]["f1_score"] == 0.0
    assert res["raw"]["specificity"] == 1.0
