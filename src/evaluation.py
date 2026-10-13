"""
Evaluation Module for CryptoPredict AI.
Computes rigorous, mathematically exact regression and classification metrics.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    roc_auc_score
)


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray
) -> Dict[str, Any]:
    """
    Calculates comprehensive regression metrics:
    - MAE (USD)
    - MSE
    - RMSE (USD)
    - MAPE (%) with safe epsilon denominator handling
    - R² Score
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))

    # MAPE with safe zero handling
    denom = np.where(np.abs(y_t) < 1e-8, 1e-8, np.abs(y_t))
    mape = float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)

    # R2 Score
    r2 = float(r2_score(y_t, y_p))

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 4),
        "r2_score": round(r2, 6),
        "formatted": {
            "MAE": f"${mae:,.2f}",
            "RMSE": f"${rmse:,.2f}",
            "MAPE": f"{mape:.2f}%",
            "R²": f"{r2:.4f}"
        }
    }


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: Optional[pd.Series | np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculates mathematically exact classification metrics:
    - Confusion Matrix: TN, FP, FN, TP
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Precision = TP / (TP + FP)
    - Sensitivity / Recall = TP / (TP + FN)
    - Specificity = TN / (TN + FP)
    - F1 = 2 * Precision * Recall / (Precision + Recall)
    - Error Rate = (FP + FN) / (TP + TN + FP + FN)
    - ROC-AUC (if probabilities provided)
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if (precision + sensitivity) > 0:
        f1 = 2.0 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1 = 0.0

    error_rate = (fp + fn) / total if total > 0 else 0.0

    roc_auc = None
    if y_prob is not None:
        try:
            # Handle 1D probabilities or 2D array
            if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                prob_pos = y_prob[:, 1]
            else:
                prob_pos = y_prob
            roc_auc = float(roc_auc_score(y_t, prob_pos))
        except Exception:
            roc_auc = None

    return {
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "total": int(total)
        },
        "raw": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "error_rate": float(error_rate),
            "roc_auc": float(roc_auc) if roc_auc is not None else None
        },
        "percentages": {
            "accuracy_pct": round(accuracy * 100.0, 2),
            "precision_pct": round(precision * 100.0, 2),
            "sensitivity_pct": round(sensitivity * 100.0, 2),
            "specificity_pct": round(specificity * 100.0, 2),
            "f1_score_pct": round(f1 * 100.0, 2),
            "error_rate_pct": round(error_rate * 100.0, 2),
            "roc_auc_pct": round(roc_auc * 100.0, 2) if roc_auc is not None else None
        },
        "formatted": {
            "Accuracy": f"{accuracy * 100.0:.2f}%",
            "Precision": f"{precision * 100.0:.2f}%",
            "Recall / Sensitivity": f"{sensitivity * 100.0:.2f}%",
            "Specificity": f"{specificity * 100.0:.2f}%",
            "F1-Score": f"{f1 * 100.0:.2f}%",
            "Error Rate": f"{error_rate * 100.0:.2f}%",
            "ROC-AUC": f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        }
    }
