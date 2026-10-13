"""
Evaluation Module for CryptoPredict AI.
Computes rigorous, mathematically exact regression and classification metrics.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    brier_score_loss
)


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    base_price: Optional[pd.Series | np.ndarray] = None,
    current_close: Optional[pd.Series | np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculates comprehensive regression metrics:
    - MAE (USD)
    - MSE
    - RMSE (USD)
    - MAPE (%) with safe epsilon denominator handling
    - R² Score
    - Directional / Sign Accuracy (%) if base_price or current_close (Close[t]) is provided
    """
    if base_price is None and current_close is not None:
        base_price = current_close
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

    # Directional Accuracy (Sign Agreement)
    directional_acc = None
    if base_price is not None:
        b_p = np.asarray(base_price, dtype=np.float64)
        actual_delta = y_t - b_p
        predicted_delta = y_p - b_p
        actual_sign = np.sign(actual_delta)
        pred_sign = np.sign(predicted_delta)
        # Treat non-negative as up
        correct_direction = (actual_sign * pred_sign >= 0)
        directional_acc = float(np.mean(correct_direction) * 100.0)

    res = {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 4),
        "r2_score": round(r2, 6),
        "directional_accuracy_pct": round(directional_acc, 2) if directional_acc is not None else None,
        "formatted": {
            "MAE": f"${mae:,.2f}",
            "RMSE": f"${rmse:,.2f}",
            "MAPE": f"{mape:.2f}%",
            "R²": f"{r2:.4f}"
        }
    }
    if directional_acc is not None:
        res["formatted"]["Directional Accuracy"] = f"{directional_acc:.2f}%"

    return res


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: Optional[pd.Series | np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculates mathematically exact classification metrics:
    - Confusion Matrix: TN, FP, FN, TP
    - Accuracy = (TP + TN) / Total
    - Precision = TP / (TP + FP)
    - Sensitivity / Recall = TP / (TP + FN)
    - Specificity = TN / (TN + FP)
    - Balanced Accuracy = (Sensitivity + Specificity) / 2
    - Matthews Correlation Coefficient (MCC)
    - F1-Score = 2 * Precision * Recall / (Precision + Recall)
    - Error Rate = (FP + FN) / Total
    - ROC-AUC & PR-AUC (if probabilities provided)
    - Brier Score (Calibration loss)
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
    balanced_acc = (sensitivity + specificity) / 2.0

    if (precision + sensitivity) > 0:
        f1 = 2.0 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1 = 0.0

    error_rate = (fp + fn) / total if total > 0 else 0.0

    # Matthews Correlation Coefficient
    try:
        mcc = float(matthews_corrcoef(y_t, y_p))
    except Exception:
        mcc = 0.0

    roc_auc = None
    pr_auc = None
    brier = None

    if y_prob is not None:
        try:
            if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                prob_pos = y_prob[:, 1]
            else:
                prob_pos = y_prob
            roc_auc = float(roc_auc_score(y_t, prob_pos))
            pr_auc = float(average_precision_score(y_t, prob_pos))
            brier = float(brier_score_loss(y_t, prob_pos))
        except Exception:
            pass

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
            "balanced_accuracy": float(balanced_acc),
            "mcc": float(mcc),
            "f1_score": float(f1),
            "error_rate": float(error_rate),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "pr_auc": float(pr_auc) if pr_auc is not None else None,
            "brier_score": float(brier) if brier is not None else None
        },
        "percentages": {
            "accuracy_pct": round(accuracy * 100.0, 2),
            "precision_pct": round(precision * 100.0, 2),
            "sensitivity_pct": round(sensitivity * 100.0, 2),
            "specificity_pct": round(specificity * 100.0, 2),
            "balanced_accuracy_pct": round(balanced_acc * 100.0, 2),
            "f1_score_pct": round(f1 * 100.0, 2),
            "error_rate_pct": round(error_rate * 100.0, 2),
            "roc_auc_pct": round(roc_auc * 100.0, 2) if roc_auc is not None else None,
            "pr_auc_pct": round(pr_auc * 100.0, 2) if pr_auc is not None else None
        },
        "formatted": {
            "Accuracy": f"{accuracy * 100.0:.2f}%",
            "Precision": f"{precision * 100.0:.2f}%",
            "Recall / Sensitivity": f"{sensitivity * 100.0:.2f}%",
            "Specificity": f"{specificity * 100.0:.2f}%",
            "Balanced Accuracy": f"{balanced_acc * 100.0:.2f}%",
            "MCC": f"{mcc:.4f}",
            "F1-Score": f"{f1 * 100.0:.2f}%",
            "Error Rate": f"{error_rate * 100.0:.2f}%",
            "ROC-AUC": f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
            "PR-AUC": f"{pr_auc:.4f}" if pr_auc is not None else "N/A",
            "Brier Score": f"{brier:.4f}" if brier is not None else "N/A"
        }
    }


def compute_threshold_curve(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    thresholds: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates classification performance across multiple probability decision thresholds.
    """
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.30, 0.75, 0.05)]

    y_t = np.asarray(y_true, dtype=int)
    if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
        prob_pos = y_prob[:, 1]
    else:
        prob_pos = np.asarray(y_prob, dtype=np.float64)

    records = []
    for thresh in thresholds:
        y_pred = (prob_pos >= thresh).astype(int)
        cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = (rec + spec) / 2.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        records.append({
            "threshold": float(thresh),
            "accuracy_pct": round(acc * 100.0, 2),
            "precision_pct": round(prec * 100.0, 2),
            "recall_pct": round(rec * 100.0, 2),
            "specificity_pct": round(spec * 100.0, 2),
            "balanced_accuracy_pct": round(bal_acc * 100.0, 2),
            "f1_score_pct": round(f1 * 100.0, 2),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        })

    return records
