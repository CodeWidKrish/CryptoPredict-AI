"""
Inference Module for CryptoPredict AI.
Handles model loading, schema validation, next-day forecasting, and historical inspection.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from config.config import MODELS_DIR, EXPERIMENTS_DIR


class CryptoPredictor:
    """
    Production inference engine for CryptoPredict AI.
    Loads validated serialized artifacts and generates safe, leakage-free predictions.
    """
    def __init__(self):
        self.reg_artifact_path = MODELS_DIR / "best_regression_model.joblib"
        self.clf_artifact_path = MODELS_DIR / "best_classification_model.joblib"

        if not self.reg_artifact_path.exists() or not self.clf_artifact_path.exists():
            raise FileNotFoundError(
                f"Trained model artifacts not found in {MODELS_DIR}. "
                "Please run the training pipeline first."
            )

        self.reg_bundle = joblib.load(self.reg_artifact_path)
        self.clf_bundle = joblib.load(self.clf_artifact_path)

        self.reg_model = self.reg_bundle["model"]
        self.reg_model_name = self.reg_bundle["model_name"]
        self.reg_features = self.reg_bundle["feature_cols"]
        self.reg_scaler = self.reg_bundle.get("scaler")
        self.reg_scaled = self.reg_bundle.get("requires_scaling", False)

        self.clf_model = self.clf_bundle["model"]
        self.clf_model_name = self.clf_bundle["model_name"]
        self.clf_features = self.clf_bundle["feature_cols"]
        self.clf_scaler = self.clf_bundle.get("scaler")
        self.clf_scaled = self.clf_bundle.get("requires_scaling", False)

        self.interval_path = EXPERIMENTS_DIR / "prediction_intervals.json"
        self.pi_info = None
        if self.interval_path.exists():
            try:
                with open(self.interval_path, "r", encoding="utf-8") as f:
                    self.pi_info = json.load(f)
            except Exception:
                self.pi_info = None

    def _prepare_features(self, df_row: pd.DataFrame, feature_cols: list, scaler: Optional[Any], requires_scaling: bool) -> pd.DataFrame:
        """Validates feature alignment and applies scaler if required."""
        missing = set(feature_cols) - set(df_row.columns)
        if missing:
            raise ValueError(f"Inference input missing required features: {missing}")

        X = df_row[feature_cols].copy()
        if requires_scaling and scaler is not None:
            X_scaled = scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        return X

    def predict_next_day(self, latest_row: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates next-day Bitcoin price and direction prediction for the most recent completed candle.
        """
        date_val = str(pd.to_datetime(latest_row["Date"].values[0]).date())
        current_close = float(latest_row["Close"].values[0])

        # Prepare regression input
        X_reg = self._prepare_features(latest_row, self.reg_features, self.reg_scaler, self.reg_scaled)
        pred_close = float(self.reg_model.predict(X_reg)[0])

        # Prepare classification input
        X_clf = self._prepare_features(latest_row, self.clf_features, self.clf_scaler, self.clf_scaled)
        pred_dir = int(self.clf_model.predict(X_clf)[0])

        dir_prob = None
        if hasattr(self.clf_model, "predict_proba"):
            probs = self.clf_model.predict_proba(X_clf)[0]
            dir_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

        delta_usd = pred_close - current_close
        pct_change = (delta_usd / (current_close + 1e-8)) * 100.0

        direction_label = "BULLISH / UP [UP]" if pred_dir == 1 else "BEARISH / DOWN [DOWN]"

        # Conformal prediction interval & uncertainty quantification
        interval_data = {}
        if self.pi_info is not None:
            try:
                q90 = self.pi_info["quantiles_usd"]["q90_usd"]
                q95 = self.pi_info["quantiles_usd"]["q95_usd"]
                interval_data = {
                    "interval_90_usd": [round(pred_close - q90, 2), round(pred_close + q90, 2)],
                    "interval_95_usd": [round(pred_close - q95, 2), round(pred_close + q95, 2)],
                    "margin_90_usd": q90,
                    "margin_95_usd": q95,
                    "volatility_regime": self.pi_info.get("volatility_regime", {}).get("current_status", "Normal Volatility")
                }
            except Exception:
                pass

        return {
            "observation_date": date_val,
            "target_prediction_date": "Next Trading Day",
            "current_close_usd": current_close,
            "predicted_close_usd": pred_close,
            "predicted_delta_usd": delta_usd,
            "predicted_pct_change": pct_change,
            "naive_baseline_usd": current_close,
            "model_vs_naive_delta_usd": delta_usd,
            "predicted_direction": pred_dir,
            "direction_label": direction_label,
            "direction_confidence": dir_prob,
            "prediction_intervals": interval_data,
            "regression_model_used": self.reg_model_name,
            "classification_model_used": self.clf_model_name,
            "probability_interpretation": (
                "Predicted probability is the model's calibrated scoring under historical feature patterns, "
                "not a guarantee of future certainty. Time-series markets remain subject to unexpected regime shocks."
            ),
            "disclaimer": (
                "DISCLAIMER: Cryptocurrency price predictions are inherently volatile, speculative, and uncertain. "
                "This output is generated by an academic machine learning model strictly for educational and research purposes. "
                "It does NOT constitute financial or investment advice."
            )
        }

    def predict_historical_date(self, df_featured: pd.DataFrame, target_date_str: str) -> Dict[str, Any]:
        """
        Runs inference on a specific historical date to inspect what the models would have forecasted
        using ONLY information available up to that timestamp.
        """
        df_featured = df_featured.copy()
        df_featured["Date_Str"] = pd.to_datetime(df_featured["Date"]).dt.strftime("%Y-%m-%d")
        matched = df_featured[df_featured["Date_Str"] == target_date_str]

        if len(matched) == 0:
            raise ValueError(f"Date {target_date_str} not found in feature dataset.")

        matched_idx = matched.index[0]
        row = df_featured.iloc[[matched_idx]]
        pred_result = self.predict_next_day(row)

        # Look up actual ground truth if available in dataset
        if matched_idx + 1 < len(df_featured):
            next_row = df_featured.iloc[matched_idx + 1]
            actual_next_close = float(next_row["Close"])
            actual_next_date = str(pd.to_datetime(next_row["Date"]).date())
            actual_dir = 1 if actual_next_close > pred_result["current_close_usd"] else 0

            error_usd = pred_result["predicted_close_usd"] - actual_next_close
            abs_pct_error = (abs(error_usd) / actual_next_close) * 100.0
            direction_correct = (pred_result["predicted_direction"] == actual_dir)

            pred_result["actual_next_date"] = actual_next_date
            pred_result["actual_next_close_usd"] = actual_next_close
            pred_result["actual_next_direction"] = actual_dir
            pred_result["actual_direction_label"] = "UP [UP]" if actual_dir == 1 else "DOWN [DOWN]"
            pred_result["prediction_error_usd"] = error_usd
            pred_result["absolute_pct_error"] = abs_pct_error
            pred_result["direction_correct"] = direction_correct
        else:
            pred_result["actual_next_close_usd"] = None
            pred_result["actual_next_date"] = "Pending completion"

        return pred_result
