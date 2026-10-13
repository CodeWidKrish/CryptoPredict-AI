"""
Preprocessing and Chronological Data Partitioning Module.
Strict time-series splits, isolated scaler fitting, and TimeSeriesSplit CV.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    REGRESSION_TARGET,
    RETURN_TARGET,
    CLASSIFICATION_TARGET
)


class DataSplitter:
    """
    Handles strictly chronological data splitting to guarantee zero future lookahead bias.
    """
    def __init__(self, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """
        Splits dataset strictly chronologically into:
        - Train (70%)
        - Validation (15%)
        - Test (15%)
        """
        df_sorted = df.copy()
        df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])
        df_sorted = df_sorted.sort_values("Date").reset_index(drop=True)
        if RETURN_TARGET not in df_sorted.columns and REGRESSION_TARGET in df_sorted.columns:
            df_sorted[RETURN_TARGET] = (df_sorted[REGRESSION_TARGET] - df_sorted["Close"]) / df_sorted["Close"]
        n = len(df_sorted)

        train_end_idx = int(n * self.train_ratio)
        val_end_idx = int(n * (self.train_ratio + self.val_ratio))

        train_df = df_sorted.iloc[:train_end_idx].copy()
        val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy()
        test_df = df_sorted.iloc[val_end_idx:].copy()

        # Fit Scaler strictly on Train fold only
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(train_df[feature_cols]),
            columns=feature_cols,
            index=train_df.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(val_df[feature_cols]),
            columns=feature_cols,
            index=val_df.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(test_df[feature_cols]),
            columns=feature_cols,
            index=test_df.index
        )

        metadata = {
            "total_samples": n,
            "train": {
                "samples": len(train_df),
                "ratio": round(len(train_df) / n, 4),
                "start_date": str(train_df["Date"].min().date()),
                "end_date": str(train_df["Date"].max().date())
            },
            "validation": {
                "samples": len(val_df),
                "ratio": round(len(val_df) / n, 4),
                "start_date": str(val_df["Date"].min().date()),
                "end_date": str(val_df["Date"].max().date())
            },
            "test": {
                "samples": len(test_df),
                "ratio": round(len(test_df) / n, 4),
                "start_date": str(test_df["Date"].min().date()),
                "end_date": str(test_df["Date"].max().date())
            }
        }

        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "X_train": train_df[feature_cols],
            "X_val": val_df[feature_cols],
            "X_test": test_df[feature_cols],
            "X_train_scaled": X_train_scaled,
            "X_val_scaled": X_val_scaled,
            "X_test_scaled": X_test_scaled,
            "y_reg_train": train_df[REGRESSION_TARGET],
            "y_reg_val": val_df[REGRESSION_TARGET],
            "y_reg_test": test_df[REGRESSION_TARGET],
            "y_return_train": train_df[RETURN_TARGET],
            "y_return_val": val_df[RETURN_TARGET],
            "y_return_test": test_df[RETURN_TARGET],
            "y_clf_train": train_df[CLASSIFICATION_TARGET].astype(int),
            "y_clf_val": val_df[CLASSIFICATION_TARGET].astype(int),
            "y_clf_test": test_df[CLASSIFICATION_TARGET].astype(int),
            "scaler": scaler,
            "feature_cols": feature_cols,
            "metadata": metadata
        }

    @staticmethod
    def get_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
        """Returns scikit-learn TimeSeriesSplit instance."""
        return TimeSeriesSplit(n_splits=n_splits)
