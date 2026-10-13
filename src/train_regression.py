"""
Regression Training and Tuning Pipeline for CryptoPredict AI.
Trains, tunes, and compares multiple regression models for next-day Bitcoin closing price.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MODELS_DIR, RANDOM_SEED
from src.utils import set_seed, save_json
from src.evaluation import compute_regression_metrics


class NaiveBaselineRegressor(BaseEstimator, RegressorMixin):
    """
    Naive baseline: Predicts tomorrow's closing price will equal today's closing price.
    Essential benchmark for time series forecasting.
    """
    def __init__(self, close_col_idx: int = 0):
        self.close_col_idx = close_col_idx

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if "Close" in X.columns:
                return X["Close"].to_numpy()
            return X.iloc[:, self.close_col_idx].to_numpy()
        return X[:, self.close_col_idx]


class ResidualXGBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Predicts next-day price delta (Close[t+1] - Close[t]) using XGBoost,
    then adds current Close[t] back. Solves tree non-extrapolation problem in non-stationary series.
    """
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model = xgb.XGBRegressor(**xgb_params)

    def fit(self, X, y):
        close_t = X["Close"].to_numpy() if isinstance(X, pd.DataFrame) else X[:, 0]
        y_delta = y.to_numpy() - close_t
        self.model.fit(X, y_delta)
        return self

    def predict(self, X):
        close_t = X["Close"].to_numpy() if isinstance(X, pd.DataFrame) else X[:, 0]
        delta_pred = self.model.predict(X)
        return close_t + delta_pred

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


def tune_xgboost_regressor(X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int = 3) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Tunes XGBoost Regressor using TimeSeriesSplit cross-validation on Training fold strictly.
    """
    print("[Train Regression] Tuning XGBoost hyperparameters with TimeSeriesSplit CV...")
    param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0.0, 1.0],
        "reg_lambda": [1.0, 5.0]
    }

    base_model = xgb.XGBRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        objective="reg:squarederror"
    )

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_model = search.best_estimator_
    print(f"[Train Regression] Best XGBoost parameters: {best_params}")
    return best_model, best_params


def train_regression_models(split_data: Dict[str, Any]) -> Tuple[pd.DataFrame, Any, Dict[str, Any], Dict[str, Any]]:
    """
    Trains all regression candidate models, ranks them using Validation RMSE,
    and conducts final unbiased evaluation on the Test set.
    """
    set_seed(RANDOM_SEED)

    X_train = split_data["X_train"]
    X_val = split_data["X_val"]
    X_test = split_data["X_test"]

    # Scaled inputs for linear models
    X_train_sc = split_data["X_train_scaled"]
    X_val_sc = split_data["X_val_scaled"]
    X_test_sc = split_data["X_test_scaled"]

    y_train = split_data["y_reg_train"]
    y_val = split_data["y_reg_val"]
    y_test = split_data["y_reg_test"]

    # Tune XGBoost on train set
    best_xgb, best_xgb_params = tune_xgboost_regressor(X_train, y_train)

    # Models dictionary
    models = {
        "Naive Previous-Close": (NaiveBaselineRegressor(), False),
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=10.0, random_state=RANDOM_SEED), True),
        "Random Forest Regressor": (RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1), False),
        "Extra Trees Regressor": (ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1), False),
        "LightGBM Regressor": (lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=RANDOM_SEED, verbose=-1), False),
        "HistGradientBoosting Regressor": (HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=5, random_state=RANDOM_SEED), False),
        "XGBoost Regressor (Raw Price)": (best_xgb, False),
        "XGBoost Regressor (Residual Delta)": (ResidualXGBoostRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05, random_state=RANDOM_SEED, n_jobs=-1
        ), False)
    }

    leaderboard_records = []
    val_predictions = {}
    test_predictions = {}
    trained_estimators = {}

    print("\n" + "="*60)
    print("REGRESSION MODEL EVALUATION (Chronological Validation & Test)")
    print("="*60)

    for name, (model, requires_scaling) in models.items():
        X_tr = X_train_sc if requires_scaling else X_train
        X_v = X_val_sc if requires_scaling else X_val
        X_te = X_test_sc if requires_scaling else X_test

        # Fit model
        model.fit(X_tr, y_train)
        trained_estimators[name] = model

        # Predict
        p_val = model.predict(X_v)
        p_test = model.predict(X_te)
        p_train = model.predict(X_tr)

        val_predictions[name] = p_val
        test_predictions[name] = p_test

        # Metrics
        train_m = compute_regression_metrics(y_train, p_train)
        val_m = compute_regression_metrics(y_val, p_val)
        test_m = compute_regression_metrics(y_test, p_test)

        leaderboard_records.append({
            "Model": name,
            "Val_RMSE_USD": val_m["rmse"],
            "Val_MAE_USD": val_m["mae"],
            "Val_MAPE_Pct": val_m["mape_pct"],
            "Val_R2": val_m["r2_score"],
            "Test_RMSE_USD": test_m["rmse"],
            "Test_MAE_USD": test_m["mae"],
            "Test_MAPE_Pct": test_m["mape_pct"],
            "Test_R2": test_m["r2_score"],
            "Train_RMSE_USD": train_m["rmse"],
            "Train_R2": train_m["r2_score"]
        })

        print(f"[{name}]")
        print(f"  Val  -> RMSE: ${val_m['rmse']:,.2f} | MAE: ${val_m['mae']:,.2f} | MAPE: {val_m['mape_pct']:.2f}% | R²: {val_m['r2_score']:.4f}")
        print(f"  Test -> RMSE: ${test_m['rmse']:,.2f} | MAE: ${test_m['mae']:,.2f} | MAPE: {test_m['mape_pct']:.2f}% | R²: {test_m['r2_score']:.4f}")

    leaderboard_df = pd.DataFrame(leaderboard_records)
    # Sort strictly by Validation RMSE (selection criterion based on validation, NOT test set)
    leaderboard_df = leaderboard_df.sort_values("Val_RMSE_USD").reset_index(drop=True)

    best_model_name = leaderboard_df.iloc[0]["Model"]
    best_estimator = trained_estimators[best_model_name]
    best_test_metrics = compute_regression_metrics(y_test, test_predictions[best_model_name])
    best_val_metrics = compute_regression_metrics(y_val, val_predictions[best_model_name])

    print(f"\n[WINNING REGRESSION MODEL (Selected by Validation RMSE)]: {best_model_name}")
    print(f"Validation RMSE: ${best_val_metrics['rmse']:,.2f} | Test RMSE: ${best_test_metrics['rmse']:,.2f}")

    # Persistence
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_reg_path = MODELS_DIR / "best_regression_model.joblib"
    joblib.dump({
        "model_name": best_model_name,
        "model": best_estimator,
        "feature_cols": split_data["feature_cols"],
        "requires_scaling": models[best_model_name][1],
        "scaler": split_data["scaler"] if models[best_model_name][1] else None,
        "best_xgb_params": best_xgb_params if "XGBoost" in best_model_name else None
    }, best_reg_path)
    print(f"[Train Regression] Saved winning model artifact to {best_reg_path}")

    # Save detailed regression report
    reg_summary = {
        "winning_model": best_model_name,
        "validation_metrics": best_val_metrics,
        "test_metrics": best_test_metrics,
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "best_xgb_hyperparameters": best_xgb_params
    }

    return leaderboard_df, best_estimator, reg_summary, test_predictions
