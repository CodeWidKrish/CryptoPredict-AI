"""
Classification Training and Tuning Pipeline for CryptoPredict AI.
Trains, tunes, and compares market-direction classification models.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MODELS_DIR, RANDOM_SEED
from src.utils import set_seed
from src.evaluation import compute_classification_metrics


def tune_xgboost_classifier(X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int = 3) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Tunes XGBoost Classifier using TimeSeriesSplit cross-validation on Training fold strictly.
    """
    print("[Train Classification] Tuning XGBoost Classifier with TimeSeriesSplit CV...")
    param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0.0, 1.0],
        "reg_lambda": [1.0, 5.0]
    }

    base_model = xgb.XGBClassifier(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss"
    )

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1",
        cv=tscv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_model = search.best_estimator_
    print(f"[Train Classification] Best XGBoost parameters: {best_params}")
    return best_model, best_params


def train_classification_models(split_data: Dict[str, Any]) -> Tuple[pd.DataFrame, Any, Dict[str, Any], Dict[str, Any]]:
    """
    Trains candidate classification models, evaluates on chronological Validation and Test sets,
    and saves the winning model.
    """
    set_seed(RANDOM_SEED)

    X_train = split_data["X_train"]
    X_val = split_data["X_val"]
    X_test = split_data["X_test"]

    # Scaled inputs for logistic regression
    X_train_sc = split_data["X_train_scaled"]
    X_val_sc = split_data["X_val_scaled"]
    X_test_sc = split_data["X_test_scaled"]

    y_train = split_data["y_clf_train"]
    y_val = split_data["y_clf_val"]
    y_test = split_data["y_clf_test"]

    # Tune XGBoost Classifier on train set
    best_xgb, best_xgb_params = tune_xgboost_classifier(X_train, y_train)

    # Candidate models
    models = {
        "Dummy Baseline (Most Frequent)": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": (LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_SEED), True),
        "Random Forest Classifier": (RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1), False),
        "Extra Trees Classifier": (ExtraTreesClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1), False),
        "LightGBM Classifier": (lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=RANDOM_SEED, verbose=-1), False),
        "HistGradientBoosting Classifier": (HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=4, random_state=RANDOM_SEED), False),
        "XGBoost Classifier (Tuned)": (best_xgb, False)
    }

    leaderboard_records = []
    val_predictions = {}
    test_predictions = {}
    test_probabilities = {}
    trained_estimators = {}

    print("\n" + "="*60)
    print("CLASSIFICATION MODEL EVALUATION (Chronological Validation & Test)")
    print("="*60)

    for name, (model, requires_scaling) in models.items():
        X_tr = X_train_sc if requires_scaling else X_train
        X_v = X_val_sc if requires_scaling else X_val
        X_te = X_test_sc if requires_scaling else X_test

        # Fit
        model.fit(X_tr, y_train)
        trained_estimators[name] = model

        # Predict
        p_val = model.predict(X_v)
        p_test = model.predict(X_te)
        p_train = model.predict(X_tr)

        # Probabilities if available
        prob_val = model.predict_proba(X_v)[:, 1] if hasattr(model, "predict_proba") else None
        prob_test = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

        val_predictions[name] = p_val
        test_predictions[name] = p_test
        test_probabilities[name] = prob_test

        # Metrics
        val_m = compute_classification_metrics(y_val, p_val, prob_val)
        test_m = compute_classification_metrics(y_test, p_test, prob_test)
        train_m = compute_classification_metrics(y_train, p_train)

        leaderboard_records.append({
            "Model": name,
            "Val_F1_Pct": val_m["percentages"]["f1_score_pct"],
            "Val_Accuracy_Pct": val_m["percentages"]["accuracy_pct"],
            "Val_Precision_Pct": val_m["percentages"]["precision_pct"],
            "Val_Recall_Pct": val_m["percentages"]["sensitivity_pct"],
            "Val_Specificity_Pct": val_m["percentages"]["specificity_pct"],
            "Val_ErrorRate_Pct": val_m["percentages"]["error_rate_pct"],
            "Val_ROC_AUC": val_m["raw"]["roc_auc"],
            "Test_F1_Pct": test_m["percentages"]["f1_score_pct"],
            "Test_Accuracy_Pct": test_m["percentages"]["accuracy_pct"],
            "Test_Precision_Pct": test_m["percentages"]["precision_pct"],
            "Test_Recall_Pct": test_m["percentages"]["sensitivity_pct"],
            "Test_Specificity_Pct": test_m["percentages"]["specificity_pct"],
            "Test_ErrorRate_Pct": test_m["percentages"]["error_rate_pct"],
            "Test_ROC_AUC": test_m["raw"]["roc_auc"],
            "Train_Accuracy_Pct": train_m["percentages"]["accuracy_pct"]
        })

        print(f"[{name}]")
        print(f"  Val  -> Acc: {val_m['percentages']['accuracy_pct']}% | F1: {val_m['percentages']['f1_score_pct']}% | Prec: {val_m['percentages']['precision_pct']}% | Spec: {val_m['percentages']['specificity_pct']}% | ErrRate: {val_m['percentages']['error_rate_pct']}%")
        print(f"  Test -> Acc: {test_m['percentages']['accuracy_pct']}% | F1: {test_m['percentages']['f1_score_pct']}% | Prec: {test_m['percentages']['precision_pct']}% | Spec: {test_m['percentages']['specificity_pct']}% | ErrRate: {test_m['percentages']['error_rate_pct']}%")

    leaderboard_df = pd.DataFrame(leaderboard_records)
    # Sort candidate models by Validation Accuracy (and Validation F1)
    leaderboard_df = leaderboard_df.sort_values(["Val_Accuracy_Pct", "Val_F1_Pct"], ascending=[False, False]).reset_index(drop=True)

    # Exclude dummy baseline from winning candidate selection
    real_candidates = leaderboard_df[~leaderboard_df["Model"].str.contains("Dummy", case=False)]
    if len(real_candidates) > 0:
        best_model_name = real_candidates.iloc[0]["Model"]
    else:
        best_model_name = leaderboard_df.iloc[0]["Model"]

    best_estimator = trained_estimators[best_model_name]
    best_test_metrics = compute_classification_metrics(y_test, test_predictions[best_model_name], test_probabilities[best_model_name])
    best_val_metrics = compute_classification_metrics(y_val, val_predictions[best_model_name])

    print(f"\n[WINNING CLASSIFICATION MODEL (Selected from ML candidates by Validation Performance)]: {best_model_name}")
    print(f"Validation Accuracy: {best_val_metrics['percentages']['accuracy_pct']}% | Test Accuracy: {best_test_metrics['percentages']['accuracy_pct']}%")
    print(f"Test Confusion Matrix: {best_test_metrics['confusion_matrix']}")

    # Save artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_clf_path = MODELS_DIR / "best_classification_model.joblib"
    joblib.dump({
        "model_name": best_model_name,
        "model": best_estimator,
        "feature_cols": split_data["feature_cols"],
        "requires_scaling": models[best_model_name][1],
        "scaler": split_data["scaler"] if models[best_model_name][1] else None,
        "best_xgb_params": best_xgb_params if "XGBoost" in best_model_name else None
    }, best_clf_path)
    print(f"[Train Classification] Saved winning model artifact to {best_clf_path}")

    clf_summary = {
        "winning_model": best_model_name,
        "validation_metrics": best_val_metrics,
        "test_metrics": best_test_metrics,
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "best_xgb_hyperparameters": best_xgb_params
    }

    return leaderboard_df, best_estimator, clf_summary, test_predictions, test_probabilities
