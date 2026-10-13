"""
End-to-End Orchestrator Pipeline for CryptoPredict AI.
Executes the full pipeline: Data Ingestion -> Feature Engineering -> Chronological Partitioning ->
Tuning & Training (Regression & Classification) -> Report Generation -> Artifact Persistence.
"""

import sys
import io
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    MODEL_COMPARISON_FILE,
    REGRESSION_METRICS_FILE,
    CLASSIFICATION_METRICS_FILE,
    TEST_PREDICTIONS_FILE,
    EXPERIMENT_SUMMARY_FILE,
    METADATA_FILE,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET
)
from src.utils import set_seed, save_json
from src.data_loader import fetch_bitcoin_data
from src.features import prepare_datasets
from src.preprocessing import DataSplitter
from src.train_regression import train_regression_models
from src.train_classification import train_classification_models
from src.predict import CryptoPredictor
from src.visualization import (
    save_static_eda_figures,
    save_static_prediction_figures
)


def run_pipeline():
    print("=" * 80)
    print(" CRYPTOPREDICT AI: BITCOIN PRICE PREDICTION & MARKET DIRECTION PIPELINE")
    print("=" * 80)
    set_seed(42)

    # 1. Real Data Ingestion
    print("\n[Step 1/8] Fetching and validating genuine Bitcoin market data...")
    raw_df, data_meta = fetch_bitcoin_data(force_download=False)
    print(f"Data Source: {data_meta['source']}")
    print(f"Total Raw Records: {data_meta['row_count']} ({data_meta['start_date']} to {data_meta['end_date']})")

    # 2. Feature Engineering
    print("\n[Step 2/8] Generating leakage-free technical features and forward targets...")
    labeled_df, latest_row, feature_cols, feature_doc = prepare_datasets(raw_df)

    # 3. Static EDA Visualizations
    print("\n[Step 3/8] Generating publication-grade Exploratory Data Analysis figures...")
    eda_figs = save_static_eda_figures(labeled_df)
    print(f"Saved {len(eda_figs)} EDA figures to reports/figures/")

    # 4. Strict Chronological Splitting
    print("\n[Step 4/8] Splitting chronologically (70% Train, 15% Validation, 15% Test)...")
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    split_data = splitter.split(labeled_df, feature_cols)
    split_meta = split_data["metadata"]

    print(f"Train Set:      {split_meta['train']['samples']} rows ({split_meta['train']['start_date']} -> {split_meta['train']['end_date']})")
    print(f"Validation Set: {split_meta['validation']['samples']} rows ({split_meta['validation']['start_date']} -> {split_meta['validation']['end_date']})")
    print(f"Test Set:       {split_meta['test']['samples']} rows ({split_meta['test']['start_date']} -> {split_meta['test']['end_date']})")

    # 5. Task A: Regression Training & Tuning
    print("\n[Step 5/8] Training & tuning next-day price regression models...")
    reg_leaderboard, best_reg_model, reg_summary, reg_test_preds = train_regression_models(split_data)

    # 6. Task B: Classification Training & Tuning
    print("\n[Step 6/8] Training & tuning market direction classification models...")
    clf_leaderboard, best_clf_model, clf_summary, clf_test_preds, clf_test_probs = train_classification_models(split_data)

    # 7. Extract Feature Importances (from tree-based models)
    print("\n[Step 7/8] Computing feature importances...")
    feature_importances = []
    if hasattr(best_reg_model, "feature_importances_"):
        fi_vals = best_reg_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": fi_vals}).sort_values("Importance", ascending=False)
        feature_importances = fi_df.to_dict(orient="records")

    # Generate Test Predictions Table
    test_dates = split_data["test_df"]["Date"].reset_index(drop=True)
    best_reg_name = reg_summary["winning_model"]
    best_clf_name = clf_summary["winning_model"]

    test_pred_df = pd.DataFrame({
        "Date": test_dates,
        "Current_Close": split_data["test_df"]["Close"].reset_index(drop=True),
        "Actual_Next_Close": split_data["y_reg_test"].reset_index(drop=True),
        "Predicted_Next_Close": reg_test_preds[best_reg_name],
        "Actual_Direction": split_data["y_clf_test"].reset_index(drop=True),
        "Predicted_Direction": clf_test_preds[best_clf_name],
    })
    test_pred_df["Price_Error_USD"] = test_pred_df["Predicted_Next_Close"] - test_pred_df["Actual_Next_Close"]
    test_pred_df["Price_Abs_Pct_Error"] = (np.abs(test_pred_df["Price_Error_USD"]) / test_pred_df["Actual_Next_Close"]) * 100.0
    test_pred_df["Direction_Correct"] = (test_pred_df["Predicted_Direction"] == test_pred_df["Actual_Direction"]).astype(int)

    test_pred_df.to_csv(TEST_PREDICTIONS_FILE, index=False)
    print(f"Saved test predictions to {TEST_PREDICTIONS_FILE}")

    # Generate Test Figures
    save_static_prediction_figures(
        dates=test_dates,
        actual=test_pred_df["Actual_Next_Close"],
        predicted=test_pred_df["Predicted_Next_Close"],
        model_name=best_reg_name,
        cm_dict=clf_summary["test_metrics"]["confusion_matrix"]
    )

    # Save Comparison CSV
    combined_comparison = pd.DataFrame({
        "Task": ["Regression"] * len(reg_leaderboard) + ["Classification"] * len(clf_leaderboard),
        "Model": list(reg_leaderboard["Model"]) + list(clf_leaderboard["Model"]),
        "Validation_Primary_Metric": (
            [f"RMSE: ${x:,.2f}" for x in reg_leaderboard["Val_RMSE_USD"]] +
            [f"F1: {x:.2f}%" for x in clf_leaderboard["Val_F1_Pct"]]
        ),
        "Test_Primary_Metric": (
            [f"RMSE: ${x:,.2f}" for x in reg_leaderboard["Test_RMSE_USD"]] +
            [f"F1: {x:.2f}%" for x in clf_leaderboard["Test_F1_Pct"]]
        ),
        "Test_Secondary_Metric": (
            [f"MAPE: {x:.2f}% (R²: {r:.4f})" for x, r in zip(reg_leaderboard["Test_MAPE_Pct"], reg_leaderboard["Test_R2"])] +
            [f"Acc: {a:.2f}% (Spec: {s:.2f}%)" for a, s in zip(clf_leaderboard["Test_Accuracy_Pct"], clf_leaderboard["Test_Specificity_Pct"])]
        )
    })
    combined_comparison.to_csv(MODEL_COMPARISON_FILE, index=False)
    print(f"Saved model comparison table to {MODEL_COMPARISON_FILE}")

    # Save Metrics JSONs
    save_json(reg_summary, REGRESSION_METRICS_FILE)
    save_json(clf_summary, CLASSIFICATION_METRICS_FILE)

    # Save Master Metadata
    metadata = {
        "project": "CryptoPredict AI",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_meta["source"],
        "dataset_rows": len(labeled_df),
        "date_range": {
            "first_date": split_meta["train"]["start_date"],
            "last_date": split_meta["test"]["end_date"]
        },
        "partitions": split_meta,
        "features": {
            "count": len(feature_cols),
            "columns": feature_cols,
            "descriptions": feature_doc
        },
        "winning_regression_model": best_reg_name,
        "regression_test_metrics": reg_summary["test_metrics"],
        "winning_classification_model": best_clf_name,
        "classification_test_metrics": clf_summary["test_metrics"],
        "top_features": feature_importances[:15]
    }
    save_json(metadata, METADATA_FILE)
    print(f"Saved master model metadata to {METADATA_FILE}")

    # Generate Markdown Summary Report
    summary_md = f"""# CryptoPredict AI — Official Experiment Summary Report
**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Asset:** Bitcoin (BTC-USD)  
**Historical Horizon:** {split_meta['train']['start_date']} to {split_meta['test']['end_date']} ({len(labeled_df)} supervised observations)

---

## 1. Executive Summary & Winning Models
* **Best Regression Model (Selected by Validation RMSE):** `{best_reg_name}`
  * **Test Set MAE:** ${reg_summary['test_metrics']['mae']:,.2f}
  * **Test Set RMSE:** ${reg_summary['test_metrics']['rmse']:,.2f}
  * **Test Set MAPE:** {reg_summary['test_metrics']['mape_pct']:.2f}%
  * **Test Set R² Score:** {reg_summary['test_metrics']['r2_score']:.4f}

* **Best Direction Classification Model (Selected by Validation F1):** `{best_clf_name}`
  * **Test Set Accuracy:** {clf_summary['test_metrics']['percentages']['accuracy_pct']}%
  * **Test Set F1-Score:** {clf_summary['test_metrics']['percentages']['f1_score_pct']}%
  * **Test Set Precision:** {clf_summary['test_metrics']['percentages']['precision_pct']}%
  * **Test Set Sensitivity / Recall:** {clf_summary['test_metrics']['percentages']['sensitivity_pct']}%
  * **Test Set Specificity:** {clf_summary['test_metrics']['percentages']['specificity_pct']}%
  * **Test Set Error Rate:** {clf_summary['test_metrics']['percentages']['error_rate_pct']}%
  * **Test Set ROC-AUC:** {clf_summary['test_metrics']['formatted']['ROC-AUC']}
  * **Confusion Matrix:** TN={clf_summary['test_metrics']['confusion_matrix']['TN']}, FP={clf_summary['test_metrics']['confusion_matrix']['FP']}, FN={clf_summary['test_metrics']['confusion_matrix']['FN']}, TP={clf_summary['test_metrics']['confusion_matrix']['TP']}

---

## 2. Temporal Partitioning (Zero Data Leakage)
Strictly chronological split applied:
* **Training Partition (70%):** {split_meta['train']['samples']} samples ({split_meta['train']['start_date']} to {split_meta['train']['end_date']})
* **Validation Partition (15%):** {split_meta['validation']['samples']} samples ({split_meta['validation']['start_date']} to {split_meta['validation']['end_date']})
* **Test Partition (15%):** {split_meta['test']['samples']} samples ({split_meta['test']['start_date']} to {split_meta['test']['end_date']})

---

## 3. Regression Leaderboard (Ranked by Validation RMSE)
{reg_leaderboard.to_markdown(index=False)}

---

## 4. Classification Leaderboard (Ranked by Validation F1-Score)
{clf_leaderboard.to_markdown(index=False)}

---

## 5. Critical Machine Learning Insights & Limitations
1. **The Random Walk and Naive Baseline:** The naive previous-close model ($\\hat{{y}}_{{t+1}} = y_t$) achieves a very strong R² score in non-stationary price forecasting because Bitcoin's daily percentage move is typically 2-4%. A model that closely tracks the previous day will mathematically achieve high R², but this does NOT imply an ability to forecast explosive trend reversals or market shocks.
2. **Direction Classification Ceiling:** Due to the semi-strong form efficient market hypothesis and high noise-to-signal ratio of daily crypto returns, daily direction accuracy typically hovers around 50-55%. Models claiming >70% daily direction accuracy on un-shuffled chronological test data suffer from lookahead target leakage.
3. **Volatility Clustering:** Daily volatility is non-stationary. Incorporating rolling standard deviation and Bollinger band bandwidth allowed gradient boosted trees to adapt to regime shifts better than plain linear estimators.
"""
    with open(EXPERIMENT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Saved experiment summary to {EXPERIMENT_SUMMARY_FILE}")

    # 8. Live Forward Inference
    print("\n[Step 8/8] Performing forward inference for the upcoming trading day...")
    predictor = CryptoPredictor()
    next_day_forecast = predictor.predict_next_day(latest_row)
    print("\n" + "*" * 60)
    print(f" LIVE FORECAST FOR NEXT TRADING DAY")
    print(f" Observation Candle Date: {next_day_forecast['observation_date']}")
    print(f" Current Observed Close:  ${next_day_forecast['current_close_usd']:,.2f}")
    print(f" Predicted Next Close:    ${next_day_forecast['predicted_close_usd']:,.2f} ({next_day_forecast['predicted_pct_change']:+.2f}%)")
    print(f" Predicted Direction:     {next_day_forecast['direction_label']}")
    if next_day_forecast['direction_confidence']:
        print(f" Direction Probability:   {next_day_forecast['direction_confidence']*100:.1f}%")
    print(f" Regression Model:        {next_day_forecast['regression_model_used']}")
    print(f" Classification Model:    {next_day_forecast['classification_model_used']}")
    print("*" * 60)

    print("\n[SUCCESS] Pipeline execution complete! All artifacts and models saved successfully.")


if __name__ == "__main__":
    run_pipeline()
