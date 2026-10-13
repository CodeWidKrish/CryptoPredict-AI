"""
Displays the complete experimental results, metrics, and forecasts.
"""

import sys
import io
import json
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

print("=" * 75)
print("  CRYPTOPREDICT AI: EXPERIMENTAL RESULTS & BENCHMARK SUMMARY")
print("=" * 75)

# Load Metadata
with open(MODELS_DIR / "model_metadata.json", "r") as f:
    meta = json.load(f)

print(f"\n[DATASET OVERVIEW]")
print(f"  Source:               {meta['data_source']}")
print(f"  Total Observations:   {meta['dataset_rows']} daily candles")
print(f"  Date Range:           {meta['date_range']['first_date']} -> {meta['date_range']['last_date']}")
print(f"  Engineered Features:  {meta['features']['count']} causal technical indicators")
print(f"  Temporal Partitions:  Train: {meta['partitions']['train']['samples']} | Val: {meta['partitions']['validation']['samples']} | Test: {meta['partitions']['test']['samples']}")

# Task A: Regression Leaderboard
print(f"\n[TASK A: NEXT-DAY PRICE REGRESSION LEADERBOARD]")
with open(REPORTS_DIR / "regression_metrics.json", "r") as f:
    reg_data = json.load(f)

reg_df = pd.DataFrame(reg_data["leaderboard"])
for col in ["Val_RMSE_USD", "Val_MAE_USD", "Test_RMSE_USD", "Test_MAE_USD"]:
    reg_df[col] = reg_df[col].apply(lambda x: f"${x:,.2f}")
for col in ["Val_MAPE_Pct", "Test_MAPE_Pct"]:
    reg_df[col] = reg_df[col].apply(lambda x: f"{x:.2f}%")
for col in ["Val_R2", "Test_R2"]:
    reg_df[col] = reg_df[col].apply(lambda x: f"{x:.4f}")

cols_to_show = ["Model", "Val_RMSE_USD", "Val_MAE_USD", "Val_R2", "Test_RMSE_USD", "Test_MAE_USD", "Test_MAPE_Pct", "Test_R2"]
print(reg_df[cols_to_show].to_string(index=False))

print(f"\n  Winning Regression Architecture: {reg_data['winning_model']}")
print(f"  Test Set MAE:  ${reg_data['test_metrics']['mae']:,.2f}")
print(f"  Test Set RMSE: ${reg_data['test_metrics']['rmse']:,.2f}")
print(f"  Test Set MAPE: {reg_data['test_metrics']['mape_pct']:.2f}%")
print(f"  Test Set R²:   {reg_data['test_metrics']['r2_score']:.4f}")

# Task B: Classification Leaderboard
print(f"\n[TASK B: MARKET DIRECTION CLASSIFICATION LEADERBOARD]")
with open(REPORTS_DIR / "classification_metrics.json", "r") as f:
    clf_data = json.load(f)

clf_df = pd.DataFrame(clf_data["leaderboard"])
clf_cols = ["Model", "Val_Accuracy_Pct", "Val_Specificity_Pct", "Val_F1_Pct", "Test_Accuracy_Pct", "Test_Specificity_Pct", "Test_Precision_Pct", "Test_F1_Pct"]
for c in ["Val_Accuracy_Pct", "Val_Specificity_Pct", "Val_F1_Pct", "Test_Accuracy_Pct", "Test_Specificity_Pct", "Test_Precision_Pct", "Test_F1_Pct"]:
    clf_df[c] = clf_df[c].apply(lambda x: f"{x:.2f}%")

print(clf_df[clf_cols].to_string(index=False))

cm = clf_data["test_metrics"]["confusion_matrix"]
print(f"\n  Winning Classification Model:  {clf_data['winning_model']}")
print(f"  Test Accuracy:                 {clf_data['test_metrics']['percentages']['accuracy_pct']}%")
print(f"  Test Specificity (Down Days):  {clf_data['test_metrics']['percentages']['specificity_pct']}%")
print(f"  Test Precision:                {clf_data['test_metrics']['percentages']['precision_pct']}%")
print(f"  Test F1-Score:                 {clf_data['test_metrics']['percentages']['f1_score_pct']}%")
print(f"  Test Confusion Matrix:         TN={cm['TN']} | FP={cm['FP']} | FN={cm['FN']} | TP={cm['TP']} (Total={cm['total']})")

# Sample Out-of-Sample Test Predictions
print(f"\n[SAMPLE OUT-OF-SAMPLE TEST PREDICTIONS (LAST 10 TRADING DAYS)]")
test_preds = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
sample_tail = test_preds.tail(10).copy()
sample_tail["Current_Close"] = sample_tail["Current_Close"].apply(lambda x: f"${x:,.2f}")
sample_tail["Actual_Next_Close"] = sample_tail["Actual_Next_Close"].apply(lambda x: f"${x:,.2f}")
sample_tail["Predicted_Next_Close"] = sample_tail["Predicted_Next_Close"].apply(lambda x: f"${x:,.2f}")
sample_tail["Price_Abs_Pct_Error"] = sample_tail["Price_Abs_Pct_Error"].apply(lambda x: f"{x:.2f}%")
sample_tail["Direction_Correct"] = sample_tail["Direction_Correct"].apply(lambda x: "CORRECT [OK]" if x == 1 else "MISSED")

sample_cols = ["Date", "Current_Close", "Predicted_Next_Close", "Actual_Next_Close", "Price_Abs_Pct_Error", "Direction_Correct"]
print(sample_tail[sample_cols].to_string(index=False))

# Live Next-Day Prediction
print("\n" + "=" * 75)
from src.predict import CryptoPredictor
predictor = CryptoPredictor()
processed_df = pd.read_csv(BASE_DIR / "data/processed/bitcoin_features.csv")
latest_row = processed_df.iloc[[-1]]
live_forecast = predictor.predict_next_day(latest_row)

print("  LIVE FORWARD PREDICTION FOR NEXT TRADING DAY:")
print(f"  Observation Date:         {live_forecast['observation_date']}")
print(f"  Current Observed Price:   ${live_forecast['current_close_usd']:,.2f}")
print(f"  Predicted Next-Day Close: ${live_forecast['predicted_close_usd']:,.2f} ({live_forecast['predicted_pct_change']:+.2f}%)")
print(f"  Predicted Market Move:    {live_forecast['direction_label']}")
if live_forecast['direction_confidence']:
    print(f"  Model Probability:        {live_forecast['direction_confidence'] * 100:.1f}%")
print("=" * 75)
