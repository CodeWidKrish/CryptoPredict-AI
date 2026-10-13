"""
Configuration file for CryptoPredict AI.
Defines paths, feature engineering parameters, splitting ratios, and model settings.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"

# Ensure essential directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Data file paths
RAW_DATA_FILE = DATA_RAW_DIR / "bitcoin_history.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "bitcoin_features.csv"
MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"
REGRESSION_METRICS_FILE = REPORTS_DIR / "regression_metrics.json"
CLASSIFICATION_METRICS_FILE = REPORTS_DIR / "classification_metrics.json"
TEST_PREDICTIONS_FILE = REPORTS_DIR / "test_predictions.csv"
EXPERIMENT_SUMMARY_FILE = REPORTS_DIR / "experiment_summary.md"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Market Data Settings
TICKER_SYMBOL = "BTC-USD"
START_DATE = "2014-09-17"  # Inception date for BTC-USD on Yahoo Finance
RANDOM_SEED = 42

# Time Series Split Ratios (Strict Chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature Engineering Parameters
PRICE_LAGS = [1, 2, 3, 7, 14, 30]
RETURN_LAGS = [1, 2, 3, 7, 14, 30]
MA_WINDOWS = [7, 14, 30, 90]
EMA_WINDOWS = [7, 14, 30, 90]
VOLATILITY_WINDOWS = [7, 14, 30]
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD_DEV = 2

# Target Variable Names
REGRESSION_TARGET = "Target_Close_Next"
RETURN_TARGET = "Target_Return_Next"
CLASSIFICATION_TARGET = "Target_Direction_Next"
