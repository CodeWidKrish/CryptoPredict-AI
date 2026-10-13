"""
Feature Engineering Module for CryptoPredict AI.
Constructs leakage-free technical indicators, lags, rolling statistics, and targets.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PRICE_LAGS,
    RETURN_LAGS,
    MA_WINDOWS,
    EMA_WINDOWS,
    VOLATILITY_WINDOWS,
    RSI_WINDOW,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_WINDOW,
    BOLLINGER_STD_DEV,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
    PROCESSED_DATA_FILE
)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes standard Relative Strength Index (RSI).
    Strictly causal (no lookahead bias).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing / exponential moving average
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_feature_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Transforms clean OHLCV dataframe into a comprehensive, leakage-free feature matrix.

    Returns:
        df_featured: DataFrame containing all features and targets.
        feature_cols: List of engineered feature column names.
        feature_documentation: Dict mapping feature names to their temporal availability & formula.
    """
    df = df.copy()

    # Sort strictly chronologically
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feature_doc: Dict[str, str] = {}

    # Base price and returns
    close = df["Close"]
    open_p = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 1. Daily percentage returns
    daily_return = close.pct_change()
    df["Daily_Return"] = daily_return
    feature_doc["Daily_Return"] = "Return from Close[t-1] to Close[t]: (Close[t] - Close[t-1]) / Close[t-1]. Available at Close[t]."

    # 2. Lagged closing prices
    for lag in PRICE_LAGS:
        col_name = f"Close_Lag_{lag}"
        df[col_name] = close.shift(lag)
        feature_doc[col_name] = f"Closing price observed {lag} day(s) ago (Close[t-{lag}]). Available at Close[t]."

    # 3. Lagged returns
    for lag in RETURN_LAGS:
        col_name = f"Return_Lag_{lag}"
        df[col_name] = daily_return.shift(lag)
        feature_doc[col_name] = f"Daily return observed {lag} day(s) ago (Daily_Return[t-{lag}]). Available at Close[t]."

    # 4. Moving Averages (SMA) and Price/MA Ratios
    for w in MA_WINDOWS:
        ma_col = f"MA_{w}"
        df[ma_col] = close.rolling(window=w).mean()
        feature_doc[ma_col] = f"{w}-day simple moving average of Close up to Close[t]. Available at Close[t]."

        ratio_col = f"Ratio_Close_MA_{w}"
        df[ratio_col] = close / (df[ma_col] + 1e-8)
        feature_doc[ratio_col] = f"Ratio of Close[t] relative to MA_{w}[t]. Normalizes price scale."

    # 5. Exponential Moving Averages (EMA)
    for w in EMA_WINDOWS:
        ema_col = f"EMA_{w}"
        df[ema_col] = close.ewm(span=w, adjust=False).mean()
        feature_doc[ema_col] = f"{w}-day exponential moving average of Close up to Close[t]. Available at Close[t]."

        ratio_col = f"Ratio_Close_EMA_{w}"
        df[ratio_col] = close / (df[ema_col] + 1e-8)
        feature_doc[ratio_col] = f"Ratio of Close[t] relative to EMA_{w}[t]."

    # 6. Rolling Standard Deviation and Volatility
    for w in VOLATILITY_WINDOWS:
        vol_col = f"Volatility_{w}"
        df[vol_col] = daily_return.rolling(window=w).std()
        feature_doc[vol_col] = f"{w}-day rolling standard deviation of daily returns up to Day t. Available at Close[t]."

        std_price_col = f"Price_Std_{w}"
        df[std_price_col] = close.rolling(window=w).std()
        feature_doc[std_price_col] = f"{w}-day rolling standard deviation of closing price up to Day t."

    # 7. Relative Strength Index (RSI)
    df["RSI_14"] = compute_rsi(close, period=RSI_WINDOW)
    feature_doc["RSI_14"] = "14-day Wilder RSI computed strictly on historical Close up to Day t. Value in [0, 100]."

    # 8. Moving Average Convergence Divergence (MACD)
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    df["MACD"] = macd_line
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist
    feature_doc["MACD"] = f"MACD line (EMA_{MACD_FAST} - EMA_{MACD_SLOW}) at Day t. Available at Close[t]."
    feature_doc["MACD_Signal"] = f"MACD Signal line ({MACD_SIGNAL}-day EMA of MACD) at Day t. Available at Close[t]."
    feature_doc["MACD_Hist"] = "MACD Histogram (MACD - Signal) at Day t. Available at Close[t]."

    # 9. Bollinger Bands (20-day, 2 std)
    bb_mid = close.rolling(window=BOLLINGER_WINDOW).mean()
    bb_std = close.rolling(window=BOLLINGER_WINDOW).std()
    bb_upper = bb_mid + (BOLLINGER_STD_DEV * bb_std)
    bb_lower = bb_mid - (BOLLINGER_STD_DEV * bb_std)

    df["Bollinger_Upper"] = bb_upper
    df["Bollinger_Lower"] = bb_lower
    df["Bollinger_Bandwidth"] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
    df["Bollinger_PctB"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
    feature_doc["Bollinger_Upper"] = "Upper Bollinger Band (20-day SMA + 2*Std). Available at Close[t]."
    feature_doc["Bollinger_Lower"] = "Lower Bollinger Band (20-day SMA - 2*Std). Available at Close[t]."
    feature_doc["Bollinger_Bandwidth"] = "Bollinger Bandwidth normalized by SMA. Measures volatility expansion/squeeze."
    feature_doc["Bollinger_PctB"] = "%B indicator showing where Close[t] sits relative to the bands."

    # 10. Intraday price range and open-close return
    df["High_Low_Range"] = (high - low) / (close + 1e-8)
    df["Open_Close_Return"] = (close - open_p) / (open_p + 1e-8)
    feature_doc["High_Low_Range"] = "Normalized intraday volatility: (High[t] - Low[t]) / Close[t]. Available at Close[t]."
    feature_doc["Open_Close_Return"] = "Intraday return: (Close[t] - Open[t]) / Open[t]. Available at Close[t]."

    # 11. Volume dynamics
    df["Volume_MA_7"] = volume.rolling(window=7).mean()
    df["Volume_Change_1d"] = volume.pct_change()
    df["Volume_Ratio_MA7"] = volume / (df["Volume_MA_7"] + 1e-8)
    feature_doc["Volume_MA_7"] = "7-day rolling average volume up to Day t."
    feature_doc["Volume_Change_1d"] = "1-day percentage change in Volume: (Volume[t] - Volume[t-1]) / Volume[t-1]."
    feature_doc["Volume_Ratio_MA7"] = "Ratio of Volume[t] to its 7-day average. Highlights volume spikes."

    # 12. Calendar Features (Known in advance at prediction time)
    df["Day_Of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Day_Of_Month"] = df["Date"].dt.day
    df["Is_Weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)
    feature_doc["Day_Of_Week"] = "Day of week (Monday=0 ... Sunday=6). Fully known at Day t."
    feature_doc["Month"] = "Month of year (1-12). Fully known at Day t."
    feature_doc["Quarter"] = "Quarter of year (1-4). Fully known at Day t."
    feature_doc["Day_Of_Month"] = "Day of month (1-31). Fully known at Day t."
    feature_doc["Is_Weekend"] = "Binary indicator: 1 if Saturday or Sunday, else 0."

    # TARGETS (Strictly shifted into the future by -1 day)
    # Target regression: Next day's closing price Close[t+1]
    df[REGRESSION_TARGET] = close.shift(-1)
    # Target classification: 1 if Next day's close > current day close, else 0
    df[CLASSIFICATION_TARGET] = (df[REGRESSION_TARGET] > close).astype(int)
    # Mark last row's classification target as NaN because future price Close[t+1] is unknown
    df.loc[df[REGRESSION_TARGET].isna(), CLASSIFICATION_TARGET] = np.nan

    # Identify all feature columns (excluding raw dates, metadata, and targets)
    non_feature_cols = {
        "Date",
        "Adj Close",
        REGRESSION_TARGET,
        CLASSIFICATION_TARGET
    }
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    return df, feature_cols, feature_doc


def prepare_datasets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, str]]:
    """
    Computes features, saves the full processed dataset, and isolates:
    1. Historical labeled dataset for training/validation/testing (dropping initial warm-up NaNs & final unclosed target)
    2. Inference row (the very latest completed candle with all historical features computed, waiting for future target)
    """
    df_featured, feature_cols, feature_doc = build_feature_pipeline(df)

    # Save complete dataset to disk for inspection and download
    PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_featured.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"[Features] Saved complete engineered dataset to {PROCESSED_DATA_FILE} ({len(df_featured)} rows)")

    # The latest row is our live inference candidate
    latest_row = df_featured.iloc[[-1]].copy()

    # Drop warm-up rows (e.g. 90-day MA window) and the last row (target unknown) for supervised modeling
    labeled_df = df_featured.dropna(subset=feature_cols + [REGRESSION_TARGET, CLASSIFICATION_TARGET]).copy()
    labeled_df = labeled_df.reset_index(drop=True)
    print(f"[Features] Labeled supervised dataset ready: {len(labeled_df)} rows, {len(feature_cols)} features.")

    return labeled_df, latest_row, feature_cols, feature_doc


if __name__ == "__main__":
    from src.data_loader import fetch_bitcoin_data
    raw_df, _ = fetch_bitcoin_data()
    labeled_df, inference_row, features, doc = prepare_datasets(raw_df)
    print(f"Features count: {len(features)}")
    print(f"First 10 features: {features[:10]}")
    print(f"Labeled data range: {labeled_df['Date'].min().date()} -> {labeled_df['Date'].max().date()}")
    print("Latest inference date:", inference_row["Date"].values[0])
