"""
Data Loader Module for CryptoPredict AI.
Fetches, validates, and stores authentic historical Bitcoin market data from yfinance.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yfinance as yf

from config.config import RAW_DATA_FILE, TICKER_SYMBOL, START_DATE


def clean_yfinance_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe returned by yfinance:
    - Flattens multi-level column indexes
    - Standardizes column names
    - Ensures Date is a sorted DateTime column
    """
    df = raw_df.copy()

    # Flatten MultiIndex columns if present (e.g. ('Close', 'BTC-USD'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Reset index if Date is the index
    if "Date" not in df.columns and (isinstance(df.index, pd.DatetimeIndex) or df.index.name == "Date"):
        df = df.reset_index()

    # Ensure Date column exists
    date_col = next((c for c in df.columns if str(c).lower() == "date"), None)
    if date_col is None:
        raise ValueError("Downloaded dataset does not contain a 'Date' column.")
    df.rename(columns={date_col: "Date"}, inplace=True)

    # Standardize column naming
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjclose": "Adj Close",
        "volume": "Volume"
    }
    df.rename(columns=lambda c: rename_map.get(str(c).lower(), c), inplace=True)

    # Convert Date to datetime and sort
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure essential columns exist
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from downloaded market data.")

    # Convert numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        numeric_cols.append("Adj Close")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicate dates if any
    df = df.drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # Drop any entirely empty rows
    df = df.dropna(subset=["Close", "Open", "High", "Low", "Volume"]).reset_index(drop=True)

    # Sanity checks for Bitcoin prices
    df = df[df["Close"] > 0].reset_index(drop=True)
    df = df[df["Volume"] >= 0].reset_index(drop=True)

    return df


def validate_market_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates the integrity of the market dataset.
    Raises ValueError if validation fails.
    """
    if df is None or len(df) == 0:
        raise ValueError("Validation failed: Market dataset is empty.")

    if len(df) < 500:
        raise ValueError(f"Validation failed: Insufficient data rows ({len(df)}). Expected >= 500.")

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Validation failed: Missing required columns: {missing}")

    if df["Date"].duplicated().any():
        raise ValueError("Validation failed: Found duplicate dates in dataset.")

    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Validation failed: Dates are not strictly chronologically increasing.")

    if df[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
        raise ValueError("Validation failed: Found unexpected NaN values in OHLCV columns.")

    summary = {
        "status": "VALID",
        "row_count": len(df),
        "start_date": str(df["Date"].min().date()),
        "end_date": str(df["Date"].max().date()),
        "columns": list(df.columns),
        "min_close": float(df["Close"].min()),
        "max_close": float(df["Close"].max()),
        "latest_close": float(df["Close"].iloc[-1]),
    }
    return summary


def fetch_bitcoin_data(force_download: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetches genuine Bitcoin data from yfinance and caches it to RAW_DATA_FILE.
    If the API call fails, falls back to the locally cached RAW_DATA_FILE.
    Never fabricates or synthesizes Bitcoin prices.
    """
    df = None
    data_source = None
    download_timestamp = None

    if not force_download and RAW_DATA_FILE.exists():
        try:
            print(f"[DataLoader] Loading cached historical data from {RAW_DATA_FILE}...")
            cached_df = pd.read_csv(RAW_DATA_FILE)
            cached_df = clean_yfinance_dataframe(cached_df)
            val_summary = validate_market_data(cached_df)
            val_summary["source"] = f"Local Cache ({RAW_DATA_FILE})"
            val_summary["download_timestamp"] = "Cached"
            print(f"[DataLoader] Successfully loaded {len(cached_df)} valid cached records.")
            return cached_df, val_summary
        except Exception as e:
            print(f"[DataLoader] Cached file check failed ({e}). Proceeding to live download...")

    print(f"[DataLoader] Downloading historical data for '{TICKER_SYMBOL}' from yfinance (start: {START_DATE})...")
    try:
        raw_df = yf.download(
            tickers=TICKER_SYMBOL,
            start=START_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if raw_df is None or len(raw_df) == 0:
            raise ConnectionError(f"yfinance returned empty dataset for {TICKER_SYMBOL}.")

        df = clean_yfinance_dataframe(raw_df)
        val_summary = validate_market_data(df)
        data_source = f"Yahoo Finance API ({TICKER_SYMBOL})"
        download_timestamp = datetime.now(timezone.utc).isoformat()

        # Cache valid real data
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_FILE, index=False)
        print(f"[DataLoader] Successfully saved {len(df)} authentic records to {RAW_DATA_FILE}")

        val_summary["source"] = data_source
        val_summary["download_timestamp"] = download_timestamp
        return df, val_summary

    except Exception as api_err:
        print(f"[DataLoader] Online download failed: {api_err}")
        if RAW_DATA_FILE.exists():
            print(f"[DataLoader] Attempting recovery using existing local cache {RAW_DATA_FILE}...")
            cached_df = pd.read_csv(RAW_DATA_FILE)
            cached_df = clean_yfinance_dataframe(cached_df)
            val_summary = validate_market_data(cached_df)
            val_summary["source"] = f"Local Cache ({RAW_DATA_FILE}) [Fallback after API error: {api_err}]"
            val_summary["download_timestamp"] = "Cached (Fallback)"
            return cached_df, val_summary
        else:
            raise RuntimeError(
                f"FATAL: Unable to retrieve genuine Bitcoin data from Yahoo Finance, "
                f"and no local backup file exists at {RAW_DATA_FILE}. "
                f"Synthetic data substitution is strictly forbidden. Error details: {api_err}"
            )


if __name__ == "__main__":
    df, meta = fetch_bitcoin_data(force_download=True)
    print("Downloaded Metadata:", meta)
    print("Head:\n", df.head())
    print("Tail:\n", df.tail())
