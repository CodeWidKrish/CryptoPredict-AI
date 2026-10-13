"""
Tests for Data Ingestion, Validation, and Caching.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import clean_yfinance_dataframe, validate_market_data, fetch_bitcoin_data
from config.config import RAW_DATA_FILE


def test_clean_yfinance_dataframe():
    """Verifies column standardization, duplicate removal, and chronological sorting."""
    sample_data = {
        "Date": ["2023-01-02", "2023-01-01", "2023-01-02"],
        "open": [16500.0, 16400.0, 16500.0],
        "high": [16700.0, 16600.0, 16700.0],
        "low": [16400.0, 16300.0, 16400.0],
        "close": [16600.0, 16500.0, 16600.0],
        "volume": [1000000, 900000, 1000000]
    }
    raw_df = pd.DataFrame(sample_data)
    cleaned = clean_yfinance_dataframe(raw_df)

    assert len(cleaned) == 2, "Duplicates should be removed"
    assert cleaned["Date"].iloc[0] < cleaned["Date"].iloc[1], "Dates must be chronologically ascending"
    assert "Open" in cleaned.columns and "Close" in cleaned.columns, "Columns must be standardized to Title Case"


def test_validate_market_data_genuine():
    """Verifies that the cached genuine Bitcoin dataset passes all integrity checks."""
    assert RAW_DATA_FILE.exists(), f"Historical data file missing: {RAW_DATA_FILE}"
    df = pd.read_csv(RAW_DATA_FILE)
    df = clean_yfinance_dataframe(df)

    summary = validate_market_data(df)
    assert summary["status"] == "VALID"
    assert summary["row_count"] >= 3000, "Bitcoin dataset should have over 3,000 daily observations"
    assert summary["min_close"] > 0, "All Bitcoin closing prices must be strictly positive"


def test_validate_market_data_rejection():
    """Verifies that invalid datasets with NaNs, missing columns, or bad dates are rejected."""
    bad_df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Close": [100.0, None]  # Contains NaN
    })
    with pytest.raises(ValueError):
        validate_market_data(bad_df)


def test_fetch_bitcoin_data_offline_fallback():
    """Verifies that fetch_bitcoin_data returns valid data even without re-fetching."""
    df, meta = fetch_bitcoin_data(force_download=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "source" in meta
