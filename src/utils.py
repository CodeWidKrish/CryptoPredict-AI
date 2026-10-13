"""
Utility functions for CryptoPredict AI.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy data types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (datetime, np.datetime64)):
            return str(obj)
        return super().default(obj)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Safely saves a dictionary to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NpEncoder)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Safely loads a dictionary from a JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_currency(value: float) -> str:
    """Formats a float as USD currency."""
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    """Formats a float (e.g. 52.3) as percentage string."""
    return f"{value:.2f}%"
