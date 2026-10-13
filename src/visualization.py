"""
Visualization Module for CryptoPredict AI.
Generates interactive Plotly charts and publication-grade Matplotlib figures.
"""

import sys
from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI compatibility
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import FIGURES_DIR


# Dark terminal palette
COLOR_BG = "#0B101D"
COLOR_CARD = "#151C2C"
COLOR_TEXT = "#E2E8F0"
COLOR_GREEN = "#00F5A0"
COLOR_RED = "#FF4B72"
COLOR_BLUE = "#00D2FF"
COLOR_PURPLE = "#7928CA"
COLOR_ACCENT = "#F59E0B"


# ==========================================
# PLOTLY INTERACTIVE CHARTS (For Dashboard)
# ==========================================

def plot_interactive_candlestick(df: pd.DataFrame, days: int = 180) -> go.Figure:
    """Creates a high-performance interactive candlestick chart with volume."""
    recent_df = df.tail(days).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=("Bitcoin (BTC/USD) Daily Candlestick", "Volume")
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=recent_df["Date"],
            open=recent_df["Open"],
            high=recent_df["High"],
            low=recent_df["Low"],
            close=recent_df["Close"],
            increasing_line_color=COLOR_GREEN,
            decreasing_line_color=COLOR_RED,
            name="BTC-USD"
        ),
        row=1, col=1
    )

    # Moving averages if present
    if "MA_7" in recent_df.columns:
        fig.add_trace(go.Scatter(x=recent_df["Date"], y=recent_df["MA_7"], line=dict(color=COLOR_BLUE, width=1.5), name="7-day SMA"), row=1, col=1)
    if "MA_30" in recent_df.columns:
        fig.add_trace(go.Scatter(x=recent_df["Date"], y=recent_df["MA_30"], line=dict(color=COLOR_ACCENT, width=1.5), name="30-day SMA"), row=1, col=1)

    # Volume Bar Chart
    colors = [COLOR_GREEN if c >= o else COLOR_RED for c, o in zip(recent_df["Close"], recent_df["Open"])]
    fig.add_trace(
        go.Bar(
            x=recent_df["Date"],
            y=recent_df["Volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.8
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Inter, sans-serif"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_interactive_actual_vs_predicted(
    dates: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
    model_name: str
) -> go.Figure:
    """Plots actual test prices vs model predicted prices."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode="lines",
        name="Actual Next-Day Close",
        line=dict(color="#94A3B8", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode="lines",
        name=f"Predicted ({model_name})",
        line=dict(color=COLOR_BLUE, width=2, dash="dash")
    ))

    fig.update_layout(
        title=f"Actual vs Predicted Bitcoin Next-Day Close ({model_name})",
        template="plotly_dark",
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_interactive_confusion_matrix(cm_dict: dict, model_name: str) -> go.Figure:
    """Plots interactive heatmap for classification confusion matrix."""
    tn, fp, fn, tp = cm_dict["TN"], cm_dict["FP"], cm_dict["FN"], cm_dict["TP"]
    z = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted Down (0)", "Predicted Up (1)"]
    y_labels = ["Actual Down (0)", "Actual Up (1)"]

    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=x_labels[j],
                y=y_labels[i],
                text=f"{val}",
                font=dict(color="white", size=18, family="sans-serif"),
                showarrow=False
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, "#1E293B"], [0.5, "#0284C7"], [1.0, "#00F5A0"]],
        showscale=False
    ))

    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        annotations=annotations,
        template="plotly_dark",
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def plot_interactive_feature_importance(importances: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Creates a horizontal bar chart of top feature importances."""
    top_df = importances.head(top_n).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=top_df["Importance"],
        y=top_df["Feature"],
        orientation="h",
        marker=dict(
            color=top_df["Importance"],
            colorscale=[[0, "#0284C7"], [1, "#00F5A0"]],
        )
    ))

    fig.update_layout(
        title=f"Top {top_n} Most Influential Features",
        template="plotly_dark",
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        xaxis_title="Relative Importance Score",
        yaxis_title="Feature",
        margin=dict(l=60, r=40, t=50, b=40),
        height=450
    )
    return fig


# ==========================================
# STATIC MATPLOTLIB FIGURES (For Report)
# ==========================================

def save_static_eda_figures(df: pd.DataFrame) -> List[Path]:
    """Generates and saves publication-quality static PNG figures for the report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # 1. Historical Close and Volume
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df["Date"], df["Close"], color="#00F5A0", linewidth=1.5, label="BTC-USD Close")
    if "MA_30" in df.columns:
        ax1.plot(df["Date"], df["MA_30"], color="#F59E0B", linewidth=1.2, linestyle="--", label="30-day MA")
    ax1.set_title("Bitcoin (BTC-USD) Historical Daily Closing Price (2014 - Present)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper left")

    ax2.bar(df["Date"], df["Volume"] / 1e9, color="#00D2FF", alpha=0.7, label="Volume (Billions USD)")
    ax2.set_ylabel("Volume ($B)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    p1 = FIGURES_DIR / "eda_price_volume_history.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    saved_files.append(p1)

    # 2. Daily Returns Distribution
    if "Daily_Return" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        returns = df["Daily_Return"].dropna() * 100
        ax.hist(returns, bins=80, color="#7928CA", edgecolor="#151C2C", alpha=0.85, density=True)
        ax.axvline(returns.mean(), color="#00F5A0", linestyle="--", label=f"Mean: {returns.mean():.2f}%")
        ax.axvline(returns.median(), color="#F59E0B", linestyle=":", label=f"Median: {returns.median():.2f}%")
        ax.set_title("Distribution of Bitcoin Daily Percentage Returns", fontsize=14, fontweight="bold")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Probability Density")
        ax.set_xlim(-25, 25)
        ax.grid(True, alpha=0.2)
        ax.legend()
        plt.tight_layout()
        p2 = FIGURES_DIR / "eda_return_distribution.png"
        plt.savefig(p2, dpi=300)
        plt.close()
        saved_files.append(p2)

    # 3. Rolling Volatility
    if "Volatility_30" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["Date"], df["Volatility_30"] * 100, color="#FF4B72", linewidth=1.5, label="30-day Rolling Volatility (%)")
        ax.set_title("Bitcoin 30-Day Rolling Annualized Daily Volatility", fontsize=14, fontweight="bold")
        ax.set_ylabel("Volatility (%)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.2)
        ax.legend()
        plt.tight_layout()
        p3 = FIGURES_DIR / "eda_rolling_volatility.png"
        plt.savefig(p3, dpi=300)
        plt.close()
        saved_files.append(p3)

    return saved_files


def save_static_prediction_figures(
    dates: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
    model_name: str,
    cm_dict: dict
) -> List[Path]:
    """Saves static test prediction and confusion matrix plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # Actual vs Predicted plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, actual, color="#94A3B8", linewidth=1.8, label="Actual Next-Day Close")
    ax.plot(dates, predicted, color="#00D2FF", linewidth=1.5, linestyle="--", label=f"Predicted ({model_name})")
    ax.set_title(f"Test Set: Actual vs. Predicted Bitcoin Closing Price ({model_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    p1 = FIGURES_DIR / "test_actual_vs_predicted.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    saved_files.append(p1)

    # Confusion matrix plot
    tn, fp, fn, tp = cm_dict["TN"], cm_dict["FP"], cm_dict["FN"], cm_dict["TP"]
    cm_arr = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm_arr, cmap="Blues")
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(cm_arr):
        ax.text(j, i, f"{z}", ha="center", va="center", color="white" if z > cm_arr.mean() else "black", fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Down (0)", "Pred Up (1)"])
    ax.set_yticklabels(["Actual Down (0)", "Actual Up (1)"])
    ax.set_title("Test Confusion Matrix (Direction Classification)", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    p2 = FIGURES_DIR / "test_confusion_matrix.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    saved_files.append(p2)

    return saved_files
