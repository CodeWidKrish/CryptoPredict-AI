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


# Pro Trading Terminal Color Tokens (Binance & TradingView Standard)
COLOR_BG = "#0B0E14"
COLOR_CARD = "#121722"
COLOR_BORDER = "#1F293D"
COLOR_TEXT = "#FFFFFF"
COLOR_MUTED = "#94A3B8"
COLOR_GREEN = "#0ECB81"   # Pro Binance Bullish Green
COLOR_RED = "#F6465D"     # Pro Binance Bearish Red
COLOR_BLUE = "#38BDF8"    # Cyan Highlight
COLOR_AMBER = "#F59E0B"   # Bitcoin Gold
COLOR_PURPLE = "#8B5CF6"


# ==========================================
# PLOTLY INTERACTIVE CHARTS (For Dashboard)
# ==========================================

def plot_interactive_candlestick(df: pd.DataFrame, days: int = 180) -> go.Figure:
    """Creates a high-performance interactive candlestick chart with volume and MA overlays."""
    recent_df = df.tail(days).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
        subplot_titles=("", "")
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
            increasing_fillcolor=COLOR_GREEN,
            decreasing_line_color=COLOR_RED,
            decreasing_fillcolor=COLOR_RED,
            name="BTC-USD Daily",
            hoverlabel=dict(bgcolor=COLOR_CARD, font=dict(family="JetBrains Mono", size=12, color=COLOR_TEXT))
        ),
        row=1, col=1
    )

    # Moving averages
    if "MA_7" in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df["Date"], y=recent_df["MA_7"],
            line=dict(color=COLOR_BLUE, width=1.75),
            name="7D SMA",
            hovertemplate="7D SMA: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)
    if "MA_30" in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df["Date"], y=recent_df["MA_30"],
            line=dict(color=COLOR_AMBER, width=1.75),
            name="30D SMA",
            hovertemplate="30D SMA: $%{y:,.2f}<extra></extra>"
        ), row=1, col=1)

    # Bollinger Bands if available
    if "Bollinger_Upper" in recent_df.columns and "Bollinger_Lower" in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df["Date"], y=recent_df["Bollinger_Upper"],
            line=dict(color="rgba(148, 163, 184, 0.35)", width=1, dash="dot"),
            name="Upper Bollinger (20,2)",
            hoverinfo="skip"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=recent_df["Date"], y=recent_df["Bollinger_Lower"],
            line=dict(color="rgba(148, 163, 184, 0.35)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.03)",
            name="Lower Bollinger (20,2)",
            hoverinfo="skip"
        ), row=1, col=1)

    # Volume Bar Chart
    colors = [COLOR_GREEN if c >= o else COLOR_RED for c, o in zip(recent_df["Close"], recent_df["Open"])]
    fig.add_trace(
        go.Bar(
            x=recent_df["Date"],
            y=recent_df["Volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.75,
            hovertemplate="Volume: $%{y:,.0f}<extra></extra>"
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(17, 26, 46, 0.8)",
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans, Inter, sans-serif"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(17, 26, 46, 0.6)",
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            font=dict(size=11)
        ),
        height=520
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        linecolor=COLOR_BORDER,
        tickfont=dict(family="JetBrains Mono", size=10, color=COLOR_MUTED)
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        linecolor=COLOR_BORDER,
        tickprefix="$",
        tickformat=",.0f",
        tickfont=dict(family="JetBrains Mono", size=10, color=COLOR_MUTED),
        row=1, col=1
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.03)",
        linecolor=COLOR_BORDER,
        tickfont=dict(family="JetBrains Mono", size=9, color=COLOR_MUTED),
        row=2, col=1
    )
    return fig


def plot_interactive_actual_vs_predicted(
    dates: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
    model_name: str
) -> go.Figure:
    """Plots actual test prices vs model predicted prices with error tracking."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode="lines",
        name="Actual Next-Day Close",
        line=dict(color="#E2E8F0", width=2.2),
        hovertemplate="Actual: $%{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode="lines",
        name=f"Predicted ({model_name})",
        line=dict(color=COLOR_BLUE, width=1.8, dash="dash"),
        hovertemplate="Predicted: $%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"Out-of-Sample Performance: Actual vs Predicted Next-Day Close",
            font=dict(size=14, color=COLOR_TEXT, family="Plus Jakarta Sans, sans-serif")
        ),
        template="plotly_dark",
        plot_bgcolor="rgba(17, 26, 46, 0.8)",
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(family="JetBrains Mono", size=10, color=COLOR_MUTED)
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickprefix="$",
            tickformat=",.0f",
            tickfont=dict(family="JetBrains Mono", size=10, color=COLOR_MUTED)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(17, 26, 46, 0.6)",
            bordercolor=COLOR_BORDER,
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=420
    )
    return fig


def plot_interactive_confusion_matrix(cm_dict: dict, model_name: str) -> go.Figure:
    """Plots interactive heatmap for classification confusion matrix with quantitative descriptors."""
    tn, fp, fn, tp = cm_dict["TN"], cm_dict["FP"], cm_dict["FN"], cm_dict["TP"]
    z = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted Down (0)", "Predicted Up (1)"]
    y_labels = ["Actual Down (0)", "Actual Up (1)"]

    cell_labels = [
        [f"<b>TN = {tn}</b><br><span style='font-size:11px'>Dips Avoided</span>",
         f"<b>FP = {fp}</b><br><span style='font-size:11px'>False Breakouts</span>"],
        [f"<b>FN = {fn}</b><br><span style='font-size:11px'>Missed Rallies</span>",
         f"<b>TP = {tp}</b><br><span style='font-size:11px'>Rallies Captured</span>"]
    ]

    annotations = []
    for i, row in enumerate(cell_labels):
        for j, text in enumerate(row):
            annotations.append(dict(
                x=x_labels[j],
                y=y_labels[i],
                text=text,
                font=dict(color="#FFFFFF", size=14, family="Plus Jakarta Sans, sans-serif"),
                showarrow=False
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, "#0F172A"], [0.3, "#1E293B"], [0.7, "#0284C7"], [1.0, "#10B981"]],
        showscale=False
    ))

    fig.update_layout(
        title=dict(
            text=f"Direction Confusion Matrix ({model_name})",
            font=dict(size=14, color=COLOR_TEXT)
        ),
        annotations=annotations,
        template="plotly_dark",
        plot_bgcolor="rgba(17, 26, 46, 0.8)",
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        xaxis=dict(tickfont=dict(size=12, color=COLOR_TEXT)),
        yaxis=dict(tickfont=dict(size=12, color=COLOR_TEXT)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=360
    )
    return fig


def plot_interactive_feature_importance(importances: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Creates a horizontal bar chart of top feature importances with electric gradient."""
    top_df = importances.head(top_n).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=top_df["Importance"],
        y=top_df["Feature"],
        orientation="h",
        marker=dict(
            color=top_df["Importance"],
            colorscale=[[0, "#0284C7"], [0.5, "#38BDF8"], [1, "#10B981"]],
            line=dict(color="rgba(255, 255, 255, 0.15)", width=1)
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Predictive Feature Weights",
            font=dict(size=14, color=COLOR_TEXT)
        ),
        template="plotly_dark",
        plot_bgcolor="rgba(17, 26, 46, 0.8)",
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        xaxis=dict(
            title=dict(text="Gini / Split Gain Importance", font=dict(size=11, color=COLOR_MUTED)),
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(family="JetBrains Mono", size=10, color=COLOR_MUTED)
        ),
        yaxis=dict(
            tickfont=dict(family="JetBrains Mono", size=11, color=COLOR_TEXT)
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=460
    )
    return fig


def plot_interactive_probability_gauge(prob: float, direction: int) -> go.Figure:
    """Creates a semicircular directional conviction gauge meter with high-contrast indicator."""
    pct = prob * 100.0
    accent_color = COLOR_GREEN if direction == 1 else COLOR_RED

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(
            suffix="%",
            font=dict(size=42, family="JetBrains Mono", color=COLOR_TEXT)
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor=COLOR_MUTED,
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0% Bear", "25%", "50% Neutral", "75%", "100% Bull"],
                tickfont=dict(family="JetBrains Mono", size=10, color=COLOR_MUTED)
            ),
            bar=dict(color=accent_color, thickness=0.32),
            bgcolor="rgba(17, 26, 46, 0.9)",
            borderwidth=1,
            bordercolor=COLOR_BORDER,
            steps=[
                dict(range=[0, 45], color="rgba(244, 63, 94, 0.18)"),
                dict(range=[45, 55], color="rgba(245, 158, 11, 0.18)"),
                dict(range=[55, 100], color="rgba(16, 185, 129, 0.18)")
            ],
            threshold=dict(
                line=dict(color="#FFFFFF", width=3),
                thickness=0.8,
                value=50.0
            )
        )
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans, sans-serif"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=240
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


def plot_interactive_walk_forward(wf_df: pd.DataFrame) -> go.Figure:
    """Plots walk-forward performance across historical regimes comparing Naive vs Linear Regression."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Regression MAPE % (Lower is Better)", "Direction Classification Accuracy % vs Baseline 50%")
    )

    # Subplot 1: MAPE %
    fig.add_trace(
        go.Bar(
            x=wf_df["Regime_Name"],
            y=wf_df["Naive_MAPE_Pct"],
            name="Naive Previous Close MAPE %",
            marker_color="#38BDF8"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=wf_df["Regime_Name"],
            y=wf_df["Linear_Regression_MAPE_Pct"],
            name="Linear Regression MAPE %",
            marker_color="#F59E0B"
        ),
        row=1, col=1
    )

    # Subplot 2: Classification Accuracy %
    fig.add_trace(
        go.Bar(
            x=wf_df["Regime_Name"],
            y=wf_df["Classifier_Accuracy_Pct"],
            name="Classifier Accuracy %",
            marker_color="#0ECB81"
        ),
        row=2, col=1
    )
    fig.add_hline(y=50.0, line_dash="dash", line_color="#94A3B8", annotation_text="50% Coin-Flip Baseline", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(17, 26, 46, 0.8)",
        paper_bgcolor=COLOR_BG,
        barmode="group",
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans, sans-serif"),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=480
    )
    return fig


def plot_interactive_threshold_curve(curve_df: pd.DataFrame) -> go.Figure:
    """Plots precision, recall, specificity, balanced accuracy, and F1 across decision threshold tau."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve_df["threshold"], y=curve_df["recall_pct"],
        name="Recall / Sensitivity %", line=dict(color="#0ECB81", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=curve_df["threshold"], y=curve_df["specificity_pct"],
        name="Specificity %", line=dict(color="#38BDF8", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=curve_df["threshold"], y=curve_df["precision_pct"],
        name="Precision %", line=dict(color="#F59E0B", width=2.0, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=curve_df["threshold"], y=curve_df["balanced_accuracy_pct"],
        name="Balanced Accuracy %", line=dict(color="#8B5CF6", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=curve_df["threshold"], y=curve_df["f1_score_pct"],
        name="F1-Score %", line=dict(color="#F6465D", width=2.0)
    ))

    # Vertical line at default tau = 0.50
    fig.add_vline(x=0.50, line_dash="dash", line_color="#EF4444", annotation_text="Default tau = 0.50")

    fig.update_layout(
        title=dict(text="Classification Threshold Optimization Curve (tau in [0.30, 0.70])", font=dict(size=14, color=COLOR_TEXT)),
        template="plotly_dark",
        plot_bgcolor="rgba(17, 26, 46, 0.8)",
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(title="Decision Threshold (tau)", gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(title="Metric Score (%)", gridcolor="rgba(255, 255, 255, 0.05)", range=[0, 100]),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=420
    )
    return fig


def plot_interactive_backtest_equity(equity_df: pd.DataFrame) -> go.Figure:
    """Plots transaction-cost-aware Strategy Equity vs Buy & Hold Benchmark with range selector."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df["Date"], y=equity_df["Strategy_Equity"],
        name="Strategy (0.10% Fee + 0.05% Slippage)",
        line=dict(color="#0ECB81", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(14, 203, 129, 0.06)",
        hovertemplate="Strategy: $%{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=equity_df["Date"], y=equity_df["Benchmark_Equity"],
        name="Buy & Hold Bitcoin Benchmark",
        line=dict(color="#38BDF8", width=2.0, dash="dash"),
        hovertemplate="Buy & Hold: $%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Portfolio Growth vs Buy & Hold Benchmark (Starting Capital: $10,000 USD)", font=dict(size=14, color=COLOR_TEXT)),
        template="plotly_dark",
        plot_bgcolor="rgba(17, 26, 46, 0.8)",
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="#141B29",
                activecolor="#0284C7",
                font=dict(color="#FFFFFF", size=11)
            ),
            rangeslider=dict(visible=False),
            type="date"
        ),
        yaxis=dict(title="Portfolio Value ($USD)", tickprefix="$", tickformat=",.0f", gridcolor="rgba(255, 255, 255, 0.05)"),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=65, b=20),
        height=450
    )
    return fig


def plot_interactive_shap_summary(shap_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Plots top features by Mean |SHAP| value with color graduation."""
    top_df = shap_df.head(top_n).sort_values("Mean_Abs_SHAP", ascending=True)

    fig = go.Figure(go.Bar(
        x=top_df["Mean_Abs_SHAP"],
        y=top_df["Feature"],
        orientation="h",
        marker=dict(
            color=top_df["Mean_Abs_SHAP"],
            colorscale=[[0, "#1E293B"], [0.5, "#0284C7"], [1.0, "#0ECB81"]],
            showscale=False
        ),
        hovertemplate="Feature: %{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"SHAP Global Feature Importance (Top {top_n} Predictors)", font=dict(size=14, color=COLOR_TEXT)),
        template="plotly_dark",
        plot_bgcolor="rgba(17, 26, 46, 0.8)",
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT, family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(title="Mean Absolute SHAP Value (Impact on Direction)", gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(tickfont=dict(family="JetBrains Mono", size=11, color=COLOR_TEXT)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=450
    )
    return fig

