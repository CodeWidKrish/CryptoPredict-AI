"""
CryptoPredict AI — Bitcoin Price Prediction & Market Direction Intelligence.
Professional Streamlit Terminal Dashboard.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    MODEL_COMPARISON_FILE,
    REGRESSION_METRICS_FILE,
    CLASSIFICATION_METRICS_FILE,
    TEST_PREDICTIONS_FILE,
    EXPERIMENT_SUMMARY_FILE,
    METADATA_FILE
)
from src.predict import CryptoPredictor
from src.visualization import (
    plot_interactive_candlestick,
    plot_interactive_actual_vs_predicted,
    plot_interactive_confusion_matrix,
    plot_interactive_feature_importance
)
from scripts.run_all import run_pipeline


# ==========================================
# PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="CryptoPredict AI | Bitcoin Market Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Terminal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #070B14;
        color: #E2E8F0;
    }
    
    /* Top Header Banner */
    .terminal-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .terminal-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .terminal-badge {
        background: rgba(0, 245, 160, 0.15);
        color: #00F5A0;
        border: 1px solid #00F5A0;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .terminal-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 6px;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    
    .metric-label {
        color: #94A3B8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    .metric-delta-pos {
        color: #00F5A0;
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
        margin-top: 4px;
    }
    
    .metric-delta-neg {
        color: #FF4B72;
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
        margin-top: 4px;
    }
    
    /* Prediction Banner */
    .pred-card {
        background: linear-gradient(145deg, #0F172A 0%, #162033 100%);
        border: 2px solid #0284C7;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 24px;
    }
    
    .bullish-badge {
        background-color: rgba(0, 245, 160, 0.15);
        color: #00F5A0;
        border: 1px solid #00F5A0;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
    }
    
    .bearish-badge {
        background-color: rgba(255, 75, 114, 0.15);
        color: #FF4B72;
        border: 1px solid #FF4B72;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
    }
    
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 12px;
        color: #FCD34D;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED DATA & MODEL LOADERS
# ==========================================
@st.cache_data(ttl=3600)
def load_datasets():
    if not PROCESSED_DATA_FILE.exists():
        return None, None
    processed_df = pd.read_csv(PROCESSED_DATA_FILE)
    processed_df["Date"] = pd.to_datetime(processed_df["Date"])
    raw_df = pd.read_csv(RAW_DATA_FILE) if RAW_DATA_FILE.exists() else processed_df
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    return processed_df, raw_df


@st.cache_data(ttl=3600)
def load_reports():
    comparison_df = pd.read_csv(MODEL_COMPARISON_FILE) if MODEL_COMPARISON_FILE.exists() else None
    test_preds_df = pd.read_csv(TEST_PREDICTIONS_FILE) if TEST_PREDICTIONS_FILE.exists() else None
    if test_preds_df is not None:
        test_preds_df["Date"] = pd.to_datetime(test_preds_df["Date"])

    meta = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            meta = json.load(f)

    reg_metrics = {}
    if REGRESSION_METRICS_FILE.exists():
        with open(REGRESSION_METRICS_FILE, "r") as f:
            reg_metrics = json.load(f)

    clf_metrics = {}
    if CLASSIFICATION_METRICS_FILE.exists():
        with open(CLASSIFICATION_METRICS_FILE, "r") as f:
            clf_metrics = json.load(f)

    return comparison_df, test_preds_df, meta, reg_metrics, clf_metrics


@st.cache_resource
def get_predictor():
    try:
        return CryptoPredictor()
    except Exception as e:
        return None


# ==========================================
# MAIN APPLICATION LOGIC
# ==========================================
def main():
    processed_df, raw_df = load_datasets()
    comparison_df, test_preds_df, meta, reg_metrics, clf_metrics = load_reports()
    predictor = get_predictor()

    # --- TOP HEADER ---
    st.markdown("""
    <div class="terminal-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="terminal-title">
                    ⚡ CryptoPredict AI
                    <span class="terminal-badge">Production v1.0</span>
                </h1>
                <p class="terminal-subtitle">
                    Bitcoin Price Forecasting & Market Direction Intelligence Platform | Zero-Leakage ML Pipeline
                </p>
            </div>
            <div style="text-align: right;">
                <span style="font-family: 'JetBrains Mono'; font-size: 13px; color: #38BDF8;">BTC-USD Real-Time Terminal</span><br>
                <span style="font-size: 12px; color: #64748B;">Evaluation Benchmark Engine</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### 🎛️ Terminal Navigation")
        app_page = st.radio(
            "Select View",
            [
                "📊 Market Overview",
                "🎯 Next-Day Price Prediction",
                "🏆 Model Performance & Leaderboard",
                "🔍 Historical Date Inspector",
                "⚙️ Pipeline Retraining",
                "📥 Data & Reports Download"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 📡 Asset Details")
        st.markdown("""
        - **Asset:** Bitcoin (`BTC-USD`)
        - **Frequency:** Daily (`1D`)
        - **Horizon:** Next Trading Day ($t+1$)
        - **Split:** 70% Train / 15% Val / 15% Test
        - **CV Strategy:** TimeSeriesSplit (No Shuffle)
        """)

        if meta and "dataset_rows" in meta:
            st.markdown("---")
            st.markdown("### 📈 Pipeline Metadata")
            st.caption(f"**Total Observations:** {meta['dataset_rows']:,}")
            st.caption(f"**Engineered Features:** {meta['features']['count']}")
            st.caption(f"**Date Span:** {meta['date_range']['first_date']} → {meta['date_range']['last_date']}")
            st.caption(f"**Best Regressor:** {meta['winning_regression_model']}")
            st.caption(f"**Best Classifier:** {meta['winning_classification_model']}")

    if processed_df is None or predictor is None:
        st.warning("⚠️ Models or datasets not initialized. Please click 'Pipeline Retraining' in the sidebar to execute the pipeline.")
        return

    latest_row = processed_df.iloc[[-1]]
    latest_close = float(latest_row["Close"].values[0])
    latest_date = str(pd.to_datetime(latest_row["Date"].values[0]).date())
    prev_close = float(latest_row["Close_Lag_1"].values[0]) if "Close_Lag_1" in latest_row else latest_close
    pct_change_24h = ((latest_close - prev_close) / prev_close) * 100.0

    # ========================================================
    # 1. MARKET OVERVIEW
    # ========================================================
    if app_page == "📊 Market Overview":
        st.subheader("📊 Bitcoin Market Overview & Technical Indicators")

        # Metric Ribbon
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta_class = "metric-delta-pos" if pct_change_24h >= 0 else "metric-delta-neg"
            arrow = "▲" if pct_change_24h >= 0 else "▼"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latest Close ({latest_date})</div>
                <div class="metric-value">${latest_close:,.2f}</div>
                <div class="{delta_class}">{arrow} {pct_change_24h:+.2f}% (24h)</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            high_24h = float(latest_row["High"].values[0])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily High</div>
                <div class="metric-value">${high_24h:,.2f}</div>
                <div style="color: #64748B; font-size: 13px; margin-top: 4px;">24h Candle Range</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            low_24h = float(latest_row["Low"].values[0])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Low</div>
                <div class="metric-value">${low_24h:,.2f}</div>
                <div style="color: #64748B; font-size: 13px; margin-top: 4px;">Intraday Support</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            vol_24h = float(latest_row["Volume"].values[0]) / 1e9
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Volume</div>
                <div class="metric-value">${vol_24h:.2f}B</div>
                <div style="color: #38BDF8; font-size: 13px; margin-top: 4px;">Liquid Trading Volume</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive Candlestick Chart
        c_col1, c_col2 = st.columns([3, 1])
        with c_col2:
            lookback_days = st.select_slider("Chart History (Days)", options=[30, 60, 90, 180, 365, 730], value=180)
        
        candlestick_fig = plot_interactive_candlestick(raw_df, days=lookback_days)
        st.plotly_chart(candlestick_fig, use_container_width=True)

        # Technical Indicators Grid
        st.markdown("### 📐 Core Technical Indicator Readings")
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)

        with tcol1:
            rsi_val = float(latest_row["RSI_14"].values[0]) if "RSI_14" in latest_row else 50.0
            rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
            rsi_color = "#FF4B72" if rsi_val > 70 else ("#00F5A0" if rsi_val < 30 else "#38BDF8")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RSI (14-Day)</div>
                <div class="metric-value" style="color: {rsi_color};">{rsi_val:.1f}</div>
                <div style="color: {rsi_color}; font-size: 13px; margin-top: 4px;">{rsi_status} Zone</div>
            </div>
            """, unsafe_allow_html=True)

        with tcol2:
            macd_val = float(latest_row["MACD"].values[0]) if "MACD" in latest_row else 0.0
            sig_val = float(latest_row["MACD_Signal"].values[0]) if "MACD_Signal" in latest_row else 0.0
            macd_bull = macd_val > sig_val
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MACD (12, 26, 9)</div>
                <div class="metric-value">{macd_val:,.1f}</div>
                <div style="color: {'#00F5A0' if macd_bull else '#FF4B72'}; font-size: 13px; margin-top: 4px;">
                    {'Bullish Divergence' if macd_bull else 'Bearish Divergence'} (Sig: {sig_val:,.1f})
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tcol3:
            bb_width = float(latest_row["Bollinger_Bandwidth"].values[0]) * 100 if "Bollinger_Bandwidth" in latest_row else 0.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Bollinger Bandwidth</div>
                <div class="metric-value">{bb_width:.2f}%</div>
                <div style="color: #64748B; font-size: 13px; margin-top: 4px;">Volatility Channel Width</div>
            </div>
            """, unsafe_allow_html=True)

        with tcol4:
            vol_30 = float(latest_row["Volatility_30"].values[0]) * 100 if "Volatility_30" in latest_row else 0.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">30-Day Volatility</div>
                <div class="metric-value">{vol_30:.2f}%</div>
                <div style="color: #64748B; font-size: 13px; margin-top: 4px;">Rolling Daily Std Dev</div>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================
    # 2. PRICE PREDICTION
    # ========================================================
    elif app_page == "🎯 Next-Day Price Prediction":
        st.subheader("🎯 Machine Learning Next-Day Forecast")

        forecast = predictor.predict_next_day(latest_row)

        pred_price = forecast["predicted_close_usd"]
        pred_delta = forecast["predicted_delta_usd"]
        pred_pct = forecast["predicted_pct_change"]
        direction = forecast["predicted_direction"]
        conf = forecast["direction_confidence"]

        badge_html = '<span class="bullish-badge">▲ BULLISH (Upward Trend Expected)</span>' if direction == 1 else '<span class="bearish-badge">▼ BEARISH (Downward / Consolidation Expected)</span>'

        st.markdown(f"""
        <div class="pred-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 16px;">
                <div>
                    <span style="color: #94A3B8; font-size: 13px; font-weight: 600; text-transform: uppercase;">Forecast Horizon:</span>
                    <strong style="color: #F8FAFC; margin-left: 6px;">Next Completed Daily Candle</strong>
                </div>
                <div>{badge_html}</div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px;">
                <div>
                    <div style="color: #94A3B8; font-size: 12px; font-weight: 600; text-transform: uppercase;">Current Day Close</div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; color: #94A3B8;">${forecast['current_close_usd']:,.2f}</div>
                </div>
                <div>
                    <div style="color: #38BDF8; font-size: 12px; font-weight: 600; text-transform: uppercase;">Predicted Next-Day Close</div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 28px; font-weight: 700; color: #38BDF8;">${pred_price:,.2f}</div>
                </div>
                <div>
                    <div style="color: #94A3B8; font-size: 12px; font-weight: 600; text-transform: uppercase;">Predicted Net Change</div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; color: {'#00F5A0' if pred_delta >= 0 else '#FF4B72'};">
                        {pred_delta:+,.2f} ({pred_pct:+.2f}%)
                    </div>
                </div>
                <div>
                    <div style="color: #94A3B8; font-size: 12px; font-weight: 600; text-transform: uppercase;">Model Direction Probability</div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; color: #F8FAFC;">
                        {conf * 100:.1f}% if conf else "Rule-Based"
                    </div>
                </div>
            </div>
            <div style="margin-top: 18px; padding-top: 14px; border-top: 1px solid #1E293B; font-size: 13px; color: #64748B;">
                <span><strong>Regression Architecture:</strong> {forecast['regression_model_used']}</span> &nbsp;|&nbsp;
                <span><strong>Classification Architecture:</strong> {forecast['classification_model_used']}</span>
            </div>
            <div class="disclaimer-box">
                {forecast['disclaimer']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Actual vs Predicted Chart on Unseen Test Partition
        if test_preds_df is not None:
            st.markdown("### 📉 Out-of-Sample Performance: Actual vs Predicted Prices (Test Set)")
            st.caption("Evaluated strictly on chronological out-of-sample test partition (2025 – Present).")
            fig_test = plot_interactive_actual_vs_predicted(
                dates=test_preds_df["Date"],
                actual=test_preds_df["Actual_Next_Close"],
                predicted=test_preds_df["Predicted_Next_Close"],
                model_name=meta.get("winning_regression_model", "Selected Model")
            )
            st.plotly_chart(fig_test, use_container_width=True)

    # ========================================================
    # 3. MODEL PERFORMANCE & LEADERBOARD
    # ========================================================
    elif app_page == "🏆 Model Performance & Leaderboard":
        st.subheader("🏆 Multi-Model Benchmark & Scientific Leaderboard")

        tab_reg, tab_clf, tab_feat = st.tabs(["📈 Task A: Price Regression Leaderboard", "🧭 Task B: Direction Classification Leaderboard", "🔥 Feature Importance Analysis"])

        with tab_reg:
            st.markdown("#### Regression Model Evaluation (Chronological Validation & Test Sets)")
            st.caption("Models ranked strictly by Validation RMSE to prevent test-set overfitting.")

            if reg_metrics and "leaderboard" in reg_metrics:
                reg_df = pd.DataFrame(reg_metrics["leaderboard"])
                
                # Format currency and percentages
                styled_df = reg_df.copy()
                styled_df["Val_RMSE_USD"] = styled_df["Val_RMSE_USD"].apply(lambda x: f"${x:,.2f}")
                styled_df["Val_MAE_USD"] = styled_df["Val_MAE_USD"].apply(lambda x: f"${x:,.2f}")
                styled_df["Val_MAPE_Pct"] = styled_df["Val_MAPE_Pct"].apply(lambda x: f"{x:.2f}%")
                styled_df["Val_R2"] = styled_df["Val_R2"].apply(lambda x: f"{x:.4f}")
                styled_df["Test_RMSE_USD"] = styled_df["Test_RMSE_USD"].apply(lambda x: f"${x:,.2f}")
                styled_df["Test_MAE_USD"] = styled_df["Test_MAE_USD"].apply(lambda x: f"${x:,.2f}")
                styled_df["Test_MAPE_Pct"] = styled_df["Test_MAPE_Pct"].apply(lambda x: f"{x:.2f}%")
                styled_df["Test_R2"] = styled_df["Test_R2"].apply(lambda x: f"{x:.4f}")

                st.dataframe(styled_df, use_container_width=True)

            # Winning Metrics Highlight
            if reg_metrics and "test_metrics" in reg_metrics:
                tm = reg_metrics["test_metrics"]
                st.markdown("##### 🎖️ Winning Regression Model Test Metrics")
                rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                rcol1.metric("Test MAE", f"${tm['mae']:,.2f}")
                rcol2.metric("Test RMSE", f"${tm['rmse']:,.2f}")
                rcol3.metric("Test MAPE", f"{tm['mape_pct']:.2f}%")
                rcol4.metric("Test R² Score", f"{tm['r2_score']:.4f}")

        with tab_clf:
            st.markdown("#### Classification Model Evaluation (Predicting Up vs Down Next Day)")
            st.caption("Positive Class: Next Day Close > Today's Close. Negative Class: Next Day Close <= Today's Close.")

            if clf_metrics and "leaderboard" in clf_metrics:
                clf_df = pd.DataFrame(clf_metrics["leaderboard"])
                styled_clf = clf_df.copy()
                for c in ["Val_F1_Pct", "Val_Accuracy_Pct", "Val_Precision_Pct", "Val_Recall_Pct", "Val_Specificity_Pct", "Val_ErrorRate_Pct",
                          "Test_F1_Pct", "Test_Accuracy_Pct", "Test_Precision_Pct", "Test_Recall_Pct", "Test_Specificity_Pct", "Test_ErrorRate_Pct"]:
                    styled_clf[c] = styled_clf[c].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                if "Test_ROC_AUC" in styled_clf:
                    styled_clf["Test_ROC_AUC"] = styled_clf["Test_ROC_AUC"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

                st.dataframe(styled_clf, use_container_width=True)

            if clf_metrics and "test_metrics" in clf_metrics:
                ctm = clf_metrics["test_metrics"]
                cm = ctm["confusion_matrix"]
                
                ccol1, ccol2 = st.columns([1, 1])
                with ccol1:
                    st.markdown("##### 🧮 Test Confusion Matrix Heatmap")
                    cm_fig = plot_interactive_confusion_matrix(cm, clf_metrics.get("winning_model", "Classifier"))
                    st.plotly_chart(cm_fig, use_container_width=True)

                with ccol2:
                    st.markdown("##### 📊 Exact Classification Metrics Table")
                    metric_rows = [
                        {"Metric": "Accuracy (TP+TN)/Total", "Value": ctm["formatted"]["Accuracy"]},
                        {"Metric": "Precision (TP/(TP+FP))", "Value": ctm["formatted"]["Precision"]},
                        {"Metric": "Sensitivity / Recall (TP/(TP+FN))", "Value": ctm["formatted"]["Recall / Sensitivity"]},
                        {"Metric": "Specificity (TN/(TN+FP))", "Value": ctm["formatted"]["Specificity"]},
                        {"Metric": "F1-Score", "Value": ctm["formatted"]["F1-Score"]},
                        {"Metric": "Error Rate (FP+FN)/Total", "Value": ctm["formatted"]["Error Rate"]},
                        {"Metric": "ROC-AUC Score", "Value": ctm["formatted"]["ROC-AUC"]}
                    ]
                    st.table(pd.DataFrame(metric_rows))

        with tab_feat:
            st.markdown("#### 🔥 Feature Importance Analysis (Gradient Boosted Trees)")
            if meta and "top_features" in meta and len(meta["top_features"]) > 0:
                top_fi_df = pd.DataFrame(meta["top_features"])
                fi_fig = plot_interactive_feature_importance(top_fi_df, top_n=15)
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                st.info("Feature importance is displayed for tree-based models (XGBoost, Random Forest, LightGBM).")

    # ========================================================
    # 4. HISTORICAL DATE INSPECTOR
    # ========================================================
    elif app_page == "🔍 Historical Date Inspector":
        st.subheader("🔍 Historical Point-in-Time Prediction Inspector")
        st.markdown(
            "Select any historical date. The model executes inference using **strictly** the features "
            "available on that chosen date, without any future data leakage, and compares the forecast "
            "with the actual ground truth observed on the following day."
        )

        all_dates = pd.to_datetime(processed_df["Date"]).dt.date.tolist()
        min_date = all_dates[90]  # Allow warmup window
        max_date = all_dates[-1]

        selected_date = st.date_input(
            "Select Historical Observation Date",
            value=all_dates[-5],
            min_value=min_date,
            max_value=max_date
        )

        date_str = str(selected_date)
        try:
            hist_result = predictor.predict_historical_date(processed_df, date_str)

            hcol1, hcol2, hcol3 = st.columns(3)
            with hcol1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Selected Date Close ({date_str})</div>
                    <div class="metric-value">${hist_result['current_close_usd']:,.2f}</div>
                    <div style="color: #64748B; font-size: 13px; margin-top: 4px;">Known Observation Base</div>
                </div>
                """, unsafe_allow_html=True)

            with hcol2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Predicted Next Close</div>
                    <div class="metric-value" style="color: #38BDF8;">${hist_result['predicted_close_usd']:,.2f}</div>
                    <div style="color: #38BDF8; font-size: 13px; margin-top: 4px;">
                        Expected Move: {hist_result['predicted_delta_usd']:+,.2f} ({hist_result['predicted_pct_change']:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with hcol3:
                actual_close = hist_result.get("actual_next_close_usd")
                if actual_close is not None:
                    error_usd = hist_result["prediction_error_usd"]
                    abs_pct = hist_result["absolute_pct_error"]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Actual Ground Truth ({hist_result.get('actual_next_date')})</div>
                        <div class="metric-value">${actual_close:,.2f}</div>
                        <div style="color: {'#00F5A0' if abs_pct < 2.0 else '#F59E0B'}; font-size: 13px; margin-top: 4px;">
                            Error: ${abs(error_usd):,.2f} ({abs_pct:.2f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Actual Ground Truth</div>
                        <div class="metric-value" style="color: #F59E0B;">Pending</div>
                        <div style="color: #64748B; font-size: 13px; margin-top: 4px;">Future candle not completed</div>
                    </div>
                    """, unsafe_allow_html=True)

            if hist_result.get("actual_next_close_usd") is not None:
                dir_correct = hist_result["direction_correct"]
                status_color = "#00F5A0" if dir_correct else "#FF4B72"
                st.markdown(f"""
                <div style="margin-top: 16px; padding: 14px; border-radius: 8px; background-color: rgba({ '0, 245, 160, 0.1' if dir_correct else '255, 75, 114, 0.1' }); border: 1px solid {status_color};">
                    <strong style="color: {status_color};">Direction Evaluation:</strong>
                    Predicted <strong>{hist_result['direction_label']}</strong> | 
                    Actual Ground Truth was <strong>{hist_result['actual_direction_label']}</strong> | 
                    Status: <strong style="color: {status_color};">{'CORRECT PREDICTION' if dir_correct else 'INCORRECT PREDICTION'}</strong>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Inference error for {date_str}: {e}")

    # ========================================================
    # 5. PIPELINE RETRAINING
    # ========================================================
    elif app_page == "⚙️ Pipeline Retraining":
        st.subheader("⚙️ Automated Model Training & Experiment Execution")
        st.markdown(
            "Trigger the complete machine learning lifecycle autonomously. "
            "Fetches fresh Bitcoin candles, computes all 58 technical features, partitions data chronologically, "
            "optimizes hyperparameters via TimeSeriesSplit, benchmarks 8 regressors and 7 classifiers, "
            "and serializes updated artifacts to `models/`."
        )

        if st.button("🚀 Run Complete Training Pipeline", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.info("Executing end-to-end ML pipeline...")
                progress_bar.progress(25)
                run_pipeline()
                progress_bar.progress(100)
                status_text.success("Pipeline executed successfully! Fresh models and reports generated.")
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                status_text.error(f"Pipeline execution encountered an error: {e}")

    # ========================================================
    # 6. DOWNLOADS & REPORTS
    # ========================================================
    elif app_page == "📥 Data & Reports Download":
        st.subheader("📥 Academic Deliverables & Exportable Data")
        st.markdown("Download raw/processed datasets, out-of-sample predictions, and experiment reports for your college submission.")

        dcol1, dcol2 = st.columns(2)

        with dcol1:
            if PROCESSED_DATA_FILE.exists():
                with open(PROCESSED_DATA_FILE, "rb") as f:
                    st.download_button(
                        label="📄 Download Processed Dataset (58 Features CSV)",
                        data=f,
                        file_name="bitcoin_features_dataset.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            if TEST_PREDICTIONS_FILE.exists():
                with open(TEST_PREDICTIONS_FILE, "rb") as f:
                    st.download_button(
                        label="🎯 Download Test Predictions (Actual vs Predicted CSV)",
                        data=f,
                        file_name="test_predictions_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        with dcol2:
            if MODEL_COMPARISON_FILE.exists():
                with open(MODEL_COMPARISON_FILE, "rb") as f:
                    st.download_button(
                        label="📊 Download Model Comparison Leaderboard (CSV)",
                        data=f,
                        file_name="model_comparison_leaderboard.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            if EXPERIMENT_SUMMARY_FILE.exists():
                with open(EXPERIMENT_SUMMARY_FILE, "r", encoding="utf-8") as f:
                    report_content = f.read()
                    st.download_button(
                        label="📑 Download Experiment Summary Report (Markdown)",
                        data=report_content,
                        file_name="experiment_summary_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

        if EXPERIMENT_SUMMARY_FILE.exists():
            st.markdown("---")
            st.markdown("### 📑 Embedded Experiment Summary Report")
            with open(EXPERIMENT_SUMMARY_FILE, "r", encoding="utf-8") as f:
                st.markdown(f.read())


if __name__ == "__main__":
    main()
