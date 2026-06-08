import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

sys.path.insert(0, ".")
from data_pipeline import load
from models.ensemble import EnsemblePredictor
from models.lstm_model import TF_AVAILABLE
from backtest import run_backtest

st.set_page_config(page_title="Finance Predictor", page_icon="📈", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Finance Predictor")
asset_type = st.sidebar.radio("Asset type", ["Crypto", "Stock"])

if asset_type == "Crypto":
    ticker = st.sidebar.selectbox("Ticker", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
else:
    ticker = st.sidebar.text_input("Ticker (e.g. AAPL, TSLA)", value="AAPL").upper()

train_split = st.sidebar.slider("Train/test split (%)", 60, 90, 80)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Settings")
if TF_AVAILABLE:
    use_lstm = st.sidebar.toggle("Enable LSTM (slower)", value=True)
    lstm_epochs = st.sidebar.slider("LSTM epochs", 5, 50, 20,
                                     help="More epochs = better LSTM, but slower training")
else:
    use_lstm = False
    lstm_epochs = 0
    st.sidebar.info("ℹ️ LSTM unavailable (TensorFlow not installed). Running XGBoost only.")

run_btn = st.sidebar.button("Run Prediction", type="primary")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("📈 Finance Predictor — XGBoost + LSTM Ensemble")
st.caption("Hybrid ML model: XGBoost for feature-based prediction + LSTM for sequence modelling.")

if not run_btn:
    st.info("👈 Select an asset in the sidebar and click **Run Prediction**.")
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Fetching data and engineering features..."):
    try:
        df = load(ticker, asset_type=asset_type.lower())
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

split = int(len(df) * train_split / 100)
train_df, test_df = df.iloc[:split], df.iloc[split:]

# ── Train ensemble ─────────────────────────────────────────────────────────────
progress_bar = st.progress(0.0, text="Starting training...")

def update_progress(msg: str, pct: float):
    progress_bar.progress(pct, text=msg)

ensemble = EnsemblePredictor(lstm_epochs=lstm_epochs if use_lstm else 0)
if not use_lstm:
    ensemble._lstm_ok = False
    ensemble.xgb.fit(train_df)
    progress_bar.progress(1.0, text="XGBoost trained.")
else:
    ensemble.fit(train_df, progress_cb=update_progress)

progress_bar.empty()

# ── Predictions ────────────────────────────────────────────────────────────────
pred = ensemble.predict(df)
metrics = ensemble.evaluate(test_df)

signal_map = {1: ("🟢 BUY", "green"), 0: ("🟡 HOLD", "orange"), -1: ("🔴 SELL", "red")}
dir_label = "🟢 UP" if pred["direction"] == 1 else "🔴 DOWN"
sig_label, sig_color = signal_map[pred["signal"]]

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Chart", "⚖️ Model Comparison", "📉 Backtest"])

# ────────────────────────────────────────────────────────────────────
# TAB 1 — Prediction
# ────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Next-Day Forecast — {ticker}")
    st.caption(f"Model: {pred.get('model', 'Ensemble')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Direction", dir_label,
                f"Confidence {pred['direction_proba']:.1%}")
    col2.metric("Ensemble Price", f"${pred['price']:,.2f}",
                f"{pred['pct_change']:+.2%}")
    col3.metric("% Change", f"{pred['pct_change']:+.2%}")
    col4.metric("Signal", sig_label)

    # Confidence gauge
    if pred.get("confidence") is not None:
        st.markdown("#### Ensemble Confidence")
        conf_pct = pred["confidence"] * 100
        conf_color = "green" if conf_pct > 60 else "orange" if conf_pct > 35 else "red"
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf_pct,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": conf_color},
                "steps": [
                    {"range": [0, 35], "color": "rgba(255,80,80,0.15)"},
                    {"range": [35, 60], "color": "rgba(255,200,0,0.15)"},
                    {"range": [60, 100], "color": "rgba(0,200,80,0.15)"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 60},
            },
            title={"text": "Prediction Confidence"},
        ))
        gauge.update_layout(height=280)
        st.plotly_chart(gauge, use_container_width=True)

    st.divider()
    st.subheader("Test Set Model Metrics (XGBoost)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.1%}")
    m2.metric("Direction F1", f"{metrics['direction_f1']:.3f}")
    m3.metric("% Change MAE", f"{metrics['pct_mae']:.4f}")
    m4.metric("Price RMSE", f"${metrics['price_rmse']:,.2f}")

# ────────────────────────────────────────────────────────────────────
# TAB 2 — Chart
# ────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Price History & Technical Indicators")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=("Price + Bollinger Bands", "RSI (14)", "MACD"))

    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                  low=df["low"], close=df["close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"],
                              line=dict(color="gray", dash="dot"), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"],
                              line=dict(color="gray", dash="dot"),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.08)",
                              name="BB Lower"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_30"],
                              line=dict(color="royalblue"), name="SMA 30"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_12"],
                              line=dict(color="orange", dash="dot"), name="EMA 12"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"],
                              line=dict(color="purple"), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="purple", opacity=0.04, row=2, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"], name="MACD Hist",
                          marker_color=np.where(df["macd_diff"] >= 0, "#26a69a", "#ef5350")),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
                              line=dict(color="royalblue"), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
                              line=dict(color="orange"), name="Signal Line"), row=3, col=1)

    fig.update_layout(height=720, xaxis_rangeslider_visible=False,
                      template="plotly_dark", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────
# TAB 3 — Model Comparison
# ────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("XGBoost vs LSTM vs Ensemble")

    if pred.get("lstm_price") is None:
        st.warning("LSTM is disabled or failed to train. Enable it in the sidebar to see comparisons.")
    else:
        current = float(df["close"].iloc[-1])
        rows = {
            "Model": ["XGBoost", "LSTM", "🏆 Ensemble"],
            "Predicted Price": [
                f"${pred['xgb_price']:,.2f}",
                f"${pred['lstm_price']:,.2f}",
                f"${pred['price']:,.2f}",
            ],
            "Direction": [
                "🟢 UP" if pred["xgb_direction_proba"] >= 0.5 else "🔴 DOWN",
                "🟢 UP" if pred["lstm_direction_proba"] >= 0.5 else "🔴 DOWN",
                "🟢 UP" if pred["direction"] == 1 else "🔴 DOWN",
            ],
            "Up Probability": [
                f"{pred['xgb_direction_proba']:.1%}",
                f"{pred['lstm_direction_proba']:.1%}",
                f"{pred['direction_proba']:.1%}",
            ],
            "Weight": ["40%", "60%", "—"],
        }
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Side-by-side price bar chart
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(name="XGBoost", x=["Predicted Price"],
                                  y=[pred["xgb_price"]], marker_color="royalblue"))
        bar_fig.add_trace(go.Bar(name="LSTM", x=["Predicted Price"],
                                  y=[pred["lstm_price"]], marker_color="orange"))
        bar_fig.add_trace(go.Bar(name="Ensemble", x=["Predicted Price"],
                                  y=[pred["price"]], marker_color="limegreen"))
        bar_fig.add_hline(y=current, line_dash="dot", line_color="white",
                           annotation_text=f"Current: ${current:,.2f}")
        bar_fig.update_layout(barmode="group", title="Price Predictions vs Current",
                               height=380, template="plotly_dark")
        st.plotly_chart(bar_fig, use_container_width=True)

        # Up-probability comparison
        prob_fig = go.Figure(go.Bar(
            x=["XGBoost", "LSTM", "Ensemble"],
            y=[pred["xgb_direction_proba"] * 100,
               pred["lstm_direction_proba"] * 100,
               pred["direction_proba"] * 100],
            marker_color=["royalblue", "orange", "limegreen"],
            text=[f"{v:.1f}%" for v in [pred["xgb_direction_proba"] * 100,
                                         pred["lstm_direction_proba"] * 100,
                                         pred["direction_proba"] * 100]],
            textposition="auto",
        ))
        prob_fig.add_hline(y=50, line_dash="dot", line_color="white",
                            annotation_text="50% threshold")
        prob_fig.update_layout(title="Up Probability by Model (%)",
                                height=340, template="plotly_dark",
                                yaxis=dict(range=[0, 100]))
        st.plotly_chart(prob_fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────
# TAB 4 — Backtest
# ────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Backtest Results — Test Set")
    signals = ensemble.get_test_signals(test_df)
    bt_metrics, portfolio = run_backtest(test_df, signals)

    b1, b2, b3, b4, b5 = st.columns(5)
    delta_vs_bh = bt_metrics["total_return"] - bt_metrics["buy_hold_return"]
    b1.metric("Strategy Return", f"{bt_metrics['total_return']}%",
              f"{delta_vs_bh:+.2f}% vs B&H")
    b2.metric("Annualised Return", f"{bt_metrics['annualized_return']}%")
    b3.metric("Sharpe Ratio", bt_metrics["sharpe_ratio"])
    b4.metric("Max Drawdown", f"{bt_metrics['max_drawdown']}%")
    b5.metric("Win Rate", f"{bt_metrics['win_rate']}%")

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["equity"],
                                 name="Strategy", line=dict(color="#26a69a", width=2)))
    eq_fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["buy_hold_equity"],
                                 name="Buy & Hold", line=dict(color="gray", dash="dash")))

    # buy/sell markers on equity curve
    buys = portfolio[portfolio["signal"] == 1]
    sells = portfolio[portfolio["signal"] == -1]
    eq_fig.add_trace(go.Scatter(x=buys.index, y=buys["equity"], mode="markers",
                                 name="Buy signal", marker=dict(color="lime", size=6, symbol="triangle-up")))
    eq_fig.add_trace(go.Scatter(x=sells.index, y=sells["equity"], mode="markers",
                                 name="Sell signal", marker=dict(color="red", size=6, symbol="triangle-down")))

    eq_fig.update_layout(title="Equity Curve vs Buy & Hold",
                          height=420, template="plotly_dark",
                          yaxis_title="Portfolio Value ($)")
    st.plotly_chart(eq_fig, use_container_width=True)

    # Drawdown chart
    rolling_max = portfolio["equity"].cummax()
    drawdown = (portfolio["equity"] - rolling_max) / rolling_max * 100
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown,
                                 fill="tozeroy", fillcolor="rgba(239,83,80,0.3)",
                                 line=dict(color="#ef5350"), name="Drawdown"))
    dd_fig.update_layout(title="Drawdown (%)", height=260,
                          template="plotly_dark", yaxis_title="%")
    st.plotly_chart(dd_fig, use_container_width=True)

    with st.expander("📋 Full Backtest Stats"):
        st.json(bt_metrics)
