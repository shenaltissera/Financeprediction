import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

sys.path.insert(0, ".")
from data_pipeline import load
from models.xgboost_model import FinanceXGB, FEATURE_COLS
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
run_btn = st.sidebar.button("Run Prediction", type="primary")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("Finance Predictor — XGBoost + LSTM")
st.caption("Stock & crypto price direction, % change, buy/sell signal, and backtesting.")

if not run_btn:
    st.info("Select an asset in the sidebar and click **Run Prediction**.")
    st.stop()

with st.spinner("Fetching data and engineering features..."):
    try:
        df = load(ticker, asset_type=asset_type.lower())
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

split = int(len(df) * train_split / 100)
train_df, test_df = df.iloc[:split], df.iloc[split:]

with st.spinner("Training XGBoost model..."):
    model = FinanceXGB().fit(train_df)

prediction = model.predict(df)
metrics = model.evaluate(test_df)

# ── Prediction cards ───────────────────────────────────────────────────────────
direction_label = "🟢 UP" if prediction["direction"] == 1 else "🔴 DOWN"
signal_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

col1, col2, col3, col4 = st.columns(4)
col1.metric("Next-day Direction", direction_label,
            f"Confidence: {prediction['direction_proba']:.1%}")
col2.metric("Predicted Price", f"${prediction['price']:,.2f}",
            f"{prediction['pct_change']:+.2%}")
col3.metric("% Change", f"{prediction['pct_change']:+.2%}")
col4.metric("Signal", signal_map[prediction["signal"]])

st.divider()

# ── Model metrics ──────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.1%}")
m2.metric("Direction F1", f"{metrics['direction_f1']:.3f}")
m3.metric("% Change MAE", f"{metrics['pct_mae']:.4f}")
m4.metric("Price RMSE", f"${metrics['price_rmse']:,.2f}")

st.divider()

# ── Price chart ────────────────────────────────────────────────────────────────
st.subheader("Price History & Technical Indicators")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=("Price + Bollinger Bands", "RSI", "MACD"))

fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                              low=df["low"], close=df["close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], line=dict(color="gray", dash="dot"),
                          name="BB Upper"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], line=dict(color="gray", dash="dot"),
                          fill="tonexty", fillcolor="rgba(128,128,128,0.1)", name="BB Lower"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["sma_30"], line=dict(color="blue"), name="SMA 30"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], line=dict(color="purple"), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"], name="MACD Diff",
                      marker_color=np.where(df["macd_diff"] >= 0, "green", "red")), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["macd"], line=dict(color="blue"), name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], line=dict(color="orange"), name="Signal"), row=3, col=1)

fig.update_layout(height=700, xaxis_rangeslider_visible=False, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ── Backtest ───────────────────────────────────────────────────────────────────
st.subheader("Backtest Results (test set)")
signals = pd.Series(
    model.signal_model.predict(model.scaler.transform(test_df[FEATURE_COLS].values)) - 1,
    index=test_df.index,
)
bt_metrics, portfolio = run_backtest(test_df, signals)

b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("Strategy Return", f"{bt_metrics['total_return']}%",
          f"vs B&H {bt_metrics['buy_hold_return']}%")
b2.metric("Ann. Return", f"{bt_metrics['annualized_return']}%")
b3.metric("Sharpe Ratio", bt_metrics["sharpe_ratio"])
b4.metric("Max Drawdown", f"{bt_metrics['max_drawdown']}%")
b5.metric("Win Rate", f"{bt_metrics['win_rate']}%")

eq_fig = go.Figure()
eq_fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["equity"],
                             name="Strategy", line=dict(color="blue")))
eq_fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["buy_hold_equity"],
                             name="Buy & Hold", line=dict(color="gray", dash="dash")))
eq_fig.update_layout(title="Equity Curve", height=350, yaxis_title="Portfolio Value ($)")
st.plotly_chart(eq_fig, use_container_width=True)
