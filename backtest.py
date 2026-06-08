import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_backtest(df: pd.DataFrame, signals: pd.Series, initial_capital: float = 10_000.0) -> dict:
    """
    signals: pd.Series aligned with df index, values in {-1, 0, 1}
    Returns performance metrics and equity curve.
    """
    prices = df["close"].copy()
    portfolio = pd.DataFrame(index=prices.index)
    portfolio["price"] = prices
    portfolio["signal"] = signals.reindex(prices.index).fillna(0)
    portfolio["position"] = portfolio["signal"].shift(1).fillna(0)  # execute next day

    portfolio["daily_return"] = prices.pct_change()
    portfolio["strategy_return"] = portfolio["position"] * portfolio["daily_return"]

    portfolio["equity"] = initial_capital * (1 + portfolio["strategy_return"]).cumprod()
    portfolio["buy_hold_equity"] = initial_capital * (1 + portfolio["daily_return"]).cumprod()

    total_return = (portfolio["equity"].iloc[-1] / initial_capital) - 1
    bh_return = (portfolio["buy_hold_equity"].iloc[-1] / initial_capital) - 1
    n_days = len(portfolio)
    ann_return = (1 + total_return) ** (252 / n_days) - 1

    rolling_max = portfolio["equity"].cummax()
    drawdown = (portfolio["equity"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    daily_std = portfolio["strategy_return"].std()
    sharpe = (portfolio["strategy_return"].mean() / daily_std * np.sqrt(252)
              if daily_std > 0 else 0)

    trades = portfolio["position"].diff().abs().sum() / 2
    win_days = (portfolio["strategy_return"] > 0).sum()
    loss_days = (portfolio["strategy_return"] < 0).sum()
    win_rate = win_days / (win_days + loss_days) if (win_days + loss_days) > 0 else 0

    metrics = {
        "total_return": round(total_return * 100, 2),
        "buy_hold_return": round(bh_return * 100, 2),
        "annualized_return": round(ann_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "num_trades": int(trades),
        "final_equity": round(portfolio["equity"].iloc[-1], 2),
    }

    return metrics, portfolio


def plot_backtest(portfolio: pd.DataFrame, ticker: str = "Asset"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(portfolio["equity"], label="Strategy", color="blue")
    axes[0].plot(portfolio["buy_hold_equity"], label="Buy & Hold", color="gray", linestyle="--")
    axes[0].set_title(f"{ticker} — Equity Curve")
    axes[0].legend()
    axes[0].set_ylabel("Portfolio Value ($)")

    axes[1].plot(portfolio["price"], color="black", linewidth=0.8)
    buy = portfolio[portfolio["signal"] == 1]
    sell = portfolio[portfolio["signal"] == -1]
    axes[1].scatter(buy.index, buy["price"], marker="^", color="green", s=40, label="Buy")
    axes[1].scatter(sell.index, sell["price"], marker="v", color="red", s=40, label="Sell")
    axes[1].set_title("Price with Signals")
    axes[1].legend()
    axes[1].set_ylabel("Price")

    rolling_max = portfolio["equity"].cummax()
    drawdown = (portfolio["equity"] - rolling_max) / rolling_max * 100
    axes[2].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4)
    axes[2].set_title("Drawdown (%)")
    axes[2].set_ylabel("Drawdown %")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import load
    from models.xgboost_model import FinanceXGB

    df = load("BTC/USDT", asset_type="crypto")
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    model = FinanceXGB().fit(train_df)
    X_test = test_df[["open","high","low","close","volume","sma_10","sma_30","ema_12","ema_26",
                       "rsi_14","macd","macd_signal","macd_diff","bb_upper","bb_lower","bb_pct",
                       "close_lag_1","close_lag_2","close_lag_3","close_lag_5","close_lag_10",
                       "volume_lag_1","volume_lag_2","volume_lag_3","volume_lag_5","volume_lag_10",
                       "volatility_5","volatility_20","daily_return","high_low_range","close_open_range"]]
    signals = pd.Series(model.signal_model.predict(model.scaler.transform(X_test.values)) - 1,
                        index=test_df.index)

    metrics, portfolio = run_backtest(test_df, signals)
    print("Backtest results:", metrics)
    fig = plot_backtest(portfolio, ticker="BTC/USDT")
    fig.savefig("backtest_result.png", dpi=150)
    print("Chart saved to backtest_result.png")
