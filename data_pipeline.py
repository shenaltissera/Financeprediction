import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def fetch_stock(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    # yfinance ≥0.2 returns a MultiIndex (metric, ticker); flatten to single-level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df.dropna()


def fetch_crypto(symbol: str = "BTC/USDT", timeframe: str = "1d", limit: int = 730) -> pd.DataFrame:
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df.set_index("date", inplace=True)
    return df.dropna()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    # Moving averages
    df["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    df["sma_30"] = SMAIndicator(close, window=30).sma_indicator()
    df["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(close, window=26).ema_indicator()

    # Momentum
    df["rsi_14"] = RSIIndicator(close, window=14).rsi()

    # MACD
    macd = MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = close.shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

    # Rolling volatility
    df["volatility_5"] = close.pct_change().rolling(5).std()
    df["volatility_20"] = close.pct_change().rolling(20).std()

    # Price-derived features
    df["daily_return"] = close.pct_change()
    df["high_low_range"] = (df["high"] - df["low"]) / close
    df["close_open_range"] = (close - df["open"]) / df["open"]

    # Targets
    df["target_direction"] = (close.shift(-1) > close).astype(int)   # 1=up, 0=down
    df["target_pct_change"] = close.pct_change(-1) * -1               # next-day % change
    df["target_price"] = close.shift(-1)                              # next-day price
    df["target_signal"] = np.where(close.shift(-1) > close * 1.01, 1,
                           np.where(close.shift(-1) < close * 0.99, -1, 0))  # buy/hold/sell

    return df.dropna()


def load(ticker: str, asset_type: str = "crypto") -> pd.DataFrame:
    """asset_type: 'stock' or 'crypto'"""
    if asset_type == "crypto":
        df = fetch_crypto(ticker)
    else:
        df = fetch_stock(ticker)
    return engineer_features(df)


if __name__ == "__main__":
    df = load("BTC/USDT", asset_type="crypto")
    print(df.shape)
    print(df.tail(3)[["close", "rsi_14", "macd", "target_direction", "target_pct_change"]])
