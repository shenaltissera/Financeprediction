import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

SEQUENCE_LEN = 60
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "macd", "macd_diff", "bb_pct",
    "daily_return", "volatility_5",
]


def build_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 3])  # close price index
    return np.array(X), np.array(y)


class FinanceLSTM:
    def __init__(self, seq_len: int = SEQUENCE_LEN):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        self.model = None

    def _build_model(self, n_features: int):
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError("Install tensorflow: pip install tensorflow")

        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.seq_len, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def fit(self, df: pd.DataFrame, epochs: int = 30, batch_size: int = 32):
        data = df[FEATURE_COLS].values
        data_scaled = self.scaler.fit_transform(data)
        X, y = build_sequences(data_scaled, self.seq_len)

        split = int(len(X) * 0.9)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.model = self._build_model(len(FEATURE_COLS))
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        return self

    def predict_next(self, df: pd.DataFrame) -> dict:
        data = df[FEATURE_COLS].values[-self.seq_len:]
        data_scaled = self.scaler.transform(data)
        X = data_scaled[np.newaxis, :, :]

        # inverse transform prediction back to price scale
        pred_scaled = self.model.predict(X, verbose=0)[0][0]
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, 3] = pred_scaled
        price = self.scaler.inverse_transform(dummy)[0, 3]

        current_price = df["close"].iloc[-1]
        return {
            "price": float(price),
            "direction": int(price > current_price),
            "pct_change": float((price - current_price) / current_price),
        }

    def save(self, path: str = "models/lstm"):
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/model.keras")
        import joblib
        joblib.dump(self.scaler, f"{path}/scaler.pkl")

    def load(self, path: str = "models/lstm"):
        from tensorflow import keras
        import joblib
        self.model = keras.models.load_model(f"{path}/model.keras")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        return self


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import load

    df = load("BTC/USDT", asset_type="crypto")
    print(f"Data shape: {df.shape}")
    print("Training LSTM...")
    model = FinanceLSTM()
    model.fit(df, epochs=10)
    print("Prediction:", model.predict_next(df))
    model.save()
