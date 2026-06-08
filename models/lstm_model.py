import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TF_AVAILABLE = True  # kept as TF_AVAILABLE for compatibility with existing imports
except ImportError:
    TF_AVAILABLE = False

SEQUENCE_LEN = 60
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "macd", "macd_diff", "bb_pct",
    "daily_return", "volatility_5",
]
CLOSE_IDX = FEATURE_COLS.index("close")


class _LSTMNet(nn.Module if TF_AVAILABLE else object):
    def __init__(self, n_features: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden, 64, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])  # last timestep
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


def _build_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, CLOSE_IDX])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class FinanceLSTM:
    def __init__(self, seq_len: int = SEQUENCE_LEN):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        self.net = None
        self.device = "mps" if (TF_AVAILABLE and torch.backends.mps.is_available()) else "cpu"

    def fit(self, df: pd.DataFrame, epochs: int = 20, batch_size: int = 32):
        data = self.scaler.fit_transform(df[FEATURE_COLS].values)
        X, y = _build_sequences(data, self.seq_len)

        split = int(len(X) * 0.9)
        loader = DataLoader(
            TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split])),
            batch_size=batch_size, shuffle=False,
        )
        val_X = torch.tensor(X[split:]).to(self.device)
        val_y = torch.tensor(y[split:]).to(self.device)

        self.net = _LSTMNet(len(FEATURE_COLS)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.net.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                opt.step()

            if (epoch + 1) % 5 == 0:
                self.net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.net(val_X), val_y).item()
                print(f"Epoch {epoch+1}/{epochs}  val_loss={val_loss:.6f}")

        return self

    def predict_next(self, df: pd.DataFrame) -> dict:
        data = self.scaler.transform(df[FEATURE_COLS].values[-self.seq_len:])
        x = torch.tensor(data[np.newaxis], dtype=torch.float32).to(self.device)

        self.net.eval()
        with torch.no_grad():
            pred_scaled = self.net(x).item()

        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, CLOSE_IDX] = pred_scaled
        price = float(self.scaler.inverse_transform(dummy)[0, CLOSE_IDX])
        current = float(df["close"].iloc[-1])
        return {
            "price": price,
            "direction": int(price > current),
            "pct_change": (price - current) / current,
        }

    def save(self, path: str = "models/lstm"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.net.state_dict(), f"{path}/model.pt")
        import joblib
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump({"seq_len": self.seq_len, "n_features": len(FEATURE_COLS)}, f"{path}/config.pkl")

    def load(self, path: str = "models/lstm"):
        import joblib
        cfg = joblib.load(f"{path}/config.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.net = _LSTMNet(cfg["n_features"]).to(self.device)
        self.net.load_state_dict(torch.load(f"{path}/model.pt", map_location=self.device))
        self.net.eval()
        return self


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import load

    df = load("BTC/USDT", asset_type="crypto")
    print(f"Data shape: {df.shape}, device: {FinanceLSTM().device}")
    model = FinanceLSTM()
    model.fit(df, epochs=10)
    print("Prediction:", model.predict_next(df))
    model.save()
