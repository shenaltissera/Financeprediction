import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "sma_10", "sma_30", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "bb_upper", "bb_lower", "bb_pct",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
    "volume_lag_1", "volume_lag_2", "volume_lag_3", "volume_lag_5", "volume_lag_10",
    "volatility_5", "volatility_20",
    "daily_return", "high_low_range", "close_open_range",
]


def prepare(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, scaler


class FinanceXGB:
    def __init__(self):
        self.direction_model = XGBClassifier(n_estimators=200, learning_rate=0.05,
                                             max_depth=6, subsample=0.8,
                                             colsample_bytree=0.8, random_state=42,
                                             eval_metric="logloss")
        self.pct_model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                      max_depth=6, subsample=0.8,
                                      colsample_bytree=0.8, random_state=42)
        self.price_model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                        max_depth=6, subsample=0.8,
                                        colsample_bytree=0.8, random_state=42)
        self.signal_model = XGBClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=6, subsample=0.8,
                                          colsample_bytree=0.8, random_state=42,
                                          eval_metric="mlogloss")
        self.scaler = None

    def fit(self, df: pd.DataFrame):
        X, self.scaler = prepare(df)
        self.direction_model.fit(X, df["target_direction"])
        self.pct_model.fit(X, df["target_pct_change"])
        self.price_model.fit(X, df["target_price"])
        # shift signal labels to 0,1,2 for XGB multiclass
        self.signal_model.fit(X, df["target_signal"] + 1)
        return self

    def predict(self, df: pd.DataFrame) -> dict:
        X = self.scaler.transform(df[FEATURE_COLS].values)
        return {
            "direction": int(self.direction_model.predict(X)[-1]),
            "direction_proba": float(self.direction_model.predict_proba(X)[-1][1]),
            "pct_change": float(self.pct_model.predict(X)[-1]),
            "price": float(self.price_model.predict(X)[-1]),
            "signal": int(self.signal_model.predict(X)[-1]) - 1,  # back to -1,0,1
        }

    def evaluate(self, df: pd.DataFrame) -> dict:
        X = self.scaler.transform(df[FEATURE_COLS].values)
        direction_pred = self.direction_model.predict(X)
        pct_pred = self.pct_model.predict(X)
        price_pred = self.price_model.predict(X)
        return {
            "direction_accuracy": accuracy_score(df["target_direction"], direction_pred),
            "direction_f1": f1_score(df["target_direction"], direction_pred),
            "pct_mae": mean_absolute_error(df["target_pct_change"], pct_pred),
            "price_rmse": mean_squared_error(df["target_price"], price_pred, squared=False),
        }

    def save(self, path: str = "models/xgb"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.direction_model, f"{path}/direction.pkl")
        joblib.dump(self.pct_model, f"{path}/pct.pkl")
        joblib.dump(self.price_model, f"{path}/price.pkl")
        joblib.dump(self.signal_model, f"{path}/signal.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")

    def load(self, path: str = "models/xgb"):
        self.direction_model = joblib.load(f"{path}/direction.pkl")
        self.pct_model = joblib.load(f"{path}/pct.pkl")
        self.price_model = joblib.load(f"{path}/price.pkl")
        self.signal_model = joblib.load(f"{path}/signal.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        return self


def train_with_cv(df: pd.DataFrame, n_splits: int = 5):
    model = FinanceXGB()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_all, scaler = prepare(df)
    results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        m = FinanceXGB()
        m.fit(train_df)
        metrics = m.evaluate(test_df)
        metrics["fold"] = fold + 1
        results.append(metrics)
        print(f"Fold {fold+1}: {metrics}")
    # final model on all data
    model.fit(df)
    return model, pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import load

    df = load("BTC/USDT", asset_type="crypto")
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    print("Training XGBoost models...")
    model = FinanceXGB().fit(train_df)
    metrics = model.evaluate(test_df)
    print("Test metrics:", metrics)
    print("Latest prediction:", model.predict(test_df))
    model.save()
