"""
Ensemble: XGBoost + LSTM hybrid predictor.

Strategy:
  - Price prediction  → weighted average (XGB 40%, LSTM 60% — LSTM better at sequences)
  - Direction         → probability-weighted vote
  - % change          → weighted average
  - Signal            → majority vote (direction proba + LSTM direction as tiebreak)
  - Confidence        → average of both model confidences
"""
import numpy as np
import pandas as pd
from models.xgboost_model import FinanceXGB
from models.lstm_model import FinanceLSTM

XGB_WEIGHT = 0.4
LSTM_WEIGHT = 0.6


class EnsemblePredictor:
    def __init__(self, lstm_epochs: int = 20):
        self.xgb = FinanceXGB()
        self.lstm = FinanceLSTM()
        self.lstm_epochs = lstm_epochs
        self._lstm_ok = False  # flag if LSTM trained successfully

    def fit(self, train_df: pd.DataFrame, progress_cb=None):
        if progress_cb:
            progress_cb("Training XGBoost...", 0.0)
        self.xgb.fit(train_df)

        if progress_cb:
            progress_cb("Training LSTM...", 0.4)
        try:
            self.lstm.fit(train_df, epochs=self.lstm_epochs, batch_size=32)
            self._lstm_ok = True
        except Exception as e:
            print(f"[Ensemble] LSTM training failed: {e}. Using XGBoost only.")
            self._lstm_ok = False

        if progress_cb:
            progress_cb("Done!", 1.0)
        return self

    def predict(self, df: pd.DataFrame) -> dict:
        xgb_pred = self.xgb.predict(df)

        if not self._lstm_ok:
            xgb_pred["model"] = "XGBoost only (LSTM unavailable)"
            xgb_pred["lstm_price"] = None
            xgb_pred["xgb_price"] = xgb_pred["price"]
            return xgb_pred

        lstm_pred = self.lstm.predict_next(df)

        # Weighted price & pct_change
        ens_price = XGB_WEIGHT * xgb_pred["price"] + LSTM_WEIGHT * lstm_pred["price"]
        ens_pct = XGB_WEIGHT * xgb_pred["pct_change"] + LSTM_WEIGHT * lstm_pred["pct_change"]

        # Direction: weighted probability vote
        xgb_up_prob = xgb_pred["direction_proba"]
        lstm_up_prob = float(lstm_pred["direction"])  # 0 or 1 — use as soft vote
        ens_up_prob = XGB_WEIGHT * xgb_up_prob + LSTM_WEIGHT * lstm_up_prob
        ens_direction = int(ens_up_prob >= 0.5)

        # Signal: if both agree use that, else use ensemble direction
        xgb_sig = xgb_pred["signal"]
        lstm_sig = 1 if lstm_pred["direction"] == 1 else -1
        if xgb_sig == lstm_sig:
            ens_signal = xgb_sig
        else:
            # tiebreak: trust direction probability
            ens_signal = 1 if ens_up_prob > 0.55 else (-1 if ens_up_prob < 0.45 else 0)

        # Confidence: how far from 0.5 the ensemble is
        confidence = abs(ens_up_prob - 0.5) * 2  # 0=uncertain, 1=certain

        return {
            "price": ens_price,
            "pct_change": ens_pct,
            "direction": ens_direction,
            "direction_proba": ens_up_prob,
            "signal": ens_signal,
            "confidence": confidence,
            "xgb_price": xgb_pred["price"],
            "lstm_price": lstm_pred["price"],
            "xgb_direction_proba": xgb_up_prob,
            "lstm_direction_proba": lstm_up_prob,
            "model": "XGBoost + LSTM Ensemble",
        }

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """XGBoost test metrics (LSTM eval is done during training via val loss)."""
        return self.xgb.evaluate(test_df)

    def get_test_signals(self, test_df: pd.DataFrame) -> pd.Series:
        """Generate per-row signals for the test set (XGBoost, fast)."""
        from models.xgboost_model import FEATURE_COLS
        raw = self.xgb.signal_model.predict(
            self.xgb.scaler.transform(test_df[FEATURE_COLS].values)
        ) - 1
        return pd.Series(raw, index=test_df.index)
