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
from models.lstm_model import FinanceLSTM, TF_AVAILABLE

XGB_WEIGHT = 0.4
LSTM_WEIGHT = 0.6


class EnsemblePredictor:
    def __init__(self, lstm_epochs: int = 20, buy_threshold: float = 0.55, sell_threshold: float = 0.45):
        self.xgb = FinanceXGB(buy_threshold=buy_threshold, sell_threshold=sell_threshold)
        self.lstm = FinanceLSTM()
        self.lstm_epochs = lstm_epochs
        self._lstm_ok = False  # flag if LSTM trained successfully

    def fit(self, train_df: pd.DataFrame, progress_cb=None):
        if progress_cb:
            progress_cb("Training XGBoost...", 0.0)
        self.xgb.fit(train_df)

        if progress_cb:
            progress_cb("Training LSTM...", 0.4)
        if not TF_AVAILABLE:
            print("[Ensemble] TensorFlow not available. Using XGBoost only.")
            self._lstm_ok = False
        else:
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

        # Signal: use ensemble probability with same thresholds as XGBoost
        buy_t = self.xgb.buy_threshold
        sell_t = self.xgb.sell_threshold
        ens_signal = 1 if ens_up_prob >= buy_t else (-1 if ens_up_prob <= sell_t else 0)

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
        """Generate per-row signals for the test set using direction probability thresholds."""
        return self.xgb.get_signals(test_df)
