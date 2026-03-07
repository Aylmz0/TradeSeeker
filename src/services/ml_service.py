import json
import logging
import os
from datetime import datetime
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from src.core.indicators import get_features_for_ml


logger = logging.getLogger(__name__)


class MLService:
    """
    Singleton Inference Service for XGBoost.
    Loads model artifacts into memory on boot and provides real-time
    predictions (BUY, HOLD, SELL probabilities) for AI hybridization.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model_path = "models/seeker_v1.xgb"
        self.scaler_path = "models/scaler.joblib"
        self.features_path = "models/feature_cols.joblib"

        self.model: xgb.XGBClassifier | None = None
        self.scaler = None
        self.feature_cols = None
        self.is_ready = False
        self.prediction_log_path = "data/ml_predictions.jsonl"
        self.last_model_mtime = 0  # Track file modification time for hot-reload

        self._load_artifacts()
        self._initialized = True

    def _load_artifacts(self) -> None:
        """Safely attempt to load ML artifacts. Fail gracefully if missing."""
        try:
            if not all(
                os.path.exists(p) for p in [self.model_path, self.scaler_path, self.features_path]
            ):
                logger.warning(
                    "[MLService] ML Artifacts missing. Service will run in DEGRADED (AI-Only) mode."
                )
                return

            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)

            # Load Scikit-Learn Scaler and Feature Names
            self.scaler = joblib.load(self.scaler_path)
            self.feature_cols = joblib.load(self.features_path)

            self.last_model_mtime = os.path.getmtime(self.model_path)
            self.is_ready = True
            logger.info(f"[MLService] XGBoost Inference Engine loaded and READY. (mtime: {self.last_model_mtime})")
        except Exception as e:
            logger.error(f"[MLService] Failed to load ML artifacts: {e}")
            self.is_ready = False

    def predict(self, df_raw: pd.DataFrame, coin: str) -> dict[str, Any] | None:
        """
        Takes raw OHLCV from Binance, extracts features, scales the latest row,
        and computes the directional multi-class probability.
        Includes automatic HOT-RELOAD if model file on disk is newer.
        """
        # --- HOT-RELOAD CHECK ---
        if os.path.exists(self.model_path):
            current_mtime = os.path.getmtime(self.model_path)
            if current_mtime > self.last_model_mtime:
                logger.info("[MLService] NEW MODEL DETECTED on disk. Triggering hot-reload...")
                self._load_artifacts()

        if not self.is_ready or df_raw.empty or len(df_raw) < 50:
            return None

        try:
            # 1. Pipeline: Raw Data -> Features
            df_features = get_features_for_ml(df_raw)
            if df_features.empty:
                return None

            # 2. Extract ONLY the very last (current) row for inference
            # Ensure we only pick the exact columns the model was trained on
            latest_features = df_features.iloc[[-1]][self.feature_cols]

            # 3. Apply StandardScaler (Using transform, NEVER fit)
            scaled_features = self.scaler.transform(latest_features)

            # 4. Predict Proba
            # [[Prob_SELL, Prob_HOLD, Prob_BUY]] -> [0.45, 0.35, 0.20]
            probs = self.model.predict_proba(scaled_features)[0]

            # Index mapping exactly as trained in train_model.py (0=SELL, 1=HOLD, 2=BUY)
            sell_prob = round(float(probs[0]) * 100, 2)
            hold_prob = round(float(probs[1]) * 100, 2)
            buy_prob = round(float(probs[2]) * 100, 2)

            # Identify dominant signal
            max_prob = max(sell_prob, hold_prob, buy_prob)
            if max_prob == sell_prob:
                dominant = "SELL"
            elif max_prob == hold_prob:
                dominant = "HOLD"
            else:
                dominant = "BUY"

            result = {
                "SELL": sell_prob,
                "HOLD": hold_prob,
                "BUY": buy_prob,
                "dominant_signal": dominant,
                "confidence": max_prob,
            }

            self._log_prediction(result, coin=coin)
            return result

        except Exception as e:
            logger.error(f"[MLService] Error during prediction: {e}")
            return None

    def _log_prediction(self, result: dict[str, Any], coin: str, interval: str = "15m") -> None:
        """Append prediction to JSONL log file (one JSON object per line)."""
        try:
            os.makedirs(os.path.dirname(self.prediction_log_path), exist_ok=True)
            log_entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "coin": coin,
                "interval": interval,
                "sell": result["SELL"],
                "hold": result["HOLD"],
                "buy": result["BUY"],
                "dominant": result["dominant_signal"],
                "confidence": result["confidence"],
            }
            with open(self.prediction_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"[MLService] Failed to log prediction: {e}")


if __name__ == "__main__":
    # Local Test Block
    print("--- Testing ML Inference Service ---")
    service = MLService()

    if service.is_ready:
        import sqlite3

        conn = sqlite3.connect("data/market_data.db")
        # Fetch 200 rows of raw data to feed the engine
        query = "SELECT * FROM market_data WHERE coin='XRP' AND interval='15m' ORDER BY timestamp DESC LIMIT 200"
        df_test = pd.read_sql_query(query, conn)
        df_test = df_test.sort_values("timestamp").reset_index(drop=True)
        df_test["timestamp"] = pd.to_datetime(df_test["timestamp"], unit="ms")
        conn.close()

        result = service.predict(df_test, coin="XRP")
        print("\n[OK] Prediction Result:")
        import json

        print(json.dumps(result, indent=2))
    else:
        print("\n[FAIL] MLService not ready. Did you run scripts/train_model.py?")
