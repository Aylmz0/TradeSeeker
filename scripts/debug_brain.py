import sys
import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime, timedelta

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.indicators import get_features_for_ml
from config.config import Config


def create_mock_data(regime="neutral", length=150):
    """Creates mock OHLCV dataframe for testing."""
    now = datetime.now()
    timestamps = [now - timedelta(minutes=15 * i) for i in range(length)]
    timestamps.reverse()

    # Base price
    prices = [100.0]
    volumes = [1000.0]

    if regime == "bullish":
        # Strong upward trend
        for i in range(1, length):
            prices.append(prices[-1] * (1 + 0.005))  # +0.5% per bar
            volumes.append(volumes[-1] * (1 + 0.01))  # Increasing volume
    elif regime == "bearish":
        # Strong downward trend
        for i in range(1, length):
            prices.append(prices[-1] * (1 - 0.005))  # -0.5% per bar
            volumes.append(volumes[-1] * (1 + 0.01))  # Increasing volume on drop
    else:
        # Neutral / Squeeze
        for i in range(1, length):
            prices.append(prices[-1] * (1 + np.random.uniform(-0.0005, 0.0005)))
            volumes.append(volumes[-1] * (1 + np.random.uniform(-0.01, 0.01)))

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": volumes,
        }
    )
    return df


def test_model():
    model_path = os.path.join(PROJECT_ROOT, "models/seeker_v1.xgb")
    scaler_path = os.path.join(PROJECT_ROOT, "models/scaler.joblib")
    features_path = os.path.join(PROJECT_ROOT, "models/feature_cols.joblib")

    if not os.path.exists(model_path):
        print(f"[ERR] Model not found at {model_path}")
        return

    print("\n" + "=" * 50)
    print("🧠 ML BRAIN VALIDATION TEST")
    print("=" * 50)

    # Load artifacts
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)

    regimes = ["neutral", "bullish", "bearish"]

    for regime in regimes:
        print(f"\n---> Testing Regime: {regime.upper()}")
        df_raw = create_mock_data(regime=regime)
        df_features = get_features_for_ml(df_raw)

        if df_features.empty:
            print(f"     [FAIL] Feature extraction returned empty for {regime}")
            continue

        latest_features = df_features.iloc[[-1]][feature_cols]
        scaled_features = scaler.transform(latest_features)

        probs = model.predict_proba(scaled_features)[0]
        sell_p, hold_p, buy_p = probs[0] * 100, probs[1] * 100, probs[2] * 100

        print(f"     [RESULT] SELL: {sell_p:.2f}% | HOLD: {hold_p:.2f}% | BUY: {buy_p:.2f}%")

        # Verdict
        if regime == "neutral" and abs(sell_p - buy_p) < 5:
            print("     [VERDICT] PASS: Model correctly identified uncertainty.")
        elif regime == "bullish" and buy_p > sell_p + 5:
            print(
                f"     [VERDICT] PASS: Model detected Bullish conviction (+{buy_p-sell_p:.2f}% spread)"
            )
        elif regime == "bearish" and sell_p > buy_p + 5:
            print(
                f"     [VERDICT] PASS: Model detected Bearish conviction (+{sell_p-buy_p:.2f}% spread)"
            )
        else:
            print("     [VERDICT] FAIL: Model conviction too low or incorrect direction.")


if __name__ == "__main__":
    test_model()
