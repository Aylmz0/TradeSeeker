"""
Phase 3.3: Hybrid Dry-Run Test
Tests the complete ML inference pipeline end-to-end:
  SQLite Raw Data -> Feature Engineering -> StandardScaler -> XGBoost -> Probability JSON

No external API dependencies (OpenAI, Binance) required.
"""

import json
import sqlite3

import pandas as pd

from src.services.ml_service import MLService


def run_dry_run(coin: str = "XRP", interval: str = "15m"):
    print("=" * 60)
    print("  TradeSeeker Hybrid Dry-Run Test (Phase 3.3)")
    print("=" * 60)

    # Step 1: Load raw OHLCV from local SQLite
    print("\n[1/4] Loading raw market data from SQLite...")
    try:
        conn = sqlite3.connect("data/market_data.db")
        query = (
            "SELECT * FROM market_data "
            f"WHERE coin='{coin}' AND interval='{interval}' "
            "ORDER BY timestamp DESC LIMIT 200"
        )
        df_raw = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"[FAIL] Cannot read SQLite: {e}")
        print("       Run data_engine.py first to populate the database.")
        return

    if df_raw.empty:
        print(f"[FAIL] No data found for {coin} ({interval}). Run data_engine.py first.")
        return

    # Chronological order
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms")
    print(
        f"[OK] Loaded {len(df_raw)} candles. Range: {df_raw['timestamp'].iloc[0]} -> {df_raw['timestamp'].iloc[-1]}"
    )

    # Step 2: Boot ML Inference Service
    print("\n[2/4] Booting MLService (Singleton)...")
    service = MLService()

    if not service.is_ready:
        print("[FAIL] MLService not ready. Train the model first:")
        print("       PYTHONPATH=. .venv/bin/python scripts/train_model.py")
        return

    print(f"[OK] Model loaded. Features: {len(service.feature_cols)} columns")
    print(f"     Scaler type: {type(service.scaler).__name__}")

    # Step 3: Run Prediction
    print("\n[3/4] Running XGBoost inference on latest candle...")
    result = service.predict(df_raw)

    if result is None:
        print("[FAIL] Prediction returned None. Check feature extraction.")
        return

    print("[OK] Prediction successful!")

    # Step 4: Display Results
    print("\n[4/4] ML Consensus Output:")
    print("-" * 40)
    print(json.dumps(result, indent=2))
    print("-" * 40)

    # Interpretation
    dominant = result["dominant_signal"]
    confidence = result["confidence"]

    if confidence >= 45:
        strength = "STRONG"
    elif confidence >= 38:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    print(f"\n[RESULT] Signal: {dominant} | Confidence: {confidence}% | Strength: {strength}")

    if strength == "STRONG":
        print(
            "[INFO] This signal would be injected into the AI prompt as a high-weight technical consensus."
        )
    elif strength == "MODERATE":
        print("[INFO] This signal would be noted by the AI but not treated as decisive alone.")
    else:
        print("[INFO] Low confidence -- AI would likely default to its own chart analysis.")

    print("\n" + "=" * 60)
    print("  Dry-Run Complete. Pipeline is operational.")
    print("=" * 60)


if __name__ == "__main__":
    run_dry_run()
