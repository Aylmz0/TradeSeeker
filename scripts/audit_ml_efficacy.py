#!/usr/bin/env python3
import json
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_LOG_FILE = os.path.join(PROJECT_ROOT, "data/ml_predictions.jsonl")
DB_FILE = os.path.join(PROJECT_ROOT, "data/market_data.db")


def audit_ml_deep():
    if not os.path.exists(ML_LOG_FILE) or not os.path.exists(DB_FILE):
        print("[ERR] Required data files missing.")
        return

    # 1. Load ML Predictions
    preds = []
    with open(ML_LOG_FILE, "r") as f:
        for line in f:
            try:
                preds.append(json.loads(line))
            except:
                continue

    df_preds = pd.DataFrame(preds)
    # Get unix timestamp in MILLISECONDS
    df_preds["ts_dt"] = pd.to_datetime(df_preds["ts"])
    df_preds["ts_ms"] = df_preds["ts_dt"].apply(lambda x: int(x.timestamp() * 1000))

    # 2. Connect to DB
    conn = sqlite3.connect(DB_FILE)

    results = []
    print(f"[INIT] Analyzing {len(df_preds)} predictions across market history...")

    for coin in df_preds["coin"].unique():
        coin_preds = df_preds[df_preds["coin"] == coin]

        # Load all 15m price data for this coin
        prices = pd.read_sql_query(
            f"SELECT timestamp, close FROM market_data WHERE coin = '{coin}' AND interval = '15m' ORDER BY timestamp ASC",
            conn,
        )
        if prices.empty:
            continue

        # Sort prices for merge_asof
        prices["timestamp"] = prices["timestamp"].astype(int)

        for _, pred in coin_preds.iterrows():
            t_start = pred["ts_ms"]
            t_end = t_start + (60 * 60 * 1000)  # +60 minutes

            # Find price at start (within 15 min window)
            start_price_mask = (prices["timestamp"] >= t_start - 900000) & (
                prices["timestamp"] <= t_start + 900000
            )
            end_price_mask = (prices["timestamp"] >= t_end - 900000) & (
                prices["timestamp"] <= t_end + 900000
            )

            start_match = prices[start_price_mask]
            end_match = prices[end_price_mask]

            if start_match.empty or end_match.empty:
                continue

            p_start = start_match.iloc[0]["close"]
            p_end = end_match.iloc[0]["close"]

            change = (p_end - p_start) / p_start
            dominant = pred["dominant"]
            conf = pred["confidence"] / 100 if pred["confidence"] > 1 else pred["confidence"]

            success = False
            if dominant == "BUY" and change > 0.0005:
                success = True
            elif dominant == "SELL" and change < -0.0005:
                success = True
            elif dominant == "HOLD" and abs(change) <= 0.0005:
                success = True

            results.append({"side": dominant, "confidence": conf, "success": success})

    conn.close()

    if not results:
        print("[ERR] Could not match ML predictions with price history. Check DB/Log timestamps.")
        return

    df_res = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("      ML EFFICACY AUDIT (60-MIN FUTURE DRIFT)")
    print("=" * 60)
    print(f"Matched Samples: {len(df_res)}")
    print(f"Overall Accuracy: {df_res['success'].mean()*100:.2f}%")

    for side in ["BUY", "SELL"]:
        side_data = df_res[df_res["side"] == side]
        if side_data.empty:
            continue

        print(f"\n[{side}] Count: {len(side_data)} | Acc: {side_data['success'].mean()*100:.2f}%")
        for low, high in [(0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
            bucket = side_data[(side_data["confidence"] >= low) & (side_data["confidence"] < high)]
            if not bucket.empty:
                print(
                    f"  - Conf {int(low*100)}-{int(high*100)}%: {bucket['success'].mean()*100:.2f}% ({len(bucket)} samples)"
                )


if __name__ == "__main__":
    audit_ml_deep()
