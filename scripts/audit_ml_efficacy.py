#!/usr/bin/env python3
import json
import os
import sqlite3
import polars as pl
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_LOG_FILE = os.path.join(PROJECT_ROOT, "data/ml_predictions.jsonl")
DB_FILE = os.path.join(PROJECT_ROOT, "data/market_data.db")


def _query_to_df(conn, query, params=None):
    """SQLite -> polars bridge."""
    cur = conn.execute(query, params) if params else conn.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    if not rows:
        return pl.DataFrame(schema=cols)
    return pl.DataFrame(rows, schema=cols)


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
            except Exception:
                continue

    df_preds = pl.DataFrame(preds)
    # Get unix timestamp in MILLISECONDS
    df_preds = df_preds.with_columns(
        pl.col("ts").str.to_datetime().alias("ts_dt"),
    ).with_columns((pl.col("ts_dt").cast(pl.Int64) // 1_000_000).alias("ts_ms"))

    # 2. Connect to DB
    conn = sqlite3.connect(DB_FILE)

    results = []
    print(f"[INIT] Analyzing {len(df_preds)} predictions across market history...")

    for coin in df_preds["coin"].unique().to_list():
        coin_preds = df_preds.filter(pl.col("coin") == coin)

        # Load all 15m price data for this coin
        prices = _query_to_df(
            conn,
            f"SELECT timestamp, close FROM market_data WHERE coin = '{coin}' AND interval = '15m' ORDER BY timestamp ASC",
        )
        if prices.is_empty():
            continue

        # Sort prices
        prices = prices.sort("timestamp").cast({"timestamp": pl.Int64})

        for pred in coin_preds.iter_rows(named=True):
            t_start = pred["ts_ms"]
            t_end = t_start + (60 * 60 * 1000)  # +60 minutes

            # Find price at start (within 15 min window)
            start_match = prices.filter(
                (pl.col("timestamp") >= t_start - 900000)
                & (pl.col("timestamp") <= t_start + 900000)
            )
            end_match = prices.filter(
                (pl.col("timestamp") >= t_end - 900000) & (pl.col("timestamp") <= t_end + 900000)
            )

            if start_match.is_empty() or end_match.is_empty():
                continue

            p_start = start_match["close"][0]
            p_end = end_match["close"][0]

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

    df_res = pl.DataFrame(results)

    print("\n" + "=" * 60)
    print("      ML EFFICACY AUDIT (60-MIN FUTURE DRIFT)")
    print("=" * 60)
    print(f"Matched Samples: {len(df_res)}")
    print(f"Overall Accuracy: {df_res['success'].mean()*100:.2f}%")

    for side in ["BUY", "SELL"]:
        side_data = df_res.filter(pl.col("side") == side)
        if side_data.is_empty():
            continue

        print(f"\n[{side}] Count: {len(side_data)} | Acc: {side_data['success'].mean()*100:.2f}%")
        for low, high in [(0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
            bucket = side_data.filter((pl.col("confidence") >= low) & (pl.col("confidence") < high))
            if not bucket.is_empty():
                print(
                    f"  - Conf {int(low*100)}-{int(high*100)}%: {bucket['success'].mean()*100:.2f}% ({len(bucket)} samples)"
                )


if __name__ == "__main__":
    audit_ml_deep()
