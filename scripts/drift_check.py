"""
Phase 4.3: Model Drift Detection
Compares live prediction accuracy against training baseline.
If live accuracy drops >10% below training accuracy, flags re-training.

Usage: PYTHONPATH=. .venv/bin/python scripts/drift_check.py
"""

import json
import os
import sqlite3

import pandas as pd


PREDICTION_LOG = "data/ml_predictions.jsonl"
DATABASE = "data/market_data.db"
TRAINING_ACCURACY = 0.431  # Baseline from Phase 2.2 training


def load_predictions() -> pd.DataFrame:
    """Load JSONL prediction log into DataFrame."""
    if not os.path.exists(PREDICTION_LOG):
        print(f"[FAIL] No prediction log found at {PREDICTION_LOG}")
        print("       Run the bot or test_hybrid.py first to generate predictions.")
        return pd.DataFrame()

    records = []
    with open(PREDICTION_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def load_closed_decisions() -> pd.DataFrame:
    """Load CLOSED decisions from SQLite."""
    if not os.path.exists(DATABASE):
        return pd.DataFrame()

    conn = sqlite3.connect(DATABASE)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM decisions WHERE status = 'CLOSED' ORDER BY timestamp ASC",
            conn,
        )
        return df
    finally:
        conn.close()


def analyze_drift():
    print("=" * 60)
    print("  TradeSeeker Model Drift Detection Report")
    print("=" * 60)

    # 1. Prediction Distribution Analysis
    print("\n[1/3] Prediction Distribution Analysis")
    print("-" * 40)
    df_pred = load_predictions()

    if df_pred.empty:
        print("[WARN] No predictions logged yet. Skipping distribution analysis.")
    else:
        total = len(df_pred)
        dist = df_pred["dominant"].value_counts()
        print(f"Total predictions logged: {total}")
        for signal, count in dist.items():
            pct = (count / total) * 100
            print(f"  {signal}: {count} ({pct:.1f}%)")

        avg_confidence = df_pred["confidence"].mean()
        print(f"\nAverage confidence: {avg_confidence:.2f}%")

        if avg_confidence < 40:
            print("[WARNING] Average confidence below 40% -- model may be uncertain.")
        else:
            print("[OK] Confidence levels are healthy.")

    # 2. Decision Feedback Analysis
    print("\n[2/3] Decision Feedback Analysis")
    print("-" * 40)
    df_decisions = load_closed_decisions()

    if df_decisions.empty:
        print("[INFO] No closed decisions yet. This section will populate after live trades.")
        live_accuracy = None
    else:
        total_trades = len(df_decisions)
        wins = len(df_decisions[df_decisions["pnl_result"] > 0])
        losses = len(df_decisions[df_decisions["pnl_result"] <= 0])
        total_pnl = df_decisions["pnl_result"].sum()
        live_accuracy = wins / total_trades if total_trades > 0 else 0

        print(f"Total closed trades: {total_trades}")
        print(f"  Wins: {wins} | Losses: {losses}")
        print(f"  Win Rate: {live_accuracy * 100:.1f}%")
        print(f"  Total PnL: ${total_pnl:.2f}")

    # 3. Drift Detection
    print("\n[3/3] Drift Detection")
    print("-" * 40)
    print(f"Training Baseline Accuracy: {TRAINING_ACCURACY * 100:.1f}%")

    if live_accuracy is not None:
        drift = TRAINING_ACCURACY - live_accuracy
        print(f"Live Accuracy: {live_accuracy * 100:.1f}%")
        print(f"Drift: {drift * 100:+.1f}%")

        if drift > 0.10:
            print("\n[ALERT] SIGNIFICANT DRIFT DETECTED!")
            print("  Live accuracy dropped >10% below training baseline.")
            print("  Recommendation: Re-train the model with fresh data.")
            print("  Command: PYTHONPATH=. .venv/bin/python scripts/train_model.py")
        elif drift > 0.05:
            print("\n[WARNING] Moderate drift detected.")
            print("  Monitor closely. Consider re-training if trend continues.")
        else:
            print("\n[OK] No significant drift. Model performance is stable.")
    else:
        print("[INFO] No live trades to compare. Drift check will activate after closed trades.")

    print("\n" + "=" * 60)
    print("  Drift Check Complete.")
    print("=" * 60)


if __name__ == "__main__":
    analyze_drift()
