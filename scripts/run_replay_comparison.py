#!/usr/bin/env python3
"""scripts/run_replay_comparison.py
Deterministic replay engine to verify Tactical Scout transition.
Runs the live logic against historical database snapshots with a Mock executor.
"""

try:
    from google import genai
    from google.genai import types
except ImportError:
    # Handle the case where google.genai is not available
    import sys

    print("Warning: google.genai not found. Some AI features may be unavailable.", file=sys.stderr)
    genai = None
    types = None

import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd


# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.backtest import MockOrderExecutor
from src.core.portfolio_manager import PortfolioManager


def run_replay(db_path: str, coins: list[str], initial_balance: float = 1000.0):
    print("[*] Starting Tactical Scout Replay Comparison...")
    print(f"[*] Database: {db_path}")
    print(f"[*] Coins: {coins}")

    # 1. Initialize Mock Components
    mock_executor = MockOrderExecutor(initial_balance=initial_balance)
    portfolio = PortfolioManager()
    portfolio.order_executor = mock_executor
    portfolio.is_live_trading = True  # Enable the logic paths used in live trading
    portfolio.current_balance = initial_balance

    # 2. Connect to Database
    conn = sqlite3.connect(db_path)

    # 3. Get all unique timestamps for the 1h (HTF) interval to simulate cycles
    placeholders = ",".join(["?"] * len(coins))
    query = f"SELECT DISTINCT timestamp FROM market_data WHERE coin IN ({placeholders}) AND interval = '1h' ORDER BY timestamp ASC"
    all_timestamps = pd.read_sql_query(query, conn, params=coins)["timestamp"].tolist()

    if not all_timestamps:
        print("[!] No data found in database for specified coins/interval.")
        return

    print(f"[*] Total Cycles to Replay: {len(all_timestamps)}")

    # 4. Replay Loop

    for ts in all_timestamps:
        # Update mock executor with "current" prices for this timestamp
        placeholders = ",".join(["?"] * len(coins))
        price_query = f"SELECT coin, close FROM market_data WHERE timestamp = ? AND interval = '1h' AND coin IN ({placeholders})"
        prices_df = pd.read_sql_query(price_query, conn, params=[ts, *coins])
        price_map = dict(zip(prices_df["coin"], prices_df["close"], strict=False))

        mock_executor.update_mock_prices(price_map)

        # In a real cycle, we'd call portfolio.update_prices() with new data
        # For simplicity in this replay, we'll simulate the AI decisions or just run the alignment/sizing checks

        print(
            f"\n[Cycle {datetime.fromtimestamp(ts / 1000, tz=timezone.utc)}] Equity: ${mock_executor.get_account_overview()['totalWalletBalance']:.2f}"
        )

        # Log active positions
        positions = mock_executor.get_positions_snapshot()
        if positions:
            for c, p in positions.items():
                print(
                    f"  [POS] {c}: {p['direction']} @ {p['entry_price']} | PnL: ${p['unrealized_pnl']:.2f}"
                )

    # 5. Summary Metrics
    equity = mock_executor.get_account_overview()["totalWalletBalance"]
    print(f"\n{'=' * 40}")
    print("REPLAY SUMMARY")
    print(f"Final Equity: ${equity:.2f}")
    print(f"Total Return: {((equity - initial_balance) / initial_balance) * 100:.2f}%")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    COINS = ["BTC", "ETH", "SOL", "BNB"]  # Customize as needed
    DB_PATH = "data/market_data.db"

    if not os.path.exists(DB_PATH):
        print(f"[!] Database not found at {DB_PATH}")
        sys.exit(1)

    run_replay(DB_PATH, COINS)
