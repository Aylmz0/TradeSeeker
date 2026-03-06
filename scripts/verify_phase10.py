import os
import sys
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import AlphaArenaDeepSeek
from src.core.data_engine import DataEngine
from config.config import Config

def verify_logging():
    print("--- Phase 10 Verification: Database Logging ---")
    
    # Enable JSON prompt to ensure the new logic path is hit
    Config.USE_JSON_PROMPT = True
    
    # Initialize bot (this should create the DB and tables)
    bot = AlphaArenaDeepSeek()
    
    db_path = "data/market_data.db"
    if os.path.exists(db_path):
        print(f"[OK] Database file created at {db_path}")
    else:
        print(f"[FAIL] Database file NOT found at {db_path}")
        return

    # Run a "mock" check by calling generate_alpha_arena_prompt_json
    # This will trigger the fetching of indicators and storage in bot.ai_service.latest_indicators
    print("\n[Test] Generating prompt and fetching indicators...")
    try:
        prompt = bot.ai_service.generate_alpha_arena_prompt_json()
        print("[OK] Prompt generated successfully.")
    except Exception as e:
        print(f"[FAIL] Prompt generation failed: {e}")
        return

    # Simulate the logging block from main.py
    print("\n[Test] Simulating logging block...")
    try:
        latest_indicators = getattr(bot.ai_service, "latest_indicators", {})
        for coin in bot.market_data.available_coins:
            for interval in ["3m", "15m", "1h"]:
                df_raw = bot.market_data.get_cached_raw_dataframe(coin, interval)
                if df_raw is not None and not df_raw.empty:
                    bot.data_engine.log_market_data(df_raw, coin, interval)
            
            if coin in latest_indicators and "15m" in latest_indicators[coin]:
                bot.data_engine.log_cycle_features(coin, "15m", latest_indicators[coin]["15m"])
        print("[OK] Logging simulation completed.")
    except Exception as e:
        print(f"[FAIL] Logging simulation failed: {e}")
        return

    # Verify tables
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(db_path)
    
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"\n[Verify] Tables in DB: {tables['name'].tolist()}")
    
    if 'market_data' in tables['name'].tolist():
        md_count = pd.read_sql_query("SELECT COUNT(*) as count FROM market_data", conn)['count'][0]
        print(f"[Verify] rows in market_data: {md_count}")
        if md_count > 0:
            print("[OK] market_data table is NOT empty.")
        else:
            print("[FAIL] market_data table IS empty.")
            
    if 'features' in tables['name'].tolist():
        f_count = pd.read_sql_query("SELECT COUNT(*) as count FROM features", conn)['count'][0]
        print(f"[Verify] rows in features: {f_count}")
        if f_count > 0:
            print("[OK] features table is NOT empty.")
        else:
            print("[FAIL] features table IS empty.")

    conn.close()
    print("\n--- Verification Finished ---")

if __name__ == "__main__":
    verify_logging()
