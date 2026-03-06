import sqlite3
import time

import pandas as pd

from src.core.indicators import get_features_for_ml

print("--- Starting feature test ---")
try:
    print("1. Connecting to DB...")
    conn = sqlite3.connect('data/market_data.db')
    query = "SELECT * FROM market_data WHERE coin='XRP' AND interval='15m' ORDER BY timestamp DESC LIMIT 500"
    df_raw = pd.read_sql_query(query, conn)
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
    conn.close()
    
    print(f"2. Fetched {len(df_raw)} candles. Extracting features...")
    st = time.time()
    
    # We will test extracting features here
    df_f = get_features_for_ml(df_raw)
    
    print(f"3. Done in {time.time() - st:.3f}s")
    print("OUTPUT ROWS:", len(df_f))
    if len(df_f) > 0:
        print("COLUMNS: ", len(df_f.columns))
        print("LAST ROW:\n", df_f.iloc[-1])
        
except Exception:
    import traceback
    traceback.print_exc()
