import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = "data/market_data.db"

def quant_audit():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Fetch all 15m data
    df = pd.read_sql_query("SELECT coin, close FROM market_data WHERE interval = '15m' ORDER BY timestamp ASC", conn)
    conn.close()
    
    if df.empty:
        print("No 15m data found.")
        return

    results = []
    lookahead = 5
    
    for coin in df['coin'].unique():
        coin_df = df[df['coin'] == coin].copy()
        coin_df['future_return'] = (coin_df['close'].shift(-lookahead) - coin_df['close']) / coin_df['close']
        coin_df.dropna(subset=['future_return'], inplace=True)
        
        # Calculate stats
        mean_ret = coin_df['future_return'].mean()
        std_ret = coin_df['future_return'].std()
        max_ret = coin_df['future_return'].max()
        min_ret = coin_df['future_return'].min()
        
        # Count how many would be BUY/SELL at 0.5% threshold
        buys = len(coin_df[coin_df['future_return'] >= 0.005])
        sells = len(coin_df[coin_df['future_return'] <= -0.005])
        total = len(coin_df)
        
        results.append({
            "coin": coin,
            "samples": total,
            "mean": f"{mean_ret:.4%}",
            "std": f"{std_ret:.4%}",
            "max": f"{max_ret:.4%}",
            "min": f"{min_ret:.4%}",
            "potential_buys_0.5%": buys,
            "potential_sells_0.5%": sells,
            "hold_percentage": f"{(total - buys - sells)/total:.1%}"
        })

    print(pd.DataFrame(results).to_markdown())

if __name__ == "__main__":
    print("--- 📉 Quantitative Market Audit: 816 Cycles ---")
    quant_audit()
