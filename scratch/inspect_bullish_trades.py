import os

import pandas as pd


PROJECT_ROOT = "/home/yilmaz/projects/TradeSeeker"
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/reconstructed_trade_logs.csv"))

bullish_longs = df[(df["regime"] == "BULLISH") & (df["direction"] == "LONG")]
print("Bullish Regime LONG Trades:")
print(bullish_longs[["symbol", "pnl", "close_reason", "volume_ratio", "rsi_15m"]])
