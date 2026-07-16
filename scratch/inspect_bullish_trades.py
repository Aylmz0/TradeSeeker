import os

import polars as pl


PROJECT_ROOT = "/home/yilmaz/projects/TradeSeeker"
df = pl.read_csv(os.path.join(PROJECT_ROOT, "data/reconstructed_trade_logs.csv"))

bullish_longs = df.filter((pl.col("regime") == "BULLISH") & (pl.col("direction") == "LONG"))
print("Bullish Regime LONG Trades:")
print(bullish_longs.select(["symbol", "pnl", "close_reason", "volume_ratio", "rsi_15m"]))
