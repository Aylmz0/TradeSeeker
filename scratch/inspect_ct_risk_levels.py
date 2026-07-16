import glob
import json
import os
import re
from datetime import datetime, timedelta

import polars as pl


PROJECT_ROOT = "/home/yilmaz/projects/TradeSeeker"
df_trades = pl.read_csv(os.path.join(PROJECT_ROOT, "data/reconstructed_trade_logs.csv"))

# We want to load the cycle records and find the computed risk level for each trade
ct_trades = df_trades.filter(pl.col("trend_alignment") == "counter_trend").clone()

# Load all cycles
all_cycles = []
active_cycle_file = os.path.join(PROJECT_ROOT, "data/cycle_history.json")
if os.path.exists(active_cycle_file):
    with open(active_cycle_file) as f:
        try:
            all_cycles.extend(json.load(f))
        except:
            pass

backup_dir = os.path.join(PROJECT_ROOT, "data/backups")
if os.path.exists(backup_dir):
    subdirs = sorted(glob.glob(os.path.join(backup_dir, "*_cycle_*")))
    for subdir in subdirs:
        cycle_file = os.path.join(subdir, "cycle_history.json")
        if os.path.exists(cycle_file):
            with open(cycle_file) as f:
                try:
                    all_cycles.extend(json.load(f))
                except:
                    pass

# Index cycles by timestamp or close time
cycles_map = {}
for c in all_cycles:
    ts = c.get("timestamp")
    if ts:
        cycles_map[ts[:16]] = c  # Match by minute

results = []
for row in ct_trades.iter_rows(named=True):
    entry_time_str = row["entry_time"]
    coin = row["symbol"]
    direction = row["direction"]
    pnl = row["pnl"]

    # Try to find a matching cycle
    matched_c = None
    # Let's match by comparing timestamp string minutes
    entry_minute = entry_time_str[:16]
    matched_c = cycles_map.get(entry_minute)

    # If not found, try to look at cycles +/- 5 minutes
    if not matched_c:
        dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
        for i in range(-5, 6):
            check_min = (dt + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M")
            if check_min in cycles_map:
                matched_c = cycles_map[check_min]
                break

    risk_level = "UNKNOWN"
    strategy = "UNKNOWN"
    confidence = 0.0

    if matched_c:
        decisions = matched_c.get("decisions", {})
        coin_dec = decisions.get(coin, {}) if isinstance(decisions, dict) else {}
        if coin_dec:
            # Check strategy/classification
            strategy = coin_dec.get("strategy", "UNKNOWN")
            confidence = coin_dec.get("confidence", 0.0)

        # Let's parse chain of thoughts to find the computed risk level
        cot = str(matched_c.get("chain_of_thoughts", ""))
        # Look for "XRP: ... risk" or check metadata if counter_trade_risk was saved
        match = re.search(
            rf"{coin}.*?risk.*?(CT_LOW_RISK|CT_MEDIUM_RISK|CT_HIGH_RISK|CT_VERY_HIGH_RISK|LOW_RISK|MEDIUM_RISK|HIGH_RISK)",
            cot,
            re.IGNORECASE,
        )
        if match:
            risk_level = match.group(1).upper()
        else:
            # Let's try general search of coin name + risk level
            m2 = re.search(
                r"(CT_LOW_RISK|CT_MEDIUM_RISK|CT_HIGH_RISK|CT_VERY_HIGH_RISK|LOW_RISK|MEDIUM_RISK|HIGH_RISK)",
                cot,
            )
            if m2:
                risk_level = m2.group(1).upper()

    results.append(
        {
            "Coin": coin,
            "Direction": direction,
            "EntryTime": entry_time_str[5:16],
            "PnL": pnl,
            "RiskLevel": risk_level,
            "Strategy": strategy,
            "Confidence": confidence,
        }
    )

df_res = pl.DataFrame(results)
print("\nCounter-Trend Trades Risk Level and Confidence Breakdown:")
print(df_res)

print("\nSummary of Risk Levels:")
print(df_res.get_column("RiskLevel").value_counts())
