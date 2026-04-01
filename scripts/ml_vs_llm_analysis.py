#!/usr/bin/env python3
import json
import os
import sys
import pandas as pd
from datetime import datetime, timezone
import re
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def parse_iso(ts_str):
    if not ts_str:
        return None
    try:
        return pd.to_datetime(ts_str, format="ISO8601", utc=True)
    except:
        return None


def analyze_ml_vs_llm():
    all_trades = []
    all_cycles = []

    # Load all trades and cycles from data/ and both backup directories
    # New: data/backups, Old: history_backups (backward compat)
    dirs_to_check = []
    backups_new = os.path.join(PROJECT_ROOT, "data/backups")
    backups_old = os.path.join(PROJECT_ROOT, "history_backups")

    if os.path.exists(backups_new):
        dirs_to_check.extend([os.path.join(backups_new, d) for d in os.listdir(backups_new)])
    if os.path.exists(backups_old):
        dirs_to_check.extend([os.path.join(backups_old, d) for d in os.listdir(backups_old)])
    dirs_to_check.append(os.path.join(PROJECT_ROOT, "data"))

    for d in dirs_to_check:
        if not os.path.isdir(d):
            continue
        tc_path = os.path.join(d, "full_trade_history.json")
        if not os.path.exists(tc_path):
            tc_path = os.path.join(d, "trade_history.json")
        cy_path = os.path.join(d, "cycle_history.json")

        if os.path.exists(tc_path):
            with open(tc_path, "r") as f:
                try:
                    ts = json.load(f)
                    all_trades.extend(ts)
                except:
                    pass
        if os.path.exists(cy_path):
            with open(cy_path, "r") as f:
                try:
                    cys = json.load(f)
                    all_cycles.extend(cys)
                except:
                    pass

    # remove duplicate trades based on entry_time and coin
    unique_trades = {}
    for t in all_trades:
        key = f"{t.get('coin', t.get('symbol'))}_{t.get('entry_time')}"
        if key not in unique_trades:
            unique_trades[key] = t

    # remove duplicate cycles based on timestamp
    unique_cycles = {}
    for c in all_cycles:
        if "timestamp" in c:
            unique_cycles[c["timestamp"]] = c

    trades = list(unique_trades.values())
    cycles = list(unique_cycles.values())

    print(f"Loaded {len(trades)} unique trades and {len(cycles)} unique cycles.")

    for c in cycles:
        c["ts"] = parse_iso(c.get("timestamp"))
    cycles = sorted([c for c in cycles if c["ts"]], key=lambda x: x["ts"])

    results = []

    patterns = [
        r"ML Consensus:\s*([A-Za-z]+)\s*(\d+\.?\d*)\%",
        r"ML\s*Consensus:\s*.*?([A-Za-z]+)\s*(\d+\.?\d*)\%",
        r"ML:.*?(SELL|BUY|HOLD)\s*(\d+\.?\d*)\%",
        r"ML\s*(leans|shows|favors|strongly favors)?\s*(SELL|BUY|HOLD)\s*\(?(\d+\.?\d*)\%\)?",
    ]

    for t in trades:
        entry_time = parse_iso(t.get("entry_time"))
        if not entry_time:
            continue

        coin = t.get("coin", t.get("symbol"))
        direction = t.get("direction", "LONG").upper()
        pnl = t.get("pnl", t.get("pnl_usd", 0))
        close_reason = t.get("close_reason", "Unknown")

        matched_cycle = None
        for i in range(len(cycles) - 1, -1, -1):
            c = cycles[i]
            if c["ts"] <= entry_time:
                # check if decs has signal
                decs = c.get("decisions", {})
                coin_dec = decs.get(coin, {})
                if coin_dec.get("signal") in [
                    "buy_to_enter",
                    "short_to_enter",
                    "open_long",
                    "open_short",
                ]:
                    matched_cycle = c
                    break
                if (entry_time - c["ts"]).total_seconds() < 90:
                    matched_cycle = c
                    break

        if matched_cycle:
            cot = matched_cycle.get("chain_of_thoughts", "")
            coin_cot = ""
            for line in cot.split("\n"):
                if line.startswith(f"{coin}:") or line.startswith(f"{coin} "):
                    coin_cot = line
                    break

            if not coin_cot:
                continue

            ml_bias = "Unknown"
            ml_conf = 0.0

            for pat in patterns:
                m = re.search(pat, coin_cot, re.IGNORECASE)
                if m:
                    groups = [g for g in m.groups() if g and g.upper() in ["BUY", "SELL", "HOLD"]]
                    nums = [g for g in m.groups() if g and g.replace(".", "", 1).isdigit()]
                    if groups:
                        ml_bias = groups[0].upper()
                    if nums:
                        ml_conf = float(nums[0])
                    break

            if ml_bias == "Unknown":
                m2 = re.search(r"ML\s*(BUY|SELL|HOLD)\s*(\d+\.?\d*)", coin_cot, re.IGNORECASE)
                if m2:
                    ml_bias = m2.group(1).upper()
                    ml_conf = float(m2.group(2))

            llm_direction = "BUY" if direction == "LONG" else "SELL"

            conflict = False
            if ml_bias in ["BUY", "SELL"] and ml_bias != llm_direction:
                conflict = True

            win = pnl > 0

            who_was_right = "N/A"
            if conflict:
                if win:
                    who_was_right = "LLM (Ignored ML and won)"
                else:
                    who_was_right = "ML (LLM ignored ML and lost)"
            else:
                if ml_bias in ["BUY", "SELL"]:
                    if win:
                        who_was_right = "Both Right"
                    else:
                        who_was_right = "Both Wrong"

            results.append(
                {
                    "Coin": coin,
                    "EntryTime": entry_time.strftime("%m-%d %H:%M"),
                    "LLM_Dir": llm_direction,
                    "ML_Bias": ml_bias,
                    "ML_Conf": f"{ml_conf}%",
                    "Conflict": conflict,
                    "WhoRight": who_was_right,
                    "PnL": f"${pnl:.2f}",
                    "LLM_Logic": coin_cot,
                }
            )

    df = pd.DataFrame(results)
    if df.empty:
        print("No matches found in 200 cycles, script failed.")
        return

    print("\n" + "=" * 80)
    print(" 🤖 ML vs LLM DIVERGENCE ANALYSIS (DEEP DATA)")
    print("=" * 80)

    df_display = df[
        ["Coin", "EntryTime", "LLM_Dir", "ML_Bias", "ML_Conf", "Conflict", "WhoRight", "PnL"]
    ]
    print(df_display.to_string(index=False))

    print("\n" + "=" * 80)
    print(" 📊 SUMMARY STATISTICS ")
    print("=" * 80)

    total_trades = len(df)
    conflicts = df[df["Conflict"] == True]
    agreements = df[(df["Conflict"] == False) & (df["ML_Bias"].isin(["BUY", "SELL"]))]
    neutral_ml = df[df["ML_Bias"].isin(["HOLD", "Unknown"])]

    print(f"Total Evaluated Trades: {total_trades}")
    print(f"Trades where ML Disagreed with LLM (Conflict): {len(conflicts)}")
    print(f"Trades where ML Agreed with LLM: {len(agreements)}")
    print(f"Trades where ML was Neutral/Hold: {len(neutral_ml)}")

    if len(conflicts) > 0:
        ml_wins = len(conflicts[conflicts["WhoRight"].str.startswith("ML")])
        llm_wins = len(conflicts[conflicts["WhoRight"].str.startswith("LLM")])

        print(f"\nIn {len(conflicts)} conflicting trades:")
        print(
            f" -> ML was right (Trade LOST, LLM should have listened): {ml_wins} times ({ml_wins / len(conflicts) * 100:.1f}%)"
        )
        print(
            f" -> LLM was right (Trade WON, ML was wrong): {llm_wins} times ({llm_wins / len(conflicts) * 100:.1f}%)"
        )

        loss_pnl = (
            conflicts[conflicts["WhoRight"].str.startswith("ML")]["PnL"]
            .str.replace("$", "")
            .astype(float)
            .sum()
        )
        win_pnl = (
            conflicts[conflicts["WhoRight"].str.startswith("LLM")]["PnL"]
            .str.replace("$", "")
            .astype(float)
            .sum()
        )

        print(f"\nPnL Impact when ML was ignored:")
        print(f" -> Losses avoided if ML was listened to: ${abs(loss_pnl):.2f}")
        print(f" -> Missed profits if ML was listened to: ${win_pnl:.2f}")

    print("\n" + "=" * 80)
    print(" 🔎 DEEP DIVE ON CONFLICTS (Why did LLM ignore ML?)")
    print("=" * 80)
    for _, row in conflicts.iterrows():
        print(
            f"\n[{row['EntryTime']}] {row['Coin']} | LLM went {row['LLM_Dir']} | ML said {row['ML_Bias']} ({row['ML_Conf']}) | Result: {row['PnL']}"
        )
        print(f"LLM Reasoning: {row['LLM_Logic']}")

    # Analyze holds vs ML
    ml_buy_llm_hold = 0
    ml_sell_llm_hold = 0
    print("\n" + "=" * 80)
    print(" 🛑 WHAT HAPPENED WHEN ML SAID BUY/SELL BUT LLM HELD?")
    print("=" * 80)

    hold_examples = []

    for c in cycles:
        decs = c.get("decisions", {})
        for coin, dec in decs.items():
            if dec.get("signal") == "hold":
                cot = c.get("chain_of_thoughts", "")
                coin_cot = ""
                for line in cot.split("\n"):
                    if line.startswith(f"{coin}:") or line.startswith(f"{coin} "):
                        coin_cot = line
                        break

                if not coin_cot:
                    continue

                ml_bias = "Unknown"

                for pat in patterns:
                    m = re.search(pat, coin_cot, re.IGNORECASE)
                    if m:
                        groups = [g for g in m.groups() if g and g.upper() in ["BUY", "SELL"]]
                        if groups:
                            ml_bias = groups[0].upper()
                        break

                if ml_bias == "Unknown":
                    m2 = re.search(r"ML\s*(BUY|SELL)\s*(\d+\.?\d*)", coin_cot, re.IGNORECASE)
                    if m2:
                        ml_bias = m2.group(1).upper()

                if "OPEN LONG" in coin_cot or "OPEN SHORT" in coin_cot:
                    continue  # Already in a trade

                if ml_bias == "BUY":
                    ml_buy_llm_hold += 1
                if ml_bias == "SELL":
                    ml_sell_llm_hold += 1

                if ml_bias in ["BUY", "SELL"] and len(hold_examples) < 5:
                    hold_examples.append(
                        f"[{c.get('timestamp')[:16]}] {coin}: ML {ml_bias} but LLM HOLD. Logic: {coin_cot[:150]}..."
                    )

    print(f"ML said BUY, but LLM decided HOLD: {ml_buy_llm_hold} times")
    print(f"ML said SELL, but LLM decided HOLD: {ml_sell_llm_hold} times")
    print("\nSample Logic for HOLDs:")
    for ex in hold_examples:
        print(ex)


if __name__ == "__main__":
    analyze_ml_vs_llm()
