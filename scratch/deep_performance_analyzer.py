import glob
import json
import os
from datetime import datetime, timezone

import polars as pl


PROJECT_ROOT = "/home/yilmaz/projects/TradeSeeker"
TRADE_HISTORY_FILE = os.path.join(PROJECT_ROOT, "data/full_trade_history.json")


def parse_iso(ts_str):
    if not ts_str:
        return None
    try:
        return pl.Series([ts_str]).str.to_datetime(format="ISO8601", time_zone="UTC")[0]
    except Exception:
        return None


def to_md_table(df):
    if df.is_empty():
        return "No data"
    # Column headers
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |\n"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |\n"

    # Rows
    rows = []
    for row in df.iter_rows(named=True):
        row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
        rows.append(row_str)

    return header + separator + "\n".join(rows)


def main():
    print("[INIT] Loading history and backups...")

    # 1. Load all trades
    if not os.path.exists(TRADE_HISTORY_FILE):
        print("[ERR] Trade history file not found.")
        return

    with open(TRADE_HISTORY_FILE) as f:
        trades = json.load(f)

    df_trades = pl.DataFrame(trades)
    df_trades = df_trades.with_columns(
        pl.col("entry_time").map_elements(parse_iso, return_dtype=pl.Datetime(time_zone="UTC")).alias("entry_dt")
    )
    df_trades = df_trades.with_columns(
        pl.col("exit_time").map_elements(parse_iso, return_dtype=pl.Datetime(time_zone="UTC")).alias("exit_dt")
    )
    df_trades = df_trades.with_columns(pl.col("pnl").cast(pl.Float64, strict=False).alias("pnl"))
    df_trades = df_trades.with_columns(
        pl.col("notional_usd").cast(pl.Float64, strict=False).alias("notional_usd")
    )

    print(f"[DATA] Loaded {len(df_trades)} unique trades.")

    # 2. Gather all cycle logs from active data and backups
    all_cycles = []

    # Check active data
    active_cycle_file = os.path.join(PROJECT_ROOT, "data/cycle_history.json")
    if os.path.exists(active_cycle_file):
        with open(active_cycle_file) as f:
            try:
                all_cycles.extend(json.load(f))
            except:
                pass

    # Check backup directories
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

    # De-duplicate cycles by timestamp
    unique_cycles = {}
    for c in all_cycles:
        if not isinstance(c, dict):
            continue
        ts = c.get("timestamp")
        if ts:
            unique_cycles[ts] = c

    cycles_list = list(unique_cycles.values())
    for c in cycles_list:
        c["dt"] = parse_iso(c.get("timestamp"))

    cycles_list = sorted([c for c in cycles_list if c["dt"]], key=lambda x: x["dt"])
    print(f"[DATA] Loaded {len(cycles_list)} unique cycles.")

    # 3. Match each trade to its corresponding cycle state AT ENTRY
    matched_trades = []

    for trade in df_trades.iter_rows(named=True):
        entry_time = trade["entry_dt"]
        coin = trade["symbol"]
        direction = trade["direction"].upper()

        # Find closest cycle that happened BEFORE or AT the trade entry time
        matched_cycle = None
        for c in reversed(cycles_list):
            if c["dt"] <= entry_time:
                # Check if this cycle mentions this coin in market data
                m_data = c.get("market_data", [])
                if isinstance(m_data, list):
                    coin_data = [item for item in m_data if item.get("coin") == coin]
                    if coin_data:
                        matched_cycle = c
                        break
                # Fallback to general time matching within 10 minutes
                if (entry_time - c["dt"]).total_seconds() < 600:
                    matched_cycle = c
                    break

        regime = "UNKNOWN"
        adx_strength = "UNKNOWN"
        volatility_state = "UNKNOWN"
        price_location = "UNKNOWN"
        momentum = "UNKNOWN"
        volume_ratio = 1.0
        volume_support = "UNKNOWN"
        structure_15m = "UNKNOWN"
        ml_signal = "UNKNOWN"
        ml_prob = 50.0
        confidence = 0.0
        trend_alignment = "UNKNOWN"
        funding_rate = 0.0

        if matched_cycle:
            m_data = matched_cycle.get("market_data", [])
            coin_data = {}
            if isinstance(m_data, list):
                for item in m_data:
                    if item.get("coin") == coin:
                        coin_data = item
                        break

            # Extract regime context
            m_ctx = coin_data.get("market_context", {}) if isinstance(coin_data, dict) else {}
            if isinstance(m_ctx, dict):
                regime = m_ctx.get("regime", "UNKNOWN")
                adx_strength = m_ctx.get("trend_strength_adx", "UNKNOWN")
                volatility_state = m_ctx.get("volatility_state", "UNKNOWN")
                price_location = m_ctx.get("price_location", "UNKNOWN")

            # Technical summary
            t_sum = coin_data.get("technical_summary", {}) if isinstance(coin_data, dict) else {}
            if isinstance(t_sum, dict):
                momentum = t_sum.get("momentum", "UNKNOWN")
                volume_ratio = t_sum.get("volume_ratio", 1.0)
                volume_support = t_sum.get("volume_support", "UNKNOWN")
                structure_15m = t_sum.get("structure_15m", "UNKNOWN")

            # ML consensus
            ml_c = coin_data.get("ml_consensus", {}) if isinstance(coin_data, dict) else {}
            if isinstance(ml_c, dict) and ml_c:
                ml_signal = ml_c.get("signal", "UNKNOWN")
                ml_prob = ml_c.get("probability", 50.0)

            # Sentiment
            sent = coin_data.get("sentiment", {}) if isinstance(coin_data, dict) else {}
            if isinstance(sent, dict):
                funding_rate = sent.get("funding_rate", 0.0)

            # AI decision details from decisions field
            decs = matched_cycle.get("decisions", {})
            coin_dec = decs.get(coin, {}) if isinstance(decs, dict) else {}
            if isinstance(coin_dec, dict):
                confidence = coin_dec.get("confidence", 0.0)
                trend_alignment = coin_dec.get("classification", "UNKNOWN")
                if trend_alignment == "UNKNOWN":
                    # Try to infer trend alignment
                    if (
                        "counter" in str(coin_dec.get("strategy", "")).lower()
                        or "counter" in str(coin_dec.get("classification", "")).lower()
                    ):
                        trend_alignment = "counter_trend"
                    elif "trend_following" in str(coin_dec.get("strategy", "")).lower():
                        trend_alignment = "trend_following"

        matched_trades.append(
            {
                "symbol": coin,
                "direction": direction,
                "pnl": trade["pnl"],
                "notional_usd": trade["notional_usd"],
                "close_reason": trade["close_reason"],
                "entry_time": trade["entry_time"],
                "exit_time": trade["exit_time"],
                "regime": regime,
                "adx_strength": adx_strength,
                "volatility_state": volatility_state,
                "price_location": price_location,
                "momentum": momentum,
                "volume_ratio": volume_ratio,
                "volume_support": volume_support,
                "structure_15m": structure_15m,
                "ml_signal": ml_signal,
                "ml_prob": ml_prob,
                "confidence": confidence,
                "trend_alignment": trend_alignment,
                "funding_rate": funding_rate,
            }
        )

    df_matched = pl.DataFrame(matched_trades)

    # 4. Generate Reports
    print("\n[ANALYSIS] Processing metrics...")

    def get_stats(df):
        if df.is_empty():
            return {"Trades": 0, "WinRate": "0%", "TotalPnL": "$0.00", "AvgPnL": "$0.00"}
        trades_count = len(df)
        wins = df.filter(pl.col("pnl") > 0)
        win_rate = len(wins) / trades_count * 100
        total_pnl = df["pnl"].sum()
        avg_pnl = df["pnl"].mean()
        return {
            "Trades": trades_count,
            "WinRate": f"{win_rate:.1f}%",
            "TotalPnL": f"${total_pnl:.2f}",
            "AvgPnL": f"${avg_pnl:.2f}",
        }

    # Group by Regime
    regime_groups = []
    for name, group in df_matched.group_by("regime"):
        stats = get_stats(group)
        stats["Regime"] = name
        regime_groups.append(stats)
    df_regime = pl.DataFrame(regime_groups)

    # Group by Coin
    coin_groups = []
    for name, group in df_matched.group_by("symbol"):
        stats = get_stats(group)
        stats["Coin"] = name
        coin_groups.append(stats)
    df_coin = pl.DataFrame(coin_groups).sort("Trades", descending=True)

    # Group by Direction
    direction_groups = []
    for name, group in df_matched.group_by("direction"):
        stats = get_stats(group)
        stats["Direction"] = name
        direction_groups.append(stats)
    df_direction = pl.DataFrame(direction_groups)

    # Group by Trend Alignment
    alignment_groups = []
    for name, group in df_matched.group_by("trend_alignment"):
        stats = get_stats(group)
        stats["Alignment"] = name
        alignment_groups.append(stats)
    df_alignment = pl.DataFrame(alignment_groups)

    # Group by Close Reason Category
    def cat_reason(r):
        r = str(r).lower()
        if "profit taking at" in r or "taking profit" in r:
            return "Take Profit Trigger"
        if "stop loss" in r:
            return "Stop Loss Trigger"
        if "ai close_position" in r:
            return "AI Close Signal"
        if "position margin" in r:
            return "Margin Limit Cut"
        if "negative for" in r:
            return "Extended Loss Timeout"
        return "Other"

    df_matched = df_matched.with_columns(
        pl.col("close_reason").map_elements(cat_reason, return_dtype=pl.Utf8).alias("close_category")
    )
    reason_groups = []
    for name, group in df_matched.group_by("close_category"):
        stats = get_stats(group)
        stats["ReasonCategory"] = name
        reason_groups.append(stats)
    df_reason = pl.DataFrame(reason_groups).sort("Trades", descending=True)

    # Group by Volume Quality
    def cat_vol(v):
        try:
            val = float(v)
            if val > 2.5:
                return "EXCELLENT (>2.5x)"
            if val > 1.8:
                return "GOOD (1.8x-2.5x)"
            if val > 1.2:
                return "FAIR (1.2x-1.8x)"
            if val > 0.7:
                return "POOR (0.7x-1.2x)"
            return "WEAK (<0.7x)"
        except:
            return "UNKNOWN"

    df_matched = df_matched.with_columns(
        pl.col("volume_ratio").map_elements(cat_vol, return_dtype=pl.Utf8).alias("vol_cat")
    )
    vol_groups = []
    for name, group in df_matched.group_by("vol_cat"):
        stats = get_stats(group)
        stats["VolumeRatio"] = name
        vol_groups.append(stats)
    df_vol = pl.DataFrame(vol_groups)

    # Group by Momentum
    mom_groups = []
    for name, group in df_matched.group_by("momentum"):
        stats = get_stats(group)
        stats["Momentum"] = name
        mom_groups.append(stats)
    df_mom = pl.DataFrame(mom_groups)

    # Write Markdown Report
    output_report = os.path.join(PROJECT_ROOT, "global_analysis_results.md")
    with open(output_report, "w") as f:
        f.write("# 📊 Deep TradeSeeker Performance & Market Analysis\n")
        f.write(f"*Analysis of {len(df_matched)} trades across {len(cycles_list)} cycles*\n")
        f.write(
            f"*Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC*\n\n"
        )

        f.write("## 📈 1. Global Metrics\n")
        total_pnl = df_matched["pnl"].sum()
        win_rate = df_matched.select((pl.col("pnl") > 0).mean()).item() * 100
        f.write(f"- **Total Trades**: {len(df_matched)}\n")
        f.write(f"- **Total Profit/Loss**: **${total_pnl:.2f}**\n")
        f.write(f"- **Win Rate**: **{win_rate:.2f}%**\n")
        f.write(f"- **Average Return/Trade**: ${df_matched['pnl'].mean():.2f}\n")
        f.write(f"- **Max Profit**: ${df_matched['pnl'].max():.2f}\n")
        f.write(f"- **Max Loss**: ${df_matched['pnl'].min():.2f}\n\n")

        f.write("## 🏛️ 2. Performance by Market Regime\n")
        f.write("> [!NOTE]\n")
        f.write(
            "> This table highlights which higher-timeframe (1h Boss) regimes generate the best and worst PnL.\n\n"
        )
        f.write(to_md_table(df_regime) + "\n\n")

        f.write("## ☣️ 3. Performance by Coin\n")
        f.write(to_md_table(df_coin) + "\n\n")

        f.write("## 🪙 4. Strategy Analysis: Long vs Short\n")
        f.write(to_md_table(df_direction) + "\n\n")

        f.write("## ↩️ 5. Trend Alignment: Trend-Following vs Counter-Trend\n")
        f.write(to_md_table(df_alignment) + "\n\n")

        f.write("## ⌛ 6. Exit Reason Breakdown\n")
        f.write(to_md_table(df_reason) + "\n\n")

        f.write("## 📊 7. Inflow Quality: Volume Ratio analysis\n")
        f.write(to_md_table(df_vol) + "\n\n")

        f.write("## ⚡ 8. Momentum Analysis at Entry\n")
        f.write(to_md_table(df_mom) + "\n\n")

        # Key Insights
        f.write("## 💡 Actionable Insights & Diagnostics\n")

        # Diagnostics
        insights = []

        # 1. Regimes
        choppy_data = df_matched.filter(pl.col("regime") == "CHOPPY")
        if not choppy_data.is_empty() and choppy_data["pnl"].sum() < 0:
            insights.append(
                f"- **[WARNING] Choppy Regimes Bleed**: Trades in **CHOPPY** regimes generated losses across {len(choppy_data)} trades. Consider reducing position sizing or increasing confidence thresholds to 0.75 in CHOPPY conditions."
            )

        # 2. Alignment
        ct_data = df_matched.filter(pl.col("trend_alignment") == "counter_trend")
        if not ct_data.is_empty() and ct_data["pnl"].sum() < 0:
            ct_pnl = ct_data["pnl"].sum()
            insights.append(
                f"- **[CRITICAL] Counter-Trend Disadvantage**: Counter-trend entries generated a total PnL of **${ct_pnl:.2f}** with a win rate of **{ct_data.select((pl.col('pnl') > 0).mean()).item()*100:.1f}%**. Counter-trend trades are highly toxic compared to trend-following setups. We should increase MIN_CONFIDENCE for counter-trend entries to **0.75**."
            )

        # 3. Volume
        poor_vol_data = df_matched.filter(pl.col("vol_cat").is_in(["POOR (0.7x-1.2x)", "WEAK (<0.7x)"]))
        if not poor_vol_data.is_empty():
            vol_pnl = poor_vol_data["pnl"].sum()
            insights.append(
                f"- **[WARNING] Low-Volume Entries**: Entering trades with POOR or WEAK volume ratios generated **${vol_pnl:.2f}** in PnL. This confirms that entering when volume is low degrades the execution edge. Stricter volume filtering is required."
            )

        # 4. Exit reason
        ai_close_data = df_matched.filter(pl.col("close_category") == "AI Close Signal")
        if not ai_close_data.is_empty():
            ai_pnl = ai_close_data["pnl"].sum()
            insights.append(
                f"- **[CRITICAL] AI Premature Closes**: AI close signals generated **${ai_pnl:.2f}** across {len(ai_close_data)} trades. The AI's own close decisions are highly unprofitable, representing a major leak in the exit strategy."
            )

        for ins in insights:
            f.write(ins + "\n")

    print(f"[OK] COMPREHENSIVE PERFORMANCE ANALYSIS COMPLETED: {output_report}")


if __name__ == "__main__":
    main()
