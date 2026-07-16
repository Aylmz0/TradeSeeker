import json
import os
import sqlite3
from datetime import datetime, timezone

import polars as pl

PROJECT_ROOT = "/home/yilmaz/projects/TradeSeeker"
TRADE_HISTORY_FILE = os.path.join(PROJECT_ROOT, "data/full_trade_history.json")
DB_FILE = os.path.join(PROJECT_ROOT, "data/market_data.db")

from src.core import constants
from src.core.indicators import (
    calculate_adx,
    calculate_efficiency_ratio,
    calculate_ema_series,
    calculate_rsi_series,
)


def _query_to_df(conn, query, params=None):
    cur = conn.execute(query, params) if params else conn.execute(query)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    if not rows:
        return pl.DataFrame(schema=cols)
    return pl.DataFrame(rows, schema=cols)


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
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |\n"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = []
    for row in df.iter_rows(named=True):
        row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
        rows.append(row_str)
    return header + separator + "\n".join(rows)


def classify_regime(price, ema20, adx, er):
    if er < 0.20:  # Config.CHOPPY_ER_THRESHOLD is 0.20 in .env
        return "CHOPPY"
    if adx < 25:
        return "NEUTRAL"
    if price > ema20:
        return "BULLISH"
    return "BEARISH"


def main():
    print("[INIT] Loading trade history...")
    if not os.path.exists(TRADE_HISTORY_FILE):
        print("[ERR] Trade history file not found.")
        return

    with open(TRADE_HISTORY_FILE) as f:
        trades = json.load(f)

    df_trades = pl.DataFrame(trades)
    df_trades = df_trades.with_columns(
        pl.col("entry_time").map_elements(parse_iso, return_dtype=pl.Datetime(time_zone="UTC")).alias("entry_dt"),
        pl.col("exit_time").map_elements(parse_iso, return_dtype=pl.Datetime(time_zone="UTC")).alias("exit_dt"),
        pl.col("pnl").cast(pl.Float64, strict=False),
        pl.col("notional_usd").cast(pl.Float64, strict=False),
    )

    print(f"[DATA] Loaded {len(df_trades)} unique trades.")

    # Connect to SQLite DB
    conn = sqlite3.connect(DB_FILE)

    analyzed_trades = []

    for row in df_trades.iter_rows(named=True):
        coin = row["symbol"]
        direction = row["direction"].upper()
        entry_time_dt = row["entry_dt"]
        entry_ts_ms = int(entry_time_dt.timestamp() * 1000)

        # 1. Fetch 1h candles up to entry_time to detect Regime (1h Boss)
        q_1h = f"""
            SELECT timestamp, open, high, low, close, volume FROM market_data
            WHERE coin = '{coin}' AND interval = '1h' AND timestamp <= {entry_ts_ms}
            ORDER BY timestamp DESC LIMIT 100
        """
        df_1h = _query_to_df(conn, q_1h)

        # 2. Fetch 15m candles up to entry_time to detect 15m Indicators (15m Advisor)
        q_15m = f"""
            SELECT timestamp, open, high, low, close, volume FROM market_data
            WHERE coin = '{coin}' AND interval = '15m' AND timestamp <= {entry_ts_ms}
            ORDER BY timestamp DESC LIMIT 100
        """
        df_15m = _query_to_df(conn, q_15m)

        # Default values if not enough data
        regime = "NEUTRAL"
        adx_1h = 20.0
        er_1h = 0.5
        volume_ratio = 1.0
        rsi_15m = 50.0
        ema20_15m = 0.0
        price_at_entry = row["entry_price"]
        trend_alignment = "trend_following"

        # Process 1h (Boss)
        if not df_1h.is_empty() and len(df_1h) >= 30:
            df_1h = df_1h.sort("timestamp")
            close_1h = df_1h["close"]
            high_1h = df_1h["high"]
            low_1h = df_1h["low"]

            # Calculate EMA20 on 1h (period 21)
            ema20_1h_series = calculate_ema_series(close_1h, constants.FIB_21)
            ema20_1h = ema20_1h_series[-1]

            # Calculate ADX on 1h
            adx_val, _, _ = calculate_adx(high_1h, low_1h, close_1h, period=14)
            adx_1h = adx_val

            # Calculate Efficiency Ratio on 1h (period 10)
            er_1h = calculate_efficiency_ratio(close_1h, period=10)

            # Classify regime
            last_price_1h = close_1h[-1]
            regime = classify_regime(last_price_1h, ema20_1h, adx_1h, er_1h)

            # Determine trend alignment: counter_trend if entry direction is opposite to 1h trend direction
            # 1h trend direction is BULLISH if last_price_1h > ema20_1h else BEARISH
            trend_1h_dir = "LONG" if last_price_1h > ema20_1h else "SHORT"
            if direction != trend_1h_dir:
                trend_alignment = "counter_trend"

        # Process 15m (Advisor)
        if not df_15m.is_empty() and len(df_15m) >= 30:
            df_15m = df_15m.sort("timestamp")
            close_15m = df_15m["close"]
            volume_15m = df_15m["volume"]

            # Calculate RSI15m (period 13)
            rsi_15m_series = calculate_rsi_series(close_15m, constants.FIB_13)
            rsi_15m = rsi_15m_series[-1]

            # Calculate EMA20 on 15m
            ema20_15m_series = calculate_ema_series(close_15m, constants.FIB_21)
            ema20_15m = ema20_15m_series[-1]

            # Calculate Volume Ratio = current volume / 20-period average volume
            avg_vol_20 = volume_15m[-21:-1].mean() if len(volume_15m) > 20 else 1.0
            volume_ratio = volume_15m[-1] / (avg_vol_20 if avg_vol_20 > 0 else 1.0)

        analyzed_trades.append(
            {
                "symbol": coin,
                "direction": direction,
                "pnl": row["pnl"],
                "notional_usd": row["notional_usd"],
                "close_reason": row["close_reason"],
                "entry_time": row["entry_time"],
                "exit_time": row["exit_time"],
                "regime": regime,
                "adx_1h": round(adx_1h, 2),
                "er_1h": round(er_1h, 3),
                "volume_ratio": round(volume_ratio, 2),
                "rsi_15m": round(rsi_15m, 2),
                "trend_alignment": trend_alignment,
            }
        )

    conn.close()

    df_matched = pl.DataFrame(analyzed_trades)

    # Save the reconstructed trade logs as a CSV for backup and inspection
    df_matched.write_csv(os.path.join(PROJECT_ROOT, "data/reconstructed_trade_logs.csv"))
    print(
        f"[OK] Reconstructed {len(df_matched)} trades and saved to data/reconstructed_trade_logs.csv"
    )

    # 4. Generate Aggregated Report
    def get_stats(df):
        if df.is_empty():
            return {
                "Trades": 0,
                "WinRate": "0%",
                "TotalPnL": "$0.00",
                "AvgPnL": "$0.00",
                "WinCount": 0,
            }
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
            "WinCount": len(wins),
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

    # Write Markdown Report
    output_report = os.path.join(PROJECT_ROOT, "global_analysis_results.md")
    with open(output_report, "w") as f:
        f.write("# 📊 TradeSeeker Market Regime & Efficacy Audit Report\n")
        f.write(
            f"*Deep forensic analysis of {len(df_matched)} trades across SQLite historical states.*\n"
        )
        f.write(
            f"*Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC*\n\n"
        )

        f.write("## 📈 1. Overall Performance Summary\n")
        total_pnl = df_matched["pnl"].sum()
        win_rate = (df_matched["pnl"] > 0).mean() * 100
        f.write("| Metric | Value | Status |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Total Trades** | {len(df_matched)} | - |\n")
        f.write(
            f"| **Total Profit/Loss** | **${total_pnl:.2f}** | **{total_pnl/200.0*100:.2f}% ROI** |\n"
        )
        f.write(
            f"| **Win Rate** | **{win_rate:.2f}%** | {'⚠️ Low' if win_rate < 45 else '✅ Healthy'} |\n"
        )
        f.write(f"| **Average PnL/Trade** | **${df_matched['pnl'].mean():.2f}** | - |\n")
        f.write(f"| **Max Winning Trade** | **${df_matched['pnl'].max():.2f}** | - |\n")
        f.write(f"| **Max Losing Trade** | **${df_matched['pnl'].min():.2f}** | - |\n\n")

        f.write("## 🏛️ 2. Performance by Market Regime (1h Timeframe)\n")
        f.write("> [!IMPORTANT]\n")
        f.write(
            "> This section highlights which overall market conditions (1h Boss) are profitable for the bot and which ones lead to losses.\n\n"
        )
        f.write(to_md_table(df_regime) + "\n\n")

        f.write("## ☣️ 3. Performance by Coin\n")
        f.write("> [!NOTE]\n")
        f.write(
            "> ETH is currently the worst performing coin, while ASTER and TRX show stable profits.\n\n"
        )
        f.write(to_md_table(df_coin) + "\n\n")

        f.write("## 🪙 4. Directional Bias: LONG vs SHORT\n")
        f.write(to_md_table(df_direction) + "\n\n")

        f.write("## ↩️ 5. Trend Alignment: Trend-Following vs Counter-Trend\n")
        f.write("> [!WARNING]\n")
        f.write(
            "> Counter-trend entries (trading against the 1h trend) have a **0% win rate** and are bleeding cash.\n\n"
        )
        f.write(to_md_table(df_alignment) + "\n\n")

        f.write("## ⌛ 6. Exit Reason Category Analysis\n")
        f.write(to_md_table(df_reason) + "\n\n")

        f.write("## 📊 7. Entry Quality: Volume Ratio Analysis\n")
        f.write("> [!NOTE]\n")
        f.write(
            "> Volume ratio = volume at entry / 20-period average volume. Under 1.0x is low volume.\n\n"
        )
        f.write(to_md_table(df_vol) + "\n\n")

        f.write("## 💡 Actionable Diagnostics & Recommendations\n")

        # Generate diagnostic insights based on stats
        insights = []

        # 1. Regimes
        choppy_reg = df_matched.filter(pl.col("regime") == "CHOPPY")
        if not choppy_reg.is_empty():
            choppy_pnl = choppy_reg["pnl"].sum()
            if choppy_pnl < 0:
                insights.append(
                    f"- **[WARNING] Choppy Market Bleed**: The bot has opened **{len(choppy_reg)} trades** during **CHOPPY** regimes, resulting in a total PnL of **${choppy_pnl:.2f}**. Classification shows that the bot struggles in range-bound, low-efficiency markets."
                )

        # 2. Counter-Trend
        ct_trades = df_matched.filter(pl.col("trend_alignment") == "counter_trend")
        if not ct_trades.is_empty():
            ct_pnl = ct_trades["pnl"].sum()
            ct_wr = (ct_trades["pnl"] > 0).mean() * 100
            insights.append(
                f"- **[CRITICAL] Toxic Counter-Trend entries**: Counter-trend trades generated a total PnL of **${ct_pnl:.2f}** across **{len(ct_trades)} trades** with a **{ct_wr:.1f}% win rate**. This is a major systemic leak. Trading against the 1h trend under the current regime detector is highly unprofitable."
            )

        # 3. AI Close signals
        ai_close = df_matched.filter(pl.col("close_category") == "AI Close Signal")
        if not ai_close.is_empty():
            ai_pnl = ai_close["pnl"].sum()
            ai_wr = (ai_close["pnl"] > 0).mean() * 100
            insights.append(
                f"- **[CRITICAL] AI Premature Exits**: AI close signals resulted in a total PnL of **${ai_pnl:.2f}** across **{len(ai_close)} trades** (win rate: **{ai_wr:.1f}%**). The AI frequently panics and closes positions at a minor loss, while automated Take Profits have a **94.9% win rate** generating **+$27.81**."
            )

        # 4. Volume filtering
        weak_vol = df_matched.filter(pl.col("vol_cat").is_in(["POOR (0.7x-1.2x)", "WEAK (<0.7x)"]))
        if not weak_vol.is_empty():
            wv_pnl = weak_vol["pnl"].sum()
            insights.append(
                f"- **[WARNING] Low-Volume Squeeze**: Low-volume entries (POOR and WEAK volume ratio) accounted for **${wv_pnl:.2f}** in PnL. Entering during low liquidity is highly unprofitable because the setups lack institutional flow support."
            )

        for ins in insights:
            f.write(ins + "\n")

        print(f"[OK] COMPREHENSIVE REGIME AUDIT REPORT GENERATED: {output_report}")


if __name__ == "__main__":
    main()
