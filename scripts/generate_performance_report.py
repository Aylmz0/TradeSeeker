#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timezone
import polars as pl

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.performance_monitor import PerformanceMonitor
from config.config import Config


def format_currency(value):
    return f"${value:,.2f}"


def format_pct(value):
    return f"{value:.2f}%"


class MasterReportingEngine:
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.output_file = "global_analysis_results.md"

    def run(self):
        print(
            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] [INIT] Starting Global History Aggregation (825+ Cycles)..."
        )

        # 1. Aggregate all data
        data = self.monitor.aggregate_all_history()
        trades = data["trades"]
        performance = data["performance"]
        cycles = data["cycles"]

        if not trades:
            print("[WARN] No trade history found. Check history_backups/ directory.")
            return

        print(
            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] [DATA] Found {len(trades)} trades and {len(performance)} performance snapshots."
        )

        # 2. Convert to DataFrames for easier analysis
        df_trades = pl.DataFrame(trades)
        df_perf = pl.DataFrame(performance)

        # Ensure numeric types
        for col in ["pnl", "notional_usd", "entry_price", "exit_price"]:
            if col in df_trades.columns:
                df_trades = df_trades.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        if "total_value" in df_perf.columns:
            df_perf = df_perf.with_columns(pl.col("total_value").cast(pl.Float64, strict=False))

        # 3. Calculate Global Metrics
        total_pnl = df_trades["pnl"].sum() if not df_trades.is_empty() else 0
        win_rate = (df_trades["pnl"] > 0).mean() * 100 if not df_trades.is_empty() else 0
        profit_factor = (
            abs(
                df_trades.filter(pl.col("pnl") > 0)["pnl"].sum()
                / df_trades.filter(pl.col("pnl") < 0)["pnl"].sum()
            )
            if not df_trades.is_empty() and df_trades.filter(pl.col("pnl") < 0)["pnl"].sum() != 0
            else float("inf")
        )
        avg_trade = df_trades["pnl"].mean() if not df_trades.is_empty() else 0

        initial_value = Config.INITIAL_BALANCE
        current_value = df_perf["total_value"][-1] if not df_perf.is_empty() else initial_value
        total_return_pct = (
            ((current_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
        )

        # 4. Deep Dive Analysis

        # Coin Performance
        coin_perf = (
            df_trades.group_by("symbol")
            .agg(
                pl.col("pnl").sum().alias("total_pnl"),
                pl.len().alias("trade_count"),
                pl.col("pnl").mean().alias("avg_pnl"),
                ((pl.col("pnl") > 0).mean() * 100).alias("win_rate"),
            )
            .sort("total_pnl")
        )  # Worst first

        # Toxic Coins (Negative PnL or Very Low Win Rate)
        toxic_coins = coin_perf.filter((pl.col("total_pnl") < 0) & (pl.col("trade_count") >= 3))

        # Directional Analysis (handle empty data)
        dir_perf = pl.DataFrame()
        if not df_trades.is_empty() and "direction" in df_trades.columns:
            dir_perf = df_trades.group_by("direction").agg(
                pl.col("pnl").sum().alias("total_pnl"),
                pl.len().alias("trade_count"),
                pl.col("pnl").mean().alias("avg_pnl"),
                ((pl.col("pnl") > 0).mean() * 100).alias("win_rate"),
            )

        # Hour of Day Analysis
        df_trades = df_trades.with_columns(
            pl.col("entry_time").str.to_datetime(format="ISO8601", time_zone="UTC")
        )
        df_trades = df_trades.with_columns(pl.col("entry_time").dt.hour().alias("hour"))
        hour_perf = df_trades.group_by("hour").agg(pl.col("pnl").sum().alias("pnl_sum"))

        # 5. Generate Markdown Report
        with open(self.output_file, "w") as f:
            f.write(f"# 📊 Global TradeSeeker Performance Report\n")
            f.write(
                f"*Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC*\n\n"
            )

            f.write(f"## 📈 Global Summary\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"| :--- | :--- |\n")
            f.write(f"| **Total Cycles Scanned** | {len(cycles)} |\n")
            f.write(f"| **Total Trades** | {len(df_trades)} |\n")
            f.write(f"| **Initial Balance** | {format_currency(initial_value)} |\n")
            f.write(f"| **Current Balance** | {format_currency(current_value)} |\n")
            f.write(f"| **Total Profit/Loss** | **{format_currency(total_pnl)}** |\n")
            f.write(f"| **Total ROI** | **{format_pct(total_return_pct)}** |\n")
            f.write(f"| **Win Rate** | {format_pct(win_rate)} |\n")
            f.write(f"| **Profit Factor** | {profit_factor:.2f} |\n")
            f.write(f"| **Average PnL/Trade** | {format_currency(avg_trade)} |\n\n")

            f.write(f"## 📉 Drawdown Analysis\n")
            peak = df_perf.with_columns(pl.col("total_value").cum_max().alias("peak"))["peak"]
            drawdown = (df_perf["total_value"] - peak) / peak * 100
            max_dd = drawdown.min()
            f.write(f"| Metric | Value |\n")
            f.write(f"| :--- | :--- |\n")
            f.write(f"| **Maximum Drawdown** | **{max_dd:.2f}%** |\n")
            f.write(f"| **Recovery Status** | {'In DD' if drawdown[-1] < 0 else 'Recovered'} |\n\n")

            f.write(f"## ☣️ Toxic Asset Detection (The 'Hit List')\n")
            f.write(f"*Coins with at least 3 trades and negative cumulative PnL*\n\n")
            f.write(f"| Coin | Trades | Win Rate | Total PnL | Avg PnL |\n")
            f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")
            for row in toxic_coins.iter_rows(named=True):
                f.write(
                    f"| {row['symbol']} | {row['trade_count']} | {format_pct(row['win_rate'])} | **{format_currency(row['total_pnl'])}** | {format_currency(row['avg_pnl'])} |\n"
                )
            f.write("\n")

            f.write(f"## 🪙 Strategy Analysis: Long vs Short\n")
            f.write(f"| Direction | Trades | Win Rate | Total PnL | Avg PnL |\n")
            f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")
            if not dir_perf.is_empty():
                for row in dir_perf.iter_rows(named=True):
                    f.write(
                        f"| {row['direction'].upper()} | {row['trade_count']} | {format_pct(row['win_rate'])} | {format_currency(row['total_pnl'])} | {format_currency(row['avg_pnl'])} |\n"
                    )
            else:
                f.write(f"| N/A | 0 | 0% | $0.00 | $0.00 |\n")
            f.write("\n")

            f.write(f"## ⏰ Temporal Performance (Hourly)\n")
            f.write(f"| Hour (UTC) | PnL Sum |\n")
            f.write(f"| :--- | :---: |\n")
            if not hour_perf.is_empty():
                for row in hour_perf.iter_rows(named=True):
                    h = row["hour"]
                    p = row["pnl_sum"]
                    f.write(f"| {h:02d}:00 | {format_currency(p)} |\n")
            else:
                f.write(f"| N/A | $0.00 |\n")
            f.write("\n")

            f.write(f"## 💡 Actionable Insights & Recommendations\n")
            insights = []
            if win_rate < 45:
                insights.append(
                    "- **[CRITICAL] Low Hit Rate**: The AI accuracy is below 45%. This suggests entry signals are too noisy or market regimes are misidentified."
                )
            if profit_factor < 1.0:
                insights.append(
                    "- **[WARNING] Negative Expectancy**: Profit Factor < 1.0 means you are losing more on losses than gaining on wins. Tighten Stop Losses or improve TP targets."
                )
            if not toxic_coins.is_empty():
                worst_coin = toxic_coins["symbol"][0]
                insights.append(
                    f"- **[ACTION] Toxic Asset**: {worst_coin} is your worst performer. Consider blacklisting it or increasing confidence requirements for this coin."
                )
            if (
                "short" in dir_perf["direction"].to_list()
                and "long" in dir_perf["direction"].to_list()
            ):
                short_pnl = dir_perf.filter(pl.col("direction") == "short")["total_pnl"][0]
                long_pnl = dir_perf.filter(pl.col("direction") == "long")["total_pnl"][0]
                if short_pnl < long_pnl * 0.5:
                    insights.append(
                        "- **[STRATEGY] Short Bias**: Short performance is significantly worse than Longs. Consider disabling shorts in non-bearish regimes."
                    )

            for insight in insights:
                f.write(f"{insight}\n")

        print(
            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] [OK] MASTER REPORT GENERATED: {self.output_file}"
        )
        self._print_cli_summary(total_pnl, win_rate, total_return_pct, toxic_coins)

    def _print_cli_summary(self, total_pnl, win_rate, total_return, toxic_coins):
        print("\n" + "=" * 50)
        print("      TRADE SEEKER MASTER PERFORMANCE SUMMARY      ")
        print("=" * 50)
        print(f" TOTAL P/L      : {format_currency(total_pnl)}")
        print(f" TOTAL ROI      : {format_pct(total_return)}")
        print(f" WIN RATE       : {format_pct(win_rate)}")
        print("-" * 50)
        if not toxic_coins.is_empty():
            print(
                f" WORST ASSET    : {toxic_coins['symbol'][0]} ({format_currency(toxic_coins['total_pnl'][0])})"
            )
        print("=" * 50)
        print(f"Detailed Markdown Report: {self.output_file}\n")


if __name__ == "__main__":
    engine = MasterReportingEngine()
    engine.run()
