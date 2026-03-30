#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np

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
        df_trades = pd.DataFrame(trades)
        df_perf = pd.DataFrame(performance)

        # Ensure numeric types
        for col in ["pnl", "notional_usd", "entry_price", "exit_price"]:
            if col in df_trades.columns:
                df_trades[col] = pd.to_numeric(df_trades[col], errors="coerce")

        if "total_value" in df_perf.columns:
            df_perf["total_value"] = pd.to_numeric(df_perf["total_value"], errors="coerce")

        # 3. Calculate Global Metrics
        total_pnl = df_trades["pnl"].sum()
        win_rate = (df_trades["pnl"] > 0).mean() * 100
        profit_factor = (
            abs(
                df_trades[df_trades["pnl"] > 0]["pnl"].sum()
                / df_trades[df_trades["pnl"] < 0]["pnl"].sum()
            )
            if df_trades[df_trades["pnl"] < 0]["pnl"].sum() != 0
            else float("inf")
        )
        avg_trade = df_trades["pnl"].mean()

        initial_value = Config.INITIAL_BALANCE
        current_value = df_perf["total_value"].iloc[-1] if not df_perf.empty else initial_value
        total_return_pct = ((current_value - initial_value) / initial_value) * 100

        # 4. Deep Dive Analysis

        # Coin Performance
        coin_perf = df_trades.groupby("symbol").agg(
            {"pnl": ["sum", "count", "mean"], "symbol": "first"}
        )
        coin_perf.columns = ["total_pnl", "trade_count", "avg_pnl", "symbol_name"]
        coin_perf["win_rate"] = df_trades.groupby("symbol").apply(
            lambda x: (x["pnl"] > 0).mean() * 100
        )
        coin_perf = coin_perf.sort_values("total_pnl", ascending=True)  # Worst first

        # Toxic Coins (Negative PnL or Very Low Win Rate)
        toxic_coins = coin_perf[(coin_perf["total_pnl"] < 0) & (coin_perf["trade_count"] >= 3)]

        # Directional Analysis
        dir_perf = df_trades.groupby("direction").agg({"pnl": ["sum", "count", "mean"]})
        dir_perf.columns = ["total_pnl", "trade_count", "avg_pnl"]
        dir_perf["win_rate"] = df_trades.groupby("direction").apply(
            lambda x: (x["pnl"] > 0).mean() * 100
        )

        # Hour of Day Analysis
        df_trades["entry_time"] = pd.to_datetime(
            df_trades["entry_time"], format="ISO8601", utc=True
        )
        df_trades["hour"] = df_trades["entry_time"].dt.hour
        hour_perf = df_trades.groupby("hour")["pnl"].sum()

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
            peak = df_perf["total_value"].cummax()
            drawdown = (df_perf["total_value"] - peak) / peak * 100
            max_dd = drawdown.min()
            f.write(f"| Metric | Value |\n")
            f.write(f"| :--- | :--- |\n")
            f.write(f"| **Maximum Drawdown** | **{max_dd:.2f}%** |\n")
            f.write(
                f"| **Recovery Status** | {'In DD' if drawdown.iloc[-1] < 0 else 'Recovered'} |\n\n"
            )

            f.write(f"## ☣️ Toxic Asset Detection (The 'Hit List')\n")
            f.write(f"*Coins with at least 3 trades and negative cumulative PnL*\n\n")
            f.write(f"| Coin | Trades | Win Rate | Total PnL | Avg PnL |\n")
            f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")
            for _, row in toxic_coins.iterrows():
                f.write(
                    f"| {row['symbol_name']} | {row['trade_count']} | {format_pct(row['win_rate'])} | **{format_currency(row['total_pnl'])}** | {format_currency(row['avg_pnl'])} |\n"
                )
            f.write("\n")

            f.write(f"## 🪙 Strategy Analysis: Long vs Short\n")
            f.write(f"| Direction | Trades | Win Rate | Total PnL | Avg PnL |\n")
            f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")
            for idx, row in dir_perf.iterrows():
                f.write(
                    f"| {idx.upper()} | {row['trade_count']} | {format_pct(row['win_rate'])} | {format_currency(row['total_pnl'])} | {format_currency(row['avg_pnl'])} |\n"
                )
            f.write("\n")

            f.write(f"## ⏰ Temporal Performance (Hourly)\n")
            f.write(f"| Hour (UTC) | PnL Sum |\n")
            f.write(f"| :--- | :---: |\n")
            for h, p in hour_perf.items():
                f.write(f"| {h:02d}:00 | {format_currency(p)} |\n")
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
            if not toxic_coins.empty:
                worst_coin = toxic_coins.index[0]
                insights.append(
                    f"- **[ACTION] Toxic Asset**: {worst_coin} is your worst performer. Consider blacklisting it or increasing confidence requirements for this coin."
                )
            if dir_perf.loc["short"]["total_pnl"] < dir_perf.loc["long"]["total_pnl"] * 0.5:
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
        if not toxic_coins.empty:
            print(
                f" WORST ASSET    : {toxic_coins.index[0]} ({format_currency(toxic_coins.iloc[0]['total_pnl'])})"
            )
        print("=" * 50)
        print(f"Detailed Markdown Report: {self.output_file}\n")


if __name__ == "__main__":
    engine = MasterReportingEngine()
    engine.run()
