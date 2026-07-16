# 📊 TradeSeeker Market Regime & Efficacy Audit Report
*Deep forensic analysis of 114 trades across SQLite historical states.*
*Generated at: 2026-06-03 04:43:13 UTC*

## 📈 1. Overall Performance Summary
| Metric | Value | Status |
| :--- | :---: | :---: |
| **Total Trades** | 114 | - |
| **Total Profit/Loss** | **$-26.57** | **-13.28% ROI** |
| **Win Rate** | **42.11%** | ⚠️ Low |
| **Average PnL/Trade** | **$-0.23** | - |
| **Max Winning Trade** | **$2.34** | - |
| **Max Losing Trade** | **$-2.63** | - |

## 🏛️ 2. Performance by Market Regime (1h Timeframe)
> [!IMPORTANT]
> This section highlights which overall market conditions (1h Boss) are profitable for the bot and which ones lead to losses.

| Trades | WinRate | TotalPnL | AvgPnL | WinCount | Regime |
| --- | --- | --- | --- | --- | --- |
| 23 | 56.5% | $4.85 | $0.21 | 13 | BEARISH |
| 27 | 29.6% | $-15.80 | $-0.59 | 8 | BULLISH |
| 47 | 48.9% | $-6.80 | $-0.14 | 23 | CHOPPY |
| 17 | 23.5% | $-8.82 | $-0.52 | 4 | NEUTRAL |

## ☣️ 3. Performance by Coin
> [!NOTE]
> ETH is currently the worst performing coin, while ASTER and TRX show stable profits.

| Trades | WinRate | TotalPnL | AvgPnL | WinCount | Coin |
| --- | --- | --- | --- | --- | --- |
| 26 | 42.3% | $-2.70 | $-0.10 | 11 | XRP |
| 25 | 36.0% | $-8.92 | $-0.36 | 9 | DOGE |
| 23 | 47.8% | $-4.59 | $-0.20 | 11 | SOL |
| 18 | 16.7% | $-15.91 | $-0.88 | 3 | ETH |
| 13 | 53.8% | $2.65 | $0.20 | 7 | TRX |
| 9 | 77.8% | $2.90 | $0.32 | 7 | ASTER |

## 🪙 4. Directional Bias: LONG vs SHORT
| Trades | WinRate | TotalPnL | AvgPnL | WinCount | Direction |
| --- | --- | --- | --- | --- | --- |
| 61 | 44.3% | $-12.93 | $-0.21 | 27 | LONG |
| 53 | 39.6% | $-13.64 | $-0.26 | 21 | SHORT |

## ↩️ 5. Trend Alignment: Trend-Following vs Counter-Trend
> [!WARNING]
> Counter-trend entries (trading against the 1h trend) have a **0% win rate** and are bleeding cash.

| Trades | WinRate | TotalPnL | AvgPnL | WinCount | Alignment |
| --- | --- | --- | --- | --- | --- |
| 20 | 15.0% | $-16.29 | $-0.81 | 3 | counter_trend |
| 94 | 47.9% | $-10.28 | $-0.11 | 45 | trend_following |

## ⌛ 6. Exit Reason Category Analysis
| Trades | WinRate | TotalPnL | AvgPnL | WinCount | ReasonCategory |
| --- | --- | --- | --- | --- | --- |
| 39 | 94.9% | $27.81 | $0.71 | 37 | Take Profit Trigger |
| 33 | 0.0% | $-27.01 | $-0.82 | 0 | Other |
| 29 | 17.2% | $-26.27 | $-0.91 | 5 | AI Close Signal |
| 11 | 45.5% | $-2.92 | $-0.27 | 5 | Margin Limit Cut |
| 1 | 0.0% | $-0.33 | $-0.33 | 0 | Extended Loss Timeout |
| 1 | 100.0% | $2.16 | $2.16 | 1 | Stop Loss Trigger |

## 📊 7. Entry Quality: Volume Ratio Analysis
> [!NOTE]
> Volume ratio = volume at entry / 20-period average volume. Under 1.0x is low volume.

| Trades | WinRate | TotalPnL | AvgPnL | WinCount | VolumeRatio |
| --- | --- | --- | --- | --- | --- |
| 18 | 44.4% | $-7.12 | $-0.40 | 8 | EXCELLENT (>2.5x) |
| 18 | 27.8% | $-9.06 | $-0.50 | 5 | FAIR (1.2x-1.8x) |
| 6 | 33.3% | $-1.94 | $-0.32 | 2 | GOOD (1.8x-2.5x) |
| 23 | 52.2% | $2.50 | $0.11 | 12 | POOR (0.7x-1.2x) |
| 49 | 42.9% | $-10.96 | $-0.22 | 21 | WEAK (<0.7x) |

## 💡 Actionable Diagnostics & Recommendations
- **[WARNING] Choppy Market Bleed**: The bot has opened **47 trades** during **CHOPPY** regimes, resulting in a total PnL of **$-6.80**. Classification shows that the bot struggles in range-bound, low-efficiency markets.
- **[CRITICAL] Toxic Counter-Trend entries**: Counter-trend trades generated a total PnL of **$-16.29** across **20 trades** with a **15.0% win rate**. This is a major systemic leak. Trading against the 1h trend under the current regime detector is highly unprofitable.
- **[CRITICAL] AI Premature Exits**: AI close signals resulted in a total PnL of **$-26.27** across **29 trades** (win rate: **17.2%**). The AI frequently panics and closes positions at a minor loss, while automated Take Profits have a **94.9% win rate** generating **+$27.81**.
- **[WARNING] Low-Volume Squeeze**: Low-volume entries (POOR and WEAK volume ratio) accounted for **$-8.45** in PnL. Entering during low liquidity is highly unprofitable because the setups lack institutional flow support.
