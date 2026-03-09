# Technical Audit: Foundation for Tactical Scout (v1.2)

This document records the baseline state of the system's concurrency and logic before the Phase 1-6 modifications.

## 1. Threading Model & Concurrency
- **Observation**: `PortfolioManager` uses a single `threading.Lock()` initialized in `__init__`.
- **Analysis**: The main bot loop in `main.py` is sequential per cycle. Parallelism is currently used mainly in `MarketData` for I/O bound kline fetching.
- **Safety**: Position modifications in `portfolio_manager.py` are properly wrapped in `with self._lock:`.
- **Conclusion**: The current `threading.Lock` is sufficient for the tactical scout transition as long as new components (Sync Guard) remain within the existing cycle context.

## 2. Regime Detection Baseline
- **Existing Logic**: `StrategyAnalyzer.detect_market_regime` uses:
    - `EMA20` delta for HTF trend.
    - `EMA20` alignment for shorter timeframes (3m/15m).
- **Consolidation**: The new `RegimeDetector` class (Week 0.2) will formalize the **ADX(14) > 25** rule to separate "Trending" from "Chop/Neutral" regimes more robustly than EMA deltas alone.

## 3. Data Integrity (The 816 Cycle Mark)
- **Baseline**: `data/market_data.db` is stable at 11,500+ intervals across 6 coins.
- **Verification**: SQLite schema confirmed (timestamp, coin, action, etc.). The 0.00% BUY recall is an artifact of the "Sniper" prompt logic and class imbalance in data/market_data.db (mostly WAIT/HOLD).
- **Target**: Move to `scale_pos_weight` in Week 2.
