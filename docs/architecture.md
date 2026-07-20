# Architecture

## System Overview

TradeSeeker uses a **Hybrid Maestro** architecture combining three decision layers:

```
┌─────────────────────────────────────────────────────────┐
│                    AlphaArenaDeepSeek                     │
│                    (main.py orchestrator)                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ XGBoost  │    │   LLM    │    │   Core Engine    │   │
│  │ (ML)     │    │ (DeepSeek│    │   (NumPy/ATR)    │   │
│  │          │    │ OpenRouter│   │                  │   │
│  │ Market   │    │ Strategic│    │ Risk discipline  │   │
│  │ Refleks  │    │ Reasoning│    │ Stop-loss, TP/SL │   │
│  └────┬─────┘    └────┬─────┘    └────────┬─────────┘   │
│       │               │                    │              │
│       └───────────────┼────────────────────┘              │
│                       │                                   │
│              ┌────────▼────────┐                          │
│              │ PortfolioManager │ ← Central Hub (113 edges)│
│              │  Entry/Exit     │                          │
│              │  Decisions      │                          │
│              └────────┬────────┘                          │
│                       │                                   │
│         ┌─────────────┼─────────────┐                    │
│         │             │             │                     │
│  ┌──────▼──────┐ ┌────▼────┐ ┌─────▼─────┐              │
│  │AccountService│ │MarketData│ │ Strategy  │              │
│  │ Binance API │ │Indicators│ │ Analyzer  │              │
│  └─────────────┘ └─────────┘ └───────────┘              │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### Trading Cycle

```
main.py:run_trading_cycle()
  │
  ├── PortfolioManager._prepare_execution_context()
  │   ├── MarketData.get_technical_indicators()    # 3m + 15m + HTF
  │   ├── MarketData.get_averaged_er()             # 3m+15m ER average
  │   ├── RegimeDetector.classify_coin_regime()    # BULLISH/BEARISH/NEUTRAL
  │   └── StrategyAnalyzer.detect_market_regime()  # choppy/trending
  │
  ├── AIService.get_trading_context()
  │   ├── PromptJSONBuilders.build_coin_state_vector()
  │   ├── EnhancedContextProvider.generate_enhanced_context()
  │   └── DeepSeekAPI.get_ai_decision()            # LLM call via LiteLLM
  │
  ├── PortfolioManager._process_single_decision()
  │   ├── _check_entry_preconditions()             # limits, cooldown, regime
  │   ├── _handle_entry_decision()                 # new position
  │   ├── _handle_exit_signal_logic()              # exit/close
  │   └── _execute_order_payload()                 # TP/SL calculation
  │
  └── AccountService.execute_live_close()           # Binance API call
```

### Indicator Pipeline

```
MarketData.get_technical_indicators(symbol, timeframe)
  │
  ├── CacheManager.fetch_all_indicators_with_cache()
  │   ├── calculate_rsi_series()        # RSI-7, RSI-13 (Fibonacci)
  │   ├── calculate_ema_series()        # EMA-9, EMA-21, EMA-50
  │   ├── calculate_adx()               # Average Directional Index
  │   ├── calculate_atr_series()        # Average True Range
  │   ├── calculate_obv()               # On-Balance Volume
  │   ├── calculate_vwap()              # Volume Weighted Average Price
  │   ├── calculate_bollinger_bands()   # Bollinger Bands
  │   ├── calculate_supertrend()        # SuperTrend
  │   ├── calculate_macd_series()       # MACD
  │   └── calculate_efficiency_ratio()  # Kaufman ER (10-period)
  │
  └── Returns: dict with all indicators for the timeframe
```

## God Nodes (Most Connected)

| Node | Edges | Role |
|------|-------|------|
| `PortfolioManager` | 113 | Central orchestrator for all trading decisions |
| `Config` | 61 | Universal configuration dependency |
| `AIService` | 31 | AI decision engine |
| `DataEngine` | 29 | SQLite data storage |
| `BinanceOrderExecutor` | 28 | Trade execution |
| `PerformanceMonitor` | 28 | Trend analysis |
| `RealMarketData` | 35 | Technical indicators |

## Communities (from graphify)

| Community | Module | Description |
|-----------|--------|-------------|
| Portfolio Manager Core | `core/portfolio_manager.py` | Entry/exit decisions, position management |
| AI Service (Decision Engine) | `core/ai_service.py` | LLM prompt generation, decision parsing |
| Main Orchestrator Loop | `main.py` | Trading cycle, simulation loop |
| Market Data Engine | `core/market_data.py` | Real-time data, indicator calculation |
| Technical Indicators | `core/indicators.py` | NumPy vectorized indicators |
| Config & DeepSeek API | `schemas/config.py` + `ai/deepseek_api.py` | Configuration + LLM integration |
| Connection & Account Service | `core/account_service.py` | Live trade execution |
| Binance Client & Account | `services/binance.py` | Binance Futures API |
| Regime Detector | `core/regime_detector.py` | Market regime classification |
| Performance Monitor | `core/performance_monitor.py` | Trend analysis, recommendations |
| Data Engine (SQLite) | `core/data_engine.py` | Data storage |
| Cache Manager | `core/cache_manager.py` | Smart caching |
| Alert System | `services/alert_system.py` | Real-time alerts |
| ML Service & Training | `services/ml_service.py` | XGBoost inference |
| Backtest Engine | `core/backtest.py` | Historical strategy testing |
| Flask Admin Dashboard | `web/admin_server_flask.py` | Web admin interface |

## Architecture Diagrams

- **Interactive Callflow**: [arch.html](arch.html) — Mermaid-based, zoomable
- **Tree View**: [tree.html](tree.html) — D3 collapsible hierarchy
- **Knowledge Graph**: [graphify-out/wiki/](../graphify-out/wiki/index.md) — 123 community articles
