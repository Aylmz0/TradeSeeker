# TradeSeeker Documentation

**TradeSeeker** is a high-performance algorithmic trading bot that combines LLM reasoning (DeepSeek/OpenRouter) with XGBoost machine learning and a vectorized indicator engine.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env    # configure API keys
python3 src/main.py     # start trading loop
```

## Documentation Map

### Core Docs (Start Here)
| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design, module map, data flow |
| [Configuration](configuration.md) | Every Config / .env option explained (120+ settings) |
| [Operations](operations.md) | Runbooks: 35-cycle reset, live mode, troubleshooting |
| [Development](development.md) | Contributing guide, how to document new functions, how to update graphify |

### Auto-Generated Docs
| Document | Description |
|----------|-------------|
| [API Reference](api/index.html) | Auto-generated from docstrings (pdoc) — open in browser |
| [Callflow Diagram](arch.html) | Interactive Mermaid-based architecture visualization — open in browser |
| [Tree View](tree.html) | D3 collapsible module hierarchy — open in browser |
| [Knowledge Graph Wiki](../graphify-out/wiki/index.md) | 123 community articles from graphify — AI-crawlable |
| [Lessons Learned](LESSONS.md) | Auto-generated from graphify Q&A memory |

### Reference
| Document | Description |
|----------|-------------|
| [AI Prompts Reference](AI_PROMPTS_REFERENCE.md) | AI prompt system documentation |
| [Architecture Guide](guide.md) | Turkish technical guide |

## Core Modules

```
main.py                    # Orchestrator (AlphaArenaDeepSeek)
├── core/
│   ├── portfolio_manager.py    # Central hub (113 edges) — entry/exit decisions
│   ├── strategy_analyzer.py    # Market regime, volume quality
│   ├── market_data.py          # Technical indicators, real-time data
│   ├── account_service.py      # Live trade execution, TP/SL
│   ├── ai_service.py           # AI decision engine, prompt generation
│   ├── regime_detector.py      # Bullish/Bearish/Neutral classification
│   ├── performance_monitor.py  # Trend analysis, recommendations
│   ├── indicators.py           # NumPy vectorized indicators (RSI, EMA, ATR, etc.)
│   ├── cache_manager.py        # Smart caching for API responses
│   └── data_engine.py          # SQLite data storage
├── ai/
│   ├── deepseek_api.py         # LiteLLM Router integration
│   ├── enhanced_context_provider.py  # Market context for AI
│   └── prompt_json_builders.py # State vector, portfolio, risk JSON builders
├── schemas/
│   ├── config.py               # All settings (Pydantic BaseSettings)
│   ├── ai.py                   # AIDecision, ExecutionReport schemas
│   ├── position.py             # Position, ExitPlan schemas
│   └── trade.py                # TradeHistoryEntry schema
├── services/
│   ├── binance.py              # Binance Futures API client
│   ├── ml_service.py           # XGBoost inference singleton
│   └── alert_system.py         # Real-time alerts
└── web/
    └── admin_server_flask.py   # Flask admin dashboard
```

## Key Concepts

- **Hybrid Decision Flow**: XGBoost (ML consensus) + LLM (strategic reasoning) + Core Engine (risk discipline)
- **35-Cycle Reset**: Every 35 cycles backs up to `data/backups/`, clears trade history
- **ER Averaging**: 3m + 15m efficiency ratio averaged for choppy market detection
- **RSI Fibonacci Periods**: RSI-7 and RSI-13 (not 14) — intentional Fibonacci-based choices
- **Graceful Degradation**: If ML data is missing, bot runs AI-only mode

## For New Contributors

1. Read [Development Guide](development.md) — how to contribute, code style, docstring conventions
2. Read [Architecture](architecture.md) — system design and data flow
3. Read [Configuration](configuration.md) — all settings and their defaults
4. Explore [Knowledge Graph Wiki](../graphify-out/wiki/index.md) — 123 community articles for deep dives

## Keeping Docs Updated

After making code changes:
```bash
# Update knowledge graph
graphify update ./src

# Regenerate API docs
.venv/bin/python -m pdoc src -o docs/api -d google --no-include-undocumented --no-show-source

# Update callflow (if architecture changed)
graphify export callflow-html --output docs/arch.html --max-sections 12

# Update lessons
graphify reflect --out docs/LESSONS.md
```

See [Development Guide](development.md) for detailed instructions.
