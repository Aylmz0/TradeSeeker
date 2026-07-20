# Configuration

All settings are loaded from `.env` via Pydantic BaseSettings (`src/schemas/config.py`).

## AI Provider Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIMARY_AI_PROVIDER` | `openrouter` | AI backend: `openrouter`, `deepseek`, `groq`, `mimo`, `zai` |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `OPENROUTER_MODEL` | `openrouter/free` | Primary model |
| `OPENROUTER_FALLBACK_MODEL` | `google/gemini-2.0-flash-exp:free` | Fallback model |
| `OPENROUTER_REASONING_ENABLED` | `true` | Enable reasoning mode |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `groq/compound` | Groq model |
| `MIMO_API_KEY` | — | Mimo API key |
| `MIMO_MODEL` | `mimo-v2-flash` | Mimo model |
| `ZAI_API_KEY` | — | ZAI API key |
| `ZAI_MODEL` | `glm-4.5-flash` | ZAI model |

## Binance

| Variable | Default | Description |
|----------|---------|-------------|
| `BINANCE_API_KEY` | — | Binance API key |
| `BINANCE_SECRET_KEY` | — | Binance secret key |
| `BINANCE_TESTNET` | `false` | Use testnet |
| `BINANCE_MARGIN_TYPE` | `ISOLATED` | `ISOLATED` or `CROSSED` |
| `BINANCE_DEFAULT_LEVERAGE` | `10` | Default leverage |
| `BINANCE_RECV_WINDOW` | `5000` | Receive window (ms) |

## Application

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Debug mode |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_DIR` | `data/logs` | Log directory |
| `MAX_RETRY_ATTEMPTS` | `3` | Max retries for API calls |
| `REQUEST_TIMEOUT` | `30` | HTTP timeout (seconds) |

## Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `simulation` | `simulation` or `live` |
| `INITIAL_BALANCE` | `200.0` | Starting balance (USD) |
| `CYCLE_INTERVAL_MINUTES` | `2` | Minutes between cycles |
| `HISTORY_RESET_INTERVAL` | `35` | Cycles before history reset |
| `MAX_CYCLES` | `0` | Max cycles (0 = unlimited) |

## Risk Management

| Variable | Default | Description |
|----------|---------|-------------|
| `RISK_LEVEL` | `medium` | `low`, `medium`, `high` |
| `MAX_LEVERAGE` | `20` | Maximum leverage |
| `MIN_CONFIDENCE` | `0.60` | Minimum AI confidence to enter |
| `CHOPPY_ER_THRESHOLD` | `0.30` | Efficiency ratio threshold for choppy detection |
| `ML_CONFIDENCE_THRESHOLD` | `40.0` | ML consensus threshold |
| `MAX_POSITIONS` | `5` | Maximum concurrent positions |
| `RISK_PER_TRADE_USD` | `3.0` | Risk per trade (USD) |

## Graduated Stop Loss

| Variable | Default | Description |
|----------|---------|-------------|
| `LOSS_MULT_L1` | `0.20` | Level 1 loss multiplier |
| `LOSS_MULT_L2` | `0.15` | Level 2 loss multiplier |
| `LOSS_MULT_L3` | `0.12` | Level 3 loss multiplier |
| `LOSS_MULT_L4` | `0.10` | Level 4 loss multiplier |
| `LOSS_MULT_BASE` | `0.08` | Base loss multiplier |

## Smart Cooldown

| Variable | Default | Description |
|----------|---------|-------------|
| `SMART_COOLDOWN_LOSS` | `3` | Cooldown cycles after loss |
| `SMART_COOLDOWN_WIN` | `2` | Cooldown cycles after win |
| `MAX_NEW_POSITIONS_PER_CYCLE` | `3` | Max new entries per cycle |
| `EXTENDED_LOSS_CYCLES` | `15` | Extended cooldown after losses |
| `EXTENDED_PROFIT_CYCLES` | `15` | Extended cooldown after profits |

## Position Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `SAME_DIRECTION_LIMIT` | `2` | Max positions in same direction |
| `DYNAMIC_DIRECTION_LIMIT` | `2` | Dynamic direction limit |

## Market Analysis

| Variable | Default | Description |
|----------|---------|-------------|
| `GLOBAL_NEUTRAL_STRENGTH_THRESHOLD` | `0.4` | Neutral regime strength threshold |
| `RSI_OVERBOUGHT_THRESHOLD` | `70.0` | RSI overbought level |
| `RSI_OVERSOLD_THRESHOLD` | `30.0` | RSI oversold level |
| `EMA_NEUTRAL_BAND_PCT` | `0.001` | EMA neutral band percentage |
| `INTRADAY_NEUTRAL_RSI_LOW` | `45.0` | Intraday neutral RSI low |
| `INTRADAY_NEUTRAL_RSI_HIGH` | `55.0` | Intraday neutral RSI high |
| `TREND_LONG_RSI_THRESHOLD` | `50.0` | Trend long RSI threshold |
| `TREND_SHORT_RSI_THRESHOLD` | `50.0` | Trend short RSI threshold |

## Enhanced Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `SHORT_ENHANCEMENT_MULTIPLIER` | `1.15` | Short position enhancement |
| `VOLUME_MINIMUM_THRESHOLD` | `0.30` | Minimum volume ratio (hard block) |
| `SIMULATION_COMMISSION_RATE` | `0.0005` | Simulation commission rate |
| `DIRECTIONAL_BULLISH_LONG_MULTIPLIER` | `1.00` | Bullish regime long multiplier |
| `DIRECTIONAL_BULLISH_SHORT_MULTIPLIER` | `0.90` | Bullish regime short multiplier |
| `DIRECTIONAL_BEARISH_LONG_MULTIPLIER` | `0.90` | Bearish regime long multiplier |
| `DIRECTIONAL_BEARISH_SHORT_MULTIPLIER` | `1.00` | Bearish regime short multiplier |
| `DIRECTIONAL_NEUTRAL_MULTIPLIER` | `0.95` | Neutral regime multiplier |

## Choppy Regime

| Variable | Default | Description |
|----------|---------|-------------|
| `CHOPPY_COIN_RATIO_MIN` | `0.5` | Min choppy coin ratio for global choppy |
| `CHOPPY_LEVERAGE` | `5` | Leverage in choppy regime |
| `CHOPPY_TP_LONG_MULTIPLIER` | `1.004` | Choppy TP long multiplier |
| `CHOPPY_TP_SHORT_MULTIPLIER` | `0.996` | Choppy TP short multiplier |
| `CHOPPY_SL_LONG_MULTIPLIER` | `0.994` | Choppy SL long multiplier |
| `CHOPPY_SL_SHORT_MULTIPLIER` | `1.006` | Choppy SL short multiplier |
| `CHOPPY_TP_SL_MULTIPLIER` | `0.004` | Choppy TP/SL multiplier |
| `CHOPPY_HIGH_ER_EXCEPTION` | `0.45` | High ER exception threshold |

## ATR

| Variable | Default | Description |
|----------|---------|-------------|
| `ATR_TP_MULTIPLIER` | `2.0` | ATR take-profit multiplier |
| `ATR_SL_MULTIPLIER` | `1.8` | ATR stop-loss multiplier |

## Flash Exit

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_EXIT_ENABLED` | `true` | Enable flash exit |
| `FLASH_EXIT_RSI_DELTA_MIN` | `15.0` | Min RSI delta for flash exit |
| `FLASH_EXIT_VOLUME_SURGE_MIN` | `3.0` | Min volume surge for flash exit |
| `FLASH_EXIT_LOSS_TRIGGER_MULTIPLIER` | `1.002` | Loss trigger multiplier |

## Exit Plan

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_STOP_LOSS_PCT` | `0.01` | Default stop-loss percentage |
| `DEFAULT_PROFIT_TARGET_PCT` | `0.015` | Default profit-target percentage |
| `MIN_EXIT_PLAN_OFFSET` | `0.0001` | Minimum exit plan offset |

## Trailing Stop

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAILING_PROGRESS_TRIGGER` | `40.0` | Progress trigger percentage |
| `TRAILING_TIME_PROGRESS_FLOOR` | `30.0` | Time progress floor |
| `TRAILING_TIME_MINUTES` | `20` | Trailing stop time (minutes) |
| `TRAILING_ATR_MULTIPLIER` | `1.2` | ATR multiplier for trailing |
| `TRAILING_FALLBACK_BUFFER_PCT` | `0.004` | Fallback buffer percentage |
| `TRAILING_VOLUME_ABSOLUTE_THRESHOLD` | `0.2` | Volume absolute threshold |
| `TRAILING_VOLUME_DROP_RATIO` | `0.5` | Volume drop ratio |
| `TRAILING_MIN_IMPROVEMENT_PCT` | `0.0005` | Minimum improvement percentage |
| `TRAILING_PROGRESS_TRIGGER_EXTREME` | `20.0` | Extreme progress trigger |

## Higher Timeframe

| Variable | Default | Description |
|----------|---------|-------------|
| `HTF_INTERVAL` | `1h` | Higher timeframe: `30m`, `1h`, `2h`, `4h` |

## Smart Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SMART_CACHE` | `true` | Enable smart cache |
| `SMART_CACHE_SAFETY_MARGIN` | `0.85` | Cache safety margin |
| `SMART_CACHE_STATS_LOGGING` | `true` | Log cache statistics |

## Performance Monitor

| Variable | Default | Description |
|----------|---------|-------------|
| `PERFORMANCE_PROFITABILITY_HIGH` | `50.0` | High profitability threshold |
| `PERFORMANCE_PROFITABILITY_LOW` | `40.0` | Low profitability threshold |
| `PERFORMANCE_PROFIT_FACTOR_LOW` | `1.2` | Low profit factor |
| `PERFORMANCE_PROFIT_FACTOR_HIGH` | `1.5` | High profit factor |
| `PERFORMANCE_DECISION_RATE_HIGH` | `60.0` | High decision rate |
| `PERFORMANCE_DECISION_RATE_LOW` | `30.0` | Low decision rate |
| `PERFORMANCE_RETURN_HIGH` | `5.0` | High return threshold |
| `PERFORMANCE_RETURN_LOW` | `0.0` | Low return threshold |
| `PERFORMANCE_DRAWDOWN_THRESHOLD` | `-10.0` | Drawdown threshold |
| `PERFORMANCE_SHARPE_HIGH` | `1.0` | High Sharpe ratio |
| `PERFORMANCE_SHARPE_LOW` | `0.0` | Low Sharpe ratio |
| `PERFORMANCE_PROFIT_FACTOR_CRITICAL` | `0.8` | Critical profit factor |

## Erosion Rate

| Variable | Default | Description |
|----------|---------|-------------|
| `EROSION_RATE_EXTREME` | `0.04` | Extreme erosion rate |
| `EROSION_RATE_NORMAL` | `0.06` | Normal erosion rate |
| `EROSION_MIN_PROFIT_USD` | `1.00` | Minimum profit for erosion |

## JSON Prompt

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_JSON_PROMPT` | `false` | Use JSON-formatted prompts |
| `JSON_PROMPT_COMPACT` | `false` | Compact JSON prompts |
| `VALIDATE_JSON_PROMPTS` | `false` | Validate JSON prompts |
| `JSON_PROMPT_VERSION` | `1.0` | JSON prompt version |
| `JSON_SERIES_MAX_LENGTH` | `30` | Max series length in JSON |

## Minimum Position

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_POSITION_MARGIN_USD` | `10.0` | Minimum position margin |
| `MIN_POSITION_CLEANUP_THRESHOLD` | `5.0` | Cleanup threshold |

## Partial Profit

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_PARTIAL_PROFIT_MARGIN_REMAINING_USD` | `10.0` | Min margin remaining for partial |
| `MAXIMUM_LIMIT_BALANCE_PCT` | `0.08` | Max limit balance percentage |

## Win Streak

| Variable | Default | Description |
|----------|---------|-------------|
| `WIN_STREAK_COOLDOWN_THRESHOLD` | `2` | Win streak cooldown threshold |
| `WIN_STREAK_COOLDOWN_CYCLES` | `1` | Cooldown cycles |

## Tactical Scout

| Variable | Default | Description |
|----------|---------|-------------|
| `SCOUT_MODE_ENABLED` | `false` | Enable scout mode |
| `MAX_ALIGNMENT_DELTA_S` | `5` | Max alignment delta |
| `WEIGHT_RECALL_TARGET` | `0.60` | Weight recall target |
| `WEIGHT_PRECISION_LIMIT` | `0.30` | Weight precision limit |
| `SHAP_STABILITY_THRESHOLD` | `0.85` | SHAP stability threshold |
| `ADX_TREND_LEVEL` | `25` | ADX trend level |
| `VOLATILITY_LIMIT_PCT` | `0.02` | Volatility limit |
| `COMMISSION_GUARD_RATIO` | `5.0` | Commission guard ratio |
| `SCOUT_LEVERAGE_MULT` | `0.5` | Scout leverage multiplier |
| `CANARY_RATIO` | `0.10` | Canary ratio |
| `CANARY_FAIL_DELTA` | `0.05` | Canary fail delta |
| `REPLAY_SEED` | `42` | Replay seed |
| `REPLAY_CHECKPOINT_CYCLES` | `50` | Replay checkpoint cycles |
| `HOLD_THRESHOLD_DEFAULT` | `0.82` | Default hold threshold |
| `HOLD_THRESHOLD_NEUTRAL` | `0.75` | Neutral hold threshold |
