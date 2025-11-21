# Alpha Arena AI Prompts Reference

This document provides a comprehensive reference for all prompts sent to the AI model (DeepSeek) via the API. It details the source files and functions responsible for generating each part of the prompt.

## 1. System Prompt

The System Prompt defines the AI's persona, core rules, risk management guidelines, and trading strategy. It is static text with some dynamic configuration values.

*   **Source File:** `alpha_arena_deepseek.py`
*   **Function:** `DeepSeekAPI.get_ai_decision`
*   **Location:** Lines ~77-271

### Content Structure
The System Prompt covers the following key areas:
*   **Persona & Goal:** Zero-shot systematic trading model, maximizing PnL.
*   **Core Rules:** 10x leverage, max 5 positions, same-direction limits.
*   **Risk Management:** 1:1.3 risk/reward, volatility-based stops.
*   **Strategy Guidance:** Symmetric strategy, NEUTRAL regime handling, Counter-trend definitions.
*   **Data Context:** Explanation of provided data (3m, 15m, HTF).
*   **JSON Data Format:** Description of the JSON sections in the user prompt.
*   **Advanced Analysis Playbook:** Multi-timeframe analysis steps.
*   **Trend & Counter-Trend Guidelines:** Specific rules for identifying opportunities.
    *   *Note:* Counter-trend is defined as "15m and 3m momentum align against the 1h structural trend".

*   **Action Format:** Instructions for `CHAIN_OF_THOUGHTS` and `DECISIONS` JSON output.

## 2. User Prompt (Wrapper)

The User Prompt acts as a wrapper that contains all the dynamic market data and state information. It uses a hybrid format where instructions are in plain text, but the data itself is structured as JSON blocks.

*   **Source File:** `alpha_arena_deepseek.py`
*   **Function:** `DeepSeekAPI.generate_alpha_arena_prompt_json`
*   **Location:** Lines ~5932-5994

### Content Structure
The User Prompt is constructed using f-strings and calls to `prompt_json_builders.py` functions to inject JSON blocks.

```text
USER_PROMPT:
It has been {minutes_running} minutes since you started trading...

{COUNTER_TRADE_ANALYSIS JSON}

{TREND_REVERSAL_DATA JSON}

{ENHANCED_CONTEXT JSON}

{DIRECTIONAL_BIAS JSON}

{COOLDOWN_STATUS JSON}



{POSITION_SLOTS JSON}

{MARKET_DATA JSON}

{HISTORICAL_CONTEXT JSON}

{RISK_STATUS JSON}

{PORTFOLIO JSON}
```

## 3. JSON Data Components

The following sections describe the specific JSON blocks injected into the User Prompt. All builder functions are located in `prompt_json_builders.py`.

### 3.1. Metadata
*   **Function:** `build_metadata_json`
*   **Description:** Basic run information.
*   **JSON Structure:**
    ```json
    {
        "minutes_running": 120,
        "current_time": "2023-10-27T10:00:00",
        "invocation_count": 45
    }
    ```

### 3.2. Counter-Trade Analysis
*   **Function:** `build_counter_trade_json`
*   **Description:** Pre-computed analysis of counter-trend conditions (RSI, MACD, Trend Alignment).
*   **JSON Structure:**
    ```json
    [
      {
        "coin": "XRP",
        "htf_trend": "BULLISH",
        "15m_trend": "BEARISH",
        "3m_trend": "BEARISH",
        "alignment_strength": "STRONG",
        "conditions": {
          "total_met": 3
        },
        "risk_level": "LOW_RISK",
        "volume_ratio": 2.5,
        "rsi_3m": 85.5
      }
    ]
    ```

### 3.3. Trend Reversal Data
*   **Function:** `build_trend_reversal_json`
*   **Description:** Signals indicating a potential reversal against *existing* positions.
*   **JSON Structure:**
    ```json
    [
      {
        "coin": "XRP",
        "has_position": true,
        "position_direction": "long",
        "position_duration_minutes": 45,
        "reversal_signals": {
          "htf_reversal": false,
          "15m_reversal": true,
          "3m_reversal": true,
          "strength": "STRONG"
        },
        "loss_risk_signal": "HIGH_LOSS_RISK",
        "current_trend_htf": "BULLISH",
        "current_trend_3m": "BEARISH"
      }
    ]
    ```

### 3.4. Enhanced Context
*   **Function:** `build_enhanced_context_json`
*   **Description:** Aggregated context including position summaries, market regime, and performance metrics.
*   **JSON Structure:**
    ```json
    {
      "position_context": { "total_positions": 3, ... },
      "market_regime": { "global_regime": "BULLISH", ... },
      "performance_insights": { "sharpe_ratio": 1.5, ... },
      "directional_feedback": { "long_performance": {...}, "short_performance": {...} },
      "risk_context": { "risk_utilization_pct": 30.0, ... },
      "suggestions": ["..."]
    }
    ```

### 3.5. Directional Bias
*   **Function:** `build_directional_bias_json`
*   **Description:** Snapshot of performance for Long vs Short trades (last 20 trades).
*   **JSON Structure:**
    ```json
    {
      "long": {
        "net_pnl": 10.5,
        "trades": 5,
        "win_rate": 60.0,
        "rolling_avg": 2.1,
        "consecutive_losses": 0,
        "consecutive_wins": 2,
        "caution_active": false
      },
      "short": { ... }
    }
    ```

### 3.6. Cooldown Status
*   **Function:** `build_cooldown_status_json`
*   **Description:** Status of trading cooldowns (global direction, specific coins).
*   **JSON Structure:**
    ```json
    {
      "directional_cooldowns": { "long": 0, "short": 2 },
      "coin_cooldowns": { "XRP": 1 },
      "counter_trend_cooldown": 0,
      "relaxed_countertrend_cycles": 0
    }
    ```



### 3.8. Position Slots
*   **Function:** `build_position_slot_json`
*   **Description:** Management of available trading slots, including same-direction limits.
*   **JSON Structure:**
    ```json
    {
      "total_open": 3,
      "max_positions": 5,
      "long_slots_used": 2,
      "short_slots_used": 1,
      "same_direction_limit": 4,
      "long_slots_available": 2,
      "short_slots_available": 3,
      "available_slots": 2,
      "weakest_position": { "coin": "DOGE", ... }
    }
    ```

### 3.9. Market Data
*   **Function:** `build_market_data_json`
*   **Description:** Detailed market data for each coin (3m, 15m, HTF indicators, sentiment, position).
*   **JSON Structure:**
    ```json
    [
      {
        "coin": "XRP",
        "market_regime": "BULLISH",
        "sentiment": { "open_interest": 100000, ... },
        "timeframes": {
          "3m": { "current": {...}, "series": {...} },
          "15m": { "current": {...}, "series": {...} },
          "htf": { "current": {...}, "series": {...} }
        },
        "position": { ... } // or null
      }
    ]
    ```

### 3.10. Historical Context
*   **Function:** `build_historical_context_json`
*   **Description:** Summary of recent market behavior and decisions.
*   **JSON Structure:**
    ```json
    {
      "total_cycles_analyzed": 50,
      "market_behavior": "Trending Up",
      "recent_decisions": [...],
      "performance_trend": "Improving"
    }
    ```

### 3.11. Risk Status
*   **Function:** `build_risk_status_json`
*   **Description:** Current risk metrics and trading limits.
*   **JSON Structure:**
    ```json
    {
      "current_positions_count": 3,
      "total_margin_used": 45.0,
      "available_cash": 155.0,
      "trading_limits": {
        "min_position": 10.0,
        "max_positions": 5,
        "available_cash_protection": 15.5,
        "position_sizing_pct": 40.0
      }
    }
    ```

### 3.12. Portfolio
*   **Function:** `build_portfolio_json`
*   **Description:** Overall portfolio performance and position summary.
*   **JSON Structure:**
    ```json
    {
      "total_return_pct": 5.2,
      "available_cash": 155.0,
      "account_value": 210.4,
      "sharpe_ratio": 1.8,
      "positions": [ ... ]
    }
    ```

## 4. Dynamic Text Templates

Some parts of the JSON data contain dynamically generated text strings that act as "mini-prompts" or direct suggestions to the AI.

### 4.1. Enhanced Context Suggestions
*   **Source File:** `enhanced_context_provider.py`
*   **Function:** `generate_suggestions`
*   **Location:** Lines ~331-346
*   **Templates:**
    *   `"[INFO] Bearish regime detected with ≥3 open positions"`
    *   `"[INFO] Bullish regime detected with zero current exposure"`

### 4.2. Performance Recommendations
*   **Source File:** `performance_monitor.py`
*   **Function:** `_generate_recommendations`
*   **Location:** Lines ~251-285
*   **Templates:**
    *   `"[INFO] Cash balance ${current} vs ${initial} initial; liquidity below 50% of baseline"`
    *   `"[INFO] Position count {count}; exceeds reference threshold (≥3)"`
    *   `"[INFO] Recorded return {return}% within analysis window; below growth target"`
    *   `"[INFO] Coins with cumulative negative PnL: {coins}"`
    *   `"[INFO] Performance metrics stable; no notable anomalies detected"`

### 4.3. Trend Reversal Recommendations
*   **Source File:** `performance_monitor.py`
*   **Function:** `_generate_reversal_recommendations`
*   **Location:** Lines ~653-686
*   **Templates:**
    *   `"[INFO] Multiple strong reversal signals detected across assets"`
    *   `"[INFO] Several strong reversal signals present across coverage set"`
    *   `"[INFO] Reversal signal percentage above {pct}%; elevated probability of directional shifts"`
    *   `"[INFO] Strong reversal readings detected in: {coins}"`
    *   `"[INFO] Reversal signals flagged; protective level proximity worth monitoring"`
    *   `"[INFO] No significant reversal signals detected; prevailing trends classified as intact"`

## 5. Source File Summary

The following files were scanned for prompt content:

| File | Role | Prompt Content |
| :--- | :--- | :--- |
| `alpha_arena_deepseek.py` | **Primary** | Contains the main System Prompt and User Prompt wrapper logic. |
| `prompt_json_builders.py` | **Primary** | Constructs all JSON data blocks injected into the User Prompt. |
| `enhanced_context_provider.py` | **Secondary** | Generates text suggestions for `ENHANCED_CONTEXT` JSON. |
| `performance_monitor.py` | **Secondary** | Generates text recommendations for `ENHANCED_CONTEXT` and `TREND_REVERSAL_DATA`. |
| `prompt_json_schemas.py` | **Structure** | Defines the JSON schema structure (documentation only). |
| `config.py` | **Config** | No direct prompt text. Configuration values affect logic. |
| `binance.py` | **API** | No prompt text. Purely API interaction. |
| `backtest.py` | **Simulation** | No prompt text. Simulation engine. |
| `utils.py` | **Utility** | No prompt text. Helper functions. |
| `prompt_json_utils.py` | **Utility** | No prompt text. JSON serialization helpers. |

## 6. Redundancy & Reinforcement Analysis

This section identifies areas where prompt instructions or data are repeated across different sections. These repetitions are intentional "reinforcements" designed to ensure the AI adheres to critical constraints.

| Topic | System Prompt | User Prompt Wrapper | JSON Data | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Data Ordering** | "All numerical sequences are ordered OLDEST → NEWEST" | "ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST" | N/A | **Critical:** Ensures AI interprets time-series data correctly. |
| **Cooldown Rules** | Implied in Risk Management | "⚠️ IMPORTANT: If a direction... is in cooldown, you MUST NOT propose..." | `COOLDOWN_STATUS` contains values | **Reinforcement:** Prevents AI from ignoring cooldowns by placing the warning directly next to the data. |
| **Counter-Trend** | Defines logic: "15m+3m align against 1h" | Mentions "We pre-compute the standard 5 counter-trend conditions" | `COUNTER_TRADE_ANALYSIS` contains results | **Context:** System Prompt gives the *theory*, User Prompt gives the *calculated result*. |
| **Trend Reversal** | Defines logic: "Reversal signals ONLY apply to EXISTING positions" | "All notes below are informational... evaluate them independently" | `TREND_REVERSAL_DATA` contains results | **Safety:** Prevents AI from overreacting to reversal signals by adding a disclaimer next to the data. |
