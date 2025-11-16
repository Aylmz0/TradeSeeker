# alpha_arena_deepseek.py
import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
import traceback # For detailed error logging
import threading
from collections import deque
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import new utility modules
from config import Config
from utils import (
    format_num, safe_file_write, safe_file_read, 
    rate_limiter, RetryManager, DataValidator, performance_monitor
)
from backtest import AdvancedRiskManager

try:
    from binance import BinanceOrderExecutor, BinanceAPIError
    BINANCE_IMPORT_ERROR = None
except Exception as import_error:
    BinanceOrderExecutor = None

    class BinanceAPIError(Exception):
        """Fallback Binance error when executor import fails."""

        pass

    BINANCE_IMPORT_ERROR = import_error

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Validate configuration
if not Config.validate_config():
    print("❌ Configuration validation failed. Please check your .env file.")
    exit(1)

Config.log_config_summary()

HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'  # Higher timeframe interval used for structural trend analysis
HTF_LABEL = HTF_INTERVAL

# --- API Class ---
class DeepSeekAPI:
    """DeepSeek API integration with enhanced error handling and rate limiting"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        self.session = RetryManager.create_session_with_retry()

        if not self.api_key:
            print("❌ DEEPSEEK_API_KEY not found!")
            print("ℹ️  Please check your .env file configuration.")

    def get_ai_decision(self, prompt: str) -> str:
        """Get trading decision from DeepSeek API"""
        if not self.api_key:
            return self._get_simulation_response(prompt)

        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"""You are a zero-shot systematic trading model participating in Alpha Arena.
Your goal is to maximize PnL (profit and loss) by trading perpetual futures on 6 assets: XRP, DOGE, ASTER, ADA, LINK, SOL.

You are given $200 starting capital and must process numerical market data to discover alpha.
Your Sharpe ratio is provided to help normalize for risky behavior.

CORE RULES:
- Make every decision using the numerical data provided; ignore external narratives.
- Always provide complete exit plans (profit_target, stop_loss, invalidation_condition).
- All entries must use fixed 10x leverage. Submit 10x on every trade; risk sizing is handled via margin rules.
- Minimum confidence is 0.4; use higher confidence for stronger quantitative edges.
- Maximum simultaneous positions across assets: 5.

RISK MANAGEMENT:
- Portfolio- and position-level risk caps are enforced automatically; focus on selecting high-quality opportunities.
- Maintain at least 1:1.3 risk/reward.
- Use objective volatility references (e.g., {HTF_LABEL} ATR) when setting stops.
- Express invalidation clearly (e.g., "If {HTF_LABEL} close is below EMA20").

SYMMETRIC STRATEGY GUIDANCE:
- Evaluate both LONG and SHORT paths for every asset. Bullish regimes support longs; bearish regimes support shorts.
- NEUTRAL regime: When a coin's market regime is NEUTRAL (1h trend doesn't align with 3m/15m), you can still take LONG or SHORT positions. NEUTRAL indicates counter-trend opportunities or mixed signals - evaluate technical conditions and take the direction with the best quantified edge. Both LONG and SHORT trades are valid in NEUTRAL conditions.
- Counter-trend trades are a valid and valuable strategy; use them when the pre-computed checklist (in the prompt) shows >2/5 conditions and you can rationalize the edge. Counter-trend opportunities often provide excellent risk/reward ratios.
- Counter-trend trades require moderate confidence (>0.65). If you see a strong counter-trend setup, don't hesitate to take it even if confidence is above 0.60.
- Only label a setup as counter-trend when your proposed trade direction is opposite the {HTF_LABEL} trend. If {HTF_LABEL} trend and trade direction align but 3m is temporarily opposing, treat it as trend-following.
- Prioritize trades with quantified momentum, participation, and risk/reward advantages.
- When regime, momentum, and participation align in your favor, favor committing capital decisively instead of waiting for perfect confirmation.
- Execute trend-following setups promptly when {HTF_LABEL} + 3m structures point the same way and volume/liquidity is supportive.
- IMPORTANT: When {HTF_LABEL} + 15m + 3m all align in the same direction, be cautious - this alignment alone does not guarantee a strong trend-following opportunity; evaluate trend duration, statistics, and market context before entering.

DATA CONTEXT:
- You receive 3m (entry/exit) and {HTF_LABEL} (trend) series plus historical indicators.
- All numerical sequences are ordered OLDEST → NEWEST; interpret momentum through time.
- Volume, Open Interest, and Funding Rate are provided for sentiment context—combine them with price action.
- Treat the supplied data as the authoritative source for every decision.
- When reporting comparisons (e.g., price vs EMA), write them explicitly as `price=2.2854 > EMA20=2.2835` or `price=... < EMA20=...` so the direction is unambiguous.
- Reference both global regime counts (bullish/bearish/neutral) and coin-specific regimes when summarizing market context to avoid contradictory statements.

JSON DATA FORMAT (Format Version 1.0):
- Market data is provided in JSON format for easier parsing and structured analysis.
- JSON sections are embedded within the prompt with clear labels (e.g., "COUNTER_TRADE_ANALYSIS (JSON):").
- Each JSON section contains structured data:
  * COUNTER_TRADE_ANALYSIS: Array of counter-trade conditions per coin with risk levels
  * TREND_REVERSAL_DATA: Array of reversal signals per coin with strength indicators
  * ENHANCED_CONTEXT: Position, market regime, performance, and risk context
  * DIRECTIONAL_BIAS: Directional performance snapshot (Last 20 trades) with long/short metrics
  * COOLDOWN_STATUS: Directional and coin-specific cooldown information
  * TREND_FLIP_GUARD: Recent trend flip guard with cooldown and history information
  * POSITION_SLOTS: Current position slot usage and availability
  * MARKET_DATA: Per-coin market data with timeframes (3m, 15m, HTF) and indicators
  * HISTORICAL_CONTEXT: Recent trading decisions and market behavior
  * RISK_STATUS: Current risk metrics and trading limits
  * PORTFOLIO: Account value, returns, and position details
- All numerical values in JSON are raw numbers (not formatted strings) for direct use in calculations.
- Series data may be compressed if longer than 50 values (first 5 and last 5 values kept, with summary stats).
- Parse JSON sections using standard JSON parsing; all data types are clearly defined (numbers, strings, booleans, arrays, objects).
- If a JSON section is missing or invalid, use the plain text context provided alongside it.

ADVANCED ANALYSIS PLAYBOOK:
- Apply long and short strategies across all coins; choose the direction that offers the superior quantified edge.
- Use {HTF_LABEL} timeframe for structural bias, 15m for medium-term momentum confirmation, and 3m for execution timing.
- Monitor volume vs. average volume, Open Interest, and Funding to measure conviction.
- Employ multi-timeframe technical analysis (EMA, RSI, MACD, ATR, etc.).
- Keep take-profit/stop-loss targets responsive (e.g. 2–4% TP, 1–2% SL) when volatility supports it.
- Manage exits proactively; do not wait for targets if data invalidates the thesis.
- High-confidence setups (0.7–0.8+) justify higher exposure within risk limits.
- Consider both trend-following and counter-trend opportunities equally; choose the setup with the best quantified edge regardless of trend direction. Counter-trend trades can be highly profitable when conditions align.
- Remember:BE AGGRESSIVE but disciplined - Take calculated risks based on technical analysis.

MULTI-TIMEFRAME PROCESS:
1. Check global and per-asset regime data (provided in the prompt).
2. Analyze {HTF_LABEL} (1h) indicators for structural trend and directional bias.
3. Analyze 15m indicators for medium-term momentum confirmation.
4. Use 3m indicators for entry/exit timing and short-term confirmation.
5. Use alignment across all three timeframes (1h + 15m + 3m) for strongest signals.
6. Incorporate volume, Open Interest, Funding, and other metrics to judge conviction.
7. In your analysis, always mention all three timeframes (1h, 15m, 3m) for each coin you evaluate.
8. Decide whether to go long, short, hold, or close based on the strongest quantified edge across all timeframes.

STARTUP BEHAVIOR:
- During the first 2-3 cycles, observe unless an exceptional, well-supported setup appears.
- Avoid impulsive entries immediately after reset.
- Maintain up to 5 concurrent positions; choose quality over quantity.

DATA NOTES:
- Series are ordered oldest → newest; interpret trends accordingly.
- Open Interest and Funding context is informational; combine with price action.
- Volume statistics highlight participation strength.

TREND & COUNTER-TREND GUIDELINES:
- When price is below {HTF_LABEL} EMA20 with bearish momentum, short setups merit priority.
- When price is above {HTF_LABEL} EMA20 with bullish momentum, long setups merit priority.
- Counter-trend trades (long or short) are encouraged when technical conditions support them. Look for oversold/overbought conditions, divergences, or reversal patterns that suggest a counter-trend move. Confidence above 0.65 is sufficient for counter-trend entries.
- Counter-trend opportunities arise when both 3m and 15m momentum align against the 1h structural trend. 
  For example: 1h bullish, but 15m bearish AND 3m bearish = counter-trend short opportunity.
- If volume ratio is ≤0.20× average, call out the weakness, reduce confidence materially, and consider skipping the trade unless another data point overwhelmingly compensates.

TREND REVERSAL DETECTION:
- Monitor positions that have been open for extended periods (1+ hours). Extended positions may need review, but don't automatically assume reversal is imminent.
- Multi-timeframe analysis: {HTF_LABEL} (1h) provides structural trend, 15m provides medium-term momentum confirmation, and 3m provides short-term entry/exit timing.
- 15m momentum provides important confirmation between {HTF_LABEL} trend and 3m momentum. When 15m aligns with {HTF_LABEL}, it strengthens the trend signal. When 15m aligns with 3m but opposes {HTF_LABEL}, it suggests potential reversal.
- 3m momentum provides supplementary context alongside {HTF_LABEL} trend analysis. Use it as one data point among many, not as a primary decision driver. Short-term 3m momentum changes are normal market noise.
- IMPORTANT: Reversal means momentum is moving AGAINST the position direction. This is an EXIT WARNING, not an entry signal. For example:
  * LONG position: Reversal = bearish momentum (price < EMA20, RSI < 50, MACD < 0) → Consider closing LONG
  * SHORT position: Reversal = bullish momentum (price > EMA20, RSI > 50, MACD > 0) → Consider closing SHORT
- Reversal signal strength (based on how many timeframes show reversal AGAINST position):
  * "STRONG": 15m + 3m BOTH show reversal against position (but {HTF_LABEL} doesn't) - strong reversal signal. This can be a COUNTER-TREND OPPORTUNITY: if {HTF_LABEL} trend is still in position's favor but 15m+3m reversed, consider opening a counter-trend position in the reversal direction (e.g., LONG position with 15m+3m bearish reversal = consider SHORT counter-trend entry, or close LONG and evaluate)
  * "MEDIUM": Only 3m shows reversal against position - medium reversal signal, continue monitoring
  * "INFORMATIONAL": Only 15m shows reversal against position - informational context, continue monitoring, prioritize {HTF_LABEL} trend
  * NOTE: If {HTF_LABEL} + 15m + 3m ALL show the same direction, this is NOT a reversal - this is the trend itself continuing. Reversal only occurs when shorter timeframes (15m+3m) oppose the position while {HTF_LABEL} may still be in favor.
- When you see reversal signals, evaluate the full context including position duration, profit/loss status, and original thesis before making any changes. Prioritize {HTF_LABEL} trend confirmation and your systematic exit plan.
- NOTE: STRONG reversal (15m+3m against position, but {HTF_LABEL} still in favor) can indicate counter-trend opportunities. Evaluate if the reversal momentum is strong enough for a counter-trend entry, but remember counter-trend trades require higher confidence (≥0.65) and proper technical conditions.

ACTION FORMAT:
- Use signals: `buy_to_enter`, `sell_to_enter`, `hold`, `close_position`.
- If a position is already open on a coin, only `hold` or `close_position` are valid.
- Provide both `CHAIN_OF_THOUGHTS` (analysis) and `DECISIONS` (JSON).
- In your CHAIN_OF_THOUGHTS, for each coin you analyze, explicitly mention: (1) {HTF_LABEL} (1h) trend assessment, (2) 15m momentum assessment, and (3) 3m momentum assessment.

Example Format (NOF1AI Advanced Style):
CHAIN_OF_THOUGHTS
[Advanced systematic analysis of all assets using {HTF_LABEL} (1h) trends, 15m momentum confirmation, and 3m entry timing. For each coin, analyze all three timeframes. Focus on market structure, volume confirmation, and risk management. Example: "XRP: 1h bullish (price > EMA20, RSI 62.5), 15m bullish momentum (price > EMA20, RSI 58), 3m bullish (price > EMA20, RSI 60). All three timeframes aligned bullish with volume confirmation. Open Interest increasing suggests institutional interest. Targeting $0.56 with stop below $0.48. Invalidation if {HTF_LABEL} price closes below EMA20."]
DECISIONS
{{
  "XRP": {{
    "signal": "buy_to_enter",
    "leverage": 10,
    "confidence": 0.75,
    "profit_target": 0.56,
    "stop_loss": 0.48,
    "risk_usd": 45.0,
    "invalidation_condition": "If {HTF_LABEL} price closes below {HTF_LABEL} EMA20"
  }},
  "SOL": {{
    "signal": "sell_to_enter",
    "leverage": 10,
    "confidence": 0.75,
    "profit_target": 185.0,
    "stop_loss": 198.0,
    "risk_usd": 45.0,
    "invalidation_condition": "If {HTF_LABEL} price closes above {HTF_LABEL} EMA20"
  }},
  "ADA": {{
    "signal": "buy_to_enter",
    "leverage": 10,
    "confidence": 0.75,
    "profit_target": 0.52,
    "stop_loss": 0.48,
    "risk_usd": 45.0,
    "invalidation_condition": "If {HTF_LABEL} price closes below {HTF_LABEL} EMA20"
  }},
  "DOGE": {{
    "signal": "sell_to_enter",
    "leverage": 10,
    "confidence": 0.75,
    "profit_target": 0.145,
    "stop_loss": 0.165,
    "risk_usd": 45.0,
    "invalidation_condition": "If {HTF_LABEL} price closes above {HTF_LABEL} EMA20"
  }},
  "LINK": {{ "signal": "hold" }},
  "ASTER": {{ "signal": "hold" }}
}}

Remember: You are a systematic trading model. Make principled decisions based on quantitative data and advanced technical analysis."""
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7, "max_tokens": 4096 # Increased max_tokens
            }

            print("🔄 Sending request to DeepSeek API...")
            response = requests.post(self.base_url, json=data, headers=headers, timeout=120) # Increased timeout further
            response.raise_for_status()

            result = response.json()
            if not result.get('choices') or not result['choices'][0].get('message'):
                raise ValueError("DeepSeek API returned unexpected structure.")
            return result['choices'][0]['message']['content']

        except requests.exceptions.Timeout:
             print("❌ DeepSeek API request timed out.")
             return self._get_error_response("API Timeout")
        except requests.exceptions.RequestException as e:
            print(f"❌ DeepSeek API request failed: {e}")
            return self._get_error_response(f"API Request Failed: {e}")
        except Exception as e:
            print(f"❌ DeepSeek API error: {e}")
            return self._get_error_response(f"General API Error: {e}")

    def _get_simulation_response(self, prompt: str) -> str:
        """Simulation response without API"""
        print("⚠️  Using simulation mode...")
        return """
CHAIN_OF_THOUGHTS
Simulation Mode: Assuming market pullback. Shorting SOL based on simulated {HTF_LABEL} resistance. Aiming for 1:1.5 R/R using simulated ATR. Holding others.
DECISIONS
{{
  "SOL": {{
    "signal": "sell_to_enter",
    "leverage": 10,
    "quantity_usd": 30,
    "confidence": 0.65,
    "profit_target": 185.0,
    "stop_loss": 198.0,
    "risk_usd": 15.0,
    "invalidation_condition": "If {HTF_LABEL} price closes above 199.0"
  }},
  "XRP": {{ "signal": "hold" }},
  "ADA": {{ "signal": "hold" }},
  "DOGE": {{ "signal": "hold" }},
  "ASTER": {{ "signal": "hold" }},
  "LINK": {{ "signal": "hold" }}
}}
"""
    def get_cached_decisions(self) -> str:
        """Get cached decisions from recent successful cycles"""
        try:
            from config import Config
            cached_cycles = safe_file_read("cycle_history.json", default_data=[])
            if not cached_cycles:
                return self.get_safe_hold_decisions()
            
            # Get the most recent successful cycle with valid decisions
            for cycle in reversed(cached_cycles[-5:]):  # Check last 5 cycles
                decisions = cycle.get('decisions', {})
                if decisions and isinstance(decisions, dict):
                    # Check if decisions are valid (not just all holds)
                    valid_signals = [d for d in decisions.values() if isinstance(d, dict) and d.get('signal') in ['buy_to_enter', 'sell_to_enter']]
                    if valid_signals:
                        print("🔄 Using cached decisions from recent successful cycle")
                        return f"""
CHAIN_OF_THOUGHTS
API Error - Using cached decisions from recent successful cycle. Continuing with established strategy.
DECISIONS
{json.dumps(decisions, indent=2)}
"""
            
            # Fallback to safe hold decisions
            return self.get_safe_hold_decisions()
            
        except Exception as e:
            print(f"⚠️ Cache retrieval error: {e}")
            return self.get_safe_hold_decisions()

    def get_safe_hold_decisions(self) -> str:
        """Generate safe hold decisions for all coins"""
        print("🛡️ Generating safe hold decisions")
        hold_decisions = {}
        for coin in ['XRP', 'DOGE', 'ASTER', 'ADA', 'LINK', 'SOL']:
            hold_decisions[coin] = {"signal": "hold", "justification": "Safe mode: Holding due to API error"}
        
        return f"""
CHAIN_OF_THOUGHTS
API Error - Operating in safe mode. Holding all positions/cash to preserve capital.
DECISIONS
{json.dumps(hold_decisions, indent=2)}
"""

    def _get_error_response(self, error_message: str) -> str:
        """Enhanced error response with intelligent recovery"""
        print(f"🔧 Enhanced error handling for: {error_message}")
        
        error_type = type(error_message).__name__ if isinstance(error_message, Exception) else str(error_message)
        
        # Connection errors: Try cache first
        if "Connection" in error_type or "Timeout" in error_type:
            return self.get_cached_decisions()
        
        # Other errors: Use safe hold decisions
        return self.get_safe_hold_decisions()


# --- Market Data Class ---
class RealMarketData:
    """Real market data from Binance Spot and Futures"""

    def __init__(self):
        self.spot_url = "https://api.binance.com/api/v3"
        self.futures_url = "https://fapi.binance.com/fapi/v1"
        self.available_coins = ['XRP', 'DOGE', 'ASTER', 'ADA', 'LINK', 'SOL'] # SHIB replaced with ASTER
        self.indicator_history_length = 10
        self.session = RetryManager.create_session_with_retry()
        self.preloaded_indicators: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def clear_preloaded_indicators(self):
        """Clear any preloaded indicator snapshots (typically once per cycle)."""
        self.preloaded_indicators = {}

    def store_preloaded_indicator(self, coin: str, interval: str, indicators: Dict[str, Any]):
        """Store a snapshot of indicators for reuse during the same cycle."""
        if not isinstance(indicators, dict):
            return
        coin_store = self.preloaded_indicators.setdefault(coin, {})
        coin_store[interval] = copy.deepcopy(indicators)

    def set_preloaded_indicators(self, cache: Dict[str, Dict[str, Dict[str, Any]]]):
        """Bulk load pre-computed indicator cache (deep copy)."""
        preloaded: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for coin, intervals in (cache or {}).items():
            preloaded[coin] = {}
            for interval, data in (intervals or {}).items():
                if isinstance(data, dict):
                    preloaded[coin][interval] = copy.deepcopy(data)
        self.preloaded_indicators = preloaded

    def get_real_time_data(self, symbol: str, interval: str = '3m', limit: int = 100) -> pd.DataFrame:
        """Get real OHLCV data from Binance Spot with enhanced error handling and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                fetch_limit = limit + self.indicator_history_length + 50
                params = {'symbol': f'{symbol}USDT', 'interval': interval, 'limit': fetch_limit}
                response = self.session.get(f"{self.spot_url}/klines", params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if len(data) < 50:
                    print(f"⚠️ Warning: Insufficient kline data for {symbol} ({interval}). Got {len(data)}.")
                    return pd.DataFrame()

                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Enhanced data validation
                if self._validate_kline_data(df, symbol, interval):
                    return df
                else:
                    print(f"❌ Data validation failed for {symbol} ({interval}) - attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print(f"❌ All retries failed for {symbol} ({interval}). Returning empty DataFrame.")
                        return pd.DataFrame()
                    
            except requests.exceptions.Timeout:
                print(f"❌ Timeout for {symbol} ({interval}) - attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"❌ All retries timed out for {symbol} ({interval})")
                    return pd.DataFrame()
            except Exception as e:
                print(f"❌ Kline data error {symbol} ({interval}) - attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"❌ All retries failed for {symbol} ({interval})")
                    return pd.DataFrame()
        
        return pd.DataFrame()

    def _validate_kline_data(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        """Validate kline data quality with enhanced volume checks"""
        if df.empty:
            print(f"⚠️ Empty DataFrame for {symbol} ({interval})")
            return False
            
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                print(f"⚠️ Invalid price data for {symbol} ({interval}): {col} contains zero/negative values")
                return False
        
        # Check for identical prices (stuck data)
        if df['close'].nunique() < 3:  # Less than 3 unique prices
            print(f"⚠️ Stuck price data for {symbol} ({interval}): only {df['close'].nunique()} unique prices")
            return False
            
        # Enhanced volume validation - check for insufficient volume
        volume_sum = df['volume'].sum()
        volume_mean = df['volume'].mean()
        
        # Check for zero volume
        if volume_sum == 0:
            print(f"⚠️ Zero volume for {symbol} ({interval})")
            return False
            
        # Check for insufficient volume (especially for low-cap coins like ASTR)
        if volume_mean < 1000:  # Minimum average volume threshold
            print(f"⚠️ Insufficient volume for {symbol} ({interval}): avg volume {volume_mean:.0f} < 1000")
            return False
            
        # Check for reasonable price movement
        price_range = df['high'].max() - df['low'].min()
        if price_range == 0:
            print(f"⚠️ No price movement for {symbol} ({interval})")
            return False
            
        return True

    def get_open_interest(self, symbol: str) -> float:
        """Get Latest Open Interest from Binance Futures"""
        try:
            params = {'symbol': f'{symbol}USDT'}
            response = self.session.get(f"{self.futures_url}/openInterest", params=params, timeout=5)
            response.raise_for_status()
            return float(response.json()['openInterest'])
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                print(f"ℹ️ OI not available for {symbol}USDT on Futures.")
            else:
                print(f"❌ OI error for {symbol}: {e}")
            return 0.0

    def get_funding_rate(self, symbol: str) -> float:
        """Get Latest Funding Rate from Binance Futures"""
        try:
            params = {'symbol': f'{symbol}USDT'}
            response = self.session.get(f"{self.futures_url}/premiumIndex", params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list): data = data[0] if data else {}

            rate = data.get('lastFundingRate')
            if rate is not None and rate != '': return float(rate)
            else:
                 # print(f"ℹ️ Using nextFundingRate for {symbol}.") # Less verbose
                 rate = data.get('nextFundingRate')
                 return float(rate) if rate is not None and rate != '' else 0.0
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and (e.response.status_code in [404, 400]):
                 print(f"ℹ️ Funding Rate not available for {symbol}USDT on Futures.")
            else:
                print(f"❌ Funding Rate error for {symbol}: {e}")
            return 0.0

    # --- Indicator Calculation Functions ---
    def calculate_ema_series(self, prices, period): return prices.ewm(span=period, adjust=False).mean()
    def calculate_rsi_series(self, prices, period=14):
        if len(prices) < period + 1: return pd.Series([np.nan] * len(prices))
        delta = prices.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean(); avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100); rsi.loc[avg_gain == 0] = 0
        return rsi
    def calculate_macd_series(self, prices, fast=12, slow=26, signal=9):
        if len(prices) < slow: return pd.Series([np.nan]*len(prices)), pd.Series([np.nan]*len(prices)), pd.Series([np.nan]*len(prices))
        ema_fast = prices.ewm(span=fast, adjust=False).mean(); ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow; macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    def calculate_atr_series(self, df_high, df_low, df_close, period=14):
        if len(df_close) < period + 1: return pd.Series([np.nan] * len(df_close))
        tr0 = abs(df_high - df_low); tr1 = abs(df_high - df_close.shift()); tr2 = abs(df_low - df_close.shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        atr = tr.ewm(com=period - 1, adjust=False).mean()
        return atr

    def get_technical_indicators(self, coin: str, interval: str) -> Dict[str, Any]:
        """Calculate technical indicators, returning history series"""
        cached = self.preloaded_indicators.get(coin, {}).get(interval)
        if isinstance(cached, dict):
            return copy.deepcopy(cached)

        df = self.get_real_time_data(coin, interval=interval)
        if df.empty or len(df) < 50:
            return {'error': f'Not enough data for {coin} {interval} (got {len(df)})'}

        close_prices = df['close']; current_price = close_prices.iloc[-1]; hist_len = self.indicator_history_length
        indicators = {'current_price': current_price}
        try:
            ema_20_series = self.calculate_ema_series(close_prices, 20); ema_50_series = self.calculate_ema_series(close_prices, 50)
            rsi_14_series = self.calculate_rsi_series(close_prices, 14); macd_line_series, macd_signal_series, macd_hist_series = self.calculate_macd_series(close_prices)
            atr_14_series = self.calculate_atr_series(df['high'], df['low'], df['close'], 14)

            indicators['ema_20'] = ema_20_series.iloc[-1]; indicators['ema_50'] = ema_50_series.iloc[-1]
            indicators['rsi_14'] = rsi_14_series.iloc[-1]; indicators['macd'] = macd_line_series.iloc[-1]
            indicators['macd_signal'] = macd_signal_series.iloc[-1]; indicators['macd_histogram'] = macd_hist_series.iloc[-1]
            indicators['atr_14'] = atr_14_series.iloc[-1] # Keep atr_14 available for AI prompt

            # Use .where(pd.notna, None) to convert NaN to None for JSON
            indicators['ema_20_series'] = ema_20_series.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()
            indicators['rsi_14_series'] = rsi_14_series.iloc[-hist_len:].round(3).where(pd.notna, None).tolist()
            indicators['macd_series'] = macd_line_series.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()

            if interval == '3m':
                 rsi_7_series = self.calculate_rsi_series(close_prices, 7)
                 indicators['rsi_7'] = rsi_7_series.iloc[-1]
                 indicators['rsi_7_series'] = rsi_7_series.iloc[-hist_len:].round(3).where(pd.notna, None).tolist()
            if interval == HTF_INTERVAL:
                 atr_3_series = self.calculate_atr_series(df['high'], df['low'], df['close'], 3)
                 indicators['atr_3'] = atr_3_series.iloc[-1]

            indicators['volume'] = df['volume'].iloc[-1]; avg_vol = df['volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
            indicators['avg_volume'] = avg_vol if pd.notna(avg_vol) else 0.0
            indicators['price_series'] = close_prices.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()

            for key, value in indicators.items():
                if isinstance(value, float) and np.isnan(value): indicators[key] = None
            self.store_preloaded_indicator(coin, interval, indicators)
            return indicators
        except Exception as e:
            print(f"❌ Indicator error {coin} ({interval}): {e}")
            traceback.print_exc()
            return {'current_price': current_price, 'error': str(e)}

    def get_all_real_prices(self) -> Dict[str, float]:
        """Get real prices for all coins from Spot with enhanced error handling"""
        prices: Dict[str, float] = {}
        symbols = [f"{coin}USDT" for coin in self.available_coins]

        def _assign_price(symbol: str, raw_price: Any):
            coin = symbol.replace("USDT", "")
            try:
                price_val = float(raw_price)
                if price_val <= 0:
                    raise ValueError(f"Non-positive price {price_val}")
                prices[coin] = price_val
            except Exception as e:
                print(f"⚠️ Invalid bulk price for {coin}: {raw_price} ({e}). Using fallback.")
                prices[coin] = self._get_fallback_price(coin)

        # First try batched endpoint (single request, lower latency)
        try:
            response = self.session.get(
                f"{self.spot_url}/ticker/price",
                params={'symbols': json.dumps(symbols, separators=(',', ':'))},
                timeout=3
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                for entry in data:
                    symbol = entry.get('symbol')
                    price_raw = entry.get('price')
                    if symbol and price_raw is not None:
                        _assign_price(symbol, price_raw)
                # Ensure we filled everything; fall back only for missing
                missing = [coin for coin in self.available_coins if coin not in prices]
                if not missing:
                    for coin, val in prices.items():
                        print(f"✅ {coin}: ${val:.4f}")
                    return prices
                else:
                    print(f"⚠️ Bulk price missing for: {', '.join(missing)}. Falling back to individual requests.")
            else:
                print("⚠️ Unexpected bulk ticker response format. Falling back to individual requests.")
        except Exception as e:
            print(f"⚠️ Bulk price fetch failed: {e}. Falling back to individual requests.")

        # Fallback to individual calls (still using session, without artificial delay)
        for coin in self.available_coins:
            try:
                response = self.session.get(
                    f"{self.spot_url}/ticker/price",
                    params={'symbol': f"{coin}USDT"},
                    timeout=3
                )
                response.raise_for_status()
                data = response.json()
                price_val = float(data.get('price', 0))
                if price_val <= 0:
                    raise ValueError(f"Non-positive price {price_val}")
                prices[coin] = price_val
                print(f"✅ {coin}: ${price_val:.4f}")
            except Exception as e:
                print(f"❌ {coin} price error: {e}. Using fallback...")
                prices[coin] = self._get_fallback_price(coin)
                
        return prices

    def _get_fallback_price(self, coin: str) -> float:
        """Get fallback price using multiple methods"""
        # Method 1: Try 1m kline data
        try:
            df = self.get_real_time_data(coin, interval='1m', limit=1)
            if not df.empty and not df['close'].empty:
                price_val = df['close'].iloc[-1]
                if price_val > 0 and pd.notna(price_val):
                    print(f"   Fallback 1m kline: ${price_val:.4f}")
                    return price_val
        except Exception as e:
            print(f"   Fallback 1m failed: {e}")
        
        # Method 2: Try 3m kline data
        try:
            df = self.get_real_time_data(coin, interval='3m', limit=1)
            if not df.empty and not df['close'].empty:
                price_val = df['close'].iloc[-1]
                if price_val > 0 and pd.notna(price_val):
                    print(f"   Fallback 3m kline: ${price_val:.4f}")
                    return price_val
        except Exception as e:
            print(f"   Fallback 3m failed: {e}")
        
        # Method 3: Use cached price from previous cycle
        try:
            from config import Config
            cached_prices = safe_file_read("portfolio_state.json", default_data={})
            if 'positions' in cached_prices:
                for pos_coin, position in cached_prices['positions'].items():
                    if pos_coin == coin and 'current_price' in position:
                        cached_price = position['current_price']
                        if cached_price > 0:
                            print(f"   Fallback cached: ${cached_price:.4f}")
                            return cached_price
        except Exception as e:
            print(f"   Fallback cache failed: {e}")
        
        # Final fallback: return 0 with warning
        print(f"   ⚠️ All fallbacks failed for {coin}. Price set to 0.")
        return 0.0

    def get_market_sentiment(self, coin: str) -> Dict[str, Any]:
        """Get Open Interest and Funding Rate (Nof1ai format)"""
        open_interest = self.get_open_interest(coin)
        funding_rate = self.get_funding_rate(coin)
        
        # Nof1ai format: "Latest: X Average: Y" for Open Interest
        avg_oi = open_interest  # Simplified average calculation
        return {
            'open_interest': open_interest,
            'open_interest_avg': avg_oi,
            'funding_rate': funding_rate
        }
# --- Portfolio Manager Class ---
class PortfolioManager:
    """Manages the portfolio state, positions, and history."""

    def __init__(self, initial_balance: float = None):
        # Dinamik initial balance - eğer verilmediyse gerçek balance'dan al veya $200 kullan
        if initial_balance is None:
            # Önce saved state'den dene, yoksa Config'den al
            saved_state = safe_file_read("portfolio_state.json", default_data={})
            if 'initial_balance' in saved_state:
                self.initial_balance = saved_state['initial_balance']
            else:
                self.initial_balance = Config.INITIAL_BALANCE
        else:
            self.initial_balance = initial_balance
            
        self.state_file = "portfolio_state.json"; self.history_file = "trade_history.json"
        self.override_file = "manual_override.json"; self.cycle_history_file = "cycle_history.json"
        self.max_cycle_history = 50; self.maintenance_margin_rate = 0.01

        self.current_balance = self.initial_balance; self.positions = {}
        self.directional_bias = self._init_directional_bias()
        self.trend_state: Dict[str, Dict[str, Any]] = {}
        self.trend_flip_cooldown = 2
        self.trend_flip_history_window = 5
        # Trend flip cooldown yönetimi PortfolioManager tarafında tutulur.
        self.indicator_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.last_execution_report: Dict[str, Any] = {}
        self.history_reset_interval = getattr(Config, "HISTORY_RESET_INTERVAL", 35)
        self.last_history_reset_cycle = 0
        self.cycles_since_history_reset = 0
        self.trading_mode = getattr(Config, "TRADING_MODE", "simulation")
        self.is_live_trading = self.trading_mode == "live"
        self.order_executor: Optional["BinanceOrderExecutor"] = None
        self.directional_cooldowns: Dict[str, int] = {'long': 0, 'short': 0}
        self.relaxed_countertrend_cycles: int = 0
        self.counter_trend_cooldown: int = 0
        self.counter_trend_consecutive_losses: int = 0
        self.coin_cooldowns: Dict[str, int] = {}  # Coin bazlı cooldown: {coin: cycles_remaining}

        self.current_cycle_number = 0

        self.trade_history = self.load_trade_history() # Load first
        self.load_state() # Loads balance, positions, trade_count
        self.cycle_history = self.load_cycle_history()
        self.risk_manager = AdvancedRiskManager()  # Initialize risk manager
        self.market_data = RealMarketData()  # Initialize market data for counter-trend detection

        # Initialize total_value before _initialize_live_trading (which calls sync_live_account)
        self.total_value = self.current_balance
        self.total_return = 0.0
        self.start_time = datetime.now()
        self.portfolio_values_history = [self.initial_balance]  # Track portfolio values for Sharpe ratio
        self.sharpe_ratio = 0.0

        if self.is_live_trading:
            self._initialize_live_trading()
        elif BINANCE_IMPORT_ERROR:
            print(f"ℹ️ Binance executor unavailable ({BINANCE_IMPORT_ERROR}). Staying in simulation mode.")

        self.update_prices({}, increment_loss_counters=False) # Calculate initial value with loaded positions

    def _init_directional_bias(self) -> Dict[str, Dict[str, Any]]:
        return {
            'long': {
                'rolling': deque(maxlen=20),
                'net_pnl': 0.0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'consecutive_losses': 0,
                'consecutive_wins': 0,
                'caution_active': False,
            'caution_win_progress': 0,
            'loss_streak_loss_usd': 0.0
            },
            'short': {
                'rolling': deque(maxlen=20),
                'net_pnl': 0.0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'consecutive_losses': 0,
                'consecutive_wins': 0,
                'caution_active': False,
            'caution_win_progress': 0,
            'loss_streak_loss_usd': 0.0
            }
        }

    def load_state(self):
        data = safe_file_read(self.state_file, default_data={})
        self.current_balance = data.get('current_balance', self.initial_balance)
        self.positions = data.get('positions', {})
        self.trade_count = data.get('trade_count', len(self.trade_history)) # Initialize from history if not in state
        print(f"✅ Loaded state ({len(self.positions)} positions, {self.trade_count} closed trades)" if data else "ℹ️ No state file found.")

        bias_state = data.get('directional_bias')
        if bias_state:
            self.directional_bias = self._init_directional_bias()
            for side in ('long', 'short'):
                stored = bias_state.get(side, {})
                stats = self.directional_bias[side]
                stats['rolling'].extend(stored.get('rolling', []))
                stats['net_pnl'] = stored.get('net_pnl', 0.0)
                stats['trades'] = stored.get('trades', 0)
                stats['wins'] = stored.get('wins', 0)
                stats['losses'] = stored.get('losses', 0)
                stats['consecutive_losses'] = stored.get('consecutive_losses', 0)
                stats['consecutive_wins'] = stored.get('consecutive_wins', 0)
                stats['caution_active'] = stored.get('caution_active', False)
                stats['caution_win_progress'] = stored.get('caution_win_progress', 0)
                stats['loss_streak_loss_usd'] = stored.get('loss_streak_loss_usd', 0.0)
        self.last_history_reset_cycle = data.get('last_history_reset_cycle', self.last_history_reset_cycle)
        self.cycles_since_history_reset = data.get('cycles_since_history_reset', self.cycles_since_history_reset)
        self.directional_cooldowns = data.get('directional_cooldowns', {'long': 0, 'short': 0})
        self.relaxed_countertrend_cycles = data.get('relaxed_countertrend_cycles', 0)
        self.counter_trend_cooldown = data.get('counter_trend_cooldown', 0)
        self.counter_trend_consecutive_losses = data.get('counter_trend_consecutive_losses', 0)
        self.coin_cooldowns = data.get('coin_cooldowns', {})

    def save_state(self):
        data = {
            'current_balance': self.current_balance,
            'positions': self.positions,
            'total_value': self.total_value,
            'total_return': self.total_return,
            'initial_balance': self.initial_balance,
            'trade_count': self.trade_count,
            'last_updated': datetime.now().isoformat(),
            'sharpe_ratio': self.sharpe_ratio,
            'directional_bias': self._serialize_directional_bias(),
            'last_history_reset_cycle': self.last_history_reset_cycle,
            'cycles_since_history_reset': self.cycles_since_history_reset,
            'directional_cooldowns': self.directional_cooldowns,
            'relaxed_countertrend_cycles': self.relaxed_countertrend_cycles,
            'counter_trend_cooldown': self.counter_trend_cooldown,
            'counter_trend_consecutive_losses': self.counter_trend_consecutive_losses,
            'coin_cooldowns': self.coin_cooldowns
        }
        safe_file_write(self.state_file, data); print(f"✅ Saved state.")

    # --- Live trading helpers -------------------------------------------------
    def _initialize_live_trading(self):
        """Configure Binance executor when live trading mode is enabled."""
        if BinanceOrderExecutor is None:
            print("❌ Live trading requested but Binance executor is unavailable.")
            self.is_live_trading = False
            return
        try:
            self.order_executor = BinanceOrderExecutor(self.market_data.available_coins)
            if not self.order_executor.is_live():
                print("⚠️ Live trading requested but executor initialized in simulation mode. Reverting to paper trading.")
                self.is_live_trading = False
                self.order_executor = None
                return
            print(f"✅ Live trading mode enabled (Binance {'TESTNET' if Config.BINANCE_TESTNET else 'MAINNET'}).")
            self.sync_live_account()
        except BinanceAPIError as exc:
            print(f"❌ Binance setup failed: {exc}. Reverting to simulation mode.")
            self.is_live_trading = False
            self.order_executor = None
        except Exception as exc:
            print(f"❌ Unexpected Binance setup error: {exc}. Reverting to simulation mode.")
            self.is_live_trading = False
            self.order_executor = None

    def _build_default_exit_plan(self, direction: str, entry_price: float) -> Dict[str, float]:
        """Generate a sensible default exit plan when AI data is unavailable."""
        try:
            entry = float(entry_price or 0.0)
        except (TypeError, ValueError):
            entry = 0.0
        if entry <= 0:
            entry = max(Config.MIN_EXIT_PLAN_OFFSET, 1.0)
        stop_pct = max(Config.DEFAULT_STOP_LOSS_PCT, 1e-4)
        tp_pct = max(Config.DEFAULT_PROFIT_TARGET_PCT, stop_pct * 1.5)
        min_offset = max(Config.MIN_EXIT_PLAN_OFFSET, 1e-6)
        stop_offset = max(entry * stop_pct, min_offset)
        tp_offset = max(entry * tp_pct, min_offset)
        if direction == 'short':
            stop_loss = entry + stop_offset
            profit_target = max(entry - tp_offset, 0.0)
        else:
            stop_loss = max(entry - stop_offset, 0.0)
            profit_target = entry + tp_offset
        return {
            'stop_loss': stop_loss,
            'profit_target': profit_target
        }

    def _ensure_exit_plan(self, position: Dict[str, Any], *candidate_plans: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure a position carries a valid exit plan, supplementing with defaults if needed."""
        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price') or position.get('current_price')
        default_plan = self._build_default_exit_plan(direction, entry_price)
        final_plan = default_plan.copy()
        provided_keys = set()
        for plan in candidate_plans:
            if not plan:
                continue
            for key, value in plan.items():
                if value is None:
                    continue
                final_plan[key] = value
                if key in ('stop_loss', 'profit_target') and isinstance(value, (int, float)) and value > 0:
                    provided_keys.add(key)
        missing_required = [key for key in ('stop_loss', 'profit_target') if key not in provided_keys]
        if missing_required:
            symbol = position.get('symbol', 'UNKNOWN')
            print(f"⚠️ Missing {', '.join(missing_required)} for {symbol} - using default exit plan offsets.")
        position['exit_plan'] = final_plan
        return final_plan

    def _merge_live_positions(self, snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Merge Binance snapshot with local runtime metadata (exit plans, confidence, etc.)."""
        merged: Dict[str, Dict[str, Any]] = {}
        existing_positions = self.positions if isinstance(self.positions, dict) else {}
        for coin, snap_pos in snapshot.items():
            previous = existing_positions.get(coin, {})
            merged_pos: Dict[str, Any] = {}
            merged_pos.update(previous)
            merged_pos.update(snap_pos)
            merged_pos['symbol'] = coin

            # Carry forward runtime metadata that Binance snapshot doesn't include
            for key in (
                'confidence',
                'loss_cycle_count',
                'trend_context',
                'trend_alignment',
                'entry_oid',
                'tp_oid',
                'sl_oid',
                'wait_for_fill',
                'entry_volume_ratio',
                'entry_volume',
                'entry_avg_volume',
                'entry_atr_14',
                'trailing'
            ):
                if key in previous and key not in merged_pos:
                    merged_pos[key] = previous[key]

            # Risk USD should reflect current margin if available
            margin_usd = merged_pos.get('margin_usd')
            if isinstance(margin_usd, (int, float)):
                merged_pos['risk_usd'] = margin_usd

            existing_plan = previous.get('exit_plan')
            snapshot_plan = snap_pos.get('exit_plan')
            self._ensure_exit_plan(merged_pos, snapshot_plan, existing_plan)
            merged[coin] = merged_pos
        return merged

    def sync_live_account(self):
        """Refresh balances and open positions from Binance when in live mode."""
        if not self.is_live_trading or not self.order_executor or not self.order_executor.is_live():
            return
        
        overview = None
        try:
            overview = self.order_executor.get_account_overview()
            if overview:
                available = overview.get("availableBalance")
                total_wallet_balance = overview.get("totalWalletBalance")
                wallet_balance = overview.get("walletBalance")
                
                # Debug: Log what we got from Binance
                print(f"🔍 Binance API Response: availableBalance={available}, totalWalletBalance={total_wallet_balance}, walletBalance={wallet_balance}")
                
                # Update available cash balance
                if available is not None and available > 0:
                    old_balance = self.current_balance
                    self.current_balance = float(available)
                    if abs(old_balance - self.current_balance) > 0.01:  # Only log if significant change
                        print(f"💰 Balance updated: ${old_balance:.2f} → ${self.current_balance:.2f}")
                
                # Note: We'll calculate total_value manually after positions are synced
                # Total value = Available cash + Margin used + Unrealized PnL
                # Binance totalWalletBalance is used for validation only
                if total_wallet_balance is not None and total_wallet_balance > 0:
                    print(f"✅ Binance totalWalletBalance: ${total_wallet_balance:.2f} (will validate against calculated value)")
                elif wallet_balance is not None and wallet_balance > 0:
                    print(f"⚠️ totalWalletBalance not available, using walletBalance: ${wallet_balance:.2f}")
                else:
                    print(f"⚠️ Neither totalWalletBalance nor walletBalance available from Binance API")
        except BinanceAPIError as exc:
            print(f"⚠️ Binance balance sync failed: {exc}")
        except Exception as exc:
            print(f"⚠️ Unexpected Binance balance sync error: {exc}")

        try:
            snapshot = self.order_executor.get_positions_snapshot()
            if isinstance(snapshot, dict):
                self.positions = self._merge_live_positions(snapshot)
                
                # Calculate total margin used from all open positions
                # For cross margin, margin_usd might be 0, so calculate from notional/leverage
                total_margin_used = 0.0
                for pos in self.positions.values():
                    margin = pos.get('margin_usd', 0.0)
                    if margin <= 0:
                        # Calculate margin from notional and leverage (for cross margin)
                        notional = pos.get('notional_usd', 0.0)
                        leverage = pos.get('leverage', 1)
                        if notional > 0 and leverage > 0:
                            margin = notional / leverage
                    if isinstance(margin, (int, float)) and margin > 0:
                        total_margin_used += margin
                
                old_total = self.total_value
                
                # Calculate total unrealized PnL
                total_unrealized_pnl = 0.0
                for pos in self.positions.values():
                    pnl = pos.get('unrealized_pnl', 0.0)
                    if isinstance(pnl, (int, float)):
                        total_unrealized_pnl += pnl
                
                # Total value = Available cash + Margin used + Unrealized PnL
                # This is the correct formula: what you have available + what's locked in positions + unrealized gains/losses
                self.total_value = self.current_balance + total_margin_used + total_unrealized_pnl
                
                if abs(old_total - self.total_value) > 0.01:
                    print(f"📊 Total value updated: ${old_total:.2f} → ${self.total_value:.2f}")
                    print(f"   (Available cash: ${self.current_balance:.2f} + Margin used: ${total_margin_used:.2f} + Unrealized PnL: ${total_unrealized_pnl:.2f})")
                    
                    # Debug: Also show what Binance says for comparison
                    if overview:
                        total_wb = overview.get("totalWalletBalance")
                        wallet_b = overview.get("walletBalance")
                        if total_wb:
                            wallet_b_str = f"${wallet_b:.2f}" if wallet_b else "N/A"
                            print(f"   (Binance totalWalletBalance: ${total_wb:.2f}, walletBalance: {wallet_b_str})")
            else:
                self.positions = {}
                # No positions, total value = totalWalletBalance (or available cash if not available)
                if overview and overview.get("totalWalletBalance") and overview.get("totalWalletBalance") > 0:
                    self.total_value = float(overview["totalWalletBalance"])
                else:
                    self.total_value = self.current_balance
        except BinanceAPIError as exc:
            print(f"⚠️ Binance position sync failed: {exc}")
        except Exception as exc:
            print(f"⚠️ Unexpected Binance position sync error: {exc}")

    @staticmethod
    def _calculate_realized_pnl(entry_price: float, exit_price: float, quantity: float, direction: str) -> float:
        if quantity <= 0 or entry_price <= 0 or exit_price <= 0:
            return 0.0
        if direction == 'long':
            return (exit_price - entry_price) * quantity
        return (entry_price - exit_price) * quantity

    def execute_live_entry(
        self,
        coin: str,
        direction: str,
        quantity: float,
        leverage: int,
        current_price: float,
        notional_usd: float,
        confidence: float,
        margin_usd: Optional[float] = None,
        stop_loss: Optional[float] = None,
        profit_target: Optional[float] = None,
        invalidation: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        try:
            order = self.order_executor.place_market_order(
                coin=coin,
                direction=direction,
                quantity=quantity,
                leverage=leverage,
                price_reference=current_price,
                reduce_only=False,
            )
            executed_qty = float(order.get("executedQty", 0.0))
            avg_price = float(order.get("avgPriceComputed", order.get("avgPrice", 0.0)))
            self.sync_live_account()
            position = self.positions.get(coin, {})
            
            # Use provided margin_usd if available, otherwise calculate from position or notional
            if margin_usd is None or margin_usd <= 0:
                if position and position.get('margin_usd', 0) > 0:
                    margin_usd = position.get('margin_usd')
                else:
                    # Calculate from executed notional (more accurate than initial notional)
                    executed_notional = executed_qty * avg_price
                    margin_usd = executed_notional / max(leverage, 1)
            
            notional_runtime = position.get('notional_usd') if position else executed_qty * avg_price
            if position:
                position['confidence'] = confidence
                # Ensure margin_usd is saved to position for later use in TP/SL checks
                if 'margin_usd' not in position or position.get('margin_usd', 0) <= 0:
                    position['margin_usd'] = margin_usd
                    print(f"💾 Saved margin_usd=${margin_usd:.2f} to position for {coin}")
                exit_plan = position.setdefault('exit_plan', {})
                if stop_loss is not None:
                    exit_plan['stop_loss'] = stop_loss
                if profit_target is not None:
                    exit_plan['profit_target'] = profit_target
                if invalidation is not None:
                    exit_plan['invalidation_condition'] = invalidation
                position['risk_usd'] = position.get('margin_usd', margin_usd)
                self._ensure_exit_plan(position, exit_plan)
                
                # Calculate and save kademeli stop loss to exit_plan (for 30-second monitoring)
                # No Binance orders - all TP/SL decisions made by 30-second monitoring loop (like simulation mode)
                if executed_qty > 0:
                    # Calculate margin-based stop loss using kademeli loss cutting
                    loss_multiplier = 0.03  # Default: %3 for margin >= 50
                    if margin_usd < 30:
                        loss_multiplier = 0.07  # %7 for margin < 30
                    elif margin_usd < 40:
                        loss_multiplier = 0.05  # %5 for margin 30-40
                    elif margin_usd < 50:
                        loss_multiplier = 0.05  # %5 for margin 40-50
                    else:
                        loss_multiplier = 0.03  # %3 for margin >= 50
                    
                    loss_threshold_usd = margin_usd * loss_multiplier
                    
                    # Calculate stop loss price from loss threshold
                    if direction == 'long':
                        margin_based_stop_loss = avg_price - (loss_threshold_usd / executed_qty)
                    else:  # short
                        margin_based_stop_loss = avg_price + (loss_threshold_usd / executed_qty)
                    
                    # Use the tighter stop loss (closer to entry price = more conservative)
                    # For long: higher stop_loss is tighter (closer to entry)
                    # For short: lower stop_loss is tighter (closer to entry)
                    if stop_loss is not None and stop_loss > 0:
                        if direction == 'long':
                            # For long: use the higher stop loss (more conservative)
                            final_stop_loss = max(stop_loss, margin_based_stop_loss)
                        else:  # short
                            # For short: use the lower stop loss (more conservative)
                            final_stop_loss = min(stop_loss, margin_based_stop_loss)
                    else:
                        final_stop_loss = margin_based_stop_loss
                    
                    # Final validation: ensure stop loss direction is correct
                    if direction == 'long':
                        if final_stop_loss >= avg_price:
                            # Stop loss cannot be at or above entry for long - recalculate from loss_threshold
                            final_stop_loss = avg_price - (loss_threshold_usd / executed_qty)
                            print(f"⚠️ Final validation: Stop loss for {coin} LONG was invalid (>= entry), recalculated to ${format_num(final_stop_loss, 4)}")
                    else:  # short
                        if final_stop_loss <= avg_price:
                            # Stop loss cannot be at or below entry for short - recalculate from loss_threshold
                            final_stop_loss = avg_price + (loss_threshold_usd / executed_qty)
                            print(f"⚠️ Final validation: Stop loss for {coin} SHORT was invalid (<= entry), recalculated to ${format_num(final_stop_loss, 4)}")
                    
                    # Save kademeli stop loss to exit_plan (will be checked by 30-second monitoring)
                    if final_stop_loss > 0:
                        exit_plan['stop_loss'] = final_stop_loss
                        print(f"💾 Kademeli stop loss saved for {coin}: ${format_num(final_stop_loss, 4)} (${loss_threshold_usd:.2f} loss limit, {loss_multiplier*100:.1f}% of ${margin_usd:.2f} margin) - will be monitored by 30s loop")
            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "margin_usd": margin_usd,
                "notional_usd": notional_runtime,
            }
        except BinanceAPIError as exc:
            print(f"❌ Binance entry order failed for {coin}: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            print(f"❌ Unexpected Binance entry error for {coin}: {exc}")
            return {"success": False, "error": str(exc)}

    def execute_live_close(
        self,
        coin: str,
        position: Dict[str, Any],
        current_price: float,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        if not position:
            return {"success": False, "error": "no_position"}

        direction = position.get('direction', 'long')
        quantity = float(position.get('quantity', 0.0) or 0.0)
        if quantity <= 0:
            return {"success": False, "error": "invalid_quantity"}

        try:
            # No Binance TP/SL orders to cancel - all managed by 30-second monitoring
            order = self.order_executor.close_position(
                coin=coin,
                direction=direction,
                quantity=quantity,
                price_reference=current_price,
            )
            executed_qty = float(order.get("executedQty", 0.0))
            avg_price = float(order.get("avgPriceComputed", order.get("avgPrice", 0.0)))
            pnl = self._calculate_realized_pnl(position.get("entry_price", 0.0), avg_price, executed_qty, direction)
            self.sync_live_account()
            history_entry = {
                "symbol": coin,
                "direction": direction,
                "entry_price": position.get("entry_price"),
                "exit_price": avg_price,
                "quantity": executed_qty,
                "notional_usd": executed_qty * avg_price,
                "pnl": pnl,
                "entry_time": position.get("entry_time"),
                "exit_time": datetime.now().isoformat(),
                "leverage": position.get("leverage"),
                "close_reason": reason or "live_close",
                "exchange_order_id": order.get("orderId"),
            }
            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "pnl": pnl,
                "history_entry": history_entry,
            }
        except BinanceAPIError as exc:
            print(f"❌ Binance close order failed for {coin}: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            print(f"❌ Unexpected Binance close error for {coin}: {exc}")
            return {"success": False, "error": str(exc)}

    def execute_live_partial_close(
        self,
        coin: str,
        position: Dict[str, Any],
        close_percent: float,
        current_price: float,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        if not position:
            return {"success": False, "error": "no_position"}

        direction = position.get('direction', 'long')
        quantity = float(position.get('quantity', 0.0) or 0.0)
        if quantity <= 0:
            return {"success": False, "error": "invalid_quantity"}
        close_percent = max(0.0, min(close_percent, 1.0))
        requested_qty = quantity * close_percent
        if requested_qty <= 0:
            return {"success": False, "error": "zero_quantity"}

        try:
            order = self.order_executor.close_position(
                coin=coin,
                direction=direction,
                quantity=requested_qty,
                price_reference=current_price,
            )
            executed_qty = float(order.get("executedQty", 0.0))
            avg_price = float(order.get("avgPriceComputed", order.get("avgPrice", 0.0)))
            if executed_qty <= 0:
                return {"success": False, "error": "no_fill"}
            pnl = self._calculate_realized_pnl(position.get("entry_price", 0.0), avg_price, executed_qty, direction)
            # Sync account balance after partial close to get updated balance
            self.sync_live_account()
            history_entry = {
                "symbol": coin,
                "direction": direction,
                "entry_price": position.get("entry_price"),
                "exit_price": avg_price,
                "quantity": executed_qty,
                "notional_usd": executed_qty * avg_price,
                "pnl": pnl,
                "entry_time": position.get("entry_time"),
                "exit_time": datetime.now().isoformat(),
                "leverage": position.get("leverage"),
                "close_reason": reason or "live_partial_close",
                "exchange_order_id": order.get("orderId"),
            }
            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "pnl": pnl,
                "history_entry": history_entry,
            }
        except BinanceAPIError as exc:
            print(f"❌ Binance partial close failed for {coin}: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            print(f"❌ Unexpected Binance partial close error for {coin}: {exc}")
            return {"success": False, "error": str(exc)}

    def _backup_historical_files(self, cycle_number: int) -> Optional[str]:
        """Create a timestamped backup for historical JSON files before wiping them."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join("history_backups", f"{timestamp}_cycle_{cycle_number}")
        files_to_backup = [
            (self.history_file, []),
            (self.cycle_history_file, []),
            ("performance_history.json", []),
            ("performance_report.json", [])  # Changed from {} to [] - now array format
        ]

        try:
            os.makedirs(backup_dir, exist_ok=True)
            backed_up = []

            for file_path, default in files_to_backup:
                data = safe_file_read(file_path, default)
                # Skip writing files that never existed and match the empty default
                if data == default and not os.path.exists(file_path):
                    continue
                target_path = os.path.join(backup_dir, os.path.basename(file_path))
                safe_file_write(target_path, data)
                
                # Calculate items count for metadata
                items_count = None
                if isinstance(data, list):
                    items_count = len(data)
                elif isinstance(data, dict):
                    # For dict, count keys (but performance_report.json should be array now)
                    items_count = len(data)
                
                backed_up.append({
                    "file": file_path,
                    "items": items_count
                })

            metadata = {
                "cycle_number": cycle_number,
                "backed_up_at": datetime.now().isoformat(),
                "files": backed_up
            }
            safe_file_write(os.path.join(backup_dir, "metadata.json"), metadata)
            print(f"💾 History backup created at {backup_dir}")
            return backup_dir
        except Exception as e:
            print(f"⚠️ History backup failed: {e}")
            return None

    def reset_historical_data(self, cycle_number: int):
        """Clear historical logs to prevent long-term bias while keeping live positions."""
        self._backup_historical_files(cycle_number)
        print(f"🧹 HISTORY RESET: Clearing logs at cycle {cycle_number}")
        self.trade_history = []
        self.trade_count = 0
        self.directional_bias = self._init_directional_bias()
        self.trend_state = {}
        self.cycle_history = []
        safe_file_write(self.history_file, [])
        safe_file_write(self.cycle_history_file, [])
        safe_file_write("performance_history.json", [])
        # Preserve existing performance reports, just add a reset marker
        existing_reports = safe_file_read("performance_report.json", [])
        if isinstance(existing_reports, dict):
            # Old format - convert to array
            if "reset_reason" not in existing_reports:
                existing_reports = [existing_reports]
            else:
                existing_reports = []
        elif not isinstance(existing_reports, list):
            existing_reports = []
        
        # Add reset marker
        reset_marker = {
            "reset_reason": "periodic_bias_control",
            "reset_at_cycle": cycle_number,
            "timestamp": datetime.now().isoformat()
        }
        existing_reports.append(reset_marker)
        
        # Keep only last 50 entries
        if len(existing_reports) > 50:
            existing_reports = existing_reports[-50:]
        
        safe_file_write("performance_report.json", existing_reports)
        self.portfolio_values_history = [self.total_value]
        for pos in self.positions.values():
            pos['loss_cycle_count'] = 0
        self.last_history_reset_cycle = cycle_number
        self.cycles_since_history_reset = 0
        self.directional_cooldowns = {'long': 0, 'short': 0}
        self.coin_cooldowns = {}  # Coin bazlı cooldown'ları da sıfırla
        self.counter_trend_cooldown = 0
        self.counter_trend_consecutive_losses = 0
        self.relaxed_countertrend_cycles = 0
        self.save_state()
        print("✅ History reset complete.")

    def _serialize_directional_bias(self) -> Dict[str, Dict[str, Any]]:
        serialized = {}
        for side, stats in self.directional_bias.items():
            serialized[side] = {
                'rolling': list(stats['rolling']),
                'net_pnl': stats['net_pnl'],
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'consecutive_losses': stats['consecutive_losses'],
                'consecutive_wins': stats.get('consecutive_wins', 0),
                'caution_active': stats.get('caution_active', False),
                'caution_win_progress': stats.get('caution_win_progress', 0),
                'loss_streak_loss_usd': stats.get('loss_streak_loss_usd', 0.0)
            }
        return serialized

    def load_trade_history(self) -> List[Dict]:
        history = safe_file_read(self.history_file, default_data=[]); print(f"✅ Loaded {len(history)} trades."); return history
    def save_trade_history(self):
        history_to_save = self.trade_history[-100:]; safe_file_write(self.history_file, history_to_save); print(f"✅ Saved {len(history_to_save)} trades.")
    def add_to_history(self, trade: Dict):
        self.trade_history.append(trade)
        self.trade_count = len(self.trade_history)
        self.save_trade_history()
        self.update_directional_bias(trade)
        self.save_state()

    def update_directional_bias(self, trade: Dict):
        direction = trade.get('direction')
        if direction not in ('long', 'short'):
            return
        stats = self.directional_bias[direction]
        pnl = float(trade.get('pnl', 0.0) or 0.0)
        print(f"📊 update_directional_bias called: {direction.upper()} trade, PnL=${pnl:.2f}")
        stats['rolling'].append(pnl)
        stats['net_pnl'] += pnl
        stats['trades'] += 1
        
        # Check if this is a counter-trend trade
        trend_alignment = trade.get('trend_alignment', 'unknown')
        is_counter_trend = (trend_alignment == 'counter_trend')
        
        if pnl > 0:
            stats['wins'] += 1
            stats['consecutive_losses'] = 0
            stats['consecutive_wins'] = stats.get('consecutive_wins', 0) + 1
            if stats.get('caution_active'):
                stats['caution_win_progress'] = stats.get('caution_win_progress', 0) + 1
                if stats['caution_win_progress'] >= 3:
                    stats['caution_active'] = False
                    stats['caution_win_progress'] = 0
            # Reset counter-trend consecutive losses on win
            if is_counter_trend:
                self.counter_trend_consecutive_losses = 0
        elif pnl < 0:
            stats['losses'] += 1
            stats['consecutive_losses'] += 1
            stats['consecutive_wins'] = 0
            stats['caution_win_progress'] = 0
            # loss_streak_loss_usd'yi güncelle - cooldown aktif olsa bile takip etmeye devam et
            # Cooldown sırasında da yeni kayıplar gelebilir ve toplam $5'i geçebilir
            current_loss_streak = stats.get('loss_streak_loss_usd', 0.0)
            stats['loss_streak_loss_usd'] = current_loss_streak + abs(pnl)
            
            # Coin bazlı cooldown: Zararla kapandığında o coin için 1 cycle cooldown
            coin_symbol = trade.get('symbol', '').upper()
            if coin_symbol:
                self.coin_cooldowns[coin_symbol] = 1
                print(f"🛡️ Coin cooldown ACTIVATED for {coin_symbol}: 1 cycle (loss: ${pnl:.2f})")
            
            if stats['consecutive_losses'] >= 3:
                stats['caution_active'] = True
                stats['caution_win_progress'] = 0
            # Cooldown: 3 consecutive losses OR $5 total loss → 3 cycle cooldown (sabit)
            # Cooldown aktif olsa bile, yeni kayıplar geldiğinde tekrar kontrol et ve gerekirse yeniden aktif et
            loss_streak_usd = stats.get('loss_streak_loss_usd', 0.0)
            consecutive = stats['consecutive_losses']
            should_activate = consecutive >= 3 or loss_streak_usd >= 5.0
            
            print(f"🔍 Cooldown check for {direction.upper()}: consecutive_losses={consecutive}, loss_streak_usd=${loss_streak_usd:.2f}, should_activate={should_activate}")
            
            if should_activate:
                self._activate_directional_cooldown(direction, 3)
                print(f"🛡️ Directional cooldown ACTIVATED for {direction.upper()}: consecutive_losses={consecutive}, loss_streak_usd=${loss_streak_usd:.2f}")
                # loss_streak_loss_usd'yi sıfırlama - bir sonraki zarar için takip etmeye devam et
                # Sadece cooldown bittiğinde sıfırlanacak
            
            # Counter-trend cooldown: 2 consecutive counter-trend losses
            if is_counter_trend:
                self.counter_trend_consecutive_losses += 1
                if self.counter_trend_consecutive_losses >= 2:
                    self.counter_trend_cooldown = 3
                    self.counter_trend_consecutive_losses = 0
                    print(f"🛡️ Counter-trend cooldown activated: 2 consecutive counter-trend losses (3 cycles cooldown).")
        else:
            stats['consecutive_losses'] = 0
            stats['consecutive_wins'] = 0
            stats['caution_win_progress'] = 0
            stats['loss_streak_loss_usd'] = 0.0
            # Reset counter-trend consecutive losses on breakeven
            if is_counter_trend:
                self.counter_trend_consecutive_losses = 0

    def count_positions_by_direction(self) -> Dict[str, int]:
        counts = {'long': 0, 'short': 0}
        for pos in self.positions.values():
            direction = pos.get('direction')
            if direction in counts:
                counts[direction] += 1
        return counts

    def _activate_directional_cooldown(self, direction: str, cycles: int = 3):
        if direction not in ('long', 'short'):
            return
        current = self.directional_cooldowns.get(direction, 0)
        # Mevcut cooldown'dan daha uzun bir süre varsa, onu kullan
        self.directional_cooldowns[direction] = max(current, cycles)
        self.relaxed_countertrend_cycles = max(self.relaxed_countertrend_cycles, 3)
        print(f"🛡️ Directional cooldown activated for {direction.upper()} trades (3 cycles). Counter-trend restrictions relaxed for 3 cycles.")

    def tick_cooldowns(self):
        print(f"⏱️ tick_cooldowns called. Current cooldowns: {self.directional_cooldowns}, Coin cooldowns: {self.coin_cooldowns}")
        for direction in ('long', 'short'):
            cycles = self.directional_cooldowns.get(direction, 0)
            if cycles > 0:
                self.directional_cooldowns[direction] = cycles - 1
                print(f"⏱️ {direction.upper()} cooldown: {cycles} → {self.directional_cooldowns[direction]} cycles remaining")
                if self.directional_cooldowns[direction] == 0:
                    # Cooldown bittiğinde loss_streak_loss_usd ve consecutive_losses'i sıfırla
                    # Çünkü cooldown bir "reset" dönemi - yeni bir başlangıç yapıyoruz
                    if direction in self.directional_bias:
                        self.directional_bias[direction]['loss_streak_loss_usd'] = 0.0
                        self.directional_bias[direction]['consecutive_losses'] = 0
                    print(f"✅ Directional cooldown cleared for {direction.upper()} trades. Loss streak reset.")
        
        # Coin bazlı cooldown'ları azalt
        coins_to_remove = []
        for coin, cycles in self.coin_cooldowns.items():
            if cycles > 0:
                self.coin_cooldowns[coin] = cycles - 1
                print(f"⏱️ {coin} coin cooldown: {cycles} → {self.coin_cooldowns[coin]} cycles remaining")
                if self.coin_cooldowns[coin] == 0:
                    coins_to_remove.append(coin)
                    print(f"✅ Coin cooldown cleared for {coin}.")
        
        # Sıfırlanan coin cooldown'larını temizle
        for coin in coins_to_remove:
            del self.coin_cooldowns[coin]
        
        if self.relaxed_countertrend_cycles > 0:
            self.relaxed_countertrend_cycles -= 1
            if self.relaxed_countertrend_cycles == 0:
                print(f"✅ Relaxed counter-trend mode cleared.")
        if self.counter_trend_cooldown > 0:
            self.counter_trend_cooldown -= 1
            if self.counter_trend_cooldown == 0:
                print(f"✅ Counter-trend cooldown cleared.")

    def get_trend_following_strength(self, coin: str, signal: str) -> Dict[str, Any]:
        """
        Hibrit yaklaşım: Trend-following gücünü 15m dahil olarak belirler
        (Confidence ve margin ayarlaması YOK, sadece bilgilendirme amaçlı)
        
        Returns:
            {
                'strength': 'STRONG' | 'MEDIUM_15M' | 'MEDIUM_3M' | 'WEAK' | None,
                'alignment_info': str,
                'trends': {
                    '1h': str,
                    '15m': str,
                    '3m': str
                }
            }
        """
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_15m = self.market_data.get_technical_indicators(coin, '15m')
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            
            if 'error' in indicators_htf or 'error' in indicators_15m or 'error' in indicators_3m:
                return None
            
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            price_15m = indicators_15m.get('current_price')
            ema20_15m = indicators_15m.get('ema_20')
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            
            # Validasyon
            if not all(isinstance(x, (int, float)) for x in [price_htf, ema20_htf, price_15m, ema20_15m, price_3m, ema20_3m]):
                return None
            
            # Trend yönleri
            trend_1h = "BULLISH" if price_htf > ema20_htf else "BEARISH"
            trend_15m = "BULLISH" if price_15m > ema20_15m else "BEARISH"
            trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
            
            # Sinyal yönü
            signal_direction = "BULLISH" if signal == 'buy_to_enter' else "BEARISH"
            
            # Counter-trend kontrolü: 1h ile sinyal zıt ise None döndür
            if trend_1h != signal_direction:
                return None
            
            # Trend-following güç seviyesi belirleme
            if trend_1h == trend_15m == trend_3m == signal_direction:
                # STRONG: 1h + 15m + 3m hepsi aynı yönde
                return {
                    'strength': 'STRONG',
                    'alignment_info': f"Perfect alignment: {HTF_LABEL}+15m+3m all {signal_direction}",
                    'trends': {
                        '1h': trend_1h,
                        '15m': trend_15m,
                        '3m': trend_3m
                    }
                }
            elif trend_1h == trend_15m == signal_direction:
                # MEDIUM_15M: 1h + 15m aynı (3m farklı)
                return {
                    'strength': 'MEDIUM_15M',
                    'alignment_info': f"Strong: {HTF_LABEL}+15m {signal_direction} (3m {trend_3m})",
                    'trends': {
                        '1h': trend_1h,
                        '15m': trend_15m,
                        '3m': trend_3m
                    }
                }
            elif trend_1h == trend_3m == signal_direction:
                # MEDIUM_3M: 1h + 3m aynı (15m farklı)
                return {
                    'strength': 'MEDIUM_3M',
                    'alignment_info': f"Moderate: {HTF_LABEL}+3m {signal_direction} (15m {trend_15m})",
                    'trends': {
                        '1h': trend_1h,
                        '15m': trend_15m,
                        '3m': trend_3m
                    }
                }
            else:
                # WEAK: Sadece 1h aynı (15m ve 3m farklı)
                return {
                    'strength': 'WEAK',
                    'alignment_info': f"Weak: Only {HTF_LABEL} {signal_direction} (15m {trend_15m}, 3m {trend_3m})",
                    'trends': {
                        '1h': trend_1h,
                        '15m': trend_15m,
                        '3m': trend_3m
                    }
                }
                
        except Exception as e:
            print(f"⚠️ Trend-following strength detection error for {coin}: {e}")
            return None

    def apply_directional_bias(self, signal: str, confidence: float, bias_metrics: Dict[str, Dict[str, Any]], current_trend: str) -> float:
        side = 'long' if signal == 'buy_to_enter' else 'short'
        stats = bias_metrics.get(side)
        if not stats:
            return confidence

        original_confidence = confidence
        adjusted_confidence = confidence

        # Hafifletilmiş caution penalty: 0.8 yerine 0.95
        if stats.get('caution_active'):
            adjusted_confidence = max(adjusted_confidence * 0.95, adjusted_confidence - 0.03)

        trend_lower = current_trend.lower() if isinstance(current_trend, str) else 'unknown'

        if trend_lower == 'neutral':
            adjusted_confidence *= Config.DIRECTIONAL_NEUTRAL_MULTIPLIER
        elif trend_lower == 'bullish':
            if side == 'long':
                adjusted_confidence *= Config.DIRECTIONAL_BULLISH_LONG_MULTIPLIER
            elif side == 'short':
                adjusted_confidence *= Config.DIRECTIONAL_BULLISH_SHORT_MULTIPLIER
        elif trend_lower == 'bearish':
            if side == 'long':
                adjusted_confidence *= Config.DIRECTIONAL_BEARISH_LONG_MULTIPLIER
            elif side == 'short':
                adjusted_confidence *= Config.DIRECTIONAL_BEARISH_SHORT_MULTIPLIER

        # Hafifletilmiş rolling avg penalty: 0.93 yerine 0.97
        rolling_avg = stats.get('rolling_avg', 0.0)
        if rolling_avg < 0:
            adjusted_confidence = max(adjusted_confidence * 0.97, adjusted_confidence - 0.02)

        # Minimum confidence floor: Orijinal değerin %90'ı altına düşmesin
        min_floor = original_confidence * 0.90
        adjusted_confidence = max(adjusted_confidence, min_floor, Config.MIN_CONFIDENCE)

        return adjusted_confidence

    def get_directional_bias_metrics(self) -> Dict[str, Dict[str, Any]]:
        metrics = {}
        for side, stats in self.directional_bias.items():
            rolling_list = list(stats['rolling'])
            rolling_sum = sum(rolling_list)
            rolling_avg = (rolling_sum / len(rolling_list)) if rolling_list else 0.0
            
            # Calculate win rate based on profit/loss amounts (not trade counts)
            # Win Rate = Total Profit / (|Total Profit| + |Total Loss|) * 100
            total_profit = sum(pnl for pnl in rolling_list if pnl > 0)
            total_loss = abs(sum(pnl for pnl in rolling_list if pnl < 0))
            
            if total_profit + total_loss > 0:
                win_rate = (total_profit / (total_profit + total_loss)) * 100
            else:
                win_rate = 0.0
            
            metrics[side] = {
                'net_pnl': stats['net_pnl'],
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': win_rate,  # Added win_rate based on profit/loss amounts
                'rolling_sum': rolling_sum,
                'rolling_avg': rolling_avg,
                'consecutive_losses': stats['consecutive_losses'],
                'consecutive_wins': stats.get('consecutive_wins', 0),
                'caution_active': stats.get('caution_active', False),
                'caution_win_progress': stats.get('caution_win_progress', 0)
            }
        return metrics

    def update_trend_state(
        self,
        coin: str,
        indicators_htf: Dict[str, Any],
        indicators_3m: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        price_htf = indicators_htf.get('current_price')
        ema20_htf = indicators_htf.get('ema_20')

        if not isinstance(price_htf, (int, float)) or not isinstance(ema20_htf, (int, float)) or ema20_htf == 0:
            return {'trend': 'unknown', 'recent_flip': False, 'last_flip_cycle': None}

        delta = (price_htf - ema20_htf) / ema20_htf
        price_neutral = abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT
        current_trend = 'neutral' if price_neutral else ('bullish' if delta > 0 else 'bearish')

        if indicators_3m and isinstance(indicators_3m, dict) and 'error' not in indicators_3m:
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20', price_3m)
            rsi_3m = indicators_3m.get('rsi_14', indicators_3m.get('rsi_7', 50))

            if isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)) and isinstance(rsi_3m, (int, float)):
                intraday_trend = 'bullish' if price_3m >= ema20_3m else 'bearish'
                if current_trend == 'bearish' and intraday_trend == 'bullish' and rsi_3m >= Config.INTRADAY_NEUTRAL_RSI_HIGH:
                    current_trend = 'neutral'
                elif current_trend == 'bullish' and intraday_trend == 'bearish' and rsi_3m <= Config.INTRADAY_NEUTRAL_RSI_LOW:
                    current_trend = 'neutral'

                if current_trend == 'neutral':
                    if price_htf <= ema20_htf and price_3m <= ema20_3m and rsi_3m <= Config.TREND_SHORT_RSI_THRESHOLD:
                        current_trend = 'bearish'
                    elif price_htf >= ema20_htf and price_3m >= ema20_3m and rsi_3m >= Config.TREND_LONG_RSI_THRESHOLD:
                        current_trend = 'bullish'

        record = self.trend_state.get(coin, {'trend': current_trend, 'last_flip_cycle': self.current_cycle_number, 'last_flip_direction': current_trend})
        previous_trend = record.get('trend', current_trend)
        recent_flip = False

        if previous_trend != current_trend:
            record['trend'] = current_trend
            if current_trend != 'neutral':
                record['last_flip_cycle'] = self.current_cycle_number
                record['last_flip_direction'] = current_trend
                recent_flip = True
        else:
            last_flip_cycle = record.get('last_flip_cycle', self.current_cycle_number)
            if current_trend != 'neutral' and self.current_cycle_number - last_flip_cycle <= self.trend_flip_cooldown:
                recent_flip = True

        record['last_seen_cycle'] = self.current_cycle_number
        self.trend_state[coin] = record
        return {
            'trend': current_trend,
            'recent_flip': recent_flip,
            'last_flip_cycle': record.get('last_flip_cycle'),
            'last_flip_direction': record.get('last_flip_direction')
        }

    def get_recent_trend_flip_summary(self) -> List[str]:
        summaries = []
        guard_window = self.trend_flip_cooldown
        history_window = max(guard_window, getattr(self, 'trend_flip_history_window', guard_window))
        entries: List[Tuple[int, str]] = []
        for coin, record in self.trend_state.items():
            last_flip_cycle = record.get('last_flip_cycle')
            if last_flip_cycle is None:
                continue
            cycles_ago = self.current_cycle_number - last_flip_cycle
            if cycles_ago < 0 or cycles_ago > history_window:
                continue
            trend_label = record.get('trend', 'unknown').upper()
            if cycles_ago <= guard_window:
                status = "GUARD"
            else:
                status = "RECENT"
            if cycles_ago == 0:
                cycles_text = "current cycle"
            elif cycles_ago == 1:
                cycles_text = "1 cycle ago"
            else:
                cycles_text = f"{cycles_ago} cycles ago"
            direction_note = record.get('last_flip_direction', trend_label)
            entries.append((cycles_ago, f"{coin}: {direction_note} since cycle {last_flip_cycle} ({status}, {cycles_text})"))
        entries.sort(key=lambda x: x[0])
        summaries = [text for _, text in entries]
        return summaries

    def load_cycle_history(self) -> List[Dict]:
        history = safe_file_read(self.cycle_history_file, default_data=[]); print(f"✅ Loaded {len(history)} cycles."); return history
    def add_to_cycle_history(
        self,
        cycle_number: int,
        prompt: str,
        thoughts: str,
        decisions: Dict,
        status: str = "ai_decision",
        metadata: Optional[Dict[str, Any]] = None
    ):
        prompt_summary = "N/A"
        if isinstance(prompt, str) and prompt not in (None, "N/A"):
            # For JSON prompts, try to extract a meaningful summary
            # Check for all JSON sections
            json_sections = [
                ("COUNTER_TRADE_ANALYSIS (JSON):", "Counter-trade analysis"),
                ("TREND_REVERSAL_DATA (JSON):", "Trend reversal"),
                ("ENHANCED_CONTEXT (JSON):", "Enhanced context"),
                ("DIRECTIONAL_BIAS (JSON):", "Directional bias"),
                ("COOLDOWN_STATUS (JSON):", "Cooldown status"),
                ("TREND_FLIP_GUARD (JSON):", "Trend flip guard"),
                ("POSITION_SLOTS (JSON):", "Position slots"),
                ("MARKET_DATA (JSON):", "Market data"),
                ("HISTORICAL_CONTEXT (JSON):", "Historical context"),
                ("RISK_STATUS (JSON):", "Risk status"),
                ("PORTFOLIO (JSON):", "Portfolio")
            ]
            
            found_sections = [name for marker, name in json_sections if marker in prompt]
            
            if found_sections:
                # JSON format prompt - create a structured summary
                try:
                    section_count = len(found_sections)
                    if section_count <= 3:
                        summary_text = ", ".join(found_sections)
                    else:
                        summary_text = f"{', '.join(found_sections[:3])} + {section_count - 3} more"
                    prompt_summary = f"JSON Format ({section_count} sections): {summary_text} | " + prompt[:200] + "..."
                except:
                    prompt_summary = prompt[:300] + "..." if len(prompt) > 300 else prompt
            else:
                # Text format prompt - use original truncation
                prompt_summary = prompt[:300] + "..." if len(prompt) > 300 else prompt
        
        # Add cooldown information to cycle data
        cooldown_info = {
            'directional_cooldowns': dict(self.directional_cooldowns),
            'relaxed_countertrend_cycles': self.relaxed_countertrend_cycles,
            'counter_trend_cooldown': self.counter_trend_cooldown,
            'coin_cooldowns': dict(self.coin_cooldowns)
        }
        
        cycle_data = {
            'cycle': cycle_number,
            'timestamp': datetime.now().isoformat(),
            'user_prompt_summary': prompt_summary,
            'chain_of_thoughts': thoughts,
            'decisions': decisions,
            'status': status,
            'cooldown_status': cooldown_info  # Always include cooldown status
        }
        if metadata:
            cycle_data['metadata'] = metadata
        self.cycle_history.append(cycle_data); self.cycle_history = self.cycle_history[-self.max_cycle_history:]
        safe_file_write(self.cycle_history_file, self.cycle_history); print(f"✅ Saved cycle {cycle_number} (Total: {len(self.cycle_history)})")
    def update_prices(self, new_prices: Dict[str, float], increment_loss_counters: bool = True):
        """Updates prices and recalculates total value."""
        total_unrealized_pnl = 0.0
        for coin, price in new_prices.items():
            if coin in self.positions and isinstance(price, (int, float)) and price > 0:
                pos = self.positions[coin]; pos['current_price'] = price
                entry = pos['entry_price']; qty = pos['quantity']; direction = pos.get('direction', 'long')
                pnl = (price - entry) * qty if direction == 'long' else (entry - price) * qty
                pos['unrealized_pnl'] = pnl; total_unrealized_pnl += pnl
                if increment_loss_counters:
                    direction = pos.get('direction', 'unknown')
                    if pnl <= 0:
                        pos['loss_cycle_count'] = pos.get('loss_cycle_count', 0) + 1
                        new_count = pos['loss_cycle_count']
                        if new_count in (5, 8, 10):
                            print(f"⏳ LOSS CYCLE WATCH: {coin} {direction} negative for {new_count} cycles (PnL ${pnl:.2f}).")
                    else:
                        pos['loss_cycle_count'] = 0
            elif coin in self.positions: print(f"⚠️ Invalid price for {coin}: {price}. PnL skip.")

        # Calculate total value correctly
        # In live mode, prefer syncing from Binance (done in sync_live_account)
        # In simulation mode or when Binance data unavailable, calculate manually
        if not self.positions:
            # No positions, total value = available cash (or totalWalletBalance if available in live mode)
            if self.is_live_trading and self.order_executor and self.order_executor.is_live():
                # Try to get totalWalletBalance from Binance
                try:
                    overview = self.order_executor.get_account_overview()
                    if overview and overview.get("totalWalletBalance") and overview.get("totalWalletBalance") > 0:
                        self.total_value = float(overview["totalWalletBalance"])
                    else:
                        self.total_value = self.current_balance
                except:
                    self.total_value = self.current_balance
            else:
                self.total_value = self.current_balance
        else:
            # With positions: Calculate margin used (for cross margin, calculate from notional/leverage)
            total_margin_used = 0.0
            total_unrealized_pnl = 0.0
            
            for pos in self.positions.values():
                # Get unrealized PnL
                pnl = pos.get('unrealized_pnl', 0.0)
                if isinstance(pnl, (int, float)):
                    total_unrealized_pnl += pnl
                
                # Get margin (for cross margin, margin_usd might be 0, so calculate from notional/leverage)
                margin = pos.get('margin_usd', 0.0)
                if margin <= 0:
                    notional = pos.get('notional_usd', 0.0)
                    leverage = pos.get('leverage', 1)
                    if notional > 0 and leverage > 0:
                        margin = notional / leverage
                if isinstance(margin, (int, float)) and margin > 0:
                    total_margin_used += margin
            
            # Total value = Available cash + Margin used + Unrealized PnL
            # This works for both live and simulation mode
            # Unrealized PnL must be included to reflect true portfolio value
            self.total_value = self.current_balance + total_margin_used + total_unrealized_pnl

        if self.initial_balance > 0: self.total_return = ((self.total_value - self.initial_balance) / self.initial_balance) * 100
        else: self.total_return = 0.0
        
        # Update portfolio history for Sharpe ratio calculation
        self.portfolio_values_history.append(self.total_value)
        if len(self.portfolio_values_history) > 100:  # Keep last 100 values
            self.portfolio_values_history = self.portfolio_values_history[-100:]
        
        # Calculate Sharpe ratio
        self.sharpe_ratio = self.calculate_sharpe_ratio()
        
        # Save updated state with Sharpe ratio
        self.save_state()
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio based on portfolio value history (Nof1ai blog style)."""
        if len(self.portfolio_values_history) < 2:
            return 0.0
        
        try:
            # Calculate simple returns (percentage changes)
            returns = []
            for i in range(1, len(self.portfolio_values_history)):
                if self.portfolio_values_history[i-1] > 0:
                    ret = (self.portfolio_values_history[i] - self.portfolio_values_history[i-1]) / self.portfolio_values_history[i-1]
                    returns.append(ret)
            
            if len(returns) < 2:
                return 0.0
            
            # Nof1ai style: Simple Sharpe ratio with 0% risk-free rate
            # Daily Sharpe ratio (assuming 2-minute cycles = 720 cycles per day)
            risk_free_rate = 0.0
            
            # Calculate excess returns
            excess_returns = [r - risk_free_rate for r in returns]
            
            # Daily return and volatility
            avg_return = np.mean(excess_returns) * 720  # Daily return (720 cycles per day)
            std_return = np.std(excess_returns) * np.sqrt(720)  # Daily volatility
            
            if std_return == 0:
                return 0.0
            
            # Daily Sharpe ratio
            sharpe = avg_return / std_return
            
            # Return as float (not annualized for simplicity)
            return float(sharpe)
            
        except Exception as e:
            print(f"⚠️ Sharpe ratio calculation error: {e}")
            return 0.0

    def get_manual_override(self) -> Dict:
        """Checks for and deletes the manual override file."""
        override_data = safe_file_read(self.override_file, default_data={})
        if override_data:
            print(f"🔔 MANUAL OVERRIDE DETECTED: {override_data}")
            try: os.remove(self.override_file); print(f"ℹ️ Override file deleted.")
            except OSError as e: print(f"❌ Could not delete override file: {e}")
        return override_data

    def _estimate_liquidation_price(self, entry_price: float, leverage: int, direction: str) -> float:
        """Estimate liquidation price."""
        if leverage <= 1 or entry_price <= 0: return 0.0
        imr = 1.0 / leverage; mmr = self.maintenance_margin_rate; margin_diff = imr - mmr
        if margin_diff <= 0: print(f"⚠️ Liq est. failed: margin diff <= 0 ({margin_diff})."); return 0.0
        liq_price = entry_price * (1 - margin_diff) if direction == 'long' else entry_price * (1 + margin_diff)
        return max(0.0, liq_price)

    # --- NEW: Enhanced Auto TP/SL Check with Advanced Exit Strategies ---
    def check_and_execute_tp_sl(self, current_prices: Dict[str, float]):
        """Checks if any open position hit TP or SL and closes them automatically with enhanced exit strategies.
        
        This function is called every 30 seconds by the monitoring loop:
        - All TP/SL decisions are made by this monitoring (like simulation mode)
        - No Binance TP/SL orders - all managed by this loop
        - Kademeli margin-based stop loss is checked and positions are closed accordingly
        """
        # Enhanced exit strategy control - check if enabled
        if hasattr(self, 'bot') and not self.bot.enhanced_exit_enabled:
            print("⏸️ Enhanced exit strategy paused during cycle")
            return False
            
        # All TP/SL decisions made by 30-second monitoring (like simulation mode)
        # No Binance TP/SL orders - all managed by monitoring loop
        print(f"🔎 Checking for TP/SL triggers (30-second monitoring mode)")
        
        closed_positions = [] # Keep track of positions closed in this check
        updated_stops = [] # Track positions with updated trailing stops
        state_changed = False
        
        for coin, position in list(self.positions.items()): # Iterate over a copy for safe deletion
            if coin not in current_prices or not isinstance(current_prices[coin], (int, float)) or current_prices[coin] <= 0:
                continue # Skip if price is invalid

            current_price = current_prices[coin]
            exit_plan = position.get('exit_plan', {})
            tp = exit_plan.get('profit_target')
            sl = exit_plan.get('stop_loss')
            direction = position.get('direction', 'long')
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            # Calculate margin_used properly - try multiple fallback methods
            margin_used = position.get('margin_usd')
            if margin_used is None or margin_used <= 0:
                # Fallback 1: Calculate from notional and leverage
                notional = position.get('notional_usd', 0)
                leverage = position.get('leverage', 1)
                if notional > 0 and leverage > 0:
                    margin_used = notional / leverage
                else:
                    # Fallback 2: Calculate from entry_price and quantity
                    if entry_price > 0 and quantity > 0:
                        notional = entry_price * quantity
                        leverage = position.get('leverage', 10)
                        margin_used = notional / leverage
                    else:
                        margin_used = 0
            
            # Debug log if margin_used is still 0
            if margin_used <= 0:
                print(f"⚠️ Warning: margin_used is 0 for {coin}. Position data: margin_usd={position.get('margin_usd')}, notional={position.get('notional_usd')}, leverage={position.get('leverage')}, entry={entry_price}, qty={quantity}")

            close_reason = None

            # Check TP
            # Convert tp/sl to float for safe comparison, handle potential errors
            try: tp = float(tp) if tp is not None else None
            except (ValueError, TypeError): tp = None
            try: sl = float(sl) if sl is not None else None
            except (ValueError, TypeError): sl = None

            # Enhanced exit strategy check - REAL-TIME ENTEGRASYON
            exit_decision = self.enhanced_exit_strategy(position, current_price)
            
            # Handle enhanced exit strategy signals - ANINDA İŞLEME
            if exit_decision['action'] == 'close_position':
                # Enhanced exit strategy wants to close the position completely
                close_reason = exit_decision['reason']
                print(f"⚡ ENHANCED EXIT CLOSE {coin} ({direction}): {close_reason} at price ${format_num(current_price, 4)}")
                state_changed = True
            elif exit_decision['action'] == 'partial_close':
                # Partial profit taking - ANINDA İŞLEME
                close_percent = exit_decision['percent']
                if self.is_live_trading:
                    live_result = self.execute_live_partial_close(
                        coin=coin,
                        position=position,
                        close_percent=close_percent,
                        current_price=current_price,
                        reason=exit_decision['reason']
                    )
                    if not live_result.get('success'):
                        print(f"🚫 Live partial close failed for {coin}: {live_result.get('error', 'unknown_error')}")
                        continue
                    history_entry = live_result.get('history_entry')
                    if history_entry:
                        self.add_to_history(history_entry)
                    print(f"⚡ PARTIAL CLOSE {coin} ({direction}) [LIVE]: {exit_decision['reason']} ({close_percent*100:.0f}% / PnL ${format_num(live_result.get('pnl', 0), 2)})")
                    state_changed = True
                    # Sync account balance after partial close in live mode
                    try:
                        self.sync_live_account()
                        print(f"✅ Account balance synced after partial close of {coin}")
                    except Exception as sync_exc:
                        print(f"⚠️ Failed to sync account after partial close: {sync_exc}")
                    continue

                close_quantity = quantity * close_percent
                
                if direction == 'long': profit = (current_price - entry_price) * close_quantity
                else: profit = (entry_price - current_price) * close_quantity
                
                # Update position quantity
                position['quantity'] = quantity * (1 - close_percent)
                position['margin_usd'] = margin_used * (1 - close_percent)
                position['notional_usd'] = position['notional_usd'] * (1 - close_percent)
                
                # Add profit to balance
                self.current_balance += (margin_used * close_percent + profit)
                
                print(f"⚡ PARTIAL CLOSE {coin} ({direction}): {exit_decision['reason']} - Closed {close_percent*100}% at price ${format_num(current_price, 4)}")
                print(f"   Partial PnL: ${format_num(profit, 2)}")
                
                history_entry = {
                    "symbol": coin, "direction": direction, "entry_price": entry_price, "exit_price": current_price,
                    "quantity": close_quantity, "notional_usd": position.get('notional_usd', 'N/A') * close_percent, 
                    "pnl": profit, "entry_time": position['entry_time'], "exit_time": datetime.now().isoformat(),
                    "leverage": position.get('leverage', 'N/A'), "close_reason": exit_decision['reason']
                }
                self.add_to_history(history_entry)
                state_changed = True
                continue  # Continue with remaining position
            
            elif exit_decision['action'] == 'update_stop':
                # Update trailing stop - ANINDA GÜNCELLEME
                updated_stops.append(coin)
                new_stop = exit_decision['new_stop']
                exit_plan['stop_loss'] = new_stop
                print(f"📈 TRAILING STOP UPDATE {coin}: New stop at ${format_num(new_stop, 4)}")
                
                # No Binance orders - stop loss updated in exit_plan, will be monitored by 30-second loop
                
                state_changed = True
                continue
            
            # Traditional TP/SL checks (only if no enhanced exit triggered)
            if close_reason is None and tp is not None:
                if direction == 'long' and current_price >= tp: close_reason = f"Profit Target ({tp}) hit"
                elif direction == 'short' and current_price <= tp: close_reason = f"Profit Target ({tp}) hit"

            # Check SL (only if TP not hit)
            # First check exit_plan stop_loss, then fallback to margin-based kademeli stop loss
            if close_reason is None:
                # Check exit_plan stop_loss first
                if sl is not None:
                    if direction == 'long' and current_price <= sl: 
                        close_reason = f"Stop Loss ({sl}) hit"
                    elif direction == 'short' and current_price >= sl: 
                        close_reason = f"Stop Loss ({sl}) hit"
                
                # If no exit_plan stop_loss or it didn't trigger, check margin-based kademeli stop loss
                # Only check if margin_used is valid (> 0)
                if close_reason is None and quantity > 0 and margin_used > 0:
                    # Calculate margin-based stop loss using kademeli loss cutting (same as entry)
                    loss_multiplier = 0.03  # Default: %3 for margin >= 50
                    if margin_used < 30:
                        loss_multiplier = 0.07  # %7 for margin < 30
                    elif margin_used < 40:
                        loss_multiplier = 0.05  # %5 for margin 30-40
                    elif margin_used < 50:
                        loss_multiplier = 0.05  # %5 for margin 40-50
                    else:
                        loss_multiplier = 0.03  # %3 for margin >= 50
                    
                    loss_threshold_usd = margin_used * loss_multiplier
                    
                    # Only proceed if loss_threshold_usd is valid
                    if loss_threshold_usd > 0:
                        # Calculate stop loss price from loss threshold
                        if direction == 'long':
                            margin_based_stop_loss = entry_price - (loss_threshold_usd / quantity)
                        else:  # short
                            margin_based_stop_loss = entry_price + (loss_threshold_usd / quantity)
                        
                        # Kademeli stop loss is calculated correctly based on margin and loss_multiplier
                        # No minimum distance adjustment needed
                        
                        # Check if current price hit margin-based stop loss
                        if direction == 'long' and current_price <= margin_based_stop_loss:
                            close_reason = f"Margin-based Stop Loss ({format_num(margin_based_stop_loss, 4)}) hit (${loss_threshold_usd:.2f} loss limit, {loss_multiplier*100:.1f}% of ${margin_used:.2f} margin)"
                        elif direction == 'short' and current_price >= margin_based_stop_loss:
                            close_reason = f"Margin-based Stop Loss ({format_num(margin_based_stop_loss, 4)}) hit (${loss_threshold_usd:.2f} loss limit, {loss_multiplier*100:.1f}% of ${margin_used:.2f} margin)"

            # Execute Close if triggered
            if close_reason:
                print(f"⚡ AUTO-CLOSE {coin} ({direction}): {close_reason} at price ${format_num(current_price, 4)}")

                if self.is_live_trading:
                    print(f"🔄 Executing LIVE close on Binance for {coin}...")
                    live_result = self.execute_live_close(
                        coin=coin,
                        position=position,
                        current_price=current_price,
                        reason=close_reason
                    )
                    if not live_result.get('success'):
                        print(f"🚫 Live auto-close failed for {coin}: {live_result.get('error', 'unknown_error')}")
                        continue
                    
                    # Log Binance order details
                    order_id = live_result.get('order', {}).get('orderId')
                    executed_qty = live_result.get('executed_qty', 0)
                    avg_price = live_result.get('avg_price', 0)
                    print(f"✅ Binance CLOSE order executed for {coin}: orderId={order_id}, qty={format_num(executed_qty, 4)}, avgPrice=${format_num(avg_price, 4)}")
                    
                    history_entry = live_result.get('history_entry')
                    if history_entry:
                        self.add_to_history(history_entry)
                    print(f"   Live Closed PnL: ${format_num(live_result.get('pnl', 0), 2)}")
                    closed_positions.append(coin)
                    state_changed = True
                    # Sync account balance after closing position in live mode
                    try:
                        self.sync_live_account()
                        print(f"✅ Account balance synced after closing {coin}")
                    except Exception as sync_exc:
                        print(f"⚠️ Failed to sync account after close: {sync_exc}")
                    continue

                if direction == 'long': profit = (current_price - entry_price) * quantity
                else: profit = (entry_price - current_price) * quantity

                self.current_balance += (margin_used + profit) # Return margin + PnL

                print(f"   Closed PnL: ${format_num(profit, 2)}")

                history_entry = {
                    "symbol": coin, "direction": direction, "entry_price": entry_price, "exit_price": current_price,
                    "quantity": quantity, "notional_usd": position.get('notional_usd', 'N/A'), "pnl": profit,
                    "entry_time": position['entry_time'], "exit_time": datetime.now().isoformat(),
                    "leverage": position.get('leverage', 'N/A'), "close_reason": close_reason # Add reason
                }
                self.add_to_history(history_entry) # This increments trade_count
                closed_positions.append(coin)
                del self.positions[coin] # Remove from active positions
                state_changed = True

        if closed_positions:
             print(f"✅ Auto-closed positions: {', '.join(closed_positions)}")
        if updated_stops:
             print(f"📈 Updated trailing stops: {', '.join(updated_stops)}")

        if state_changed:
            self.save_state()
             
        return len(closed_positions) > 0  # Indicate if any positions were closed

    def calculate_dynamic_position_size(self, coin: str, confidence: float, market_regime: str, trend_strength: int) -> float:
        """Calculate dynamic position size based on multiple factors"""
        base_risk = 25.0  # Reduced maximum risk to $25
        
        # Confidence factor
        confidence_multiplier = confidence
        
        # Market regime factor
        if market_regime == "BULLISH":
            regime_multiplier = 1.2
        elif market_regime == "BEARISH":
            regime_multiplier = 0.8
        else:
            regime_multiplier = 1.0
        
        # Trend strength factor
        trend_multiplier = 1.0 + (trend_strength * 0.1)
        
        # Volume consideration
        try:
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            volume = indicators_3m.get('volume', 0)
            avg_volume = indicators_3m.get('avg_volume', 0)
            
            # Volume multiplier: higher volume = higher confidence
            if volume > avg_volume * 2:
                volume_multiplier = 1.2
            elif volume > avg_volume:
                volume_multiplier = 1.1
            else:
                volume_multiplier = 0.8  # Penalize low volume
        except:
            volume_multiplier = 1.0
        
        # Dynamic risk calculation
        dynamic_risk = base_risk * confidence_multiplier * regime_multiplier * trend_multiplier * volume_multiplier
        
        # Maximum risk limit
        return min(dynamic_risk, 25.0)

    def get_profit_levels_by_notional(self, notional_usd: float) -> Dict[str, float]:
        """Get dynamic profit levels based on notional size"""
        if notional_usd < 200:
            # Small positions: aggressive profit taking
            return {
                'level1': 0.006,  # %0.7
                'level2': 0.007,  # %0.9
                'level3': 0.08,  # %1.1
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        elif notional_usd < 300:
            # Medium positions: balanced profit taking
            return {
                'level1': 0.005,  # %0.7
                'level2': 0.006,  # %0.9
                'level3': 0.007,  # %1.1
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        elif notional_usd < 400:
            # Large positions: conservative profit taking
            return {
                'level1': 0.004,  # %0.6
                'level2': 0.005,  # %0.8
                'level3': 0.006,  # %1.0
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        elif notional_usd < 500:
            # xLarge positions: conservative profit taking
            return {
                'level1': 0.003,  # %0.5
                'level2': 0.004,  # %0.7
                'level3': 0.005,  # %0.9
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        elif notional_usd < 600:
            # xxLarge positions: conservative profit taking
            return {
                'level1': 0.002,  # %0.
                'level2': 0.003,  # %0.6
                'level3': 0.004,  # %0.8
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        else:
            # Very large positions: very conservative profit taking
            return {
                'level1': 0.002,  # %0.3
                'level2': 0.004,  # %0.5
                'level3': 0.006,  # %0.7
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }

    def get_dynamic_stop_loss_percentage(self, total_portfolio_value: float) -> float:
        """Get dynamic stop-loss percentage based on portfolio value"""
        if total_portfolio_value < 300:
            return 0.07  # %1.0
        elif total_portfolio_value < 400:
            return 0.006 # %0.8
        elif total_portfolio_value < 500:
            return 0.005 # %0.7
        else:
            return 0.004 # %0.5

    def enhanced_exit_strategy(self, position: Dict, current_price: float) -> Dict[str, Any]:
        """Enhanced exit strategy with dynamic profit taking and KADEMELİ loss cutting"""
        entry_price = position.get('entry_price')
        if entry_price is None:
            entry_price = position.get('current_price', 0)
            position['entry_price'] = entry_price
        direction = position.get('direction', 'long')
        exit_plan = self._ensure_exit_plan(position, position.get('exit_plan'))
        stop_loss = exit_plan.get('stop_loss')
        profit_target = exit_plan.get('profit_target')
        notional_usd = position.get('notional_usd', 0)
        
        exit_decision = {"action": "hold", "reason": "No exit trigger"}
        
        current_margin = position.get('margin_usd', 0)
        margin_used = position.get('margin_usd', position.get('notional_usd', 0) / max(position.get('leverage', 1), 1))
        loss_cycle_count = position.get('loss_cycle_count', 0)
        unrealized_pnl = position.get('unrealized_pnl', 0)
        if loss_cycle_count >= 10 and unrealized_pnl <= 0:
            reason = f"Position negative for {loss_cycle_count} cycles"
            print(f"⏳ Extended loss exit: {position['symbol']} {direction} closed ({reason}).")
            return {"action": "close_position", "reason": reason}
        
        # --- KADEMELİ LOSS CUTTING MEKANİZMASI (Margin tabanlı) ---
        loss_multiplier = 0.03  # Default: %3 for margin >= 50
        if margin_used < 30:
            loss_multiplier = 0.07  # %7 for margin < 30
        elif margin_used < 40:
            loss_multiplier = 0.05  # %5 for margin 30-40
        elif margin_used < 50:
            loss_multiplier = 0.05  # %5 for margin 40-50
        else:
            loss_multiplier = 0.03  # %3 for margin >= 50

        loss_threshold_usd = margin_used * loss_multiplier
        
        if direction == 'long':
            unrealized_loss_usd = max(0.0, (entry_price - current_price) * position['quantity'])
        else:
            unrealized_loss_usd = max(0.0, (current_price - entry_price) * position['quantity'])

        if unrealized_loss_usd >= loss_threshold_usd and loss_threshold_usd > 0:
            print(f"🛑 KADEMELİ LOSS CUTTING: {direction} {position['symbol']} ${unrealized_loss_usd:.2f} zarar (eşik: ${loss_threshold_usd:.2f}). Pozisyon kapatılıyor.")
            return {"action": "close_position", "reason": f"Margin-based loss cut ${unrealized_loss_usd:.2f} ≥ ${loss_threshold_usd:.2f}"}
        
        # Get dynamic profit levels based on notional size
        profit_levels = self.get_profit_levels_by_notional(notional_usd)
        level1 = profit_levels['level1']
        level2 = profit_levels['level2']
        level3 = profit_levels['level3']
        take1 = profit_levels['take1']
        take2 = profit_levels['take2']
        take3 = profit_levels['take3']
        
        print(f"📊 Dynamic profit levels for ${notional_usd:.2f} notional: {level1*100:.1f}%/{level2*100:.1f}%/{level3*100:.1f}%")
        
        if direction == 'long':
            unrealized_pnl_usd = max(0.0, (current_price - entry_price) * position['quantity'])
            unrealized_pnl_percent = (unrealized_pnl_usd / notional_usd) if notional_usd else 0.0
            
            # Dynamic Profit Taking Levels based on notional size
            if unrealized_pnl_percent >= level3:  # Level 3 profit - take 75%
                take_profit_percent = take3
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level3*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            elif unrealized_pnl_percent >= level2:  # Level 2 profit - take 50%
                take_profit_percent = take2
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level2*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            elif unrealized_pnl_percent >= level1:  # Level 1 profit - take 25%
                take_profit_percent = take1
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level1*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            
            trailing_action = self._evaluate_trailing_stop(
                position=position,
                current_price=current_price,
                profit_target=profit_target,
                direction=direction,
                entry_price=entry_price,
                unrealized_pnl_percent=unrealized_pnl_percent,
                profit_levels=profit_levels
            )
            if trailing_action:
                return trailing_action
        
        elif direction == 'short':
            unrealized_pnl_usd = max(0.0, (entry_price - current_price) * position['quantity'])
            unrealized_pnl_percent = (unrealized_pnl_usd / notional_usd) if notional_usd else 0.0
            
            # Dynamic Profit Taking Levels for shorts based on notional size
            if unrealized_pnl_percent >= level3:  # Level 3 profit - take 75%
                take_profit_percent = take3
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level3*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            elif unrealized_pnl_percent >= level2:  # Level 2 profit - take 50%
                take_profit_percent = take2
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level2*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            elif unrealized_pnl_percent >= level1:  # Level 1 profit - take 25%
                take_profit_percent = take1
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level1*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            
            trailing_action = self._evaluate_trailing_stop(
                position=position,
                current_price=current_price,
                profit_target=profit_target,
                direction=direction,
                entry_price=entry_price,
                unrealized_pnl_percent=unrealized_pnl_percent,
                profit_levels=profit_levels
            )
            if trailing_action:
                return trailing_action
        
        return exit_decision

    def _evaluate_trailing_stop(
        self,
        position: Dict[str, Any],
        current_price: float,
        profit_target: Optional[float],
        direction: str,
        entry_price: float,
        unrealized_pnl_percent: float,
        profit_levels: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate advanced trailing stop conditions based on progress, time, volume and ATR."""
        if unrealized_pnl_percent <= 0 or not isinstance(current_price, (int, float)) or current_price <= 0:
            return None

        symbol = position.get('symbol')
        exit_plan = position.get('exit_plan') or {}

        level1_threshold = 0.0
        if isinstance(profit_levels, dict):
            try:
                level1_threshold = float(profit_levels.get('level1', 0.0) or 0.0)
            except (TypeError, ValueError):
                level1_threshold = 0.0
        if unrealized_pnl_percent < max(level1_threshold * 0.5, 0.0):
            return None

        existing_stop = exit_plan.get('stop_loss')
        try:
            existing_stop = float(existing_stop) if existing_stop is not None else None
        except (TypeError, ValueError):
            existing_stop = None

        # Calculate progress toward profit target (in %)
        progress_pct = 0.0
        progress_valid = False
        if isinstance(profit_target, (int, float)) and profit_target > 0 and profit_target != entry_price:
            if direction == 'long':
                denominator = profit_target - entry_price
                if denominator > 0:
                    progress_pct = ((current_price - entry_price) / denominator) * 100
                    progress_valid = True
            elif direction == 'short':
                denominator = entry_price - profit_target
                if denominator > 0:
                    progress_pct = ((entry_price - current_price) / denominator) * 100
                    progress_valid = True
        progress_pct = max(0.0, min(progress_pct, 200.0))

        pnl_percent = max(0.0, unrealized_pnl_percent * 100.0)
        progress_score = progress_pct if progress_valid else pnl_percent

        # Time in trade (minutes)
        time_in_trade = 0.0
        entry_time_str = position.get('entry_time')
        if entry_time_str:
            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                time_in_trade = max(0.0, (datetime.now() - entry_time).total_seconds() / 60.0)
            except Exception:
                time_in_trade = 0.0

        progress_triggered = progress_score >= Config.TRAILING_PROGRESS_TRIGGER
        time_triggered = (
            time_in_trade >= Config.TRAILING_TIME_MINUTES
            and progress_score >= Config.TRAILING_TIME_PROGRESS_FLOOR
        )
        if not (progress_triggered or time_triggered):
            return None

        # Fetch current 3m indicators for ATR & volume context
        current_volume_ratio = None
        atr_value = None
        try:
            indicators_3m = self.market_data.get_technical_indicators(symbol, '3m') if self.market_data else {}
        except Exception as exc:
            print(f"⚠️ Trailing stop indicator fetch failed for {symbol}: {exc}")
            indicators_3m = {}

        if isinstance(indicators_3m, dict):
            volume_now = indicators_3m.get('volume')
            avg_volume_now = indicators_3m.get('avg_volume')
            if isinstance(volume_now, (int, float)) and isinstance(avg_volume_now, (int, float)) and avg_volume_now > 0:
                current_volume_ratio = volume_now / avg_volume_now
            atr_value = indicators_3m.get('atr_14')

        if not isinstance(atr_value, (int, float)) or atr_value <= 0:
            atr_value = position.get('entry_atr_14')
        if not isinstance(atr_value, (int, float)) or atr_value <= 0:
            atr_value = current_price * Config.TRAILING_FALLBACK_BUFFER_PCT

        entry_volume_ratio = position.get('entry_volume_ratio')
        volume_drop_triggered = False
        if isinstance(current_volume_ratio, (int, float)):
            if current_volume_ratio <= Config.TRAILING_VOLUME_ABSOLUTE_THRESHOLD:
                volume_drop_triggered = True
            elif isinstance(entry_volume_ratio, (int, float)) and entry_volume_ratio > 0:
                if current_volume_ratio <= entry_volume_ratio * Config.TRAILING_VOLUME_DROP_RATIO:
                    volume_drop_triggered = True

        min_improvement_abs = max(
            current_price * Config.TRAILING_MIN_IMPROVEMENT_PCT,
            max(Config.MIN_EXIT_PLAN_OFFSET, 1e-7)
        )
        atr_buffer = max(atr_value * Config.TRAILING_ATR_MULTIPLIER, min_improvement_abs)

        reason_tokens: List[str] = []
        if progress_triggered:
            reason_tokens.append(f"progress {progress_score:.1f}%")
        if time_triggered:
            reason_tokens.append(f"time {time_in_trade:.1f}m")
        if volume_drop_triggered and isinstance(current_volume_ratio, (int, float)):
            reason_tokens.append(f"volume {current_volume_ratio:.2f}x")

        if not reason_tokens:
            reason_tokens.append("trailing criteria met")

        new_stop: Optional[float] = None
        if direction == 'long':
            baseline_stop = current_price - atr_buffer
            baseline_stop = min(baseline_stop, current_price - min_improvement_abs)
            if progress_triggered:
                baseline_stop = max(baseline_stop, entry_price + min_improvement_abs)
            elif time_triggered:
                baseline_stop = max(baseline_stop, entry_price + Config.MIN_EXIT_PLAN_OFFSET)

            if existing_stop is not None:
                baseline_stop = max(baseline_stop, existing_stop + min_improvement_abs)

            if baseline_stop >= current_price:
                baseline_stop = current_price - min_improvement_abs

            if existing_stop is not None and baseline_stop <= existing_stop + min_improvement_abs:
                return None

            if baseline_stop <= 0:
                return None

            new_stop = baseline_stop
        else:  # short
            baseline_stop = current_price + atr_buffer
            baseline_stop = max(baseline_stop, current_price + min_improvement_abs)
            if progress_triggered:
                baseline_stop = min(baseline_stop, entry_price - min_improvement_abs)
            elif time_triggered:
                baseline_stop = min(baseline_stop, entry_price - Config.MIN_EXIT_PLAN_OFFSET)

            if existing_stop is not None:
                baseline_stop = min(baseline_stop, existing_stop - min_improvement_abs)

            if baseline_stop <= current_price:
                baseline_stop = current_price + min_improvement_abs

            if progress_triggered and baseline_stop > entry_price - min_improvement_abs:
                baseline_stop = entry_price - min_improvement_abs

            if baseline_stop <= current_price:
                return None

            if existing_stop is not None and (existing_stop - baseline_stop) <= min_improvement_abs:
                return None

            new_stop = baseline_stop

        if new_stop is None:
            return None

        new_stop = round(max(0.0, new_stop), 6)
        if direction == 'long' and new_stop >= current_price:
            return None
        if direction == 'short' and new_stop <= current_price:
            return None

        # Persist updated stop and trailing metadata
        exit_plan['stop_loss'] = new_stop
        position['exit_plan'] = exit_plan

        trailing_meta = position.setdefault('trailing', {})
        trailing_meta.update({
            'active': True,
            'last_update_cycle': getattr(self, 'current_cycle_number', None),
            'last_reason': ", ".join(reason_tokens),
            'last_stop': new_stop,
            'progress_percent': round(progress_score, 2),
            'time_in_trade_min': round(time_in_trade, 2)
        })
        if isinstance(current_volume_ratio, (int, float)):
            trailing_meta['last_volume_ratio'] = round(current_volume_ratio, 4)

        reason = f"Trailing stop tightened ({', '.join(reason_tokens)})"
        return {"action": "update_stop", "new_stop": new_stop, "reason": reason}

    def _execute_new_positions_only(
        self,
        decisions: Dict,
        valid_prices: Dict,
        cycle_number: int,
        indicator_cache: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Execute only new position entries after AI close_position signal"""
        print("🔄 Executing new positions only (after close_position signal)")
        
        # KADEMELİ POZİSYON SİSTEMİ: Cycle bazlı pozisyon limiti
        max_positions_for_cycle = self.get_max_positions_for_cycle(cycle_number)
        current_positions = len(self.positions)
        
        decisions_to_execute = {}
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                continue
                
            signal = trade.get('signal')
            if signal in ['buy_to_enter', 'sell_to_enter']:
                # Apply kademeli position limit
                if current_positions >= max_positions_for_cycle:
                    print(f"⚠️ KADEMELİ POZİSYON LİMİTİ (Cycle {cycle_number}): Max {max_positions_for_cycle} positions allowed. Skipping {coin} entry.")
                    continue
                current_positions += 1
                
                decisions_to_execute[coin] = trade

        if decisions_to_execute:
            self.execute_decision(decisions_to_execute, valid_prices, indicator_cache=indicator_cache)

    def get_max_positions_for_cycle(self, cycle_number: int) -> int:
        """Cycle bazlı maximum pozisyon limiti - Kademeli artış sistemi, MAX_POSITIONS ile sınırlı"""
        from config import Config
        max_allowed = Config.MAX_POSITIONS
        
        if cycle_number == 1:
            return min(1, max_allowed)  # Cycle 1: max 1 pozisyon (veya MAX_POSITIONS)
        elif cycle_number == 2:
            return min(2, max_allowed)  # Cycle 2: max 2 pozisyon (veya MAX_POSITIONS)
        elif cycle_number == 3:
            return min(3, max_allowed)  # Cycle 3: max 3 pozisyon (veya MAX_POSITIONS)
        elif cycle_number == 4:
            return min(4, max_allowed)  # Cycle 4: max 4 pozisyon (veya MAX_POSITIONS)
        else:
            return max_allowed  # Cycle 5+: MAX_POSITIONS değerini kullan

    def _get_indicator_snapshot(
        self,
        coin: str,
        indicator_cache: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Fetch indicators for 3m and higher timeframe from cache if available, otherwise from market data."""
        cache_source = indicator_cache if indicator_cache is not None else getattr(self, 'indicator_cache', {})
        cached_entry = cache_source.get(coin) if isinstance(cache_source, dict) else None

        indicators_3m = None
        indicators_htf = None

        if isinstance(cached_entry, dict):
            indicators_3m = copy.deepcopy(cached_entry.get('3m'))
            cached_htf = cached_entry.get(HTF_INTERVAL)
            if cached_htf is None and HTF_INTERVAL != '4h':
                cached_htf = cached_entry.get('4h')  # backward compatibility
            indicators_htf = copy.deepcopy(cached_htf)

        if not isinstance(indicators_3m, dict) or 'error' in indicators_3m:
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
        if not isinstance(indicators_htf, dict) or 'error' in indicators_htf:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)

        return indicators_3m, indicators_htf

    def _execute_normal_decisions(
        self,
        decisions: Dict,
        valid_prices: Dict,
        cycle_number: int,
        positions_closed_by_tp_sl: bool,
        indicator_cache: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Execute normal AI decisions with partial profit active"""
        print("🔄 Executing normal AI decisions (partial profit active)")
        
        # KADEMELİ POZİSYON SİSTEMİ: Cycle bazlı pozisyon limiti
        max_positions_for_cycle = self.get_max_positions_for_cycle(cycle_number)
        current_positions = len(self.positions)
        
        decisions_to_execute = {}
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                continue
                
            signal = trade.get('signal')
            if signal in ['buy_to_enter', 'sell_to_enter']:
                # Apply kademeli position limit
                if current_positions >= max_positions_for_cycle:
                    print(f"⚠️ KADEMELİ POZİSYON LİMİTİ (Cycle {cycle_number}): Max {max_positions_for_cycle} positions allowed. Skipping {coin} entry.")
                    continue
                current_positions += 1
                
                decisions_to_execute[coin] = trade
            else:
                # Execute all other decisions (hold, close_position)
                decisions_to_execute[coin] = trade

        if decisions_to_execute:
            self.execute_decision(decisions_to_execute, valid_prices, indicator_cache=indicator_cache)

    def _calculate_maximum_limit(self) -> float:
        """Calculate maximum limit: $15 fixed OR 15% of available cash, whichever is larger"""
        max_from_percentage = self.current_balance * 0.15
        max_limit = max(15.0, max_from_percentage)
        print(f"📊 Maximum limit: ${max_limit:.2f} (${15.0} fixed vs ${max_from_percentage:.2f} 15% of ${self.current_balance:.2f} available cash)")
        return max_limit


    def _adjust_partial_sale_for_max_limit(self, position: Dict, proposed_percent: float) -> Tuple[float, bool, Optional[str]]:
        """Adjust partial sale percentage to ensure position doesn't go below maximum limit"""
        current_margin = position.get('margin_usd', 0)
        
        # Calculate maximum limit: $15 fixed OR 15% of available cash, whichever is larger
        max_limit = self._calculate_maximum_limit()
        
        if current_margin <= max_limit:
            # Position already at or below maximum limit, don't sell - close completely
            print(f"🛑 Partial sale blocked: Position margin ${current_margin:.2f} <= maximum limit ${max_limit:.2f}. Position will be closed.")
            return 0.0, True, f"Position margin ${current_margin:.2f} <= maximum limit ${max_limit:.2f}"
        
        # Calculate remaining margin after proposed sale
        remaining_after_proposed = current_margin * (1 - proposed_percent)
        
        if remaining_after_proposed >= max_limit:
            # Proposed sale keeps us above maximum limit, use as-is
            return proposed_percent, False, None
        else:
            # Adjust sale to leave exactly max_limit margin
            adjusted_sale_amount = current_margin - max_limit
            adjusted_percent = adjusted_sale_amount / current_margin
            
            print(f"📊 Adjusted partial sale: {proposed_percent*100:.0f}% → {adjusted_percent*100:.0f}% to maintain ${max_limit:.2f} maximum limit")
            return adjusted_percent, False, None

    def _adjust_partial_sale_for_min_limit(self, position: Dict, proposed_percent: float) -> float:
        """Adjust partial sale percentage to ensure minimum limit remains after sale"""
        current_margin = position.get('margin_usd', 0)
        
        # Calculate dynamic minimum limit: $15 fixed OR 10% of available cash, whichever is larger
        min_remaining = self._calculate_dynamic_minimum_limit()
        
        if current_margin <= min_remaining:
            # Position already at or below minimum, don't sell
            print(f"🛑 Partial sale blocked: Position margin ${current_margin:.2f} <= minimum limit ${min_remaining:.2f}")
            return 0.0
        
        # Calculate remaining margin after proposed sale
        remaining_after_proposed = current_margin * (1 - proposed_percent)
        
        if remaining_after_proposed >= min_remaining:
            # Proposed sale keeps us above minimum, use as-is
            return proposed_percent
        else:
            # Adjust sale to leave exactly min_remaining margin
            adjusted_sale_amount = current_margin - min_remaining
            adjusted_percent = adjusted_sale_amount / current_margin
            
            print(f"📊 Adjusted partial sale: {proposed_percent*100:.0f}% → {adjusted_percent*100:.0f}% to maintain ${min_remaining:.2f} minimum limit")
            return adjusted_percent

    def _is_counter_trend_trade(self, coin: str, signal: str, indicators_3m: Dict, indicators_htf: Dict) -> bool:
        """Check if trade is counter-trend based on higher timeframe trend vs 3m signal"""
        try:
            if 'error' in indicators_3m or 'error' in indicators_htf:
                return False
            
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            
            # Determine higher timeframe trend direction
            trend_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"
            
            # Determine 3m trend direction
            trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
            
            # Check if trade is counter-trend (signal vs higher timeframe trend)
            if signal == 'buy_to_enter' and trend_htf == "BEARISH":
                return True  # Long against bearish higher timeframe trend
            elif signal == 'sell_to_enter' and trend_htf == "BULLISH":
                return True  # Short against bullish higher timeframe trend
            
            return False
            
        except Exception as e:
            print(f"⚠️ Counter-trend detection error for {coin}: {e}")
            return False

    def apply_market_regime_adjustment(self, confidence: float, signal: str, market_regime: str) -> float:
        """Apply market regime based confidence adjustment (hafifletilmiş - 0.7 yerine 0.92)"""
        original_confidence = confidence
        if market_regime == "BEARISH" and signal == "buy_to_enter":
            # Long in bearish market - counter-trade
            # Hafifletilmiş: 0.7 yerine 0.92 (sadece %8 düşüş)
            adjusted_confidence = max(confidence * 0.92, confidence - 0.05, original_confidence * 0.90)
            print(f"📊 Market regime adjustment: BEARISH market, LONG signal → confidence {confidence:.2f} → {adjusted_confidence:.2f}")
            return adjusted_confidence
        elif market_regime == "BULLISH" and signal == "sell_to_enter":
            # Short in bullish market - counter-trade
            # Hafifletilmiş: 0.7 yerine 0.92 (sadece %8 düşüş)
            adjusted_confidence = max(confidence * 0.92, confidence - 0.05, original_confidence * 0.90)
            print(f"📊 Market regime adjustment: BULLISH market, SHORT signal → confidence {confidence:.2f} → {adjusted_confidence:.2f}")
            return adjusted_confidence
        else:
            # Trend-following trade - no adjustment
            return confidence

    def validate_counter_trade(self, coin: str, signal: str, indicators_3m: Dict, indicators_htf: Dict) -> Dict[str, Any]:
        """Validate counter-trade with multiple technical conditions"""
        try:
            if 'error' in indicators_3m or 'error' in indicators_htf:
                return {"valid": False, "reason": "Indicator data error"}
            conditions_met: List[str] = []
            conditions_met_count = 0
            total_conditions_available = 5
            score = 0.0
            score_breakdown: List[str] = []
            relaxed_cycles = getattr(self, 'relaxed_countertrend_cycles', 0)
            relax_mode_active = relaxed_cycles > 0
            
            def _register(condition: bool, description: str, weight: float) -> None:
                nonlocal conditions_met_count, score
                if condition:
                    conditions_met_count += 1
                    conditions_met.append(description)
                    score += weight
                    score_breakdown.append(f"{description} (+{weight:.1f})")

            # Condition 1: 3m momentum supportive of the counter direction
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            if isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)):
                if signal == 'buy_to_enter':
                    _register(price_3m > ema20_3m, "3m momentum supportive (price > EMA20)", 0.9)
                elif signal == 'sell_to_enter':
                    _register(price_3m < ema20_3m, "3m momentum supportive (price < EMA20)", 0.9)
            
            # Condition 2: Volume confirmation (>1.5x average)
            current_volume = indicators_3m.get('volume', 0)
            avg_volume = indicators_3m.get('avg_volume', 1)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            if volume_ratio > 1.5:
                _register(True, f"Volume {volume_ratio:.1f}x average", 1.1)
            elif volume_ratio > 1.0:
                _register(True, f"Volume {volume_ratio:.1f}x average", 0.9)
            elif volume_ratio > 0.8:
                _register(True, f"Volume {volume_ratio:.1f}x average", 0.8)
            else:
                score_breakdown.append(f"Volume {volume_ratio:.2f}x average (no boost)")
            
            # Condition 3: Extreme RSI
            rsi_3m = indicators_3m.get('rsi_14', 50)
            if signal == 'buy_to_enter':
                _register(rsi_3m < 35, f"Extreme RSI ({rsi_3m:.1f})", 1.0)
            elif signal == 'sell_to_enter':
                _register(rsi_3m > 65, f"Extreme RSI ({rsi_3m:.1f})", 1.0)
            
            # Condition 4: Price close to EMA20 (< 1%)
            if isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)) and price_3m:
                price_ema_distance = abs(price_3m - ema20_3m) / price_3m * 100
                ema_distance_threshold = 1.8 if not relax_mode_active else 2.5
                _register(price_ema_distance < ema_distance_threshold, f"Price within {ema_distance_threshold:.1f}% of EMA20 ({price_ema_distance:.2f}%)", 0.6)
            
            # Condition 5: MACD divergence supportive
            macd_3m = indicators_3m.get('macd', 0)
            macd_signal_3m = indicators_3m.get('macd_signal', 0)
            if signal == 'buy_to_enter':
                _register(macd_3m > macd_signal_3m, "MACD divergence supportive", 0.8)
            elif signal == 'sell_to_enter':
                _register(macd_3m < macd_signal_3m, "MACD divergence supportive", 0.8)

            # Volume safety rail: reject extremely illiquid environments
            if volume_ratio is not None:
                volume_safety_floor = 0.15
                if relax_mode_active:
                    volume_safety_floor = 0.10
                if volume_ratio < volume_safety_floor:
                    return {
                        "valid": False,
                        "conditions_met": conditions_met,
                        "total_conditions": conditions_met_count,
                        "conditions_required": total_conditions_available,
                        "score": score,
                        "score_threshold": None,
                        "score_breakdown": score_breakdown,
                        "reason": (
                            f"Volume ratio {volume_ratio:.2f}x is below {volume_safety_floor:.2f}x minimum for counter-trend entries"
                            if not relax_mode_active else
                            f"Volume ratio {volume_ratio:.2f}x is below relaxed minimum {volume_safety_floor:.2f}x while cooldown active"
                        )
                    }

            # Dynamic score threshold based on liquidity context
            score_threshold = 2.3
            if volume_ratio is not None:
                if volume_ratio >= 1.5:
                    score_threshold = 2.0
                elif volume_ratio >= 1.0:
                    score_threshold = 2.1
                elif volume_ratio >= 0.8:
                    score_threshold = 2.2
                else:
                    score_threshold = 2.3
            if relax_mode_active:
                score_threshold = max(1.8, score_threshold - 0.3)

            valid = score >= score_threshold
            return {
                "valid": valid,
                "conditions_met": conditions_met,
                "total_conditions": conditions_met_count,
                "conditions_required": total_conditions_available,
                "score": round(score, 2),
                "score_threshold": round(score_threshold, 2),
                "score_breakdown": score_breakdown,
                "reason": (
                    f"Counter-trend score {score:.2f} ≥ threshold {score_threshold:.2f} ({conditions_met_count}/{total_conditions_available} signals)"
                    if valid else
                    f"Counter-trend score {score:.2f} < threshold {score_threshold:.2f} ({conditions_met_count}/{total_conditions_available} signals)"
                ),
                "relax_mode_active": relax_mode_active,
                "relaxed_cycles_remaining": relaxed_cycles if relax_mode_active else 0
            }
        except Exception as e:
            print(f"⚠️ Counter-trade validation error for {coin}: {e}")
            return {"valid": False, "reason": f"Validation error: {str(e)}"}

    def calculate_dynamic_risk(self, market_regime: str, confidence: float) -> float:
        """Calculate dynamic risk based on market regime and confidence"""
        base_risk = 50.0  # $50 base risk
        
        # Market regime adjustment
        if "BEARISH" in market_regime:
            base_risk *= 0.8  # Bearish market: reduce risk to $40
        elif "BULLISH" in market_regime:
            base_risk *= 1.2  # Bullish market: increase risk to $60
        
        # Confidence adjustment
        if confidence >= 0.7:
            base_risk *= 1.1  # High confidence: +10%
        elif confidence <= 0.5:
            base_risk *= 0.9  # Low confidence: -10%
            
        return min(base_risk, 60.0)  # Cap at $60 maximum risk


    def calculate_confidence_based_margin(self, confidence: float, available_cash: float) -> float:
        """Calculate margin based on confidence level and available cash (new simplified formula)"""
        # Max margin = 40% of available cash × confidence
        margin = available_cash * 0.40 * confidence
        
        # Apply minimum margin limit ($10)
        margin = max(margin, Config.MIN_POSITION_MARGIN_USD)
        
        print(f"📊 Confidence-based margin: ${margin:.2f} (confidence: {confidence:.2f}, available cash: ${available_cash:.2f})")
        return margin

    def get_volume_threshold(self, market_regime: str, signal: str) -> float:
        """Get volume threshold based on market regime and signal type"""
        if signal == "buy_to_enter":
            if "BULLISH" in market_regime:
                return 0.6  # Bullish market: LONG >60% volume
            else:
                return 0.8  # Other markets: LONG >80% volume
        elif signal == "sell_to_enter":
            if "BEARISH" in market_regime:
                return 0.3  # Bearish market: SHORT >30% volume
            else:
                return 0.4  # Other markets: SHORT >40% volume
        return 0.8  # Default threshold

    def calculate_volume_quality_score(
        self,
        coin: str,
        indicators_3m: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate volume quality score (0-100) based on Config thresholds"""
        try:
            if indicators_3m is None or not isinstance(indicators_3m, dict):
                indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            if 'error' in indicators_3m:
                return 0.0
            
            current_volume = indicators_3m.get('volume', 0)
            avg_volume = indicators_3m.get('avg_volume', 0)
            
            if avg_volume <= 0:
                return 0.0
            
            volume_ratio = current_volume / avg_volume
            
            # Calculate score based on Config thresholds
            if volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['excellent']:
                return 90.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['good']:
                return 75.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['fair']:
                return 60.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['poor']:
                return 40.0
            else:
                return 20.0
                
        except Exception as e:
            print(f"⚠️ Volume quality score calculation error for {coin}: {e}")
            return 0.0

    def detect_market_regime_overall(self) -> str:
        """Detect overall market regime across all coins"""
        try:
            from alpha_arena_deepseek import RealMarketData
            market_data = RealMarketData()
            
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            for coin in market_data.available_coins:
                indicators_htf = market_data.get_technical_indicators(coin, HTF_INTERVAL)
                if 'error' in indicators_htf:
                    continue
                
                price = indicators_htf.get('current_price')
                ema20 = indicators_htf.get('ema_20')
                
                if not isinstance(price, (int, float)) or not isinstance(ema20, (int, float)) or ema20 == 0:
                    continue

                delta = (price - ema20) / ema20
                if abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT:
                    neutral_count += 1
                elif price > ema20:
                    bullish_count += 1
                else:
                    bearish_count += 1
            
            if bullish_count >= 4:
                return "BULLISH"
            if bearish_count >= 4:
                return "BEARISH"
            if neutral_count >= 4:
                return "NEUTRAL"

            if bullish_count > bearish_count:
                return "BULLISH" if bullish_count >= 3 else "NEUTRAL"
            if bearish_count > bullish_count:
                return "BEARISH" if bearish_count >= 3 else "NEUTRAL"
                return "NEUTRAL"
                
        except Exception as e:
            print(f"⚠️ Market regime detection error: {e}")
            return "NEUTRAL"

    def should_enhance_short_sizing(self, coin: str) -> bool:
        """Check if short position should be enhanced (%15 daha büyük)"""
        try:
            # Import market data to get indicators
            from alpha_arena_deepseek import RealMarketData
            market_data = RealMarketData()
            
            indicators_3m = market_data.get_technical_indicators(coin, '3m')
            indicators_htf = market_data.get_technical_indicators(coin, HTF_INTERVAL)
            
            if 'error' in indicators_3m or 'error' in indicators_htf:
                return False
            
            # Enhanced short conditions:
            # 1. 3m RSI > 70 (aşırı alım)
            rsi_3m = indicators_3m.get('rsi_14', 50)
            # 2. Volume > 1.5x average
            volume_ratio = indicators_3m.get('volume', 0) / indicators_3m.get('avg_volume', 1)
            # 3. Higher timeframe trend bearish
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            trend_bearish = price_htf < ema20_htf
            
            # All conditions must be met
            return rsi_3m > 70 and volume_ratio > 1.5 and trend_bearish
            
        except Exception as e:
            print(f"⚠️ Enhanced short sizing check error for {coin}: {e}")
            return False

    def execute_decision(
        self,
        decisions: Dict[str, Dict],
        current_prices: Dict[str, float],
        indicator_cache: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Executes trading decisions from AI, incorporating dynamic sizing and enhanced features."""
        print("\n⚡ EXECUTING AI DECISIONS...")
        if not isinstance(decisions, dict): print(f"❌ Invalid decisions format: {type(decisions)}"); return

        # Import Config inside the function to avoid scope issues
        from config import Config

        bias_metrics = getattr(self, 'latest_bias_metrics', self.get_directional_bias_metrics())
        execution_report = {
            'executed': [],
            'blocked': [],
            'skipped': [],
            'holds': [],
            'notes': [],
            'timestamp': datetime.now().isoformat()
        }
        live_trading = getattr(self, 'is_live_trading', False)
        if live_trading:
            self.sync_live_account()

        for coin, trade in decisions.items():
            if not isinstance(trade, dict): print(f"⚠️ Invalid trade data for {coin}: {type(trade)}"); continue
            if coin not in current_prices or not isinstance(current_prices[coin], (int, float)) or current_prices[coin] <= 0:
                print(f"⚠️ Skipping {coin}: Invalid price data."); continue

            signal = trade.get('signal'); current_price = current_prices[coin]; position = self.positions.get(coin)

            if signal == 'buy_to_enter' or signal == 'sell_to_enter':
                if position:
                    print(f"⚠️ {signal.upper()} {coin}: Position already open.")
                    execution_report['skipped'].append({'coin': coin, 'reason': 'position_exists', 'signal': signal})
                    trade['runtime_decision'] = 'skipped_existing_position'
                    continue

                confidence = trade.get('confidence', 0.5) # Default 50% confidence if missing
                leverage = trade.get('leverage')
                if leverage in (None, "", 0):
                    leverage = 8
                # Ensure confidence and leverage are valid numbers
                try:
                    confidence = float(confidence)
                    leverage = int(leverage)
                except (ValueError, TypeError): print(f"⚠️ Invalid confidence ({confidence}) or leverage ({leverage}) for {coin}. Skipping."); continue
                if leverage < 1:
                    leverage = 1
                # Enforce maximum leverage limit from config
                if leverage > Config.MAX_LEVERAGE: 
                    print(f"⚠️ Leverage {leverage}x exceeds maximum limit of {Config.MAX_LEVERAGE}x. Reducing to {Config.MAX_LEVERAGE}x.")
                    leverage = Config.MAX_LEVERAGE
                # Clamp leverage into [8, 10] operational band for new entries
                if signal in ['buy_to_enter', 'sell_to_enter']:
                    if leverage < 8:
                        print(f"ℹ️ Adjusting leverage from {leverage}x to minimum operational level 8x for {coin}.")
                        leverage = 8
                    elif leverage > 10:
                        print(f"ℹ️ Adjusting leverage from {leverage}x to maximum operational level 10x for {coin}.")
                        leverage = 10
                if not (0.0 <= confidence <= 1.0): confidence = 0.5 # Clamp confidence to 0.0-1.0
                
                # 2. Market Regime Position Sizing
                market_regime = self.detect_market_regime_overall()
                market_regime_multiplier = Config.MARKET_REGIME_MULTIPLIERS.get(market_regime, 1.0)
                partial_margin_factor = 1.0

                direction = 'long' if signal == 'buy_to_enter' else 'short'
                dominant_direction = None
                if market_regime == 'BULLISH':
                    dominant_direction = 'long'
                elif market_regime == 'BEARISH':
                    dominant_direction = 'short'

                # Coin bazlı cooldown kontrolü (öncelikli - zararlı trade'den sonra aynı coin'i engelle)
                coin_cooldowns = getattr(self, 'coin_cooldowns', {})
                coin_upper = coin.upper()
                coin_cooldown_remaining = coin_cooldowns.get(coin_upper, 0)
                if coin_cooldown_remaining > 0:
                    print(f"⏸️ Coin cooldown active: Blocking {coin} entry ({coin_cooldown_remaining} cycles remaining - previous loss).")
                    execution_report['blocked'].append({
                        'coin': coin,
                        'reason': 'coin_cooldown',
                        'cooldown_remaining': coin_cooldown_remaining
                    })
                    trade['runtime_decision'] = 'blocked_coin_cooldown'
                    continue

                # Cooldown kontrolü: PortfolioManager'dan cooldown durumunu al
                cooldowns = getattr(self, 'directional_cooldowns', {'long': 0, 'short': 0})
                cooldown_remaining = cooldowns.get(direction, 0)
                print(f"🔍 Cooldown check for {coin} {direction.upper()}: cooldown_remaining={cooldown_remaining}, cooldowns={cooldowns}")
                if cooldown_remaining > 0:
                    print(f"⏸️ Directional cooldown active: Blocking {direction.upper()} entry for {coin} ({cooldown_remaining} cycles remaining).")
                    execution_report['blocked'].append({
                        'coin': coin,
                        'reason': 'directional_cooldown',
                        'direction': direction,
                        'cooldown_remaining': cooldown_remaining
                    })
                    trade['runtime_decision'] = 'blocked_directional_cooldown'
                    continue

                # Check SAME_DIRECTION_LIMIT for ALL directions (not just dominant)
                directional_counts = self.count_positions_by_direction()
                current_same_direction = directional_counts.get(direction, 0)
                if current_same_direction >= Config.SAME_DIRECTION_LIMIT:
                    print(
                        f"🚫 SAME-DIRECTION LIMIT: {coin} {signal} blocked. "
                        f"{current_same_direction}/{Config.SAME_DIRECTION_LIMIT} {direction.upper()} positions already open."
                    )
                    execution_report['blocked'].append({
                        'coin': coin,
                        'reason': 'same_direction_limit',
                        'direction': direction,
                        'current': current_same_direction,
                        'limit': Config.SAME_DIRECTION_LIMIT
                    })
                    trade['runtime_decision'] = 'blocked_same_direction_limit'
                    continue
                
                # 3. Enhanced Short Sizing (increase by 15% when criteria met)
                if signal == 'sell_to_enter':
                    # Check enhanced short conditions
                    if self.should_enhance_short_sizing(coin):
                        print(f"📈 ENHANCED SHORT: increasing {coin} short exposure by 15% based on conditions")
                        if 'quantity_usd' in trade:
                            trade['quantity_usd'] *= Config.SHORT_ENHANCEMENT_MULTIPLIER
                
                # 4. Coin-specific dynamic stop-loss adjustment
                stop_loss = trade.get('stop_loss')
                try:
                    stop_loss = float(stop_loss) if stop_loss is not None else None
                except (ValueError, TypeError):
                    stop_loss = None
                
                # Apply dynamic stop-loss multiplier
                if stop_loss is not None:
                    stop_loss_multiplier = Config.COIN_SPECIFIC_STOP_LOSS_MULTIPLIERS.get(coin, 1.0)
                    if signal == 'buy_to_enter':
                        stop_loss = current_price - ((current_price - stop_loss) * stop_loss_multiplier)
                    else:  # sell_to_enter
                        stop_loss = current_price + ((stop_loss - current_price) * stop_loss_multiplier)
                    print(f"📊 Dynamic Stop-Loss: applied {stop_loss_multiplier}x multiplier for {coin}")
                
                # 5. Counter-Trend detection (validate only for counter trades) & indicator alignment
                # Use cached indicators when available to keep prompt/execution consistent
                current_trend = 'unknown'
                volume_ratio = None
                trend_classification = 'unknown'
                try:
                    indicators_3m, indicators_htf = self._get_indicator_snapshot(coin, indicator_cache)
                    if ('error' in indicators_3m) or ('error' in indicators_htf):
                        print(f"⚠️ Indicator fetch error for {coin}: {indicators_3m.get('error', '') or indicators_htf.get('error', '')}")
                        execution_report['blocked'].append({'coin': coin, 'reason': 'indicator_error'})
                        trade['runtime_decision'] = 'blocked_indicator_error'
                        continue
                    # Volume quality scoring using the same data the AI saw
                    volume_quality_score = self.calculate_volume_quality_score(coin, indicators_3m=indicators_3m)
                    confidence = min(1.0, confidence + (volume_quality_score / 1000))
                    trade['volume_quality_score'] = volume_quality_score

                    current_volume = indicators_3m.get('volume', 0)
                    avg_volume = indicators_3m.get('avg_volume', 1)
                    volume_ratio = current_volume / avg_volume if avg_volume and avg_volume > 0 else 0.0
                    trade['volume_ratio_runtime'] = round(volume_ratio, 4)
                    relax_cycles_global = getattr(self, 'relaxed_countertrend_cycles', 0)
                    relax_mode_active_global = relax_cycles_global > 0
                    if volume_ratio is not None:
                        low_volume_threshold = 0.20 if not relax_mode_active_global else 0.15
                        if volume_ratio < low_volume_threshold:
                            if relax_mode_active_global:
                                print(f"⚡ Relaxed mode: skipping low-volume penalty for {coin} (ratio {volume_ratio:.2f}x).")
                            else:
                                original_confidence = confidence
                                # Hafifletilmiş penalty: 0.8 yerine 0.92 (sadece %8 düşüş)
                                confidence = max(confidence * 0.92, confidence - 0.05)
                                # Minimum confidence floor: AI'nın verdiği değerin %85'i altına düşmesin
                                min_floor = original_confidence * 0.85
                                confidence = max(confidence, min_floor, Config.MIN_CONFIDENCE)
                                print(f"🥶 LOW VOLUME PENALTY: {coin} volume ratio {volume_ratio:.2f}x. Confidence {original_confidence:.2f} → {confidence:.2f}")
                                if confidence < Config.MIN_CONFIDENCE:
                                    print(f"🚫 Low volume block: {coin} confidence {confidence:.2f} below minimum after penalty.")
                                    execution_report['blocked'].append({'coin': coin, 'reason': 'low_volume', 'volume_ratio': volume_ratio, 'confidence': confidence})
                                    trade['runtime_decision'] = 'blocked_low_volume'
                                    continue
                                trade['confidence'] = confidence
                    trend_info = self.update_trend_state(coin, indicators_htf, indicators_3m)
                    current_trend = trend_info.get('trend', 'unknown')
                    flip_cycle = trend_info.get('last_flip_cycle')
                    last_flip_direction = trend_info.get('last_flip_direction')
                    guard_cycles_since_flip = None
                    guard_window = self.trend_flip_cooldown
                    if isinstance(flip_cycle, int):
                        guard_cycles_since_flip = max(0, self.current_cycle_number - flip_cycle)
                    trade['trend_runtime'] = current_trend

                    pre_bias_confidence = confidence
                    confidence = self.apply_directional_bias(signal, confidence, bias_metrics, current_trend)
                    if confidence != pre_bias_confidence:
                        print(f"🧭 Directional bias adjustment: {coin} {signal} confidence {pre_bias_confidence:.2f} → {confidence:.2f}")
                        trade['confidence'] = confidence
                    is_counter_trend = self._is_counter_trend_trade(coin, signal, indicators_3m, indicators_htf)
                    trend_classification = 'counter_trend' if is_counter_trend else 'trend_following'
                    trade['trend_alignment'] = trend_classification
                    snapshot_parts = []
                    price_htf = indicators_htf.get('current_price')
                    ema20_htf = indicators_htf.get('ema_20')
                    price_3m = indicators_3m.get('current_price')
                    ema20_3m = indicators_3m.get('ema_20')
                    def _fmt(val):
                        return f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
                    comparison_htf = "?" if not isinstance(price_htf, (int, float)) or not isinstance(ema20_htf, (int, float)) else (">" if price_htf > ema20_htf else "<" if price_htf < ema20_htf else "=")
                    comparison_3m = "?" if not isinstance(price_3m, (int, float)) or not isinstance(ema20_3m, (int, float)) else (">" if price_3m > ema20_3m else "<" if price_3m < ema20_3m else "=")
                    snapshot_parts.append(f"{HTF_LABEL} price={_fmt(price_htf)} {comparison_htf} EMA20={_fmt(ema20_htf)}")
                    snapshot_parts.append(f"3m price={_fmt(price_3m)} {comparison_3m} EMA20={_fmt(ema20_3m)}")
                    snapshot_parts.append(f"volume_ratio={volume_ratio:.2f}x")
                    snapshot_parts.append(f"counter_trend={is_counter_trend}")
                    snapshot_parts.append(f"trend_state={current_trend.upper()}")
                    print(f"🧾 EXECUTION SNAPSHOT {coin}: " + " | ".join(snapshot_parts))

                    direction = 'long' if signal == 'buy_to_enter' else 'short'
                    
                    if is_counter_trend:
                        # Check counter-trend cooldown
                        counter_trend_cooldown = getattr(self, 'counter_trend_cooldown', 0)
                        if counter_trend_cooldown > 0:
                            print(f"🚫 Counter-trend cooldown active: Blocking {coin} {signal} ({counter_trend_cooldown} cycles remaining).")
                            execution_report['blocked'].append({'coin': coin, 'reason': 'counter_trend_cooldown', 'classification': trend_classification})
                            trade['runtime_decision'] = 'blocked_counter_trend_cooldown'
                            continue
                        
                        relaxed_countertrend = getattr(self, 'relaxed_countertrend_cycles', 0) > 0
                        if relaxed_countertrend:
                            remaining_relax = getattr(self, 'relaxed_countertrend_cycles', 0)
                            print(f"⚡ RELAXED COUNTER-TREND MODE: {coin} - skipping flip guard & validation ({remaining_relax} cycles remaining).")
                        else:
                            guard_active = guard_cycles_since_flip is not None and guard_cycles_since_flip <= guard_window
                            if guard_active:
                                if guard_cycles_since_flip == 0:
                                    min_conf = 0.63
                                    if confidence < min_conf:
                                        print(f"🚫 Flip guard confidence floor: {coin} {signal} confidence {confidence:.2f} < {min_conf:.2f} in same cycle after flip.")
                                        execution_report['blocked'].append({'coin': coin, 'reason': 'trend_flip_guard_confidence', 'classification': trend_classification})
                                        trade['runtime_decision'] = 'blocked_trend_flip_confidence'
                                        continue
                                    partial_margin_factor = min(partial_margin_factor, 0.7)
                                    print(f"⏳ Trend flip guard: {coin} sizing capped at 40% in same-cycle counter-trend attempt (confidence {confidence:.2f}).")
                                elif guard_cycles_since_flip == 1:
                                    min_conf = 0.70
                                    if confidence < min_conf:
                                        print(f"🚫 Flip guard confidence floor: {coin} {signal} confidence {confidence:.2f} < {min_conf:.2f} one cycle after flip.")
                                        execution_report['blocked'].append({'coin': coin, 'reason': 'trend_flip_guard_confidence', 'classification': trend_classification})
                                        trade['runtime_decision'] = 'blocked_trend_flip_confidence'
                                        continue
                                    partial_margin_factor = min(partial_margin_factor, 0.8)
                                    print(f"⏳ Trend flip guard (counter-trend): {coin} sizing capped at 60% one cycle after flip.")
                                elif guard_cycles_since_flip == 2:
                                    partial_margin_factor = min(partial_margin_factor, 0.9)
                                    print(f"⏳ Trend flip guard (counter-trend): {coin} sizing capped at 80% two cycles after flip.")
                            counter_confidence_floor = 0.65 if not relaxed_countertrend else 0.60
                            if confidence < counter_confidence_floor:
                                if relaxed_countertrend:
                                    print(f"⚠️ Relaxed mode: counter-trend confidence {confidence:.2f} below {counter_confidence_floor:.2f}, but allowing due to cooldown.")
                                else:
                                    print(f"⚠️ WARNING: Counter-trend confidence {confidence:.2f} below recommended {counter_confidence_floor:.2f} - proceeding with AI decision")
                            print(f"⚠️ COUNTER-TREND DETECTED: {coin} - respecting AI decision with additional validation")
                            
                            # Perform validation for counter-trend trades only
                            validation_result = self.validate_counter_trade(coin, signal, indicators_3m, indicators_htf)
                            
                            if validation_result['valid']:
                                print(f"✅ COUNTER-TRADE STRONG: {validation_result['reason']}")
                                print(f"   Conditions met: {validation_result.get('conditions_met', [])}")
                            else:
                                print(f"⚠️ COUNTER-TRADE WEAK: {validation_result['reason']}")
                                print(f"   Conditions met: {validation_result.get('conditions_met', [])}")
                                print(f"⚠️ WARNING: Counter-trend validation shows weak conditions - proceeding with AI decision")
                    else:
                        guard_active = guard_cycles_since_flip is not None and guard_cycles_since_flip <= guard_window
                        if guard_active and last_flip_direction:
                            is_trend_direction = ((last_flip_direction == 'bullish' and direction == 'long') or (last_flip_direction == 'bearish' and direction == 'short'))
                            if is_trend_direction:
                                original_conf = confidence
                                if guard_cycles_since_flip == 0:
                                    # Hafifletilmiş: 0.90 yerine 0.97
                                    confidence = max(confidence * 0.97, confidence - 0.02, original_conf * 0.95)
                                    partial_margin_factor = min(partial_margin_factor, 0.7)
                                    print(f"⏳ Trend flip guard (trend-following): {coin} confidence {original_conf:.2f} → {confidence:.2f} & sizing 50% immediately after flip.")
                                elif guard_cycles_since_flip == 1:
                                    # Hafifletilmiş: 0.95 yerine 0.98
                                    confidence = max(confidence * 0.98, confidence - 0.01, original_conf * 0.97)
                                    partial_margin_factor = min(partial_margin_factor, 0.8)
                                    print(f"⏳ Trend flip guard (trend-following): {coin} confidence {original_conf:.2f} → {confidence:.2f} & sizing 70% one cycle after flip.")
                                elif guard_cycles_since_flip == 2:
                                    # Hafifletilmiş: 0.98 yerine 0.99
                                    confidence = max(confidence * 0.99, original_conf * 0.98)
                                    partial_margin_factor = min(partial_margin_factor, 0.90)
                                    print(f"⏳ Trend flip guard (trend-following): {coin} confidence {original_conf:.2f} → {confidence:.2f} & sizing 85% two cycles after flip.")
                                trade['confidence'] = confidence
                        # Trend-following trade path - Hibrit yaklaşım (15m dahil)
                        trend_strength_result = self.get_trend_following_strength(coin, signal)

                        if trend_strength_result and trend_strength_result['strength']:
                            # 15m dahil multi-timeframe analizi başarılı
                            strength = trend_strength_result['strength']
                            alignment_info = trend_strength_result['alignment_info']
                            trends = trend_strength_result['trends']
                            
                            # Mevcut trend_aligned kontrolü (geriye dönük uyumluluk için)
                            price_htf_follow = indicators_htf.get('current_price')
                            ema20_htf_follow = indicators_htf.get('ema_20')
                            ema20_3m = indicators_3m.get('ema_20')
                            price_3m = indicators_3m.get('current_price')
                            trend_aligned = False
                            if isinstance(price_htf_follow, (int, float)) and isinstance(ema20_htf_follow, (int, float)) \
                                    and isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)):
                                if signal == 'buy_to_enter' and price_htf_follow >= ema20_htf_follow and price_3m >= ema20_3m:
                                    trend_aligned = True
                                elif signal == 'sell_to_enter' and price_htf_follow <= ema20_htf_follow and price_3m <= ema20_3m:
                                    trend_aligned = True
                            
                            # Logging - Güç seviyesi ve 15m bilgisi ile
                            ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else "n/a"
                            print(f"✅ TREND-FOLLOWING ({strength}): {coin} {alignment_info}")
                            print(f"   Timeframes: 1h={trends['1h']}, 15m={trends['15m']}, 3m={trends['3m']} | Volume: {ratio_str}x")
                            
                            # Mevcut volume kontrolü (değişiklik yok)
                            if trend_aligned:
                                if volume_ratio is not None and volume_ratio >= 0.5:
                                    if volume_ratio < 0.8:
                                        partial_margin_factor = 0.5
                                        print(f"🧪 Low-volume trend-following: using 50% margin for {coin} (volume ratio {volume_ratio:.2f})")
                        else:
                            # Fallback: Eski mantık (hata durumunda veya 15m verisi yoksa)
                            price_htf_follow = indicators_htf.get('current_price')
                            ema20_htf_follow = indicators_htf.get('ema_20')
                            ema20_3m = indicators_3m.get('ema_20')
                            price_3m = indicators_3m.get('current_price')
                            trend_aligned = False
                            if isinstance(price_htf_follow, (int, float)) and isinstance(ema20_htf_follow, (int, float)) \
                                    and isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)):
                                if signal == 'buy_to_enter' and price_htf_follow >= ema20_htf_follow and price_3m >= ema20_3m:
                                    trend_aligned = True
                                elif signal == 'sell_to_enter' and price_htf_follow <= ema20_htf_follow and price_3m <= ema20_3m:
                                    trend_aligned = True
                            if trend_aligned:
                                if volume_ratio is not None and volume_ratio >= 0.5:
                                    if volume_ratio < 0.8:
                                        partial_margin_factor = 0.5
                                        print(f"🧪 Low-volume trend-following: using 50% margin for {coin} (volume ratio {volume_ratio:.2f})")
                                ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else "n/a"
                                print(f"✅ TREND-FOLLOWING: {coin} aligns with {HTF_LABEL} trend direction (volume ratio {ratio_str})")
                            else:
                                print(f"✅ TREND-FOLLOWING: {coin} aligns with {HTF_LABEL} trend direction")
                            
                except Exception as e:
                    print(f"⚠️ Counter-trend detection failed for {coin}: {e}")
                    # Detection hatasında trade'e izin ver
                    execution_report['notes'].append({'coin': coin, 'warning': str(e)})
                
                # Use dynamic confidence-based margin calculation instead of AI's quantity_usd
                # This ensures position sizing is ratio-based and dynamic
                calculated_margin = self.calculate_confidence_based_margin(confidence, self.current_balance)
                
                # Apply market regime multiplier
                calculated_margin *= market_regime_multiplier
                if partial_margin_factor < 1.0:
                    standard_margin = calculated_margin
                    reduced_margin = standard_margin * partial_margin_factor
                    print(f"📉 Applying partial margin ({partial_margin_factor*100:.0f}%): ${standard_margin:.2f} → ${reduced_margin:.2f}")
                    calculated_margin = max(reduced_margin, Config.MIN_POSITION_MARGIN_USD)
                
                # MINIMUM $10 COIN MIKTARI KONTROLÜ
                if calculated_margin < Config.MIN_POSITION_MARGIN_USD:
                    print(f"ℹ️ Calculated margin ${calculated_margin:.2f} below minimum ${Config.MIN_POSITION_MARGIN_USD:.2f}. Using minimum margin instead.")
                    calculated_margin = Config.MIN_POSITION_MARGIN_USD
                
                # AVAILABLE CASH KORUMA KONTROLÜ
                min_available_cash = self.current_balance * 0.10
                if (self.current_balance - calculated_margin) < min_available_cash:
                    print(f"⚠️ Trade would reduce available cash below minimum ${min_available_cash:.2f}. Trade blocked.")
                    execution_report['blocked'].append({'coin': coin, 'reason': 'available_cash_guard', 'calculated_margin': calculated_margin})
                    trade['runtime_decision'] = 'blocked_cash_guard'
                    continue
                
                # Convert margin to notional using leverage
                calculated_notional_usd = calculated_margin * leverage
                print(f"   Dynamic confidence-based sizing: ${calculated_notional_usd:.2f} notional (${calculated_margin:.2f} margin)")
                
                # Check risk management constraints with new simplified system
                # Minimum $10 ve available cash koruma zaten kontrol edildi
                # Risk manager artık sadece position limit kontrolü yapacak
                risk_decision = self.risk_manager.should_enter_trade(
                    symbol=coin,
                    current_positions=self.positions,
                    current_prices=current_prices,
                    confidence=confidence,
                    proposed_notional=calculated_notional_usd,
                    current_balance=self.current_balance
                )
                
                if not risk_decision['should_enter']:
                    print(f"⚠️ Risk management blocked trade: {risk_decision['reason']}")
                    execution_report['blocked'].append({'coin': coin, 'reason': f"risk_manager:{risk_decision['reason']}"})
                    trade['runtime_decision'] = 'blocked_risk_manager'
                    continue
                
                notional_usd = calculated_notional_usd
                margin_usd = notional_usd / leverage # Margin required

                if margin_usd <= 0: print(f"⚠️ {signal.upper()} {coin}: Calculated margin is zero/negative. Skipping."); continue
                if margin_usd > self.current_balance: print(f"⚠️ {signal.upper()} {coin}: Not enough cash for margin (${margin_usd:.2f} > ${self.current_balance:.2f})"); continue

                quantity_coin = notional_usd / current_price
                
                if live_trading:
                    live_result = self.execute_live_entry(
                        coin=coin,
                        direction=direction,
                        quantity=quantity_coin,
                        leverage=leverage,
                        current_price=current_price,
                        notional_usd=notional_usd,
                        confidence=confidence,
                        margin_usd=margin_usd,  # Pass calculated margin_usd
                        stop_loss=stop_loss,
                        profit_target=trade.get('profit_target'),
                        invalidation=trade.get('invalidation_condition')
                    )
                    if not live_result.get('success'):
                        error_msg = live_result.get('error', 'unknown_error')
                        print(f"🚫 LIVE ORDER FAILED: {coin} {signal} ({error_msg})")
                        execution_report['blocked'].append({
                            'coin': coin,
                            'reason': 'live_order_failed',
                            'error': error_msg
                        })
                        trade['runtime_decision'] = 'blocked_live_order'
                        continue

                    execution_report['executed'].append({
                        'coin': coin,
                        'signal': signal,
                        'confidence': confidence,
                        'classification': trend_classification,
                        'volume_ratio': volume_ratio,
                        'margin_usd': live_result.get('margin_usd'),
                        'mode': 'live',
                        'order_id': live_result.get('order', {}).get('orderId')
                    })
                    trade['runtime_decision'] = 'executed_live'
                    continue

                self.current_balance -= margin_usd # Deduct margin (simulation)

                direction = 'long' if signal == 'buy_to_enter' else 'short'
                estimated_liq_price = self._estimate_liquidation_price(current_price, leverage, direction)
                
                self.positions[coin] = {
                    'symbol': coin,
                    'direction': direction,
                    'quantity': quantity_coin,
                    'entry_price': current_price,
                    'entry_time': datetime.now().isoformat(),
                    'current_price': current_price,
                    'unrealized_pnl': 0.0,
                    'notional_usd': notional_usd,
                    'margin_usd': margin_usd,
                    'leverage': leverage,
                    'liquidation_price': estimated_liq_price,
                    'confidence': confidence,
                    'exit_plan': {
                        'profit_target': trade.get('profit_target'),
                        'stop_loss': stop_loss,
                        'invalidation_condition': trade.get('invalidation_condition')
                    },
                    'risk_usd': margin_usd,
                    'loss_cycle_count': 0,
                    'entry_volume': indicators_3m.get('volume') if isinstance(indicators_3m, dict) else None,
                    'entry_avg_volume': indicators_3m.get('avg_volume') if isinstance(indicators_3m, dict) else None,
                    'entry_volume_ratio': volume_ratio,
                    'entry_atr_14': indicators_3m.get('atr_14') if isinstance(indicators_3m, dict) else None,
                    'trend_alignment': trend_classification,
                    'trend_context': {
                        'trend_at_entry': current_trend,
                        'alignment': trend_classification,
                        'cycle': self.current_cycle_number
                    },
                    'trailing': {},
                    'sl_oid': -1,
                    'tp_oid': -1,
                    'entry_oid': -1,
                    'wait_for_fill': False
                }
                print(f"✅ {signal.upper()}: Opened {direction} {coin} ({format_num(quantity_coin, 4)} @ ${format_num(current_price, 4)} / Notional ${format_num(notional_usd, 2)} / Margin ${format_num(margin_usd, 2)} / Est. Liq: ${format_num(estimated_liq_price, 4)})")
                execution_report['executed'].append({
                    'coin': coin,
                    'signal': signal,
                    'confidence': confidence,
                    'classification': trend_classification,
                    'volume_ratio': volume_ratio,
                    'margin_usd': margin_usd
                })
                trade['runtime_decision'] = 'executed'

            elif signal == 'close_position':
                if not position:
                    print(f"⚠️ CLOSE {coin}: No position to close.")
                    execution_report['skipped'].append({'coin': coin, 'reason': 'no_position_to_close'})
                    trade['runtime_decision'] = 'skipped_no_position'
                    continue

                if live_trading:
                    live_result = self.execute_live_close(
                        coin=coin,
                        position=position,
                        current_price=current_price,
                        reason=trade.get('justification')
                    )
                    if not live_result.get('success'):
                        error_msg = live_result.get('error', 'unknown_error')
                        print(f"🚫 LIVE CLOSE FAILED: {coin} ({error_msg})")
                        execution_report['blocked'].append({
                            'coin': coin,
                            'reason': 'live_close_failed',
                            'error': error_msg
                        })
                        trade['runtime_decision'] = 'blocked_live_close'
                    else:
                        history_entry = live_result.get('history_entry')
                        if history_entry:
                            self.add_to_history(history_entry)
                        execution_report['executed'].append({
                            'coin': coin,
                            'signal': 'close_position',
                            'pnl': live_result.get('pnl'),
                            'direction': position.get('direction'),
                            'mode': 'live',
                            'order_id': live_result.get('order', {}).get('orderId')
                        })
                        trade['runtime_decision'] = 'executed_live'
                    continue

                sell_quantity = position['quantity']; direction = position.get('direction', 'long')
                entry_price = position['entry_price']
                margin_used = position.get('margin_usd', position.get('notional_usd', 0) / position.get('leverage', 1))

                profit = (current_price - entry_price) * sell_quantity if direction == 'long' else (entry_price - current_price) * sell_quantity
                if not live_trading:
                    self.current_balance += (margin_used + profit)

                print(f"✅ CLOSE (AI): Closed {direction} {coin} @ ${format_num(current_price, 4)} (PnL: ${format_num(profit, 2)})")
                execution_report['executed'].append({'coin': coin, 'signal': 'close_position', 'pnl': profit, 'direction': direction})
                trade['runtime_decision'] = 'executed'

                history_entry = {
                    "symbol": coin, "direction": direction, "entry_price": entry_price, "exit_price": current_price,
                    "quantity": position['quantity'], "notional_usd": position.get('notional_usd', 'N/A'), "pnl": profit,
                    "entry_time": position['entry_time'], "exit_time": datetime.now().isoformat(),
                    "leverage": position.get('leverage', 'N/A'), "close_reason": f"AI Decision: {trade.get('justification', 'N/A')}"
                }
                self.add_to_history(history_entry)
                del self.positions[coin]

            elif signal == 'hold':
                # For hold signals, just log the decision - no action needed
                if position:
                    print(f"ℹ️ HOLD: Holding {position.get('direction', 'long')} {coin} position.")
                else:
                    print(f"ℹ️ HOLD: Staying cash in {coin}.")
                execution_report['holds'].append({'coin': coin, 'has_position': bool(position)})
                trade['runtime_decision'] = 'hold'
            else: print(f"⚠️ Unknown signal '{signal}' for {coin}. Skipping.")

        self.last_execution_report = execution_report
# --- Main Bot Class ---
# format_num is global

class AlphaArenaDeepSeek:
    """Alpha Arena-like DeepSeek integration with auto TP/SL, dynamic sizing, and advanced risk management."""

    def __init__(self, api_key: str = None):
        self.market_data = RealMarketData()
        self.portfolio = PortfolioManager()
        self.deepseek = DeepSeekAPI(api_key)
        self.risk_manager = AdvancedRiskManager()
        self.invocation_count = 0 # Track AI calls since bot start
        self.tp_sl_timer = None
        self.is_running = False
        self.enhanced_exit_enabled = True  # Enhanced exit strategy control flag
        self.cycle_active = False  # Track whether a trading cycle is executing
        self.current_cycle_number = 0
        # Trend flip cooldown yönetimi PortfolioManager tarafında tutulur.
        self.latest_indicator_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.history_reset_interval = Config.HISTORY_RESET_INTERVAL

    def get_max_positions_for_cycle(self, cycle_number: int) -> int:
        """Cycle bazlı maximum pozisyon limiti - Kademeli artış sistemi, MAX_POSITIONS ile sınırlı"""
        from config import Config
        max_allowed = Config.MAX_POSITIONS
        
        if cycle_number == 1:
            return min(1, max_allowed)  # Cycle 1: max 1 pozisyon (veya MAX_POSITIONS)
        elif cycle_number == 2:
            return min(2, max_allowed)  # Cycle 2: max 2 pozisyon (veya MAX_POSITIONS)
        elif cycle_number == 3:
            return min(3, max_allowed)  # Cycle 3: max 3 pozisyon (veya MAX_POSITIONS)
        elif cycle_number == 4:
            return min(4, max_allowed)  # Cycle 4: max 4 pozisyon (veya MAX_POSITIONS)
        else:
            return max_allowed  # Cycle 5+: MAX_POSITIONS değerini kullan
    def _apply_directional_capacity_filter(self, decisions: Dict[str, Dict]) -> Tuple[Dict[str, Dict], bool]:
        """Convert entry signals to hold when directional capacity is full."""
        if not isinstance(decisions, dict):
            return decisions, False

        directional_counts = self.portfolio.count_positions_by_direction()
        limit = Config.SAME_DIRECTION_LIMIT
        blocked = {
            'long': directional_counts.get('long', 0) >= limit,
            'short': directional_counts.get('short', 0) >= limit
        }
        cooldowns = getattr(self.portfolio, 'directional_cooldowns', {'long': 0, 'short': 0})
        if (
            not blocked['long'] and not blocked['short'] and
            cooldowns.get('long', 0) == 0 and cooldowns.get('short', 0) == 0
        ):
            return decisions, False

        filtered: Dict[str, Dict] = {}
        changed = False
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                filtered[coin] = trade
                continue

            signal = trade.get('signal')
            direction = None
            if signal == 'buy_to_enter':
                direction = 'long'
            elif signal == 'sell_to_enter':
                direction = 'short'

            if direction and cooldowns.get(direction, 0) > 0:
                changed = True
                remaining = cooldowns.get(direction, 0)
                filtered[coin] = {
                    'signal': 'hold',
                    'justification': f'Directional cooldown active ({remaining} cycles remaining)'
                }
                print(f"⏸️ Directional cooldown: Blocking {direction.upper()} entry for {coin} ({remaining} cycles remaining).")
                continue

            if direction and blocked.get(direction):
                changed = True
                filtered[coin] = {
                    'signal': 'hold',
                    'justification': f"{direction.upper()} capacity full ({directional_counts.get(direction, 0)}/{limit}); evaluate exits or opposite-side setups."
                }
            else:
                filtered[coin] = trade

        return filtered, changed

    def maybe_reset_history(self, cycle_number: int):
        """Reset historical logs at configured intervals to prevent long-term bias."""
        interval = getattr(self, "history_reset_interval", 35)
        if interval <= 0:
            return
        cycles_elapsed = getattr(self.portfolio, "cycles_since_history_reset", 0)
        if cycles_elapsed >= interval:
            print(f"🧭 Bias control: {cycles_elapsed} cycles since last reset (interval {interval}). Resetting history.")
            self.portfolio.reset_historical_data(cycle_number)
            self.invocation_count = 0

    def check_trend_alignment(self, coin: str) -> bool:
        """Check if trends are aligned across multiple timeframes (1h + 15m + 3m)"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_15m = self.market_data.get_technical_indicators(coin, '15m')
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            
            if 'error' in indicators_htf or 'error' in indicators_15m or 'error' in indicators_3m:
                return False
            
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            price_15m = indicators_15m.get('current_price')
            ema20_15m = indicators_15m.get('ema_20')
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            
            # Trend alignment: All three timeframes in same direction (strongest signal)
            trend_aligned = (price_htf > ema20_htf and price_15m > ema20_15m and price_3m > ema20_3m) or \
                           (price_htf < ema20_htf and price_15m < ema20_15m and price_3m < ema20_3m)
            
            return trend_aligned
            
        except Exception as e:
            print(f"⚠️ Trend alignment error for {coin}: {e}")
            return False

    def check_momentum_alignment(self, coin: str) -> bool:
        """Check if momentum indicators are aligned across timeframes (1h + 15m + 3m)"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_15m = self.market_data.get_technical_indicators(coin, '15m')
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            
            if 'error' in indicators_htf or 'error' in indicators_15m or 'error' in indicators_3m:
                return False
            
            rsi_3m = indicators_3m.get('rsi_14', 50)
            rsi_15m = indicators_15m.get('rsi_14', 50)
            rsi_htf = indicators_htf.get('rsi_14', 50)
            macd_3m = indicators_3m.get('macd', 0)
            macd_15m = indicators_15m.get('macd', 0)
            macd_htf = indicators_htf.get('macd', 0)
            
            # Momentum alignment: All three timeframes showing same momentum direction (strongest signal)
            momentum_aligned = (rsi_3m > 50 and rsi_15m > 50 and rsi_htf > 50 and macd_3m > 0 and macd_15m > 0 and macd_htf > 0) or \
                              (rsi_3m < 50 and rsi_15m < 50 and rsi_htf < 50 and macd_3m < 0 and macd_15m < 0 and macd_htf < 0)
            
            return momentum_aligned
            
        except Exception as e:
            print(f"⚠️ Momentum alignment error for {coin}: {e}")
            return False

    def enhanced_trend_detection(self, coin: str) -> Dict[str, Any]:
        """Enhanced trend detection with simple trend strength and counter-trade detection"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_15m = self.market_data.get_technical_indicators(coin, '15m')
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            
            if 'error' in indicators_htf or 'error' in indicators_15m or 'error' in indicators_3m:
                return {'trend_strength': 0, 'trend_direction': 'NEUTRAL', 'ema_comparison': 'N/A', 'volume_confidence': 0.0}
            
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            ema50_htf = indicators_htf.get('ema_50')
            price_15m = indicators_15m.get('current_price')
            ema20_15m = indicators_15m.get('ema_20')
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            
            trend_label_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"

            # Nof1AI Blog Style: EMA20 vs EMA50 comparison
            ema_comparison = f"20-Period EMA: {format_num(ema20_htf)} vs. 50-Period EMA: {format_num(ema50_htf)}"
            
            # Simple trend strength calculation (used mainly for counter-trend context)
            trend_strength = 0
            trend_direction = 'NEUTRAL'
            
            # Higher timeframe EMA alignment (strong trend indicator) - Nof1AI Blog Style
            if ema20_htf > ema50_htf and price_htf > ema20_htf:
                trend_strength += 3  # Strong bullish (EMA20 > EMA50 + price > EMA20)
                trend_direction = 'STRONG_BULLISH'
            elif ema20_htf < ema50_htf and price_htf < ema20_htf:
                trend_strength += 3  # Strong bearish (EMA20 < EMA50 + price < EMA20)
                trend_direction = 'STRONG_BEARISH'
            elif ema20_htf > ema50_htf:
                trend_strength += 1  # Weak bullish (EMA20 > EMA50 but price < EMA20)
                trend_direction = 'WEAK_BULLISH'
            elif ema20_htf < ema50_htf:
                trend_strength += 1  # Weak bearish (EMA20 < EMA50 but price > EMA20)
                trend_direction = 'WEAK_BEARISH'
            
            # Multi-timeframe alignment bonus (1h + 15m + 3m)
            alignment_count = 0
            if price_htf > ema20_htf and price_15m > ema20_15m and price_3m > ema20_3m:
                alignment_count = 3
            elif price_htf < ema20_htf and price_15m < ema20_15m and price_3m < ema20_3m:
                alignment_count = 3
            elif (price_htf > ema20_htf and price_15m > ema20_15m) or (price_htf > ema20_htf and price_3m > ema20_3m) or (price_15m > ema20_15m and price_3m > ema20_3m):
                alignment_count = 2
            elif (price_htf < ema20_htf and price_15m < ema20_15m) or (price_htf < ema20_htf and price_3m < ema20_3m) or (price_15m < ema20_15m and price_3m < ema20_3m):
                alignment_count = 2
            
            if alignment_count >= 3:
                trend_strength += 2  # Strong multi-timeframe alignment (all 3 timeframes)
            elif alignment_count == 2:
                trend_strength += 1  # Medium multi-timeframe alignment (2 of 3 timeframes)
            
            # Volume Confirmation (Nof1AI Blog Style)
            volume_confidence = self.calculate_volume_confidence(coin)
            
            # Counter-trade detection information
            counter_trade_info = self.get_counter_trade_information(coin)
            
            return {
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'ema_comparison': ema_comparison,
                'price_vs_ema20_htf': 'ABOVE' if price_htf > ema20_htf else 'BELOW',
                'price_vs_ema20_3m': 'ABOVE' if price_3m > ema20_3m else 'BELOW',
                'volume_confidence': volume_confidence,
                'counter_trade_info': counter_trade_info,
                'trend_htf': trend_label_htf
            }
            
        except Exception as e:
            print(f"⚠️ Enhanced trend detection error for {coin}: {e}")
            return {'trend_strength': 0, 'trend_direction': 'NEUTRAL', 'ema_comparison': 'ERROR', 'volume_confidence': 0.0}

    def get_counter_trade_information(self, coin: str) -> Dict[str, Any]:
        """Get counter-trade information for AI decision making (information only, no blocking)"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            
            if 'error' in indicators_htf or 'error' in indicators_3m:
                return {'counter_trade_risk': 'UNKNOWN', 'conditions_met': 0, 'total_conditions': 5}
            
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            rsi_3m = indicators_3m.get('rsi_14', 50)
            volume_3m = indicators_3m.get('volume', 0)
            avg_volume_3m = indicators_3m.get('avg_volume', 1)
            macd_3m = indicators_3m.get('macd', 0)
            macd_signal_3m = indicators_3m.get('macd_signal', 0)
            
            trend_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"
            trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
            
            conditions_met = 0
            total_conditions = 5
            conditions_details: List[str] = []
            
            # Condition 1: 3m trend alignment
            if (trend_htf == "BULLISH" and price_3m < ema20_3m) or (trend_htf == "BEARISH" and price_3m > ema20_3m):
                conditions_met += 1
                conditions_details.append("✅ 3m trend alignment")
            else:
                conditions_details.append("❌ 3m trend misalignment")
            
            # Condition 2: Volume confirmation (>1.5x average)
            volume_ratio = volume_3m / avg_volume_3m if avg_volume_3m > 0 else 0
            if volume_ratio > 1.5:
                conditions_met += 1
                conditions_details.append(f"✅ Volume {volume_ratio:.1f}x average")
            else:
                conditions_details.append(f"❌ Volume {volume_ratio:.1f}x average (need >1.5x)")
            
            # Condition 3: Extreme RSI
            if (trend_htf == "BULLISH" and rsi_3m < 25) or (trend_htf == "BEARISH" and rsi_3m > 75):
                conditions_met += 1
                conditions_details.append(f"✅ Extreme RSI: {rsi_3m:.1f}")
            else:
                conditions_details.append(f"❌ RSI: {rsi_3m:.1f} (need <25 for LONG, >75 for SHORT)")
            
            # Condition 4: Strong technical levels (price near EMA)
            price_ema_distance = abs(price_3m - ema20_3m) / price_3m * 100 if price_3m else 100
            if price_ema_distance < 1.0:
                conditions_met += 1
                conditions_details.append(f"✅ Strong technical level ({price_ema_distance:.2f}% from EMA)")
            else:
                conditions_details.append(f"❌ Weak technical level ({price_ema_distance:.2f}% from EMA)")
            
            # Condition 5: MACD divergence
            if (trend_htf == "BULLISH" and macd_3m > macd_signal_3m) or (trend_htf == "BEARISH" and macd_3m < macd_signal_3m):
                conditions_met += 1
                conditions_details.append("✅ MACD divergence")
            else:
                conditions_details.append("❌ No MACD divergence")
            
            if conditions_met >= 4:
                risk_level = "LOW_RISK"
                recommendation = "STRONG COUNTER-TRADE SETUP - Consider with high confidence (>0.75)"
            elif conditions_met >= 3:
                risk_level = "MEDIUM_RISK"
                recommendation = "MODERATE COUNTER-TRADE SETUP - Consider with medium confidence (>0.65)"
            elif conditions_met >= 2:
                risk_level = "HIGH_RISK"
                recommendation = "WEAK COUNTER-TRADE SETUP - Avoid or use very low confidence"
            else:
                risk_level = "VERY_HIGH_RISK"
                recommendation = "NO COUNTER-TRADE SETUP - Focus on trend-following"
            
            return {
                'counter_trade_risk': risk_level,
                'conditions_met': conditions_met,
                'total_conditions': total_conditions,
                'recommendation': recommendation,
                'conditions_details': conditions_details,
                'trend_htf': trend_htf,
                'trend_3m': trend_3m,
                'volume_ratio': round(volume_ratio, 2),
                'rsi_3m': round(rsi_3m, 2),
                'price_ema_distance_pct': round(price_ema_distance, 2)
            }
            
        except Exception as e:
            print(f"⚠️ Counter-trade information error for {coin}: {e}")
            return {'counter_trade_risk': 'ERROR', 'conditions_met': 0, 'total_conditions': 5}

    def calculate_comprehensive_trend_strength(self, coin: str) -> Dict[str, Any]:
        """Calculate comprehensive trend strength using 5 technical indicators with weighted scoring"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            
            if 'error' in indicators_htf or 'error' in indicators_3m:
                return {'strength_score': 0, 'trend_direction': 'UNCLEAR', 'component_scores': {}}
            
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            ema50_htf = indicators_htf.get('ema_50')
            rsi_htf = indicators_htf.get('rsi_14', 50)
            macd_htf = indicators_htf.get('macd', 0)
            volume_htf = indicators_htf.get('volume', 0)
            avg_volume_htf = indicators_htf.get('avg_volume', 1)
            
            # 1. RSI Strength (20% weight)
            rsi_strength = self.analyze_rsi_strength(rsi_htf)
            
            # 2. MACD Strength (25% weight - most important)
            macd_strength = self.analyze_macd_strength(macd_htf)
            
            # 3. Volume Strength (15% weight)
            volume_strength = self.analyze_volume_strength(volume_htf, avg_volume_htf)
            
            # 4. Bollinger Bands Strength (20% weight)
            bb_strength = self.analyze_bollinger_bands_strength(indicators_htf)
            
            # 5. Moving Averages Strength (20% weight)
            ma_strength = self.analyze_moving_averages_strength(price_htf, ema20_htf, ema50_htf)
            
            # Weighted average - each indicator has different importance
            total_strength = (
                rsi_strength * 0.20 +      # %20 ağırlık
                macd_strength * 0.25 +     # %25 ağırlık (en önemli)
                volume_strength * 0.15 +   # %15 ağırlık
                bb_strength * 0.20 +       # %20 ağırlık
                ma_strength * 0.20         # %20 ağırlık
            )
            
            # Determine trend direction
            trend_direction = self.determine_trend_direction(price_htf, ema20_htf, ema50_htf, rsi_htf, macd_htf)
            
            return {
                'strength_score': total_strength,
                'trend_direction': trend_direction,
                'component_scores': {
                    'rsi': rsi_strength,
                    'macd': macd_strength, 
                    'volume': volume_strength,
                    'bollinger_bands': bb_strength,
                    'moving_averages': ma_strength
                },
                'confidence_level': self.get_confidence_level(total_strength)
            }
            
        except Exception as e:
            print(f"⚠️ Comprehensive trend strength error for {coin}: {e}")
            return {'strength_score': 0, 'trend_direction': 'UNCLEAR', 'component_scores': {}}

    def analyze_rsi_strength(self, rsi: float) -> float:
        """Analyze RSI strength (0-1 scale)"""
        if rsi > 70:
            return 0.9  # Overbought - strong trend continuation
        elif rsi > 60:
            return 0.7  # Bullish momentum
        elif rsi > 50:
            return 0.5  # Neutral bullish
        elif rsi > 40:
            return 0.3  # Neutral bearish
        elif rsi > 30:
            return 0.1  # Bearish momentum
        else:
            return 0.0  # Oversold - weak trend

    def analyze_macd_strength(self, macd: float) -> float:
        """Analyze MACD strength (0-1 scale)"""
        if macd > 0.01:
            return 1.0  # Strong bullish
        elif macd > 0.005:
            return 0.8  # Moderate bullish
        elif macd > 0:
            return 0.6  # Weak bullish
        elif macd > -0.005:
            return 0.4  # Weak bearish
        elif macd > -0.01:
            return 0.2  # Moderate bearish
        else:
            return 0.0  # Strong bearish

    def analyze_volume_strength(self, volume: float, avg_volume: float) -> float:
        """Analyze volume strength (0-1 scale)"""
        if avg_volume <= 0:
            return 0.0
            
        volume_ratio = volume / avg_volume
        
        if volume_ratio >= 1.8:  # High volume: >1.8x average
            return 1.0
        elif volume_ratio >= 1.3:  # Medium-high volume: >1.3x average
            return 0.8
        elif volume_ratio >= 0.8:  # Normal volume: >0.8x average
            return 0.6
        elif volume_ratio >= 0.5:  # Low volume: >0.5x average
            return 0.3
        else:  # Very low volume: <0.5x average
            return 0.1

    def analyze_bollinger_bands_strength(self, indicators: Dict) -> float:
        """Analyze Bollinger Bands strength (0-1 scale)"""
        try:
            price = indicators.get('current_price', 0)
            ema20 = indicators.get('ema_20', price)
            atr_14 = indicators.get('atr_14', 0)
            
            if atr_14 <= 0:
                return 0.5  # Neutral if no volatility data
                
            # Calculate distance from EMA as percentage of ATR
            distance = abs(price - ema20) / atr_14
            
            if distance > 2.0:
                return 1.0  # Strong trend (price far from EMA)
            elif distance > 1.0:
                return 0.7  # Moderate trend
            elif distance > 0.5:
                return 0.4  # Weak trend
            else:
                return 0.2  # No trend (consolidation)
                
        except Exception as e:
            print(f"⚠️ Bollinger Bands analysis error: {e}")
            return 0.5

    def analyze_moving_averages_strength(self, price: float, ema20: float, ema50: float) -> float:
        """Analyze Moving Averages strength (0-1 scale)"""
        try:
            # EMA alignment strength
            if ema20 > ema50 and price > ema20:
                return 1.0  # Strong bullish alignment
            elif ema20 < ema50 and price < ema20:
                return 1.0  # Strong bearish alignment
            elif ema20 > ema50:
                return 0.6  # Weak bullish alignment
            elif ema20 < ema50:
                return 0.6  # Weak bearish alignment
            else:
                return 0.3  # No clear alignment
                
        except Exception as e:
            print(f"⚠️ Moving Averages analysis error: {e}")
            return 0.5

    def determine_trend_direction(self, price: float, ema20: float, ema50: float, rsi: float, macd: float) -> str:
        """Determine overall trend direction based on multiple indicators"""
        bullish_signals = 0
        bearish_signals = 0
        
        # Price vs EMA20
        if price > ema20:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        # EMA20 vs EMA50
        if ema20 > ema50:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        # RSI direction
        if rsi > 50:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        # MACD direction
        if macd > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if bullish_signals >= 3:
            return "STRONG_BULLISH"
        elif bearish_signals >= 3:
            return "STRONG_BEARISH"
        elif bullish_signals > bearish_signals:
            return "WEAK_BULLISH"
        elif bearish_signals > bullish_signals:
            return "WEAK_BEARISH"
        else:
            return "NEUTRAL"

    def get_confidence_level(self, strength_score: float) -> str:
        """Get confidence level based on trend strength score"""
        if strength_score > 0.75:
            return "VERY_HIGH"
        elif strength_score > 0.60:
            return "HIGH" 
        elif strength_score > 0.45:
            return "MEDIUM"
        elif strength_score > 0.30:
            return "LOW"
        else:
            return "VERY_LOW"

    def calculate_volume_confidence(self, coin: str) -> float:
        """Calculate volume confidence based on current vs average volume (Nof1AI Blog Style)"""
        try:
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            if 'error' in indicators_3m:
                return 0.0
            
            current_volume = indicators_3m.get('volume', 0)
            avg_volume = indicators_3m.get('avg_volume', 0)
            
            if avg_volume <= 0:
                return 0.0
            
            # Nof1AI Blog Style: "Current Volume: 53.457 vs. Average Volume: 4329.191"
            volume_ratio = current_volume / avg_volume
            
            # Volume confidence scoring
            if volume_ratio >= 1.8:  # High volume: >1.8x average
                return 1.0
            elif volume_ratio >= 1.3:  # Medium-high volume: >1.3x average
                return 0.8
            elif volume_ratio >= 0.8:  # Normal volume: >0.8x average
                return 0.6
            elif volume_ratio >= 0.5:  # Low volume: >0.5x average
                return 0.3
            else:  # Very low volume: <0.5x average
                return 0.1
                
        except Exception as e:
            print(f"⚠️ Volume confidence calculation error for {coin}: {e}")
            return 0.0

    def calculate_volume_quality_score(
        self,
        coin: str,
        indicators_3m: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate volume quality score (0-100) based on Config thresholds"""
        try:
            if indicators_3m is None or not isinstance(indicators_3m, dict):
                indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            if 'error' in indicators_3m:
                return 0.0
            
            current_volume = indicators_3m.get('volume', 0)
            avg_volume = indicators_3m.get('avg_volume', 0)
            
            if avg_volume <= 0:
                return 0.0
            
            volume_ratio = current_volume / avg_volume
            
            # Calculate score based on Config thresholds
            if volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['excellent']:
                return 90.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['good']:
                return 75.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['fair']:
                return 60.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['poor']:
                return 40.0
            else:
                return 20.0
                
        except Exception as e:
            print(f"⚠️ Volume quality score calculation error for {coin}: {e}")
            return 0.0

    def should_enhance_short_sizing(self, coin: str) -> bool:
        """Check if short position should be enhanced (%15 daha büyük)"""
        try:
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            
            if 'error' in indicators_3m or 'error' in indicators_htf:
                return False
            
            # Enhanced short conditions:
            # 1. 3m RSI > 70 (aşırı alım)
            rsi_3m = indicators_3m.get('rsi_14', 50)
            # 2. Volume > 1.5x average
            volume_ratio = indicators_3m.get('volume', 0) / indicators_3m.get('avg_volume', 1)
            # 3. Higher timeframe trend bearish
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            trend_bearish = price_htf < ema20_htf
            
            # All conditions must be met
            return rsi_3m > 70 and volume_ratio > 1.5 and trend_bearish
            
        except Exception as e:
            print(f"⚠️ Enhanced short sizing check error for {coin}: {e}")
            return False

    def generate_advanced_exit_plan(self, coin: str, direction: str, entry_price: float) -> Dict[str, Any]:
        """Generate advanced exit plan with momentum failure detection (Nof1AI Blog Style)"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            
            if 'error' in indicators_htf or 'error' in indicators_3m:
                return {
                    'profit_target': None,
                    'stop_loss': None,
                    'invalidation_condition': 'Unable to generate exit plan due to data error'
                }
            
            current_price = indicators_3m.get('current_price', entry_price)
            atr_14 = indicators_htf.get('atr_14', 0)
            rsi_14 = indicators_htf.get('rsi_14', 50)
            ema_20 = indicators_htf.get('ema_20', current_price)
            htf_upper = HTF_LABEL.upper()
            
            # Calculate TP/SL based on ATR (Nof1AI Blog Style)
            if direction == 'long':
                # Long position: TP = entry + 2x ATR, SL = entry - 1x ATR
                profit_target = entry_price + (atr_14 * 2)
                stop_loss = entry_price - atr_14
                
                # Advanced invalidation conditions (Nof1AI Blog Style)
                if rsi_14 > 70:
                    invalidation_condition = f"If {htf_upper} RSI breaks back below 60, signaling momentum failure"
                elif rsi_14 < 40:
                    invalidation_condition = f"If {htf_upper} RSI breaks above 50, signaling momentum recovery"
                else:
                    invalidation_condition = f"If {htf_upper} price closes below {htf_upper} EMA20, signaling trend reversal"
                    
            else:  # short
                # Short position: TP = entry - 2x ATR, SL = entry + 1x ATR
                profit_target = entry_price - (atr_14 * 2)
                stop_loss = entry_price + atr_14
                
                # Advanced invalidation conditions (Nof1AI Blog Style)
                if rsi_14 < 30:
                    invalidation_condition = f"If {htf_upper} RSI breaks back above 40, signaling momentum failure"
                elif rsi_14 > 60:
                    invalidation_condition = f"If {htf_upper} RSI breaks below 50, signaling momentum recovery"
                else:
                    invalidation_condition = f"If {htf_upper} price closes above {htf_upper} EMA20, signaling trend reversal"
            
            return {
                'profit_target': round(profit_target, 4),
                'stop_loss': round(stop_loss, 4),
                'invalidation_condition': invalidation_condition,
                'atr_based': True,
                'rsi_context': f"{htf_upper} RSI: {rsi_14:.1f}",
                'ema_context': f"{htf_upper} EMA20: {ema_20:.4f}"
            }
            
        except Exception as e:
            print(f"⚠️ Advanced exit plan generation error for {coin}: {e}")
            return {
                'profit_target': None,
                'stop_loss': None,
                'invalidation_condition': f'Error generating exit plan: {str(e)}'
            }

    def get_market_regime(self) -> str:
        """Detect overall market regime across all coins"""
        try:
            bullish_count = 0
            bearish_count = 0
            
            for coin in self.market_data.available_coins:
                indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
                if 'error' in indicators_htf:
                    continue
                
                price = indicators_htf.get('current_price')
                ema20 = indicators_htf.get('ema_20')
                
                if price > ema20:
                    bullish_count += 1
                else:
                    bearish_count += 1
            
            if bullish_count >= 4:  # At least 4 coins bullish
                return "BULLISH"
            elif bearish_count >= 4:  # At least 4 coins bearish
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            print(f"⚠️ Market regime detection error: {e}")
            return "NEUTRAL"

    def detect_market_regime(
        self,
        coin: str,
        indicators_htf: Optional[Dict[str, Any]] = None,
        indicators_3m: Optional[Dict[str, Any]] = None,
        indicators_15m: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Detect market condition based on multi-timeframe indicators.
        
        Rule: For a coin to be BULLISH, 1h must be bullish AND (3m OR 15m must be bullish).
        For a coin to be BEARISH, 1h must be bearish AND (3m OR 15m must be bearish).
        Otherwise, return NEUTRAL.
        """
        try:
            if indicators_htf is None:
                indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            if not isinstance(indicators_htf, dict) or 'error' in indicators_htf:
                return "UNCLEAR"
            
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')

            if not isinstance(price_htf, (int, float)) or not isinstance(ema20_htf, (int, float)) or ema20_htf == 0:
                return "UNCLEAR"

            # Determine 1h trend
            delta_htf = (price_htf - ema20_htf) / ema20_htf
            price_neutral = abs(delta_htf) <= Config.EMA_NEUTRAL_BAND_PCT
            htf_trend = None
            if not price_neutral:
                htf_trend = "bullish" if delta_htf > 0 else "bearish"
            else:
                return "NEUTRAL"

            # Get 3m trend
            if indicators_3m is None:
                indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            trend_3m = None
            if isinstance(indicators_3m, dict) and 'error' not in indicators_3m:
                price_3m = indicators_3m.get('current_price')
                ema20_3m = indicators_3m.get('ema_20')
                if isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)) and ema20_3m > 0:
                    trend_3m = "bullish" if price_3m >= ema20_3m else "bearish"

            # Get 15m trend
            if indicators_15m is None:
                indicators_15m = self.market_data.get_technical_indicators(coin, '15m')
            trend_15m = None
            if isinstance(indicators_15m, dict) and 'error' not in indicators_15m:
                price_15m = indicators_15m.get('current_price')
                ema20_15m = indicators_15m.get('ema_20')
                if isinstance(price_15m, (int, float)) and isinstance(ema20_15m, (int, float)) and ema20_15m > 0:
                    trend_15m = "bullish" if price_15m >= ema20_15m else "bearish"

            # Apply rule: 1h + (3m OR 15m) must align for BULLISH/BEARISH
            if htf_trend == "bullish":
                # For BULLISH: 1h bullish AND (3m bullish OR 15m bullish)
                if trend_3m == "bullish" or trend_15m == "bullish":
                    return "BULLISH"
                else:
                    # 1h bullish but shorter timeframes bearish = NEUTRAL (counter-trend opportunity)
                    return "NEUTRAL"
            elif htf_trend == "bearish":
                # For BEARISH: 1h bearish AND (3m bearish OR 15m bearish)
                if trend_3m == "bearish" or trend_15m == "bearish":
                    return "BEARISH"
                else:
                    # 1h bearish but shorter timeframes bullish = NEUTRAL (counter-trend opportunity)
                    return "NEUTRAL"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            print(f"⚠️ Regime detection error for {coin}: {e}")
            return "UNCLEAR"
    def get_trading_context(self) -> Dict[str, Any]:
        """Get historical context from recent cycles - Enhanced with 5 cycle analysis"""
        try:
            if len(self.portfolio.cycle_history) < 2:
                return {
                    "recent_decisions": [], 
                    "market_behavior": "Initial cycles - observing",
                    "total_cycles_analyzed": len(self.portfolio.cycle_history),
                    "performance_trend": "No data yet"
                }
            
            # Use last 5 cycles for enhanced analysis
            recent_cycles = self.portfolio.cycle_history[-5:]
            recent_decisions = []
            
            for cycle in recent_cycles:
                decisions = cycle.get('decisions', {})
                for coin, trade in decisions.items():
                    if isinstance(trade, dict) and trade.get('signal') in ['buy_to_enter', 'sell_to_enter']:
                        recent_decisions.append({
                            'coin': coin,
                            'signal': trade.get('signal'),
                            'cycle': cycle.get('cycle'),
                            'confidence': trade.get('confidence', 0.5),
                            'timestamp': cycle.get('timestamp')
                        })
            
            # Enhanced market behavior analysis
            market_behavior = self._analyze_market_behavior(recent_cycles)
            performance_trend = self._analyze_performance_trend(recent_cycles)
            
            return {
                "recent_decisions": recent_decisions,
                "market_behavior": market_behavior,
                "performance_trend": performance_trend,
                "total_cycles_analyzed": len(recent_cycles),
                "analysis_period": f"Last {len(recent_cycles)} cycles"
            }
            
        except Exception as e:
            print(f"⚠️ Trading context error: {e}")
            return {
                "recent_decisions": [], 
                "market_behavior": "Error in context analysis",
                "performance_trend": "Unknown",
                "total_cycles_analyzed": 0
            }

    def _analyze_market_behavior(self, recent_cycles: List[Dict]) -> str:
        """Analyze market behavior based on recent trading decisions"""
        if not recent_cycles:
            return "No recent activity"
        
        recent_decisions = []
        for cycle in recent_cycles:
            decisions = cycle.get('decisions', {})
            for coin, trade in decisions.items():
                if isinstance(trade, dict) and trade.get('signal') in ['buy_to_enter', 'sell_to_enter']:
                    recent_decisions.append(trade)
        
        if not recent_decisions:
            return "Consolidating - No recent entries"
        
        long_count = sum(1 for d in recent_decisions if d.get('signal') == 'buy_to_enter')
        short_count = sum(1 for d in recent_decisions if d.get('signal') == 'sell_to_enter')
        
        # Enhanced analysis with confidence weighting
        long_confidence = sum(d.get('confidence', 0.5) for d in recent_decisions if d.get('signal') == 'buy_to_enter')
        short_confidence = sum(d.get('confidence', 0.5) for d in recent_decisions if d.get('signal') == 'sell_to_enter')
        
        if long_count > short_count and long_confidence > short_confidence:
            return f"Strong Bullish bias ({long_count} longs, avg confidence: {long_confidence/long_count:.2f})"
        elif short_count > long_count and short_confidence > long_confidence:
            return f"Strong Bearish bias ({short_count} shorts, avg confidence: {short_confidence/short_count:.2f})"
        elif long_count > short_count:
            return f"Bullish bias ({long_count} longs)"
        elif short_count > long_count:
            return f"Bearish bias ({short_count} shorts)"
        else:
            return "Balanced market"

    def _analyze_performance_trend(self, recent_cycles: List[Dict]) -> str:
        """Analyze performance trend based on recent cycles"""
        if len(recent_cycles) < 3:
            return "Insufficient data for trend analysis"
        
        # Analyze decision patterns
        entry_signals = 0
        hold_signals = 0
        close_signals = 0
        
        for cycle in recent_cycles:
            decisions = cycle.get('decisions', {})
            for trade in decisions.values():
                if isinstance(trade, dict):
                    signal = trade.get('signal')
                    if signal == 'buy_to_enter' or signal == 'sell_to_enter':
                        entry_signals += 1
                    elif signal == 'hold':
                        hold_signals += 1
                    elif signal == 'close_position':
                        close_signals += 1
        
        total_signals = entry_signals + hold_signals + close_signals
        if total_signals == 0:
            return "No trading activity"
        
        entry_rate = entry_signals / total_signals
        close_rate = close_signals / total_signals
        
        if entry_rate > 0.4 and close_rate < 0.2:
            return "Aggressive accumulation phase"
        elif close_rate > 0.3:
            return "Profit-taking phase"
        elif hold_signals > entry_signals + close_signals:
            return "Consolidation phase"
        else:
            return "Balanced trading"

    def get_enhanced_context(self) -> Dict[str, Any]:
        """Get enhanced context for AI decision making"""
        try:
            from enhanced_context_provider import EnhancedContextProvider
            provider = EnhancedContextProvider()
            return provider.generate_enhanced_context()
        except Exception as e:
            print(f"⚠️ Enhanced context error: {e}")
            return {"error": f"Enhanced context failed: {str(e)}"}

    def format_position_context(self, position_context: Dict) -> str:
        """Format position context for prompt"""
        if not position_context:
            return "No open positions"
        
        formatted = ""
        for symbol, data in position_context.items():
            pnl = data.get('unrealized_pnl', 0)
            remaining_pct = data.get('remaining_to_target_pct')
            if remaining_pct is None:
                progress = data.get('profit_target_progress', 0)
                remaining_pct = max(0.0, round(100 - progress, 2))
            time_in_trade = data.get('time_in_trade_minutes', 0)
            direction = (data.get('direction') or 'long').upper()
            trend_alignment = str(data.get('trend_alignment', 'unknown')).upper()
            trend_context = data.get('trend_context') or {}
            trend_entry = str(trend_context.get('trend_at_entry', 'unknown')).upper()
            entry_cycle = trend_context.get('cycle')
            confidence = data.get('confidence')

            cycle_note = f", entry cycle {entry_cycle}" if isinstance(entry_cycle, (int, float)) else ""
            confidence_note = f", entry confidence {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
            counter_note = ""
            if trend_alignment == 'COUNTER_TREND':
                counter_note = " | Counter-trend position — hold unless invalidation triggers."

            formatted += (
                f"  {symbol}: ${pnl:.2f} PnL, {remaining_pct}% to target, {time_in_trade}min in trade | "
                f"{direction} {trend_alignment} vs HTF trend {trend_entry}{cycle_note}{confidence_note}{counter_note}\n"
            )
        return formatted

    def format_market_regime_context(self, market_regime: Dict) -> str:
        """Format market regime context for prompt"""
        if not market_regime:
            return "Market regime: Unknown"
        
        current = market_regime.get('current_regime', 'unknown')
        strength = market_regime.get('regime_strength', 0)
        bull_count = market_regime.get('bullish_count', 0)
        bear_count = market_regime.get('bearish_count', 0)
        neutral_count = market_regime.get('neutral_count', 0)
        total_coins = market_regime.get('total_coins', bull_count + bear_count + neutral_count)
        coin_regimes = market_regime.get('coin_regimes', {})
        
        formatted = (
            f"Global regime: {current} "
            f"(strength {strength}, bullish={bull_count}, bearish={bear_count}, neutral={neutral_count}, total={total_coins})\n"
        )
        if coin_regimes:
            formatted += "  Coin regimes:\n"
            for coin, data in coin_regimes.items():
                regime = data.get('regime', 'unknown')
                score = data.get('score', 0)
                price_relation = data.get('price_vs_ema20', 'unknown')
                formatted += f"    - {coin}: {regime} (score {score}, price {price_relation} EMA20)\n"
        else:
            formatted += "  Coin regimes: No data\n"
        return formatted.rstrip()

    def format_performance_insights(self, performance_insights: Dict) -> str:
        """Format performance insights for prompt"""
        if not performance_insights:
            return "No performance insights available"
        
        insights = performance_insights.get('insights', [])
        if not insights:
            return "No performance insights available"
        
        formatted = ""
        for insight in insights:
            formatted += f"  • {insight}\n"
        return formatted

    def format_directional_feedback(self, directional_feedback: Dict) -> str:
        """Format long/short feedback for prompt"""
        if not directional_feedback:
            return "No directional feedback available"
        
        lines = []
        for direction in ("long", "short"):
            stats = directional_feedback.get(direction, {})
            trades = stats.get("trades", 0)
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            win_rate = stats.get("win_rate", 0.0)
            avg_pnl = stats.get("avg_pnl", 0.0)
            total_pnl = stats.get("total_pnl", 0.0)
            lines.append(
                f"  {direction.upper()}: trades={trades}, wins={wins}, losses={losses}, win_rate={win_rate}%, avg_pnl=${avg_pnl:.2f}, total_pnl=${total_pnl:.2f}"
            )
        return "\n".join(lines)

    def format_risk_context(self, risk_context: Dict) -> str:
        """Format risk context for prompt"""
        if not risk_context:
            return "Risk context: Unknown"
        
        total_risk = risk_context.get('total_risk_usd', 0)
        position_count = risk_context.get('position_count', 0)
        
        return f"Total Risk: ${total_risk:.2f}, Positions: {position_count}"

    def _get_counter_trade_analysis_from_indicators(self, all_indicators: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
        """Get counter-trade analysis using pre-fetched indicators (OPTIMIZATION: No re-fetch)"""
        analysis = []
        
        for coin in self.market_data.available_coins:
            try:
                # Use pre-fetched indicators
                indicators_3m = all_indicators.get(coin, {}).get('3m', {})
                indicators_htf = all_indicators.get(coin, {}).get(HTF_INTERVAL, {})
                
                if 'error' in indicators_3m or 'error' in indicators_htf:
                    analysis.append(f"❌ {coin}: Data error - cannot analyze counter-trade conditions")
                    continue
                
                # Extract key indicators
                price_htf = indicators_htf.get('current_price')
                ema20_htf = indicators_htf.get('ema_20')
                price_3m = indicators_3m.get('current_price')
                ema20_3m = indicators_3m.get('ema_20')
                rsi_3m = indicators_3m.get('rsi_14', 50)
                volume_3m = indicators_3m.get('volume', 0)
                avg_volume_3m = indicators_3m.get('avg_volume', 1)
                macd_3m = indicators_3m.get('macd', 0)
                macd_signal_3m = indicators_3m.get('macd_signal', 0)
                
                # Determine higher timeframe trend direction
                trend_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"
                trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
                
                # Counter-trade conditions analysis
                conditions_met = 0
                total_conditions = 5
                conditions_details = []
                
                # Condition 1: 3m trend alignment
                if (trend_htf == "BULLISH" and trend_3m == "BEARISH") or (trend_htf == "BEARISH" and trend_3m == "BULLISH"):
                    conditions_met += 1
                    conditions_details.append("✅ 3m trend alignment")
                else:
                    conditions_details.append("❌ 3m trend misalignment")
                
                # Condition 2: Volume confirmation (>1.5x average)
                volume_ratio = volume_3m / avg_volume_3m if avg_volume_3m > 0 else 0
                if volume_ratio > 1.5:
                    conditions_met += 1
                    conditions_details.append(f"✅ Volume {volume_ratio:.1f}x average")
                else:
                    conditions_details.append(f"❌ Volume {volume_ratio:.1f}x average (need >1.5x)")
                
                # Condition 3: Extreme RSI
                if (trend_htf == "BULLISH" and rsi_3m < 25) or (trend_htf == "BEARISH" and rsi_3m > 75):
                    conditions_met += 1
                    conditions_details.append(f"✅ Extreme RSI: {rsi_3m:.1f}")
                else:
                    conditions_details.append(f"❌ RSI: {rsi_3m:.1f} (need <25 for LONG, >75 for SHORT)")
                
                # Condition 4: Strong technical levels (price near EMA)
                price_ema_distance = abs(price_3m - ema20_3m) / price_3m * 100
                if price_ema_distance < 1.0:
                    conditions_met += 1
                    conditions_details.append(f"✅ Strong technical level ({price_ema_distance:.2f}% from EMA)")
                else:
                    conditions_details.append(f"❌ Weak technical level ({price_ema_distance:.2f}% from EMA)")
                
                # Condition 5: MACD divergence
                if (trend_htf == "BULLISH" and macd_3m > macd_signal_3m) or (trend_htf == "BEARISH" and macd_3m < macd_signal_3m):
                    conditions_met += 1
                    conditions_details.append("✅ MACD divergence")
                else:
                    conditions_details.append("❌ No MACD divergence")
                
                # Determine counter-trade risk level and recommendation
                if conditions_met >= 4:
                    risk_level = "LOW_RISK"
                    recommendation = "STRONG COUNTER-TRADE SETUP - Consider with high confidence (>0.75)"
                elif conditions_met >= 3:
                    risk_level = "MEDIUM_RISK"
                    recommendation = "MODERATE COUNTER-TRADE SETUP - Consider with medium confidence (>0.65)"
                elif conditions_met >= 2:
                    risk_level = "HIGH_RISK"
                    recommendation = "WEAK COUNTER-TRADE SETUP - Avoid or use very low confidence"
                else:
                    risk_level = "VERY_HIGH_RISK"
                    recommendation = "NO COUNTER-TRADE SETUP - Focus on trend-following"
                
                # Build analysis string for this coin
                htf_upper = HTF_LABEL.upper()
                coin_analysis = f"""
{coin} COUNTER-TRADE ANALYSIS:
  {htf_upper} Trend: {trend_htf} | 3m Trend: {trend_3m}
  Conditions Met: {conditions_met}/{total_conditions}
  Risk Level: {risk_level}
  Recommendation: {recommendation}
  Conditions:
    {chr(10).join(f'    {detail}' for detail in conditions_details)}
"""
                analysis.append(coin_analysis)
                
            except Exception as e:
                analysis.append(f"❌ {coin}: Analysis error - {str(e)}")
        
        # Combine all analyses
        if not analysis:
            return "No counter-trade analysis available due to data errors"
        
        return "\n".join(analysis)

    def format_suggestions(self, suggestions: List[str]) -> str:
        """Format suggestions for prompt"""
        if not suggestions:
            return "No suggestions at this time"
        
        formatted = ""
        for suggestion in suggestions:
            formatted += f"  • {suggestion}\n"
        return formatted

    def format_trend_reversal_analysis(self, trend_reversal_analysis: Dict) -> str:
        """Format trend reversal analysis for prompt"""
        if not trend_reversal_analysis or 'error' in trend_reversal_analysis:
            return "Trend reversal analysis: No data available"
        
        formatted = ""
        for coin, analysis in trend_reversal_analysis.items():
            if coin == 'error':
                continue
                
            reversal_signals = analysis.get('reversal_signals', [])
            if not reversal_signals:
                continue
                
            formatted += f"\n{coin} TREND REVERSAL SIGNALS:\n"
            for signal in reversal_signals:
                signal_type = signal.get('type', 'Unknown')
                strength = signal.get('strength', 'Unknown')
                description = signal.get('description', 'No description')
                formatted += f"  • {signal_type} ({strength}): {description}\n"
        
        if not formatted:
            return "Trend reversal analysis: No reversal signals detected"
        
        return formatted

    def format_volume_ratio(self, volume: Any, avg_volume: Any) -> str:
        """Format volume ratio with guard rails for extremely low values."""
        try:
            if not isinstance(volume, (int, float)) or not isinstance(avg_volume, (int, float)):
                return "N/A"
            if avg_volume <= 0:
                return "N/A"
            ratio = volume / avg_volume
            if ratio == 0:
                return "0.00x"
            if ratio < 0.0005:
                return "<0.0005x"
            if ratio < 0.01:
                return f"{ratio:.4f}x"
            if ratio < 1:
                return f"{ratio:.3f}x"
            return f"{ratio:.2f}x"
        except Exception:
            return "N/A"

    def format_list(self, lst, precision=4):
        """Helper function to format lists for prompt display"""
        if not isinstance(lst, list): return []
        return [format_num(x, precision) if x is not None else 'N/A' for x in lst]

    def _fetch_all_indicators_parallel(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Fetch all indicators for all coins in parallel (OPTIMIZATION: Single fetch, shared data)"""
        print("🔄 Fetching all indicators in parallel...")
        start_time = time.time()
        
        all_indicators = {}  # {coin: {interval: indicators}}
        all_sentiment = {}   # {coin: sentiment}
        
        def fetch_indicators_for_coin(coin: str) -> tuple:
            """Fetch all indicators and sentiment for a single coin"""
            try:
                indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
                indicators_15m = self.market_data.get_technical_indicators(coin, '15m')
                indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
                sentiment = self.market_data.get_market_sentiment(coin)
                return (coin, {
                    '3m': indicators_3m,
                    '15m': indicators_15m,
                    HTF_INTERVAL: indicators_htf
                }, sentiment)
            except Exception as e:
                print(f"⚠️ Error fetching indicators for {coin}: {e}")
                return (coin, {
                    '3m': {'error': str(e)},
                    '15m': {'error': str(e)},
                    HTF_INTERVAL: {'error': str(e)}
                }, {})
        
        # Fetch all indicators in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_indicators_for_coin, coin): coin 
                      for coin in self.market_data.available_coins}
            
            for future in as_completed(futures):
                coin, indicators, sentiment = future.result()
                all_indicators[coin] = indicators
                all_sentiment[coin] = sentiment
        
        elapsed = time.time() - start_time
        print(f"✅ Fetched all indicators in {elapsed:.2f}s (parallel)")
        
        return all_indicators, all_sentiment

    def generate_alpha_arena_prompt(self) -> str:
        """Generate prompt with enhanced data, indicator history and AI decision context"""
        current_time = datetime.now(); minutes_running = int((current_time - self.portfolio.start_time).total_seconds() / 60)
        # Use internal invocation counter, don't increment here, do it in run_cycle
        # self.invocation_count += 1

        # OPTIMIZATION 1 & 2: Fetch all indicators in parallel ONCE, then share
        all_indicators, all_sentiment = self._fetch_all_indicators_parallel()
        
        # OPTIMIZATION 3: Get enhanced context and other data in parallel (non-blocking)
        # These don't need fresh market data, so can run in parallel
        enhanced_context = self.get_enhanced_context()
        
        # Get counter-trade analysis using pre-fetched indicators (no re-fetch)
        counter_trade_analysis = self._get_counter_trade_analysis_from_indicators(all_indicators)
        
        # Get trend reversal detection using pre-fetched indicators (OPTIMIZATION 3: No re-fetch)
        from performance_monitor import PerformanceMonitor
        performance_monitor = PerformanceMonitor()
        trend_reversal_analysis = performance_monitor.detect_trend_reversal_for_all_coins(
            self.market_data.available_coins,
            indicators_cache=all_indicators  # Pass pre-fetched indicators
        )
        
        bias_metrics = getattr(self, 'latest_bias_metrics', self.get_directional_bias_metrics())
        bias_lines = []
        for side in ('long', 'short'):
            stats = bias_metrics.get(side, {})
            bias_lines.append(
                f"  • {side.upper()}: net_pnl=${format_num(stats.get('net_pnl', 0.0), 2)}, "
                f"trades={stats.get('trades', 0)}, win_rate={format_num(stats.get('win_rate', 0.0), 2)}%, "
                f"rolling_avg=${format_num(stats.get('rolling_avg', 0.0), 2)}, consecutive_losses={stats.get('consecutive_losses', 0)}"
            )
        bias_section = "\n".join(bias_lines) if bias_lines else "  • No directional trades recorded"

        # Get cooldown status
        cooldowns = getattr(self.portfolio, 'directional_cooldowns', {'long': 0, 'short': 0})
        cooldown_lines = []
        for side in ('long', 'short'):
            cycles_remaining = cooldowns.get(side, 0)
            if cycles_remaining > 0:
                stats = bias_metrics.get(side, {})
                consecutive_losses = stats.get('consecutive_losses', 0)
                loss_streak_usd = stats.get('loss_streak_loss_usd', 0.0)
                reason = []
                if consecutive_losses >= 3:
                    reason.append(f"{consecutive_losses} consecutive losses")
                if loss_streak_usd >= 5.0:
                    reason.append(f"${loss_streak_usd:.2f} total loss")
                reason_str = " + ".join(reason) if reason else "unknown"
                cooldown_lines.append(
                    f"  • {side.upper()}: COOLDOWN ACTIVE ({cycles_remaining} cycles remaining) - Reason: {reason_str}"
                )
            else:
                cooldown_lines.append(f"  • {side.upper()}: No cooldown (active)")
        cooldown_section = "\n".join(cooldown_lines) if cooldown_lines else "  • No cooldowns active"

        # Get coin cooldown status
        coin_cooldowns = getattr(self.portfolio, 'coin_cooldowns', {})
        coin_cooldown_lines = []
        if coin_cooldowns:
            for coin, cycles in sorted(coin_cooldowns.items()):
                if cycles > 0:
                    coin_cooldown_lines.append(f"  • {coin}: COOLDOWN ACTIVE ({cycles} cycles remaining - previous loss)")
        coin_cooldown_section = "\n".join(coin_cooldown_lines) if coin_cooldown_lines else "  • No coin cooldowns active"

        recent_flips = self.portfolio.get_recent_trend_flip_summary()
        flip_history_window = getattr(self.portfolio, 'trend_flip_history_window', self.portfolio.trend_flip_cooldown)
        if recent_flips:
            trend_flip_section = "\n".join(f"  • {entry}" for entry in recent_flips)
        else:
            trend_flip_section = f"  • No trend flips in last {flip_history_window} cycles"
        
        prompt = f"""
USER_PROMPT:
It has been {minutes_running} minutes since you started trading. The current time is {current_time} and you've been invoked {self.invocation_count} times. Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. Below that is your current account information, value, performance, positions, etc.

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST
Timeframes note: Unless stated otherwise in a section title, intraday series are provided at 3‑minute intervals. If a coin uses a different interval, it is explicitly stated in that coin's section.

{'='*20} REAL-TIME COUNTER-TRADE ANALYSIS {'='*20}

We pre-compute the standard 5 counter-trend conditions for every coin. Review these findings first; only recalc if you detect inconsistencies or need extra validation.

{counter_trade_analysis}

{'='*20} TREND REVERSAL DETECTION {'='*20}

All notes below are informational statistics about potential reversals; evaluate them independently before acting.

{self.format_trend_reversal_analysis(trend_reversal_analysis)}

{'='*20} ENHANCED DECISION CONTEXT (Non-binding suggestions) {'='*20}

Metrics and remarks in this section are informational only. You must weigh them yourself before making any trading decision.

POSITION MANAGEMENT CONTEXT:
{self.format_position_context(enhanced_context.get('position_context', {}))}

MARKET REGIME ANALYSIS:
{self.format_market_regime_context(enhanced_context.get('market_regime', {}))}

PERFORMANCE INSIGHTS:
{self.format_performance_insights(enhanced_context.get('performance_insights', {}))}

DIRECTIONAL FEEDBACK (LONG vs SHORT):
{self.format_directional_feedback(enhanced_context.get('directional_feedback', {}))}

DIRECTIONAL PERFORMANCE SNAPSHOT (Last 20 trades max):
{bias_section}

DIRECTIONAL COOLDOWN STATUS (CRITICAL - DO NOT PROPOSE TRADES IN COOLDOWN DIRECTIONS):
{cooldown_section}

⚠️ IMPORTANT: If a direction (LONG or SHORT) is in cooldown, you MUST NOT propose any new trades in that direction. The system will block them, but you should avoid proposing them in the first place. Cooldown is activated after 3 consecutive losses OR $5+ total loss in a direction.

COIN COOLDOWN STATUS (CRITICAL - DO NOT PROPOSE TRADES FOR COINS IN COOLDOWN):
{coin_cooldown_section}

⚠️ IMPORTANT: If a coin is in cooldown, you MUST NOT propose any new trades for that coin (LONG or SHORT). The system will block them, but you should avoid proposing them in the first place. Coin cooldown is activated after a loss on that coin and lasts for 1 cycle.

RECENT TREND FLIP GUARD (Cooldown = {self.portfolio.trend_flip_cooldown} cycles | History = {flip_history_window} cycles):
{trend_flip_section}

RISK MANAGEMENT CONTEXT:
{self.format_risk_context(enhanced_context.get('risk_context', {}))}

SUGGESTIONS (Non-binding):
{self.format_suggestions(enhanced_context.get('suggestions', []))}

REMEMBER: These are suggestions only. You make the final trading decisions based on your systematic analysis.
"""

        directional_counts = self.portfolio.count_positions_by_direction()
        positions_by_direction: Dict[str, List[Dict[str, Any]]] = {'long': [], 'short': []}
        now_ts = datetime.now()
        for coin, position in self.portfolio.positions.items():
            direction = position.get('direction', 'long')
            if direction not in positions_by_direction:
                continue
            pnl = position.get('unrealized_pnl', 0.0)
            entry_time_str = position.get('entry_time')
            minutes_in_trade = None
            if entry_time_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_time_str)
                    minutes_in_trade = max(0, int((now_ts - entry_dt).total_seconds() // 60))
                except Exception:
                    minutes_in_trade = None
            positions_by_direction[direction].append({
                'coin': coin,
                'pnl': pnl,
                'minutes': minutes_in_trade,
                'loss_cycles': position.get('loss_cycle_count', 0)
            })

        long_open = directional_counts.get('long', 0)
        short_open = directional_counts.get('short', 0)
        same_direction_limit = Config.SAME_DIRECTION_LIMIT
        total_open_positions = len(self.portfolio.positions)
        cycle_for_limits = max(1, getattr(self, 'current_cycle_number', 1))
        cycle_position_cap = self.get_max_positions_for_cycle(cycle_for_limits)

        slot_lines = [
            f"  • Total open positions: {total_open_positions}/{cycle_position_cap} (cycle cap)",
            f"  • Long slots used: {long_open}/{same_direction_limit}",
            f"  • Short slots used: {short_open}/{same_direction_limit}"
        ]
        if long_open >= same_direction_limit:
            weakest_long = None
            if positions_by_direction['long']:
                weakest_long = min(positions_by_direction['long'], key=lambda x: x['pnl'])
            if weakest_long:
                wl_minutes = f"{weakest_long['minutes']}min" if weakest_long['minutes'] is not None else "N/A"
                slot_lines.append(
                    f"  • Weakest LONG → {weakest_long['coin']} (PnL ${weakest_long['pnl']:.2f}, in trade {wl_minutes}, "
                    f"loss_cycles={weakest_long['loss_cycles']}). Evaluate trimming/closing this before proposing a new long."
                )
            slot_lines.append(
                "  • Long capacity FULL → System blocks new longs. Provide either (a) a close/trim plan for a current long "
                "OR (b) a SHORT setup (trend-following if regimes align, otherwise counter-trend with ≥0.75 confidence)."
            )
        if short_open >= same_direction_limit:
            weakest_short = None
            if positions_by_direction['short']:
                weakest_short = min(positions_by_direction['short'], key=lambda x: x['pnl'])
            if weakest_short:
                ws_minutes = f"{weakest_short['minutes']}min" if weakest_short['minutes'] is not None else "N/A"
                slot_lines.append(
                    f"  • Weakest SHORT → {weakest_short['coin']} (PnL ${weakest_short['pnl']:.2f}, in trade {ws_minutes}, "
                    f"loss_cycles={weakest_short['loss_cycles']}). Evaluate trimming/closing this before proposing a new short."
                )
            slot_lines.append(
                f"  • Short capacity FULL → System blocks new shorts. Provide either (a) a close/trim plan for a current short "
                f"OR (b) a LONG alternative (trend-following with {HTF_LABEL} alignment or counter-trend with ≥0.75 confidence)."
            )

        prompt += f"\n{'='*20} POSITION SLOT STATUS {'='*20}\n" + "\n".join(slot_lines) + "\n"

        # --- Loop through available coins ---
        # OPTIMIZATION: Use pre-fetched indicators instead of re-fetching
        self.latest_indicator_cache = {}

        for coin in self.market_data.available_coins:
            prompt += f"\n{'='*20} ALL {coin} DATA {'='*20}\n"
            # Use pre-fetched indicators (no re-fetch)
            indicators_3m = all_indicators.get(coin, {}).get('3m', {})
            indicators_15m = all_indicators.get(coin, {}).get('15m', {})
            indicators_htf = all_indicators.get(coin, {}).get(HTF_INTERVAL, {})
            sentiment = all_sentiment.get(coin, {})
            self.latest_indicator_cache[coin] = {
                '3m': copy.deepcopy(indicators_3m),
                '15m': copy.deepcopy(indicators_15m),
                HTF_INTERVAL: copy.deepcopy(indicators_htf)
            }
            
            # Add market regime detection
            market_regime = self.detect_market_regime(coin, indicators_htf=indicators_htf, indicators_3m=indicators_3m, indicators_15m=indicators_15m)
            prompt += f"--- MARKET REGIME: {market_regime} ---\n"
            
            prompt += f"--- Market Sentiment for {coin} Perps ---\n"
            prompt += f"Open Interest: Latest: {format_num(sentiment.get('open_interest', 'N/A'), 2)}\n"
            funding_rate = sentiment.get('funding_rate', 0.0)
            prompt += f"Funding Rate: {format_num(funding_rate, 8)} ({format_num(funding_rate*100, 4)}%)\n\n"

            # --- Inner function to format indicators ---
            def format_indicators(indicators, prefix=""):
                if not isinstance(indicators, dict) or 'error' in indicators:
                     error_msg = indicators.get('error', 'Unknown error') if isinstance(indicators, dict) else 'Invalid indicator data'
                     return f"{prefix}Error fetching indicator data: {error_msg}\n"
                # Format numbers using global helper
                output = f"{prefix}current_price = {format_num(indicators.get('current_price', 'N/A'))}\n"
                output += f"{prefix}Mid prices (last {len(indicators.get('price_series',[]))}): {self.format_list(indicators.get('price_series',[]))}\n"
                output += f"{prefix}EMA indicators (20‑period): {self.format_list(indicators.get('ema_20_series',[]))}\n"
                if 'rsi_7_series' in indicators: output += f"{prefix}RSI indicators (7‑Period): {self.format_list(indicators.get('rsi_7_series',[]), precision=3)}\n"
                output += f"{prefix}RSI indicators (14‑Period): {self.format_list(indicators.get('rsi_14_series',[]), precision=3)}\n"
                output += f"{prefix}MACD indicators: {self.format_list(indicators.get('macd_series',[]))}\n"
                atr_3 = indicators.get('atr_3'); atr_14 = indicators.get('atr_14'); atr_str = ""
                if atr_3 is not None and pd.notna(atr_3): atr_str += f"{prefix}3‑Period ATR: {format_num(atr_3)} vs "
                atr_str += f"14‑Period ATR: {format_num(atr_14)}\n"; output += atr_str
                current_volume = indicators.get('volume', 'N/A')
                avg_volume = indicators.get('avg_volume', 'N/A')
                output += f"{prefix}Current Volume: {format_num(current_volume, 3)} vs. Average Volume: {format_num(avg_volume, 3)}\n"
                output += f"{prefix}Volume ratio (current/avg): {self.format_volume_ratio(current_volume, avg_volume)}\n"
                return output
            # --- End inner function ---

            prompt += "--- Intraday series (3‑minute intervals) ---\n"; prompt += format_indicators(indicators_3m)
            prompt += "\n--- Medium-term context (15‑minute intervals) ---\n"; prompt += format_indicators(indicators_15m)
            prompt += f"\n--- Longer‑term context ({HTF_LABEL} timeframe) ---\n"; prompt += format_indicators(indicators_htf)

            # --- Add current position details if open ---
            if coin in self.portfolio.positions:
                position = self.portfolio.positions[coin]; prompt += "\n--- CURRENT OPEN POSITION & YOUR PLAN ---\n"
                prompt += f"You have an open {position.get('direction', 'long').upper()} position.\n"; prompt += f"  Symbol: {position.get('symbol', 'N/A')}\n"
                prompt += f"  Quantity: {format_num(position.get('quantity', 0), 6)}\n"; prompt += f"  Entry Price: ${format_num(position.get('entry_price', 0))}\n"
                prompt += f"  Current Price: ${format_num(position.get('current_price', 0))}\n"; prompt += f"  Liquidation Price (Est.): ${format_num(position.get('liquidation_price', 0))}\n"
                prompt += f"  Unrealized PnL: ${format_num(position.get('unrealized_pnl', 0), 2)}\n"; prompt += f"  Leverage: {position.get('leverage', 1)}x\n"
                prompt += f"  Notional Value: ${format_num(position.get('notional_usd', 0), 2)}\n"
                
                # Calculate position duration
                entry_time_str = position.get('entry_time')
                position_duration_minutes = None
                position_duration_hours = None
                if entry_time_str:
                    try:
                        entry_dt = datetime.fromisoformat(entry_time_str)
                        position_duration_minutes = max(0, int((datetime.now() - entry_dt).total_seconds() // 60))
                        position_duration_hours = position_duration_minutes / 60.0
                    except Exception:
                        pass
                
                if position_duration_minutes is not None:
                    if position_duration_hours >= 1:
                        prompt += f"  Position Duration: {position_duration_hours:.1f} hours ({position_duration_minutes} minutes)\n"
                    else:
                        prompt += f"  Position Duration: {position_duration_minutes} minutes\n"
                
                # Get current trend state
                trend_info = self.portfolio.update_trend_state(coin, indicators_htf, indicators_3m)
                current_trend = trend_info.get('trend', 'unknown')
                trend_direction = position.get('direction', 'long').lower()
                
                # Determine 3m momentum
                price_3m = indicators_3m.get('current_price')
                ema20_3m = indicators_3m.get('ema_20')
                rsi_3m = indicators_3m.get('rsi_14', indicators_3m.get('rsi_7', 50))
                momentum_3m = 'unknown'
                if isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)) and ema20_3m > 0:
                    if price_3m > ema20_3m:
                        momentum_3m = 'bullish'
                    elif price_3m < ema20_3m:
                        momentum_3m = 'bearish'
                
                # Determine 15m momentum
                price_15m = indicators_15m.get('current_price')
                ema20_15m = indicators_15m.get('ema_20')
                rsi_15m = indicators_15m.get('rsi_14', indicators_15m.get('rsi_7', 50))
                momentum_15m = 'unknown'
                if isinstance(price_15m, (int, float)) and isinstance(ema20_15m, (int, float)) and ema20_15m > 0:
                    if price_15m > ema20_15m:
                        momentum_15m = 'bullish'
                    elif price_15m < ema20_15m:
                        momentum_15m = 'bearish'
                
                # Check for potential trend reversal using HTF trend, 15m momentum, and 3m momentum
                trend_reversal_warning = ""
                reversal_signals = []
                
                # HTF trend reversal check
                if current_trend == 'bullish' and trend_direction == 'short':
                    reversal_signals.append(f"{HTF_LABEL} trend flipped to BULLISH")
                elif current_trend == 'bearish' and trend_direction == 'long':
                    reversal_signals.append(f"{HTF_LABEL} trend flipped to BEARISH")
                
                # 15m momentum reversal check (medium-term confirmation)
                if momentum_15m == 'bullish' and trend_direction == 'short':
                    reversal_signals.append("15m momentum turned BULLISH")
                elif momentum_15m == 'bearish' and trend_direction == 'long':
                    reversal_signals.append("15m momentum turned BEARISH")
                
                # 3m momentum reversal check (more sensitive, earlier signal)
                if momentum_3m == 'bullish' and trend_direction == 'short':
                    reversal_signals.append("3m momentum turned BULLISH")
                elif momentum_3m == 'bearish' and trend_direction == 'long':
                    reversal_signals.append("3m momentum turned BEARISH")
                
                # Generate warning based on signal strength
                trend_reversal_warning = ""
                if reversal_signals:
                    # Count signals: HTF=structural, 15m=medium, 3m=short
                    htf_signal = any(f"{HTF_LABEL}" in s for s in reversal_signals)
                    signal_15m = any("15m" in s for s in reversal_signals)
                    signal_3m = any("3m" in s for s in reversal_signals)
                    
                    # If 1h + 15m + 3m all show reversal, this is NOT a reversal - it's the trend itself
                    # Only create reversal signal if 15m+3m oppose position (with or without 1h)
                    if htf_signal and signal_15m and signal_3m:
                        # All timeframes aligned - this is trend continuation, not reversal
                        # Don't create reversal warning in this case
                        trend_reversal_warning = ""
                    elif signal_15m and signal_3m:
                        # 15m + 3m both show reversal (strong reversal signal)
                        signal_strength = "STRONG"
                        signals_text = " & ".join([s for s in reversal_signals if "15m" in s or "3m" in s])
                        
                        if trend_direction == 'short':
                            trend_reversal_warning = f"ℹ️ {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a SHORT position but momentum is showing bullish signs. "
                            trend_reversal_warning += "15m and 3m momentum both show bullish signs - strong reversal signal. This can be a counter-trend opportunity. Evaluate your exit plan and consider if the position thesis is still valid."
                        else:  # long position
                            trend_reversal_warning = f"ℹ️ {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a LONG position but momentum is showing bearish signs. "
                            trend_reversal_warning += "15m and 3m momentum both show bearish signs - strong reversal signal. This can be a counter-trend opportunity. Evaluate your exit plan and consider if the position thesis is still valid."
                    elif signal_3m:
                        # Only 3m shows reversal (medium reversal signal)
                        signal_strength = "MEDIUM"
                        signals_text = " & ".join([s for s in reversal_signals if "3m" in s])
                        
                        if trend_direction == 'short':
                            trend_reversal_warning = f"ℹ️ {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a SHORT position but 3m momentum is showing bullish signs. "
                            trend_reversal_warning += "3m momentum shows bullish signs - medium reversal signal. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                        else:  # long position
                            trend_reversal_warning = f"ℹ️ {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a LONG position but 3m momentum is showing bearish signs. "
                            trend_reversal_warning += "3m momentum shows bearish signs - medium reversal signal. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                    elif signal_15m:
                        # Only 15m shows reversal (informational)
                        signal_strength = "INFORMATIONAL"
                        signals_text = " & ".join([s for s in reversal_signals if "15m" in s])
                        
                        if trend_direction == 'short':
                            trend_reversal_warning = f"ℹ️ {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a SHORT position but 15m momentum is showing bullish signs. "
                            trend_reversal_warning += "15m momentum shows bullish signs - this is informational context. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                        else:  # long position
                            trend_reversal_warning = f"ℹ️ {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a LONG position but 15m momentum is showing bearish signs. "
                            trend_reversal_warning += "15m momentum shows bearish signs - this is informational context. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                    else:
                        # Only HTF signal (shouldn't happen, but handle it)
                        signal_strength = "INFORMATIONAL"
                        signals_text = " & ".join(reversal_signals)
                        
                        if trend_direction == 'short':
                            trend_reversal_warning = f"ℹ️ {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a SHORT position but momentum is showing bullish signs. "
                            trend_reversal_warning += "Short-term momentum shows bullish signs - this is informational context. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                        else:  # long position
                            trend_reversal_warning = f"ℹ️ {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a LONG position but momentum is showing bearish signs. "
                            trend_reversal_warning += "Short-term momentum shows bearish signs - this is informational context. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                
                # Extended position duration warning
                if position_duration_hours is not None and position_duration_hours >= 4:
                    if trend_reversal_warning:
                        trend_reversal_warning += f"\n  ℹ️ POSITION DURATION: This {trend_direction.upper()} position has been open for {position_duration_hours:.1f} hours. Review your exit plan and ensure it's still aligned with current market conditions."
                    else:
                        trend_reversal_warning = f"ℹ️ POSITION DURATION: This {trend_direction.upper()} position has been open for {position_duration_hours:.1f} hours. This is informational - ensure your exit plan remains valid."
                
                if trend_reversal_warning:
                    prompt += f"\n  {trend_reversal_warning}\n"
                
                prompt += f"  Current {HTF_LABEL} Trend: {current_trend.upper()}\n"
                prompt += f"  Current 15m Momentum: {momentum_15m.upper()}\n"
                prompt += f"  Current 3m Momentum: {momentum_3m.upper()}\n"
                if isinstance(rsi_15m, (int, float)):
                    prompt += f"  15m RSI: {rsi_15m:.1f}\n"
                if isinstance(rsi_3m, (int, float)):
                    prompt += f"  3m RSI: {rsi_3m:.1f}\n"
                
                exit_plan = position.get('exit_plan', {}); prompt += f"  YOUR ACTIVE EXIT PLAN:\n"
                prompt += f"    Profit Target: {exit_plan.get('profit_target', 'N/A')}\n"; prompt += f"    Stop Loss: {exit_plan.get('stop_loss', 'N/A')}\n"
                prompt += f"    Invalidation: {exit_plan.get('invalidation_condition', 'N/A')}\n"; prompt += f"  Your Confidence: {position.get('confidence', 'N/A')}\n"
                prompt += f"  Estimated Risk USD: {position.get('risk_usd', 'N/A')}\n"; prompt += "REMINDER: You can only 'hold' or 'close_position'.\n"
        # --- End coin loop ---

        self.portfolio.indicator_cache = copy.deepcopy(self.latest_indicator_cache)


        # Add historical context section
        trading_context = self.get_trading_context()
        
        # Calculate current risk status - NEW SIMPLIFIED SYSTEM
        total_margin_used = sum(pos.get('margin_usd', 0) for pos in self.portfolio.positions.values())
        current_positions_count = len(self.portfolio.positions)
        max_positions = 5
        
        prompt += f"""
{'='*20} HISTORICAL CONTEXT (Last {trading_context['total_cycles_analyzed']} Cycles) {'='*20}

Market Behavior: {trading_context['market_behavior']}
Recent Trading Decisions: {json.dumps(trading_context['recent_decisions'], indent=2)}
{'='*20} REAL-TIME RISK STATUS {'='*20}

CURRENT STATUS: {current_positions_count} positions open, ${format_num(total_margin_used, 2)} margin used
AVAILABLE CASH: ${format_num(self.portfolio.current_balance, 2)}
TRADING LIMITS:
- Minimum position: $10
- Maximum positions: 5
- Available cash protection: Never below ${format_num(self.portfolio.current_balance * 0.10, 2)}
- Position sizing: Automatic based on confidence (up to 40% of available cash)

{'='*20} HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE {'='*20}

Current Total Return (percent): {format_num(self.portfolio.total_return, 2)}%
Available Cash: {format_num(self.portfolio.current_balance, 2)}
Current Account Value: {format_num(self.portfolio.total_value, 2)}
Sharpe Ratio: {format_num(self.portfolio.sharpe_ratio, 3)}

Current live positions & performance:"""

        if not self.portfolio.positions: 
            prompt += " No open positions. (100% cash)"
        else:
            for coin, pos in self.portfolio.positions.items():
                prompt += f"\n{{'symbol': '{coin}', 'quantity': {format_num(pos.get('quantity', 0), 4)}, 'entry_price': {format_num(pos.get('entry_price', 0))}, 'current_price': {format_num(pos.get('current_price', 0))}, 'liquidation_price': {format_num(pos.get('liquidation_price', 0))}, 'unrealized_pnl': {format_num(pos.get('unrealized_pnl', 0), 2)}, 'leverage': {pos.get('leverage', 1)}, 'exit_plan': {json.dumps(pos.get('exit_plan', {}))}, 'confidence': {pos.get('confidence', 0.5)}, 'risk_usd': {pos.get('risk_usd', 'N/A')}, 'sl_oid': -1, 'tp_oid': -1, 'wait_for_fill': False, 'entry_oid': -1, 'notional_usd': {format_num(pos.get('notional_usd', 0), 2)}}}"
        return prompt

    def generate_alpha_arena_prompt_json(self) -> str:
        """
        Generate hybrid JSON prompt with structured data sections.
        Uses JSON for data, plain text for instructions and warnings.
        """
        from prompt_json_builders import (
            build_metadata_json,
            build_counter_trade_json,
            build_trend_reversal_json,
            build_enhanced_context_json,
            build_cooldown_status_json,
            build_position_slot_json,
            build_market_data_json,
            build_portfolio_json,
            build_risk_status_json,
            build_historical_context_json,
            build_directional_bias_json,
            build_trend_flip_guard_json
        )
        from prompt_json_utils import safe_json_dumps, create_json_section, compare_token_usage
        from prompt_json_schemas import JSON_PROMPT_VERSION
        from config import Config
        
        current_time = datetime.now()
        minutes_running = int((current_time - self.portfolio.start_time).total_seconds() / 60)
        
        # Fetch all indicators in parallel (same as original)
        all_indicators, all_sentiment = self._fetch_all_indicators_parallel()
        
        # Get enhanced context and other data
        enhanced_context = self.get_enhanced_context()
        counter_trade_analysis = self._get_counter_trade_analysis_from_indicators(all_indicators)
        
        # Get trend reversal detection
        from performance_monitor import PerformanceMonitor
        performance_monitor = PerformanceMonitor()
        trend_reversal_analysis = performance_monitor.detect_trend_reversal_for_all_coins(
            self.market_data.available_coins,
            indicators_cache=all_indicators
        )
        
        # Get cooldown status
        directional_cooldowns = getattr(self.portfolio, 'directional_cooldowns', {'long': 0, 'short': 0})
        coin_cooldowns = getattr(self.portfolio, 'coin_cooldowns', {})
        counter_trend_cooldown = getattr(self.portfolio, 'counter_trend_cooldown', 0)
        relaxed_countertrend_cycles = getattr(self.portfolio, 'relaxed_countertrend_cycles', 0)
        
        # Get trading context
        trading_context = self.get_trading_context()
        
        # Get directional bias metrics (for performance snapshot)
        bias_metrics = getattr(self, 'latest_bias_metrics', self.get_directional_bias_metrics())
        
        # Get trend flip summary
        recent_flips = self.portfolio.get_recent_trend_flip_summary()
        flip_history_window = getattr(self.portfolio, 'trend_flip_history_window', self.portfolio.trend_flip_cooldown)
        
        # Build JSON sections
        compact = Config.JSON_PROMPT_COMPACT
        
        # Metadata
        metadata_json = build_metadata_json(minutes_running, current_time, self.invocation_count)
        
        # Counter-trade analysis
        counter_trade_json = build_counter_trade_json(
            counter_trade_analysis,
            all_indicators,
            self.market_data.available_coins,
            HTF_INTERVAL
        )
        
        # Trend reversal
        trend_reversal_json = build_trend_reversal_json(
            trend_reversal_analysis,
            self.portfolio.positions
        )
        
        # Enhanced context
        enhanced_context_json = build_enhanced_context_json(enhanced_context)
        
        # Cooldown status
        cooldown_status_json = build_cooldown_status_json(
            directional_cooldowns,
            coin_cooldowns,
            counter_trend_cooldown,
            relaxed_countertrend_cycles
        )
        
        # Position slot status
        max_positions = self.get_max_positions_for_cycle(max(1, getattr(self, 'current_cycle_number', 1)))
        position_slot_json = build_position_slot_json(self.portfolio.positions, max_positions)
        
        # Market data (per coin)
        market_data_json = []
        for coin in self.market_data.available_coins:
            indicators_3m = all_indicators.get(coin, {}).get('3m', {})
            indicators_15m = all_indicators.get(coin, {}).get('15m', {})
            indicators_htf = all_indicators.get(coin, {}).get(HTF_INTERVAL, {})
            sentiment = all_sentiment.get(coin, {})
            
            # Detect market regime
            market_regime = self.detect_market_regime(coin, indicators_htf=indicators_htf, indicators_3m=indicators_3m, indicators_15m=indicators_15m)
            
            # Get position if exists
            position = self.portfolio.positions.get(coin)
            
            coin_market_data = build_market_data_json(
                coin,
                market_regime,
                sentiment,
                indicators_3m,
                indicators_15m,
                indicators_htf,
                position,
                max_series_length=Config.JSON_SERIES_MAX_LENGTH
            )
            market_data_json.append(coin_market_data)
        
        # Portfolio
        portfolio_json = build_portfolio_json(self.portfolio)
        
        # Risk status
        risk_status_json = build_risk_status_json(self.portfolio, max_positions)
        
        # Historical context
        historical_context_json = build_historical_context_json(trading_context)
        
        # Directional bias (performance snapshot)
        directional_bias_json = build_directional_bias_json(bias_metrics)
        
        # Trend flip guard
        trend_flip_guard_json = build_trend_flip_guard_json(
            recent_flips,
            self.portfolio.trend_flip_cooldown,
            flip_history_window
        )
        
        # Build hybrid prompt
        prompt = f"""
USER_PROMPT:
It has been {minutes_running} minutes since you started trading. The current time is {current_time} and you've been invoked {self.invocation_count} times. Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. Below that is your current account information, value, performance, positions, etc.

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST
Timeframes note: Unless stated otherwise in a section title, intraday series are provided at 3‑minute intervals. If a coin uses a different interval, it is explicitly stated in that coin's section.

{'='*20} REAL-TIME COUNTER-TRADE ANALYSIS {'='*20}

We pre-compute the standard 5 counter-trend conditions for every coin. Review these findings first; only recalc if you detect inconsistencies or need extra validation.

{create_json_section("COUNTER_TRADE_ANALYSIS", counter_trade_json, compact=compact)}

{'='*20} TREND REVERSAL DETECTION {'='*20}

All notes below are informational statistics about potential reversals; evaluate them independently before acting.

{create_json_section("TREND_REVERSAL_DATA", trend_reversal_json, compact=compact)}

{'='*20} ENHANCED DECISION CONTEXT (Non-binding suggestions) {'='*20}

Metrics and remarks in this section are informational only. You must weigh them yourself before making any trading decision.

{create_json_section("ENHANCED_CONTEXT", enhanced_context_json, compact=compact)}

DIRECTIONAL PERFORMANCE SNAPSHOT (Last 20 trades max):
{create_json_section("DIRECTIONAL_BIAS", directional_bias_json, compact=compact)}

⚠️ IMPORTANT: If a direction (LONG or SHORT) is in cooldown, you MUST NOT propose any new trades in that direction. The system will block them, but you should avoid proposing them in the first place. Cooldown is activated after 3 consecutive losses OR $5+ total loss in a direction.

{create_json_section("COOLDOWN_STATUS", cooldown_status_json, compact=compact)}

⚠️ IMPORTANT: If a coin is in cooldown, you MUST NOT propose any new trades for that coin (LONG or SHORT). The system will block them, but you should avoid proposing them in the first place. Coin cooldown is activated after a loss on that coin and lasts for 1 cycle.

RECENT TREND FLIP GUARD (Cooldown = {self.portfolio.trend_flip_cooldown} cycles | History = {flip_history_window} cycles):
{create_json_section("TREND_FLIP_GUARD", trend_flip_guard_json, compact=compact)}

{'='*20} POSITION SLOT STATUS {'='*20}

{create_json_section("POSITION_SLOTS", position_slot_json, compact=compact)}

{'='*20} MARKET DATA {'='*20}

All market data is provided in JSON format below. Each coin contains:
- market_regime: Current market regime (BULLISH/BEARISH/NEUTRAL)
- sentiment: Open Interest and Funding Rate
- timeframes: 3m, 15m, {HTF_INTERVAL} indicators with historical series
- position: Current position details (if exists)

{create_json_section("MARKET_DATA", market_data_json, compact=compact)}

{'='*20} HISTORICAL CONTEXT (Last {trading_context.get('total_cycles_analyzed', 0)} Cycles) {'='*20}

{create_json_section("HISTORICAL_CONTEXT", historical_context_json, compact=compact)}

{'='*20} REAL-TIME RISK STATUS {'='*20}

{create_json_section("RISK_STATUS", risk_status_json, compact=compact)}

{'='*20} HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE {'='*20}

{create_json_section("PORTFOLIO", portfolio_json, compact=compact)}
"""
        
        # Validate JSON if enabled
        if Config.VALIDATE_JSON_PROMPTS:
            from prompt_json_schemas import validate_json_against_schema, get_full_prompt_schema
            # Note: Full validation would require building the complete JSON structure
            # For now, we validate individual sections
            pass
        
        return prompt

    def parse_ai_response(self, response: str) -> Dict:
        """Parse the AI response and clean up hold decisions"""
        thoughts = "Error: Could not parse AI response."; decisions = {}
        try:
            parts = response.split("DECISIONS", 1)
            if len(parts) == 2: thoughts_raw = parts[0].replace("CHAIN_OF_THOUGHTS", "").strip(); json_part = parts[1].strip(); thoughts = thoughts_raw if thoughts_raw else "No thoughts."
            else: print("⚠️ AI response keywords missing."); json_part = response; thoughts = "Keywords missing."
            start_idx = json_part.find('{'); end_idx = json_part.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = json_part[start_idx:end_idx]
                try:
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, dict): 
                        # Clean up hold decisions - remove unnecessary 0 values
                        decisions = self._clean_ai_decisions(parsed_json)
                    else: print(f"❌ Parsed JSON not dict: {type(parsed_json)}"); thoughts += f"\nError: Parsed JSON not dict."
                except json.JSONDecodeError as json_e: print(f"❌ JSON decode error: {json_e}"); print(f"   Invalid JSON: {json_str}"); thoughts = f"JSON Error: {json_e}\nRaw:\n{response}"
            else: print("⚠️ No valid JSON block found."); thoughts += f"\nError: No JSON block."
        except Exception as e: print(f"❌ General parse error: {e}"); thoughts = f"Parse Error: {e}\nRaw:\n{response}"
        return {"chain_of_thoughts": thoughts, "decisions": decisions}

    def _clean_ai_decisions(self, decisions: Dict) -> Dict:
        """Clean up AI decisions - preserve position data for hold signals"""
        cleaned_decisions = {}
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                cleaned_decisions[coin] = trade
                continue
                
            signal = trade.get('signal')
            if signal == 'hold':
                # For hold signals, preserve all position data for risk management
                cleaned_trade = {'signal': 'hold'}
                
                # If there's an open position, preserve ALL position data
                if coin in self.portfolio.positions:
                    position = self.portfolio.positions[coin]
                    cleaned_trade.update({
                        'leverage': position.get('leverage', 1),
                        'quantity_usd': position.get('margin_usd', 0),
                        'confidence': position.get('confidence', 0.5),
                        'profit_target': position.get('exit_plan', {}).get('profit_target'),
                        'stop_loss': position.get('exit_plan', {}).get('stop_loss'),
                        'risk_usd': position.get('risk_usd', 0),
                        'invalidation_condition': position.get('exit_plan', {}).get('invalidation_condition'),
                        'entry_price': position.get('entry_price', 0),
                        'current_price': position.get('current_price', 0),
                        'unrealized_pnl': position.get('unrealized_pnl', 0),
                        'notional_usd': position.get('notional_usd', 0),
                        'direction': position.get('direction', 'long')
                    })
                cleaned_decisions[coin] = cleaned_trade
            else:
                # For other signals, keep all data as is
                cleaned_decisions[coin] = trade
                
        return cleaned_decisions

    def check_coin_rotation(self, coin: str) -> bool:
        """Coin rotation disabled - always allow trading"""
        # Coin rotation system disabled - always return True to allow trading
        return True

    def detect_market_regime_overall(self) -> str:
        """Detect overall market regime across all coins"""
        try:
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            for coin in self.market_data.available_coins:
                indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
                if 'error' in indicators_htf:
                    continue
                
                price = indicators_htf.get('current_price')
                ema20 = indicators_htf.get('ema_20')
                
                if not isinstance(price, (int, float)) or not isinstance(ema20, (int, float)) or ema20 == 0:
                    continue

                delta = (price - ema20) / ema20
                if abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT:
                    neutral_count += 1
                elif price > ema20:
                    bullish_count += 1
                else:
                    bearish_count += 1
            
            if bullish_count >= 4:
                return "BULLISH"
            if bearish_count >= 4:
                return "BEARISH"
            if neutral_count >= 4:
                return "NEUTRAL"

            if bullish_count > bearish_count:
                return "BULLISH" if bullish_count >= 3 else "NEUTRAL"
            if bearish_count > bullish_count:
                return "BEARISH" if bearish_count >= 3 else "NEUTRAL"
                return "NEUTRAL"
                
        except Exception as e:
            print(f"⚠️ Market regime detection error: {e}")
            return "NEUTRAL"

    def calculate_optimal_cycle_frequency(self) -> int:
        """Calculate optimal cycle frequency based on market volatility"""
        try:
            atr_values = []
            # Tüm coin'leri dahil et (ASTER dahil)
            for coin in self.market_data.available_coins:
                indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
                if 'error' not in indicators_3m:
                    atr = indicators_3m.get('atr_14', 0)
                    # Küçük ATR değerlerini de dahil et (floating-point hassasiyetini düzelt)
                    if atr is not None and atr > 1e-6:  # 0.000001'den büyük olanları al
                        atr_values.append(atr)
                        print(f"📊 {coin} ATR: {atr:.6f}")
            
            if not atr_values:
                print("⚠️ No valid ATR values found, using default 2 minutes")
                return 120
            
            avg_atr = sum(atr_values) / len(atr_values)
            print(f"📊 Average ATR: {avg_atr:.6f}")
            
            # Adjust cycle frequency based on volatility
            if avg_atr < 0.3:    # Düşük volatility
                return 240       # 4 dakikada bir cycle
            elif avg_atr < 0.6:  # Orta volatility  
                return 180       # 3 dakikada bir cycle
            else:                # Yüksek volatility
                return 120       # 2 dakikada bir cycle
                
        except Exception as e:
            print(f"⚠️ Cycle frequency calculation error: {e}")
            return 120  # Default 2 minutes

    def adjust_position_size_by_trend(self, base_size: float, market_regime: str, signal: str) -> float:
        """Adjust position size based on market regime and signal direction"""
        if market_regime == "BULLISH" and signal == "buy_to_enter":
            return base_size * 1.2  # %20 daha büyük long
        elif market_regime == "BEARISH" and signal == "sell_to_enter":
            return base_size * 1.2  # %20 daha büyük short
        else:
            return base_size * 0.8  # %20 daha küçük counter-trend

    def track_performance_metrics(self, cycle_number: int):
        """Her cycle'da temel performans metriklerini kaydet"""
        try:
            metrics = {
                "cycle": cycle_number,
                "timestamp": datetime.now().isoformat(),
                "total_value": self.portfolio.total_value,
                "total_return": self.portfolio.total_return,
                "sharpe_ratio": self.portfolio.sharpe_ratio,
                "open_positions": len(self.portfolio.positions),
                "available_cash": self.portfolio.current_balance,
                "total_trades": self.portfolio.trade_count
            }
            
            # Performance history dosyasına kaydet
            performance_history = safe_file_read("performance_history.json", [])
            performance_history.append(metrics)
            safe_file_write("performance_history.json", performance_history[-100:])  # Son 100 cycle
            
        except Exception as e:
            print(f"⚠️ Performance tracking error: {e}")

    def should_run_performance_analysis(self, cycle_number: int) -> bool:
        """10 cycle'da bir veya kritik durumlarda analiz çalıştır"""
        # Her 10 cycle'da bir
        if cycle_number % 10 == 0:
            return True
        
        # Büyük PnL değişikliklerinde
        if abs(self.portfolio.total_return) > 10:  # %10'dan fazla değişim
            return True
        
        # Çok fazla pozisyon açıldığında
        if len(self.portfolio.positions) >= 4:
            return True
        
        return False

    def run_trading_cycle(self, cycle_number: int):
        """Run a single trading cycle with auto TP/SL and enhanced features"""
        print(f"\n{'='*80}\n🔄 TRADING CYCLE {cycle_number} | ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}")
        
        # Check bot control at cycle start
        control = self._read_bot_control()
        if control.get("status") == "paused":
            print(f"⏸️ Cycle {cycle_number} SKIPPED - Bot is PAUSED")
            return
        elif control.get("status") == "stopped":
            print(f"🛑 Cycle {cycle_number} STOPPED - Bot STOP command received")
            return
        
        self.current_cycle_number = cycle_number
        self.portfolio.current_cycle_number = cycle_number
        if hasattr(self.portfolio, 'tick_cooldowns'):
            self.portfolio.tick_cooldowns()
        self.market_data.clear_preloaded_indicators()
        self.portfolio.cycles_since_history_reset += 1
        self.maybe_reset_history(cycle_number)
        self.latest_bias_metrics = self.portfolio.get_directional_bias_metrics()
        self.portfolio.latest_bias_metrics = self.latest_bias_metrics
        prompt, thoughts, decisions = "N/A", "N/A", {}
        self.cycle_active = True
        cycle_timing: Dict[str, float] = {}
        try:
            # Enhanced exit strategy control - pause during cycle
            print("⏸️ Enhanced exit strategy paused during cycle")
            self.enhanced_exit_enabled = False
            
            # Track performance metrics every cycle
            self.track_performance_metrics(cycle_number)
            
            # Run performance analysis every 10 cycles or on critical conditions
            if self.should_run_performance_analysis(cycle_number):
                print(f"📊 PERFORMANCE ANALYSIS - Cycle {cycle_number}")
                from performance_monitor import PerformanceMonitor
                monitor = PerformanceMonitor()
                report = monitor.analyze_performance(last_n_cycles=10)
                monitor.print_performance_summary(report)
            print("\n📊 FETCHING MARKET DATA...")
            md_start = time.perf_counter()
            real_prices = self.market_data.get_all_real_prices()
            valid_prices = {k: v for k, v in real_prices.items() if isinstance(v, (int, float)) and v > 0}
            cycle_timing['market_data_ms'] = round((time.perf_counter() - md_start) * 1000, 2)
            if not valid_prices: raise ValueError("No valid market prices received.")
            
            # Check bot control before live account sync (can be slow)
            control = self._read_bot_control()
            if control.get("status") == "paused":
                print(f"⏸️ Cycle {cycle_number} PAUSED before account sync - stopping cycle")
                self.cycle_active = False
                return
            elif control.get("status") == "stopped":
                print(f"🛑 Cycle {cycle_number} STOPPED - Bot STOP command received")
                self.cycle_active = False
                return
            
            if self.portfolio.is_live_trading:
                self.portfolio.sync_live_account()
            self.portfolio.update_prices(valid_prices, increment_loss_counters=True) # Update PnL before checking TP/SL

            # --- Auto TP/SL Check ---
            positions_closed_by_tp_sl = self.portfolio.check_and_execute_tp_sl(valid_prices)
            # --- End Auto TP/SL Check ---

            manual_override = self.portfolio.get_manual_override()
            auto_exit_triggered = bool(positions_closed_by_tp_sl)

            if manual_override:
                print("🔔 APPLYING MANUAL OVERRIDE...")
                decisions = manual_override.get('decisions', {}); thoughts = "Manual override."; prompt = "N/A (Manual)"
                print("\n🎯 MANUAL DECISIONS:", json.dumps(decisions, indent=2))
            # Only ask AI if no TP/SL triggered AND no manual override
            else:
                if auto_exit_triggered:
                    print("ℹ️ Auto TP/SL/extended exit triggered earlier this cycle — proceeding with AI analysis.")
                # Check bot control before AI call (can be slow in live mode)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    print(f"⏸️ Cycle {cycle_number} PAUSED before AI call - stopping cycle")
                    self.cycle_active = False
                    return
                elif control.get("status") == "stopped":
                    print(f"🛑 Cycle {cycle_number} STOPPED - Bot STOP command received")
                    self.cycle_active = False
                    return
                
                ai_timer_start = time.perf_counter()
                print("\n🤖 GENERATING PROMPT...")
                self.invocation_count += 1 # Increment AI call count
                # Use JSON prompt if enabled, with fallback to text format
                prompt = None
                prompt_format_used = "text"
                json_serialization_error = None
                
                if Config.USE_JSON_PROMPT:
                    try:
                        prompt = self.generate_alpha_arena_prompt_json()
                        prompt_format_used = "json"
                        print(f"✅ Using JSON prompt format (version {Config.JSON_PROMPT_VERSION})")
                    except Exception as e:
                        json_serialization_error = str(e)
                        print(f"⚠️ JSON prompt generation failed: {e}")
                        print("   Falling back to text format...")
                        prompt = self.generate_alpha_arena_prompt()
                        prompt_format_used = "json_fallback"
                else:
                    prompt = self.generate_alpha_arena_prompt()
                print("📋 USER PROMPT (summary): " + prompt[:200] + "...")

                # Check bot control before AI API call (can be slow in live mode)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    print(f"⏸️ Cycle {cycle_number} PAUSED before AI API call - stopping cycle")
                    self.cycle_active = False
                    return
                elif control.get("status") == "stopped":
                    print(f"🛑 Cycle {cycle_number} STOPPED - Bot STOP command received")
                    self.cycle_active = False
                    return

                print("\n💭 DEEPSEEK ANALYZING...")
                ai_response = self.deepseek.get_ai_decision(prompt)
                
                # Check bot control after AI API call (may have taken time in live mode)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    print(f"⏸️ Cycle {cycle_number} PAUSED after AI call - stopping cycle")
                    self.cycle_active = False
                    return
                elif control.get("status") == "stopped":
                    print(f"🛑 Cycle {cycle_number} STOPPED - Bot STOP command received")
                    self.cycle_active = False
                    return
                
                parsed_response = self.parse_ai_response(ai_response)
                thoughts = parsed_response.get("chain_of_thoughts", "Parse Error.")
                decisions = parsed_response.get("decisions", {})
                if auto_exit_triggered and isinstance(thoughts, str):
                    thoughts += "\n[Auto Exit Note: TP/SL or extended-loss closure executed before this analysis]"
                cycle_timing['ai_ms'] = round((time.perf_counter() - ai_timer_start) * 1000, 2)

                if not isinstance(decisions, dict):
                    print(f"❌ AI decisions not dict ({type(decisions)}). Resetting."); thoughts += f"\nError: Decisions not dict."; decisions = {}

                print("\n🔍 CHAIN_OF_THOUGHTS:\n", thoughts)
                print("\n🎯 AI TRADING DECISIONS:", json.dumps(decisions, indent=2) if decisions else "{}")
                
                # KADEMELİ POZİSYON SİSTEMİ: Cycle bazlı pozisyon limiti
                max_positions_for_cycle = self.get_max_positions_for_cycle(cycle_number)
                current_positions = len(self.portfolio.positions)
                
                if current_positions >= max_positions_for_cycle:
                    print(f"🛡️ POSITION LIMIT REACHED (Cycle {cycle_number}): Max {max_positions_for_cycle} positions allowed")
                    # Pozisyon limiti dolduysa yeni entry sinyallerini hold'a çevir
                    filtered_decisions = {}
                    for coin, trade in decisions.items():
                        if isinstance(trade, dict):
                            signal = trade.get('signal')
                            if signal in ['buy_to_enter', 'sell_to_enter']:
                                print(f"   ⚠️ {coin} {signal} → HOLD (Position limit: {max_positions_for_cycle})")
                                filtered_decisions[coin] = {'signal': 'hold', 'justification': f'Position limit reached - Cycle {cycle_number} (max {max_positions_for_cycle} positions)'}
                            else:
                                filtered_decisions[coin] = trade
                        else:
                            filtered_decisions[coin] = trade
                    decisions = filtered_decisions
                    thoughts += f"\n[Position Limit: Cycle {cycle_number} - Max {max_positions_for_cycle} positions allowed]"
                decisions, directional_filtered = self._apply_directional_capacity_filter(decisions)
                if directional_filtered:
                    thoughts += "\n[Directional Capacity: Entry signals converted to HOLD until opposite-side exposure is reduced]"
            # Execute AI decisions only if it's a valid dict and NOT empty AND no manual override was active
            execution_elapsed = None
            if isinstance(decisions, dict) and decisions and not manual_override:
                # Check bot control before execution (live mode can be slow)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    print(f"⏸️ Cycle {cycle_number} PAUSED before execution - stopping cycle")
                    self.cycle_active = False
                    return
                elif control.get("status") == "stopped":
                    print(f"🛑 Cycle {cycle_number} STOPPED - Bot STOP command received")
                    self.cycle_active = False
                    return
                
                exec_start = time.perf_counter()
                # AI ÖNCELİKLİ SİSTEM: "close_position" sinyali varsa tüm pozisyon kapatılır
                has_close_position_signal = any(
                    trade.get('signal') == 'close_position' 
                    for trade in decisions.values() 
                    if isinstance(trade, dict)
                )

                close_execution_report = {
                    "executed": [],
                    "blocked": [],
                    "skipped": [],
                    "holds": [],
                    "notes": [],
                    "timestamp": datetime.now().isoformat()
                }
                
                if has_close_position_signal:
                    print("🚨 AI CLOSE_POSITION SİNYALİ: Sadece belirtilen pozisyonlar kapatılıyor")
                    # Sadece close_position sinyali verilen coin'leri kapat
                    for coin, trade in decisions.items():
                        if not isinstance(trade, dict):
                            continue
                        if trade.get('signal') == 'close_position' and coin in self.portfolio.positions:
                            if coin in valid_prices:
                                position = self.portfolio.positions[coin]
                                current_price = valid_prices[coin]
                                direction = position.get('direction', 'long')
                                entry_price = position['entry_price']
                                quantity = position['quantity']
                                margin_used = position.get('margin_usd', 0)

                                if self.portfolio.is_live_trading:
                                    live_result = self.portfolio.execute_live_close(
                                        coin=coin,
                                        position=position,
                                        current_price=current_price,
                                        reason="AI close_position signal"
                                    )
                                    if not live_result.get('success'):
                                        error_msg = live_result.get('error', 'unknown_error')
                                        print(f"🚫 AI LIVE CLOSE FAILED: {coin} ({error_msg})")
                                        close_execution_report["blocked"].append({
                                            "coin": coin,
                                            "reason": "live_close_failed",
                                            "error": error_msg
                                        })
                                    else:
                                        history_entry = live_result.get('history_entry')
                                        if history_entry:
                                            self.portfolio.add_to_history(history_entry)
                                        close_execution_report["executed"].append({
                                            "coin": coin,
                                            "signal": "close_position",
                                            "pnl": live_result.get('pnl'),
                                            "direction": direction,
                                            "price": current_price,
                                            "mode": "live"
                                        })
                                        print(f"✅ AI LIVE CLOSE: Closed {direction} {coin} @ ${format_num(current_price, 4)} (PnL: ${format_num(live_result.get('pnl', 0), 2)})")
                                    continue
                                
                                if direction == 'long': 
                                    profit = (current_price - entry_price) * quantity
                                else: 
                                    profit = (entry_price - current_price) * quantity
                                
                                self.portfolio.current_balance += (margin_used + profit)
                                
                                print(f"✅ AI CLOSE: Closed {direction} {coin} @ ${format_num(current_price, 4)} (PnL: ${format_num(profit, 2)})")
                                
                                history_entry = {
                                    "symbol": coin, "direction": direction, "entry_price": entry_price, "exit_price": current_price,
                                    "quantity": quantity, "notional_usd": position.get('notional_usd', 'N/A'), "pnl": profit,
                                    "entry_time": position['entry_time'], "exit_time": datetime.now().isoformat(),
                                    "leverage": position.get('leverage', 'N/A'), "close_reason": "AI close_position signal"
                                }
                                self.portfolio.add_to_history(history_entry)

                                close_execution_report["executed"].append({
                                    "coin": coin,
                                    "signal": "close_position",
                                    "pnl": profit,
                                    "direction": direction,
                                    "price": current_price
                                })
                                del self.portfolio.positions[coin]
                            else:
                                close_execution_report["skipped"].append({
                                    "coin": coin,
                                    "reason": "no_price_data_for_close"
                                })
                        elif isinstance(trade, dict):
                            close_execution_report["holds"].append({
                                "coin": coin,
                                "has_position": coin in self.portfolio.positions
                            })

                    # Combine with normal execution report for logging
                    previous_report = getattr(self.portfolio, "last_execution_report", {})
                    merged_report = {
                        "executed": (previous_report.get("executed", []) + close_execution_report["executed"]),
                        "blocked": (previous_report.get("blocked", []) + close_execution_report["blocked"]),
                        "skipped": (previous_report.get("skipped", []) + close_execution_report["skipped"]),
                        "holds": close_execution_report["holds"] or previous_report.get("holds", []),
                        "notes": previous_report.get("notes", []) + close_execution_report["notes"],
                        "timestamp": close_execution_report["timestamp"]
                    }
                    self.portfolio.last_execution_report = merged_report
                    
                    # AI'nin diğer kararlarını işleme (sadece yeni pozisyonlar)
                    self.portfolio._execute_new_positions_only(
                        decisions,
                        valid_prices,
                        cycle_number,
                        indicator_cache=self.latest_indicator_cache
                    )
                else:
                    # Normal karar işleme (partial profit aktif)
                    self.portfolio._execute_normal_decisions(
                        decisions,
                        valid_prices,
                        cycle_number,
                        positions_closed_by_tp_sl,
                        indicator_cache=self.latest_indicator_cache
                    )
                execution_elapsed = time.perf_counter() - exec_start

            # Execute manual override decisions if present
            elif isinstance(decisions, dict) and decisions and manual_override:
                 exec_start = time.perf_counter()
                 self.portfolio.execute_decision(decisions, valid_prices, indicator_cache=self.latest_indicator_cache)
                 execution_elapsed = time.perf_counter() - exec_start

            elif isinstance(decisions, dict): print("ℹ️ No AI/Manual trading actions to execute this cycle.")

            if execution_elapsed is not None:
                cycle_timing['execution_ms'] = round(execution_elapsed * 1000, 2)

            # Save state and history at the end of the cycle
            self.portfolio.save_state()
            # Log regardless of errors (log contains error info if applicable)
            execution_report = getattr(self.portfolio, 'last_execution_report', {})
            if manual_override:
                cycle_status = "manual_override"
            elif positions_closed_by_tp_sl and not decisions:
                cycle_status = "tp_sl_only"
            elif isinstance(decisions, dict) and decisions:
                cycle_status = "ai_decision"
            else:
                cycle_status = "idle"

            cycle_metadata: Dict[str, Any] = {
                'positions_closed_by_tp_sl': bool(positions_closed_by_tp_sl),
                'manual_override': bool(manual_override),
                'cooldown_status': {
                    'directional_cooldowns': dict(self.portfolio.directional_cooldowns),
                    'relaxed_countertrend_cycles': self.portfolio.relaxed_countertrend_cycles,
                    'counter_trend_cooldown': self.portfolio.counter_trend_cooldown
                },
                'prompt_format': prompt_format_used if 'prompt_format_used' in locals() else 'text',
                'json_serialization_error': json_serialization_error if 'json_serialization_error' in locals() else None
            }
            if execution_report:
                cycle_metadata['execution_report'] = execution_report
            if cycle_timing:
                cycle_metadata['performance'] = cycle_timing
                timing_summary = []
                if 'market_data_ms' in cycle_timing:
                    timing_summary.append(f"market {cycle_timing['market_data_ms']:.2f}ms")
                if 'ai_ms' in cycle_timing:
                    timing_summary.append(f"ai {cycle_timing['ai_ms']:.2f}ms")
                if 'execution_ms' in cycle_timing:
                    timing_summary.append(f"exec {cycle_timing['execution_ms']:.2f}ms")
                if timing_summary:
                    print(f"⏱️ Cycle timers → " + " | ".join(timing_summary))

            self.portfolio.add_to_cycle_history(
                cycle_number,
                prompt,
                thoughts,
                decisions,
                status=cycle_status,
                metadata=cycle_metadata if cycle_metadata else None
            )
            
            # Enhanced exit strategy control - re-enable after cycle completion
            print("▶️ Enhanced exit strategy re-enabled after cycle completion")
            self.show_status()

        except Exception as e:
            print(f"❌ CRITICAL CYCLE ERROR: {e}"); traceback.print_exc()
            try:
                 decisions_log = decisions if isinstance(decisions, dict) else {}
                 self.portfolio.add_to_cycle_history(
                     cycle_number,
                     prompt,
                     f"CRITICAL CYCLE ERROR: {e}\n{traceback.format_exc()}",
                     decisions_log,
                     status="error",
                     metadata={'exception': str(e)}
                 )
            except Exception as log_e: print(f"❌ Failed to save error to cycle history: {log_e}")
        finally:
            self.cycle_active = False
            self.enhanced_exit_enabled = True
            # Check bot control after exception - if paused/stopped, don't continue
            try:
                control = self._read_bot_control()
                if control.get("status") == "stopped":
                    print(f"🛑 Cycle {cycle_number} exception handler: Bot STOP command received")
                    raise SystemExit("Bot stopped by user command")
            except SystemExit:
                raise
            except Exception as control_e:
                print(f"⚠️ Failed to check bot control after exception: {control_e}")

    def show_status(self):
        """Show current status in the console"""
        print(f"\n📊 CURRENT STATUS:")
        print(f"💰 Total Value: ${format_num(self.portfolio.total_value, 2)} (Initial: ${format_num(self.portfolio.initial_balance, 2)})")
        print(f"📈 Total Return: {format_num(self.portfolio.total_return, 2)}%")
        print(f"💵 Available Cash: ${format_num(self.portfolio.current_balance, 2)}")
        print(f"🔄 Total Closed Trades: {self.portfolio.trade_count}")
        print(f"\n📦 CURRENT POSITIONS ({len(self.portfolio.positions)} open):")
        if not self.portfolio.positions: print("  No open positions.")
        else:
            for coin, pos in self.portfolio.positions.items():
                pnl = pos.get('unrealized_pnl', 0.0); pnl_sign = "+" if pnl >= 0 else ""
                direction = pos.get('direction', 'long').upper(); leverage = pos.get('leverage', 1)
                notional = pos.get('notional_usd', 0.0); liq = pos.get('liquidation_price', 0.0)
                entry = pos.get('entry_price', 0.0); qty = pos.get('quantity', 0.0)
                print(f"  {coin} ({direction} {leverage}x): {format_num(qty, 4)} units | Notional ${format_num(notional, 2)} | Entry: ${format_num(entry, 4)} | PnL: {pnl_sign}${format_num(pnl, 2)} | Liq Est: ${format_num(liq, 4)}")

    def start_tp_sl_monitoring(self):
        """Start TP/SL monitoring timer that runs every 1 minute"""
        if self.tp_sl_timer and self.tp_sl_timer.is_alive():
            print("ℹ️ TP/SL monitoring already running")
            return
        
        self.is_running = True
        self.tp_sl_timer = threading.Thread(target=self._tp_sl_monitoring_loop, daemon=True)
        self.tp_sl_timer.start()
        print("✅ TP/SL monitoring started (30 second interval)")

    def stop_tp_sl_monitoring(self):
        """Stop TP/SL monitoring timer"""
        self.is_running = False
        if self.tp_sl_timer and self.tp_sl_timer.is_alive():
            self.tp_sl_timer.join(timeout=5)
            print("🛑 TP/SL monitoring stopped")
        else:
            print("ℹ️ TP/SL monitoring was not running")

    def _tp_sl_monitoring_loop(self):
        """Background thread that checks TP/SL every 30 seconds"""
        print("🔄 TP/SL monitoring loop started (30 second interval)")
        while self.is_running:
            try:
                # Check bot control file for pause/stop command
                control = self._read_bot_control()
                if control.get("status") == "stopped":
                    print("🛑 TP/SL monitoring: STOP command received. Stopping monitoring loop...")
                    self.is_running = False
                    break
                elif control.get("status") == "paused":
                    print("⏸️ TP/SL monitoring: Bot is PAUSED. Waiting for resume...")
                    # Wait in smaller intervals to check for resume
                    while True:
                        time.sleep(10)
                        control = self._read_bot_control()
                        if control.get("status") == "running":
                            print("▶️ TP/SL monitoring: Bot RESUMED. Continuing monitoring...")
                            break
                        elif control.get("status") == "stopped":
                            print("🛑 TP/SL monitoring: STOP command received. Stopping monitoring loop...")
                            self.is_running = False
                            break
                    if control.get("status") == "stopped":
                        break
                    continue
                
                # Enhanced exit strategy control - check if enabled
                if getattr(self, 'cycle_active', False):
                    # Trading cycle active; wait until it completes
                    for _ in range(5):
                        if not self.is_running:
                            break
                        # Check bot control during wait
                        control = self._read_bot_control()
                        if control.get("status") == "stopped":
                            self.is_running = False
                            break
                        if not getattr(self, 'cycle_active', False):
                            break
                        time.sleep(1)
                    if control.get("status") == "stopped":
                        break
                    continue

                if not self.enhanced_exit_enabled:
                    print("⏸️ Enhanced exit strategy paused during cycle - TP/SL monitoring waiting")
                    # Wait 10 seconds and check again
                    for _ in range(10):
                        if not self.is_running:
                            break
                        # Check bot control during wait
                        control = self._read_bot_control()
                        if control.get("status") == "stopped":
                            self.is_running = False
                            break
                        time.sleep(1)
                    if control.get("status") == "stopped":
                        break
                    continue
                
                # Get current prices
                real_prices = self.market_data.get_all_real_prices()
                valid_prices = {k: v for k, v in real_prices.items() if isinstance(v, (int, float)) and v > 0}
                
                if valid_prices:
                    # Update portfolio prices
                    self.portfolio.update_prices(valid_prices, increment_loss_counters=False)
                    
                    # Run TP/SL check - all decisions made by 30-second monitoring (like simulation mode)
                    # No Binance TP/SL orders - all managed by monitoring loop
                    positions_closed = self.portfolio.check_and_execute_tp_sl(valid_prices)
                    
                    if positions_closed:
                        print(f"⏰ 30-SECOND TP/SL CHECK: Positions closed")
                    else:
                        print(f"⏰ 30-SECOND TP/SL CHECK: No triggers ({len(self.portfolio.positions)} positions monitored)")
                else:
                    print("⚠️ TP/SL monitoring: No valid prices available")
                
                # Check bot control before sleep
                control = self._read_bot_control()
                if control.get("status") == "stopped":
                    print("🛑 TP/SL monitoring: STOP command received. Stopping monitoring loop...")
                    self.is_running = False
                    break
                
            except Exception as e:
                print(f"❌ TP/SL monitoring error: {e}")
                # Check bot control after exception
                try:
                    control = self._read_bot_control()
                    if control.get("status") == "stopped":
                        print("🛑 TP/SL monitoring: STOP command received after exception")
                        self.is_running = False
                        break
                except Exception as control_e:
                    print(f"⚠️ Failed to check bot control after TP/SL exception: {control_e}")
            
            # Wait 30 seconds before next check
            if self.is_running:
                time.sleep(30)

    def run_simulation(self, total_duration_hours: int = 168, cycle_interval_minutes: int = 2):
        """Run the simulation with dynamic cycle frequency and TP/SL monitoring"""
        print(f"🚀 ALPHA ARENA - DEEPSEEK INTEGRATION (V{VERSION})")
        print(f"💡 Simulating with ${format_num(self.portfolio.initial_balance, 2)} budget for {total_duration_hours} hours.")
        print(f"   Trading: {', '.join(self.market_data.available_coins)}")
        print(f"   State File: {self.portfolio.state_file}")
        print(f"   Trade History File: {self.portfolio.history_file}")
        print(f"   Cycle History File: {self.portfolio.cycle_history_file}")
        print(f"   Override File Check: {self.portfolio.override_file}")
        print(f"   Dynamic Cycle Frequency: Enabled (2-4 minutes based on volatility)")
        print(f"   TP/SL Monitoring: Enabled (30 second interval)")

        # Start TP/SL monitoring
        self.start_tp_sl_monitoring()

        end_time = datetime.now() + timedelta(hours=total_duration_hours)
        start_cycle = len(self.portfolio.cycle_history) + 1
        print(f"   Resuming from Cycle {start_cycle}...")
        self.invocation_count = start_cycle - 1; current_cycle_number = start_cycle - 1

        # Initialize bot control file
        bot_control_file = "bot_control.json"
        self._write_bot_control({"status": "running", "last_updated": datetime.now().isoformat()})
        
        try:
            while datetime.now() < end_time:
                # Check bot control file for pause/stop command BEFORE starting cycle
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    print(f"⏸️ Bot is PAUSED. Waiting for resume command... (checking every 10 seconds)")
                    # Wait in smaller intervals to check for resume
                    while True:
                        time.sleep(10)
                        control = self._read_bot_control()
                        if control.get("status") == "running":
                            print(f"▶️ Bot RESUMED. Continuing trading cycles...")
                            break
                        elif control.get("status") == "stopped":
                            print(f"🛑 Bot STOP command received. Stopping gracefully...")
                            break
                    if control.get("status") == "stopped":
                        break
                    continue
                elif control.get("status") == "stopped":
                    print(f"🛑 Bot STOP command received. Stopping gracefully...")
                    break
                
                current_cycle_number += 1; cycle_start_time = time.time()
                
                # Calculate dynamic cycle frequency
                dynamic_cycle_interval = self.calculate_optimal_cycle_frequency()
                print(f"🔄 Dynamic cycle frequency: {dynamic_cycle_interval} seconds ({dynamic_cycle_interval/60:.1f} minutes)")
                
                self.run_trading_cycle(current_cycle_number)
                if datetime.now() >= end_time: break
                elapsed_time = time.time() - cycle_start_time
                sleep_time = max(0, dynamic_cycle_interval - elapsed_time)
                print(f"\n⏳ Cycle {current_cycle_number} complete in {format_num(elapsed_time,2)}s. Next cycle in {format_num(sleep_time/60, 2)} mins... (Ctrl+C to stop)")
                time.sleep(max(sleep_time, 0.5))
                
                # Check bot control file AFTER sleep (before next cycle)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    print(f"⏸️ Bot is PAUSED. Waiting for resume command... (checking every 10 seconds)")
                    # Wait in smaller intervals to check for resume
                    while True:
                        time.sleep(10)
                        control = self._read_bot_control()
                        if control.get("status") == "running":
                            print(f"▶️ Bot RESUMED. Continuing trading cycles...")
                            break
                        elif control.get("status") == "stopped":
                            print(f"🛑 Bot STOP command received. Stopping gracefully...")
                            break
                    if control.get("status") == "stopped":
                        break
                    continue
                elif control.get("status") == "stopped":
                    print(f"🛑 Bot STOP command received. Stopping gracefully...")
                    break

        except KeyboardInterrupt:
            print("\n⏹️ Program stopped by user.")
        finally:
            # Stop TP/SL monitoring
            self.stop_tp_sl_monitoring()

        print(f"\n{'='*80}\n🏁 SIMULATION COMPLETED\n{'='*80}"); self.show_status()

    def _adjust_partial_sale_for_min_limit(self, position: Dict, proposed_percent: float) -> float:
        """Adjust partial sale percentage to ensure minimum limit remains after sale"""
        current_margin = position.get('margin_usd', 0)
        
        # Calculate dynamic minimum limit: $15 fixed OR 10% of available cash, whichever is larger
        min_remaining = self._calculate_dynamic_minimum_limit()
        
        if current_margin <= min_remaining:
            # Position already at or below minimum, don't sell
            print(f"🛑 Partial sale blocked: Position margin ${current_margin:.2f} <= minimum limit ${min_remaining:.2f}")
            return 0.0
        
        # Calculate remaining margin after proposed sale
        remaining_after_proposed = current_margin * (1 - proposed_percent)
        
        if remaining_after_proposed >= min_remaining:
            # Proposed sale keeps us above minimum, use as-is
            return proposed_percent
        else:
            # Adjust sale to leave exactly min_remaining margin
            adjusted_sale_amount = current_margin - min_remaining
            adjusted_percent = adjusted_sale_amount / current_margin
            
            print(f"📊 Adjusted partial sale: {proposed_percent*100:.0f}% → {adjusted_percent*100:.0f}% to maintain ${min_remaining:.2f} minimum limit")
            return adjusted_percent

    def update_trend_state(
        self,
        coin: str,
        indicators_htf: Dict[str, Any],
        indicators_3m: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Delegate trend state updates to PortfolioManager for backward compatibility."""
        return self.portfolio.update_trend_state(coin, indicators_htf, indicators_3m)

    def get_recent_trend_flip_summary(self) -> List[str]:
        """Expose portfolio trend flip summary for existing integrations."""
        return self.portfolio.get_recent_trend_flip_summary()

    def count_positions_by_direction(self) -> Dict[str, int]:
        return self.portfolio.count_positions_by_direction()
    
    def _read_bot_control(self) -> Dict[str, Any]:
        """Read bot control file to check for pause/stop commands."""
        try:
            return safe_file_read("bot_control.json", {"status": "running", "last_updated": datetime.now().isoformat()})
        except Exception as e:
            print(f"⚠️ Failed to read bot_control.json: {e}")
            # Return default running state if file read fails (fail-safe)
            return {"status": "running", "last_updated": datetime.now().isoformat()}
    
    def _write_bot_control(self, data: Dict[str, Any]):
        """Write bot control file."""
        safe_file_write("bot_control.json", data)

    def apply_directional_bias(self, signal: str, confidence: float, bias_metrics: Dict[str, Dict[str, Any]], current_trend: str) -> float:
        return self.portfolio.apply_directional_bias(signal, confidence, bias_metrics, current_trend)

    def get_directional_bias_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Proxy to portfolio directional bias metrics."""
        return self.portfolio.get_directional_bias_metrics()

    def load_cycle_history(self) -> List[Dict]:
        history = safe_file_read(self.cycle_history_file, default_data=[]); print(f"✅ Loaded {len(history)} cycles."); return history
    def add_to_cycle_history(
        self,
        cycle_number: int,
        prompt: str,
        thoughts: str,
        decisions: Dict,
        status: str = "ai_decision",
        metadata: Optional[Dict[str, Any]] = None
    ):
        return self.portfolio.add_to_cycle_history(cycle_number, prompt, thoughts, decisions, status=status, metadata=metadata)

# Define VERSION at the top level
VERSION = "9 - Auto TP/SL, Dynamic Size, Prompt Eng"

def main():
    """Main application entry point"""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key: print("⚠️ No DEEPSEEK_API_KEY found. Running simulation mode...");
        arena = AlphaArenaDeepSeek(api_key)
        arena.run_simulation(total_duration_hours=168, cycle_interval_minutes=2)
    except KeyboardInterrupt: print("\n⏹️ Program stopped by user.")
    except Exception as e: print(f"\n❌ Unexpected critical error in main: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main()