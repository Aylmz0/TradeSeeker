import requests
import json
import time
from typing import Dict, List, Any, Optional
from config.config import Config
from src.utils import RetryManager, safe_file_read

# HTF_INTERVAL used in prompt, we can get it from Config
HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'
HTF_LABEL = HTF_INTERVAL

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

You are given ${Config.INITIAL_BALANCE} starting capital and must process numerical market data to discover alpha.
Your Sharpe ratio is provided to help normalize for risky behavior.

CORE RULES:
- Make every decision using the numerical data provided; ignore external narratives.
- Always provide complete exit plans (profit_target, stop_loss, invalidation_condition).
- All entries must use fixed 10x leverage. Submit 10x on every trade; risk sizing is handled via margin rules.
- Minimum confidence is 0.4; use higher confidence for stronger quantitative edges.
- Maximum simultaneous positions across assets: 5.
- SAME-DIRECTION LIMIT: You can open a maximum of {Config.SAME_DIRECTION_LIMIT} positions in the SAME direction (LONG or SHORT).
  * If you already have {Config.SAME_DIRECTION_LIMIT} LONG positions, you CANNOT open another LONG position.
  * If you already have {Config.SAME_DIRECTION_LIMIT} SHORT positions, you CANNOT open another SHORT position.
  * When same-direction limit is reached:
    - DO NOT propose new entries in that direction
    - Instead, evaluate exit plans for existing positions in that direction
    - Look for counter-trend opportunities in the opposite direction
    - Or hold and wait for better setups
  * This limit is shown in POSITION_SLOTS JSON as "same_direction_limit", "long_slots_used", "short_slots_used", "long_slots_available", "short_slots_available".
  * CRITICAL: Always check POSITION_SLOTS JSON before proposing new entries. If "long_slots_available" is 0, do NOT propose LONG entries. If "short_slots_available" is 0, do NOT propose SHORT entries.

RISK MANAGEMENT:
- Portfolio- and position-level risk caps are enforced automatically; focus on selecting high-quality opportunities.
- Maintain at least 1:1.3 risk/reward.
- Use objective volatility references (e.g., {HTF_LABEL} ATR) when setting stops.
- Express invalidation clearly (e.g., "If {HTF_LABEL} close is below EMA20").

SYMMETRIC STRATEGY GUIDANCE:
- Evaluate both LONG and SHORT paths for every asset. Bullish regimes support longs; bearish regimes support shorts.
- NEUTRAL regime: When a coin's market regime is NEUTRAL (1h trend doesn't align with BOTH 3m AND 15m). Specifically: 1h bullish but both 3m and 15m are bearish, OR 1h bearish but both 3m and 15m are bullish. In NEUTRAL conditions, you can take LONG or SHORT positions based on the best quantified edge. NEUTRAL indicates counter-trend opportunities or mixed signals - evaluate technical conditions and take the direction with the best quantified edge. Both LONG and SHORT trades are valid in NEUTRAL conditions.
- Counter-trend trades are a valid and valuable strategy; use them when the pre-computed checklist (in the prompt) shows >2/5 conditions and you can rationalize the edge. Counter-trend opportunities often provide excellent risk/reward ratios.
- Counter-trend trades require confidence ≥0.65 for standard setups. For exceptionally strong counter-trend setups (STRONG alignment, 3+ conditions met, LOW_RISK rating), confidence ≥0.60 is acceptable.
- Only label a setup as counter-trend when your proposed trade direction is opposite the {HTF_LABEL} trend. If {HTF_LABEL} trend and trade direction align but 3m is temporarily opposing, treat it as trend-following.
- Prioritize trades with quantified momentum, participation, and risk/reward advantages.
- When regime, momentum, and participation align in your favor, favor committing capital decisively instead of waiting for perfect confirmation.
- Execute trend-following setups promptly when {HTF_LABEL} + 3m structures point the same way and volume/liquidity is supportive.
- IMPORTANT: When {HTF_LABEL} + 15m + 3m all align in the same direction, this is a STRONG trend-following signal. Evaluate trend duration and market context, but this alignment indicates high-probability trend continuation.

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
  Examples:
    - 1h BULLISH, but 15m BEARISH AND 3m BEARISH = counter-trend SHORT opportunity
    - 1h BEARISH, but 15m BULLISH AND 3m BULLISH = counter-trend LONG opportunity
  CRITICAL: To determine counter-trend direction:
    - Look at 15m and 3m trends (NOT 1h trend)
    - If 15m+3m are BULLISH → Counter-trend LONG
    - If 15m+3m are BEARISH → Counter-trend SHORT
    - The 1h trend is OPPOSITE to your trade direction
    - Remember: Counter-trend direction = 15m+3m direction, NOT 1h direction
- If volume ratio is ≤0.20× average, call out the weakness, reduce confidence materially, and consider skipping the trade unless another data point overwhelmingly compensates.
- CRITICAL: If you identify a valid counter-trend opportunity (e.g., LONG) but cannot execute it due to limits (e.g., no LONG slots), you MUST NOT open a trend-following trade in the opposite direction (e.g., SHORT). The existence of a valid counter-trend signal invalidates the trend-following setup. In this case, simply HOLD.

TREND REVERSAL DETECTION:
- Monitor positions that have been open for extended periods (1+ hours). Extended positions may need review, but don't automatically assume reversal is imminent.
- Multi-timeframe analysis: {HTF_LABEL} (1h) provides structural trend, 15m provides medium-term momentum confirmation, and 3m provides short-term entry/exit timing.
- 15m momentum provides important confirmation between {HTF_LABEL} trend and 3m momentum. When 15m aligns with {HTF_LABEL}, it strengthens the trend signal. When 15m aligns with 3m but opposes {HTF_LABEL}, it suggests potential reversal.
- 3m momentum provides supplementary context alongside {HTF_LABEL} trend analysis. Use it as one data point among many, not as a primary decision driver. Short-term 3m momentum changes are normal market noise.
- CRITICAL: Reversal signals ONLY apply to EXISTING positions.
    - If you have NO position, reversal signals are NOT relevant
    - Reversal = momentum moving AGAINST your current position
    - For new entries, use counter-trend analysis, NOT reversal
- IMPORTANT: Reversal means momentum is moving AGAINST the position direction. This is an EXIT WARNING, not an entry signal. For example:
  * LONG position: Reversal = bearish momentum (price < EMA20, RSI < 50, MACD < 0) → Consider closing LONG
  * SHORT position: Reversal = bullish momentum (price > EMA20, RSI > 50, MACD > 0) → Consider closing SHORT
- Reversal signal strength (based on how many timeframes show reversal AGAINST position):
  * "STRONG": 15m + 3m BOTH show reversal against position (but {HTF_LABEL} doesn't) - strong reversal signal. Consider closing the position if it's losing money or the thesis is invalidated.
  * "MEDIUM": Only 3m shows reversal against position - medium reversal signal, continue monitoring. 3m momentum changes are normal market noise; don't overreact.
  * "INFORMATIONAL": Only 15m shows reversal against position - informational context, continue monitoring, prioritize {HTF_LABEL} trend
  * NOTE: If {HTF_LABEL} + 15m + 3m ALL show the same direction, this is NOT a reversal - this is the trend itself continuing. Reversal only occurs when shorter timeframes (15m+3m) oppose the position while {HTF_LABEL} may still be in favor.
- IMPORTANT: Reversal signals are EXIT warnings for existing positions. Counter-trend opportunities are ENTRY signals for new positions. These are SEPARATE decisions:
  1. If you see reversal signals against an existing position:
     - Evaluate position duration, PnL, and original thesis
     - Consider closing if reversal is STRONG and position is losing
     - Don't close immediately due to 3m momentum changes alone
  2. After closing a position (or for new entries):
     - If 15m+3m align against 1h trend, this is a counter-trend opportunity in the 15m+3m direction
     - Use counter-trend analysis, NOT reversal signals, for new entries
  3. Do NOT mix reversal and counter-trend in the same decision. Reversal = exit, Counter-trend = entry (separate evaluations)


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
    "invalidation_condition": "If {HTF_LABEL} price closes below {HTF_LABEL} EMA20"
  }},
  "SOL": {{
    "signal": "sell_to_enter",
    "leverage": 10,
    "confidence": 0.75,
    "profit_target": 185.0,
    "stop_loss": 198.0,
    "invalidation_condition": "If {HTF_LABEL} price closes above {HTF_LABEL} EMA20"
  }},
  "ADA": {{
    "signal": "buy_to_enter",
    "leverage": 10,
    "confidence": 0.75,
    "profit_target": 0.52,
    "stop_loss": 0.48,
    "invalidation_condition": "If {HTF_LABEL} price closes below {HTF_LABEL} EMA20"
  }},
  "DOGE": {{
    "signal": "sell_to_enter",
    "leverage": 10,
    "confidence": 0.75,
    "profit_target": 0.145,
    "stop_loss": 0.165,
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
            cached_cycles = safe_file_read("data/cycle_history.json", default_data=[])
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
