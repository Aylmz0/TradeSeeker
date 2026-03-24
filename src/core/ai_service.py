import copy
import json
import warnings
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config.config import Config
from src.ai.enhanced_context_provider import EnhancedContextProvider
from src.core import constants
from src.core.cache_manager import fetch_all_indicators_parallel, fetch_all_indicators_with_cache
from src.core.performance_monitor import PerformanceMonitor
from src.services.ml_service import MLService
from src.utils import format_num


HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"
HTF_LABEL = HTF_INTERVAL


class AIService:
    def __init__(self, portfolio, market_data, strategy_analyzer):
        self.portfolio = portfolio
        self.market_data = market_data
        self.strategy_analyzer = strategy_analyzer
        self.invocation_count = 0

    def _fetch_all_indicators_parallel(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Fetch all indicators for all coins in parallel with smart caching."""
        if Config.USE_SMART_CACHE:
            return fetch_all_indicators_with_cache(
                self.market_data,
                self.market_data.available_coins,
                HTF_INTERVAL,
                use_cache=True,
            )
        return fetch_all_indicators_parallel(
            self.market_data,
            self.market_data.available_coins,
            HTF_INTERVAL,
        )

    def get_enhanced_context(self) -> dict[str, Any]:
        """Get enhanced context for AI decision making"""
        try:
            provider = EnhancedContextProvider()
            return provider.generate_enhanced_context()
        except Exception as e:
            print(f"[WARN]  Enhanced context error: {e}")
            return {"error": f"Enhanced context failed: {e!s}"}

    def get_directional_bias_metrics(self) -> dict[str, dict[str, Any]]:
        """Get directional bias metrics from portfolio"""
        return self.portfolio.get_directional_bias_metrics()

    def get_trading_context(self) -> dict[str, Any]:
        """Get historical context from recent cycles - Enhanced with 5 cycle analysis"""
        try:
            if len(self.portfolio.cycle_history) < constants.MIN_HISTORY_FOR_ANALYSIS:
                return {
                    "recent_decisions": [],
                    "market_behavior": "Initial cycles - observing",
                    "total_cycles_analyzed": len(self.portfolio.cycle_history),
                    "performance_trend": "No data yet",
                }

            # Use last 5 cycles for enhanced analysis
            recent_cycles = self.portfolio.cycle_history[-5:]
            recent_decisions = []

            for cycle in recent_cycles:
                decisions = cycle.get("decisions", {})
                for coin, trade in decisions.items():
                    if isinstance(trade, dict) and trade.get("signal") in [
                        "buy_to_enter",
                        "sell_to_enter",
                    ]:
                        recent_decisions.append(
                            {
                                "coin": coin,
                                "signal": trade.get("signal"),
                                "cycle": cycle.get("cycle"),
                                "confidence": trade.get("confidence", 0.5),
                                "timestamp": cycle.get("timestamp"),
                            },
                        )

            # Enhanced market behavior analysis
            market_behavior = self._analyze_market_behavior(recent_cycles)
            performance_trend = self._analyze_performance_trend(recent_cycles)

            return {
                "recent_decisions": recent_decisions,
                "market_behavior": market_behavior,
                "performance_trend": performance_trend,
                "total_cycles_analyzed": len(recent_cycles),
                "analysis_period": f"Last {len(recent_cycles)} cycles",
            }

        except Exception as e:
            print(f"[WARN]  Trading context error: {e}")
            return {
                "recent_decisions": [],
                "market_behavior": "Error in context analysis",
                "performance_trend": "Unknown",
                "total_cycles_analyzed": 0,
            }

    def _analyze_market_behavior(self, recent_cycles: list[dict]) -> str:
        """Analyze market behavior based on recent trading decisions"""
        if not recent_cycles:
            return "No recent activity"

        recent_decisions = []
        for cycle in recent_cycles:
            decisions = cycle.get("decisions", {})
            for _coin, trade in decisions.items():
                if isinstance(trade, dict) and trade.get("signal") in [
                    "buy_to_enter",
                    "sell_to_enter",
                ]:
                    recent_decisions.append(trade)

        if not recent_decisions:
            return "Consolidating - No recent entries"

        long_count = sum(1 for d in recent_decisions if d.get("signal") == "buy_to_enter")
        short_count = sum(1 for d in recent_decisions if d.get("signal") == "sell_to_enter")

        # Enhanced analysis with confidence weighting
        long_confidence = sum(
            d.get("confidence", 0.5) for d in recent_decisions if d.get("signal") == "buy_to_enter"
        )
        short_confidence = sum(
            d.get("confidence", 0.5) for d in recent_decisions if d.get("signal") == "sell_to_enter"
        )

        if long_count > short_count and long_confidence > short_confidence:
            return f"Strong Bullish bias ({long_count} longs, avg confidence: {long_confidence / long_count:.2f})"
        if short_count > long_count and short_confidence > long_confidence:
            return f"Strong Bearish bias ({short_count} shorts, avg confidence: {short_confidence / short_count:.2f})"
        if long_count > short_count:
            return f"Bullish bias ({long_count} longs)"
        if short_count > long_count:
            return f"Bearish bias ({short_count} shorts)"
        return "Balanced market"

    def _analyze_performance_trend(self, recent_cycles: list[dict]) -> str:
        """Analyze performance trend based on recent cycles"""
        if len(recent_cycles) < constants.REVERSAL_SCORE_MODERATE:
            return "Insufficient data for trend analysis"

        # Analyze decision patterns
        entry_signals = 0
        hold_signals = 0
        close_signals = 0

        for cycle in recent_cycles:
            decisions = cycle.get("decisions", {})
            for trade in decisions.values():
                if isinstance(trade, dict):
                    signal = trade.get("signal")
                    if signal in {"buy_to_enter", "sell_to_enter"}:
                        entry_signals += 1
                    elif signal == "hold":
                        hold_signals += 1
                    elif signal == "close_position":
                        close_signals += 1

        total_signals = entry_signals + hold_signals + close_signals
        if total_signals == 0:
            return "No trading activity"

        entry_rate = entry_signals / total_signals
        close_rate = close_signals / total_signals

        if (
            entry_rate > constants.ENTRY_RATE_AGGRESSIVE
            and close_rate < constants.CLOSE_RATE_AGGRESSIVE
        ):
            return "Aggressive accumulation phase"
        if close_rate > constants.CLOSE_RATE_PROFIT_TAKING:
            return "Profit-taking phase"
        if hold_signals > entry_signals + close_signals:
            return "Consolidation phase"
        return "Balanced trading"

    def get_max_positions_for_cycle(self, cycle_number: int) -> int:
        """Delegate to portfolio manager"""
        return self.portfolio.get_max_positions_for_cycle(cycle_number)

    def generate_alpha_arena_prompt(self) -> str:
        """Generate prompt with enhanced data, indicator history and AI decision context

        .. deprecated:: 1.0
            Use :meth:`generate_alpha_arena_prompt_json` instead.
            This function is kept for backward compatibility and fallback scenarios.

        Returns
        -------
            str: Text-formatted prompt (legacy format)

        """
        warnings.warn(
            "generate_alpha_arena_prompt() is deprecated. "
            "Use generate_alpha_arena_prompt_json() instead. "
            "This function is kept for backward compatibility and fallback scenarios.",
            DeprecationWarning,
            stacklevel=2,
        )
        current_time = datetime.now(timezone.utc)
        minutes_running = int((current_time - self.portfolio.start_time).total_seconds() / 60)
        # Use internal invocation counter, don't increment here, do it in run_cycle
        # self.invocation_count += 1

        # OPTIMIZATION 1 & 2: Fetch all indicators in parallel ONCE, then share
        all_indicators, all_sentiment = self._fetch_all_indicators_parallel()
        self.latest_indicators = all_indicators
        self.latest_sentiment = all_sentiment

        # OPTIMIZATION 3: Get enhanced context and other data in parallel (non-blocking)
        # These don't need fresh market data, so can run in parallel
        enhanced_context = self.get_enhanced_context()

        # NOTE: counter_trade_analysis is now computed directly inside build_counter_trade_json

        # Get trend reversal detection using pre-fetched indicators (OPTIMIZATION 3: No re-fetch)
        from src.core.performance_monitor import PerformanceMonitor

        performance_monitor = PerformanceMonitor()
        trend_reversal_analysis = performance_monitor.detect_trend_reversal_for_all_coins(
            self.market_data.available_coins,
            indicators_cache=all_indicators,  # Pass pre-fetched indicators
        )

        bias_metrics = getattr(self, "latest_bias_metrics", self.get_directional_bias_metrics())
        bias_lines = []
        for side in ("long", "short"):
            stats = bias_metrics.get(side, {})
            bias_lines.append(
                f"  - {side.upper()}: net_pnl=${format_num(stats.get('net_pnl', 0.0), 2)}, "
                f"trades={stats.get('trades', 0)}, win_rate={format_num(stats.get('win_rate', 0.0), 2)}%, "
                f"rolling_avg=${format_num(stats.get('rolling_avg', 0.0), 2)}, consecutive_losses={stats.get('consecutive_losses', 0)}",
            )
        bias_section = "\n".join(bias_lines) if bias_lines else "  - No directional trades recorded"

        # Get cooldown status
        cooldowns = self.portfolio.directional_cooldowns
        cooldown_lines = []
        for side in ("long", "short"):
            cycles_remaining = cooldowns.get(side, 0)
            if cycles_remaining > 0:
                stats = bias_metrics.get(side, {})
                consecutive_losses = stats.get("consecutive_losses", 0)
                loss_streak_usd = stats.get("loss_streak_loss_usd", 0.0)
                reason = []
                if consecutive_losses >= constants.REVERSAL_SCORE_MODERATE:
                    reason.append(f"{consecutive_losses} consecutive losses")
                if loss_streak_usd >= constants.REGIME_PERFORMANCE_THRESHOLD:
                    reason.append(f"${loss_streak_usd:.2f} total loss")
                reason_str = " + ".join(reason) if reason else "unknown"
                cooldown_lines.append(
                    f"  - {side.upper()}: COOLDOWN ACTIVE ({cycles_remaining} cycles remaining) - Reason: {reason_str}",
                )
            else:
                cooldown_lines.append(f"  - {side.upper()}: No cooldown (active)")
        cooldown_section = (
            "\n".join(cooldown_lines) if cooldown_lines else "  - No cooldowns active"
        )

        # Get coin cooldown status
        coin_cooldowns = self.portfolio.coin_cooldowns
        coin_cooldown_lines = []
        if coin_cooldowns:
            for coin, cycles in sorted(coin_cooldowns.items()):
                if cycles > 0:
                    coin_cooldown_lines.append(
                        f"  - {coin}: COOLDOWN ACTIVE ({cycles} cycles remaining - previous loss)",
                    )
        coin_cooldown_section = (
            "\n".join(coin_cooldown_lines)
            if coin_cooldown_lines
            else "  - No coin cooldowns active"
        )

        recent_flips = self.portfolio.get_recent_trend_flip_summary()
        flip_history_window = getattr(
            self.portfolio,
            "trend_flip_history_window",
            self.portfolio.trend_flip_cooldown,
        )
        if recent_flips:
            trend_flip_section = "\n".join(f"  - {entry}" for entry in recent_flips)
        else:
            trend_flip_section = f"  - No trend flips in last {flip_history_window} cycles"

        # Use JSON builder for prompt generation (Enables new features like slot constraint instructions)
        from src.ai.prompt_json_builders import build_position_slot_json

        # Calculate slot status for prompt context
        position_slots = build_position_slot_json(
            self.portfolio.positions,
            self.get_max_positions_for_cycle(self.current_cycle_number),
        )

        # Add slot constraint instruction to the prompt if applicable
        slot_instruction = ""
        if position_slots.get("constraint_mode") != "NORMAL":
            slot_instruction = (
                f"\nIMPORTANT CONSTRAINT: {position_slots.get('constraint_instruction')}\n"
            )

        prompt = f"""
USER_PROMPT:
It has been {minutes_running} minutes since you started trading. The current time is {current_time} and you've been invoked {self.invocation_count} times. Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. Below that is your current account information, value, performance, etc.

{slot_instruction}

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST -> NEWEST
Timeframes note: Unless stated otherwise in a section title, intraday series are provided at 3-minute intervals. If a coin uses a different interval, it is explicitly stated in that coin's section.

{"=" * 20} REAL-TIME COUNTER-TRADE ANALYSIS {"=" * 20}

(Counter-trade analysis is now included in JSON format below)

{"=" * 20} TREND REVERSAL DETECTION {"=" * 20}

{self.format_trend_reversal_analysis(trend_reversal_analysis)}

{"=" * 20} ENHANCED DECISION CONTEXT (Non-binding suggestions) {"=" * 20}

POSITION MANAGEMENT CONTEXT:
{self.format_position_context(enhanced_context.get("position_context", {}))}

MARKET REGIME ANALYSIS:
{self.format_market_regime_context(enhanced_context.get("market_regime", {}))}

PERFORMANCE INSIGHTS:
{self.format_performance_insights(enhanced_context.get("performance_insights", {}))}

DIRECTIONAL FEEDBACK (LONG vs SHORT):
{self.format_directional_feedback(enhanced_context.get("directional_feedback", {}))}

DIRECTIONAL PERFORMANCE SNAPSHOT (Last 20 trades max):
{bias_section}

DIRECTIONAL COOLDOWN STATUS (CRITICAL - DO NOT PROPOSE TRADES IN COOLDOWN DIRECTIONS):
{cooldown_section}

[WARNING] IMPORTANT: If a direction (LONG or SHORT) is in cooldown, you MUST NOT propose any new trades in that direction. The system will block them, but you should avoid proposing them in the first place. Cooldown is activated after 3 consecutive losses OR $5+ total loss in a direction.

COIN COOLDOWN STATUS (CRITICAL - DO NOT PROPOSE TRADES FOR COINS IN COOLDOWN):
{coin_cooldown_section}

[WARNING] IMPORTANT: If a coin is in cooldown, you MUST NOT propose any new trades for that coin (LONG or SHORT). The system will block them, but you should avoid proposing them in the first place. Coin cooldown is activated after a loss on that coin and lasts for 1 cycle.

RECENT TREND FLIP GUARD (Cooldown = {self.portfolio.trend_flip_cooldown} cycles | History = {flip_history_window} cycles):
{trend_flip_section}

RISK MANAGEMENT CONTEXT:
{self.format_risk_context(enhanced_context.get("risk_context", {}))}

SUGGESTIONS (Non-binding):
{self.format_suggestions(enhanced_context.get("suggestions", []))}

REMEMBER: These are suggestions only. You make the final trading decisions based on your systematic analysis.
"""

        directional_counts = self.portfolio.count_positions_by_direction()
        positions_by_direction: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
        now_ts = datetime.now(timezone.utc)
        for coin, position in self.portfolio.positions.items():
            direction = position.get("direction", "long")
            if direction not in positions_by_direction:
                continue
            pnl = position.get("unrealized_pnl", 0.0)
            entry_time_str = position.get("entry_time")
            minutes_in_trade = None
            if entry_time_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_time_str)
                    minutes_in_trade = max(0, int((now_ts - entry_dt).total_seconds() // 60))
                except (ValueError, TypeError):
                    # FIX: Specific exception handling for datetime parsing
                    minutes_in_trade = None
            positions_by_direction[direction].append(
                {
                    "coin": coin,
                    "pnl": pnl,
                    "minutes": minutes_in_trade,
                    "loss_cycles": position.get("loss_cycle_count", 0),
                },
            )

        long_open = directional_counts.get("long", 0)
        short_open = directional_counts.get("short", 0)
        same_direction_limit = Config.SAME_DIRECTION_LIMIT
        total_open_positions = len(self.portfolio.positions)
        cycle_for_limits = max(1, getattr(self, "current_cycle_number", 1))
        cycle_position_cap = self.get_max_positions_for_cycle(cycle_for_limits)

        slot_lines = [
            f"  - Total open positions: {total_open_positions}/{cycle_position_cap} (cycle cap)",
            f"  - Long slots used: {long_open}/{same_direction_limit}",
            f"  - Short slots used: {short_open}/{same_direction_limit}",
        ]
        if long_open >= same_direction_limit:
            weakest_long = None
            if positions_by_direction["long"]:
                weakest_long = min(positions_by_direction["long"], key=lambda x: x["pnl"])
            if weakest_long:
                wl_minutes = (
                    f"{weakest_long['minutes']}min"
                    if weakest_long["minutes"] is not None
                    else "N/A"
                )
                slot_lines.append(
                    f"  - Weakest LONG -> {weakest_long['coin']} (PnL ${weakest_long['pnl']:.2f}, in trade {wl_minutes}, "
                    f"loss_cycles={weakest_long['loss_cycles']}). Evaluate trimming/closing this before proposing a new long.",
                )
            slot_lines.append(
                "  - Long capacity FULL -> System blocks new longs. Provide either (a) a close/trim plan for a current long "
                "OR (b) a SHORT setup (ONLY if no counter-trend LONG signal exists). CRITICAL: If a counter-trend LONG signal exists, DO NOT open a SHORT.",
            )
        if short_open >= same_direction_limit:
            weakest_short = None
            if positions_by_direction["short"]:
                weakest_short = min(positions_by_direction["short"], key=lambda x: x["pnl"])
            if weakest_short:
                ws_minutes = (
                    f"{weakest_short['minutes']}min"
                    if weakest_short["minutes"] is not None
                    else "N/A"
                )
                slot_lines.append(
                    f"  - Weakest SHORT -> {weakest_short['coin']} (PnL ${weakest_short['pnl']:.2f}, in trade {ws_minutes}, "
                    f"loss_cycles={weakest_short['loss_cycles']}). Evaluate trimming/closing this before proposing a new short.",
                )
            slot_lines.append(
                "  - Short capacity FULL -> System blocks new shorts. Provide either (a) a close/trim plan for a current short "
                "OR (b) a LONG alternative (ONLY if no counter-trend SHORT signal exists). CRITICAL: If a counter-trend SHORT signal exists, DO NOT open a LONG.",
            )

        prompt += f"\n{'=' * 20} POSITION SLOT STATUS {'=' * 20}\n" + "\n".join(slot_lines) + "\n"

        # --- Loop through available coins ---
        # OPTIMIZATION: Use pre-fetched indicators instead of re-fetching
        self.latest_indicator_cache = {}

        for coin in self.market_data.available_coins:
            prompt += f"\n{'=' * 20} ALL {coin} DATA {'=' * 20}\n"
            # Use pre-fetched indicators (no re-fetch)
            indicators_3m = all_indicators.get(coin, {}).get("3m", {})
            indicators_15m = all_indicators.get(coin, {}).get("15m", {})
            indicators_htf = all_indicators.get(coin, {}).get(HTF_INTERVAL, {})
            sentiment = all_sentiment.get(coin, {})
            self.latest_indicator_cache[coin] = {
                "3m": copy.deepcopy(indicators_3m),
                "15m": copy.deepcopy(indicators_15m),
                HTF_INTERVAL: copy.deepcopy(indicators_htf),
            }

            # Key level detection
            current_price = indicators_htf.get("current_price")
            nearest_support = indicators_htf.get("nearest_support")
            nearest_resistance = indicators_htf.get("nearest_resistance")
            key_level = None

            # Check support
            if nearest_support:
                distance_pct = (current_price - nearest_support) / current_price * 100
                if distance_pct < constants.LEVEL_PROXIMITY_THRESHOLD:
                    key_level = {
                        "type": "support",
                        "price": nearest_support,
                        "distance_pct": round(distance_pct, 2),
                    }

            # Check resistance
            if nearest_resistance:
                distance_pct = (nearest_resistance - current_price) / current_price * 100

                if distance_pct < constants.LEVEL_PROXIMITY_THRESHOLD:
                    key_level = {
                        "type": "resistance",
                        "price": nearest_resistance,
                        "distance_pct": round(distance_pct, 2),
                    }
                    prompt += f"[LEVEL] Nearest Resistance: ${format_num(nearest_resistance)} ({key_level['distance_pct']}% distance)\n"

            # Add market regime detection
            market_regime = self.strategy_analyzer.detect_market_regime(
                coin,
                indicators_htf=indicators_htf,
                indicators_3m=indicators_3m,
                indicators_15m=indicators_15m,
            )
            prompt += f"--- MARKET REGIME: {market_regime} ---\n"

            prompt += f"--- Market Sentiment for {coin} Perps ---\n"
            prompt += (
                f"Open Interest: Latest: {format_num(sentiment.get('open_interest', 'N/A'), 2)}\n"
            )
            funding_rate = sentiment.get("funding_rate", 0.0)
            prompt += f"Funding Rate: {format_num(funding_rate, 8)} ({format_num(funding_rate * 100, 4)}%)\n\n"

            # --- Inner function to format indicators ---
            def format_indicators(indicators, prefix=""):
                if not isinstance(indicators, dict) or "error" in indicators:
                    error_msg = (
                        indicators.get("error", "Unknown error")
                        if isinstance(indicators, dict)
                        else "Invalid indicator data"
                    )
                    return f"{prefix}Error fetching indicator data: {error_msg}\n"
                # Format numbers using global helper
                current_price = indicators.get("current_price")
                period_low = indicators.get("low_series", [current_price])[-1]
                period_high = indicators.get("high_series", [current_price])[-1]
                price_range = period_high - period_low

                # Zone detection
                percentile = (
                    ((current_price - period_low) / price_range) * 100 if price_range > 0 else 50
                )
                zone = (
                    "LOWER_10"
                    if percentile <= constants.EXTREME_PERCENTILE_LOW
                    else (
                        "UPPER_10" if percentile >= constants.EXTREME_PERCENTILE_HIGH else "MIDDLE"
                    )
                )

                output = f"{prefix}current_price = {format_num(indicators.get('current_price', 'N/A'))} (Zone: {zone})\n"
                output += f"{prefix}Mid prices (last {len(indicators.get('price_series', []))}): {self.format_list(indicators.get('price_series', []))}\n"
                output += f"{prefix}EMA indicators (20-period): {self.format_list(indicators.get('ema_20_series', []))}\n"
                if "rsi_7_series" in indicators:
                    output += f"{prefix}RSI indicators (7-Period): {self.format_list(indicators.get('rsi_7_series', []), precision=3)}\n"
                output += f"{prefix}RSI indicators (14-Period): {self.format_list(indicators.get('rsi_14_series', []), precision=3)}\n"
                output += f"{prefix}MACD indicators: {self.format_list(indicators.get('macd_series', []))}\n"
                atr_3 = indicators.get("atr_3")
                atr_14 = indicators.get("atr_14")
                atr_str = ""
                if atr_3 is not None and pd.notna(atr_3):
                    atr_str += f"{prefix}3-Period ATR: {format_num(atr_3)} vs "
                atr_str += f"14-Period ATR: {format_num(atr_14)}\n"
                output += atr_str
                current_volume = indicators.get("volume", "N/A")
                avg_volume = indicators.get("avg_volume", "N/A")
                output += f"{prefix}Current Volume: {format_num(current_volume, 3)} vs. Average Volume: {format_num(avg_volume, 3)}\n"
                output += f"{prefix}Volume ratio (current/avg): {self.format_volume_ratio(current_volume, avg_volume)}\n"
                return output

            # --- End inner function ---

            prompt += "--- LIQUIDITY & MICRO-REACTION (3-Minute Noise Layer) ---\n"
            prompt += "[CONTEXT] Use this section ONLY for identifying liquidity surges (Volume Ratio) or immediate exhaustion spikes. Do NOT use it as a primary trend gate.\n"
            prompt += format_indicators(indicators_3m)
            prompt += "\n--- Medium-term context (15-minute intervals) ---\n"
            prompt += format_indicators(indicators_15m)
            prompt += f"\n--- Longer-term context ({HTF_LABEL} timeframe) ---\n"
            prompt += format_indicators(indicators_htf)

            # --- Add current position details if open ---
            if coin in self.portfolio.positions:
                position = self.portfolio.positions[coin]
                prompt += "\n--- CURRENT OPEN POSITION & YOUR PLAN ---\n"
                prompt += (
                    f"You have an open {position.get('direction', 'long').upper()} position.\n"
                )
                prompt += f"  Symbol: {position.get('symbol', 'N/A')}\n"
                prompt += f"  Quantity: {format_num(position.get('quantity', 0), 6)}\n"
                prompt += f"  Entry Price: ${format_num(position.get('entry_price', 0))}\n"
                prompt += f"  Current Price: ${format_num(position.get('current_price', 0))}\n"
                prompt += f"  Liquidation Price (Est.): ${format_num(position.get('liquidation_price', 0))}\n"
                prompt += f"  Unrealized PnL: ${format_num(position.get('unrealized_pnl', 0), 2)}\n"
                prompt += f"  Leverage: {position.get('leverage', 1)}x\n"
                prompt += f"  Notional Value: ${format_num(position.get('notional_usd', 0), 2)}\n"

                # Calculate position duration
                entry_time_str = position.get("entry_time")
                position_duration_minutes = None
                position_duration_hours = None
                if entry_time_str:
                    try:
                        entry_dt = datetime.fromisoformat(entry_time_str)
                        position_duration_minutes = max(
                            0,
                            int((datetime.now(timezone.utc) - entry_dt).total_seconds() // 60),
                        )
                        position_duration_hours = position_duration_minutes / 60.0
                    except (ValueError, TypeError, AttributeError):
                        # FIX: Specific exception handling for datetime calculation
                        pass

                if position_duration_minutes is not None:
                    if position_duration_hours >= 1:
                        prompt += f"  Position Duration: {position_duration_hours:.1f} hours ({position_duration_minutes} minutes)\n"
                    else:
                        prompt += f"  Position Duration: {position_duration_minutes} minutes\n"

                # Get current trend state
                trend_info = self.portfolio.update_trend_state(coin, indicators_htf, indicators_3m)
                current_trend = trend_info.get("trend", "unknown")
                trend_direction = position.get("direction", "long").lower()

                # Determine 3m momentum
                price_3m = indicators_3m.get("current_price")
                ema20_3m = indicators_3m.get("ema_20")
                rsi_3m = indicators_3m.get("rsi_14", indicators_3m.get("rsi_7", 50))
                momentum_3m = "unknown"
                if (
                    isinstance(price_3m, (int, float))
                    and isinstance(ema20_3m, (int, float))
                    and ema20_3m > 0
                ):
                    # Trend structure detection
                    slope_pct = indicators_3m.get("slope_pct", 0)
                    peaks = indicators_3m.get("peaks", [])
                    valleys = indicators_3m.get("valleys", [])
                    structure = "SIDEWAYS"
                    if slope_pct > constants.TREND_SLOPE_THRESHOLD:
                        structure = "UPTREND"
                        if len(peaks) >= constants.MIN_PEAKS_VALLEYS and peaks[-1] < peaks[-2]:
                            structure = "UPTREND_LOSING_MOMENTUM"
                    elif slope_pct < -constants.TREND_SLOPE_THRESHOLD:
                        structure = "DOWNTREND"
                        if (
                            len(valleys) >= constants.MIN_PEAKS_VALLEYS
                            and valleys[-1] > valleys[-2]
                        ):
                            structure = "DOWNTREND_LOSING_MOMENTUM"

                    prompt += f"  3m Price Structure: {structure}\n"
                    if price_3m > ema20_3m:
                        momentum_3m = "bullish"
                    elif price_3m < ema20_3m:
                        momentum_3m = "bearish"

                # Determine 15m momentum
                price_15m = indicators_15m.get("current_price")
                ema20_15m = indicators_15m.get("ema_20")
                rsi_15m = indicators_15m.get("rsi_14", indicators_15m.get("rsi_7", 50))
                momentum_15m = "unknown"
                if (
                    isinstance(price_15m, (int, float))
                    and isinstance(ema20_15m, (int, float))
                    and ema20_15m > 0
                ):
                    if price_15m > ema20_15m:
                        momentum_15m = "bullish"
                    elif price_15m < ema20_15m:
                        momentum_15m = "bearish"

                # Check for potential trend reversal using HTF trend, 15m momentum, and 3m momentum
                trend_reversal_warning = ""
                reversal_signals = []

                # HTF trend reversal check
                if current_trend == "bullish" and trend_direction == "short":
                    reversal_signals.append(f"{HTF_LABEL} trend flipped to BULLISH")
                elif current_trend == "bearish" and trend_direction == "long":
                    reversal_signals.append(f"{HTF_LABEL} trend flipped to BEARISH")

                # 15m momentum reversal check (medium-term confirmation)
                if momentum_15m == "bullish" and trend_direction == "short":
                    reversal_signals.append("15m momentum turned BULLISH")
                elif momentum_15m == "bearish" and trend_direction == "long":
                    reversal_signals.append("15m momentum turned BEARISH")

                # 3m momentum reversal check (more sensitive, earlier signal)
                if momentum_3m == "bullish" and trend_direction == "short":
                    reversal_signals.append("3m momentum turned BULLISH")
                elif momentum_3m == "bearish" and trend_direction == "long":
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
                        signals_text = " & ".join(
                            [s for s in reversal_signals if "15m" in s or "3m" in s],
                        )

                        if trend_direction == "short":
                            trend_reversal_warning = f"[INFO] {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a SHORT position but momentum is showing bullish signs. "
                            trend_reversal_warning += "15m and 3m momentum both show bullish signs - strong reversal signal. This can be a counter-trend opportunity. Evaluate your exit plan and consider if the position thesis is still valid."
                        else:  # long position
                            trend_reversal_warning = f"[INFO] {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a LONG position but momentum is showing bearish signs. "
                            trend_reversal_warning += "15m and 3m momentum both show bearish signs - strong reversal signal. This can be a counter-trend opportunity. Evaluate your exit plan and consider if the position thesis is still valid."
                    elif signal_3m:
                        # Only 3m shows reversal (medium reversal signal)
                        signal_strength = "MEDIUM"
                        signals_text = " & ".join([s for s in reversal_signals if "3m" in s])
                        if trend_direction == "short":
                            trend_reversal_warning = f"[INFO] {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a SHORT position but 3m momentum is showing bullish signs. "
                            trend_reversal_warning += "3m momentum shows bullish signs - medium reversal signal. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                        else:  # long position
                            trend_reversal_warning = f"[INFO] {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a LONG position but 3m momentum is showing bearish signs. "
                            trend_reversal_warning += "3m momentum shows bearish signs - medium reversal signal. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                    elif signal_15m:
                        # Only 15m shows reversal (informational)
                        signal_strength = "INFORMATIONAL"
                        signals_text = " & ".join([s for s in reversal_signals if "15m" in s])
                        if trend_direction == "short":
                            trend_reversal_warning = f"[INFO] {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a SHORT position but 15m momentum is showing bullish signs. "
                            trend_reversal_warning += "15m momentum shows bullish signs - this is informational context. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                        else:  # long position
                            trend_reversal_warning = f"[INFO] {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a LONG position but 15m momentum is showing bearish signs. "
                            trend_reversal_warning += "15m momentum shows bearish signs - this is informational context. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                    else:
                        # Only HTF signal (shouldn't happen, but handle it)
                        signal_strength = "INFORMATIONAL"
                        signals_text = " & ".join(reversal_signals)
                        if trend_direction == "short":
                            trend_reversal_warning = f"[INFO] {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a SHORT position but momentum is showing bullish signs. "
                            trend_reversal_warning += "Short-term momentum shows bullish signs - this is informational context. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."
                        else:  # long position
                            trend_reversal_warning = f"[INFO] {signal_strength} REVERSAL SIGNAL ({signals_text}): You have a LONG position but momentum is showing bearish signs. "
                            trend_reversal_warning += "Short-term momentum shows bearish signs - this is informational context. Continue monitoring but prioritize {HTF_LABEL} trend confirmation before making exit decisions."

                # Extended position duration warning
                if (
                    position_duration_hours is not None
                    and position_duration_hours >= constants.PNL_TREND_LOOKBACK
                ):
                    if trend_reversal_warning:
                        trend_reversal_warning += f"\n  [INFO] POSITION DURATION: This {trend_direction.upper()} position has been open for {position_duration_hours:.1f} hours. Review your exit plan and ensure it's still aligned with current market conditions."
                    else:
                        trend_reversal_warning = f"[INFO] POSITION DURATION: This {trend_direction.upper()} position has been open for {position_duration_hours:.1f} hours. This is informational - ensure your exit plan remains valid."

                if trend_reversal_warning:
                    prompt += f"\n  {trend_reversal_warning}\n"

                prompt += f"  Current {HTF_LABEL} Trend: {current_trend.upper()}\n"
                prompt += f"  Current 15m Momentum: {momentum_15m.upper()}\n"
                prompt += f"  Current 3m Momentum: {momentum_3m.upper()}\n"
                if isinstance(rsi_15m, (int, float)):
                    prompt += f"  15m RSI: {rsi_15m:.1f}\n"
                if isinstance(rsi_3m, (int, float)):
                    prompt += f"  3m RSI: {rsi_3m:.1f}\n"

                exit_plan = position.get("exit_plan", {})
                prompt += "  YOUR ACTIVE EXIT PLAN:\n"
                prompt += f"    Profit Target: {exit_plan.get('profit_target', 'N/A')}\n"
                prompt += f"    Stop Loss: {exit_plan.get('stop_loss', 'N/A')}\n"
                prompt += f"    Invalidation: {exit_plan.get('invalidation_condition', 'N/A')}\n"
                prompt += f"  Your Confidence: {position.get('confidence', 'N/A')}\n"
                prompt += f"  Estimated Risk USD: {position.get('risk_usd', 'N/A')}\n"
                prompt += "REMINDER: You can only 'hold' or 'close_position'.\n"
        # --- End coin loop ---

        self.portfolio.indicator_cache = copy.deepcopy(self.latest_indicator_cache)

        # Add historical context section
        trading_context = self.get_trading_context()

        # Calculate current risk status - NEW SIMPLIFIED SYSTEM
        total_margin_used = sum(
            pos.get("margin_usd", 0) for pos in self.portfolio.positions.values()
        )
        current_positions_count = len(self.portfolio.positions)

        prompt += f"""
{"=" * 20} HISTORICAL CONTEXT (Last {trading_context["total_cycles_analyzed"]} Cycles) {"=" * 20}

Market Behavior: {trading_context["market_behavior"]}
Recent Trading Decisions: {json.dumps(trading_context["recent_decisions"], indent=2)}
{"=" * 20} REAL-TIME RISK STATUS {"=" * 20}

CURRENT STATUS: {current_positions_count} positions open, ${format_num(total_margin_used, 2)} margin used
AVAILABLE CASH: ${format_num(self.portfolio.current_balance, 2)}
TRADING LIMITS:
- Minimum position: $10
- Maximum positions: 5
- Available cash protection: Never below ${format_num(self.portfolio.current_balance * 0.10, 2)}
- Position sizing: Automatic based on confidence (up to 40% of available cash)

{"=" * 20} HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE {"=" * 20}

Current Total Return (percent): {format_num(self.portfolio.total_return, 2)}%
Available Cash: {format_num(self.portfolio.current_balance, 2)}
Current Account Value: {format_num(self.portfolio.total_value, 2)}
Sharpe Ratio: {format_num(self.portfolio.sharpe_ratio, 3)}

Current live positions & performance:"""

        if not self.portfolio.positions:
            prompt += " No open positions. (100% cash)"

        return prompt

    def generate_alpha_arena_prompt_json(self) -> str:
        """Generate hybrid JSON prompt with structured data sections.

        Uses JSON for data, plain text for instructions and warnings.
        This is the recommended method for prompt generation.

        Returns:
        -------
            str: Hybrid prompt with JSON sections and text instructions

        Note:
        ----
            Falls back to text format if JSON serialization fails.
            See :meth:`generate_alpha_arena_prompt` for text-only format (deprecated).

        """
        from config.config import Config
        from src.ai.prompt_json_builders import (
            build_coin_state_vector,
            build_cooldown_status_json,
            build_counter_trade_json,
            build_directional_bias_json,
            build_historical_context_json,
            build_metadata_json,
            build_portfolio_json,
            build_position_slot_json,
            build_risk_status_json,
            build_trend_reversal_json,
        )
        from src.ai.prompt_json_utils import (
            create_json_section,
        )

        current_time = datetime.now(timezone.utc)
        minutes_running = int((current_time - self.portfolio.start_time).total_seconds() / 60)

        # Boot ML Inference Service
        ml_service = MLService()

        # Fetch all indicators in parallel (same as original)
        all_indicators, all_sentiment = self._fetch_all_indicators_parallel()
        self.latest_indicators = all_indicators
        self.latest_sentiment = all_sentiment

        # Get enhanced context and other data
        # NOTE: enhanced_context removed - data was 100% redundant with PORTFOLIO/RISK_STATUS

        # Get trend reversal detection

        performance_monitor = PerformanceMonitor()
        trend_reversal_analysis = performance_monitor.detect_trend_reversal_for_all_coins(
            self.market_data.available_coins,
            indicators_cache=all_indicators,
        )

        # Get cooldown status
        directional_cooldowns = self.portfolio.directional_cooldowns
        coin_cooldowns = self.portfolio.coin_cooldowns
        counter_trend_cooldown = self.portfolio.counter_trend_cooldown
        relaxed_countertrend_cycles = self.portfolio.relaxed_countertrend_cycles

        # Get trading context
        trading_context = self.get_trading_context()

        # Get directional bias metrics (for performance snapshot)
        bias_metrics = getattr(self, "latest_bias_metrics", self.get_directional_bias_metrics())

        # Get trend flip summary
        self.portfolio.get_recent_trend_flip_summary()
        getattr(
            self.portfolio,
            "trend_flip_history_window",
            self.portfolio.trend_flip_cooldown,
        )

        # Build JSON sections
        compact = Config.JSON_PROMPT_COMPACT

        # === COOLDOWN COIN FILTERING ===
        # Skip coins in cooldown that have no position from prompt (save tokens + focus AI)
        # Data collection continues normally for regime calculation
        # NOTE: Volume filtering removed - AI handles via prompt rules (0.3x threshold)
        coins_to_analyze = []
        skipped_cooldown_coins = []

        for coin in self.market_data.available_coins:
            has_position = coin in self.portfolio.positions
            cooldown_cycles = coin_cooldowns.get(coin, 0)

            if cooldown_cycles > 0 and not has_position:
                skipped_cooldown_coins.append(f"{coin}({cooldown_cycles})")
            else:
                # NOTE: Removed volume hard filter - AI will handle via prompt rules (0.3x threshold)
                coins_to_analyze.append(coin)

        if skipped_cooldown_coins:
            print(
                f"[INFO]  Prompt optimization: Skipped cooldown coins (no position): {skipped_cooldown_coins}",
            )

        # Metadata
        build_metadata_json(minutes_running, current_time, self.invocation_count)

        # Counter-trade risk analysis (compact dict: {coin: {risk_level, alignment_strength, conditions_met}})
        counter_trade_risks = build_counter_trade_json(
            "",  # Legacy parameter, not used - function calculates internally
            all_indicators,
            coins_to_analyze,  # Filtered list
            HTF_INTERVAL,
            self.market_data,  # INFO: Integrated for Funding Rate calculation
        )

        # Trend reversal threats (compact dict: {coin: {strength}})
        reversal_threats = build_trend_reversal_json(
            trend_reversal_analysis,
            self.portfolio.positions,
        )

        # Cooldown status
        cooldown_status_json = build_cooldown_status_json(
            directional_cooldowns,
            coin_cooldowns,
            counter_trend_cooldown,
            relaxed_countertrend_cycles,
        )

        # Position slot status
        max_positions = self.get_max_positions_for_cycle(
            max(1, getattr(self, "current_cycle_number", 1)),
        )
        effective_limit = self.portfolio.get_effective_same_direction_limit()
        position_slot_json = build_position_slot_json(
            self.portfolio.positions,
            max_positions,
            same_direction_limit=effective_limit,
        )

        # State Vector data (per coin) - only for tradeable coins
        market_data_json = []
        for coin in coins_to_analyze:  # Filtered list (excludes cooldown coins without position)
            indicators_3m = all_indicators.get(coin, {}).get("3m", {})
            indicators_15m = all_indicators.get(coin, {}).get("15m", {})
            indicators_htf = all_indicators.get(coin, {}).get(HTF_INTERVAL, {})
            sentiment = all_sentiment.get(coin, {})

            # Detect market regime
            market_regime = self.strategy_analyzer.detect_market_regime(
                coin,
                indicators_htf=indicators_htf,
                indicators_3m=indicators_3m,
                indicators_15m=indicators_15m,
            )

            # Get position if exists
            position = self.portfolio.positions.get(coin)

            # Get ML Consensus for Hybridization
            ml_consensus = None
            if ml_service.is_ready:
                # Reuse already-fetched 15m data from this cycle (no redundant API call)
                df_raw_15m = self.market_data.get_cached_raw_dataframe(coin, "15m")
                if df_raw_15m is not None and not df_raw_15m.empty:
                    ml_consensus = ml_service.predict(df_raw_15m, coin)
                else:
                    # Fallback: fetch fresh if cache miss (safety net)
                    df_raw_15m = self.market_data.get_real_time_data(coin, "15m", limit=150)
                    if not df_raw_15m.empty:
                        ml_consensus = ml_service.predict(df_raw_15m, coin)

            # Calculate Bias Deviation Score (BDS) - Professional Mitigation Layer
            ml_bias_label = "Neutral"
            if ml_consensus:
                ml_prob_buy = ml_consensus.get("BUY", 0)
                ml_prob_sell = ml_consensus.get("SELL", 0)
                htf_direction = indicators_htf.get("trend_direction", "neutral")

                # Flag deviation if ML is strong opposite to HTF trend
                if ml_prob_sell > Config.ML_CONFIDENCE_THRESHOLD and htf_direction == "bullish":
                    ml_bias_label = "Trend-Averse (Bearish ML vs Bullish HTF)"
                elif ml_prob_buy > Config.ML_CONFIDENCE_THRESHOLD and htf_direction == "bearish":
                    ml_bias_label = "Trend-Averse (Bullish ML vs Bearish HTF)"

            # Build State Vector (labels + numerical anchors)
            coin_state = build_coin_state_vector(
                coin,
                market_regime,
                sentiment,
                indicators_3m,
                indicators_15m,
                indicators_htf,
                position,
                ml_consensus=ml_consensus,
                ml_bias_label=ml_bias_label,  # Injecting the mitigation label
                counter_trade_result=counter_trade_risks.get(coin),
                reversal_result=reversal_threats.get(coin),
            )
            market_data_json.append(coin_state)

        # Portfolio
        portfolio_json = build_portfolio_json(self.portfolio)

        # Risk status
        risk_status_json = build_risk_status_json(self.portfolio, max_positions)

        # Historical context
        historical_context_json = build_historical_context_json(trading_context)

        # Directional bias (performance snapshot)
        directional_bias_json = build_directional_bias_json(bias_metrics)

        # Build hybrid prompt
        prompt = f"""
USER_PROMPT:
It has been {minutes_running} minutes since you started trading. The current time is {current_time} and you've been invoked {self.invocation_count} times. Below is your State Vector data for each coin, followed by account information.

DIRECTIONAL PERFORMANCE SNAPSHOT (Last 20 trades max):
{create_json_section("DIRECTIONAL_BIAS", directional_bias_json, compact=compact)}

[WARNING] IMPORTANT: If a direction (LONG or SHORT) is in cooldown, you MUST NOT propose any new trades in that direction. The system will block them, but you should avoid proposing them in the first place. Cooldown is activated after 3 consecutive losses OR $5+ total loss in a direction.

{create_json_section("COOLDOWN_STATUS", cooldown_status_json, compact=compact)}

[WARNING] IMPORTANT: If a coin is in cooldown, you MUST NOT propose any new trades for that coin (LONG or SHORT). The system will block them, but you should avoid proposing them in the first place. Coin cooldown is activated after a loss on that coin and lasts for 1 cycle.

{"=" * 20} POSITION_SLOTS {"=" * 20}

{create_json_section("POSITION_SLOTS", position_slot_json, compact=compact)}

[WARNING] CRITICAL: If "long_slots_available" is 0, do NOT propose LONG entries. If "short_slots_available" is 0, do NOT propose SHORT entries.
[WARNING] CRITICAL: If you identify a valid counter-trend opportunity (e.g. LONG) but cannot execute it because slots are full, you MUST NOT open a trend-following trade in the opposite direction (e.g. SHORT). The counter-trend signal invalidates the trend-following setup. Simply HOLD.

{"=" * 20} MARKET STATE VECTORS {"=" * 20}

Each coin below contains a State Vector with:
- ml_consensus: XGBoost probability (>45% BUY/SELL = strong signal). Combine with your own analysis.
- ml_bias_label: Indicates if ML is currently deviating from the HTF trend (Trend-Averse status).
- market_context: Regime, volatility state, price location labels.
- technical_summary: Trend alignment, momentum, volume, structure labels.
- key_levels: price, ema20_htf, rsi_15m, atr_htf for your independent reasoning.
- risk_profile: Counter-trade risk and reversal threat assessments.
- sentiment: Funding rate and open interest.
- position: Current position details (if exists).

{create_json_section("MARKET_DATA", market_data_json, compact=compact)}

{"=" * 20} HISTORICAL CONTEXT (Last {trading_context.get("total_cycles_analyzed", 0)} Cycles) {"=" * 20}

{create_json_section("HISTORICAL_CONTEXT", historical_context_json, compact=compact)}

{"=" * 20} REAL-TIME RISK STATUS {"=" * 20}

{create_json_section("RISK_STATUS", risk_status_json, compact=compact)}

{"=" * 20} HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE {"=" * 20}

{create_json_section("PORTFOLIO", portfolio_json, compact=compact)}

[DIRECTIVE] STRATEGIC PRIORITY:
1. Technical Confluence (HTF Trend + 15m Momentum + Volume) is your PRIMARY source of truth.
2. ML Consensus is a statistical probability tool. If it shows "Trend-Averse" bias, DE-PRIORITIZE it and stick to the clear technical trend.
3. Only perform counter-trend trades if Counter-Trade Risk is LOW and price is at extreme exhaustion (RSI/BB).
"""

        # Validate JSON if enabled
        if Config.VALIDATE_JSON_PROMPTS:
            pass

        return prompt

    def parse_ai_response(self, response: str) -> dict[str, Any]:
        """Parse AI response - expects clean JSON string from DeepSeekAPI"""
        try:
            parsed_json = json.loads(response)
            if not isinstance(parsed_json, dict):
                print(f"[ERR]   Parsed JSON not dict: {type(parsed_json)}")
                return {"chain_of_thoughts": "Error: Parsed JSON not dict.", "decisions": {}}

            thoughts = parsed_json.get("CHAIN_OF_THOUGHTS", "No thoughts provided.")
            decisions = parsed_json.get("DECISIONS", {})
            decisions = self._clean_ai_decisions(decisions)
            return {"chain_of_thoughts": thoughts, "decisions": decisions}
        except json.JSONDecodeError as e:
            print(f"[ERR]   JSON decode error: {e}")
            return {"chain_of_thoughts": f"Error: JSON decode failed: {e}", "decisions": {}}
        except Exception as e:
            print(f"[ERR]   General parse error: {e}")
            return {"chain_of_thoughts": f"Error: {e}", "decisions": {}}

    def _clean_ai_decisions(self, decisions: dict) -> dict:
        """Clean up AI decisions - preserve position data for hold signals"""
        cleaned_decisions = {}
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                cleaned_decisions[coin] = trade
                continue
            signal = trade.get("signal")
            if signal == "hold":
                cleaned_trade = {"signal": "hold"}
                if coin in self.portfolio.positions:
                    position = self.portfolio.positions[coin]
                    cleaned_trade.update(
                        {
                            "leverage": position.get("leverage", 1),
                            "quantity_usd": position.get("margin_usd", 0),
                            "confidence": position.get("confidence", 0.5),
                            "profit_target": position.get("exit_plan", {}).get("profit_target"),
                            "stop_loss": position.get("exit_plan", {}).get("stop_loss"),
                            "risk_usd": position.get("risk_usd", 0),
                            "invalidation_condition": position.get("exit_plan", {}).get(
                                "invalidation_condition",
                            ),
                            "entry_price": position.get("entry_price", 0),
                            "current_price": position.get("current_price", 0),
                            "unrealized_pnl": position.get("unrealized_pnl", 0),
                            "notional_usd": position.get("notional_usd", 0),
                            "direction": position.get("direction", "long"),
                        },
                    )
                cleaned_decisions[coin] = cleaned_trade
            else:
                cleaned_decisions[coin] = trade
        return cleaned_decisions

    # --- Formatting Methods (v1.2 Restoration) ---
    def format_list(self, items: list, precision: int = 5) -> str:
        """Formats a list of numbers into a readable string."""
        if not items:
            return "[]"
        formatted = []
        for x in items:
            if isinstance(x, (int, float)):
                formatted.append(f"{x:.{precision}f}")
            else:
                formatted.append(str(x))
        return "[" + ", ".join(formatted) + "]"

    def format_volume_ratio(self, current: float, avg: float) -> str:
        """Calculates and formats volume ratio."""
        try:
            if not avg or avg == 0:
                return "1.00x"
            return f"{(current / avg):.2f}x"
        except:
            return "1.00x"

    def format_trend_reversal_analysis(self, analysis: dict) -> str:
        """Formats trend reversal analysis for AI prompt."""
        if not analysis:
            return "No active reversal threats identified."
        lines = []
        for coin, data in analysis.items():
            strength = data.get("strength_score", 0)
            if strength > constants.REVERSAL_STRENGTH_MODERATE:
                lines.append(
                    f"  - {coin}: REVERSAL THREAT (strength={strength:.2f}). Reason: {data.get('reason_summary', 'Unknown')}"
                )
        return "\n".join(lines) if lines else "No significant reversal threats identified."

    def format_position_context(self, context: dict) -> str:
        """Formats enhanced position context."""
        if not context:
            return "No enhanced position data available."
        lines = []
        for coin, data in context.items():
            lines.append(
                f"  - {coin}: PnL ${data.get('unrealized_pnl', 0):.2f}, Progress {data.get('profit_target_progress', 0):.1f}% to target, TiT {data.get('time_in_trade_minutes', 0):.1f}m"
            )
        return "\n".join(lines)

    def format_market_regime_context(self, context: dict) -> str:
        """Formats market regime context."""
        if not context:
            return "Market regime data unavailable."
        return f"  Overall Regime: {context.get('current_regime', 'unknown').upper()} (Strength: {context.get('regime_strength', 0):.2f})"

    def format_performance_insights(self, context: dict) -> str:
        """Formats performance insights."""
        insights = context.get("insights", [])
        if not insights:
            return "No specific performance insights for this cycle."
        return "\n".join(f"  - {i}" for i in insights)

    def format_directional_feedback(self, context: dict) -> str:
        """Formats directional bias feedback."""
        if not context:
            return "No directional bias feedback available."
        lines = []
        for side in ["long", "short"]:
            data = context.get(side, {})
            lines.append(
                f"  - {side.upper()}: PI={data.get('profitability_index', 0)}% (Trades: {data.get('trades', 0)})"
            )
        return "\n".join(lines)

    def format_risk_context(self, context: dict) -> str:
        """Formats risk management context."""
        if not context:
            return "Risk context unavailable."
        return f"  Total Risk: ${context.get('total_risk_usd', 0):.2f} across {context.get('position_count', 0)} positions. Div-score: {context.get('diversification_score', 0):.1f}"

    def format_suggestions(self, suggestions: list) -> str:
        """Formats suggestions list."""
        if not suggestions:
            return "Continue executing core logic."
        return "\n".join(f"  - {s}" for s in suggestions)
