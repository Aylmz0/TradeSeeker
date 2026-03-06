"""
JSON builder functions for AI prompt generation.
Converts data structures to JSON format for hybrid prompt.
"""

from datetime import datetime
from typing import Any

from config.config import Config
from src.ai.prompt_json_utils import format_number_for_json


# ============================================================================
# State Vector Architecture — Helper Functions
# ============================================================================


def _sv_fmt(value) -> float | int | None:
    """Shorthand for format_number_for_json in state vector context."""
    return format_number_for_json(value)


def _sv_trend_alignment(
    indicators_htf: dict[str, Any],
    indicators_15m: dict[str, Any],
    indicators_3m: dict[str, Any],
) -> str:
    """
    Determine multi-timeframe trend alignment.
    Returns: FULL_BULLISH | FULL_BEARISH | MIXED_BULLISH | MIXED_BEARISH | CONFLICTED
    """

    def _trend(ind: dict[str, Any]) -> str | None:
        if not ind or "error" in ind:
            return None
        price = ind.get("current_price")
        ema = ind.get("ema_20")
        if price is None or ema is None:
            return None
        return "BULLISH" if price > ema else "BEARISH"

    t_htf = _trend(indicators_htf)
    t_15m = _trend(indicators_15m)
    t_3m = _trend(indicators_3m)

    trends = [t for t in [t_htf, t_15m, t_3m] if t is not None]
    if not trends:
        return "UNKNOWN"

    bullish_count = trends.count("BULLISH")
    bearish_count = trends.count("BEARISH")

    if bullish_count == len(trends):
        return "FULL_BULLISH"
    if bearish_count == len(trends):
        return "FULL_BEARISH"
    if bullish_count > bearish_count:
        return "MIXED_BULLISH"
    if bearish_count > bullish_count:
        return "MIXED_BEARISH"
    return "CONFLICTED"


def _sv_momentum(indicators_15m: dict[str, Any]) -> str:
    """
    Extract momentum from 15m smart_sparkline.
    Returns: STRENGTHENING | STABLE | WEAKENING | UNKNOWN
    """
    if not indicators_15m or "error" in indicators_15m:
        return "UNKNOWN"
    sparkline = indicators_15m.get("smart_sparkline", {})
    if isinstance(sparkline, dict):
        return sparkline.get("momentum", "STABLE")
    return "STABLE"


def _sv_price_location(indicators_15m: dict[str, Any]) -> str:
    """
    Extract price location zone from 15m smart_sparkline.
    Returns: UPPER_10 | LOWER_10 | MIDDLE
    """
    if not indicators_15m or "error" in indicators_15m:
        return "MIDDLE"
    sparkline = indicators_15m.get("smart_sparkline", {})
    if isinstance(sparkline, dict):
        loc = sparkline.get("price_location", {})
        if isinstance(loc, dict):
            return loc.get("zone", "MIDDLE")
        if isinstance(loc, str):
            return loc
    return "MIDDLE"


def _sv_structure(indicators_15m: dict[str, Any]) -> str:
    """
    Extract market structure from 15m smart_sparkline.
    Returns: HH_HL | LH_LL | RANGE | UNCLEAR
    """
    if not indicators_15m or "error" in indicators_15m:
        return "UNCLEAR"
    sparkline = indicators_15m.get("smart_sparkline", {})
    if isinstance(sparkline, dict):
        return sparkline.get("structure", "UNCLEAR")
    return "UNCLEAR"


def _sv_volatility_state(indicators_htf: dict[str, Any]) -> str:
    """
    Determine volatility state from BB squeeze.
    Returns: SQUEEZE | EXPANDING | NORMAL
    """
    if not indicators_htf or "error" in indicators_htf:
        return "NORMAL"
    if indicators_htf.get("bb_squeeze"):
        return "SQUEEZE"
    bb_upper = indicators_htf.get("bb_upper")
    bb_lower = indicators_htf.get("bb_lower")
    bb_middle = indicators_htf.get("bb_middle")
    if bb_upper and bb_lower and bb_middle and bb_middle > 0:
        width_pct = ((bb_upper - bb_lower) / bb_middle) * 100
        if width_pct > 4.0:
            return "EXPANDING"
    return "NORMAL"


def _sv_volume_label(indicators_3m: dict[str, Any]) -> str:
    """
    Convert volume ratio to linguistic label.
    Returns: EXCELLENT | GOOD | FAIR | POOR | LOW
    """
    if not indicators_3m or "error" in indicators_3m:
        return "LOW"
    vol = indicators_3m.get("volume", 0)
    avg_vol = indicators_3m.get("avg_volume", 1)
    if not avg_vol or avg_vol == 0:
        return "LOW"

    # Use pre-calculated volume_ratio if available
    if "volume_ratio" in indicators_3m:
        ratio = indicators_3m["volume_ratio"]
    else:
        ratio = (vol or 0) / (avg_vol or 1)

    if ratio > 2.5:
        return "EXCELLENT"
    if ratio > 1.8:
        return "GOOD"
    if ratio > 1.2:
        return "FAIR"
    if ratio > 0.7:
        return "POOR"
    return "LOW"


def _sv_build_position(position: dict[str, Any]) -> dict[str, Any]:
    """Build compact position data for state vector."""
    pos = {
        "direction": position.get("direction", "long"),
        "entry_price": _sv_fmt(position.get("entry_price", 0)),
        "current_price": _sv_fmt(position.get("current_price", 0)),
        "unrealized_pnl": _sv_fmt(position.get("unrealized_pnl", 0)),
        "leverage": position.get("leverage", 1),
        "confidence": _sv_fmt(position.get("confidence", 0.5)),
        "exit_plan": {
            "profit_target": _sv_fmt(position.get("exit_plan", {}).get("profit_target")),
            "stop_loss": _sv_fmt(position.get("exit_plan", {}).get("stop_loss")),
            "invalidation_condition": position.get("exit_plan", {}).get(
                "invalidation_condition",
            ),
        },
    }
    # Profit erosion tracking — only include if meaningful
    erosion_status = position.get("erosion_status", "NONE")
    pos["erosion_status"] = erosion_status
    if erosion_status != "NONE":
        pos["peak_pnl"] = _sv_fmt(position.get("peak_pnl", 0))
        pos["erosion_pct"] = _sv_fmt(position.get("erosion_pct", 0))
    return pos


def build_coin_state_vector(
    coin: str,
    market_regime: str,
    sentiment: dict[str, Any],
    indicators_3m: dict[str, Any],
    indicators_15m: dict[str, Any],
    indicators_htf: dict[str, Any],
    position: dict[str, Any] | None = None,
    ml_consensus: dict[str, Any] | None = None,
    counter_trade_result: dict[str, Any] | None = None,
    reversal_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a refined State Vector for a single coin.

    Combines pre-processed labels (for fast AI orientation) with key numerical
    anchors (for independent AI reasoning). This replaces the old
    build_market_data_json which dumped 51 raw indicator fields per coin.

    Args:
        coin: Coin symbol
        market_regime: Market regime (BULLISH/BEARISH/NEUTRAL)
        sentiment: Sentiment data (OI, funding rate)
        indicators_3m: 3-minute timeframe indicators
        indicators_15m: 15-minute timeframe indicators
        indicators_htf: Higher timeframe indicators
        position: Current position dict (or None)
        ml_consensus: XGBoost prediction dict (or None)
        counter_trade_result: Output from build_counter_trade_risk() for this coin
        reversal_result: Output from build_reversal_threat() for this coin

    Returns:
        State vector dict with labels + numerical anchors
    """
    # Efficiency ratio for choppy detection
    efficiency_ratio = (
        indicators_3m.get("efficiency_ratio", 0.5)
        if indicators_3m and "error" not in indicators_3m
        else 0.5
    )

    # Volume ratio (numerical anchor for AI autonomy)
    vol_ratio = None
    if indicators_3m and "error" not in indicators_3m:
        if "volume_ratio" in indicators_3m:
            vol_ratio = _sv_fmt(indicators_3m["volume_ratio"])
        else:
            vol = indicators_3m.get("volume", 0)
            avg_vol = indicators_3m.get("avg_volume", 1)
            if avg_vol and avg_vol > 0:
                vol_ratio = _sv_fmt((vol or 0) / avg_vol)

    # Defaults for risk inputs
    ct = counter_trade_result or {"risk_level": "VERY_HIGH_RISK", "alignment_strength": "NONE"}
    rv = reversal_result or {"strength": "NONE"}

    state = {
        "coin": coin,
        # ML Consensus — untouched, AI's statistical tie-breaker
        "ml_consensus": ml_consensus,
        # Market Context — regime + environment labels
        "market_context": {
            "regime": market_regime,
            "efficiency_ratio": _sv_fmt(efficiency_ratio),
            "volatility_state": _sv_volatility_state(indicators_htf),
            "price_location": _sv_price_location(indicators_15m),
        },
        # Technical Summary — processed labels for fast orientation
        "technical_summary": {
            "trend_alignment": _sv_trend_alignment(indicators_htf, indicators_15m, indicators_3m),
            "momentum": _sv_momentum(indicators_15m),
            "volume_ratio": vol_ratio,
            "volume_support": _sv_volume_label(indicators_3m),
            "structure_15m": _sv_structure(indicators_15m),
        },
        # Key Levels — numerical anchors for independent AI reasoning
        "key_levels": {
            "price": _sv_fmt(indicators_htf.get("current_price") if indicators_htf else None),
            "ema20_htf": _sv_fmt(indicators_htf.get("ema_20") if indicators_htf else None),
            "rsi_15m": _sv_fmt(indicators_15m.get("rsi_14") if indicators_15m else None),
            "atr_htf": _sv_fmt(indicators_htf.get("atr_14") if indicators_htf else None),
        },
        # Risk Profile — pre-computed risk labels
        "risk_profile": {
            "counter_trade_risk": ct.get("risk_level", "VERY_HIGH_RISK"),
            "alignment_strength": ct.get("alignment_strength", "NONE"),
            "reversal_threat": rv.get("strength", "NONE"),
        },
        # Sentiment — funding rate & open interest
        "sentiment": {
            "funding_rate": _sv_fmt(sentiment.get("funding_rate")) if sentiment else None,
            "open_interest": _sv_fmt(sentiment.get("open_interest")) if sentiment else None,
        },
        # Position — compact position data (or null)
        "position": _sv_build_position(position) if position else None,
    }

    return state


def build_metadata_json(
    minutes_running: int,
    current_time: datetime,
    invocation_count: int,
) -> dict[str, Any]:
    """Build metadata JSON section."""
    return {
        "minutes_running": minutes_running,
        "current_time": current_time.isoformat()
        if isinstance(current_time, datetime)
        else str(current_time),
        "invocation_count": invocation_count,
    }


def build_counter_trade_json(
    counter_trade_analysis: str,
    all_indicators: dict[str, dict[str, dict[str, Any]]],
    available_coins: list[str],
    htf_interval: str,
    market_data=None,  # NEW: market_data parameter for Funding Rate
) -> list[dict[str, Any]]:
    """
    Build counter-trade analysis JSON from text analysis or indicators.

    Args:
        counter_trade_analysis: Text analysis (legacy format)
        all_indicators: Pre-fetched indicators dict
        available_coins: List of coins to analyze
        htf_interval: Higher timeframe interval (e.g., '1h')
        market_data: RealMarketData instance for funding rate (optional)

    Returns:
        Dict keyed by coin: {risk_level, alignment_strength, conditions_met}
    """
    analysis_list = {}

    for coin in available_coins:
        try:
            indicators_3m = all_indicators.get(coin, {}).get("3m", {})
            indicators_15m = all_indicators.get(coin, {}).get("15m", {})
            indicators_htf = all_indicators.get(coin, {}).get(htf_interval, {})

            if "error" in indicators_3m or "error" in indicators_htf:
                continue

            has_15m = indicators_15m and "error" not in indicators_15m

            # Extract key indicators
            price_htf = format_number_for_json(indicators_htf.get("current_price"))
            ema20_htf = format_number_for_json(indicators_htf.get("ema_20"))
            price_3m = format_number_for_json(indicators_3m.get("current_price"))
            ema20_3m = format_number_for_json(indicators_3m.get("ema_20"))

            if price_htf is None or ema20_htf is None or price_3m is None or ema20_3m is None:
                continue

            rsi_3m = format_number_for_json(indicators_3m.get("rsi_14", 50))
            volume_3m = format_number_for_json(indicators_3m.get("volume", 0))
            avg_volume_3m = format_number_for_json(indicators_3m.get("avg_volume", 1))
            macd_3m = format_number_for_json(indicators_3m.get("macd", 0))
            macd_signal_3m = format_number_for_json(indicators_3m.get("macd_signal", 0))

            # Extract 15m indicators (if available)
            price_15m = None
            ema20_15m = None
            trend_15m = None
            if has_15m:
                price_15m = format_number_for_json(indicators_15m.get("current_price"))
                ema20_15m = format_number_for_json(indicators_15m.get("ema_20"))
                if price_15m is not None and ema20_15m is not None:
                    trend_15m = "BULLISH" if price_15m > ema20_15m else "BEARISH"

            # Determine trend directions
            trend_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"
            trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"

            # Determine alignment strength for counter-trend
            # STRONG: 15m + 3m both align against 1h
            # MEDIUM: 15m OR 3m align against 1h
            # NONE: 15m AND 3m both follow 1h (no counter-trend)
            alignment_strength = "NONE"  # Default to NONE (not Python None)
            if trend_15m and trend_3m:
                # Counter-trend: 1h trend vs 3m/15m trend
                if trend_htf == "BULLISH":
                    # Counter-trend SHORT: 15m and 3m should be BEARISH
                    if trend_15m == "BEARISH" and trend_3m == "BEARISH":
                        alignment_strength = "STRONG"  # 15m+3m both BEARISH (against 1h BULLISH)
                    elif trend_15m == "BEARISH" or trend_3m == "BEARISH":
                        alignment_strength = "MEDIUM"  # 15m OR 3m BEARISH
                elif trend_htf == "BEARISH":
                    # Counter-trend LONG: 15m and 3m should be BULLISH
                    if trend_15m == "BULLISH" and trend_3m == "BULLISH":
                        alignment_strength = "STRONG"  # 15m+3m both BULLISH (against 1h BEARISH)
                    elif trend_15m == "BULLISH" or trend_3m == "BULLISH":
                        alignment_strength = "MEDIUM"  # 15m OR 3m BULLISH

            # Evaluate 5 conditions
            # Condition 1: Funding Rate Extreme (NEW - timeframe independent)
            # Negative funding = too many shorts = LONG counter-trend favored
            # Positive funding = too many longs = SHORT counter-trend favored
            condition_1 = False
            if market_data:
                try:
                    funding_rate = market_data.get_funding_rate(coin)
                    if funding_rate is not None:
                        # BEARISH trend + negative funding = LONG counter-trend favored
                        # BULLISH trend + positive funding = SHORT counter-trend favored
                        if (trend_htf == "BEARISH" and funding_rate < -0.0003) or (
                            trend_htf == "BULLISH" and funding_rate > 0.0003
                        ):  # -0.03%
                            condition_1 = True
                except Exception:
                    pass

            condition_2 = (
                (volume_3m or 0) / (avg_volume_3m or 1) > 1.5 if avg_volume_3m else False
            )  # 1.5x volume threshold for counter-trend
            # Condition 3: Extreme RSI (Counter-trend)
            # If Bullish trend, we want to Short -> Need Overbought (>70)
            # If Bearish trend, we want to Long -> Need Oversold (<30)
            condition_3 = (
                trend_htf == "BULLISH" and (rsi_3m or 50) > Config.RSI_OVERBOUGHT_THRESHOLD
            ) or (trend_htf == "BEARISH" and (rsi_3m or 50) < Config.RSI_OVERSOLD_THRESHOLD)
            # condition_4 (EMA Proximity) REMOVED - not a meaningful counter-trend signal
            # Condition 5: MACD divergence (Counter-trend)
            # If Bullish trend, we want to Short -> Need Bearish MACD (MACD < Signal)
            # If Bearish trend, we want to Long -> Need Bullish MACD (MACD > Signal)
            condition_5 = (trend_htf == "BULLISH" and (macd_3m or 0) < (macd_signal_3m or 0)) or (
                trend_htf == "BEARISH" and (macd_3m or 0) > (macd_signal_3m or 0)
            )

            # Condition 6: Zone + Weakening (Counter-trend favorable setup)
            # LOWER_10 + WEAKENING (BEARISH trend) = favorable for LONG counter-trade
            # UPPER_10 + WEAKENING (BULLISH trend) = favorable for SHORT counter-trade
            momentum_15m = indicators_15m.get("momentum", None) if has_15m else None
            price_location_15m = indicators_15m.get("price_location", None) if has_15m else None
            zone_15m = (
                price_location_15m.get("zone", "MIDDLE")
                if isinstance(price_location_15m, dict)
                else price_location_15m
            )
            condition_6 = False
            if momentum_15m == "WEAKENING":
                if trend_htf == "BEARISH" and zone_15m == "LOWER_10":
                    condition_6 = True  # Favorable for LONG counter-trade
                elif trend_htf == "BULLISH" and zone_15m == "UPPER_10":
                    condition_6 = True  # Favorable for SHORT counter-trade

            # ==================== NEW CONDITIONS (v6.0) ====================
            # Condition 7: VWAP Reversion (Price extended from fair value)
            # If BULLISH trend and price > VWAP by 2% -> favorable for SHORT counter-trade
            # If BEARISH trend and price < VWAP by 2% -> favorable for LONG counter-trade
            condition_7 = False
            if has_15m and "vwap" in indicators_15m:
                vwap = indicators_15m.get("vwap")
                price = indicators_15m.get("current_price")
                if vwap and price and vwap > 0:
                    vwap_dist = ((price - vwap) / vwap) * 100
                    if trend_htf == "BULLISH" and vwap_dist > 2.0:
                        condition_7 = True  # Price expensive -> SHORT favorable
                    elif trend_htf == "BEARISH" and vwap_dist < -2.0:
                        condition_7 = True  # Price cheap -> LONG favorable

            # Condition 8: BB Extreme (Price at band extremes)
            # If BULLISH trend and price at upper band -> favorable for SHORT counter-trade
            # If BEARISH trend and price at lower band -> favorable for LONG counter-trade
            condition_8 = False
            if has_15m and "bb_signal" in indicators_15m:
                bb_signal = indicators_15m.get("bb_signal")
                if trend_htf == "BULLISH" and bb_signal == "OVERBOUGHT":
                    condition_8 = True  # Price at upper band -> SHORT favorable
                elif trend_htf == "BEARISH" and bb_signal == "OVERSOLD":
                    condition_8 = True  # Price at lower band -> LONG favorable

            # Condition 9: OBV Divergence (Volume not confirming price)
            # If BULLISH trend but OBV BEARISH divergence -> favorable for SHORT counter-trade
            # If BEARISH trend but OBV BULLISH divergence -> favorable for LONG counter-trade
            condition_9 = False
            if has_15m and "obv_divergence" in indicators_15m:
                obv_div = indicators_15m.get("obv_divergence")
                if trend_htf == "BULLISH" and obv_div == "BEARISH":
                    condition_9 = True  # Volume not confirming rally -> SHORT favorable
                elif trend_htf == "BEARISH" and obv_div == "BULLISH":
                    condition_9 = True  # Volume accumulating in downtrend -> LONG favorable
            # ==================== END NEW CONDITIONS ====================

            total_met = sum(
                [
                    condition_1,
                    condition_2,
                    condition_3,
                    condition_5,
                    condition_6,
                    condition_7,
                    condition_8,
                    condition_9,
                ],
            )  # 8 conditions (EMA removed)

            # Determine risk level (Updated Logic - User Request)
            # Counter-trade risk assessment based on alignment and conditions met
            if alignment_strength == "STRONG" and total_met >= 4:
                risk_level = "LOW_RISK"  # STRONG + 4 or more conditions
            elif alignment_strength == "STRONG" and total_met >= 3:
                risk_level = "MEDIUM_RISK"  # STRONG + 3 conditions
            elif alignment_strength == "MEDIUM" and total_met >= 5:
                risk_level = "LOW_RISK"  # MEDIUM + 5 or more conditions
            elif alignment_strength == "MEDIUM" and total_met >= 4:
                risk_level = "MEDIUM_RISK"  # MEDIUM + 4 conditions
            elif alignment_strength == "MEDIUM":
                risk_level = "HIGH_RISK"  # MEDIUM + less than 4 conditions
            elif alignment_strength == "NONE" and total_met >= 8:
                risk_level = "LOW_RISK"  # NONE + 8 conditions (All)
            elif alignment_strength == "NONE" and total_met >= 7:
                risk_level = "MEDIUM_RISK"  # NONE + 7 conditions
            else:
                risk_level = "VERY_HIGH_RISK"  # No alignment and < 7 conditions

            # NOTE: Zone + Weakening is now Condition 6 (calculated above)
            # No longer modifies risk level - it's counted as a condition instead

            # Store compact result keyed by coin
            analysis_list[coin] = {
                "risk_level": risk_level,
                "alignment_strength": alignment_strength,
                "conditions_met": total_met,
            }

        except Exception:
            # Skip coins with errors
            continue

    return analysis_list


def build_trend_reversal_json(
    trend_reversal_analysis: dict[str, Any],
    portfolio_positions: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Build trend reversal detection data from performance_monitor output.

    Args:
        trend_reversal_analysis: Output from detect_trend_reversal_for_all_coins()
        portfolio_positions: Current portfolio positions

    Returns:
        Dict keyed by coin: {strength} for State Vector integration
    """
    reversal_dict = {}

    if not trend_reversal_analysis or "error" in trend_reversal_analysis:
        return reversal_dict

    for coin, analysis in trend_reversal_analysis.items():
        if coin == "error":
            continue

        strength = analysis.get("strength", "NONE")
        reversal_dict[coin] = {"strength": strength}

    return reversal_dict


def build_cooldown_status_json(
    directional_cooldowns: dict[str, int],
    coin_cooldowns: dict[str, int],
    counter_trend_cooldown: int,
    relaxed_countertrend_cycles: int,
) -> dict[str, Any]:
    """Build cooldown status JSON."""
    return {
        "directional_cooldowns": {k: v for k, v in directional_cooldowns.items()},
        "coin_cooldowns": {k: v for k, v in coin_cooldowns.items()},
        "counter_trend_cooldown": counter_trend_cooldown,
        "relaxed_countertrend_cycles": relaxed_countertrend_cycles,
    }


def build_position_slot_json(
    portfolio_positions: dict[str, Any],
    max_positions: int,
    same_direction_limit: int = None,
) -> dict[str, Any]:
    """Build position slot status JSON."""
    from config.config import Config

    total_open = len(portfolio_positions)
    # Fix: Check direction without default value to avoid logic error
    long_slots = sum(
        1 for p in portfolio_positions.values() if str(p.get("direction", "")).lower() == "long"
    )
    short_slots = sum(
        1 for p in portfolio_positions.values() if str(p.get("direction", "")).lower() == "short"
    )

    # Get same direction limit from config if not provided
    if same_direction_limit is None:
        same_direction_limit = Config.SAME_DIRECTION_LIMIT

    long_slots_available = same_direction_limit - long_slots
    short_slots_available = same_direction_limit - short_slots

    # Find weakest position
    weakest_position = None
    if portfolio_positions:
        weakest = min(
            portfolio_positions.items(),
            key=lambda x: x[1].get("unrealized_pnl", float("inf")),
        )
        weakest_position = {
            "coin": weakest[0],
            "unrealized_pnl": format_number_for_json(weakest[1].get("unrealized_pnl", 0)),
            "confidence": format_number_for_json(weakest[1].get("confidence", 0)),
        }

    # Check for forced rotation/constraint conditions
    constraint_mode = "NORMAL"
    if long_slots_available <= 0 and short_slots_available > 0:
        constraint_mode = "LONG_FULL_SHORT_AVAILABLE"
    elif short_slots_available <= 0 and long_slots_available > 0:
        constraint_mode = "SHORT_FULL_LONG_AVAILABLE"
    elif long_slots_available <= 0 and short_slots_available <= 0:
        constraint_mode = "ALL_SLOTS_FULL"

    return {
        "total_open": total_open,
        "max_positions": max_positions,
        "long_slots_used": long_slots,
        "short_slots_used": short_slots,
        "same_direction_limit": same_direction_limit,
        "long_slots_available": long_slots_available,
        "short_slots_available": short_slots_available,
        "available_slots": max_positions - total_open,
        "weakest_position": weakest_position,
        "constraint_mode": constraint_mode,
        "constraint_instruction": "If a direction is FULL, do NOT force trades in the other direction unless they are LOW_RISK or MEDIUM_RISK (High Confidence alone is NOT enough).",
    }


def build_portfolio_json(portfolio: Any) -> dict[str, Any]:
    """
    Build portfolio JSON.

    Args:
        portfolio: Portfolio object with attributes like total_return, current_balance, etc.

    Returns:
        Portfolio JSON object
    """
    positions_list = []
    if hasattr(portfolio, "positions") and portfolio.positions:
        for coin, pos in portfolio.positions.items():
            positions_list.append(
                {
                    "symbol": coin,
                    "direction": pos.get("direction", "long"),  # [INFO] Added
                    "quantity": format_number_for_json(pos.get("quantity", 0)),
                    "entry_price": format_number_for_json(pos.get("entry_price", 0)),
                    "current_price": format_number_for_json(pos.get("current_price", 0)),
                    "unrealized_pnl": format_number_for_json(pos.get("unrealized_pnl", 0)),
                    "leverage": pos.get("leverage", 1),
                    "confidence": format_number_for_json(pos.get("confidence", 0.5)),
                },
            )

    return {
        "total_return_pct": format_number_for_json(
            portfolio.total_return if hasattr(portfolio, "total_return") else 0,
        ),
        "available_cash": format_number_for_json(
            portfolio.current_balance if hasattr(portfolio, "current_balance") else 0,
        ),
        "account_value": format_number_for_json(
            portfolio.total_value if hasattr(portfolio, "total_value") else 0,
        ),
        "sharpe_ratio": format_number_for_json(
            portfolio.sharpe_ratio if hasattr(portfolio, "sharpe_ratio") else None,
        ),
        "positions": positions_list,
    }


def build_risk_status_json(portfolio: Any, max_positions: int = 5) -> dict[str, Any]:
    """Build risk status JSON."""
    current_positions_count = len(portfolio.positions) if hasattr(portfolio, "positions") else 0
    total_margin_used = sum(
        p.get("margin_usd", 0)
        for p in (portfolio.positions.values() if hasattr(portfolio, "positions") else [])
    )
    available_cash = portfolio.current_balance if hasattr(portfolio, "current_balance") else 0

    return {
        "current_positions_count": current_positions_count,
        "total_margin_used": format_number_for_json(total_margin_used),
        "available_cash": format_number_for_json(available_cash),
        "trading_limits": {
            "min_position": Config.MIN_POSITION_MARGIN_USD,
            "max_positions": max_positions,
            "available_cash_protection": format_number_for_json(available_cash * 0.10),
            "position_sizing_pct": 40.0,  # Up to 40% of available cash
        },
    }


def build_historical_context_json(trading_context: dict[str, Any]) -> dict[str, Any]:
    """Build historical context JSON."""
    return {
        "total_cycles_analyzed": trading_context.get("total_cycles_analyzed", 0),
        "market_behavior": trading_context.get("market_behavior", "Unknown"),
        "recent_decisions": trading_context.get("recent_decisions", []),
        "performance_trend": trading_context.get("performance_trend", "Unknown"),
    }


def build_directional_bias_json(bias_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Build directional bias metrics JSON (Last 20 trades snapshot).

    Args:
        bias_metrics: Output from get_directional_bias_metrics()

    Returns:
        Directional bias JSON object
    """
    result = {}
    for side in ("long", "short"):
        stats = bias_metrics.get(side, {})
        result[side] = {
            "net_pnl": format_number_for_json(stats.get("net_pnl", 0.0)),
            "trades": stats.get("trades", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "profitability_index": format_number_for_json(stats.get("profitability_index", 0.0)),
            "rolling_avg": format_number_for_json(stats.get("rolling_avg", 0.0)),
            "consecutive_losses": stats.get("consecutive_losses", 0),
            "consecutive_wins": stats.get("consecutive_wins", 0),
            "caution_active": stats.get("caution_active", False),
        }
    return result


def build_trend_flip_guard_json(
    trend_flip_summary: list[str],
    trend_flip_cooldown: int,
    trend_flip_history_window: int,
) -> dict[str, Any]:
    """
    Build trend flip guard JSON.

    Args:
        trend_flip_summary: Output from get_recent_trend_flip_summary()
        trend_flip_cooldown: Cooldown period in cycles
        trend_flip_history_window: History window in cycles

    Returns:
        Trend flip guard JSON object
    """
    return {
        "cooldown_cycles": trend_flip_cooldown,
        "history_window_cycles": trend_flip_history_window,
        "recent_flips": trend_flip_summary or [],
        "flip_count": len(trend_flip_summary) if trend_flip_summary else 0,
    }
