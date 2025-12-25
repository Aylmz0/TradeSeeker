"""
JSON builder functions for AI prompt generation.
Converts data structures to JSON format for hybrid prompt.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.ai.prompt_json_utils import (
    safe_json_dumps,
    create_json_section,
    format_number_for_json
)
from src.ai.prompt_json_schemas import JSON_PROMPT_VERSION
from config.config import Config
import math


def build_metadata_json(
    minutes_running: int,
    current_time: datetime,
    invocation_count: int
) -> Dict[str, Any]:
    """Build metadata JSON section."""
    return {
        "minutes_running": minutes_running,
        "current_time": current_time.isoformat() if isinstance(current_time, datetime) else str(current_time),
        "invocation_count": invocation_count
    }


def build_counter_trade_json(
    counter_trade_analysis: str,
    all_indicators: Dict[str, Dict[str, Dict[str, Any]]],
    available_coins: List[str],
    htf_interval: str,
    market_data = None  # YENİ: Funding Rate için market_data parametresi
) -> List[Dict[str, Any]]:
    """
    Build counter-trade analysis JSON from text analysis or indicators.
    
    Args:
        counter_trade_analysis: Text analysis (legacy format)
        all_indicators: Pre-fetched indicators dict
        available_coins: List of coins to analyze
        htf_interval: Higher timeframe interval (e.g., '1h')
        market_data: RealMarketData instance for funding rate (optional)
    
    Returns:
        List of counter-trade analysis objects with 15m+3m alignment information
    """
    analysis_list = []
    
    for coin in available_coins:
        try:
            indicators_3m = all_indicators.get(coin, {}).get('3m', {})
            indicators_15m = all_indicators.get(coin, {}).get('15m', {})
            indicators_htf = all_indicators.get(coin, {}).get(htf_interval, {})
            
            if 'error' in indicators_3m or 'error' in indicators_htf:
                continue
            
            has_15m = indicators_15m and 'error' not in indicators_15m
            
            # Extract key indicators
            price_htf = format_number_for_json(indicators_htf.get('current_price'))
            ema20_htf = format_number_for_json(indicators_htf.get('ema_20'))
            price_3m = format_number_for_json(indicators_3m.get('current_price'))
            ema20_3m = format_number_for_json(indicators_3m.get('ema_20'))
            
            if price_htf is None or ema20_htf is None or price_3m is None or ema20_3m is None:
                continue
            
            rsi_3m = format_number_for_json(indicators_3m.get('rsi_14', 50))
            volume_3m = format_number_for_json(indicators_3m.get('volume', 0))
            avg_volume_3m = format_number_for_json(indicators_3m.get('avg_volume', 1))
            macd_3m = format_number_for_json(indicators_3m.get('macd', 0))
            macd_signal_3m = format_number_for_json(indicators_3m.get('macd_signal', 0))
            
            # Extract 15m indicators (if available)
            price_15m = None
            ema20_15m = None
            trend_15m = None
            if has_15m:
                price_15m = format_number_for_json(indicators_15m.get('current_price'))
                ema20_15m = format_number_for_json(indicators_15m.get('ema_20'))
                if price_15m is not None and ema20_15m is not None:
                    trend_15m = "BULLISH" if price_15m > ema20_15m else "BEARISH"
            
            
            # Determine trend directions
            trend_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"
            trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
            
            # Determine alignment strength for counter-trend
            # STRONG: 15m + 3m both align against 1h
            # MEDIUM: 15m VEYA 3m align against 1h
            # NONE: 15m AND 3m both follow 1h (no counter-trend)
            alignment_strength = "NONE"  # Default to NONE (not Python None)
            if trend_15m and trend_3m:
                # Counter-trend: 1h trend vs 3m/15m trend
                if trend_htf == "BULLISH":
                    # Counter-trend SHORT: 15m and 3m should be BEARISH
                    if trend_15m == "BEARISH" and trend_3m == "BEARISH":
                        alignment_strength = "STRONG"  # 15m+3m both BEARISH (against 1h BULLISH)
                    elif trend_15m == "BEARISH" or trend_3m == "BEARISH":
                        alignment_strength = "MEDIUM"  # 15m VEYA 3m BEARISH
                elif trend_htf == "BEARISH":
                    # Counter-trend LONG: 15m and 3m should be BULLISH
                    if trend_15m == "BULLISH" and trend_3m == "BULLISH":
                        alignment_strength = "STRONG"  # 15m+3m both BULLISH (against 1h BEARISH)
                    elif trend_15m == "BULLISH" or trend_3m == "BULLISH":
                        alignment_strength = "MEDIUM"  # 15m VEYA 3m BULLISH
            
            # Evaluate 5 conditions
            # Condition 1: Funding Rate Extreme (YENİ - zaman diliminden bağımsız)
            # Negative funding = too many shorts = LONG counter-trend favored
            # Positive funding = too many longs = SHORT counter-trend favored
            condition_1 = False
            if market_data:
                try:
                    funding_rate = market_data.get_funding_rate(coin)
                    if funding_rate is not None:
                        # BEARISH trend + negative funding = LONG counter-trend favored
                        # BULLISH trend + positive funding = SHORT counter-trend favored
                        if trend_htf == "BEARISH" and funding_rate < -0.0003:  # -0.03%
                            condition_1 = True
                        elif trend_htf == "BULLISH" and funding_rate > 0.0003:  # +0.03%
                            condition_1 = True
                except Exception:
                    pass
            
            condition_2 = (volume_3m or 0) / (avg_volume_3m or 1) > Config.VOLUME_QUALITY_THRESHOLDS['good'] if avg_volume_3m else False
            # Condition 3: Extreme RSI (Counter-trend)
            # If Bullish trend, we want to Short -> Need Overbought (>70)
            # If Bearish trend, we want to Long -> Need Oversold (<30)
            condition_3 = (trend_htf == "BULLISH" and (rsi_3m or 50) > Config.RSI_OVERBOUGHT_THRESHOLD) or (trend_htf == "BEARISH" and (rsi_3m or 50) < Config.RSI_OVERSOLD_THRESHOLD)
            condition_4 = abs((price_3m or 0) - (ema20_3m or 0)) / (price_3m or 1) * 100 < 1.0 if price_3m and ema20_3m else False
            # Condition 5: MACD divergence (Counter-trend)
            # If Bullish trend, we want to Short -> Need Bearish MACD (MACD < Signal)
            # If Bearish trend, we want to Long -> Need Bullish MACD (MACD > Signal)
            condition_5 = (trend_htf == "BULLISH" and (macd_3m or 0) < (macd_signal_3m or 0)) or (trend_htf == "BEARISH" and (macd_3m or 0) > (macd_signal_3m or 0))
            
            # Condition 6: Zone + Weakening (Counter-trend favorable setup)
            # LOWER_10 + WEAKENING (BEARISH trend) = favorable for LONG counter-trade
            # UPPER_10 + WEAKENING (BULLISH trend) = favorable for SHORT counter-trade
            momentum_15m = indicators_15m.get('momentum', None) if has_15m else None
            price_location_15m = indicators_15m.get('price_location', None) if has_15m else None
            zone_15m = price_location_15m.get('zone', 'MIDDLE') if isinstance(price_location_15m, dict) else price_location_15m
            condition_6 = False
            if momentum_15m == "WEAKENING":
                if trend_htf == "BEARISH" and zone_15m == "LOWER_10":
                    condition_6 = True  # Favorable for LONG counter-trade
                elif trend_htf == "BULLISH" and zone_15m == "UPPER_10":
                    condition_6 = True  # Favorable for SHORT counter-trade
            
            total_met = sum([condition_1, condition_2, condition_3, condition_4, condition_5, condition_6])
            
            # Determine risk level (Updated Logic - User Request Dec 10)
            # STRONG alignment = 15m+3m both counter
            # MEDIUM alignment = 15m OR 3m counter (one of them)
            if alignment_strength == "STRONG" and total_met >= 3:
                risk_level = "LOW_RISK"  # STRONG + 3 or more conditions
            elif alignment_strength == "STRONG" and total_met >= 1:
                risk_level = "MEDIUM_RISK"  # STRONG + 1-2 conditions
            elif alignment_strength == "MEDIUM" and total_met >= 4:
                risk_level = "LOW_RISK"  # MEDIUM + 4 or more conditions
            elif alignment_strength == "MEDIUM" and total_met == 3:
                risk_level = "MEDIUM_RISK"  # MEDIUM + exactly 3 conditions
            elif alignment_strength == "MEDIUM":
                risk_level = "HIGH_RISK"  # MEDIUM + less than 3 conditions
            else:
                risk_level = "VERY_HIGH_RISK"  # No alignment (15m AND 3m both follow HTF trend)
            
            # NOTE: Zone + Weakening is now Condition 6 (calculated above)
            # No longer modifies risk level - it's counted as a condition instead
            
            analysis_list.append({
                "coin": coin,
                "htf_trend": trend_htf,
                "15m_trend": trend_15m,
                "3m_trend": trend_3m,
                "alignment_strength": alignment_strength,
                "conditions": {
                    "total_met": total_met
                },
                "risk_level": risk_level,
                "volume_ratio": format_number_for_json((volume_3m or 0) / avg_volume_3m) if avg_volume_3m and avg_volume_3m > 0 else 0.0,
                "volume_strength": "STRONG" if (avg_volume_3m and avg_volume_3m > 0 and ((volume_3m or 0) / avg_volume_3m) > Config.VOLUME_QUALITY_THRESHOLDS['good']) else "WEAK" if (avg_volume_3m and avg_volume_3m > 0 and ((volume_3m or 0) / avg_volume_3m) < Config.VOLUME_QUALITY_THRESHOLDS['poor']) else "NORMAL",
                "rsi_3m": rsi_3m
            })
        
        except Exception as e:
            # Skip coins with errors
            continue
    
    return analysis_list


def build_trend_reversal_json(
    trend_reversal_analysis: Dict[str, Any],
    portfolio_positions: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Build trend reversal detection JSON from performance_monitor output.
    
    Args:
        trend_reversal_analysis: Output from detect_trend_reversal_for_all_coins()
        portfolio_positions: Current portfolio positions
    
    Returns:
        List of trend reversal objects
    """
    reversal_list = []
    
    if not trend_reversal_analysis or 'error' in trend_reversal_analysis:
        return reversal_list
    
    for coin, analysis in trend_reversal_analysis.items():
        if coin == 'error':
            continue
        
        has_position = coin in portfolio_positions
        position = portfolio_positions.get(coin, {})
        position_direction = position.get('direction', None)
        
        # Calculate position duration if available
        position_duration_minutes = None
        if has_position and 'entry_time' in position:
            try:
                entry_time = datetime.fromisoformat(position['entry_time']) if isinstance(position['entry_time'], str) else position['entry_time']
                if isinstance(entry_time, datetime):
                    position_duration_minutes = (datetime.now() - entry_time).total_seconds() / 60
            except:
                pass
        
        # Extract reversal signals
        loss_risk_signals = analysis.get('loss_risk_signals', [])
        signal_strength = analysis.get('signal_strength', 'NO_LOSS_RISK')
        
        # Get trend directions for reversal detection
        trend_htf = analysis.get('current_trend_4h', analysis.get('current_trend_1h', 'UNKNOWN'))
        trend_3m = analysis.get('current_trend_3m', 'UNKNOWN')
        trend_15m = analysis.get('current_trend_15m', None)  # May not be available
        
        # Check if this is a counter-trend position
        # Counter-trend: position direction is OPPOSITE to HTF trend at entry
        is_counter_trend = False
        if has_position and position_direction and position:
            trend_alignment = position.get('trend_alignment', 'trend_following')
            trend_context = position.get('trend_context', {})
            if trend_alignment == 'counter_trend' or trend_context.get('alignment') == 'counter_trend':
                is_counter_trend = True
        
        # Detect reversal against position direction
        htf_reversal = False
        fifteen_m_reversal = False
        three_m_reversal = len(loss_risk_signals) > 0
        
        if has_position and position_direction:
            # HTF reversal: HTF trend opposes position
            if position_direction == 'long' and trend_htf == 'BEARISH':
                htf_reversal = True
            elif position_direction == 'short' and trend_htf == 'BULLISH':
                htf_reversal = True
            
            # 15m reversal: 15m trend opposes position (if available)
            if trend_15m:
                if position_direction == 'long' and trend_15m == 'BEARISH':
                    fifteen_m_reversal = True
                elif position_direction == 'short' and trend_15m == 'BULLISH':
                    fifteen_m_reversal = True
        
        # Map reversal strength based on 15m and 3m reversal detection (15m > 3m priority)
        # IMPORTANT: For counter-trend positions, htf_reversal is expected (was already against at entry)
        # So we only consider 15m and 3m for strength calculation
        if is_counter_trend:
            # Counter-trend: ignore htf_reversal, only look at 15m/3m
            if fifteen_m_reversal and three_m_reversal:
                strength = "STRONG"  # Both 15m and 3m reversed - trend flip happening
            elif fifteen_m_reversal:
                strength = "MEDIUM"  # 15m reversed against counter-trend position
            elif three_m_reversal:
                strength = "INFORMATIONAL"  # Only 3m, may be noise
            else:
                strength = "NONE"  # htf_reversal alone doesn't count for counter-trend
        else:
            # Trend-following: htf_reversal matters
            if fifteen_m_reversal and three_m_reversal:
                strength = "STRONG"  # Both 15m and 3m show reversal
            elif fifteen_m_reversal:
                strength = "MEDIUM"  # Only 15m shows reversal (more reliable)
            elif three_m_reversal:
                strength = "INFORMATIONAL"  # Only 3m shows reversal (may be noise)
            else:
                strength = "NONE"
        
        reversal_list.append({
            "coin": coin,
            "has_position": has_position,
            "position_direction": position_direction,
            "position_duration_minutes": format_number_for_json(position_duration_minutes),
            "is_counter_trend": is_counter_trend,  # Added for AI awareness
            "reversal_signals": {
                "htf_reversal": htf_reversal,  # Still reported, but interpreted differently for counter-trend
                "15m_reversal": fifteen_m_reversal,
                "3m_reversal": three_m_reversal,
                "strength": strength
            },
            "loss_risk_signal": signal_strength,
            "current_trend_htf": trend_htf,
            "current_trend_3m": trend_3m
        })
    
    return reversal_list


def build_enhanced_context_json(
    enhanced_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build enhanced context JSON from enhanced_context_provider output.
    
    Args:
        enhanced_context: Output from generate_enhanced_context()
    
    Returns:
        Enhanced context JSON object
    """
    if not enhanced_context or 'error' in enhanced_context:
        return {}
    
    position_context = enhanced_context.get('position_context', {})
    market_regime = enhanced_context.get('market_regime', {})
    performance_insights = enhanced_context.get('performance_insights', {})
    directional_feedback = enhanced_context.get('directional_feedback', {})
    risk_context = enhanced_context.get('risk_context', {})
    suggestions = enhanced_context.get('suggestions', [])
    
    return {
        "position_context": {
            "total_positions": position_context.get('total_positions', 0),
            "long_positions": position_context.get('long_positions', 0),
            "short_positions": position_context.get('short_positions', 0),
            "total_margin_used": format_number_for_json(position_context.get('total_margin_used', 0)),
            "total_unrealized_pnl": format_number_for_json(position_context.get('total_unrealized_pnl', 0))
        },
        "market_regime": {
            "global_regime": market_regime.get('current_regime', 'UNKNOWN'),
            "bullish_count": market_regime.get('bullish_count', 0),
            "bearish_count": market_regime.get('bearish_count', 0),
            "neutral_count": market_regime.get('neutral_count', 0),
            "regime_strength": format_number_for_json(market_regime.get('regime_strength', 0))
        },
        "performance_insights": {
            "total_return": format_number_for_json(performance_insights.get('total_return', 0)),
            "sharpe_ratio": format_number_for_json(performance_insights.get('sharpe_ratio')),
            "profitability_index": format_number_for_json(performance_insights.get('profitability_index'))
        },
        "directional_feedback": {
            "long_performance": {
                "net_pnl": format_number_for_json(directional_feedback.get('long', {}).get('total_pnl', 0)),
                "trades": directional_feedback.get('long', {}).get('trades', 0),
                "profitability_index": format_number_for_json(directional_feedback.get('long', {}).get('profitability_index', 0))
            },
            "short_performance": {
                "net_pnl": format_number_for_json(directional_feedback.get('short', {}).get('total_pnl', 0)),
                "trades": directional_feedback.get('short', {}).get('trades', 0),
                "profitability_index": format_number_for_json(directional_feedback.get('short', {}).get('profitability_index', 0))
            }
        },
        "risk_context": {
            "current_risk_usd": format_number_for_json(risk_context.get('total_risk_usd', 0)),
            "max_risk_allowed": format_number_for_json(risk_context.get('max_risk_allowed', 0)),
            "risk_utilization_pct": format_number_for_json(risk_context.get('risk_utilization_pct', 0))
        },
        "suggestions": suggestions if isinstance(suggestions, list) else []
    }


def build_cooldown_status_json(
    directional_cooldowns: Dict[str, int],
    coin_cooldowns: Dict[str, int],
    counter_trend_cooldown: int,
    relaxed_countertrend_cycles: int
) -> Dict[str, Any]:
    """Build cooldown status JSON."""
    return {
        "directional_cooldowns": {k: v for k, v in directional_cooldowns.items()},
        "coin_cooldowns": {k: v for k, v in coin_cooldowns.items()},
        "counter_trend_cooldown": counter_trend_cooldown,
        "relaxed_countertrend_cycles": relaxed_countertrend_cycles
    }


def build_position_slot_json(
    portfolio_positions: Dict[str, Any],
    max_positions: int,
    same_direction_limit: int = None
) -> Dict[str, Any]:
    """Build position slot status JSON."""
    from config.config import Config
    
    total_open = len(portfolio_positions)
    # Fix: Check direction without default value to avoid logic error
    long_slots = sum(1 for p in portfolio_positions.values() if p.get('direction') == 'long')
    short_slots = sum(1 for p in portfolio_positions.values() if p.get('direction') == 'short')
    
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
            key=lambda x: x[1].get('unrealized_pnl', float('inf'))
        )
        weakest_position = {
            "coin": weakest[0],
            "unrealized_pnl": format_number_for_json(weakest[1].get('unrealized_pnl', 0)),
            "confidence": format_number_for_json(weakest[1].get('confidence', 0))
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
        "constraint_instruction": "If a direction is FULL, do NOT force trades in the other direction unless they are LOW_RISK or MEDIUM_RISK (High Confidence alone is NOT enough)."
    }


def build_market_data_json(
    coin: str,
    market_regime: str,
    sentiment: Dict[str, Any],
    indicators_3m: Dict[str, Any],
    indicators_15m: Dict[str, Any],
    indicators_htf: Dict[str, Any],
    position: Optional[Dict[str, Any]] = None,
    max_series_length: int = 50
) -> Dict[str, Any]:
    """
    Build market data JSON for a single coin.
    
    Args:
        coin: Coin symbol
        market_regime: Market regime (BULLISH/BEARISH/NEUTRAL)
        sentiment: Sentiment data (OI, funding rate)
        indicators_3m: 3m indicators
        indicators_15m: 15m indicators
        indicators_htf: HTF indicators
        position: Current position (if exists)
        max_series_length: Maximum series length before compression
    
    Returns:
        Market data JSON object
    """
    def build_timeframe_data(indicators: Dict[str, Any], has_atr: bool = False, has_volume: bool = False):
        """Helper to build timeframe data structure."""
        if not indicators or 'error' in indicators:
            return {
                "current": {},
                "series": {}
            }
        
        current = {
            "price": format_number_for_json(indicators.get('current_price')),
            "ema20": format_number_for_json(indicators.get('ema_20')),
            "rsi": format_number_for_json(indicators.get('rsi_14')),
            "macd": format_number_for_json(indicators.get('macd'))
        }
        
        if has_atr:
            current["atr"] = format_number_for_json(indicators.get('atr_14'))
        if has_volume:
            vol = indicators.get('volume')
            avg_vol = indicators.get('avg_volume')
            current["volume"] = format_number_for_json(vol)
            
            # Use pre-calculated volume_ratio (based on closed candles) if available
            # This ensures consistency with 'tags' which use the closed candle ratio
            if 'volume_ratio' in indicators:
                ratio = indicators['volume_ratio']
            elif avg_vol:
                ratio = (vol or 0) / (avg_vol or 1)
            else:
                ratio = 0
                
            current["volume_ratio"] = format_number_for_json(ratio)
        if has_volume:
            vol = indicators.get('volume')
            avg_vol = indicators.get('avg_volume')
            current["volume"] = format_number_for_json(vol)
            
            # Use pre-calculated volume_ratio (based on closed candles) if available
            # This ensures consistency with 'tags' which use the closed candle ratio
            if 'volume_ratio' in indicators:
                ratio = indicators['volume_ratio']
            elif avg_vol:
                ratio = (vol or 0) / (avg_vol or 1)
            else:
                ratio = 0
                
            current["volume_ratio"] = format_number_for_json(ratio)
            current["volume_strength"] = "STRONG" if ratio > 1.5 else "WEAK" if ratio < 0.5 else "NORMAL"

        # Add Smart Sparkline Data if available
        if 'smart_sparkline' in indicators:
            current["smart_sparkline"] = indicators['smart_sparkline']
        
        # ==================== NEW INDICATORS (v5.0) ====================
        # Add new indicator data to AI prompt
        
        # ADX (Trend Strength) - Keep in prompt
        if 'adx' in indicators:
            current["adx"] = format_number_for_json(indicators.get('adx'))
            current["trend_strength_adx"] = indicators.get('trend_strength_adx', 'UNKNOWN')
        
        # NOTE: The following indicators are NOT sent to prompt anymore (v5.1 optimization)
        # They only affect confidence in backend:
        # - VWAP: affects confidence ±5%
        # - Bollinger Bands: squeeze penalty -5%
        # - OBV: divergence penalty -15%
        # - SuperTrend: alignment ±5%
        
        # ==================== END NEW INDICATORS ====================
        
        # Build series with compression if needed
        price_series = indicators.get('price_series', [])
        rsi_series = indicators.get('rsi_14_series', [])
        
        series = {}
        
        # Enforce max series length from Config
        max_len = Config.JSON_SERIES_MAX_LENGTH
        
        # Compress series if too long
        if len(price_series) > max_len:
            compressed_price = compress_series(price_series, max_length=max_len)
            series["price"] = compressed_price
        else:
            series["price"] = [format_number_for_json(p) for p in price_series]
        
        if len(rsi_series) > max_len:
            compressed_rsi = compress_series(rsi_series, max_length=max_len)
            series["rsi"] = compressed_rsi
        else:
            series["rsi"] = [format_number_for_json(r) for r in rsi_series]
        
        return {
            "current": current,
            "series": series
        }
    
    
    # Extract Efficiency Ratio from 3m indicators for choppy detection
    efficiency_ratio = indicators_3m.get('efficiency_ratio', 0.5) if indicators_3m and 'error' not in indicators_3m else 0.5
    
    # Determine market condition based on ER
    from config.config import Config
    if efficiency_ratio < Config.CHOPPY_ER_THRESHOLD:
        market_condition = "CHOPPY"
    else:
        market_condition = "TRENDING"
    
    market_data = {
        "coin": coin,
        "market_regime": market_regime,
        "efficiency_ratio": format_number_for_json(efficiency_ratio),
        "market_condition": market_condition,
        "tags": indicators_htf.get('tags') if indicators_htf else [],
        "sentiment": {
            "open_interest": format_number_for_json(sentiment.get('open_interest')),
            "funding_rate": format_number_for_json(sentiment.get('funding_rate')),
            "funding_rate_24h_avg": format_number_for_json(sentiment.get('funding_rate_24h_avg'))
        },
        "timeframes": {
            "3m": build_timeframe_data(indicators_3m, has_atr=True, has_volume=True),
            "15m": build_timeframe_data(indicators_15m),
            "htf": build_timeframe_data(indicators_htf, has_atr=True)
        }
    }
    
    # Add position if exists
    if position:
        market_data["position"] = {
            "symbol": position.get('symbol', coin),
            "direction": position.get('direction', 'long'),
            "quantity": format_number_for_json(position.get('quantity', 0)),
            "entry_price": format_number_for_json(position.get('entry_price', 0)),
            "current_price": format_number_for_json(position.get('current_price', 0)),
            "liquidation_price": format_number_for_json(position.get('liquidation_price', 0)),
            "unrealized_pnl": format_number_for_json(position.get('unrealized_pnl', 0)),
            "leverage": position.get('leverage', 1),
            "confidence": format_number_for_json(position.get('confidence', 0.5)),
            "risk_usd": position.get('risk_usd', 'N/A'),
            "notional_usd": format_number_for_json(position.get('notional_usd', 0)),
            "exit_plan": {
                "profit_target": format_number_for_json(position.get('exit_plan', {}).get('profit_target')),
                "stop_loss": format_number_for_json(position.get('exit_plan', {}).get('stop_loss')),
                "invalidation_condition": position.get('exit_plan', {}).get('invalidation_condition')
            }
        }
        # Profit erosion tracking - only send details if meaningful (not NONE)
        # This prevents AI confusion from high erosion_pct when peak_pnl was tiny
        erosion_status = position.get('erosion_status', 'NONE')
        if erosion_status != 'NONE':
            market_data["position"]["peak_pnl"] = format_number_for_json(position.get('peak_pnl', 0))
            market_data["position"]["erosion_pct"] = format_number_for_json(position.get('erosion_pct', 0))
            market_data["position"]["erosion_status"] = erosion_status
        else:
            market_data["position"]["erosion_status"] = "NONE"
    else:
        market_data["position"] = None
    
    return market_data


def build_portfolio_json(
    portfolio: Any
) -> Dict[str, Any]:
    """
    Build portfolio JSON.
    
    Args:
        portfolio: Portfolio object with attributes like total_return, current_balance, etc.
    
    Returns:
        Portfolio JSON object
    """
    positions_list = []
    if hasattr(portfolio, 'positions') and portfolio.positions:
        for coin, pos in portfolio.positions.items():
            positions_list.append({
                "symbol": coin,
                "direction": pos.get('direction', 'long'),  # ✅ Eklendi
                "quantity": format_number_for_json(pos.get('quantity', 0)),
                "entry_price": format_number_for_json(pos.get('entry_price', 0)),
                "current_price": format_number_for_json(pos.get('current_price', 0)),
                "unrealized_pnl": format_number_for_json(pos.get('unrealized_pnl', 0)),
                "leverage": pos.get('leverage', 1),
                "confidence": format_number_for_json(pos.get('confidence', 0.5))
            })
    
    return {
        "total_return_pct": format_number_for_json(portfolio.total_return if hasattr(portfolio, 'total_return') else 0),
        "available_cash": format_number_for_json(portfolio.current_balance if hasattr(portfolio, 'current_balance') else 0),
        "account_value": format_number_for_json(portfolio.total_value if hasattr(portfolio, 'total_value') else 0),
        "sharpe_ratio": format_number_for_json(portfolio.sharpe_ratio if hasattr(portfolio, 'sharpe_ratio') else None),
        "positions": positions_list
    }


def build_risk_status_json(
    portfolio: Any,
    max_positions: int = 5
) -> Dict[str, Any]:
    """Build risk status JSON."""
    current_positions_count = len(portfolio.positions) if hasattr(portfolio, 'positions') else 0
    total_margin_used = sum(
        p.get('margin_usd', 0) for p in (portfolio.positions.values() if hasattr(portfolio, 'positions') else [])
    )
    available_cash = portfolio.current_balance if hasattr(portfolio, 'current_balance') else 0
    
    return {
        "current_positions_count": current_positions_count,
        "total_margin_used": format_number_for_json(total_margin_used),
        "available_cash": format_number_for_json(available_cash),
        "trading_limits": {
            "min_position": Config.MIN_POSITION_MARGIN_USD,
            "max_positions": max_positions,
            "available_cash_protection": format_number_for_json(available_cash * 0.10),
            "position_sizing_pct": 40.0  # Up to 40% of available cash
        }
    }


def build_historical_context_json(
    trading_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Build historical context JSON."""
    return {
        "total_cycles_analyzed": trading_context.get('total_cycles_analyzed', 0),
        "market_behavior": trading_context.get('market_behavior', 'Unknown'),
        "recent_decisions": trading_context.get('recent_decisions', []),
        "performance_trend": trading_context.get('performance_trend', 'Unknown')
    }


def build_directional_bias_json(
    bias_metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build directional bias metrics JSON (Last 20 trades snapshot).
    
    Args:
        bias_metrics: Output from get_directional_bias_metrics()
    
    Returns:
        Directional bias JSON object
    """
    result = {}
    for side in ('long', 'short'):
        stats = bias_metrics.get(side, {})
        result[side] = {
            "net_pnl": format_number_for_json(stats.get('net_pnl', 0.0)),
            "trades": stats.get('trades', 0),
            "wins": stats.get('wins', 0),
            "losses": stats.get('losses', 0),
            "profitability_index": format_number_for_json(stats.get('profitability_index', 0.0)),
            "rolling_avg": format_number_for_json(stats.get('rolling_avg', 0.0)),
            "consecutive_losses": stats.get('consecutive_losses', 0),
            "consecutive_wins": stats.get('consecutive_wins', 0),
            "caution_active": stats.get('caution_active', False)
        }
    return result


def build_trend_flip_guard_json(
    trend_flip_summary: List[str],
    trend_flip_cooldown: int,
    trend_flip_history_window: int
) -> Dict[str, Any]:
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
        "recent_flips": trend_flip_summary if trend_flip_summary else [],
        "flip_count": len(trend_flip_summary) if trend_flip_summary else 0
    }

