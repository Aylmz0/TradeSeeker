"""
JSON builder functions for AI prompt generation.
Converts data structures to JSON format for hybrid prompt.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from prompt_json_utils import (
    safe_json_dumps,
    format_number_for_json,
    compress_series,
    create_json_section
)
from prompt_json_schemas import JSON_PROMPT_VERSION
from config import Config
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
    htf_interval: str
) -> List[Dict[str, Any]]:
    """
    Build counter-trade analysis JSON from text analysis or indicators.
    
    Args:
        counter_trade_analysis: Text analysis (legacy format)
        all_indicators: Pre-fetched indicators dict
        available_coins: List of coins to analyze
        htf_interval: Higher timeframe interval (e.g., '1h')
    
    Returns:
        List of counter-trade analysis objects
    """
    analysis_list = []
    
    for coin in available_coins:
        try:
            indicators_3m = all_indicators.get(coin, {}).get('3m', {})
            indicators_htf = all_indicators.get(coin, {}).get(htf_interval, {})
            
            if 'error' in indicators_3m or 'error' in indicators_htf:
                continue
            
            # Extract key indicators
            price_htf = format_number_for_json(indicators_htf.get('current_price'))
            ema20_htf = format_number_for_json(indicators_htf.get('ema_20'))
            price_3m = format_number_for_json(indicators_3m.get('current_price'))
            ema20_3m = format_number_for_json(indicators_3m.get('ema_20'))
            rsi_3m = format_number_for_json(indicators_3m.get('rsi_14', 50))
            volume_3m = format_number_for_json(indicators_3m.get('volume', 0))
            avg_volume_3m = format_number_for_json(indicators_3m.get('avg_volume', 1))
            macd_3m = format_number_for_json(indicators_3m.get('macd', 0))
            macd_signal_3m = format_number_for_json(indicators_3m.get('macd_signal', 0))
            
            if price_htf is None or ema20_htf is None or price_3m is None or ema20_3m is None:
                continue
            
            # Determine trend directions
            trend_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"
            trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
            
            # Evaluate conditions
            conditions = {
                "condition_1": (trend_htf == "BULLISH" and trend_3m == "BEARISH") or (trend_htf == "BEARISH" and trend_3m == "BULLISH"),
                "condition_2": (volume_3m or 0) / (avg_volume_3m or 1) > 1.5 if avg_volume_3m else False,
                "condition_3": (trend_htf == "BULLISH" and (rsi_3m or 50) < 25) or (trend_htf == "BEARISH" and (rsi_3m or 50) > 75),
                "condition_4": abs((price_3m or 0) - (ema20_3m or 0)) / (price_3m or 1) * 100 < 1.0 if price_3m and ema20_3m else False,
                "condition_5": (trend_htf == "BULLISH" and (macd_3m or 0) > (macd_signal_3m or 0)) or (trend_htf == "BEARISH" and (macd_3m or 0) < (macd_signal_3m or 0))
            }
            
            total_met = sum(1 for v in conditions.values() if v)
            
            # Determine risk level
            if total_met >= 4:
                risk_level = "LOW_RISK"
            elif total_met >= 3:
                risk_level = "MEDIUM_RISK"
            elif total_met >= 2:
                risk_level = "HIGH_RISK"
            else:
                risk_level = "VERY_HIGH_RISK"
            
            analysis_list.append({
                "coin": coin,
                "htf_trend": trend_htf,
                "three_m_trend": trend_3m,
                "conditions": {
                    **conditions,
                    "total_met": total_met
                },
                "risk_level": risk_level,
                "volume_ratio": format_number_for_json((volume_3m or 0) / (avg_volume_3m or 1)) if avg_volume_3m else None,
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
        
        # Map signal strength to reversal strength
        if signal_strength == "HIGH_LOSS_RISK":
            strength = "STRONG"
        elif signal_strength == "MEDIUM_LOSS_RISK":
            strength = "MEDIUM"
        elif signal_strength == "LOW_LOSS_RISK":
            strength = "INFORMATIONAL"
        else:
            strength = "NONE"
        
        reversal_list.append({
            "coin": coin,
            "has_position": has_position,
            "position_direction": position_direction,
            "position_duration_minutes": format_number_for_json(position_duration_minutes),
            "reversal_signals": {
                "htf_reversal": False,  # Would need more detailed analysis
                "fifteen_m_reversal": False,
                "three_m_reversal": len(loss_risk_signals) > 0,
                "strength": strength
            },
            "loss_risk_signal": signal_strength,
            "current_trend_htf": analysis.get('current_trend_4h', 'UNKNOWN'),
            "current_trend_3m": analysis.get('current_trend_3m', 'UNKNOWN')
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
            "win_rate": format_number_for_json(performance_insights.get('win_rate'))
        },
        "directional_feedback": {
            "long_performance": {
                "net_pnl": format_number_for_json(directional_feedback.get('long', {}).get('total_pnl', 0)),
                "trades": directional_feedback.get('long', {}).get('trades', 0),
                "win_rate": format_number_for_json(directional_feedback.get('long', {}).get('win_rate', 0))
            },
            "short_performance": {
                "net_pnl": format_number_for_json(directional_feedback.get('short', {}).get('total_pnl', 0)),
                "trades": directional_feedback.get('short', {}).get('trades', 0),
                "win_rate": format_number_for_json(directional_feedback.get('short', {}).get('win_rate', 0))
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
    max_positions: int
) -> Dict[str, Any]:
    """Build position slot status JSON."""
    total_open = len(portfolio_positions)
    long_slots = sum(1 for p in portfolio_positions.values() if p.get('direction', 'long') == 'long')
    short_slots = sum(1 for p in portfolio_positions.values() if p.get('direction', 'long') == 'short')
    
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
    
    return {
        "total_open": total_open,
        "max_positions": max_positions,
        "long_slots_used": long_slots,
        "short_slots_used": short_slots,
        "available_slots": max_positions - total_open,
        "weakest_position": weakest_position
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
            current["volume"] = format_number_for_json(indicators.get('volume'))
        
        # Build series with compression if needed
        price_series = indicators.get('price_series', [])
        ema_series = indicators.get('ema_20_series', [])
        rsi_series = indicators.get('rsi_14_series', [])
        macd_series = indicators.get('macd_series', [])
        
        series = {}
        
        # Compress series if too long
        if len(price_series) > max_series_length:
            compressed_price = compress_series(price_series, max_length=max_series_length)
            series["price"] = compressed_price
        else:
            series["price"] = [format_number_for_json(p) for p in price_series]
        
        if len(ema_series) > max_series_length:
            compressed_ema = compress_series(ema_series, max_length=max_series_length)
            series["ema20"] = compressed_ema
        else:
            series["ema20"] = [format_number_for_json(e) for e in ema_series]
        
        if len(rsi_series) > max_series_length:
            compressed_rsi = compress_series(rsi_series, max_length=max_series_length)
            series["rsi"] = compressed_rsi
        else:
            series["rsi"] = [format_number_for_json(r) for r in rsi_series]
        
        if len(macd_series) > max_series_length:
            compressed_macd = compress_series(macd_series, max_length=max_series_length)
            series["macd"] = compressed_macd
        else:
            series["macd"] = [format_number_for_json(m) for m in macd_series]
        
        if has_atr:
            atr_series = indicators.get('atr_14_series', [])
            if len(atr_series) > max_series_length:
                compressed_atr = compress_series(atr_series, max_length=max_series_length)
                series["atr"] = compressed_atr
            else:
                series["atr"] = [format_number_for_json(a) for a in atr_series]
        
        if has_volume:
            volume_series = indicators.get('volume_series', [])
            if len(volume_series) > max_series_length:
                compressed_volume = compress_series(volume_series, max_length=max_series_length)
                series["volume"] = compressed_volume
            else:
                series["volume"] = [format_number_for_json(v) for v in volume_series]
        
        return {
            "current": current,
            "series": series
        }
    
    market_data = {
        "coin": coin,
        "market_regime": market_regime,
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
            "min_position": 10.0,  # $10 minimum
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
            "win_rate": format_number_for_json(stats.get('win_rate', 0.0)),
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

