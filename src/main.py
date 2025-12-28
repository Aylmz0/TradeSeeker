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
import sys
import os

# Add project root to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from src.utils import (
    format_num, safe_file_write, safe_file_read, 
    rate_limiter, RetryManager, DataValidator
)
from src.core.performance_monitor import PerformanceMonitor
from src.core.backtest import AdvancedRiskManager
from src.core.cache_manager import fetch_all_indicators_parallel, fetch_all_indicators_with_cache
from src.core.market_data import RealMarketData
from src.core.portfolio_manager import PortfolioManager
from src.ai.deepseek_api import DeepSeekAPI
from src.services.binance import BinanceOrderExecutor, BinanceAPIError
from src.ai.enhanced_context_provider import EnhancedContextProvider

# Define constants
HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'
HTF_LABEL = HTF_INTERVAL

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
        from config.config import Config
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
        cooldowns = self.portfolio.directional_cooldowns
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

            # EMA20 vs EMA50 comparison
            ema_comparison = f"20-Period EMA: {format_num(ema20_htf)} vs. 50-Period EMA: {format_num(ema50_htf)}"
            
            # Simple trend strength calculation (used mainly for counter-trend context)
            trend_strength = 0
            trend_direction = 'NEUTRAL'
            
            # Higher timeframe EMA alignment (strong trend indicator)
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
            
            # Volume Confirmation
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
            
            # Condition 2: Volume confirmation (>1.0x average)
            volume_ratio = volume_3m / avg_volume_3m if avg_volume_3m > 0 else 0
            if volume_ratio > 1.0:
                conditions_met += 1
                conditions_details.append(f"✅ Volume {volume_ratio:.1f}x average")
            else:
                conditions_details.append(f"❌ Volume {volume_ratio:.1f}x average (need >1.0x)")
            
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
        """Calculate volume confidence based on current vs average volume"""
        try:
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            if 'error' in indicators_3m:
                return 0.0
            
            current_volume = indicators_3m.get('volume', 0)
            avg_volume = indicators_3m.get('avg_volume', 0)
            
            if avg_volume <= 0:
                return 0.0
            
            # Volume comparison
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
        """Generate advanced exit plan with momentum failure detection"""
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
            
            # Calculate TP/SL based on ATR
            if direction == 'long':
                # Long position: TP = entry + 2x ATR, SL = entry - 1x ATR
                profit_target = entry_price + (atr_14 * 2)
                stop_loss = entry_price - atr_14
                
                # Advanced invalidation conditions
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
                
                # Advanced invalidation conditions
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
            # from src.ai.enhanced_context_provider import EnhancedContextProvider
            provider = EnhancedContextProvider()
            return provider.generate_enhanced_context()
        except Exception as e:
            print(f"⚠️ Enhanced context error: {e}")
            return {"error": f"Enhanced context failed: {str(e)}"}

    def format_position_context(self, position_context: Dict) -> str:
        """
        Format position context for prompt.
        
        .. deprecated:: 1.0
            Use :func:`build_position_slot_json` from prompt_json_builders instead.
            This function is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "format_position_context() is deprecated. "
            "Use build_position_slot_json() from prompt_json_builders instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
        """
        Format market regime context for prompt.
        
        .. deprecated:: 1.0
            Use JSON builders from prompt_json_builders instead.
            This function is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "format_market_regime_context() is deprecated. "
            "Use JSON builders from prompt_json_builders instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
        """
        Format performance insights for prompt.
        
        .. deprecated:: 1.0
            Use JSON builders from prompt_json_builders instead.
            This function is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "format_performance_insights() is deprecated. "
            "Use JSON builders from prompt_json_builders instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
        """
        Format long/short feedback for prompt.
        
        .. deprecated:: 1.0
            Use JSON builders from prompt_json_builders instead.
            This function is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "format_directional_feedback() is deprecated. "
            "Use JSON builders from prompt_json_builders instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
        """
        Format risk context for prompt.
        
        .. deprecated:: 1.0
            Use :func:`build_risk_status_json` from prompt_json_builders instead.
            This function is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "format_risk_context() is deprecated. "
            "Use build_risk_status_json() from prompt_json_builders instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if not risk_context:
            return "Risk context: Unknown"
        
        total_risk = risk_context.get('total_risk_usd', 0)
        position_count = risk_context.get('position_count', 0)
        
        return f"Total Risk: ${total_risk:.2f}, Positions: {position_count}"

    # NOTE: _get_counter_trade_analysis_from_indicators function REMOVED
    # Reason: build_counter_trade_json in prompt_json_builders.py performs its own calculation
    # and includes zone+weakening risk modifier logic. The legacy function was redundant.

    def format_suggestions(self, suggestions: List[str]) -> str:
        """
        Format suggestions for prompt.
        
        .. deprecated:: 1.0
            Use JSON builders from prompt_json_builders instead.
            This function is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "format_suggestions() is deprecated. "
            "Use JSON builders from prompt_json_builders instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if not suggestions:
            return "No suggestions at this time"
        
        formatted = ""
        for suggestion in suggestions:
            formatted += f"  • {suggestion}\n"
        return formatted

    def format_trend_reversal_analysis(self, trend_reversal_analysis: Dict) -> str:
        """
        Format trend reversal analysis for prompt.
        
        .. deprecated:: 1.0
            Use :func:`build_trend_reversal_json` from prompt_json_builders instead.
            This function is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "format_trend_reversal_analysis() is deprecated. "
            "Use build_trend_reversal_json() from prompt_json_builders instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
        """
        Format volume ratio with guard rails for extremely low values.
        
        .. deprecated:: 1.0
            Use JSON builders from prompt_json_builders instead.
            This function is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "format_volume_ratio() is deprecated. "
            "Use JSON builders from prompt_json_builders instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
        """
        Helper function to format lists for prompt display.
        
        .. deprecated:: 1.0
            Use JSON builders from prompt_json_builders instead.
            This function is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "format_list() is deprecated. "
            "Use JSON builders from prompt_json_builders instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if not isinstance(lst, list): return []
        return [format_num(x, precision) if x is not None else 'N/A' for x in lst]

    def _fetch_all_indicators_parallel(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Fetch all indicators for all coins in parallel with smart caching.
        
        Note:
            - Delegates to cache_manager functions for better code organization
            - Uses SmartIndicatorCache if Config.USE_SMART_CACHE is enabled
            - Cache strategy:
                * 3m: Always fresh (NEVER cached)
                * 15m: Cached with dynamic TTL (~75% hit rate)
                * HTF: Cached with dynamic TTL (~93% hit rate for 1h)
        """
        if Config.USE_SMART_CACHE:
            # Use cache-aware fetch
            return fetch_all_indicators_with_cache(
                self.market_data,
                self.market_data.available_coins,
                HTF_INTERVAL,
                use_cache=True
            )
        else:
            # Fallback to non-cached version
            return fetch_all_indicators_parallel(
                self.market_data,
                self.market_data.available_coins,
                HTF_INTERVAL
            )

    def generate_alpha_arena_prompt(self) -> str:
        """
        Generate prompt with enhanced data, indicator history and AI decision context
        
        .. deprecated:: 1.0
            Use :meth:`generate_alpha_arena_prompt_json` instead.
            This function is kept for backward compatibility and fallback scenarios.
        
        Returns:
            str: Text-formatted prompt (legacy format)
        """
        import warnings
        warnings.warn(
            "generate_alpha_arena_prompt() is deprecated. "
            "Use generate_alpha_arena_prompt_json() instead. "
            "This function is kept for backward compatibility and fallback scenarios.",
            DeprecationWarning,
            stacklevel=2
        )
        current_time = datetime.now(); minutes_running = int((current_time - self.portfolio.start_time).total_seconds() / 60)
        # Use internal invocation counter, don't increment here, do it in run_cycle
        # self.invocation_count += 1

        # OPTIMIZATION 1 & 2: Fetch all indicators in parallel ONCE, then share
        all_indicators, all_sentiment = self._fetch_all_indicators_parallel()
        
        # OPTIMIZATION 3: Get enhanced context and other data in parallel (non-blocking)
        # These don't need fresh market data, so can run in parallel
        enhanced_context = self.get_enhanced_context()
        
        # NOTE: counter_trade_analysis is now computed directly inside build_counter_trade_json
        
        # Get trend reversal detection using pre-fetched indicators (OPTIMIZATION 3: No re-fetch)
        from src.core.performance_monitor import PerformanceMonitor
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
        cooldowns = self.portfolio.directional_cooldowns
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
        coin_cooldowns = self.portfolio.coin_cooldowns
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
        
        # Use JSON builder for prompt generation (Enables new features like slot constraint instructions)
        from src.ai.prompt_json_builders import build_position_slot_json
        
        # Calculate slot status for prompt context
        position_slots = build_position_slot_json(
            self.portfolio.positions, 
            self.get_max_positions_for_cycle(self.current_cycle_number)
        )
        
        # Add slot constraint instruction to the prompt if applicable
        slot_instruction = ""
        if position_slots.get('constraint_mode') != "NORMAL":
            slot_instruction = f"\nIMPORTANT CONSTRAINT: {position_slots.get('constraint_instruction')}\n"

        prompt = f"""
USER_PROMPT:
It has been {minutes_running} minutes since you started trading. The current time is {current_time} and you've been invoked {self.invocation_count} times. Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. Below that is your current account information, value, performance, positions, etc.

{slot_instruction}

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST
Timeframes note: Unless stated otherwise in a section title, intraday series are provided at 3‑minute intervals. If a coin uses a different interval, it is explicitly stated in that coin's section.

{'='*20} REAL-TIME COUNTER-TRADE ANALYSIS {'='*20}

(Counter-trade analysis is now included in JSON format below)

{'='*20} TREND REVERSAL DETECTION {'='*20}

{self.format_trend_reversal_analysis(trend_reversal_analysis)}

{'='*20} ENHANCED DECISION CONTEXT (Non-binding suggestions) {'='*20}

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
                "OR (b) a SHORT setup (ONLY if no counter-trend LONG signal exists). CRITICAL: If a counter-trend LONG signal exists, DO NOT open a SHORT."
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
                f"OR (b) a LONG alternative (ONLY if no counter-trend SHORT signal exists). CRITICAL: If a counter-trend SHORT signal exists, DO NOT open a LONG."
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
        
        return prompt

    def generate_alpha_arena_prompt_json(self) -> str:
        """
        Generate hybrid JSON prompt with structured data sections.
        
        Uses JSON for data, plain text for instructions and warnings.
        This is the recommended method for prompt generation.
        
        Returns:
            str: Hybrid prompt with JSON sections and text instructions
            
        Note:
            Falls back to text format if JSON serialization fails.
            See :meth:`generate_alpha_arena_prompt` for text-only format (deprecated).
        """
        from src.ai.prompt_json_builders import (
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
            build_directional_bias_json
        )
        from src.ai.prompt_json_utils import safe_json_dumps, create_json_section, compare_token_usage
        from src.ai.prompt_json_schemas import JSON_PROMPT_VERSION
        from config.config import Config
        
        current_time = datetime.now()
        minutes_running = int((current_time - self.portfolio.start_time).total_seconds() / 60)
        
        # Fetch all indicators in parallel (same as original)
        all_indicators, all_sentiment = self._fetch_all_indicators_parallel()
        
        # Get enhanced context and other data
        enhanced_context = self.get_enhanced_context()
        # NOTE: counter_trade_analysis is now computed directly inside build_counter_trade_json
        
        # Get trend reversal detection
        from src.core.performance_monitor import PerformanceMonitor
        performance_monitor = PerformanceMonitor()
        trend_reversal_analysis = performance_monitor.detect_trend_reversal_for_all_coins(
            self.market_data.available_coins,
            indicators_cache=all_indicators
        )
        
        # Get cooldown status
        directional_cooldowns = self.portfolio.directional_cooldowns
        coin_cooldowns = self.portfolio.coin_cooldowns
        counter_trend_cooldown = self.portfolio.counter_trend_cooldown
        relaxed_countertrend_cycles = self.portfolio.relaxed_countertrend_cycles
        
        # Get trading context
        trading_context = self.get_trading_context()
        
        # Get directional bias metrics (for performance snapshot)
        bias_metrics = getattr(self, 'latest_bias_metrics', self.get_directional_bias_metrics())
        
        # Get trend flip summary
        recent_flips = self.portfolio.get_recent_trend_flip_summary()
        flip_history_window = getattr(self.portfolio, 'trend_flip_history_window', self.portfolio.trend_flip_cooldown)
        
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
            print(f"⏭️ Prompt optimization: Skipped cooldown coins (no position): {skipped_cooldown_coins}")
        
        # Metadata
        metadata_json = build_metadata_json(minutes_running, current_time, self.invocation_count)
        
        # Counter-trade analysis (only for tradeable coins)
        counter_trade_json = build_counter_trade_json(
            "",  # Legacy parameter, not used - function calculates internally
            all_indicators,
            coins_to_analyze,  # Filtered list
            HTF_INTERVAL,
            self.market_data  # YENİ: Funding Rate hesaplaması için
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
        effective_limit = self.portfolio.get_effective_same_direction_limit()
        position_slot_json = build_position_slot_json(self.portfolio.positions, max_positions, same_direction_limit=effective_limit)
        
        # Market data (per coin) - only for tradeable coins
        market_data_json = []
        for coin in coins_to_analyze:  # Filtered list (excludes cooldown coins without position)
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
        

        # Build hybrid prompt
        prompt = f"""
USER_PROMPT:
It has been {minutes_running} minutes since you started trading. The current time is {current_time} and you've been invoked {self.invocation_count} times. Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. Below that is your current account information, value, performance, positions, etc.

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST -> NEWEST
Timeframes note: Unless stated otherwise in a section title, intraday series are provided at 3-minute intervals. If a coin uses a different interval, it is explicitly stated in that coin's section.

{'='*20} REAL-TIME COUNTER-TRADE ANALYSIS {'='*20}

{create_json_section("COUNTER_TRADE_ANALYSIS", counter_trade_json, compact=compact)}
""" 
        
        # OPTIMIZATION: Only include TREND_REVERSAL_DATA if positions exist
        if any(self.portfolio.positions.values()):
            prompt += f"""
{'='*20} TREND REVERSAL DETECTION {'='*20}

{create_json_section("TREND_REVERSAL_DATA", trend_reversal_json, compact=compact)}

"""
        
        prompt += f"""
{'='*20} ENHANCED DECISION CONTEXT {'='*20}

{create_json_section("ENHANCED_CONTEXT", enhanced_context_json, compact=compact)}

DIRECTIONAL PERFORMANCE SNAPSHOT (Last 20 trades max):
{create_json_section("DIRECTIONAL_BIAS", directional_bias_json, compact=compact)}

⚠️ IMPORTANT: If a direction (LONG or SHORT) is in cooldown, you MUST NOT propose any new trades in that direction. The system will block them, but you should avoid proposing them in the first place. Cooldown is activated after 3 consecutive losses OR $5+ total loss in a direction.

{create_json_section("COOLDOWN_STATUS", cooldown_status_json, compact=compact)}

⚠️ IMPORTANT: If a coin is in cooldown, you MUST NOT propose any new trades for that coin (LONG or SHORT). The system will block them, but you should avoid proposing them in the first place. Coin cooldown is activated after a loss on that coin and lasts for 1 cycle.


{'='*20} POSITION_SLOTS {'='*20}

{create_json_section("POSITION_SLOTS", position_slot_json, compact=compact)}

⚠️ CRITICAL: If "long_slots_available" is 0, do NOT propose LONG entries. If "short_slots_available" is 0, do NOT propose SHORT entries.
⚠️ CRITICAL: If you identify a valid counter-trend opportunity (e.g. LONG) but cannot execute it because slots are full, you MUST NOT open a trend-following trade in the opposite direction (e.g. SHORT). The counter-trend signal invalidates the trend-following setup. Simply HOLD.

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
            from prompt_json_schemas import validate_json_against_schema
            pass
        
        return prompt

    def parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response - expects clean JSON string from DeepSeekAPI"""
        try:
            parsed_json = json.loads(response)
            if not isinstance(parsed_json, dict):
                print(f"❌ Parsed JSON not dict: {type(parsed_json)}")
                return {"chain_of_thoughts": "Error: Parsed JSON not dict.", "decisions": {}}
            
            thoughts = parsed_json.get("CHAIN_OF_THOUGHTS", "No thoughts provided.")
            decisions = parsed_json.get("DECISIONS", {})
            decisions = self._clean_ai_decisions(decisions)
            return {"chain_of_thoughts": thoughts, "decisions": decisions}
        except Exception as e:
            print(f"❌ General parse error: {e}")
            return {"chain_of_thoughts": f"Parse Error: {e}", "decisions": {}}

    def _clean_ai_decisions(self, decisions: Dict) -> Dict:
        """Clean up AI decisions - preserve position data for hold signals"""
        cleaned_decisions = {}
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                cleaned_decisions[coin] = trade
                continue
            signal = trade.get('signal')
            if signal == 'hold':
                cleaned_trade = {'signal': 'hold'}
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
                cleaned_decisions[coin] = trade
        return cleaned_decisions

    def check_coin_rotation(self, coin: str) -> bool:
        """Coin rotation disabled - always allow trading"""
        return True


    def calculate_optimal_cycle_frequency(self) -> int:
        """Calculate optimal cycle frequency based on market volatility
        Uses CYCLE_INTERVAL_MINUTES from .env as the minimum value"""
        try:
            # Get minimum from .env
            min_interval = Config.CYCLE_INTERVAL_MINUTES * 60  # Convert to seconds
            
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
                print(f"⚠️ No valid ATR values found, using .env value: {Config.CYCLE_INTERVAL_MINUTES} minutes")
                return min_interval
            
            avg_atr = sum(atr_values) / len(atr_values)
            print(f"📊 Average ATR: {avg_atr:.6f}")
            
            # Adjust cycle frequency based on volatility
            # But never go below .env minimum
            if avg_atr < 0.3:    # Düşük volatility
                calculated = 240   # 4 dakikada bir cycle
            elif avg_atr < 0.6:  # Orta volatility  
                calculated = 180   # 3 dakikada bir cycle
            else:                # Yüksek volatility
                calculated = 120   # 2 dakikada bir cycle
            
            # Use the larger of calculated and .env minimum
            result = max(calculated, min_interval)
            print(f"🔄 Cycle interval: {result}s (min from .env: {min_interval}s)")
            return result
                
        except Exception as e:
            print(f"⚠️ Cycle frequency calculation error: {e}")
            return Config.CYCLE_INTERVAL_MINUTES * 60  # Use .env value as fallback


    def track_performance_metrics(self, cycle_number: int):
        """Record basic performance metrics for each cycle"""
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
            performance_history = safe_file_read("data/performance_history.json", [])
            performance_history.append(metrics)
            safe_file_write("data/performance_history.json", performance_history[-100:])  # Son 100 cycle
            
        except Exception as e:
            print(f"⚠️ Performance tracking error: {e}")

    def should_run_performance_analysis(self, cycle_number: int) -> bool:
        """Run analysis every 10 cycles or in critical situations"""
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
        # ✅ FIX: tick_cooldowns() prompt oluşturulduktan SONRA çağrılmalı
        # Çünkü prompt oluşturulurken cooldown değerlerine ihtiyaç var
        # tick_cooldowns() cooldown'ları azaltıyor, bu yüzden prompt'tan SONRA çağrılmalı
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
                from src.core.performance_monitor import PerformanceMonitor
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

            # --- Flash Exit Check (V-Reversal Protection) ---
            # Checks for RSI Spike + Volume Surge in losing positions
            flash_exits_triggered = False
            for coin, position in list(self.portfolio.positions.items()):
                if coin in valid_prices:
                    if self.portfolio.check_flash_exit_conditions(coin, position):
                        print(f"🚨 EXECUTING FLASH EXIT for {coin}...")
                        current_price = valid_prices[coin]
                        
                        # Close position immediately
                        if self.portfolio.is_live_trading:
                            result = self.portfolio.execute_live_close(coin, position, current_price, reason="Flash Exit (V-Reversal)")
                            if result.get('success'):
                                flash_exits_triggered = True
                                if coin in self.portfolio.positions:
                                    del self.portfolio.positions[coin] 
                        else:
                            # Paper trading close
                            self.portfolio.close_position(coin, current_price, reason="Flash Exit (V-Reversal)")
                            flash_exits_triggered = True
            
            if flash_exits_triggered:
                print("ℹ️ Flash Exits triggered. Continuing cycle...")
            # --- End Flash Exit Check ---

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

                print("\n💭 AI ANALYZING...")
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
                                
                                # Deduct commission for simulation realism (round-trip: entry + exit)
                                notional = (entry_price + current_price) / 2 * quantity
                                commission = notional * Config.SIMULATION_COMMISSION_RATE * 2
                                profit -= commission
                                
                                self.portfolio.current_balance += (margin_used + profit)
                                
                                print(f"✅ AI CLOSE: Closed {direction} {coin} @ ${format_num(current_price, 4)} (PnL: ${format_num(profit, 2)}, Commission: ${format_num(commission, 3)})")
                                
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
            
            # ✅ FIX: tick_cooldowns() prompt oluşturulduktan SONRA çağrılmalı
            # Çünkü prompt oluşturulurken cooldown değerlerine ihtiyaç var
            # tick_cooldowns() cooldown'ları azaltıyor, bu yüzden prompt'tan SONRA çağrılmalı
            if hasattr(self.portfolio, 'tick_cooldowns'):
                self.portfolio.tick_cooldowns()
            
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
        print("🔄 TP/SL monitoring loop started (20 second interval)")
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
                    
                    # Flash Exit Check (V-Reversal) - now runs every 20 seconds
                    if Config.FLASH_EXIT_ENABLED and self.portfolio.positions:
                        flash_exits_triggered = False
                        for coin, position in list(self.portfolio.positions.items()):
                            if self.portfolio.check_flash_exit_conditions(coin, position):
                                current_price = valid_prices.get(coin)
                                if current_price:
                                    if self.portfolio.is_live_trading:
                                        live_result = self.portfolio.execute_live_close(
                                            coin=coin, position=position, current_price=current_price,
                                            reason="Flash Exit (V-Reversal) - 20s monitor"
                                        )
                                        if live_result.get('success'):
                                            history_entry = live_result.get('history_entry')
                                            if history_entry:
                                                self.portfolio.add_to_history(history_entry)
                                            del self.portfolio.positions[coin]
                                            flash_exits_triggered = True
                                    else:
                                        # Paper trading close
                                        self.portfolio.close_position(coin, current_price, reason="Flash Exit (V-Reversal) - 20s monitor")
                                        flash_exits_triggered = True
                        if flash_exits_triggered:
                            print("🚨 20-SECOND FLASH EXIT: V-Reversal detected and closed")
                    
                    # Run TP/SL check - all decisions made by 20-second monitoring (like simulation mode)
                    # No Binance TP/SL orders - all managed by monitoring loop
                    positions_closed = self.portfolio.check_and_execute_tp_sl(valid_prices)
                    
                    if positions_closed:
                        print(f"⏰ 20-SECOND TP/SL CHECK: Positions closed")
                    else:
                        print(f"⏰ 20-SECOND TP/SL CHECK: No triggers ({len(self.portfolio.positions)} positions monitored)")
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
                time.sleep(20)

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
        # Calculate correct cycle number: reset_cycle + cycles_since_reset
        last_reset = getattr(self.portfolio, 'last_history_reset_cycle', 0) or 0
        cycles_since_reset = len(self.portfolio.cycle_history)
        start_cycle = last_reset + cycles_since_reset + 1
        print(f"   Resuming from Cycle {start_cycle}... (last reset: {last_reset}, cycles since: {cycles_since_reset})")
        self.invocation_count = start_cycle - 1; current_cycle_number = start_cycle - 1

        # Initialize bot control file
        bot_control_file = "data/bot_control.json"
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
                
                # Check MAX_CYCLES limit - auto-stop at configured cycle number
                if Config.MAX_CYCLES > 0 and current_cycle_number > Config.MAX_CYCLES:
                    print(f"🛑 MAX_CYCLES limit reached ({Config.MAX_CYCLES}). Auto-stopping bot...")
                    break
                
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
            return safe_file_read("data/bot_control.json", {"status": "running", "last_updated": datetime.now().isoformat()})
        except Exception as e:
            print(f"⚠️ Failed to read bot_control.json: {e}")
            # Return default running state if file read fails (fail-safe)
            return {"status": "running", "last_updated": datetime.now().isoformat()}
    
    def _write_bot_control(self, data: Dict[str, Any]):
        """Write bot control file."""
        safe_file_write("data/bot_control.json", data)

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