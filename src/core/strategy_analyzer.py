import copy
import json
import re
from typing import Any
from config.config import Config
from src.utils import format_num

HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'
HTF_LABEL = HTF_INTERVAL

class StrategyAnalyzer:
    def __init__(self, market_data):
        self.market_data = market_data

    def check_trend_alignment(self, coin: str) -> bool:
        """Check if trends are aligned across multiple timeframes (1h + 15m + 3m)"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_15m = self.market_data.get_technical_indicators(coin, "15m")
            indicators_3m = self.market_data.get_technical_indicators(coin, "3m")

            if "error" in indicators_htf or "error" in indicators_15m or "error" in indicators_3m:
                return False

            price_htf = indicators_htf.get("current_price")
            ema20_htf = indicators_htf.get("ema_20")
            price_15m = indicators_15m.get("current_price")
            ema20_15m = indicators_15m.get("ema_20")
            price_3m = indicators_3m.get("current_price")
            ema20_3m = indicators_3m.get("ema_20")

            # Trend alignment: All three timeframes in same direction (strongest signal)
            trend_aligned = (
                price_htf > ema20_htf and price_15m > ema20_15m and price_3m > ema20_3m
            ) or (price_htf < ema20_htf and price_15m < ema20_15m and price_3m < ema20_3m)

            return trend_aligned

        except Exception as e:
            print(f"[WARNING] Trend alignment error for {coin}: {e}")
            return False

    def check_momentum_alignment(self, coin: str) -> bool:
        """Check if momentum indicators are aligned across timeframes (1h + 15m + 3m)"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_15m = self.market_data.get_technical_indicators(coin, "15m")
            indicators_3m = self.market_data.get_technical_indicators(coin, "3m")

            if "error" in indicators_htf or "error" in indicators_15m or "error" in indicators_3m:
                return False

            rsi_3m = indicators_3m.get("rsi_14", 50)
            rsi_15m = indicators_15m.get("rsi_14", 50)
            rsi_htf = indicators_htf.get("rsi_14", 50)
            macd_3m = indicators_3m.get("macd", 0)
            macd_15m = indicators_15m.get("macd", 0)
            macd_htf = indicators_htf.get("macd", 0)

            # Momentum alignment: All three timeframes showing same momentum direction (strongest signal)
            momentum_aligned = (
                rsi_3m > 50
                and rsi_15m > 50
                and rsi_htf > 50
                and macd_3m > 0
                and macd_15m > 0
                and macd_htf > 0
            ) or (
                rsi_3m < 50
                and rsi_15m < 50
                and rsi_htf < 50
                and macd_3m < 0
                and macd_15m < 0
                and macd_htf < 0
            )

            return momentum_aligned

        except Exception as e:
            print(f"[WARNING] Momentum alignment error for {coin}: {e}")
            return False

    def enhanced_trend_detection(self, coin: str) -> dict[str, Any]:
        """Enhanced trend detection with simple trend strength and counter-trade detection"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_15m = self.market_data.get_technical_indicators(coin, "15m")
            indicators_3m = self.market_data.get_technical_indicators(coin, "3m")

            if "error" in indicators_htf or "error" in indicators_15m or "error" in indicators_3m:
                return {
                    "trend_strength": 0,
                    "trend_direction": "NEUTRAL",
                    "ema_comparison": "N/A",
                    "volume_confidence": 0.0,
                }

            price_htf = indicators_htf.get("current_price")
            ema20_htf = indicators_htf.get("ema_20")
            ema50_htf = indicators_htf.get("ema_50")
            price_15m = indicators_15m.get("current_price")
            ema20_15m = indicators_15m.get("ema_20")
            price_3m = indicators_3m.get("current_price")
            ema20_3m = indicators_3m.get("ema_20")

            trend_label_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"

            # EMA20 vs EMA50 comparison
            ema_comparison = (
                f"20-Period EMA: {format_num(ema20_htf)} vs. 50-Period EMA: {format_num(ema50_htf)}"
            )

            # Simple trend strength calculation (used mainly for counter-trend context)
            trend_strength = 0
            trend_direction = "NEUTRAL"

            # Higher timeframe EMA alignment (strong trend indicator)
            if ema20_htf > ema50_htf and price_htf > ema20_htf:
                trend_strength += 3  # Strong bullish (EMA20 > EMA50 + price > EMA20)
                trend_direction = "STRONG_BULLISH"
            elif ema20_htf < ema50_htf and price_htf < ema20_htf:
                trend_strength += 3  # Strong bearish (EMA20 < EMA50 + price < EMA20)
                trend_direction = "STRONG_BEARISH"
            elif ema20_htf > ema50_htf:
                trend_strength += 1  # Weak bullish (EMA20 > EMA50 but price < EMA20)
                trend_direction = "WEAK_BULLISH"
            elif ema20_htf < ema50_htf:
                trend_strength += 1  # Weak bearish (EMA20 < EMA50 but price > EMA20)
                trend_direction = "WEAK_BEARISH"

            # Multi-timeframe alignment bonus (1h + 15m + 3m)
            alignment_count = 0
            if (
                price_htf > ema20_htf
                and price_15m > ema20_15m
                and price_3m > ema20_3m
                or price_htf < ema20_htf
                and price_15m < ema20_15m
                and price_3m < ema20_3m
            ):
                alignment_count = 3
            elif (
                (price_htf > ema20_htf and price_15m > ema20_15m)
                or (price_htf > ema20_htf and price_3m > ema20_3m)
                or (price_15m > ema20_15m and price_3m > ema20_3m)
            ) or (
                (price_htf < ema20_htf and price_15m < ema20_15m)
                or (price_htf < ema20_htf and price_3m < ema20_3m)
                or (price_15m < ema20_15m and price_3m < ema20_3m)
            ):
                alignment_count = 2

            if alignment_count >= 3:
                trend_strength += 2  # Strong multi-timeframe alignment (all 3 timeframes)
            elif alignment_count == 2:
                trend_strength += 1  # Medium multi-timeframe alignment (2 of 3 timeframes)

            # Volume Confirmation
            volume_confidence = self.calculate_volume_confidence(coin)

            return {
                "trend_strength": trend_strength,
                "trend_direction": trend_direction,
                "ema_comparison": ema_comparison,
                "price_vs_ema20_htf": "ABOVE" if price_htf > ema20_htf else "BELOW",
                "price_vs_ema20_3m": "ABOVE" if price_3m > ema20_3m else "BELOW",
                "volume_confidence": volume_confidence,
                "trend_htf": trend_label_htf,
            }

        except Exception as e:
            print(f"[WARNING] Enhanced trend detection error for {coin}: {e}")
            return {
                "trend_strength": 0,
                "trend_direction": "NEUTRAL",
                "ema_comparison": "ERROR",
                "volume_confidence": 0.0,
            }

    # NOTE: Legacy get_counter_trade_information function REMOVED
    # Counter-trade analysis is now handled by build_counter_trade_json in prompt_json_builders.py

    def calculate_comprehensive_trend_strength(self, coin: str) -> dict[str, Any]:
        """Calculate comprehensive trend strength using 5 technical indicators with weighted scoring"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_3m = self.market_data.get_technical_indicators(coin, "3m")

            if "error" in indicators_htf or "error" in indicators_3m:
                return {"strength_score": 0, "trend_direction": "UNCLEAR", "component_scores": {}}

            price_htf = indicators_htf.get("current_price")
            ema20_htf = indicators_htf.get("ema_20")
            ema50_htf = indicators_htf.get("ema_50")
            rsi_htf = indicators_htf.get("rsi_14", 50)
            macd_htf = indicators_htf.get("macd", 0)
            volume_htf = indicators_htf.get("volume", 0)
            avg_volume_htf = indicators_htf.get("avg_volume", 1)

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
                + bb_strength * 0.20  # 20% weight
                + ma_strength * 0.20  # 20% weight
            )

            # Determine trend direction
            trend_direction = self.determine_trend_direction(
                price_htf, ema20_htf, ema50_htf, rsi_htf, macd_htf
            )

            return {
                "strength_score": total_strength,
                "trend_direction": trend_direction,
                "component_scores": {
                    "rsi": rsi_strength,
                    "macd": macd_strength,
                    "volume": volume_strength,
                    "bollinger_bands": bb_strength,
                    "moving_averages": ma_strength,
                },
                "confidence_level": self.get_confidence_level(total_strength),
            }

        except Exception as e:
            print(f"[WARNING] Comprehensive trend strength error for {coin}: {e}")
            return {"strength_score": 0, "trend_direction": "UNCLEAR", "component_scores": {}}

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

    def analyze_bollinger_bands_strength(self, indicators: dict) -> float:
        """Analyze Bollinger Bands strength (0-1 scale)"""
        try:
            price = indicators.get("current_price", 0)
            ema20 = indicators.get("ema_20", price)
            atr_14 = indicators.get("atr_14", 0)

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
            print(f"[WARNING] Bollinger Bands analysis error: {e}")
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
            print(f"[WARNING] Moving Averages analysis error: {e}")
            return 0.5

    def determine_trend_direction(
        self, price: float, ema20: float, ema50: float, rsi: float, macd: float
    ) -> str:
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
            indicators_3m = self.market_data.get_technical_indicators(coin, "3m")
            if "error" in indicators_3m:
                return 0.0

            current_volume = indicators_3m.get("volume", 0)
            avg_volume = indicators_3m.get("avg_volume", 0)

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
            print(f"[WARNING] Volume confidence calculation error for {coin}: {e}")
            return 0.0

    def calculate_volume_quality_score(
        self, coin: str, indicators_3m: dict[str, Any] | None = None
    ) -> float:
        """Calculate volume quality score (0-100) based on Config thresholds"""
        try:
            if indicators_3m is None or not isinstance(indicators_3m, dict):
                indicators_3m = self.market_data.get_technical_indicators(coin, "3m")
            if "error" in indicators_3m:
                return 0.0

            current_volume = indicators_3m.get("volume", 0)
            avg_volume = indicators_3m.get("avg_volume", 0)

            if avg_volume <= 0:
                return 0.0

            volume_ratio = current_volume / avg_volume

            # Calculate score based on Config thresholds
            if volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS["excellent"]:
                return 90.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS["good"]:
                return 75.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS["fair"]:
                return 60.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS["poor"]:
                return 40.0
            else:
                return 20.0

        except Exception as e:
            print(f"[WARNING] Volume quality score calculation error for {coin}: {e}")
            return 0.0

    def should_enhance_short_sizing(self, coin: str) -> bool:
        """Check if short position should be enhanced (15% larger)"""
        try:
            indicators_3m = self.market_data.get_technical_indicators(coin, "3m")
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)

            if "error" in indicators_3m or "error" in indicators_htf:
                return False

            # Enhanced short conditions:
            # 1. 3m RSI > 70 (overbought)
            rsi_3m = indicators_3m.get("rsi_14", 50)
            # 2. Volume > 1.5x average
            volume_ratio = indicators_3m.get("volume", 0) / indicators_3m.get("avg_volume", 1)
            # 3. Higher timeframe trend bearish
            price_htf = indicators_htf.get("current_price")
            ema20_htf = indicators_htf.get("ema_20")
            trend_bearish = price_htf < ema20_htf

            # All conditions must be met
            return rsi_3m > 70 and volume_ratio > 1.5 and trend_bearish

        except Exception as e:
            print(f"[WARNING] Enhanced short sizing check error for {coin}: {e}")
            return False

    def generate_advanced_exit_plan(
        self, coin: str, direction: str, entry_price: float
    ) -> dict[str, Any]:
        """Generate advanced exit plan - TP/SL is now handled by execute_live_entry using Config multipliers"""
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)

            if "error" in indicators_htf:
                return {
                    "profit_target": None,
                    "stop_loss": None,
                    "invalidation_condition": "Unable to generate exit plan due to data error",
                }

            rsi_14 = indicators_htf.get("rsi_14", 50)
            htf_upper = HTF_LABEL.upper()

            # Only generate invalidation conditions - TP/SL handled by execute_live_entry
            if direction == "long":
                if rsi_14 > 70:
                    invalidation_condition = (
                        f"If {htf_upper} RSI breaks back below 60, signaling momentum failure"
                    )
                elif rsi_14 < 40:
                    invalidation_condition = (
                        f"If {htf_upper} RSI breaks above 50, signaling momentum recovery"
                    )
                else:
                    invalidation_condition = f"If {htf_upper} price closes below {htf_upper} EMA20, signaling trend reversal"
            elif rsi_14 < 30:
                invalidation_condition = (
                    f"If {htf_upper} RSI breaks back above 40, signaling momentum failure"
                )
            elif rsi_14 > 60:
                invalidation_condition = (
                    f"If {htf_upper} RSI breaks below 50, signaling momentum recovery"
                )
            else:
                invalidation_condition = (
                    f"If {htf_upper} price closes above {htf_upper} EMA20, signaling trend reversal"
                )

            return {
                "profit_target": None,  # Handled by execute_live_entry
                "stop_loss": None,  # Handled by execute_live_entry
                "invalidation_condition": invalidation_condition,
                "rsi_context": f"{htf_upper} RSI: {rsi_14:.1f}",
            }

        except Exception as e:
            print(f"[WARNING] Advanced exit plan generation error for {coin}: {e}")
            return {
                "profit_target": None,
                "stop_loss": None,
                "invalidation_condition": f"Error generating exit plan: {str(e)}",
            }

    def detect_market_regime(
        self,
        coin: str,
        indicators_htf: dict[str, Any] | None = None,
        indicators_3m: dict[str, Any] | None = None,
        indicators_15m: dict[str, Any] | None = None,
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
            if not isinstance(indicators_htf, dict) or "error" in indicators_htf:
                return "UNCLEAR"

            price_htf = indicators_htf.get("current_price")
            ema20_htf = indicators_htf.get("ema_20")

            if (
                not isinstance(price_htf, (int, float))
                or not isinstance(ema20_htf, (int, float))
                or ema20_htf == 0
            ):
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
                indicators_3m = self.market_data.get_technical_indicators(coin, "3m")
            trend_3m = None
            if isinstance(indicators_3m, dict) and "error" not in indicators_3m:
                price_3m = indicators_3m.get("current_price")
                ema20_3m = indicators_3m.get("ema_20")
                if (
                    isinstance(price_3m, (int, float))
                    and isinstance(ema20_3m, (int, float))
                    and ema20_3m > 0
                ):
                    trend_3m = "bullish" if price_3m >= ema20_3m else "bearish"

            # Get 15m trend
            if indicators_15m is None:
                indicators_15m = self.market_data.get_technical_indicators(coin, "15m")
            trend_15m = None
            if isinstance(indicators_15m, dict) and "error" not in indicators_15m:
                price_15m = indicators_15m.get("current_price")
                ema20_15m = indicators_15m.get("ema_20")
                if (
                    isinstance(price_15m, (int, float))
                    and isinstance(ema20_15m, (int, float))
                    and ema20_15m > 0
                ):
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
            print(f"[WARNING] Regime detection error for {coin}: {e}")
            return "UNCLEAR"

