from typing import Any

from loguru import logger

from config.config import Config
from src.core import constants
from src.utils import format_num


HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"
HTF_LABEL = HTF_INTERVAL


class StrategyAnalyzer:
    def __init__(self, market_data):
        self.market_data = market_data

    # NOTE: Legacy get_counter_trade_information function REMOVED
    # Counter-trade analysis is now handled by build_counter_trade_json in prompt_json_builders.py

    def calculate_volume_quality_score(
        self,
        coin: str,
        indicators_3m: dict[str, Any] | None = None,
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
            if volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS["good"]:
                return 75.0
            if volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS["fair"]:
                return 60.0
            if volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS["poor"]:
                return 40.0
            return 20.0

        except Exception as e:
            logger.warning("Volume quality score calculation error for {}: {}", coin, e)
            return 0.0

    def detect_market_regime(
        self,
        coin: str,
        indicators_htf: dict[str, Any] | None = None,
        indicators_3m: dict[str, Any] | None = None,
        indicators_15m: dict[str, Any] | None = None,
    ) -> str:
        """Detect market condition based on multi-timeframe indicators.

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

            # 0. Choppy Veto (Prioritize market noise over price position)
            # Use averaged 3m + 15m ER (each 10 candles) for balanced choppy detection
            er_val = self.market_data.get_averaged_er(coin, ["3m", "15m"], period=10)

            if er_val < getattr(Config, "CHOPPY_ER_THRESHOLD", 0.30):
                return "CHOPPY"

            # Determine 1h trend
            delta_htf = (price_htf - ema20_htf) / ema20_htf
            price_neutral = abs(delta_htf) <= Config.EMA_NEUTRAL_BAND_PCT
            htf_trend = None
            if not price_neutral:
                htf_trend = "bullish" if delta_htf > 0 else "bearish"
            else:
                return "NEUTRAL"

            # Get 3m and 15m trends
            if indicators_3m is None:
                indicators_3m = self.market_data.get_technical_indicators(coin, "3m")
            trend_3m = None
            if isinstance(indicators_3m, dict) and "error" not in indicators_3m:
                p3m = indicators_3m.get("current_price")
                e3m = indicators_3m.get("ema_20")
                if p3m and e3m:
                    trend_3m = "bullish" if p3m >= e3m else "bearish"

            if indicators_15m is None:
                indicators_15m = self.market_data.get_technical_indicators(coin, "15m")
            trend_15m = None
            if isinstance(indicators_15m, dict) and "error" not in indicators_15m:
                p15m = indicators_15m.get("current_price")
                e15m = indicators_15m.get("ema_20")
                if p15m and e15m:
                    trend_15m = "bullish" if p15m >= e15m else "bearish"

            # Layered Trend Detection Logic (LLM-Safe Unique Prefixes)
            if htf_trend == "bullish":
                # STRONG: All 3 timeframes BULLISH
                if trend_15m == "bullish" and trend_3m == "bullish":
                    return "TF_STRONG_BULLISH"
                # STABLE: 1h + 15m BULLISH (The ideal "Pullback" entry if 3m is bearish)
                if trend_15m == "bullish":
                    return "TF_STABLE_BULLISH"
                # WEAK: Only 1h is BULLISH
                return "TF_WEAK_BULLISH"

            if htf_trend == "bearish":
                # STRONG: All 3 timeframes BEARISH
                if trend_15m == "bearish" and trend_3m == "bearish":
                    return "TF_STRONG_BEARISH"
                # STABLE: 1h + 15m BEARISH
                if trend_15m == "bearish":
                    return "TF_STABLE_BEARISH"
                # WEAK: Only 1h is BEARISH
                return "TF_WEAK_BEARISH"

            return "TF_NEUTRAL"

        except Exception as e:
            logger.warning("Regime detection error for {}: {}", coin, e)
            return "UNCLEAR"
