"""src/core/regime_detector.py
Formalized Market Regime Detection for Tactical Scout v1.2.
Uses ADX, ATR, and EMA to classify market states.
"""

import logging
from typing import Any

from config.config import Config
from src.core import constants


logger = logging.getLogger(__name__)


class RegimeDetector:
    """Classifies market regimes based on technical indicators."""

    @staticmethod
    def classify_coin_regime(indicators: dict[str, Any]) -> str:
        """Classifies the regime for a single coin.
        Returns: BULLISH, BEARISH, NEUTRAL, VOLATILE, CHOPPY
        """
        try:
            adx = indicators.get("adx_14", 0)
            atr = indicators.get("atr_14", 0)
            price = indicators.get("current_price", 0)
            ema20 = indicators.get("ema_20", 0)
            er = indicators.get("efficiency_ratio", 1.0)

            # 1. Choppy Check (Efficiency Ratio)
            if er < getattr(Config, "CHOPPY_ER_THRESHOLD", 0.35):
                return "CHOPPY"

            # 3. Trending vs Neutral (ADX)
            if adx < getattr(Config, "ADX_TREND_LEVEL", 25):
                return "TF_NEUTRAL"

            # 4. Trend Direction
            if price > ema20:
                return "TF_BULLISH"
            return "TF_BEARISH"

        except Exception as e:
            logger.warning(f"Regime classification error: {e}")
            return "TF_NEUTRAL"

    @classmethod
    def detect_overall_regime(cls, coin_indicators: dict[str, dict[str, Any]]) -> str:
        """Detects the global market regime by aggregating coin-level regimes."""
        regimes = [cls.classify_coin_regime(ind) for ind in coin_indicators.values()]
        if not regimes:
            return "TF_NEUTRAL"

        # Count occurrences
        counts = {r: regimes.count(r) for r in set(regimes)}

        # Priority logic
        if counts.get("CHOPPY", 0) >= constants.CHOPPY_THRESHOLD_COUNT:
            return "CHOPPY"
        if counts.get("TF_BULLISH", 0) >= constants.TRENDING_THRESHOLD_COUNT:
            return "TF_BULLISH"
        if counts.get("TF_BEARISH", 0) >= constants.TRENDING_THRESHOLD_COUNT:
            return "TF_BEARISH"

        return "TF_NEUTRAL"

    @classmethod
    def calculate_regime_strength(cls, coin_indicators: dict[str, dict[str, Any]]) -> float:
        """Calculates market regime strength (0.0 to 1.0) based on coin alignment.
        Replaces the legacy get_market_regime_strength in PortfolioManager.
        """
        try:
            bullish_count = 0
            bearish_count = 0
            total_valid = 0

            for indicators in coin_indicators.values():
                price = indicators.get("current_price")
                ema20 = indicators.get("ema_20")

                if (
                    isinstance(price, (int, float))
                    and isinstance(ema20, (int, float))
                    and ema20 > 0
                ):
                    total_valid += 1
                    if price > ema20:
                        bullish_count += 1
                    else:
                        bearish_count += 1

            if total_valid == 0:
                return 0.0

            # Strength = max(bullish, bearish) / total
            return max(bullish_count, bearish_count) / total_valid

        except Exception as e:
            logger.warning(f"Regime strength calculation error: {e}")
            return 0.0
