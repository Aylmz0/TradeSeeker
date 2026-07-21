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
    def classify_coin_regime(indicators: dict[str, Any], averaged_er: float | None = None) -> str:
        """Classify the regime for a single coin.

        Uses ER (choppy check), ADX (trending vs neutral), and price vs EMA20
        (direction) to determine regime.

        Args:
            indicators: Technical indicators dictionary for the coin.
            averaged_er: Pre-computed averaged ER from 3m+15m (optional).

        Returns:
            Regime string: "BULLISH", "BEARISH", "NEUTRAL", "VOLATILE", or "CHOPPY".
        """
        try:
            adx = indicators.get("adx_14", 0)
            atr = indicators.get("atr_14", 0)
            price = indicators.get("current_price", 0)
            ema20 = indicators.get("ema_20", 0)
            er = averaged_er if averaged_er is not None else indicators.get("efficiency_ratio", 1.0)

            # 1. Choppy Check (Efficiency Ratio: averaged 3m+15m)
            if er < getattr(Config, "CHOPPY_ER_THRESHOLD", 0.35):
                return "CHOPPY"

            # 3. Trending vs Neutral (ADX)
            if adx < getattr(Config, "ADX_TREND_LEVEL", 25):
                return "NEUTRAL"

            # 4. Trend Direction
            if price > ema20:
                return "BULLISH"
            return "BEARISH"

        except Exception as e:
            logger.warning("Regime classification error: {}", e)
            return "NEUTRAL"

    @classmethod
    def detect_overall_regime(
        cls,
        coin_indicators: dict[str, dict[str, Any]],
        averaged_ers: dict[str, float] | None = None,
    ) -> str:
        """Detect global market regime by aggregating coin-level regimes.

        Uses priority logic: CHOPPY > BULLISH/BEARISH > NEUTRAL.

        Args:
            coin_indicators: Dictionary mapping coin symbols to their indicators.
            averaged_ers: Pre-computed averaged ER per coin (optional).

        Returns:
            Overall regime string.
        """
        regimes = [
            cls.classify_coin_regime(ind, averaged_er=(averaged_ers or {}).get(coin))
            for coin, ind in coin_indicators.items()
        ]
        if not regimes:
            return "NEUTRAL"

        # Count occurrences
        counts = {r: regimes.count(r) for r in set(regimes)}

        # Priority logic
        if counts.get("CHOPPY", 0) >= constants.CHOPPY_THRESHOLD_COUNT:
            return "CHOPPY"
        if counts.get("BULLISH", 0) >= constants.TRENDING_THRESHOLD_COUNT:
            return "BULLISH"
        if counts.get("BEARISH", 0) >= constants.TRENDING_THRESHOLD_COUNT:
            return "BEARISH"

        return "NEUTRAL"

    @classmethod
    def calculate_regime_strength(cls, coin_indicators: dict[str, dict[str, Any]]) -> float:
        """Calculate market regime strength based on coin alignment.

        Strength = max(bullish_count, bearish_count) / total_valid.

        Args:
            coin_indicators: Dictionary mapping coin symbols to their indicators.

        Returns:
            Regime strength between 0.0 (no alignment) and 1.0 (full alignment).
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
            logger.warning("Regime strength calculation error: {}", e)
            return 0.0
