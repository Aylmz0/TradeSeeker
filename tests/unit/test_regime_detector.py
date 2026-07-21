"""Tests for src/core/regime_detector.py."""

from unittest.mock import patch

import pytest

from config.config import Config
from src.core import constants
from src.core.regime_detector import RegimeDetector


class TestClassifyCoinRegime:
    """Tests for RegimeDetector.classify_coin_regime."""

    def test_classify_bullish(self):
        """price > ema20 + ADX > 25 → BULLISH."""
        indicators = {
            "current_price": 2.50,
            "ema_20": 2.40,
            "adx_14": 30,
            "efficiency_ratio": 0.5,
        }
        result = RegimeDetector.classify_coin_regime(indicators)
        assert result == "BULLISH"

    def test_classify_bearish(self):
        """price < ema20 + ADX > 25 → BEARISH."""
        indicators = {
            "current_price": 2.40,
            "ema_20": 2.50,
            "adx_14": 30,
            "efficiency_ratio": 0.5,
        }
        result = RegimeDetector.classify_coin_regime(indicators)
        assert result == "BEARISH"

    def test_classify_neutral(self):
        """ADX < 25 → NEUTRAL."""
        indicators = {
            "current_price": 2.50,
            "ema_20": 2.40,
            "adx_14": 20,
            "efficiency_ratio": 0.5,
        }
        result = RegimeDetector.classify_coin_regime(indicators)
        assert result == "NEUTRAL"

    def test_classify_choppy(self):
        """ER < 0.35 → CHOPPY."""
        indicators = {
            "current_price": 2.50,
            "ema_20": 2.40,
            "adx_14": 30,
            "efficiency_ratio": 0.20,
        }
        result = RegimeDetector.classify_coin_regime(indicators)
        assert result == "CHOPPY"

    def test_classify_neutral_low_adx(self):
        """ADX < threshold → NEUTRAL."""
        indicators = {
            "current_price": 3.00,
            "ema_20": 2.00,
            "adx_14": 10,
            "efficiency_ratio": 0.6,
        }
        result = RegimeDetector.classify_coin_regime(indicators)
        assert result == "NEUTRAL"

    def test_classify_bullish_with_averaged_er(self):
        """averaged_er overrides indicators' efficiency_ratio."""
        indicators = {
            "current_price": 2.50,
            "ema_20": 2.40,
            "adx_14": 30,
            "efficiency_ratio": 0.1,  # would be CHOPPY if used
        }
        result = RegimeDetector.classify_coin_regime(indicators, averaged_er=0.5)
        assert result == "BULLISH"

    def test_classify_exception_handling(self):
        """Invalid indicators → NEUTRAL (exception path)."""
        result = RegimeDetector.classify_coin_regime({})
        assert result == "NEUTRAL"


class TestDetectOverallRegime:
    """Tests for RegimeDetector.detect_overall_regime."""

    def test_overall_regime_bullish(self):
        """Majority BULLISH → BULLISH."""
        coin_indicators = {
            f"COIN{i}": {
                "current_price": 2.50 + i * 0.1,
                "ema_20": 2.40,
                "adx_14": 30,
                "efficiency_ratio": 0.5,
            }
            for i in range(constants.TRENDING_THRESHOLD_COUNT)
        }
        result = RegimeDetector.detect_overall_regime(coin_indicators)
        assert result == "BULLISH"

    def test_overall_regime_bearish(self):
        """Majority BEARISH → BEARISH."""
        coin_indicators = {
            f"COIN{i}": {
                "current_price": 2.30,
                "ema_20": 2.50,
                "adx_14": 30,
                "efficiency_ratio": 0.5,
            }
            for i in range(constants.TRENDING_THRESHOLD_COUNT)
        }
        result = RegimeDetector.detect_overall_regime(coin_indicators)
        assert result == "BEARISH"

    def test_overall_regime_neutral(self):
        """Mixed regimes → NEUTRAL."""
        bullish_indicators = {
            "current_price": 2.50,
            "ema_20": 2.40,
            "adx_14": 30,
            "efficiency_ratio": 0.5,
        }
        bearish_indicators = {
            "current_price": 2.30,
            "ema_20": 2.50,
            "adx_14": 30,
            "efficiency_ratio": 0.5,
        }
        coin_indicators = {
            "BTC": bullish_indicators,
            "ETH": bearish_indicators,
            "SOL": bullish_indicators,
            "DOGE": bearish_indicators,
        }
        result = RegimeDetector.detect_overall_regime(coin_indicators)
        assert result == "NEUTRAL"

    def test_overall_regime_empty(self):
        """No coins → NEUTRAL."""
        result = RegimeDetector.detect_overall_regime({})
        assert result == "NEUTRAL"


class TestCalculateRegimeStrength:
    """Tests for RegimeDetector.calculate_regime_strength."""

    def test_regime_strength(self):
        """Alignment calculation is correct."""
        coin_indicators = {
            "BTC": {"current_price": 2.50, "ema_20": 2.40},
            "ETH": {"current_price": 2.50, "ema_20": 2.40},
            "SOL": {"current_price": 2.30, "ema_20": 2.50},
        }
        # 2 bullish, 1 bearish → max(2,1)/3 = 0.666...
        strength = RegimeDetector.calculate_regime_strength(coin_indicators)
        assert strength == pytest.approx(2 / 3, abs=1e-9)

    def test_regime_strength_all_bullish(self):
        """All bullish → strength 1.0."""
        coin_indicators = {
            "BTC": {"current_price": 2.50, "ema_20": 2.40},
            "ETH": {"current_price": 3.00, "ema_20": 2.90},
        }
        strength = RegimeDetector.calculate_regime_strength(coin_indicators)
        assert strength == 1.0

    def test_regime_strength_no_valid(self):
        """No valid indicators → 0.0."""
        strength = RegimeDetector.calculate_regime_strength({})
        assert strength == 0.0

    def test_regime_strength_missing_ema(self):
        """Invalid indicator data excluded from calculation."""
        coin_indicators = {
            "BTC": {"current_price": 2.50, "ema_20": 2.40},
            "ETH": {"current_price": 2.50},  # missing ema_20
        }
        strength = RegimeDetector.calculate_regime_strength(coin_indicators)
        assert strength == 1.0  # only 1 valid coin, bullish

    def test_classify_price_equals_ema(self):
        """price == ema20 → BEARISH (not strictly greater)."""
        indicators = {
            "current_price": 2.50,
            "ema_20": 2.50,
            "adx_14": 30,
            "efficiency_ratio": 0.5,
        }
        result = RegimeDetector.classify_coin_regime(indicators)
        assert result == "BEARISH"

    def test_overall_regime_with_averaged_ers(self):
        """averaged_ers dict overrides per-coin ER, preventing choppy classification."""
        indicators = {
            "BTC": {
                "current_price": 2.50,
                "ema_20": 2.40,
                "adx_14": 30,
                "efficiency_ratio": 0.1,  # low ER → CHOPPY if used
            },
            "ETH": {
                "current_price": 2.60,
                "ema_20": 2.50,
                "adx_14": 30,
                "efficiency_ratio": 0.1,
            },
            "SOL": {
                "current_price": 2.70,
                "ema_20": 2.60,
                "adx_14": 30,
                "efficiency_ratio": 0.1,
            },
            "DOGE": {
                "current_price": 2.80,
                "ema_20": 2.70,
                "adx_14": 30,
                "efficiency_ratio": 0.1,
            },
        }
        averaged_ers = {
            "BTC": 0.5,
            "ETH": 0.5,
            "SOL": 0.5,
            "DOGE": 0.5,
        }
        result = RegimeDetector.detect_overall_regime(indicators, averaged_ers=averaged_ers)
        assert result == "BULLISH"

    def test_regime_strength_all_bearish(self):
        """All bearish → strength 1.0."""
        coin_indicators = {
            "BTC": {"current_price": 2.30, "ema_20": 2.50},
            "ETH": {"current_price": 1.80, "ema_20": 2.00},
            "SOL": {"current_price": 140.0, "ema_20": 150.0},
        }
        strength = RegimeDetector.calculate_regime_strength(coin_indicators)
        assert strength == 1.0
