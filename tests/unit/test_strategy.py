"""Tests for src/core/strategy_analyzer.py."""

from unittest.mock import MagicMock

from config.config import Config
from src.core.strategy_analyzer import StrategyAnalyzer


def _make_analyzer():
    return StrategyAnalyzer(market_data=MagicMock())


class TestVolumeQualityScore:
    """Tests for calculate_volume_quality_score."""

    def test_volume_quality_score_excellent(self):
        """High volume ratio (>=2.5) returns 90.0."""
        analyzer = _make_analyzer()
        indicators = {"volume": 500, "avg_volume": 200}
        assert analyzer.calculate_volume_quality_score("XRP", indicators) == 90.0

    def test_volume_quality_score_good(self):
        """Medium volume ratio (>=1.8) returns 75.0."""
        analyzer = _make_analyzer()
        indicators = {"volume": 360, "avg_volume": 200}
        assert analyzer.calculate_volume_quality_score("XRP", indicators) == 75.0

    def test_volume_quality_score_poor(self):
        """Low volume ratio (>=0.7) returns 40.0."""
        analyzer = _make_analyzer()
        indicators = {"volume": 140, "avg_volume": 200}
        assert analyzer.calculate_volume_quality_score("XRP", indicators) == 40.0

    def test_volume_quality_score_no_data(self):
        """No indicators returns 0.0."""
        analyzer = _make_analyzer()
        analyzer.market_data.get_technical_indicators.return_value = {"error": "no data"}
        assert analyzer.calculate_volume_quality_score("XRP") == 0.0


class TestDetectMarketRegime:
    """Tests for detect_market_regime."""

    def test_detect_market_regime_bullish(self):
        """All timeframes bullish returns TF_STRONG_BULLISH."""
        analyzer = _make_analyzer()
        analyzer.market_data.get_averaged_er.return_value = 0.60

        indicators_htf = {"current_price": 2.50, "ema_20": 2.40}
        indicators_15m = {"current_price": 2.50, "ema_20": 2.40}
        indicators_3m = {"current_price": 2.50, "ema_20": 2.40}

        result = analyzer.detect_market_regime(
            "XRP",
            indicators_htf=indicators_htf,
            indicators_15m=indicators_15m,
            indicators_3m=indicators_3m,
        )
        assert result == "TF_STRONG_BULLISH"

    def test_detect_market_regime_choppy(self):
        """Low ER returns CHOPPY."""
        analyzer = _make_analyzer()
        analyzer.market_data.get_averaged_er.return_value = 0.10

        indicators_htf = {"current_price": 2.50, "ema_20": 2.40}

        result = analyzer.detect_market_regime("XRP", indicators_htf=indicators_htf)
        assert result == "CHOPPY"

    def test_bearish_regime(self):
        """All timeframes bearish returns TF_STRONG_BEARISH."""
        analyzer = _make_analyzer()
        analyzer.market_data.get_averaged_er.return_value = 0.60

        indicators_htf = {"current_price": 2.30, "ema_20": 2.50}
        indicators_15m = {"current_price": 2.30, "ema_20": 2.50}
        indicators_3m = {"current_price": 2.30, "ema_20": 2.50}

        result = analyzer.detect_market_regime(
            "XRP",
            indicators_htf=indicators_htf,
            indicators_15m=indicators_15m,
            indicators_3m=indicators_3m,
        )
        assert result == "TF_STRONG_BEARISH"

    def test_zero_avg_volume(self):
        """avg_volume=0 returns 0.0 (division-by-zero guard)."""
        analyzer = _make_analyzer()
        indicators = {"volume": 500, "avg_volume": 0}
        assert analyzer.calculate_volume_quality_score("XRP", indicators) == 0.0

    def test_unclear_regime(self):
        """Missing HTF indicators (None) returns UNCLEAR."""
        analyzer = _make_analyzer()
        result = analyzer.detect_market_regime(
            "XRP",
            indicators_htf=None,
        )
        assert result == "UNCLEAR"
