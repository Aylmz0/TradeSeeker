"""Tests for src/ai/enhanced_context_provider.py."""

from unittest.mock import patch, MagicMock

from src.ai.enhanced_context_provider import EnhancedContextProvider


def _make_provider():
    return EnhancedContextProvider()


class TestPositionContext:
    """Tests for get_enhanced_position_context."""

    def test_position_context(self):
        """Enhanced position context has correct keys."""
        provider = _make_provider()
        portfolio_state = {
            "positions": {
                "XRP": {
                    "entry_price": 2.0,
                    "current_price": 2.10,
                    "unrealized_pnl": 5.0,
                    "direction": "long",
                    "entry_time": "2026-01-01T00:00:00+00:00",
                    "exit_plan": {"profit_target": 2.50, "stop_loss": 1.90},
                    "trend_context": {},
                    "confidence": 0.8,
                },
            },
        }
        result = provider.get_enhanced_position_context(portfolio_state)
        assert "XRP" in result
        pos = result["XRP"]
        assert pos["unrealized_pnl"] == 5.0
        assert pos["direction"] == "long"
        assert "profit_target_progress" in pos
        assert "time_in_trade_minutes" in pos
        assert "distance_to_stop_pct" in pos
        assert "risk_reward_ratio" in pos


class TestRiskContext:
    """Tests for get_risk_context."""

    def test_risk_context(self):
        """Risk context has correct structure."""
        provider = _make_provider()
        portfolio_state = {
            "positions": {
                "XRP": {"risk_usd": 10.0, "direction": "long"},
                "DOGE": {"risk_usd": 5.0, "direction": "short"},
            },
        }
        result = provider.get_risk_context(portfolio_state)
        assert result["total_risk_usd"] == 15.0
        assert result["position_count"] == 2
        assert "diversification_score" in result
        assert "XRP" in result["long_positions"]
        assert "DOGE" in result["short_positions"]


class TestDirectionalFeedback:
    """Tests for get_directional_feedback."""

    def test_directional_feedback(self):
        """Long/short stats are correct."""
        provider = _make_provider()
        trade_history = [
            {"direction": "long", "pnl": 10.0},
            {"direction": "long", "pnl": -5.0},
            {"direction": "short", "pnl": 3.0},
        ]
        result = provider.get_directional_feedback(trade_history)
        assert result["long"]["trades"] == 2
        assert result["long"]["wins"] == 1
        assert result["long"]["losses"] == 1
        assert result["short"]["trades"] == 1
        assert result["short"]["wins"] == 1


class TestGenerateSuggestions:
    """Tests for generate_suggestions."""

    def test_generate_suggestions(self):
        """Suggestions are generated."""
        provider = _make_provider()
        portfolio_state = {"positions": {}}
        market_regime = {"current_regime": "bullish"}
        result = provider.generate_suggestions(portfolio_state, market_regime)
        assert isinstance(result, list)
        assert any("Bullish regime" in s for s in result)


class TestGenerateEnhancedContext:
    """Tests for generate_enhanced_context."""

    def test_generate_enhanced_context(self):
        """Full context has all required keys."""
        provider = _make_provider()
        with patch.object(provider, "safe_file_read") as mock_read:
            mock_read.side_effect = lambda path, default: default
            with patch.object(
                provider, "get_market_regime_context", return_value={"current_regime": "NEUTRAL"}
            ):
                result = provider.generate_enhanced_context()
        assert "position_context" in result
        assert "market_regime" in result
        assert "performance_insights" in result
        assert "directional_feedback" in result
        assert "risk_context" in result
        assert "suggestions" in result
        assert "timestamp" in result


class TestPerformanceInsights:
    """Tests for get_performance_insights."""

    def test_performance_insights(self):
        """Performance insights return correct structure with trade history."""
        provider = _make_provider()
        trade_history = [
            {"symbol": "XRP", "pnl": 10.0},
            {"symbol": "XRP", "pnl": -5.0},
            {"symbol": "DOGE", "pnl": -3.0},
        ]
        portfolio_state = {"total_return": 5.0}
        result = provider.get_performance_insights(trade_history, portfolio_state)
        assert "insights" in result
        assert "coin_performance" in result
        assert result["coin_performance"]["XRP"]["trades"] == 2
        assert result["coin_performance"]["XRP"]["wins"] == 1
        assert result["coin_performance"]["DOGE"]["total_pnl"] == -3.0

    def test_performance_insights_empty_history(self):
        """Empty trade history returns fallback insight."""
        provider = _make_provider()
        result = provider.get_performance_insights([], {})
        assert result["insights"] == ["No trading history available"]


class TestMarketRegimeContext:
    """Tests for get_market_regime_context."""

    @patch("src.core.market_data.RealMarketData")
    def test_market_regime_context(self, mock_market_data_cls):
        """Market regime context returns correct structure."""
        mock_market_data = MagicMock()
        mock_market_data.available_coins = ["XRP"]
        mock_market_data.get_technical_indicators.return_value = {
            "current_price": 2.5,
            "ema_20": 2.3,
            "ema_50": 2.0,
        }
        mock_market_data_cls.return_value = mock_market_data
        provider = _make_provider()
        result = provider.get_market_regime_context()
        assert "current_regime" in result
        assert "regime_strength" in result
        assert "coin_regimes" in result
        assert result["coin_regimes"]["XRP"]["regime"] == "bullish"


class TestEmptyPositions:
    """Edge case: position context with empty positions dict."""

    def test_empty_positions(self):
        """Empty positions dict returns empty enhanced context."""
        provider = _make_provider()
        result = provider.get_enhanced_position_context({"positions": {}})
        assert result == {}

    def test_empty_positions_risk_context(self):
        """Empty positions dict returns zero risk and full diversification score."""
        provider = _make_provider()
        result = provider.get_risk_context({"positions": {}})
        assert result["total_risk_usd"] == 0
        assert result["position_count"] == 0
        assert result["diversification_score"] == 100.0


class TestBearishSuggestions:
    """Edge case: bearish regime with >=3 positions generates suggestion."""

    def test_bearish_suggestions(self):
        """Bearish regime with 3 positions generates a suggestion."""
        provider = _make_provider()
        portfolio_state = {
            "positions": {
                "XRP": {"direction": "long"},
                "DOGE": {"direction": "long"},
                "SOL": {"direction": "short"},
            },
        }
        market_regime = {"current_regime": "bearish"}
        result = provider.generate_suggestions(portfolio_state, market_regime)
        assert any("Bearish regime" in s for s in result)
