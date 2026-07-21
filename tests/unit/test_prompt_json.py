"""Tests for src/ai/prompt_json_builders.py."""

from datetime import datetime, timezone

import pytest

from src.ai.prompt_json_builders import (
    build_coin_state_vector,
    build_metadata_json,
    build_counter_trade_json,
    build_cooldown_status_json,
    build_position_slot_json,
    build_portfolio_json,
    build_risk_status_json,
    build_historical_context_json,
    build_directional_bias_json,
)


class TestBuildCoinStateVector:
    """Tests for build_coin_state_vector."""

    def test_build_coin_state_vector_minimal(self):
        result = build_coin_state_vector(
            coin="XRP",
            market_regime="BULLISH",
            sentiment=None,
            indicators_3m={"current_price": 2.5, "rsi_13": 55, "efficiency_ratio": 0.45},
            indicators_15m={},
            indicators_htf={},
        )

        assert result["coin"] == "XRP"
        assert "market_context" in result
        assert result["market_context"]["regime"] == "BULLISH"
        assert "technical_summary" in result
        assert "key_levels" in result
        assert "risk_profile" in result
        assert "sentiment" in result
        assert "position" in result
        assert result["position"] is None

    def test_build_coin_state_vector_with_position(self):
        class FakeExitPlan:
            profit_target = 3.0
            stop_loss = 2.0
            invalidation_condition = "CLOSE_BELOW_EMA20"

        class FakePosition:
            direction = "long"
            entry_price = 2.5
            current_price = 2.6
            unrealized_pnl = 0.1
            leverage = 10
            confidence = 0.75
            exit_plan = FakeExitPlan()
            erosion_status = "MINOR"
            peak_pnl = 0.15
            erosion_pct = 0.05

        result = build_coin_state_vector(
            coin="ETH",
            market_regime="NEUTRAL",
            sentiment={"funding_rate": 0.0001, "open_interest": 50000},
            indicators_3m={},
            indicators_15m={},
            indicators_htf={},
            position=FakePosition(),
        )

        assert result["coin"] == "ETH"
        assert result["position"] is not None
        assert result["position"]["direction"] == "long"
        assert result["position"]["entry_price"] == 2.5


class TestBuildMetadataJson:
    """Tests for build_metadata_json."""

    def test_build_metadata_json(self):
        now = datetime(2026, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = build_metadata_json(minutes_running=60, current_time=now, invocation_count=10)

        assert result["minutes_running"] == 60
        assert result["invocation_count"] == 10
        assert "2026" in result["current_time"]

    def test_build_metadata_json_string_time(self):
        result = build_metadata_json(
            minutes_running=5, current_time="2026-01-01T00:00:00Z", invocation_count=1
        )
        assert result["current_time"] == "2026-01-01T00:00:00Z"


class TestBuildCounterTradeJson:
    """Tests for build_counter_trade_json."""

    def test_build_counter_trade_json(self):
        all_indicators = {
            "XRP": {
                "3m": {
                    "current_price": 2.5,
                    "ema_20": 2.45,
                    "rsi_13": 55,
                    "volume": 100000,
                    "avg_volume": 80000,
                    "macd": 0.01,
                    "macd_signal": 0.005,
                },
                "15m": {
                    "current_price": 2.5,
                    "ema_20": 2.48,
                },
                "1h": {
                    "current_price": 2.5,
                    "ema_20": 2.42,
                },
            }
        }

        result = build_counter_trade_json(
            counter_trade_analysis="",
            all_indicators=all_indicators,
            available_coins=["XRP"],
            htf_interval="1h",
        )

        assert "XRP" in result
        assert "risk_level" in result["XRP"]
        assert "alignment_strength" in result["XRP"]
        assert "conditions_met" in result["XRP"]

    def test_build_counter_trade_json_missing_coin_gets_default(self):
        result = build_counter_trade_json(
            counter_trade_analysis="",
            all_indicators={},
            available_coins=["DOGE"],
            htf_interval="1h",
        )

        assert "DOGE" in result
        assert result["DOGE"]["risk_level"] == "CT_VERY_HIGH_RISK"


class TestBuildCooldownStatusJson:
    """Tests for build_cooldown_status_json."""

    def test_build_cooldown_status_json(self):
        result = build_cooldown_status_json(
            directional_cooldowns={"long": 2, "short": 0},
            coin_cooldowns={"XRP": 3, "DOGE": 1},
            counter_trend_cooldown=5,
            relaxed_countertrend_cycles=2,
        )

        assert result["directional_cooldowns"]["long"] == 2
        assert result["coin_cooldowns"]["XRP"] == 3
        assert result["counter_trend_cooldown"] == 5
        assert result["relaxed_countertrend_cycles"] == 2

    def test_build_cooldown_status_json_empty(self):
        result = build_cooldown_status_json(
            directional_cooldowns={},
            coin_cooldowns={},
            counter_trend_cooldown=0,
            relaxed_countertrend_cycles=0,
        )

        assert result["directional_cooldowns"] == {}
        assert result["coin_cooldowns"] == {}


class TestBuildPositionSlotJson:
    """Tests for build_position_slot_json."""

    def test_build_position_slot_json_empty(self):
        class FakePortfolio:
            positions = {}

        result = build_position_slot_json(
            portfolio_positions={},
            max_positions=5,
        )

        assert result["total_open"] == 0
        assert result["max_positions"] == 5
        assert result["available_slots"] == 5
        assert result["constraint_mode"] == "NORMAL"

    def test_build_position_slot_json_with_positions(self):
        class FakePos:
            direction = "long"
            unrealized_pnl = 5.0
            confidence = 0.8

        positions = {"XRP": FakePos(), "ETH": FakePos()}
        result = build_position_slot_json(portfolio_positions=positions, max_positions=5)

        assert result["total_open"] == 2
        assert result["long_slots_used"] == 2
        assert result["weakest_position"] is not None


class TestBuildPortfolioJson:
    """Tests for build_portfolio_json."""

    def test_build_portfolio_json(self):
        class FakePos:
            direction = "long"
            quantity = 100
            entry_price = 2.5
            current_price = 2.6
            unrealized_pnl = 10.0
            leverage = 10
            confidence = 0.75
            margin_usd = 25.0

        class FakePortfolio:
            total_return = 5.0
            current_balance = 200.0
            total_value = 210.0
            sharpe_ratio = 1.2
            positions = {"XRP": FakePos()}

        result = build_portfolio_json(FakePortfolio())

        assert "total_return_pct" in result
        assert "available_cash" in result
        assert "account_value" in result
        assert "sharpe_ratio" in result
        assert "positions" in result
        assert len(result["positions"]) == 1
        assert result["positions"][0]["symbol"] == "XRP"


class TestBuildRiskStatusJson:
    """Tests for build_risk_status_json."""

    def test_build_risk_status_json(self):
        class FakePos:
            margin_usd = 25.0

        class FakePortfolio:
            positions = {"XRP": FakePos()}
            current_balance = 175.0

        result = build_risk_status_json(FakePortfolio(), max_positions=5)

        assert result["current_positions_count"] == 1
        assert result["total_margin_used"] == 25.0
        assert result["available_cash"] == 175.0
        assert "trading_limits" in result
        assert result["trading_limits"]["max_positions"] == 5

    def test_build_risk_status_json_no_positions(self):
        class FakePortfolio:
            positions = {}
            current_balance = 200.0

        result = build_risk_status_json(FakePortfolio())
        assert result["current_positions_count"] == 0
        assert result["total_margin_used"] == 0


class TestBuildHistoricalContextJson:
    """Tests for build_historical_context_json."""

    def test_build_historical_context_json(self):
        context = {
            "total_cycles_analyzed": 50,
            "market_behavior": "BULLISH trend",
            "recent_decisions": ["buy XRP", "hold ETH"],
            "performance_trend": "improving",
        }

        result = build_historical_context_json(context)

        assert result["total_cycles_analyzed"] == 50
        assert result["market_behavior"] == "BULLISH trend"
        assert len(result["recent_decisions"]) == 2
        assert result["performance_trend"] == "improving"

    def test_build_historical_context_json_defaults(self):
        result = build_historical_context_json({})
        assert result["total_cycles_analyzed"] == 0
        assert result["market_behavior"] == "Unknown"
        assert result["recent_decisions"] == []


class TestBuildDirectionalBiasJson:
    """Tests for build_directional_bias_json."""

    def test_build_directional_bias_json(self):
        bias = {
            "long": {
                "net_pnl": 15.0,
                "trades": 10,
                "wins": 7,
                "losses": 3,
                "profitability_index": 70.0,
                "rolling_avg": 1.5,
                "consecutive_losses": 1,
                "consecutive_wins": 3,
                "caution_active": False,
            },
            "short": {
                "net_pnl": -5.0,
                "trades": 5,
                "wins": 2,
                "losses": 3,
                "profitability_index": 40.0,
                "rolling_avg": -1.0,
                "consecutive_losses": 2,
                "consecutive_wins": 0,
                "caution_active": True,
            },
        }

        result = build_directional_bias_json(bias)

        assert "long" in result
        assert "short" in result
        assert result["long"]["net_pnl"] == 15.0
        assert result["long"]["trades"] == 10
        assert result["short"]["caution_active"] is True
        assert result["long"]["caution_active"] is False

    def test_build_directional_bias_json_empty(self):
        result = build_directional_bias_json({})
        assert "long" in result
        assert "short" in result
        assert result["long"]["net_pnl"] == 0
        assert result["short"]["net_pnl"] == 0
