"""Tests for Pydantic schemas in src/schemas/."""

import pytest
from pydantic import ValidationError

from src.schemas.config import Settings
from src.schemas.position import ExitPlan, Position, TrendContext, TrailingMeta
from src.schemas.trade import TradeHistoryEntry


class TestPosition:
    def test_position_frozen(self):
        pos = Position(symbol="XRP", direction="long")
        with pytest.raises(ValidationError):
            pos.symbol = "DOGE"

    def test_position_required_fields(self):
        with pytest.raises(ValidationError):
            Position()

        with pytest.raises(ValidationError):
            Position(symbol="XRP")

        with pytest.raises(ValidationError):
            Position(direction="long")

    def test_position_defaults(self):
        pos = Position(symbol="XRP", direction="long")
        assert pos.quantity == 0.0
        assert pos.entry_price == 0.0
        assert pos.leverage == 10
        assert pos.confidence == 0.0
        assert pos.exit_plan == ExitPlan()
        assert pos.trailing == TrailingMeta()
        assert pos.trend_context == TrendContext()
        assert pos.partial_exit_flags == {}

    def test_exit_plan_frozen(self):
        ep = ExitPlan(stop_loss=2.45)
        with pytest.raises(ValidationError):
            ep.stop_loss = 2.40


class TestTradeHistoryEntry:
    def test_trade_history_entry(self):
        entry = TradeHistoryEntry(
            symbol="XRP",
            direction="long",
            entry_price=2.50,
            exit_price=2.60,
            quantity=100.0,
        )
        assert entry.symbol == "XRP"
        assert entry.direction == "long"
        assert entry.entry_price == 2.50
        assert entry.exit_price == 2.60
        assert entry.quantity == 100.0
        assert entry.pnl == 0.0
        assert entry.leverage == 10

    def test_trade_history_entry_required_fields(self):
        with pytest.raises(ValidationError):
            TradeHistoryEntry()

    def test_trade_history_entry_frozen(self):
        entry = TradeHistoryEntry(
            symbol="XRP",
            direction="long",
            entry_price=2.50,
            exit_price=2.60,
            quantity=100.0,
        )
        with pytest.raises(ValidationError):
            entry.symbol = "DOGE"


class TestSettings:
    def test_config_defaults(self):
        defaults = Settings.model_fields
        assert defaults["PRIMARY_AI_PROVIDER"].default == "openrouter"
        assert defaults["DEBUG"].default is False
        assert defaults["LOG_LEVEL"].default == "INFO"
        assert defaults["INITIAL_BALANCE"].default == 200.0
        assert defaults["CYCLE_INTERVAL_MINUTES"].default == 2
        assert defaults["MAX_LEVERAGE"].default == 20
        assert defaults["MIN_CONFIDENCE"].default == 0.60
        assert defaults["RISK_LEVEL"].default == "medium"
        assert defaults["TRADING_MODE"].default == "simulation"
        assert defaults["BINANCE_TESTNET"].default is False
        assert defaults["BINANCE_MARGIN_TYPE"].default == "ISOLATED"
        assert defaults["BINANCE_DEFAULT_LEVERAGE"].default == 10
        assert defaults["MAX_POSITIONS"].default == 5
        assert defaults["USE_SMART_CACHE"].default is True
        assert defaults["FLASH_EXIT_ENABLED"].default is True

    def test_config_validators(self):
        with pytest.raises(ValidationError):
            Settings(LOG_LEVEL="INVALID")

        with pytest.raises(ValidationError):
            Settings(TRADING_MODE="paper")

        with pytest.raises(ValidationError):
            Settings(RISK_LEVEL="extreme")

        with pytest.raises(ValidationError):
            Settings(HTF_INTERVAL="5m")

        with pytest.raises(ValidationError):
            Settings(BINANCE_MARGIN_TYPE="CROSS")
