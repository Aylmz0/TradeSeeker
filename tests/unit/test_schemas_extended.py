"""Tests for Pydantic schemas not covered in test_schemas.py."""

import pytest
from pydantic import ValidationError

from src.schemas.position import TrailingMeta, TrendContext
from src.schemas.trade import CycleHistoryEntry, DirectionalBias


class TestTrailingMeta:
    def test_creation_defaults(self):
        meta = TrailingMeta()
        assert meta.active is False
        assert meta.stop_price is None
        assert meta.highest_pnl is None
        assert meta.lowest_pnl is None
        assert meta.last_update_cycle is None
        assert meta.last_reason is None
        assert meta.last_stop is None
        assert meta.progress_percent is None
        assert meta.time_in_trade_min is None
        assert meta.last_volume_ratio is None

    def test_creation_with_values(self):
        meta = TrailingMeta(
            active=True,
            stop_price=2.50,
            highest_pnl=120.5,
            lowest_pnl=-15.0,
            last_update_cycle=42,
            last_reason="pnl_threshold",
            last_stop=2.45,
            progress_percent=65.3,
            time_in_trade_min=18.5,
            last_volume_ratio=1.8,
        )
        assert meta.active is True
        assert meta.stop_price == 2.50
        assert meta.highest_pnl == 120.5
        assert meta.lowest_pnl == -15.0
        assert meta.last_update_cycle == 42
        assert meta.last_reason == "pnl_threshold"
        assert meta.last_stop == 2.45
        assert meta.progress_percent == 65.3
        assert meta.time_in_trade_min == 18.5
        assert meta.last_volume_ratio == 1.8

    def test_frozen(self):
        meta = TrailingMeta(active=True, stop_price=2.50)
        with pytest.raises(ValidationError):
            meta.active = False
        with pytest.raises(ValidationError):
            meta.stop_price = 3.00


class TestTrendContext:
    def test_creation_defaults(self):
        ctx = TrendContext()
        assert ctx.trend_at_entry is None
        assert ctx.alignment is None
        assert ctx.cycle is None

    def test_creation_with_values(self):
        ctx = TrendContext(
            trend_at_entry="uptrend",
            alignment="strong_bullish",
            cycle=7,
        )
        assert ctx.trend_at_entry == "uptrend"
        assert ctx.alignment == "strong_bullish"
        assert ctx.cycle == 7

    def test_frozen(self):
        ctx = TrendContext(trend_at_entry="downtrend")
        with pytest.raises(ValidationError):
            ctx.trend_at_entry = "sideways"


class TestDirectionalBias:
    def test_creation_defaults(self):
        bias = DirectionalBias()
        assert bias.net_pnl == 0.0
        assert bias.trades == 0
        assert bias.wins == 0
        assert bias.losses == 0
        assert bias.consecutive_losses == 0
        assert bias.consecutive_wins == 0
        assert bias.caution_active is False
        assert bias.caution_win_progress == 0
        assert bias.loss_streak_loss_usd == 0.0

    def test_creation_with_values(self):
        bias = DirectionalBias(
            net_pnl=350.25,
            trades=20,
            wins=12,
            losses=8,
            consecutive_losses=3,
            consecutive_wins=5,
            caution_active=True,
            caution_win_progress=2,
            loss_streak_loss_usd=-45.0,
        )
        assert bias.net_pnl == 350.25
        assert bias.trades == 20
        assert bias.wins == 12
        assert bias.losses == 8
        assert bias.consecutive_losses == 3
        assert bias.consecutive_wins == 5
        assert bias.caution_active is True
        assert bias.caution_win_progress == 2
        assert bias.loss_streak_loss_usd == -45.0

    def test_frozen(self):
        bias = DirectionalBias(net_pnl=100.0, trades=5)
        with pytest.raises(ValidationError):
            bias.net_pnl = 200.0
        with pytest.raises(ValidationError):
            bias.trades = 10


class TestCycleHistoryEntry:
    def test_creation_defaults(self):
        entry = CycleHistoryEntry(cycle=1)
        assert entry.cycle == 1
        assert entry.timestamp == ""
        assert entry.user_prompt_summary == ""
        assert entry.chain_of_thoughts == ""
        assert entry.decisions == {}
        assert entry.status == "idle"
        assert entry.cooldown_status == {}
        assert entry.metadata is None

    def test_creation_with_values(self):
        entry = CycleHistoryEntry(
            cycle=5,
            timestamp="2025-07-21T10:30:00Z",
            user_prompt_summary="Check BTC momentum",
            chain_of_thoughts="BTC is trending up with volume confirmation",
            decisions={"action": "open_long", "symbol": "BTC"},
            status="ai_decision",
            cooldown_status={"active": False},
            metadata={"model": "gpt-4", "latency_ms": 1200},
        )
        assert entry.cycle == 5
        assert entry.timestamp == "2025-07-21T10:30:00Z"
        assert entry.user_prompt_summary == "Check BTC momentum"
        assert entry.chain_of_thoughts == "BTC is trending up with volume confirmation"
        assert entry.decisions == {"action": "open_long", "symbol": "BTC"}
        assert entry.status == "ai_decision"
        assert entry.cooldown_status == {"active": False}
        assert entry.metadata == {"model": "gpt-4", "latency_ms": 1200}

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            CycleHistoryEntry()

    def test_frozen(self):
        entry = CycleHistoryEntry(cycle=1)
        with pytest.raises(ValidationError):
            entry.cycle = 2
        with pytest.raises(ValidationError):
            entry.status = "error"

    def test_status_values(self):
        for status in ("ai_decision", "tp_sl_only", "manual_override", "error", "idle"):
            entry = CycleHistoryEntry(cycle=1, status=status)
            assert entry.status == status
