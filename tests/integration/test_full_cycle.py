"""Integration tests for simplified full trading cycle persistence."""

import json
import os

import pytest

from src.utils import safe_file_read, safe_file_write


class TestStatePersistence:
    """Tests for portfolio state save/load cycle."""

    def test_state_save_load(self, temp_dir):
        """Save state and load it back."""
        state_path = os.path.join(temp_dir, "portfolio_state.json")
        state = {
            "current_balance": 1500.0,
            "positions": {},
            "total_value": 1500.0,
            "total_return": 0.0,
            "initial_balance": 1000.0,
            "trade_count": 5,
            "sharpe_ratio": 1.2,
            "coin_cooldowns": {"BTC": 2},
        }

        success = safe_file_write(state_path, state)
        assert success is True

        loaded = safe_file_read(state_path)
        assert loaded["current_balance"] == 1500.0
        assert loaded["trade_count"] == 5
        assert loaded["coin_cooldowns"] == {"BTC": 2}

    def test_state_save_load_with_positions(self, temp_dir):
        """State with positions persists correctly."""
        state_path = os.path.join(temp_dir, "state.json")
        state = {
            "current_balance": 800.0,
            "positions": {
                "ETH": {
                    "symbol": "ETH",
                    "direction": "long",
                    "entry_price": 2500.0,
                    "size": 0.1,
                    "leverage": 10,
                },
            },
            "trade_count": 1,
        }

        safe_file_write(state_path, state)
        loaded = safe_file_read(state_path)

        assert len(loaded["positions"]) == 1
        assert loaded["positions"]["ETH"]["entry_price"] == 2500.0


class TestTradeHistoryPersistence:
    """Tests for trade history file persistence."""

    def test_trade_history_persistence(self, temp_dir):
        """Trade history persists correctly across save/load."""
        history_path = os.path.join(temp_dir, "trade_history.json")
        history = [
            {
                "symbol": "XRP",
                "direction": "long",
                "entry_price": 2.50,
                "exit_price": 2.60,
                "pnl": 10.0,
                "entry_time": "2026-01-01T00:00:00Z",
                "exit_time": "2026-01-01T01:00:00Z",
            },
            {
                "symbol": "DOGE",
                "direction": "short",
                "entry_price": 0.15,
                "exit_price": 0.14,
                "pnl": 5.0,
                "entry_time": "2026-01-01T02:00:00Z",
                "exit_time": "2026-01-01T03:00:00Z",
            },
        ]

        safe_file_write(history_path, history)
        loaded = safe_file_read(history_path, default_data=[])

        assert len(loaded) == 2
        assert loaded[0]["symbol"] == "XRP"
        assert loaded[1]["pnl"] == 5.0

    def test_trade_history_append(self, temp_dir):
        """Trade history can be appended and reloaded."""
        history_path = os.path.join(temp_dir, "trade_history.json")
        trade = {
            "symbol": "SOL",
            "direction": "long",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "pnl": 15.0,
        }

        safe_file_write(history_path, [trade])
        existing = safe_file_read(history_path, default_data=[])
        existing.append(trade)
        safe_file_write(history_path, existing)

        loaded = safe_file_read(history_path, default_data=[])
        assert len(loaded) == 2
        assert loaded[0]["symbol"] == "SOL"
        assert loaded[1]["symbol"] == "SOL"


class TestCycleHistoryPersistence:
    """Tests for cycle history file persistence."""

    def test_cycle_history_persistence(self, temp_dir):
        """Cycle history persists correctly."""
        cycle_path = os.path.join(temp_dir, "cycle_history.json")
        cycles = [
            {
                "cycle": 1,
                "symbol": "XRP",
                "direction": "long",
                "pnl": 10.0,
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "cycle": 2,
                "symbol": "DOGE",
                "direction": "short",
                "pnl": -3.0,
                "timestamp": "2026-01-01T01:00:00Z",
            },
        ]

        safe_file_write(cycle_path, cycles)
        loaded = safe_file_read(cycle_path, default_data=[])

        assert len(loaded) == 2
        assert loaded[0]["cycle"] == 1
        assert loaded[1]["pnl"] == -3.0

    def test_cycle_history_empty_default(self, temp_dir):
        """Missing cycle history file returns empty list."""
        cycle_path = os.path.join(temp_dir, "nonexistent.json")
        loaded = safe_file_read(cycle_path, default_data=[])
        assert loaded == []
