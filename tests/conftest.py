"""Shared test fixtures for TradeSeeker test suite."""

import json
import os
import tempfile

import polars as pl
import pytest

from src.core import constants


@pytest.fixture
def sample_ohlcv():
    """Sample 3m OHLCV DataFrame with 100 candles."""
    n = 100
    base_price = 2.50
    prices = [base_price + (i % 10) * 0.01 for i in range(n)]
    base_ts = 1704067200000  # 2024-01-01T00:00:00Z in epoch ms
    return pl.DataFrame(
        {
            "timestamp": [base_ts + i * 180000 for i in range(n)],
            "open": prices,
            "high": [p + 0.02 for p in prices],
            "low": [p - 0.02 for p in prices],
            "close": prices,
            "volume": [100000.0 + i * 1000 for i in range(n)],
            "close_time": [base_ts + i * 180000 + 179999 for i in range(n)],
            "quote_asset_volume": [250000.0 + i * 2500 for i in range(n)],
            "number_of_trades": [500 + i * 10 for i in range(n)],
            "taker_buy_base_asset_volume": [50000.0 + i * 500 for i in range(n)],
            "taker_buy_quote_asset_volume": [125000.0 + i * 1250 for i in range(n)],
            "ignore": [0.0] * n,
        }
    )


@pytest.fixture
def sample_close_prices():
    """Sample close price series as Polars Series."""
    return pl.Series(
        "close",
        [2.50 + (i % 10) * 0.01 for i in range(100)],
    )


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_db():
    """Temporary SQLite database for DataEngine tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield db_path


@pytest.fixture
def sample_indicators():
    """Sample technical indicators dictionary."""
    return {
        "current_price": 2.50,
        "ema_20": 2.45,
        "ema_50": 2.40,
        "rsi_13": 55.0,
        "rsi_7": 58.0,
        "macd": 0.01,
        "macd_signal": 0.005,
        "macd_histogram": 0.005,
        "atr_14": 0.05,
        "adx": 25.0,
        "plus_di": 30.0,
        "minus_di": 20.0,
        "volume": 1000000.0,
        "avg_volume": 800000.0,
        "last_closed_volume": 900000.0,
        "volume_ratio": 1.125,
        "efficiency_ratio": 0.45,
        "vwap": 2.48,
        "vwap_distance_pct": 0.8,
        "price_vs_vwap": "ABOVE",
        "bb_upper": 2.60,
        "bb_lower": 2.40,
        "bb_bandwidth": 0.08,
        "bb_squeeze": False,
        "bb_signal": "NORMAL",
        "obv_trend": "UP",
        "obv_divergence": "NONE",
        "supertrend": 2.42,
        "supertrend_direction": "BULLISH",
        "price_slope_label": "MODERATE_ASCEND",
        "rsi_divergence_label": "NONE",
        "ema_stretch_label": "TIGHT",
        "volatility_pulse_label": "STAGNANT",
        "trend_strength_adx": "MODERATE",
    }


@pytest.fixture
def sample_portfolio_state():
    """Sample portfolio state dictionary."""
    return {
        "current_balance": 1000.0,
        "initial_balance": 1000.0,
        "total_return": 0.0,
        "positions": {},
        "trade_history": [],
        "cycle_history": [],
    }


@pytest.fixture
def sample_position():
    """Sample position dictionary."""
    return {
        "symbol": "XRP",
        "direction": "long",
        "entry_price": 2.50,
        "current_price": 2.55,
        "size": 100.0,
        "leverage": 10,
        "unrealized_pnl": 5.0,
        "entry_time": "2026-01-01T00:00:00Z",
        "exit_plan": {
            "profit_target": 2.60,
            "stop_loss": 2.45,
            "atr_stop_loss": 2.44,
        },
        "risk_usd": 10.0,
    }


@pytest.fixture
def sample_trade_history():
    """Sample trade history list."""
    return [
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
            "exit_price": 0.16,
            "pnl": -5.0,
            "entry_time": "2026-01-01T02:00:00Z",
            "exit_time": "2026-01-01T03:00:00Z",
        },
        {
            "symbol": "SOL",
            "direction": "long",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "pnl": 15.0,
            "entry_time": "2026-01-01T04:00:00Z",
            "exit_time": "2026-01-01T05:00:00Z",
        },
    ]


@pytest.fixture
def mock_binance_klines():
    """Mock Binance kline API response."""
    return [
        [
            1704067200000,  # timestamp
            "2.50",  # open
            "2.60",  # high
            "2.40",  # low
            "2.55",  # close
            "100000",  # volume
            1704067380000,  # close_time
            "250000",  # quote_asset_volume
            500,  # number_of_trades
            "50000",  # taker_buy_base
            "125000",  # taker_buy_quote
            "0",  # ignore
        ]
    ]


@pytest.fixture
def write_test_json(temp_dir):
    """Helper to write test JSON files."""

    def _write(filename, data):
        path = os.path.join(temp_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    return _write
