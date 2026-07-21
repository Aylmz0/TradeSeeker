"""Unit tests for src/core/market_data.py — isolated tests (no API calls)."""

import polars as pl

from src.core.market_data import KLINE_COLUMNS, RealMarketData


def _make_df(rows=5, price=100.0, volume=1000.0):
    """Build a minimal valid kline DataFrame."""
    data = []
    for i in range(rows):
        data.append(
            {
                "timestamp": 1700000000000 + i * 60000,
                "open": price + i * 0.5,
                "high": price + i * 0.5 + 1.0,
                "low": price + i * 0.5 - 1.0,
                "close": price + i * 0.5 + 0.5,
                "volume": volume,
                "close_time": 1700000059999 + i * 60000,
                "quote_asset_volume": volume * price,
                "number_of_trades": 100,
                "taker_buy_base_asset_volume": volume * 0.5,
                "taker_buy_quote_asset_volume": volume * price * 0.5,
                "ignore": 0,
            }
        )
    return pl.DataFrame(data, schema=KLINE_COLUMNS, orient="row")


def test_build_empty_df():
    """Empty DataFrame has correct schema columns."""
    md = RealMarketData.__new__(RealMarketData)
    df = md._build_empty_df()
    assert df.is_empty()
    assert df.columns == KLINE_COLUMNS


def test_validate_kline_valid():
    """Valid kline data passes validation."""
    md = RealMarketData.__new__(RealMarketData)
    df = _make_df(rows=50)
    assert md._validate_kline_data(df, "XRPUSDT", "3m") is True


def test_validate_kline_empty():
    """Empty DataFrame fails validation."""
    md = RealMarketData.__new__(RealMarketData)
    df = pl.DataFrame(schema=KLINE_COLUMNS)
    assert md._validate_kline_data(df, "XRPUSDT", "3m") is False


def test_validate_kline_zero_volume():
    """Zero-volume data fails validation."""
    md = RealMarketData.__new__(RealMarketData)
    df = _make_df(rows=50, volume=0.0)
    assert md._validate_kline_data(df, "XRPUSDT", "3m") is False


def test_validate_kline_zero_price():
    """Zero-price data fails validation."""
    md = RealMarketData.__new__(RealMarketData)
    df = _make_df(rows=50, price=0.0)
    assert md._validate_kline_data(df, "XRPUSDT", "3m") is False


def test_calculate_max_drawdown():
    """Max drawdown calculation is correct for known input."""
    md = RealMarketData.__new__(RealMarketData)
    history = [100.0, 110.0, 90.0, 95.0, 80.0]
    # Peak goes 100→110, drawdown from 110→80 = 30/110 ≈ 0.2727
    result = md._calculate_max_drawdown(history)
    assert abs(result - 30.0 / 110.0) < 1e-6


def test_averaged_er_fallback():
    """get_averaged_er returns 1.0 when no data available."""
    md = RealMarketData.__new__(RealMarketData)
    md.preloaded_indicators = {}
    md._raw_dataframes = {}
    md.indicator_history_length = 10
    # Mock get_technical_indicators to return error
    md.get_technical_indicators = lambda coin, tf: {"error": "no data"}
    result = md.get_averaged_er("XRP")
    assert result == 1.0


def test_drawdown_same_prices():
    """All identical prices → drawdown is 0."""
    md = RealMarketData.__new__(RealMarketData)
    history = [50.0, 50.0, 50.0, 50.0, 50.0]
    result = md._calculate_max_drawdown(history)
    assert result == 0.0


def test_validate_negative_volume():
    """Negative volume with non-zero sum passes validation."""
    md = RealMarketData.__new__(RealMarketData)
    df = _make_df(rows=50, volume=-1000.0)
    assert md._validate_kline_data(df, "XRPUSDT", "3m") is True
