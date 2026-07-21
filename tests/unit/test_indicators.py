"""Unit tests for src/core/indicators.py"""

import polars as pl

from src.core.indicators import (
    _linear_slope,
    _sign,
    calculate_adx,
    calculate_atr_series,
    calculate_bollinger_bands,
    calculate_efficiency_ratio,
    calculate_ema_series,
    calculate_obv,
    calculate_rsi_series,
    calculate_supertrend,
    calculate_vwap,
    generate_smart_sparkline,
)


def test_calculate_rsi_series(sample_close_prices):
    rsi = calculate_rsi_series(sample_close_prices, period=14)
    assert isinstance(rsi, pl.Series)
    assert len(rsi) == len(sample_close_prices)
    non_nan = rsi.drop_nulls()
    valid = [v for v in non_nan.to_list() if v == v]  # filter NaN
    assert len(valid) > 0
    assert all(0 <= v <= 100 for v in valid)


def test_calculate_ema_series(sample_close_prices):
    ema = calculate_ema_series(sample_close_prices, period=20)
    assert isinstance(ema, pl.Series)
    assert len(ema) == len(sample_close_prices)


def _make_ohlcv(n=100, base_price=2.50):
    prices = [base_price + (i % 10) * 0.01 for i in range(n)]
    return pl.DataFrame(
        {
            "open": prices,
            "high": [p + 0.02 for p in prices],
            "low": [p - 0.02 for p in prices],
            "close": prices,
            "volume": [100000.0 + i * 1000 for i in range(n)],
        }
    )


def test_calculate_adx():
    ohlcv = _make_ohlcv()
    result = calculate_adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
    assert isinstance(result, tuple)
    assert len(result) == 3
    adx, plus_di, minus_di = result
    assert isinstance(adx, float)
    assert isinstance(plus_di, float)
    assert isinstance(minus_di, float)
    assert adx >= 0
    assert plus_di >= 0
    assert minus_di >= 0


def test_calculate_atr_series():
    ohlcv = _make_ohlcv()
    atr = calculate_atr_series(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
    assert isinstance(atr, pl.Series)
    assert len(atr) == len(ohlcv)
    non_nan = atr.drop_nulls()
    valid = [v for v in non_nan.to_list() if v == v]
    assert len(valid) > 0
    assert all(v > 0 for v in valid)


def test_calculate_bollinger_bands(sample_close_prices):
    result = calculate_bollinger_bands(sample_close_prices, period=20, std_dev=2.0)
    assert isinstance(result, tuple)
    assert len(result) == 5
    upper, middle, lower, bandwidth, percent_b = result
    assert upper >= lower
    assert middle >= lower
    assert upper >= middle
    assert isinstance(bandwidth, float)
    assert isinstance(percent_b, float)


def test_calculate_obv(sample_close_prices):
    volume = pl.Series("volume", [100000.0 + i * 1000 for i in range(len(sample_close_prices))])
    result = calculate_obv(sample_close_prices, volume)
    assert isinstance(result, tuple)
    assert len(result) == 3
    obv_val, trend, divergence = result
    assert isinstance(obv_val, float)
    assert trend in ("RISING", "FALLING", "FLAT")
    assert divergence in ("BULLISH", "BEARISH", "NONE")


def test_calculate_supertrend():
    ohlcv = _make_ohlcv()
    result = calculate_supertrend(
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        period=10,
        multiplier=3.0,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    line, direction = result
    assert isinstance(line, float)
    assert direction in ("UP", "DOWN")


def test_calculate_vwap():
    ohlcv = _make_ohlcv()
    vwap = calculate_vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], period=60)
    assert isinstance(vwap, float)
    assert vwap > 0


def test_calculate_efficiency_ratio(sample_close_prices):
    er = calculate_efficiency_ratio(sample_close_prices, period=10)
    assert isinstance(er, float)
    assert 0 <= er <= 1


def test_generate_smart_sparkline(sample_close_prices):
    result = generate_smart_sparkline(sample_close_prices, period=24)
    assert isinstance(result, dict)
    assert "structure" in result
    assert "momentum" in result
    assert "price_location" in result


def test_linear_slope():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 2.0, 4.0, 6.0, 8.0]
    slope = _linear_slope(x, y)
    assert isinstance(slope, float)
    assert slope == 2.0


def test_sign():
    assert _sign(5.0) == 1
    assert _sign(-3.0) == -1
    assert _sign(0.0) == 0


def test_rsi_period_too_large():
    prices = pl.Series("close", [10.0, 11.0, 12.0])
    rsi = calculate_rsi_series(prices, period=14)
    assert isinstance(rsi, pl.Series)
    assert len(rsi) == len(prices)


def test_ema_period_one():
    prices = pl.Series("close", [10.0, 20.0, 30.0, 40.0, 50.0])
    ema = calculate_ema_series(prices, period=1)
    assert isinstance(ema, pl.Series)
    assert len(ema) == len(prices)
    result = [v for v in ema.to_list() if v == v]
    expected = prices.to_list()
    assert result == expected


def test_adx_single_point():
    ohlcv = _make_ohlcv(n=20)
    result = calculate_adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
    assert isinstance(result, tuple)
    assert len(result) == 3
    adx, plus_di, minus_di = result
    assert isinstance(adx, float)
    assert isinstance(plus_di, float)
    assert isinstance(minus_di, float)


def test_bb_insufficient_data():
    prices = pl.Series("close", [10.0, 11.0, 12.0])
    result = calculate_bollinger_bands(prices, period=20, std_dev=2.0)
    assert isinstance(result, tuple)
    assert len(result) == 5
