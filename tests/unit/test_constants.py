"""Tests for src/core/constants.py."""

from src.core import constants


def test_constants_exist():
    assert hasattr(constants, "FIB_3")
    assert hasattr(constants, "FIB_8")
    assert hasattr(constants, "FIB_13")
    assert hasattr(constants, "FIB_21")
    assert hasattr(constants, "FIB_55")
    assert hasattr(constants, "ADX_STRONG_THRESHOLD")
    assert hasattr(constants, "ADX_MODERATE_THRESHOLD")
    assert hasattr(constants, "ADX_WEAK_THRESHOLD")
    assert hasattr(constants, "MIN_KLINE_DATA_POINTS")
    assert hasattr(constants, "RSI_OVERBOUGHT")
    assert hasattr(constants, "RSI_OVERSOLD")
    assert hasattr(constants, "EMA_FAST")
    assert hasattr(constants, "MACD_FAST")
    assert hasattr(constants, "MACD_SLOW")
    assert hasattr(constants, "MACD_SIGNAL")


def test_fib_values():
    assert constants.FIB_3 == 3
    assert constants.FIB_8 == 8
    assert constants.FIB_13 == 13
    assert constants.FIB_21 == 21
    assert constants.FIB_55 == 55


def test_thresholds_in_range():
    assert 0 < constants.ADX_STRONG_THRESHOLD <= 100
    assert 0 < constants.ADX_MODERATE_THRESHOLD <= 100
    assert 0 < constants.ADX_WEAK_THRESHOLD <= 100

    assert 0 < constants.RSI_OVERBOUGHT <= 100
    assert 0 < constants.RSI_OVERSOLD <= 100

    assert (
        constants.ADX_STRONG_THRESHOLD
        > constants.ADX_MODERATE_THRESHOLD
        > constants.ADX_WEAK_THRESHOLD
    )

    assert 0 < constants.MAINTENANCE_MARGIN_RATE < 1
    assert 0 < constants.BB_SQUEEZE_THRESHOLD < 1
    assert constants.MIN_KLINE_DATA_POINTS > 0
    assert constants.CIRCUIT_BREAKER_THRESHOLD > 0
    assert constants.API_TIMEOUT_SECONDS > 0
