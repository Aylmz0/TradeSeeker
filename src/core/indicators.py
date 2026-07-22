from collections.abc import Sequence
from typing import Any

import polars as pl

from loguru import logger

from config.config import Config
from src.core import constants


def determine_trend(price: float, ema20: float) -> str:
    """Determine trend direction relative to EMA with a neutral band.

    Args:
        price: Current asset price.
        ema20: The EMA 20 value.

    Returns:
        String representing the trend direction ('BULLISH', 'BEARISH', or 'NEUTRAL').
    """
    if not isinstance(price, (int, float)) or not isinstance(ema20, (int, float)) or ema20 == 0:
        return "NEUTRAL"
    delta = (price - ema20) / ema20
    if abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT:
        return "NEUTRAL"
    return "BULLISH" if delta > 0 else "BEARISH"


def _linear_slope(x: Sequence[int | float], y: Sequence[int | float]) -> float:
    """Calculate the slope of a linear regression line.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        The slope of the best-fit line, or 0.0 if insufficient data.
    """
    n = len(x)
    if n < 2:
        return 0.0
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denominator


def _sign(x: float) -> int:
    """Return the sign of a number as an integer.

    Args:
        x: The number to check.

    Returns:
        1 if positive, -1 if negative, 0 if zero.
    """
    return 1 if x > 0 else (-1 if x < 0 else 0)


def calculate_ema_series(prices: pl.Series, period: int) -> pl.Series:
    """Calculate Exponential Moving Average for a price series.

    Args:
        prices: Series of price values.
        period: The lookback period for the EMA.

    Returns:
        Series containing the EMA values.
    """
    return prices.ewm_mean(span=period, adjust=False)


def calculate_rsi_series(prices: pl.Series, period: int = 14) -> pl.Series:
    """Calculate Relative Strength Index (RSI) for a price series.

    Args:
        prices: Series of close prices.
        period: The lookback period for RSI calculation.

    Returns:
        Series of RSI values between 0 and 100.
    """
    if len(prices) < period + 1:
        return pl.Series([float("nan")] * len(prices))

    df = pl.DataFrame({"close": prices})
    df = df.with_columns(pl.col("close").diff().alias("delta"))

    df = df.with_columns(
        [
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0.0).alias("loss"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("gain").ewm_mean(com=period - 1, adjust=False).alias("avg_gain"),
            pl.col("loss").ewm_mean(com=period - 1, adjust=False).alias("avg_loss"),
        ]
    )

    df = df.with_columns(
        (pl.col("avg_gain") / pl.col("avg_loss").replace(0, float("nan"))).alias("rs")
    )
    df = df.with_columns((100 - (100 / (1 + pl.col("rs")))).alias("rsi_raw"))
    df = df.with_columns(pl.col("rsi_raw").fill_nan(100).alias("rsi"))
    df = df.with_columns(
        pl.when(pl.col("avg_gain") == 0).then(0.0).otherwise(pl.col("rsi")).alias("rsi")
    )

    return df["rsi"]


def calculate_macd_series(
    prices: pl.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Calculate MACD (Moving Average Convergence Divergence) components.

    Args:
        prices: Series of close prices.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.

    Returns:
        Tuple of (macd_line, signal_line, histogram) series.
    """
    if len(prices) < slow:
        nan_series = pl.Series([float("nan")] * len(prices))
        return nan_series, nan_series, nan_series

    ema_fast = prices.ewm_mean(span=fast, adjust=False)
    ema_slow = prices.ewm_mean(span=slow, adjust=False)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm_mean(span=signal, adjust=False)
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram


def calculate_atr_series(
    df_high: pl.Series, df_low: pl.Series, df_close: pl.Series, period: int = 14
) -> pl.Series:
    """Calculate Average True Range (ATR) for price series.

    Args:
        df_high: Series of high prices.
        df_low: Series of low prices.
        df_close: Series of close prices.
        period: The lookback period for ATR smoothing.

    Returns:
        Series of ATR values.
    """
    if len(df_close) < period + 1:
        return pl.Series([float("nan")] * len(df_close))

    tr0 = (df_high - df_low).abs()
    tr1 = (df_high - df_close.shift()).abs()
    tr2 = (df_low - df_close.shift()).abs()

    temp_df = pl.DataFrame({"tr0": tr0, "tr1": tr1, "tr2": tr2})
    tr = temp_df.select(pl.max_horizontal("tr0", "tr1", "tr2")).to_series()

    return tr.ewm_mean(com=period - 1, adjust=False)


def calculate_adx(
    high: pl.Series, low: pl.Series, close: pl.Series, period: int = 14
) -> tuple[float, float, float]:
    """Calculate ADX (Average Directional Index) and directional indicators.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        period: The lookback period for smoothing.

    Returns:
        Tuple of (adx, plus_di, minus_di) values for the latest bar.
    """
    if len(close) < period + 1:
        return 0.0, 0.0, 0.0

    df = pl.DataFrame({"high": high, "low": low, "close": close})

    tr1 = pl.col("high") - pl.col("low")
    tr2 = (pl.col("high") - pl.col("close").shift(1)).abs()
    tr3 = (pl.col("low") - pl.col("close").shift(1)).abs()

    df = df.with_columns(pl.max_horizontal(tr1, tr2, tr3).alias("tr"))

    df = df.with_columns(
        [
            (pl.col("high") - pl.col("high").shift(1)).alias("up_move"),
            (pl.col("low").shift(1) - pl.col("low")).alias("down_move"),
        ]
    )

    df = df.with_columns(
        [
            pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
            .then(pl.col("up_move"))
            .otherwise(0.0)
            .alias("plus_dm"),
            pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
            .then(pl.col("down_move"))
            .otherwise(0.0)
            .alias("minus_dm"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("tr").ewm_mean(span=period, adjust=False).alias("atr"),
            pl.col("plus_dm").ewm_mean(span=period, adjust=False).alias("plus_dm_smooth"),
            pl.col("minus_dm").ewm_mean(span=period, adjust=False).alias("minus_dm_smooth"),
        ]
    )

    df = df.with_columns(
        [
            (100 * (pl.col("plus_dm_smooth") / pl.col("atr").replace(0, float("nan")))).alias(
                "plus_di"
            ),
            (100 * (pl.col("minus_dm_smooth") / pl.col("atr").replace(0, float("nan")))).alias(
                "minus_di"
            ),
        ]
    )

    df = df.with_columns(
        [
            pl.col("plus_di").fill_nan(0).alias("plus_di"),
            pl.col("minus_di").fill_nan(0).alias("minus_di"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("plus_di") + pl.col("minus_di")).alias("di_sum"),
            (pl.col("plus_di") - pl.col("minus_di")).abs().alias("di_diff"),
        ]
    )

    df = df.with_columns(
        (100 * (pl.col("di_diff") / pl.col("di_sum").replace(0, float("nan")))).alias("dx")
    )
    df = df.with_columns(pl.col("dx").fill_nan(0).alias("adx_raw"))
    df = df.with_columns(pl.col("adx_raw").ewm_mean(span=period, adjust=False).alias("adx"))

    return float(df["adx"][-1]), float(df["plus_di"][-1]), float(df["minus_di"][-1])


def calculate_vwap(
    high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series, period: int = 60
) -> float:
    """Calculate Rolling VWAP (Volume Weighted Average Price).

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        volume: Series of volume values.
        period: Rolling window size for VWAP calculation.

    Returns:
        The latest VWAP value.
    """
    if len(close) < period:
        return float(close[-1]) if len(close) > 0 else 0.0

    typical_price = (high + low + close) / 3
    tp_volume = typical_price * volume

    df = pl.DataFrame({"tp_vol": tp_volume, "vol": volume})
    df = df.with_columns(
        [
            pl.col("tp_vol").rolling_sum(window_size=period).alias("rolling_tp_vol"),
            pl.col("vol").rolling_sum(window_size=period).alias("rolling_vol"),
        ]
    )

    vwap = df.select(
        (pl.col("rolling_tp_vol") / pl.col("rolling_vol").replace(0, float("nan"))).alias("vwap")
    )["vwap"]

    last_val = vwap[-1]
    if last_val is not None and last_val == last_val:  # NaN check
        return float(last_val)
    return float(close[-1])


def calculate_bollinger_bands(
    close: pl.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[float, float, float, float, float]:
    """Calculate Bollinger Bands and related metrics.

    Args:
        close: Series of close prices.
        period: The lookback period for the moving average.
        std_dev: Number of standard deviations for the bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band, bandwidth, percent_b).
    """
    if len(close) < period:
        price = float(close[-1]) if len(close) > 0 else 0.0
        return price, price, price, 0.0, 0.5

    df = pl.DataFrame({"close": close})
    df = df.with_columns(
        [
            pl.col("close").rolling_mean(window_size=period).alias("middle"),
            pl.col("close").rolling_std(window_size=period).alias("std"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("middle") + (std_dev * pl.col("std"))).alias("upper"),
            (pl.col("middle") - (std_dev * pl.col("std"))).alias("lower"),
        ]
    )

    df = df.with_columns(
        [
            ((pl.col("upper") - pl.col("lower")) / pl.col("middle")).fill_nan(0).alias("bandwidth"),
            (
                (pl.col("close") - pl.col("lower"))
                / (pl.col("upper") - pl.col("lower")).replace(0, float("nan"))
            )
            .fill_nan(0.5)
            .alias("percent_b"),
        ]
    )

    return (
        float(df["upper"][-1]),
        float(df["middle"][-1]),
        float(df["lower"][-1]),
        float(df["bandwidth"][-1]),
        float(df["percent_b"][-1]),
    )


def calculate_obv(close: pl.Series, volume: pl.Series) -> tuple[float, str, str]:
    """Calculate On Balance Volume and its trend.

    Args:
        close: Series of close prices.
        volume: Series of volume values.

    Returns:
        Tuple of (obv_value, trend_direction, divergence_type).
    """
    if len(close) < constants.INDICATOR_HISTORY_DEFAULT:
        return 0.0, "FLAT", "NONE"

    df = pl.DataFrame({"close": close, "volume": volume})
    df = df.with_columns(pl.col("close").diff().fill_null(0).alias("delta"))

    direction = df["delta"].map_elements(_sign, return_dtype=pl.Int64).cast(pl.Float64)
    direction = direction.fill_null(0.0)
    direction[0] = 0.0

    df = df.with_columns((pl.col("volume") * pl.lit(direction)).cum_sum().alias("obv"))

    current_obv = float(df["obv"][-1])

    hist_len = constants.INDICATOR_HISTORY_DEFAULT
    obv_change = float(df["obv"][-1]) - float(df["obv"][-hist_len])
    obv_trend = "RISING" if obv_change > 0 else ("FALLING" if obv_change < 0 else "FLAT")

    price_change = float(close[-1]) - float(close[-hist_len])
    divergence = "NONE"
    if price_change > 0 and obv_change < 0:
        divergence = "BEARISH"
    elif price_change < 0 and obv_change > 0:
        divergence = "BULLISH"

    return current_obv, obv_trend, divergence


def calculate_supertrend(
    high: pl.Series, low: pl.Series, close: pl.Series, period: int = 10, multiplier: float = 3.0
) -> tuple[float, str]:
    """Calculate SuperTrend indicator.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        period: The lookback period for ATR calculation.
        multiplier: ATR multiplier for band width.

    Returns:
        Tuple of (supertrend_line_value, direction).
    """
    if len(close) < period + 1:
        return float(close[-1]) if len(close) > 0 else 0.0, "UP"

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    temp_df = pl.DataFrame({"tr0": tr1, "tr1": tr2, "tr2": tr3})
    tr = temp_df.select(pl.max_horizontal("tr0", "tr1", "tr2")).to_series()
    atr = tr.ewm_mean(span=period, adjust=False)

    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    n = len(close)
    supertrend = [0.0] * n
    direction = [0] * n

    close_vals = close.to_list()
    ub_vals = upper_band.to_list()
    lb_vals = lower_band.to_list()

    supertrend[0] = ub_vals[0]
    direction[0] = 1

    for i in range(1, n):
        st_prev = supertrend[i - 1]
        if close_vals[i] > st_prev:
            supertrend[i] = (
                lb_vals[i] if lb_vals[i] > st_prev or direction[i - 1] == -1 else st_prev
            )
            direction[i] = 1
        else:
            supertrend[i] = ub_vals[i] if ub_vals[i] < st_prev or direction[i - 1] == 1 else st_prev
            direction[i] = -1

    current_st = supertrend[-1]
    current_dir = "UP" if direction[-1] == 1 else "DOWN"

    return current_st, current_dir


def calculate_efficiency_ratio(prices: pl.Series, period: int = 10) -> float:
    """Calculate Kaufman Efficiency Ratio (ER) to detect Choppy vs Trending markets.

    Args:
        prices: Series of close prices.
        period: The lookback period for the calculation.

    Returns:
        Efficiency ratio between 0 and 1, where higher values indicate stronger trends.
    """
    if len(prices) < period + 1:
        return 0.5

    change = abs(float(prices[-1]) - float(prices[-period - 1]))
    volatility = prices.diff().abs()[-period:].sum()

    if volatility == 0:
        return 1.0

    return change / volatility


def extract_semantic_features(prices: pl.Series, period: int = 24) -> dict[str, Any]:
    """Extract semantic features from price series.

    Args:
        prices: Series of close prices.
        period: The lookback window for feature extraction.

    Returns:
        Dictionary containing slope, peaks, valleys, volatility_state, and structure.
    """
    if len(prices) < period:
        return {}

    subset = prices[-period:].to_list()
    current_price = subset[-1]

    x = [float(i) for i in range(len(subset))]
    slope = _linear_slope(x, [float(v) for v in subset])
    slope_pct = (slope / subset[0]) * 100 if subset[0] != 0 else 0

    peaks = []
    valleys = []
    for i in range(1, len(subset) - 1):
        if subset[i] > subset[i - 1] and subset[i] > subset[i + 1]:
            peaks.append(float(subset[i]))
        elif subset[i] < subset[i - 1] and subset[i] < subset[i + 1]:
            valleys.append(float(subset[i]))

    std_dev = prices[-period:].std()
    mean_price = prices[-period:].mean()
    std_val = float(std_dev) if isinstance(std_dev, (int, float)) else 0.0
    mean_val = float(mean_price) if isinstance(mean_price, (int, float)) else 0.0
    volatility_ratio = std_val / mean_val if mean_val != 0 else 0.0

    volatility_state = "STABLE"
    if volatility_ratio > constants.VOLATILITY_THRESHOLD:
        volatility_state = "HIGH_VOLATILITY"
    elif volatility_ratio > constants.VOLATILITY_THRESHOLD / 2:
        volatility_state = "EXPANDING"
    elif volatility_ratio < constants.VOLATILITY_THRESHOLD / 4:
        volatility_state = "COMPRESSED"

    structure = "SIDEWAYS"
    if slope_pct > constants.TREND_SLOPE_THRESHOLD:
        structure = "UPTREND"
        if len(peaks) >= constants.MIN_PEAKS_VALLEYS and peaks[-1] < peaks[-2]:
            structure = "UPTREND_LOSING_MOMENTUM"
    elif slope_pct < -constants.TREND_SLOPE_THRESHOLD:
        structure = "DOWNTREND"
        if len(valleys) >= constants.MIN_PEAKS_VALLEYS and valleys[-1] > valleys[-2]:
            structure = "DOWNTREND_LOSING_MOMENTUM"

    return {
        "slope": float(slope),
        "slope_pct": float(slope_pct),
        "peaks": peaks[-2:],
        "valleys": valleys[-2:],
        "volatility_state": volatility_state,
        "structure": structure,
    }


def calculate_slope_label(prices: pl.Series, period: int = 20) -> str:
    """Calculate linear regression slope and return categorical label.

    Args:
        prices: Series of close prices.
        period: The lookback period for slope calculation.

    Returns:
        Categorical label indicating trend strength and direction.
    """
    if len(prices) < period:
        return "FLAT"
    subset = prices[-period:].to_list()
    x = [float(i) for i in range(len(subset))]
    slope = _linear_slope(x, [float(v) for v in subset])
    slope_pct = (slope / subset[0]) * 100 if subset[0] != 0 else 0

    if slope_pct > 0.2:
        return "AGGRESSIVE_ASCEND"
    if slope_pct > 0.05:
        return "MODERATE_ASCEND"
    if slope_pct < -0.2:
        return "AGGRESSIVE_DESCEND"
    if slope_pct < -0.05:
        return "MODERATE_DESCEND"
    return "FLAT"


def calculate_ema_stretch_label(current_price: float, ema20: float) -> str:
    """Calculate how far price is from EMA20 and return categorical label.

    Args:
        current_price: The current price value.
        ema20: The 20-period EMA value.

    Returns:
        Categorical label indicating price extension from EMA.
    """
    if not current_price or not ema20 or ema20 == 0:
        return "NORMAL"
    diff_pct = (current_price - ema20) / ema20 * 100

    if abs(diff_pct) < 0.2:
        return "TIGHT"
    if diff_pct > 1.5:
        return "OVEREXTENDED_UP"
    if diff_pct < -1.5:
        return "OVEREXTENDED_DOWN"
    return "NORMAL"


def calculate_rsi_divergence_label(prices: pl.Series, rsi: pl.Series, period: int = 20) -> str:
    """Detect RSI-Price divergence using slope comparison.

    Args:
        prices: Series of close prices.
        rsi: Series of RSI values.
        period: The lookback period for divergence detection.

    Returns:
        Categorical label indicating divergence type (BULLISH_DIVERGENCE, BEARISH_DIVERGENCE, or NONE).
    """
    if len(prices) < period or len(rsi) < period:
        return "NONE"

    p_subset = prices[-period:].to_list()
    r_subset = rsi[-period:].to_list()
    x = list(range(len(p_subset)))

    p_slope = _linear_slope(x, p_subset)
    r_slope = _linear_slope(x, r_subset)

    if p_slope < -0.02 and r_slope > 0.5:
        return "BULLISH_DIVERGENCE"
    if p_slope > 0.02 and r_slope < -0.5:
        return "BEARISH_DIVERGENCE"

    return "NONE"


def calculate_volatility_pulse_label(atr_3: float, atr_14: float) -> str:
    """Compare short-term vs long-term ATR to detect volatility expansion.

    Args:
        atr_3: Short-term ATR value (e.g., 3-period).
        atr_14: Long-term ATR value (e.g., 14-period).

    Returns:
        Categorical label indicating volatility state (STRETCHING, STAGNANT, or NORMAL).
    """
    if not atr_3 or not atr_14 or atr_14 == 0:
        return "NORMAL"
    ratio = atr_3 / atr_14

    if ratio > 1.3:
        return "STRETCHING"
    if ratio < 0.7:
        return "STAGNANT"
    return "NORMAL"


def generate_smart_sparkline(prices: pl.Series, period: int = 24) -> dict[str, Any]:
    """Generate Smart Sparkline v2.1 with key level, structure, and momentum.

    Args:
        prices: Series of close prices.
        period: The lookback window for sparkline generation.

    Returns:
        Dictionary with key_level, structure, momentum, and price_location.
    """
    if len(prices) < period:
        return {"key_level": None, "structure": "UNCLEAR", "momentum": "STABLE"}

    subset = prices[-period:].to_list()
    current_price = subset[-1]
    tolerance_pct = constants.SPARKLINE_TOLERANCE

    idx = list(range(2, len(subset) - 2))
    peaks = []
    valleys = []
    for i in idx:
        if (
            subset[i] > subset[i - 1]
            and subset[i] > subset[i - 2]
            and subset[i] > subset[i + 1]
            and subset[i] > subset[i + 2]
        ):
            peaks.append(float(subset[i]))
        elif (
            subset[i] < subset[i - 1]
            and subset[i] < subset[i - 2]
            and subset[i] < subset[i + 1]
            and subset[i] < subset[i + 2]
        ):
            valleys.append(float(subset[i]))

    key_level = None
    supports = [v for v in valleys if v < current_price]
    if supports:
        nearest_support = max(supports)
        strength = sum(
            1 for v in valleys if abs(v - nearest_support) / nearest_support < tolerance_pct
        )
        distance_pct = (current_price - nearest_support) / current_price * 100

        if distance_pct < constants.LEVEL_PROXIMITY_THRESHOLD:
            key_level = {
                "type": "support",
                "level": round(nearest_support, 6),
                "strength": min(strength, 5),
                "distance_pct": round(distance_pct, 2),
            }

    if key_level is None:
        resistances = [p for p in peaks if p > current_price]
        if resistances:
            nearest_resistance = min(resistances)
            strength = sum(
                1 for p in peaks if abs(p - nearest_resistance) / nearest_resistance < tolerance_pct
            )
            distance_pct = (nearest_resistance - current_price) / current_price * 100

            if distance_pct < constants.LEVEL_PROXIMITY_THRESHOLD:
                key_level = {
                    "type": "resistance",
                    "level": round(nearest_resistance, 6),
                    "strength": min(strength, 5),
                    "distance_pct": round(distance_pct, 2),
                }

    structure = "RANGE"
    if len(peaks) >= constants.MIN_PEAKS_VALLEYS and len(valleys) >= constants.MIN_PEAKS_VALLEYS:
        last_peaks = peaks[-constants.MIN_PEAKS_VALLEYS :]
        last_valleys = valleys[-constants.MIN_PEAKS_VALLEYS :]

        if last_peaks[1] > last_peaks[0] and last_valleys[1] > last_valleys[0]:
            structure = "HH_HL"
        elif last_peaks[1] < last_peaks[0] and last_valleys[1] < last_valleys[0]:
            structure = "LH_LL"
        else:
            price_range_val = max(subset) - min(subset)
            if price_range_val / current_price < constants.RANGE_STRICT_THRESHOLD:
                structure = "RANGE"

    mid = len(subset) // 2
    raw_first_change = (subset[mid] - subset[0]) / subset[0] if subset[0] != 0 else 0
    raw_second_change = (subset[-1] - subset[mid]) / subset[mid] if subset[mid] != 0 else 0

    first_half_change = abs(raw_first_change)
    second_half_change = abs(raw_second_change)

    direction_changed = (raw_first_change * raw_second_change) < 0

    if direction_changed:
        momentum = "WEAKENING"
    elif (
        first_half_change > 0
        and second_half_change > first_half_change * constants.MOMENTUM_ACCELERATION_THRESHOLD
    ):
        momentum = "STRENGTHENING"
    elif (
        first_half_change > 0
        and second_half_change < first_half_change * constants.MOMENTUM_DECELERATION_THRESHOLD
    ):
        momentum = "WEAKENING"
    else:
        momentum = "STABLE"

    period_high = max(subset)
    period_low = min(subset)
    price_range = period_high - period_low

    percentile = ((current_price - period_low) / price_range) * 100 if price_range > 0 else 50
    zone = (
        "LOWER_10"
        if percentile <= constants.EXTREME_PERCENTILE_LOW
        else ("UPPER_10" if percentile >= constants.EXTREME_PERCENTILE_HIGH else "MIDDLE")
    )

    return {
        "key_level": key_level,
        "structure": structure,
        "momentum": momentum,
        "price_location": {"zone": zone, "percentile": round(percentile, 0)},
    }


def calculate_pivots(df: pl.DataFrame, periods: int = 24) -> dict[str, float]:
    """Calculate High/Low pivots over N periods.

    Args:
        df: DataFrame with 'high' and 'low' columns.
        periods: The lookback window for pivot calculation.

    Returns:
        Dictionary with 'high' and 'low' pivot values, or empty dict if insufficient data.
    """
    if len(df) < periods:
        return {}
    subset = df.tail(periods)
    high_val = subset["high"].max()
    low_val = subset["low"].min()
    return {
        "high": float(high_val) if isinstance(high_val, (int, float)) else 0.0,
        "low": float(low_val) if isinstance(low_val, (int, float)) else 0.0,
    }


def generate_tags(indicators: dict[str, Any]) -> list[str]:
    """Generate analytical tags based on indicators.

    Args:
        indicators: Dictionary containing calculated indicator values.

    Returns:
        List of categorical tags describing market conditions.
    """
    tags = []

    if indicators.get("volume_ratio", 0) > constants.VOLUME_RATIO_HIGH:
        tags.append("Vol_High")
    elif indicators.get("volume_ratio", 0) < constants.VOLUME_RATIO_LOW:
        tags.append("Vol_Low")

    price = indicators.get("current_price", 0)
    ema20 = indicators.get("ema_20", 0)
    ema50 = indicators.get("ema_50", 0)

    if price > ema20 > ema50:
        tags.append("Trend_Strong_Bull")
    elif price < ema20 < ema50:
        tags.append("Trend_Strong_Bear")
    elif price > ema20 and price < ema50:
        tags.append("Trend_Correction_Bull")
    elif price < ema20 and price > ema50:
        tags.append("Trend_Correction_Bear")

    rsi = indicators.get("rsi_13", 50)
    if rsi > constants.RSI_OVERBOUGHT:
        tags.append("RSI_Overbought")
    elif rsi < constants.RSI_OVERSOLD:
        tags.append("RSI_Oversold")

    atr = indicators.get("atr_14", 0)
    if price > 0 and atr / price > constants.VOLATILITY_THRESHOLD:
        tags.append("High_Volatility")

    return tags


def _compute_adx_series(
    df: pl.DataFrame, period: int = 14
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Compute full ADX/DI series (not just last bar) for ML feature extraction.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        period: EMA smoothing period for ADX calculation.

    Returns:
        Tuple of (adx_series, plus_di_series, minus_di_series).
    """
    if len(df) < period + 1:
        nan = pl.Series([float("nan")] * len(df))
        return nan, nan, nan

    adx_df = df.select(["high", "low", "close"])

    tr1 = pl.col("high") - pl.col("low")
    tr2 = (pl.col("high") - pl.col("close").shift(1)).abs()
    tr3 = (pl.col("low") - pl.col("close").shift(1)).abs()
    adx_df = adx_df.with_columns(pl.max_horizontal(tr1, tr2, tr3).alias("tr"))

    adx_df = adx_df.with_columns(
        [
            (pl.col("high") - pl.col("high").shift(1)).alias("up_move"),
            (pl.col("low").shift(1) - pl.col("low")).alias("down_move"),
        ]
    )

    adx_df = adx_df.with_columns(
        [
            pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
            .then(pl.col("up_move"))
            .otherwise(0.0)
            .alias("plus_dm"),
            pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
            .then(pl.col("down_move"))
            .otherwise(0.0)
            .alias("minus_dm"),
        ]
    )

    adx_df = adx_df.with_columns(pl.col("tr").ewm_mean(span=period, adjust=False).alias("atr_ewm"))
    adx_df = adx_df.with_columns(pl.col("atr_ewm").replace(0, float("nan")).alias("atr_ewm_safe"))

    adx_df = adx_df.with_columns(
        [
            (
                100
                * (pl.col("plus_dm").ewm_mean(span=period, adjust=False) / pl.col("atr_ewm_safe"))
            ).alias("plus_di"),
            (
                100
                * (pl.col("minus_dm").ewm_mean(span=period, adjust=False) / pl.col("atr_ewm_safe"))
            ).alias("minus_di"),
        ]
    )

    adx_df = adx_df.with_columns(
        [
            (pl.col("plus_di") + pl.col("minus_di")).alias("di_sum"),
            (pl.col("plus_di") - pl.col("minus_di")).abs().alias("di_diff"),
        ]
    )

    adx_df = adx_df.with_columns(
        (100 * (pl.col("di_diff") / pl.col("di_sum").replace(0, float("nan")))).alias("dx")
    )

    adx_series = adx_df["dx"].fill_nan(0).ewm_mean(span=period, adjust=False)
    plus_di_series = adx_df["plus_di"].fill_nan(0)
    minus_di_series = adx_df["minus_di"].fill_nan(0)

    return adx_series, plus_di_series, minus_di_series


def _compute_momentum_features(df: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Compute momentum indicators: RSI, MACD, and their slopes.

    Args:
        df: Raw OHLCV DataFrame.
        features: Partial features DataFrame.

    Returns:
        Features DataFrame with momentum columns added.
    """
    features = features.with_columns(
        [
            calculate_rsi_series(df["close"], constants.ATR_PERIOD_DEFAULT).alias("rsi_14"),
            calculate_rsi_series(df["close"], constants.FIB_8).alias("rsi_7"),
        ]
    )

    macd_line, _macd_signal, macd_hist = calculate_macd_series(df["close"])
    features = features.with_columns(
        [
            macd_hist.alias("macd_hist"),
            macd_line.alias("macd_line"),
        ]
    )

    features = features.with_columns(
        [
            (pl.col("rsi_14") - pl.col("rsi_14").shift(1)).alias("rsi_14_slope"),
            (pl.col("rsi_7") - pl.col("rsi_7").shift(1)).alias("rsi_7_slope"),
        ]
    )

    features = features.with_columns(
        [
            pl.col("macd_hist").shift(1).alias("macd_hist_lag1"),
            pl.col("macd_hist").shift(2).alias("macd_hist_lag2"),
        ]
    )
    features = features.with_columns(
        (pl.col("macd_hist") - pl.col("macd_hist_lag1")).alias("macd_slope")
    )

    return features


def _compute_trend_features(df: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Compute trend indicators: EMA, ADX, DI cross, alignment.

    Args:
        df: Raw OHLCV DataFrame.
        features: Partial features DataFrame.

    Returns:
        Features DataFrame with trend columns added.
    """
    features = features.with_columns(
        [
            calculate_ema_series(df["close"], constants.EMA_FAST).alias("ema_20"),
            calculate_ema_series(df["close"], constants.EMA_MEDIUM).alias("ema_50"),
        ]
    )
    features = features.with_columns(
        ((df["close"] - pl.col("ema_20")) / pl.col("ema_20")).alias("ema_20_dist")
    )
    features = features.with_columns(
        ((df["close"] - pl.col("ema_50")) / pl.col("ema_50")).alias("ema_50_dist")
    )
    features = features.with_columns(
        (pl.col("ema_20_dist") - pl.col("ema_50_dist")).alias("ema_alignment")
    )
    features = features.with_columns(
        [
            ((pl.col("ema_20") - pl.col("ema_20").shift(1)) / pl.col("ema_20").shift(1)).alias(
                "ema_20_slope"
            ),
            ((pl.col("ema_50") - pl.col("ema_50").shift(1)) / pl.col("ema_50").shift(1)).alias(
                "ema_50_slope"
            ),
        ]
    )

    adx_s, plus_di_s, minus_di_s = _compute_adx_series(df)
    features = features.with_columns(
        [
            adx_s.alias("adx_14"),
            plus_di_s.alias("plus_di"),
            minus_di_s.alias("minus_di"),
        ]
    )

    features = features.with_columns(
        [
            pl.col("plus_di").shift(1).alias("plus_di_lag1"),
            pl.col("minus_di").shift(1).alias("minus_di_lag1"),
        ]
    )
    features = features.with_columns((pl.col("plus_di") - pl.col("minus_di")).alias("di_cross"))
    features = features.with_columns(
        [
            pl.col("di_cross").shift(1).alias("di_cross_lag1"),
            pl.col("adx_14").shift(1).alias("adx_14_lag1"),
        ]
    )
    features = features.with_columns((pl.col("adx_14") - pl.col("adx_14_lag1")).alias("adx_growth"))

    return features


def _compute_volatility_features(df: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Compute volatility indicators: ATR and Bollinger Bands.

    Args:
        df: Raw OHLCV DataFrame.
        features: Partial features DataFrame.

    Returns:
        Features DataFrame with volatility columns added.
    """
    features = features.with_columns(
        [
            calculate_atr_series(
                df["high"], df["low"], df["close"], constants.ATR_PERIOD_DEFAULT
            ).alias("atr_14"),
        ]
    )
    features = features.with_columns((pl.col("atr_14") / df["close"]).alias("atr_ratio"))

    bb_df = pl.DataFrame({"close": df["close"]})
    bb_df = bb_df.with_columns(
        [
            pl.col("close").rolling_mean(window_size=constants.EMA_FAST).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=constants.EMA_FAST).alias("bb_std"),
        ]
    )
    bb_df = bb_df.with_columns(
        [
            (pl.col("bb_middle") + (constants.BB_STD_DEV_NORMAL * pl.col("bb_std"))).alias(
                "bb_upper"
            ),
            (pl.col("bb_middle") - (constants.BB_STD_DEV_NORMAL * pl.col("bb_std"))).alias(
                "bb_lower"
            ),
        ]
    )

    features = features.with_columns(
        [
            ((bb_df["bb_upper"] - bb_df["bb_lower"]) / bb_df["bb_middle"])
            .fill_nan(0)
            .alias("bb_bandwidth"),
            ((df["close"] - bb_df["bb_lower"]) / (bb_df["bb_upper"] - bb_df["bb_lower"]))
            .fill_nan(0)
            .alias("bb_percent_b"),
        ]
    )

    return features


def _compute_derived_features(df: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """Compute derived features: returns, ROC, and lag columns.

    Args:
        df: Raw OHLCV DataFrame.
        features: Partial features DataFrame with base indicators.

    Returns:
        Features DataFrame with derived columns added.
    """
    features = features.with_columns(
        [
            df["close"].pct_change().alias("return_1p"),
            df["volume"].pct_change().alias("return_vol"),
        ]
    )

    features = features.with_columns(
        [
            df["close"].pct_change(n=constants.INDICATOR_HISTORY_DEFAULT).alias("roc_10"),
        ]
    )

    cols_to_lag = [
        "return_1p",
        "return_vol",
        "rsi_14",
        "rsi_7",
        "macd_hist",
        "bb_bandwidth",
        "di_cross",
        "adx_14",
        "rsi_14_slope",
    ]
    for col in cols_to_lag:
        if f"{col}_lag1" not in features.columns:
            features = features.with_columns(pl.col(col).shift(1).alias(f"{col}_lag1"))
        if f"{col}_lag2" not in features.columns:
            features = features.with_columns(pl.col(col).shift(2).alias(f"{col}_lag2"))

    features = features.fill_null(strategy="forward")
    features = features.fill_nan(None)
    features = features.drop_nulls()

    return features


def get_features_for_ml(df: pl.DataFrame) -> pl.DataFrame:
    """Convert raw OHLCV DataFrame into an ML-ready Feature Matrix.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume.

    Returns:
        DataFrame with engineered features for ML model input.
    """
    if len(df) < constants.MIN_KLINE_DATA_POINTS:
        return pl.DataFrame()

    features = pl.DataFrame({"timestamp": df["timestamp"]})

    features = _compute_momentum_features(df, features)
    features = _compute_trend_features(df, features)
    features = _compute_volatility_features(df, features)
    features = _compute_derived_features(df, features)

    return features


if __name__ == "__main__":
    import sqlite3

    logger.info("--- Testing ML Feature Extraction ---")
    try:
        conn = sqlite3.connect("data/market_data.db")
        query = "SELECT * FROM market_data WHERE coin='XRP' AND interval='15m' ORDER BY timestamp DESC LIMIT 500"
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logger.error("No raw data in DB. Run Phase 1.2 first.")
        else:
            df_raw = pl.DataFrame(rows, schema=columns)
            df_raw = df_raw.sort("timestamp").with_row_index()
            df_raw = df_raw.drop("index")
            df_raw = df_raw.with_columns(
                pl.col("timestamp").cast(pl.Datetime(time_unit="ms")).alias("timestamp")
            )

            df_features = get_features_for_ml(df_raw)
            logger.success("Extracted features from {} raw candles.", len(df_raw))
            logger.info(
                "Generated {} ML rows, {} features.", df_features.height, len(df_features.columns)
            )
            logger.info("Last Row Sample: {}", df_features.tail(1))
    except Exception as e:
        logger.error("Test Failed: {}", e)
