import numpy as np
import pandas as pd
from typing import Any

def calculate_ema_series(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()

def calculate_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    if len(prices) < period + 1:
        return pd.Series([np.nan] * len(prices))
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    rsi.loc[avg_gain == 0] = 0
    return rsi

def calculate_macd_series(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    if len(prices) < slow:
        return (
            pd.Series([np.nan] * len(prices)),
            pd.Series([np.nan] * len(prices)),
            pd.Series([np.nan] * len(prices)),
        )
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def calculate_atr_series(df_high: pd.Series, df_low: pd.Series, df_close: pd.Series, period: int = 14) -> pd.Series:
    if len(df_close) < period + 1:
        return pd.Series([np.nan] * len(df_close))
    tr0 = abs(df_high - df_low)
    tr1 = abs(df_high - df_close.shift())
    tr2 = abs(df_low - df_close.shift())
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, adjust=False).mean()
    return atr

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[float, float, float]:
    """
    Calculate ADX (Average Directional Index) and DI values.
    Returns: (adx, plus_di, minus_di)
    """
    if len(close) < period + 1:
        return 0.0, 0.0, 0.0

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / atr.replace(0, np.nan)).fillna(0)
    minus_di = 100 * (minus_dm_smooth / atr.replace(0, np.nan)).fillna(0)

    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    dx = 100 * (di_diff / di_sum.replace(0, np.nan)).fillna(0)
    adx = dx.ewm(span=period, adjust=False).mean()

    return float(adx.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 60) -> float:
    """
    Calculate Rolling VWAP (Volume Weighted Average Price).
    """
    if len(close) < period:
        return float(close.iloc[-1]) if len(close) > 0 else 0.0

    typical_price = (high + low + close) / 3
    tp_volume = typical_price * volume
    rolling_tp_vol = tp_volume.rolling(window=period).sum()
    rolling_vol = volume.rolling(window=period).sum()

    vwap = rolling_tp_vol / rolling_vol.replace(0, np.nan)
    return float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else float(close.iloc[-1])

def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[float, float, float, float, float]:
    """
    Calculate Bollinger Bands.
    Returns: (upper_band, middle_band, lower_band, bandwidth, percent_b)
    """
    if len(close) < period:
        price = float(close.iloc[-1]) if len(close) > 0 else 0.0
        return price, price, price, 0.0, 0.5

    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    bandwidth = ((upper - lower) / middle).fillna(0)
    band_range = upper - lower
    percent_b = ((close - lower) / band_range.replace(0, np.nan)).fillna(0.5)

    return (
        float(upper.iloc[-1]),
        float(middle.iloc[-1]),
        float(lower.iloc[-1]),
        float(bandwidth.iloc[-1]),
        float(percent_b.iloc[-1]),
    )

def calculate_obv(close: pd.Series, volume: pd.Series) -> tuple[float, str, str]:
    """
    Calculate On Balance Volume and its trend using vectorized operations.
    Returns: (obv, obv_trend, obv_divergence)
    """
    if len(close) < 10:
        return 0.0, "FLAT", "NONE"

    # Vectorized direction calculation: 1 if up, -1 if down, 0 if flat
    direction = np.sign(close.diff().fillna(0))
    # Fill the first element with 1 to match legacy behavior where OBV starts at 0 + volume[1]
    # actually, legacy behavior starts obv at 0. Let's just cumsum the volume adjusted by direction.
    direction.iloc[0] = 0 
    
    # Vectorized OBV calculation
    obv_series = (volume * direction).cumsum()
    current_obv = float(obv_series.iloc[-1])

    # Trend calculation
    obv_change = obv_series.iloc[-1] - obv_series.iloc[-10]
    obv_trend = "RISING" if obv_change > 0 else ("FALLING" if obv_change < 0 else "FLAT")

    # Divergence calculation
    price_change = close.iloc[-1] - close.iloc[-10]
    divergence = "NONE"
    if price_change > 0 and obv_change < 0:
        divergence = "BEARISH"
    elif price_change < 0 and obv_change > 0:
        divergence = "BULLISH"

    return current_obv, obv_trend, divergence

def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> tuple[float, str]:
    """
    Calculate SuperTrend indicator using optimized vectorization.
    Returns: (supertrend_line, direction)
    """
    if len(close) < period + 1:
        return float(close.iloc[-1]) if len(close) > 0 else 0.0, "UP"

    # Vectorized True Range and ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    # Initialize tracking arrays
    supertrend = np.zeros(len(close))
    direction = np.zeros(len(close), dtype=int)
    
    close_vals = close.values
    ub_vals = upper_band.values
    lb_vals = lower_band.values

    supertrend[0] = ub_vals[0]
    direction[0] = 1

    # Numba-style loop over arrays is significantly faster than pandas series access
    # Given the recursive nature of supertrend (depends on previous step), 
    # a pure pandas vectorized form is complex, but numpy array iteration 
    # provides near C-speed execution over O(N).
    for i in range(1, len(close_vals)):
        st_prev = supertrend[i - 1]
        if close_vals[i] > st_prev:
            supertrend[i] = lb_vals[i] if lb_vals[i] > st_prev or direction[i-1] == -1 else st_prev
            direction[i] = 1
        else:
            supertrend[i] = ub_vals[i] if ub_vals[i] < st_prev or direction[i-1] == 1 else st_prev
            direction[i] = -1

    current_st = float(supertrend[-1])
    current_dir = "UP" if direction[-1] == 1 else "DOWN"

    return current_st, current_dir

def calculate_efficiency_ratio(prices: pd.Series, period: int = 10) -> float:
    """
    Calculate Kaufman Efficiency Ratio (ER) to detect Choppy vs Trending markets.
    """
    if len(prices) < period + 1:
        return 0.5

    change = abs(prices.iloc[-1] - prices.iloc[-period - 1])
    volatility = prices.diff().abs().iloc[-period:].sum()

    if volatility == 0:
        return 1.0

    return change / volatility

def extract_semantic_features(prices: pd.Series, period: int = 24) -> dict[str, Any]:
    """Extract semantic features from price series using numpy"""
    if len(prices) < period:
        return {}

    subset = prices.iloc[-period:].values

    x = np.arange(len(subset))
    slope, _ = np.polyfit(x, subset, 1)
    slope_pct = (slope / subset[0]) * 100

    peaks = []
    valleys = []
    for i in range(1, len(subset) - 1):
        if subset[i] > subset[i - 1] and subset[i] > subset[i + 1]:
            peaks.append(float(subset[i]))
        elif subset[i] < subset[i - 1] and subset[i] < subset[i + 1]:
            valleys.append(float(subset[i]))

    std_dev = np.std(subset)
    mean_price = np.mean(subset)
    volatility_ratio = std_dev / mean_price

    volatility_state = "STABLE"
    if volatility_ratio > 0.02:
        volatility_state = "HIGH_VOLATILITY"
    elif volatility_ratio > 0.01:
        volatility_state = "EXPANDING"
    elif volatility_ratio < 0.005:
        volatility_state = "COMPRESSED"

    structure = "SIDEWAYS"
    if slope_pct > 0.05:
        structure = "UPTREND"
        if len(peaks) >= 2 and peaks[-1] < peaks[-2]:
            structure = "UPTREND_LOSING_MOMENTUM"
    elif slope_pct < -0.05:
        structure = "DOWNTREND"
        if len(valleys) >= 2 and valleys[-1] > valleys[-2]:
            structure = "DOWNTREND_LOSING_MOMENTUM"

    return {
        "slope": float(slope),
        "slope_pct": float(slope_pct),
        "peaks": peaks[-2:],
        "valleys": valleys[-2:],
        "volatility_state": volatility_state,
        "structure": structure,
    }

def generate_smart_sparkline(prices: pd.Series, period: int = 24) -> dict[str, Any]:
    """Generate Smart Sparkline v2.1 with key level, structure, and momentum using NumPy."""
    if len(prices) < period:
        return {"key_level": None, "structure": "UNCLEAR", "momentum": "STABLE"}

    subset = prices.iloc[-period:].values
    current_price = float(subset[-1])
    tolerance_pct = 0.005

    # Peak/Valley detection via SciPy concepts (numpy rolling comparisons)
    # Finding local maxima/minima with a window of 5 (2 before, 2 after)
    idx = np.arange(2, len(subset) - 2)
    is_peak = (subset[idx] > subset[idx-1]) & (subset[idx] > subset[idx-2]) & \
              (subset[idx] > subset[idx+1]) & (subset[idx] > subset[idx+2])
    is_valley = (subset[idx] < subset[idx-1]) & (subset[idx] < subset[idx-2]) & \
                (subset[idx] < subset[idx+1]) & (subset[idx] < subset[idx+2])
    
    peaks = subset[idx][is_peak].tolist()
    valleys = subset[idx][is_valley].tolist()

    key_level = None
    supports = [v for v in valleys if v < current_price]
    if supports:
        nearest_support = max(supports)
        strength = sum(1 for v in valleys if abs(v - nearest_support) / nearest_support < tolerance_pct)
        distance_pct = (current_price - nearest_support) / current_price * 100

        if distance_pct < 2.0:
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
            strength = sum(1 for p in peaks if abs(p - nearest_resistance) / nearest_resistance < tolerance_pct)
            distance_pct = (nearest_resistance - current_price) / current_price * 100

            if distance_pct < 2.0:
                key_level = {
                    "type": "resistance",
                    "level": round(nearest_resistance, 6),
                    "strength": min(strength, 5),
                    "distance_pct": round(distance_pct, 2),
                }

    structure = "RANGE"
    if len(peaks) >= 2 and len(valleys) >= 2:
        last_peaks = peaks[-2:]
        last_valleys = valleys[-2:]

        if last_peaks[1] > last_peaks[0] and last_valleys[1] > last_valleys[0]:
            structure = "HH_HL"
        elif last_peaks[1] < last_peaks[0] and last_valleys[1] < last_valleys[0]:
            structure = "LH_LL"
        else:
            price_range_val = np.ptp(subset) # Peak-to-peak (max - min)
            if price_range_val / current_price < 0.015:
                structure = "RANGE"

    mid = len(subset) // 2
    first_half_change = abs(subset[mid] - subset[0]) / subset[0] if subset[0] != 0 else 0
    second_half_change = abs(subset[-1] - subset[mid]) / subset[mid] if subset[mid] != 0 else 0

    if first_half_change > 0 and second_half_change > first_half_change * 1.3:
        momentum = "STRENGTHENING"
    elif first_half_change > 0 and second_half_change < first_half_change * 0.7:
        momentum = "WEAKENING"
    else:
        momentum = "STABLE"

    period_high = np.max(subset)
    period_low = np.min(subset)
    price_range = period_high - period_low

    percentile = ((current_price - period_low) / price_range) * 100 if price_range > 0 else 50
    zone = "LOWER_10" if percentile <= 10 else ("UPPER_10" if percentile >= 90 else "MIDDLE")

    return {
        "key_level": key_level,
        "structure": structure,
        "momentum": momentum,
        "price_location": {"zone": zone, "percentile": round(percentile, 0)},
    }

def calculate_pivots(df: pd.DataFrame, periods: int = 24) -> dict[str, float]:
    """Calculate High/Low pivots over N periods"""
    if len(df) < periods:
        return {}
    subset = df.iloc[-periods:]
    return {"high": float(subset["high"].max()), "low": float(subset["low"].min())}

def generate_tags(indicators: dict[str, Any]) -> list[str]:
    """Generate analytical tags based on indicators"""
    tags = []

    if indicators.get("volume_ratio", 0) > 1.5:
        tags.append("Vol_High")
    elif indicators.get("volume_ratio", 0) < 0.5:
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

    rsi = indicators.get("rsi_14", 50)
    if rsi > 70:
        tags.append("RSI_Overbought")
    elif rsi < 30:
        tags.append("RSI_Oversold")

    atr = indicators.get("atr_14", 0)
    if price > 0 and atr / price > 0.02:
        tags.append("High_Volatility")

    return tags
