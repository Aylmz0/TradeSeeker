from typing import Any

import numpy as np
import pandas as pd

from src.core import constants


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


def calculate_macd_series(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
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


def calculate_atr_series(
    df_high: pd.Series, df_low: pd.Series, df_close: pd.Series, period: int = 14
) -> pd.Series:
    if len(df_close) < period + 1:
        return pd.Series([np.nan] * len(df_close))
    tr0 = abs(df_high - df_low)
    tr1 = abs(df_high - df_close.shift())
    tr2 = abs(df_low - df_close.shift())
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[float, float, float]:
    """Calculate ADX (Average Directional Index) and DI values.
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


def calculate_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 60
) -> float:
    """Calculate Rolling VWAP (Volume Weighted Average Price)."""
    if len(close) < period:
        return float(close.iloc[-1]) if len(close) > 0 else 0.0

    typical_price = (high + low + close) / 3
    tp_volume = typical_price * volume
    rolling_tp_vol = tp_volume.rolling(window=period).sum()
    rolling_vol = volume.rolling(window=period).sum()

    vwap = rolling_tp_vol / rolling_vol.replace(0, np.nan)
    return float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else float(close.iloc[-1])


def calculate_bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[float, float, float, float, float]:
    """Calculate Bollinger Bands.
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
    """Calculate On Balance Volume and its trend using vectorized operations.
    Returns: (obv, obv_trend, obv_divergence)
    """
    if len(close) < constants.INDICATOR_HISTORY_DEFAULT:
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
    obv_change = obv_series.iloc[-1] - obv_series.iloc[-constants.INDICATOR_HISTORY_DEFAULT]
    obv_trend = "RISING" if obv_change > 0 else ("FALLING" if obv_change < 0 else "FLAT")

    # Divergence calculation
    price_change = close.iloc[-1] - close.iloc[-constants.INDICATOR_HISTORY_DEFAULT]
    divergence = "NONE"
    if price_change > 0 and obv_change < 0:
        divergence = "BEARISH"
    elif price_change < 0 and obv_change > 0:
        divergence = "BULLISH"

    return current_obv, obv_trend, divergence


def calculate_supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0
) -> tuple[float, str]:
    """Calculate SuperTrend indicator using optimized vectorization.
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
            supertrend[i] = (
                lb_vals[i] if lb_vals[i] > st_prev or direction[i - 1] == -1 else st_prev
            )
            direction[i] = 1
        else:
            supertrend[i] = ub_vals[i] if ub_vals[i] < st_prev or direction[i - 1] == 1 else st_prev
            direction[i] = -1

    current_st = float(supertrend[-1])
    current_dir = "UP" if direction[-1] == 1 else "DOWN"

    return current_st, current_dir


def calculate_efficiency_ratio(prices: pd.Series, period: int = 10) -> float:
    """Calculate Kaufman Efficiency Ratio (ER) to detect Choppy vs Trending markets."""
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


def generate_smart_sparkline(prices: pd.Series, period: int = 24) -> dict[str, Any]:
    """Generate Smart Sparkline v2.1 with key level, structure, and momentum using NumPy."""
    if len(prices) < period:
        return {"key_level": None, "structure": "UNCLEAR", "momentum": "STABLE"}

    subset = prices.iloc[-period:].values
    current_price = float(subset[-1])
    tolerance_pct = constants.SPARKLINE_TOLERANCE

    # Peak/Valley detection via SciPy concepts (numpy rolling comparisons)
    # Finding local maxima/minima with a window of 5 (2 before, 2 after)
    idx = np.arange(2, len(subset) - 2)
    is_peak = (
        (subset[idx] > subset[idx - 1])
        & (subset[idx] > subset[idx - 2])
        & (subset[idx] > subset[idx + 1])
        & (subset[idx] > subset[idx + 2])
    )
    is_valley = (
        (subset[idx] < subset[idx - 1])
        & (subset[idx] < subset[idx - 2])
        & (subset[idx] < subset[idx + 1])
        & (subset[idx] < subset[idx + 2])
    )

    peaks = subset[idx][is_peak].tolist()
    valleys = subset[idx][is_valley].tolist()

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
            price_range_val = np.ptp(subset)  # Peak-to-peak (max - min)
            if price_range_val / current_price < constants.RANGE_STRICT_THRESHOLD:
                structure = "RANGE"

    mid = len(subset) // 2
    first_half_change = abs(subset[mid] - subset[0]) / subset[0] if subset[0] != 0 else 0
    second_half_change = abs(subset[-1] - subset[mid]) / subset[mid] if subset[mid] != 0 else 0

    if (
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

    period_high = np.max(subset)
    period_low = np.min(subset)
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


def calculate_pivots(df: pd.DataFrame, periods: int = 24) -> dict[str, float]:
    """Calculate High/Low pivots over N periods"""
    if len(df) < periods:
        return {}
    subset = df.iloc[-periods:]
    return {"high": float(subset["high"].max()), "low": float(subset["low"].min())}


def generate_tags(indicators: dict[str, Any]) -> list[str]:
    """Generate analytical tags based on indicators"""
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

    rsi = indicators.get("rsi_14", 50)
    if rsi > constants.RSI_OVERBOUGHT:
        tags.append("RSI_Overbought")
    elif rsi < constants.RSI_OVERSOLD:
        tags.append("RSI_Oversold")

    atr = indicators.get("atr_14", 0)
    if price > 0 and atr / price > constants.VOLATILITY_THRESHOLD:
        tags.append("High_Volatility")

    return tags


def get_features_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw OHLCV Dataframe into an ML-ready Feature Matrix.
    Calculates technical indicators, applies stationarity (pct_change),
    and adds lag features to capture temporal dependencies (t-1, t-2).
    """
    if len(df) < constants.MIN_KLINE_DATA_POINTS:
        return pd.DataFrame()

    features = pd.DataFrame(index=df.index)
    features["timestamp"] = df["timestamp"]

    # 1. Base Prices & Stationarity (Return over previous bar)
    features["return_1p"] = df["close"].pct_change()
    features["return_vol"] = df["volume"].pct_change()

    # 2. Technical Indicators (Vectorized across the entire DataFrame)
    features["rsi_14"] = calculate_rsi_series(df["close"], constants.ATR_PERIOD_DEFAULT)
    features["rsi_7"] = calculate_rsi_series(df["close"], constants.FIB_8)

    macd_line, _macd_signal, macd_hist = calculate_macd_series(df["close"])
    features["macd_hist"] = macd_hist
    features["macd_line"] = macd_line

    features["atr_14"] = calculate_atr_series(
        df["high"], df["low"], df["close"], constants.ATR_PERIOD_DEFAULT
    )
    features["atr_ratio"] = features["atr_14"] / df["close"]  # Normalize ATR by price

    features["ema_20"] = calculate_ema_series(df["close"], constants.EMA_FAST)
    features["ema_50"] = calculate_ema_series(df["close"], constants.EMA_MEDIUM)
    features["ema_20_dist"] = (df["close"] - features["ema_20"]) / features["ema_20"]

    # ADX and DI
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_14_ewm = tr.ewm(span=14, adjust=False).mean()
    # Avoid div by zero which causes Inf -> NaN
    atr_14_ewm_safe = atr_14_ewm.replace(0, np.nan)

    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_14_ewm_safe)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_14_ewm_safe)

    di_sum = plus_di + minus_di
    di_diff_safe = abs(plus_di - minus_di)
    dx = 100 * (di_diff_safe / di_sum.replace(0, np.nan))

    features["adx_14"] = dx.ewm(span=constants.ATR_PERIOD_DEFAULT, adjust=False).mean().fillna(0)
    features["plus_di"] = plus_di.fillna(0)
    features["minus_di"] = minus_di.fillna(0)

    # Volatility / Bollinger Bandwidth
    middle = df["close"].rolling(window=constants.EMA_FAST).mean()
    std = df["close"].rolling(window=constants.EMA_FAST).std()
    upper = middle + (constants.BB_STD_DEV_NORMAL * std)
    lower = middle - (constants.BB_STD_DEV_NORMAL * std)
    features["bb_bandwidth"] = (upper - lower) / middle
    features["bb_percent_b"] = (df["close"] - lower) / (upper - lower)

    # 2.5 CATEGORY A: Directional Lags (DI Crossings)
    features["plus_di_lag1"] = features["plus_di"].shift(1)
    features["minus_di_lag1"] = features["minus_di"].shift(1)
    features["di_cross"] = features["plus_di"] - features["minus_di"]
    features["di_cross_lag1"] = features["di_cross"].shift(1)
    features["adx_14_lag1"] = features["adx_14"].shift(1)
    features["adx_growth"] = features["adx_14"] - features["adx_14_lag1"]

    # 2.6 CATEGORY B: Momentum Lags (RSI History/Slope)
    features["rsi_14_slope"] = features["rsi_14"] - features["rsi_14"].shift(1)
    features["rsi_7_slope"] = features["rsi_7"] - features["rsi_7"].shift(1)

    # 2.7 CATEGORY C: Structural Spacing (HTF Anchors)
    features["ema_50_dist"] = (df["close"] - features["ema_50"]) / features["ema_50"]
    features["ema_alignment"] = features["ema_20_dist"] - features["ema_50_dist"]

    # 2.8 CATEGORY D: Enhanced Slopes
    features["ema_20_slope"] = (features["ema_20"] - features["ema_20"].shift(1)) / features[
        "ema_20"
    ].shift(1)
    features["ema_50_slope"] = (features["ema_50"] - features["ema_50"].shift(1)) / features[
        "ema_50"
    ].shift(1)
    features["macd_hist_lag1"] = features["macd_hist"].shift(1)
    features["macd_hist_lag2"] = features["macd_hist"].shift(2)
    features["macd_slope"] = features["macd_hist"] - features["macd_hist_lag1"]

    # Momentum (Price Rate of Change)
    features["roc_10"] = df["close"].pct_change(periods=constants.INDICATOR_HISTORY_DEFAULT)

    # 3. Lag Features (Temporal History t-1, t-2)
    # XGBoost only sees one row at a time. It needs lag features to understand velocity.
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
            features[f"{col}_lag1"] = features[col].shift(1)
        if f"{col}_lag2" not in features.columns:
            features[f"{col}_lag2"] = features[col].shift(2)

    # 4. Cleanup
    # Forward fill non-critical NaNs
    features.ffill(inplace=True)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows that have NaNs due to lookback periods (e.g. at the start of the data)
    # Usually the first 50 rows will have some missing data (EMA_50 needs 50 bars)
    features.dropna(inplace=True)

    return features


if __name__ == "__main__":
    import sqlite3

    print("\n--- Testing ML Feature Extraction ---")
    try:
        conn = sqlite3.connect("data/market_data.db")
        # Direct query to avoid import deadlocks from DataEngine/RealMarketData
        query = "SELECT * FROM market_data WHERE coin='XRP' AND interval='15m' ORDER BY timestamp DESC LIMIT 500"
        df_raw = pd.read_sql_query(query, conn)
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms")
        conn.close()

        if df_raw.empty:
            print("[FAIL] No raw data in DB. Run Phase 1.2 first.")
        else:
            df_features = get_features_for_ml(df_raw)
            print(f"[OK] Extracted features from {len(df_raw)} raw candles.")
            print(f"[INFO] Generated {len(df_features)} ML rows.")
            print(f"[INFO] Feature Count: {len(df_features.columns)}")
            print("\nLast Row Sample:")
            pd.set_option("display.max_columns", None)
            print(df_features.tail(1))
    except Exception as e:
        print(f"[FAIL] Test Failed: {e}")
