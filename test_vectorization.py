import numpy as np
import pandas as pd
from src.core.indicators import calculate_obv, calculate_supertrend, generate_smart_sparkline

np.random.seed(42)

def generate_test_data(size=100):
    prices = [100.0]
    highs = []
    lows = []
    volumes = []
    for _ in range(size):
        change = np.random.normal(0, 1)
        new_price = prices[-1] + change
        prices.append(new_price)
        highs.append(new_price + abs(np.random.normal(0, 1)))
        lows.append(new_price - abs(np.random.normal(0, 1)))
        volumes.append(abs(np.random.normal(100, 20)))
    
    # Trim initial to match sizes
    prices = prices[1:]
    
    df = pd.DataFrame({
        'close': prices,
        'high': highs,
        'low': lows,
        'volume': volumes
    })
    return df

def calculate_obv_legacy(close: pd.Series, volume: pd.Series):
    if len(close) < 10:
        return 0.0, "FLAT", "NONE"

    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])

    obv_series = pd.Series(obv)
    current_obv = float(obv_series.iloc[-1])

    obv_change = obv_series.iloc[-1] - obv_series.iloc[-10]
    if obv_change > 0:
        obv_trend = "RISING"
    elif obv_change < 0:
        obv_trend = "FALLING"
    else:
        obv_trend = "FLAT"

    price_change = close.iloc[-1] - close.iloc[-10]
    divergence = "NONE"

    if price_change > 0 and obv_change < 0:
        divergence = "BEARISH"
    elif price_change < 0 and obv_change > 0:
        divergence = "BULLISH"

    return current_obv, obv_trend, divergence

def calculate_supertrend_legacy(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0):
    if len(close) < period + 1:
        return float(close.iloc[-1]) if len(close) > 0 else 0.0, "UP"

    tr = pd.concat(
        [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
    ).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(close)):
        if close.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1

    current_st = float(supertrend.iloc[-1])
    current_dir = "UP" if direction.iloc[-1] == 1 else "DOWN"

    return current_st, current_dir


if __name__ == "__main__":
    df = generate_test_data(500)
    
    print("Testing OBV Vectorization Parity...")
    leg_obv, leg_tr, leg_div = calculate_obv_legacy(df['close'], df['volume'])
    vec_obv, vec_tr, vec_div = calculate_obv(df['close'], df['volume'])
    
    assert np.isclose(leg_obv, vec_obv), f"OBV mismatch: Legacy {leg_obv} != Vectorized {vec_obv}"
    assert leg_tr == vec_tr, f"OBV trend mismatch"
    assert leg_div == vec_div, f"OBV divergence mismatch"
    print("✅ OBV Vectorization passed!")
    
    print("Testing Supertrend Vectorization Parity...")
    leg_st, leg_dir = calculate_supertrend_legacy(df['high'], df['low'], df['close'])
    vec_st, vec_dir = calculate_supertrend(df['high'], df['low'], df['close'])
    
    assert np.isclose(leg_st, vec_st), f"Supertrend mismatch: Legacy {leg_st} != Vectorized {vec_st}"
    assert leg_dir == vec_dir, f"Supertrend direction mismatch"
    print("✅ Supertrend Vectorization passed!")
    
    print("Testing Smart Sparkline Execution (no crash)...")
    res = generate_smart_sparkline(df['close'], 24)
    assert res['momentum'] in ["STRENGTHENING", "WEAKENING", "STABLE"]
    print("✅ Smart Sparkline passed!")
    
    print("\nALL PARITY TESTS SUCCESSFUL. MATHEMATICAL INTEGRITY MAINTAINED.")
