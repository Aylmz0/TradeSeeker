import copy
import json
import threading
import time
import traceback
from typing import Any
from typing import Any, List

import numpy as np
import pandas as pd
import requests

from config.config import Config
from src.core.indicators import (
    calculate_adx,
    calculate_atr_series,
    calculate_bollinger_bands,
    calculate_efficiency_ratio,
    calculate_ema_series,
    calculate_macd_series,
    calculate_obv,
    calculate_pivots,
    calculate_rsi_series,
    calculate_supertrend,
    calculate_vwap,
    generate_smart_sparkline,
    generate_tags,
)
from src.core.schemas.alignment import AlignmentResult, AlignmentError
from src.utils import RetryManager, safe_file_read_cached, safe_file_write


# add this ass a cycle counter analysez

# HTF_INTERVAL used in main.py, we can get it from Config or define it here
HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"


class RealMarketData:
    """Real market data from Binance Spot and Futures"""

    def __init__(self):
        self.spot_url = "https://api.binance.com/api/v3"
        self.futures_url = "https://fapi.binance.com/fapi/v1"
        self.available_coins = ["XRP", "DOGE", "ASTER", "TRX", "ETH", "SOL"]
        self.indicator_history_length = 10
        self.session = RetryManager.create_session_with_retry()
        self.preloaded_indicators: dict[str, dict[str, dict[str, Any]]] = {}
        self._raw_dataframes: dict[str, dict[str, pd.DataFrame]] = {}

        # FIX: Circuit breaker state to prevent hammering Binance during outages
        # Thread-safe: uses lock to protect state mutations from concurrent access
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5  # Trip after 5 consecutive failures
        self._circuit_breaker_timeout = 60  # Stay open for 60 seconds
        self._circuit_breaker_until = 0  # Timestamp when breaker resets
        self._circuit_breaker_lock = threading.Lock()  # Thread safety for circuit breaker state

        # FIX: Short-lived cache for bulk price fetches to prevent duplicate logs/calls during cycle start
        self._last_price_fetch_time = 0.0
        self._last_price_cache: dict[str, float] = {}
        self._price_cache_lock = threading.Lock()

    def clear_preloaded_indicators(self):
        """Clear any preloaded indicator snapshots (typically once per cycle)."""
        self.preloaded_indicators = {}
        self._raw_dataframes = {}

    def get_cached_raw_dataframe(self, coin: str, interval: str):
        """Get a cached raw OHLCV DataFrame from the current cycle, or None."""
        return self._raw_dataframes.get(coin, {}).get(interval)

    def store_preloaded_indicator(self, coin: str, interval: str, indicators: dict[str, Any]):
        """Store a snapshot of indicators for reuse during the same cycle."""
        if not isinstance(indicators, dict):
            return
        coin_store = self.preloaded_indicators.setdefault(coin, {})
        coin_store[interval] = copy.deepcopy(indicators)

    def set_preloaded_indicators(self, cache: dict[str, dict[str, dict[str, Any]]]):
        """Bulk load pre-computed indicator cache (deep copy)."""
        preloaded: dict[str, dict[str, dict[str, Any]]] = {}
        for coin, intervals in (cache or {}).items():
            preloaded[coin] = {}
            for interval, data in (intervals or {}).items():
                if isinstance(data, dict):
                    preloaded[coin][interval] = copy.deepcopy(data)
        self.preloaded_indicators = preloaded

    def get_real_time_data(
        self,
        symbol: str,
        interval: str = "3m",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get real OHLCV data from Binance Spot with enhanced error handling, retry logic, and circuit breaker"""
        import time as _time_module

        # FIX: Circuit breaker check - if tripped, fail fast to avoid hammering API
        # Thread-safe: check is atomic (no lock needed for read)
        current_time = _time_module.time()
        if current_time < self._circuit_breaker_until:
            print(
                f"[CIRCUIT] OPEN for {symbol} {interval} - cooling down until {self._circuit_breaker_until:.0f}"
            )
            return pd.DataFrame()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                fetch_limit = limit + self.indicator_history_length + 50
                params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": fetch_limit}
                response = self.session.get(f"{self.spot_url}/klines", params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                # FIX: Success - reset circuit breaker (thread-safe)
                with self._circuit_breaker_lock:
                    if self._circuit_breaker_failures > 0:
                        print(f"[CIRCUIT] Reset after {self._circuit_breaker_failures} failures")
                        self._circuit_breaker_failures = 0
                        self._circuit_breaker_until = 0

                if len(data) < 50:
                    print(
                        f"[WARN] Insufficient kline data for {symbol} ({interval}). Got {len(data)}.",
                    )
                    return pd.DataFrame()

                df = pd.DataFrame(
                    data,
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base_asset_volume",
                        "taker_buy_quote_asset_volume",
                        "ignore",
                    ],
                )
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float).round(8)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                # Sanitize corrupted candles
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                if df.isna().any().any():
                    print(
                        f"[WARN] Invalid candles (NaN/Inf) detected for {symbol} ({interval}). Dropping invalid rows."
                    )
                    df.dropna(inplace=True)

                # Enhanced data validation
                if self._validate_kline_data(df, symbol, interval):
                    # Cache raw DataFrame for reuse within the same cycle (e.g., ML inference)
                    coin_key = symbol.replace("USDT", "")
                    self._raw_dataframes.setdefault(coin_key, {})[interval] = df.copy()
                    return df
                print(
                    f"[ERR]   Data validation failed for {symbol} ({interval}) - attempt {attempt + 1}/{max_retries}",
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                    continue
                print(
                    f"[ERR]   All retries failed for {symbol} ({interval}). Returning empty DataFrame.",
                )
                # FIX: Circuit breaker - increment failure count and trip if threshold reached (thread-safe)
                with self._circuit_breaker_lock:
                    self._circuit_breaker_failures += 1
                    if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
                        self._circuit_breaker_until = _time_module.time() + self._circuit_breaker_timeout
                        print(
                            f"[CIRCUIT] TRIPPED for {symbol} {interval} - open for {self._circuit_breaker_timeout}s (failures: {self._circuit_breaker_failures})"
                        )
                return pd.DataFrame()

            except requests.exceptions.Timeout:
                print(
                    f"[ERR]   Timeout for {symbol} ({interval}) - attempt {attempt + 1}/{max_retries}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                print(f"[ERR]   All retries timed out for {symbol} ({interval})")
                # FIX: Circuit breaker - increment failure count and trip if threshold reached (thread-safe)
                with self._circuit_breaker_lock:
                    self._circuit_breaker_failures += 1
                    if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
                        self._circuit_breaker_until = _time_module.time() + self._circuit_breaker_timeout
                        print(
                            f"[CIRCUIT] TRIPPED for {symbol} {interval} - open for {self._circuit_breaker_timeout}s (failures: {self._circuit_breaker_failures})"
                        )
                return pd.DataFrame()
            except Exception as e:
                print(
                    f"[ERR]   Kline data error {symbol} ({interval}) - attempt {attempt + 1}/{max_retries}: {e}",
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                print(f"[ERR]   All retries failed for {symbol} ({interval})")
                # FIX: Circuit breaker - increment failure count and trip if threshold reached (thread-safe)
                with self._circuit_breaker_lock:
                    self._circuit_breaker_failures += 1
                    if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
                        self._circuit_breaker_until = _time_module.time() + self._circuit_breaker_timeout
                        print(
                            f"[CIRCUIT] TRIPPED for {symbol} {interval} - open for {self._circuit_breaker_timeout}s (failures: {self._circuit_breaker_failures})"
                        )
                return pd.DataFrame()

        return pd.DataFrame()

    def _validate_kline_data(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        """Validate kline data quality with enhanced volume checks"""
        if df.empty:
            print(f"[WARN] Empty DataFrame for {symbol} ({interval})")
            return False

        # Check for zero or negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if (df[col] <= 0).any():
                print(
                    f"[WARN] Invalid price data for {symbol} ({interval}): {col} contains zero/negative values",
                )
                return False

        # Check for identical prices (stuck data)
        if df["close"].nunique() < 3:  # Less than 3 unique prices
            print(
                f"[WARN] Stuck price data for {symbol} ({interval}): only {df['close'].nunique()} unique prices",
            )
            return False

        # Volume validation - only check for zero/invalid volume
        volume_sum = df["volume"].sum()

        # Check for zero volume (data quality issue)
        if volume_sum == 0:
            print(f"[WARN] Zero volume for {symbol} ({interval})")
            return False

        # NOTE: Removed hard volume threshold filter
        # Volume quality is now handled by AI via prompt rules (0.3x threshold)
        # This allows AI to see low-volume coins and make informed decisions

        # Check for reasonable price movement
        price_range = df["high"].max() - df["low"].min()
        if price_range == 0:
            print(f"[WARN] No price movement for {symbol} ({interval})")
            return False

        return True

    def get_open_interest(self, symbol: str) -> float:
        """Get Latest Open Interest from Binance Futures"""
        try:
            params = {"symbol": f"{symbol}USDT"}
            response = self.session.get(
                f"{self.futures_url}/openInterest",
                params=params,
                timeout=5,
            )
            response.raise_for_status()
            return float(response.json()["openInterest"])
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                print(f"[INFO] OI not available for {symbol}USDT on Futures.")
            else:
                print(f"[ERR]   OI error for {symbol}: {e}")
            return 0.0

    def get_funding_rate(self, symbol: str) -> float:
        """Get Latest Funding Rate from Binance Futures"""
        try:
            params = {"symbol": f"{symbol}USDT"}
            response = self.session.get(
                f"{self.futures_url}/premiumIndex",
                params=params,
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                data = data[0] if data else {}

            rate = data.get("lastFundingRate")
            if rate is not None and rate != "":
                return float(rate)
            # print(f"[INFO] Using nextFundingRate for {symbol}.")
            rate = data.get("nextFundingRate")
            return float(rate) if rate is not None and rate != "" else 0.0
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and (
                e.response.status_code in [404, 400]
            ):
                print(f"[INFO] Funding Rate not available for {symbol}USDT on Futures.")
            else:
                print(f"[ERR]   Funding Rate error for {symbol}: {e}")
            return 0.0

    # --- Indicator Calculation Functions ---
    def get_technical_indicators(self, coin: str, interval: str) -> dict[str, Any]:
        """Calculate technical indicators, returning history series"""
        cached = self.preloaded_indicators.get(coin, {}).get(interval)
        if isinstance(cached, dict):
            return copy.deepcopy(cached)

        df = self.get_real_time_data(coin, interval=interval)
        if df.empty or len(df) < 50:
            return {"error": f"Not enough data for {coin} {interval} (got {len(df)})"}

        close_prices = df["close"]
        # FIX: Bounds check for empty series (defensive)
        if len(close_prices) == 0:
            return {"error": f"Empty close price series for {coin} {interval}"}
        current_price = close_prices.iloc[-1]
        hist_len = self.indicator_history_length
        indicators = {"current_price": current_price}
        try:
            ema_20_series = calculate_ema_series(close_prices, 21)
            ema_50_series = calculate_ema_series(close_prices, 55)  # Fibonacci: 21, 55
            rsi_14_series = calculate_rsi_series(close_prices, 13)
            macd_line_series, macd_signal_series, macd_hist_series = calculate_macd_series(
                close_prices,
            )  # Fibonacci: 13
            atr_14_series = calculate_atr_series(df["high"], df["low"], df["close"], 14)

            indicators["ema_20"] = ema_20_series.iloc[-1]
            indicators["ema_50"] = ema_50_series.iloc[-1]
            indicators["rsi_14"] = rsi_14_series.iloc[-1]
            indicators["macd"] = macd_line_series.iloc[-1]
            indicators["macd_signal"] = macd_signal_series.iloc[-1]
            indicators["macd_histogram"] = macd_hist_series.iloc[-1]
            indicators["atr_14"] = atr_14_series.iloc[-1]  # Keep atr_14 available for AI prompt

            # Use .where(pd.notna, None) to convert NaN to None for JSON
            indicators["ema_20_series"] = (
                ema_20_series.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()
            )
            indicators["rsi_14_series"] = (
                rsi_14_series.iloc[-hist_len:].round(3).where(pd.notna, None).tolist()
            )
            indicators["macd_series"] = (
                macd_line_series.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()
            )

            if interval == "3m":
                rsi_7_series = calculate_rsi_series(close_prices, 8)  # Fibonacci: 8
                indicators["rsi_7"] = rsi_7_series.iloc[-1]  # Keep key as rsi_7 for compatibility
                indicators["rsi_7_series"] = (
                    rsi_7_series.iloc[-hist_len:].round(3).where(pd.notna, None).tolist()
                )
            if interval == HTF_INTERVAL:
                atr_3_series = calculate_atr_series(df["high"], df["low"], df["close"], 3)
                indicators["atr_3"] = atr_3_series.iloc[-1]

            # Volume Analysis: Use last CLOSED candle for consistent ratio
            # iloc[-1] is current incomplete candle. iloc[-2] is last closed candle.
            current_vol = df["volume"].iloc[-1]
            last_closed_vol = df["volume"].iloc[-2]

            # Calculate average volume based on LAST 20 CLOSED candles (excluding current partial)
            # We take slice [-21:-1] which gives 20 candles ending at iloc[-2]
            avg_vol_closed = df["volume"].iloc[-21:-1].mean()

            indicators["volume"] = current_vol  # Keep current volume for AI context
            indicators["last_closed_volume"] = last_closed_vol
            indicators["avg_volume"] = (
                avg_vol_closed if pd.notna(avg_vol_closed) and avg_vol_closed > 0 else 1.0
            )

            # Pre-calculate ratio for consistency
            indicators["volume_ratio"] = last_closed_vol / indicators["avg_volume"]

            # Efficiency Ratio (ER) Calculation for Choppy Regime Detection
            # Using 10 periods (30 mins for 3m interval)
            indicators["efficiency_ratio"] = calculate_efficiency_ratio(
                close_prices,
                period=10,
            )

            # ==================== NEW INDICATORS (v5.0) ====================

            # 1. ADX/DMI - Trend Strength
            adx, plus_di, minus_di = calculate_adx(
                df["high"],
                df["low"],
                df["close"],
                period=14,
            )
            indicators["adx"] = adx
            indicators["plus_di"] = plus_di
            indicators["minus_di"] = minus_di

            if adx >= 40:
                indicators["trend_strength_adx"] = "STRONG"
            elif adx >= 25:
                indicators["trend_strength_adx"] = "MODERATE"
            elif adx >= 15:
                indicators["trend_strength_adx"] = "WEAK"
            else:
                indicators["trend_strength_adx"] = "NO_TREND"

            # 2. VWAP - Rolling 4-hour (60 bars for 4min cycle)
            vwap = calculate_vwap(df["high"], df["low"], df["close"], df["volume"], period=60)
            indicators["vwap"] = vwap
            if vwap > 0:
                vwap_distance_pct = ((current_price - vwap) / vwap) * 100
                indicators["vwap_distance_pct"] = round(vwap_distance_pct, 3)
                indicators["price_vs_vwap"] = "ABOVE" if current_price > vwap else "BELOW"
            else:
                indicators["vwap_distance_pct"] = 0.0
                indicators["price_vs_vwap"] = "UNKNOWN"

            # 3. Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b = calculate_bollinger_bands(
                close_prices
            )
            indicators["bb_upper"] = bb_upper
            indicators["bb_lower"] = bb_lower
            indicators["bb_bandwidth"] = bb_bandwidth
            indicators["bb_squeeze"] = bb_bandwidth < 0.03  # Squeeze detection

            if current_price > bb_upper:
                indicators["bb_signal"] = "OVERBOUGHT"
            elif current_price < bb_lower:
                indicators["bb_signal"] = "OVERSOLD"
            else:
                indicators["bb_signal"] = "NORMAL"

            # 4. OBV - On Balance Volume
            obv, obv_trend, obv_divergence = calculate_obv(close_prices, df["volume"])
            indicators["obv_trend"] = obv_trend
            indicators["obv_divergence"] = obv_divergence

            # 5. SuperTrend
            st_line, st_direction = calculate_supertrend(df["high"], df["low"], close_prices)
            indicators["supertrend"] = st_line
            indicators["supertrend_direction"] = st_direction

            # ==================== END NEW INDICATORS ====================

            indicators["price_series"] = (
                close_prices.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()
            )

            # Enhanced Context Integration (Sparklines, Pivots, Tags)
            # Smart Sparkline v2.1: HTF (1h) gets full data, 15m gets structure+momentum only
            if interval == HTF_INTERVAL:
                indicators["smart_sparkline"] = generate_smart_sparkline(
                    close_prices,
                    period=24,
                )
            elif interval == "15m":
                # 15m: structure, momentum, and price_location (no key_level for token efficiency)
                full_sparkline = generate_smart_sparkline(close_prices, period=24)
                indicators["smart_sparkline"] = {
                    "structure": full_sparkline.get("structure", "UNCLEAR"),
                    "momentum": full_sparkline.get("momentum", "STABLE"),
                    "price_location": full_sparkline.get(
                        "price_location",
                        {"zone": "MIDDLE", "percentile": 50},
                    ),
                }
            indicators["pivots"] = calculate_pivots(df, periods=24)
            indicators["tags"] = generate_tags(indicators)

            for key, value in indicators.items():
                if isinstance(value, float) and np.isnan(value):
                    indicators[key] = None
            self.store_preloaded_indicator(coin, interval, indicators)
            return indicators
        except Exception as e:
            print(f"[ERR]   Indicator error {coin} ({interval}): {e}")
            traceback.print_exc()
            return {"current_price": current_price, "error": str(e)}

    def get_all_real_prices(self) -> dict[str, float]:
        """Get real prices for all coins from Spot with enhanced error handling and short-lived caching"""
        with self._price_cache_lock:
            # FIX: Check cache first (2-second TTL) to prevent duplicate logs/calls during concurrent cycle start
            current_time = time.time()
            if current_time - self._last_price_fetch_time < 2.0 and self._last_price_cache:
                return copy.deepcopy(self._last_price_cache)

            prices: dict[str, float] = {}
            symbols = [f"{coin}USDT" for coin in self.available_coins]

            def _assign_price(symbol: str, raw_price: Any):
                coin = symbol.replace("USDT", "")
                try:
                    price_val = round(float(raw_price), 8)
                    if price_val <= 0 or np.isnan(price_val) or np.isinf(price_val):
                        raise ValueError(f"Invalid price value {price_val}")
                    prices[coin] = price_val
                except Exception as e:
                    print(f"[WARN] Invalid bulk price for {coin}: {raw_price} ({e}). Using fallback.")
                    prices[coin] = self._get_fallback_price(coin)

            # First try batched endpoint (single request, lower latency)
            try:
                response = self.session.get(
                    f"{self.spot_url}/ticker/price",
                    params={"symbols": json.dumps(symbols, separators=(",", ":"))},
                    timeout=3,
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    for entry in data:
                        symbol = entry.get("symbol")
                        price_raw = entry.get("price")
                        if symbol and price_raw is not None:
                            _assign_price(symbol, price_raw)
                    # Ensure we filled everything; fall back only for missing
                    missing = [coin for coin in self.available_coins if coin not in prices]
                    if not missing:
                        prices_str = " | ".join([f"{coin}: ${val:.4f}" for coin, val in prices.items()])
                        print(f"[OK]    Prices: {prices_str}")
                        
                        # Update cache
                        self._last_price_fetch_time = time.time()
                        self._last_price_cache = copy.deepcopy(prices)
                            
                        return prices
                    print(
                        f"[WARN] Bulk price missing for: {', '.join(missing)}. Falling back to individual requests.",
                    )
                else:
                    print(
                        "[WARN] Unexpected bulk ticker response format. Falling back to individual requests.",
                    )
            except Exception as e:
                print(f"[WARN] Bulk price fetch failed: {e}. Falling back to individual requests.")

            # Fallback to individual calls (still using session, without artificial delay)
            for coin in self.available_coins:
                try:
                    response = self.session.get(
                        f"{self.spot_url}/ticker/price",
                        params={"symbol": f"{coin}USDT"},
                        timeout=3,
                    )
                    response.raise_for_status()
                    data = response.json()
                    price_val = round(float(data.get("price", 0)), 8)
                    if price_val <= 0 or np.isnan(price_val) or np.isinf(price_val):
                        raise ValueError(f"Invalid price value {price_val}")
                    prices[coin] = price_val
                except Exception as e:
                    print(f"[ERR]   {coin} price error: {e}. Using fallback...")
                    prices[coin] = self._get_fallback_price(coin)

            if len(prices) > 0:
                prices_str = " | ".join([f"{c}: ${p:.4f}" for c, p in prices.items()])
                print(f"[OK]    Prices: {prices_str}")
                
                # Update cache (fallback case)
                self._last_price_fetch_time = time.time()
                self._last_price_cache = copy.deepcopy(prices)

            return prices

    def _get_fallback_price(self, coin: str) -> float:
        """Get fallback price using multiple methods"""
        # Method 1: Try 1m kline data
        try:
            df = self.get_real_time_data(coin, interval="1m", limit=1)
            if not df.empty and not df["close"].empty:
                price_val = df["close"].iloc[-1]
                if price_val > 0 and pd.notna(price_val):
                    print(f"   Fallback 1m kline: ${price_val:.4f}")
                    return price_val
        except Exception as e:
            print(f"   Fallback 1m failed: {e}")

        # Method 2: Try 3m kline data
        try:
            df = self.get_real_time_data(coin, interval="3m", limit=1)
            if not df.empty and not df["close"].empty:
                price_val = df["close"].iloc[-1]
                if price_val > 0 and pd.notna(price_val):
                    print(f"   Fallback 3m kline: ${price_val:.4f}")
                    return price_val
        except Exception as e:
            print(f"   Fallback 3m failed: {e}")

        # Method 3: Use cached price from previous cycle
        try:
            from src.utils import safe_file_read

            cached_prices = safe_file_read("data/portfolio_state.json", default_data={})
            if "positions" in cached_prices:
                for pos_coin, position in cached_prices["positions"].items():
                    if pos_coin == coin and "current_price" in position:
                        cached_price = position["current_price"]
                        if cached_price > 0:
                            print(f"   Fallback cached: ${cached_price:.4f}")
                            return cached_price
        except Exception as e:
            print(f"   Fallback cache failed: {e}")

        # Final fallback: return 0 with warning
        print(f"   [WARNING] All fallbacks failed for {coin}. Price set to 0.")
        return 0.0

    def verify_sync_alignment(self, coin: str, intervals: List[str] = ["3m", "15m", "1h"]) -> AlignmentResult:
        """
        Verifies that the latest kline timestamps for multiple intervals are within 
        Config.MAX_ALIGNMENT_DELTA_S.
        """
        timestamps = {}
        errors = []
        
        try:
            for interval in intervals:
                # Use internal cache to avoid redundant API hits
                # In a real cycle, indices should already be preloaded or fetched
                klines = self.get_real_time_data(coin, interval=interval, limit=1)
                
                if klines is None or klines.empty:
                    return AlignmentResult(
                        aligned=False, 
                        error_type=AlignmentError.INSUFFICIENT_DATA,
                        error_message=f"No kline data for {coin} @ {interval}"
                    )
                
                # Use the latest closed candle timestamp
                # The 'timestamp' column is the open time of the candle.
                # The 'close_time' column is the close time of the candle.
                # For alignment, we care about the latest *completed* candle.
                # Binance klines 'timestamp' is the open time, 'close_time' is the close time.
                # We want the close time of the last *closed* candle.
                # If limit=1, df.iloc[-1] is the latest candle, which might be incomplete.
                # However, for alignment, we usually compare the *start* of the latest candle.
                # Let's assume 'timestamp' (open time) is what we need for alignment check.
                latest_ts = int(klines.iloc[-1]['timestamp'].timestamp() * 1000) # Convert datetime to ms timestamp
                timestamps[interval] = latest_ts
                
            if not timestamps:
                return AlignmentResult(aligned=False, error_type=AlignmentError.INSUFFICIENT_DATA)

            # Check deltas between all pairs
            ts_values = list(timestamps.values())
            max_ts = max(ts_values)
            min_ts = min(ts_values)
            delta_ms = max_ts - min_ts
            delta_s = delta_ms / 1000.0
            
            is_aligned = delta_s <= Config.MAX_ALIGNMENT_DELTA_S
            
            mismatches = []
            if not is_aligned:
                for interval, ts in timestamps.items():
                    mismatches.append({"interval": interval, "timestamp": ts, "delta_from_max": (max_ts - ts)/1000.0})

            return AlignmentResult(
                aligned=is_aligned,
                max_delta_seconds=delta_s,
                mismatches=mismatches if not is_aligned else [],
                error_type=AlignmentError.NONE if is_aligned else AlignmentError.EXCESSIVE_MISMATCH
            )

        except Exception as e:
            return AlignmentResult(
                aligned=False,
                error_type=AlignmentError.API_FAILURE,
                error_message=str(e)
            )

    def get_market_sentiment(self, coin: str) -> dict[str, Any]:
        """Get Open Interest and Funding Rate (Nof1ai format)"""
        open_interest = self.get_open_interest(coin)
        funding_rate = self.get_funding_rate(coin)

        # Nof1ai format: "Latest: X Average: Y" for Open Interest
        avg_oi = open_interest  # Simplified average calculation
        return {
            "open_interest": open_interest,
            "open_interest_avg": avg_oi,
            "funding_rate": funding_rate,
        }

    def detect_trend_reversal_signals(
        self,
        coin: str,
        indicators_3m: dict[str, Any],
        indicators_htf: dict[str, Any],
        indicators_15m: dict[str, Any] = None,
        position_direction: str = None,
    ) -> dict[str, Any]:
        """
        Detect potential trend reversal signals with weighted scoring.

        Weights:
        - HTF trend reversal: +3
        - 15m structure conflict: +3
        - 15m momentum reversal: +2
        - 3m trend reversal: +1
        - RSI extreme: +1
        - MACD divergence: +1

        Strength levels:
        - NONE: score = 0
        - WEAK: score 1-2
        - MODERATE: score 3-4
        - STRONG: score 5-7
        - CRITICAL: score 8+
        """
        score = 0
        signals = []

        if not indicators_3m or not indicators_htf:
            return {
                "signals": [],
                "score": 0,
                "strength": "NONE",
                "trend_htf": None,
                "trend_15m": None,
                "trend_3m": None,
            }

        # Extract indicators
        price_3m = indicators_3m.get("current_price")
        ema20_3m = indicators_3m.get("ema_20")
        rsi_3m = indicators_3m.get("rsi_14")
        macd_3m = indicators_3m.get("macd")
        macd_signal_3m = indicators_3m.get("macd_signal")

        price_htf = indicators_htf.get("current_price")
        ema20_htf = indicators_htf.get("ema_20")

        if None in [price_3m, ema20_3m, price_htf, ema20_htf]:
            return {
                "signals": [],
                "score": 0,
                "strength": "NONE",
                "trend_htf": None,
                "trend_15m": None,
                "trend_3m": None,
            }

        # Determine trends
        def _determine_trend(price: float, ema20: float) -> str:
            if ema20 == 0:
                return "UNKNOWN"
            delta = (price - ema20) / ema20
            if abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT:
                return "NEUTRAL"
            return "BULLISH" if delta > 0 else "BEARISH"

        trend_3m = _determine_trend(price_3m, ema20_3m)
        trend_htf = _determine_trend(price_htf, ema20_htf)
        trend_15m = None
        structure_15m = None

        # Extract 15m data if available
        if indicators_15m and "error" not in indicators_15m:
            price_15m = indicators_15m.get("current_price")
            ema20_15m = indicators_15m.get("ema_20")
            sparkline_15m = indicators_15m.get("smart_sparkline", {})
            structure_15m = (
                sparkline_15m.get("structure", "UNCLEAR")
                if isinstance(sparkline_15m, dict)
                else "UNCLEAR"
            )

            if price_15m and ema20_15m:
                trend_15m = _determine_trend(price_15m, ema20_15m)

        # If no position direction, detect general reversal signals
        if not position_direction:
            # Return basic trend info without scoring
            return {
                "signals": [],
                "score": 0,
                "strength": "NONE",
                "trend_htf": trend_htf,
                "trend_15m": trend_15m,
                "trend_3m": trend_3m,
            }

        # ===== WEIGHTED SCORING =====

        # 1. HTF trend reversal (+3)
        if position_direction == "long" and trend_htf == "BEARISH":
            score += 3
            signals.append("htf_bearish_vs_long(+3)")
        elif position_direction == "short" and trend_htf == "BULLISH":
            score += 3
            signals.append("htf_bullish_vs_short(+3)")

        # 2. 15m structure conflict (+3)
        if structure_15m:
            if position_direction == "long" and structure_15m == "LH_LL":
                score += 3
                signals.append("15m_lhll_vs_long(+3)")
            elif position_direction == "short" and structure_15m == "HH_HL":
                score += 3
                signals.append("15m_hhhl_vs_short(+3)")

        # 3. 15m momentum reversal (+2)
        if trend_15m:
            if position_direction == "long" and trend_15m == "BEARISH":
                score += 2
                signals.append("15m_bearish_vs_long(+2)")
            elif position_direction == "short" and trend_15m == "BULLISH":
                score += 2
                signals.append("15m_bullish_vs_short(+2)")

        # 4. 3m trend reversal (+1)
        if position_direction == "long" and trend_3m == "BEARISH":
            score += 1
            signals.append("3m_bearish_vs_long(+1)")
        elif position_direction == "short" and trend_3m == "BULLISH":
            score += 1
            signals.append("3m_bullish_vs_short(+1)")

        # 5. RSI extreme (+1)
        if rsi_3m is not None:
            if position_direction == "long" and rsi_3m > Config.RSI_OVERBOUGHT_THRESHOLD:
                score += 1
                signals.append(f"rsi_overbought_{rsi_3m:.0f}(+1)")
            elif position_direction == "short" and rsi_3m < Config.RSI_OVERSOLD_THRESHOLD:
                score += 1
                signals.append(f"rsi_oversold_{rsi_3m:.0f}(+1)")

        # 6. MACD divergence (+1)
        if macd_3m is not None and macd_signal_3m is not None:
            if position_direction == "long" and macd_3m < macd_signal_3m:
                score += 1
                signals.append("macd_bearish_cross(+1)")
            elif position_direction == "short" and macd_3m > macd_signal_3m:
                score += 1
                signals.append("macd_bullish_cross(+1)")

        # Determine strength from score
        if score >= 8:
            strength = "CRITICAL"
        elif score >= 5:
            strength = "STRONG"
        elif score >= 3:
            strength = "MODERATE"
        elif score >= 1:
            strength = "WEAK"
        else:
            strength = "NONE"

        return {
            "signals": signals,
            "score": score,
            "strength": strength,
            "trend_htf": trend_htf,
            "trend_15m": trend_15m,
            "trend_3m": trend_3m,
        }
