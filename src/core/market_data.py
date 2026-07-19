import copy

from loguru import logger
import json
import math
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import polars as pl
import requests

from config.config import Config
from src.core import constants
from src.core.indicators import (
    calculate_adx,
    calculate_atr_series,
    calculate_bollinger_bands,
    calculate_efficiency_ratio,
    calculate_ema_series,
    calculate_ema_stretch_label,
    calculate_macd_series,
    calculate_obv,
    calculate_pivots,
    calculate_rsi_divergence_label,
    calculate_rsi_series,
    calculate_slope_label,
    calculate_supertrend,
    calculate_volatility_pulse_label,
    calculate_vwap,
    generate_smart_sparkline,
    generate_tags,
)
from src.core.schemas.alignment import AlignmentError, AlignmentResult
from src.utils import RetryManager

HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"

KLINE_COLUMNS = [
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
]


class RealMarketData:
    """Real market data from Binance Spot and Futures"""

    def __init__(self):
        self.spot_url = "https://api.binance.com/api/v3"
        self.futures_url = "https://fapi.binance.com/fapi/v1"
        self.available_coins = ["XRP", "DOGE", "ASTER", "TRX", "ETH", "SOL"]
        self.indicator_history_length = 10
        self.session = RetryManager.create_session_with_retry()
        self.preloaded_indicators: dict[str, dict[str, dict[str, Any]]] = {}
        self._raw_dataframes: dict[str, dict[str, pl.DataFrame]] = {}

        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = constants.CIRCUIT_BREAKER_THRESHOLD
        self._circuit_breaker_timeout = constants.CIRCUIT_BREAKER_TIMEOUT_S
        self._circuit_breaker_until = 0
        self._circuit_breaker_lock = threading.RLock()

        self._last_price_fetch_time = 0.0
        self._last_price_cache: dict[str, float] = {}
        self._price_cache_lock = threading.RLock()

    def clear_preloaded_indicators(self):
        self.preloaded_indicators = {}
        self._raw_dataframes = {}

    def get_cached_raw_dataframe(self, coin: str, interval: str):
        return self._raw_dataframes.get(coin, {}).get(interval)

    def store_preloaded_indicator(self, coin: str, interval: str, indicators: dict[str, Any]):
        if not isinstance(indicators, dict):
            return
        coin_store = self.preloaded_indicators.setdefault(coin, {})
        coin_store[interval] = copy.deepcopy(indicators)

    def set_preloaded_indicators(self, cache: dict[str, dict[str, dict[str, Any]]]):
        preloaded: dict[str, dict[str, dict[str, Any]]] = {}
        for coin, intervals in (cache or {}).items():
            preloaded[coin] = {}
            for interval, data in (intervals or {}).items():
                if isinstance(data, dict):
                    preloaded[coin][interval] = copy.deepcopy(data)
        self.preloaded_indicators = preloaded

    def _build_empty_df(self) -> pl.DataFrame:
        return pl.DataFrame(schema=KLINE_COLUMNS)

    def get_real_time_data(
        self,
        symbol: str,
        interval: str = "3m",
        limit: int = 100,
    ) -> pl.DataFrame:
        import time as _time_module

        current_time = _time_module.time()
        if current_time < self._circuit_breaker_until:
            logger.warning(
                "CIRCUIT OPEN for {} {} - cooling down until {:.0f}",
                symbol,
                interval,
                self._circuit_breaker_until,
            )
            return self._build_empty_df()

        max_retries = constants.MAX_API_RETRIES
        for attempt in range(max_retries):
            try:
                fetch_limit = (
                    limit + self.indicator_history_length + constants.MIN_KLINE_DATA_POINTS
                )
                params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": fetch_limit}
                response = self.session.get(
                    f"{self.spot_url}/klines", params=params, timeout=constants.API_TIMEOUT_SECONDS
                )
                response.raise_for_status()
                data = response.json()

                with self._circuit_breaker_lock:
                    if self._circuit_breaker_failures > 0:
                        logger.info(
                            "CIRCUIT Reset after {} failures", self._circuit_breaker_failures
                        )
                        self._circuit_breaker_failures = 0
                        self._circuit_breaker_until = 0

                if len(data) < constants.MIN_KLINE_DATA_POINTS:
                    logger.warning(
                        "Insufficient kline data for {} ({}). Got {}.", symbol, interval, len(data)
                    )
                    return self._build_empty_df()

                df = pl.DataFrame(data, schema=KLINE_COLUMNS, orient="row")

                for col in ["open", "high", "low", "close", "volume"]:
                    df = df.with_columns(pl.col(col).cast(pl.Float64).round(8))

                df = df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime(time_unit="ms")).alias("timestamp")
                )

                df = df.fill_nan(None)
                has_nan = df.select(pl.any_horizontal(pl.col(pl.Float64).is_nan()).any()).item()
                if has_nan:
                    logger.warning(
                        "Invalid candles (NaN/Inf) detected for {} ({}). Dropping invalid rows.",
                        symbol,
                        interval,
                    )
                    df = df.drop_nulls()

                if self._validate_kline_data(df, symbol, interval):
                    coin_key = symbol.replace("USDT", "")
                    self._raw_dataframes.setdefault(coin_key, {})[interval] = df.clone()
                    return df
                logger.error(
                    "Data validation failed for {} ({}) - attempt {}/{}",
                    symbol,
                    interval,
                    attempt + 1,
                    max_retries,
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                logger.error(
                    "All retries failed for {} ({}). Returning empty DataFrame.", symbol, interval
                )
                with self._circuit_breaker_lock:
                    self._circuit_breaker_failures += 1
                    if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
                        self._circuit_breaker_until = (
                            _time_module.time() + self._circuit_breaker_timeout
                        )
                        logger.warning(
                            "CIRCUIT TRIPPED for {} {} - open for {}s (failures: {})",
                            symbol,
                            interval,
                            self._circuit_breaker_timeout,
                            self._circuit_breaker_failures,
                        )
                return self._build_empty_df()

            except requests.exceptions.Timeout:
                logger.error(
                    "Timeout for {} ({}) - attempt {}/{}",
                    symbol,
                    interval,
                    attempt + 1,
                    max_retries,
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                logger.error("All retries timed out for {} ({})", symbol, interval)
                with self._circuit_breaker_lock:
                    self._circuit_breaker_failures += 1
                    if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
                        self._circuit_breaker_until = (
                            _time_module.time() + self._circuit_breaker_timeout
                        )
                        logger.warning(
                            "CIRCUIT TRIPPED for {} {} - open for {}s (failures: {})",
                            symbol,
                            interval,
                            self._circuit_breaker_timeout,
                            self._circuit_breaker_failures,
                        )
                return self._build_empty_df()
            except Exception as e:
                logger.error(
                    "Kline data error {} ({}) - attempt {}/{}: {}",
                    symbol,
                    interval,
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                logger.error("All retries failed for {} ({})", symbol, interval)
                with self._circuit_breaker_lock:
                    self._circuit_breaker_failures += 1
                    if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
                        self._circuit_breaker_until = (
                            _time_module.time() + self._circuit_breaker_timeout
                        )
                        logger.warning(
                            "CIRCUIT TRIPPED for {} {} - open for {}s (failures: {})",
                            symbol,
                            interval,
                            self._circuit_breaker_timeout,
                            self._circuit_breaker_failures,
                        )
                return self._build_empty_df()

        return self._build_empty_df()

    def _validate_kline_data(self, df: pl.DataFrame, symbol: str, interval: str) -> bool:
        if df.is_empty():
            logger.warning("Empty DataFrame for {} ({})", symbol, interval)
            return False

        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if (df[col] <= 0).any():
                logger.warning(
                    "Invalid price data for {} ({}): {} contains zero/negative values",
                    symbol,
                    interval,
                    col,
                )
                return False

        if df["close"].n_unique() < constants.STUCK_DATA_PRICE_COUNT:
            logger.warning(
                "Stuck price data for {} ({}): only {} unique prices",
                symbol,
                interval,
                df["close"].n_unique(),
            )
            return False

        volume_sum = df["volume"].sum()
        if volume_sum == 0:
            logger.warning("Zero volume for {} ({})", symbol, interval)
            return False

        price_range = df["high"].max() - df["low"].min()
        if price_range == 0:
            logger.warning("No price movement for {} ({})", symbol, interval)
            return False

        return True

    def get_open_interest(self, symbol: str) -> float:
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
            if (
                isinstance(e, requests.exceptions.HTTPError)
                and e.response.status_code == constants.OI_NOT_AVAILABLE_STATUS
            ):
                logger.info("OI not available for {}USDT on Futures.", symbol)
            else:
                logger.error("OI error for {}: {}", symbol, e)
            return 0.0

    def _calculate_max_drawdown(self, value_history: list[float]) -> float:
        if not value_history or len(value_history) < constants.MIN_HISTORY_FOR_ANALYSIS:
            return 0.0
        peak = value_history[0]
        max_drawdown = 0.0
        for value in value_history:
            peak = max(peak, value)
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown

    def get_funding_rate(self, symbol: str) -> float:
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
            rate = data.get("nextFundingRate")
            return float(rate) if rate is not None and rate != "" else 0.0
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and (
                e.response.status_code in [404, 400]
            ):
                logger.info("Funding Rate not available for {}USDT on Futures.", symbol)
            else:
                logger.error("Funding Rate error for {}: {}", symbol, e)
            return 0.0

    def get_technical_indicators(self, coin: str, interval: str) -> dict[str, Any]:
        cached = self.preloaded_indicators.get(coin, {}).get(interval)
        if isinstance(cached, dict):
            return copy.deepcopy(cached)

        df = self.get_real_time_data(coin, interval=interval)
        if df.is_empty() or len(df) < constants.MIN_KLINE_DATA_POINTS:
            return {"error": f"Not enough data for {coin} {interval} (got {len(df)})"}

        close_prices = df["close"]
        if len(close_prices) == 0:
            return {"error": f"Empty close price series for {coin} {interval}"}
        current_price = float(close_prices[-1])
        hist_len = self.indicator_history_length
        indicators: dict[str, Any] = {"current_price": current_price}
        try:
            ema_20_series = calculate_ema_series(close_prices, constants.FIB_21)
            ema_50_series = calculate_ema_series(close_prices, constants.FIB_55)
            rsi_14_series = calculate_rsi_series(close_prices, constants.FIB_13)
            macd_line_series, macd_signal_series, macd_hist_series = calculate_macd_series(
                close_prices,
            )
            atr_14_series = calculate_atr_series(df["high"], df["low"], df["close"], 14)

            indicators["ema_20"] = float(ema_20_series[-1])
            indicators["ema_50"] = float(ema_50_series[-1])
            indicators["rsi_14"] = float(rsi_14_series[-1])
            indicators["macd"] = float(macd_line_series[-1])
            indicators["macd_signal"] = float(macd_signal_series[-1])
            indicators["macd_histogram"] = float(macd_hist_series[-1])
            indicators["atr_14"] = float(atr_14_series[-1])

            indicators["ema_20_series"] = (
                ema_20_series[-hist_len:].round(4).fill_nan(None).to_list()
            )
            indicators["rsi_14_series"] = (
                rsi_14_series[-hist_len:].round(3).fill_nan(None).to_list()
            )
            indicators["macd_series"] = (
                macd_line_series[-hist_len:].round(4).fill_nan(None).to_list()
            )

            if interval == "3m":
                rsi_7_series = calculate_rsi_series(close_prices, constants.FIB_8)
                indicators["rsi_7"] = float(rsi_7_series[-1])
                indicators["rsi_7_series"] = (
                    rsi_7_series[-hist_len:].round(3).fill_nan(None).to_list()
                )
            if interval == HTF_INTERVAL:
                atr_3_series = calculate_atr_series(
                    df["high"], df["low"], df["close"], constants.FIB_3
                )
                indicators["atr_3"] = float(atr_3_series[-1])

            current_vol = float(df["volume"][-1])
            last_closed_vol = float(df["volume"][-2])

            vol_slice = df["volume"][-(constants.VOLUME_WINDOW_SIZE + 1) : -1]
            avg_vol_closed = vol_slice.mean() if len(vol_slice) > 0 else None

            indicators["volume"] = current_vol
            indicators["last_closed_volume"] = last_closed_vol
            indicators["avg_volume"] = (
                float(avg_vol_closed) if avg_vol_closed is not None and avg_vol_closed > 0 else 1.0
            )

            indicators["volume_ratio"] = last_closed_vol / indicators["avg_volume"]

            indicators["efficiency_ratio"] = calculate_efficiency_ratio(
                close_prices,
                period=10,
            )

            adx, plus_di, minus_di = calculate_adx(
                df["high"],
                df["low"],
                df["close"],
                period=14,
            )
            indicators["adx"] = adx
            indicators["plus_di"] = plus_di
            indicators["minus_di"] = minus_di

            if adx >= constants.ADX_STRONG_THRESHOLD:
                indicators["trend_strength_adx"] = "STRONG"
            elif adx >= constants.ADX_MODERATE_THRESHOLD:
                indicators["trend_strength_adx"] = "MODERATE"
            elif adx >= constants.ADX_WEAK_THRESHOLD:
                indicators["trend_strength_adx"] = "WEAK"
            else:
                indicators["trend_strength_adx"] = "NO_TREND"

            vwap = calculate_vwap(
                df["high"], df["low"], df["close"], df["volume"], period=constants.VWAP_WINDOW_SIZE
            )
            indicators["vwap"] = vwap
            if vwap > 0:
                vwap_distance_pct = ((current_price - vwap) / vwap) * 100
                indicators["vwap_distance_pct"] = round(vwap_distance_pct, 3)
                indicators["price_vs_vwap"] = "ABOVE" if current_price > vwap else "BELOW"
            else:
                indicators["vwap_distance_pct"] = 0.0
                indicators["price_vs_vwap"] = "UNKNOWN"

            bb_upper, _bb_middle, bb_lower, bb_bandwidth, _bb_percent_b = calculate_bollinger_bands(
                close_prices
            )
            indicators["bb_upper"] = bb_upper
            indicators["bb_lower"] = bb_lower
            indicators["bb_bandwidth"] = bb_bandwidth
            indicators["bb_squeeze"] = bb_bandwidth < constants.BB_SQUEEZE_THRESHOLD

            if current_price > bb_upper:
                indicators["bb_signal"] = "OVERBOUGHT"
            elif current_price < bb_lower:
                indicators["bb_signal"] = "OVERSOLD"
            else:
                indicators["bb_signal"] = "NORMAL"

            _obv, obv_trend, obv_divergence = calculate_obv(close_prices, df["volume"])
            indicators["obv_trend"] = obv_trend
            indicators["obv_divergence"] = obv_divergence

            st_line, st_direction = calculate_supertrend(df["high"], df["low"], close_prices)
            indicators["supertrend"] = st_line
            indicators["supertrend_direction"] = st_direction

            indicators["price_slope_label"] = calculate_slope_label(close_prices)
            indicators["rsi_divergence_label"] = calculate_rsi_divergence_label(
                close_prices, rsi_14_series
            )
            indicators["ema_stretch_label"] = calculate_ema_stretch_label(
                current_price, indicators.get("ema_20")
            )
            atr_3_val = indicators.get("atr_3")
            atr_14_val = indicators.get("atr_14")
            indicators["volatility_pulse_label"] = calculate_volatility_pulse_label(
                atr_3_val, atr_14_val
            )

            indicators["price_series"] = close_prices[-hist_len:].round(4).fill_nan(None).to_list()

            if interval == HTF_INTERVAL:
                indicators["smart_sparkline"] = generate_smart_sparkline(
                    close_prices,
                    period=constants.SPARKLINE_WINDOW,
                )
            elif interval == "15m":
                full_sparkline = generate_smart_sparkline(
                    close_prices, period=constants.SPARKLINE_WINDOW
                )
                indicators["smart_sparkline"] = {
                    "structure": full_sparkline.get("structure", "UNCLEAR"),
                    "momentum": full_sparkline.get("momentum", "STABLE"),
                    "price_location": full_sparkline.get(
                        "price_location",
                        {"zone": "MIDDLE", "percentile": 50},
                    ),
                }
            indicators["pivots"] = calculate_pivots(df, periods=constants.SPARKLINE_WINDOW)
            indicators["tags"] = generate_tags(indicators)

            for key, value in indicators.items():
                if isinstance(value, float) and (value != value):  # NaN check
                    indicators[key] = None
            self.store_preloaded_indicator(coin, interval, indicators)
            return indicators
        except Exception as e:
            logger.error("Indicator error {} ({}): {}", coin, interval, e)
            traceback.print_exc()
            return {"current_price": current_price, "error": str(e)}

    def get_averaged_er(
        self, coin: str, timeframes: list[str] | None = None, period: int = 10
    ) -> float:
        """Calculate averaged Efficiency Ratio across multiple timeframes.
        Default: 3m + 15m, each with 10-candle period.
        Returns averaged ER or 1.0 (no-error fallback) if both fail.
        """
        if timeframes is None:
            timeframes = ["3m", "15m"]

        er_values: list[float] = []
        for tf in timeframes:
            indicators = self.get_technical_indicators(coin, tf)
            if isinstance(indicators, dict) and "efficiency_ratio" in indicators:
                er_val = indicators["efficiency_ratio"]
                if isinstance(er_val, (int, float)) and er_val == er_val:  # NaN check
                    er_values.append(float(er_val))

        if not er_values:
            return 1.0

        avg_er = sum(er_values) / len(er_values)
        logger.debug(
            "Averaged ER for {}: {} (from {} timeframes: {})",
            coin,
            round(avg_er, 4),
            len(er_values),
            timeframes[: len(er_values)],
        )
        return avg_er

    def get_all_real_prices(self) -> dict[str, float]:
        with self._price_cache_lock:
            current_time = time.time()
            if (
                current_time - self._last_price_fetch_time < constants.PRICE_CACHE_TTL_S
                and self._last_price_cache
            ):
                return copy.deepcopy(self._last_price_cache)

            prices: dict[str, float] = {}
            symbols = [f"{coin}USDT" for coin in self.available_coins]

            def _assign_price(symbol: str, raw_price: Any):
                coin = symbol.replace("USDT", "")
                try:
                    price_val = round(float(raw_price), 8)
                    if price_val <= 0 or price_val != price_val or math.isinf(price_val):
                        raise ValueError(f"Invalid price value {price_val}")
                    prices[coin] = price_val
                except Exception as e:
                    logger.warning(
                        "Invalid bulk price for {}: {} ({}). Using fallback.", coin, raw_price, e
                    )
                    prices[coin] = self._get_fallback_price(coin)

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
                    missing = [coin for coin in self.available_coins if coin not in prices]
                    if not missing:
                        prices_str = " | ".join(
                            [f"{coin}: ${val:.4f}" for coin, val in prices.items()]
                        )
                        logger.debug("Prices: {}", prices_str)
                        self._last_price_fetch_time = time.time()
                        self._last_price_cache = copy.deepcopy(prices)
                        return prices
                    logger.warning(
                        "Bulk price missing for: {}. Falling back to individual requests.",
                        ", ".join(missing),
                    )
                else:
                    logger.warning(
                        "Unexpected bulk ticker response format. Falling back to individual requests."
                    )
            except Exception as e:
                logger.warning(
                    "Bulk price fetch failed: {}. Falling back to individual requests.", e
                )

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
                    if price_val <= 0 or price_val != price_val or math.isinf(price_val):
                        raise ValueError(f"Invalid price value {price_val}")
                    prices[coin] = price_val
                except Exception as e:
                    logger.error("{} price error: {}. Using fallback...", coin, e)
                    prices[coin] = self._get_fallback_price(coin)

            if len(prices) > 0:
                prices_str = " | ".join([f"{c}: ${p:.4f}" for c, p in prices.items()])
                logger.debug("Prices: {}", prices_str)
                self._last_price_fetch_time = time.time()
                self._last_price_cache = copy.deepcopy(prices)

            return prices

    def _get_fallback_price(self, coin: str) -> float:
        try:
            df = self.get_real_time_data(coin, interval="1m", limit=1)
            if not df.is_empty() and len(df["close"]) > 0:
                price_val = float(df["close"][-1])
                if price_val > 0 and price_val == price_val:
                    logger.debug("Fallback 1m kline: ${:.4f}", price_val)
                    return price_val
        except Exception as e:
            logger.debug("Fallback 1m failed: {}", e)

        try:
            df = self.get_real_time_data(coin, interval="3m", limit=1)
            if not df.is_empty() and len(df["close"]) > 0:
                price_val = float(df["close"][-1])
                if price_val > 0 and price_val == price_val:
                    logger.debug("Fallback 3m kline: ${:.4f}", price_val)
                    return price_val
        except Exception as e:
            logger.debug("Fallback 3m failed: {}", e)

        try:
            from src.utils import safe_file_read

            cached_prices = safe_file_read("data/portfolio_state.json", default_data={})
            if "positions" in cached_prices:
                for pos_coin, position in cached_prices["positions"].items():
                    if pos_coin == coin and "current_price" in position:
                        cached_price = position["current_price"]
                        if cached_price > 0:
                            logger.debug("Fallback cached: ${:.4f}", cached_price)
                            return cached_price
        except Exception as e:
            logger.debug("Fallback cache failed: {}", e)

        logger.warning("All fallbacks failed for {}. Price set to 0.", coin)
        return 0.0

    def verify_sync_alignment(
        self, coin: str, intervals: list[str] | None = None
    ) -> AlignmentResult:
        if intervals is None:
            intervals = ["3m", "15m", "1h"]
        timestamps = {}

        try:
            for interval in intervals:
                klines = self.get_real_time_data(coin, interval=interval, limit=1)

                if klines is None or klines.is_empty():
                    return AlignmentResult(
                        aligned=False,
                        error_type=AlignmentError.INSUFFICIENT_DATA,
                        error_message=f"No kline data for {coin} @ {interval}",
                    )

                ts_val = klines["timestamp"][-1]
                if isinstance(ts_val, datetime):
                    latest_ts = int(ts_val.replace(tzinfo=timezone.utc).timestamp() * 1000)
                else:
                    latest_ts = int(ts_val)
                timestamps[interval] = latest_ts

            if not timestamps:
                return AlignmentResult(aligned=False, error_type=AlignmentError.INSUFFICIENT_DATA)

            ts_values = list(timestamps.values())
            max_ts = max(ts_values)
            min_ts = min(ts_values)
            delta_ms = max_ts - min_ts
            delta_s = delta_ms / 1000.0

            is_aligned = delta_s <= Config.MAX_ALIGNMENT_DELTA_S

            mismatches = []
            if not is_aligned:
                for interval, ts in timestamps.items():
                    mismatches.append(
                        {
                            "interval": interval,
                            "timestamp": ts,
                            "delta_from_max": (max_ts - ts) / 1000.0,
                        }
                    )

            return AlignmentResult(
                aligned=is_aligned,
                max_delta_seconds=delta_s,
                mismatches=mismatches if not is_aligned else [],
                error_type=AlignmentError.NONE if is_aligned else AlignmentError.EXCESSIVE_MISMATCH,
            )

        except Exception as e:
            return AlignmentResult(
                aligned=False, error_type=AlignmentError.API_FAILURE, error_message=str(e)
            )

    def get_market_sentiment(self, coin: str) -> dict[str, Any]:
        open_interest = self.get_open_interest(coin)
        funding_rate = self.get_funding_rate(coin)
        avg_oi = open_interest
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
        indicators_15m: dict[str, Any] | None = None,
        position_direction: str | None = None,
    ) -> dict[str, Any]:
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

        if not position_direction:
            return {
                "signals": [],
                "score": 0,
                "strength": "NONE",
                "trend_htf": trend_htf,
                "trend_15m": trend_15m,
                "trend_3m": trend_3m,
            }

        if position_direction == "long" and trend_htf == "BEARISH":
            score += 3
            signals.append("htf_bearish_vs_long(+3)")
        elif position_direction == "short" and trend_htf == "BULLISH":
            score += 3
            signals.append("htf_bullish_vs_short(+3)")

        if structure_15m:
            if position_direction == "long" and structure_15m == "LH_LL":
                score += 3
                signals.append("15m_lhll_vs_long(+3)")
            elif position_direction == "short" and structure_15m == "HH_HL":
                score += 3
                signals.append("15m_hhhl_vs_short(+3)")

        if trend_15m:
            if position_direction == "long" and trend_15m == "BEARISH":
                score += 2
                signals.append("15m_bearish_vs_long(+2)")
            elif position_direction == "short" and trend_15m == "BULLISH":
                score += 2
                signals.append("15m_bullish_vs_short(+2)")

        if position_direction == "long" and trend_3m == "BEARISH":
            score += 1
            signals.append("3m_bearish_vs_long(+1)")
        elif position_direction == "short" and trend_3m == "BULLISH":
            score += 1
            signals.append("3m_bullish_vs_short(+1)")

        if rsi_3m is not None:
            if position_direction == "long" and rsi_3m > Config.RSI_OVERBOUGHT_THRESHOLD:
                score += 1
                signals.append(f"rsi_overbought_{rsi_3m:.0f}(+1)")
            elif position_direction == "short" and rsi_3m < Config.RSI_OVERSOLD_THRESHOLD:
                score += 1
                signals.append(f"rsi_oversold_{rsi_3m:.0f}(+1)")

        if macd_3m is not None and macd_signal_3m is not None:
            if position_direction == "long" and macd_3m < macd_signal_3m:
                score += 1
                signals.append("macd_bearish_cross(+1)")
            elif position_direction == "short" and macd_3m > macd_signal_3m:
                score += 1
                signals.append("macd_bullish_cross(+1)")

        ml_consensus = indicators_3m.get("ml_consensus", {})
        if position_direction == "long" and ml_consensus.get("SELL", 0) > 40:
            score += 2
            signals.append(f"ml_reverse_sell({ml_consensus.get('SELL', 0):.0f})(+2)")
        elif position_direction == "short" and ml_consensus.get("BUY", 0) > 40:
            score += 2
            signals.append(f"ml_reverse_buy({ml_consensus.get('BUY', 0):.0f})(+2)")

        if score >= constants.REVERSAL_SCORE_CRITICAL:
            strength = "CRITICAL"
        elif score >= constants.REVERSAL_SCORE_STRONG:
            strength = "STRONG"
        elif score >= constants.REVERSAL_SCORE_WEAK:
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
