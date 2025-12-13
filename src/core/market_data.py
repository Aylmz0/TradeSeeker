import requests
import pandas as pd
import numpy as np
import time
import json
import copy
import traceback
from typing import Dict, List, Any, Optional
from config.config import Config
from src.utils import RetryManager

#add this ass a cycle counter analysez              

# HTF_INTERVAL used in main.py, we can get it from Config or define it here
HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'

class RealMarketData:
    """Real market data from Binance Spot and Futures"""

    def __init__(self):
        self.spot_url = "https://api.binance.com/api/v3"
        self.futures_url = "https://fapi.binance.com/fapi/v1"
        self.available_coins = ['XRP', 'DOGE', 'ASTER', 'TRX', 'LINK', 'SOL']
        self.indicator_history_length = 10
        self.session = RetryManager.create_session_with_retry()
        self.preloaded_indicators: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def clear_preloaded_indicators(self):
        """Clear any preloaded indicator snapshots (typically once per cycle)."""
        self.preloaded_indicators = {}

    def store_preloaded_indicator(self, coin: str, interval: str, indicators: Dict[str, Any]):
        """Store a snapshot of indicators for reuse during the same cycle."""
        if not isinstance(indicators, dict):
            return
        coin_store = self.preloaded_indicators.setdefault(coin, {})
        coin_store[interval] = copy.deepcopy(indicators)

    def set_preloaded_indicators(self, cache: Dict[str, Dict[str, Dict[str, Any]]]):
        """Bulk load pre-computed indicator cache (deep copy)."""
        preloaded: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for coin, intervals in (cache or {}).items():
            preloaded[coin] = {}
            for interval, data in (intervals or {}).items():
                if isinstance(data, dict):
                    preloaded[coin][interval] = copy.deepcopy(data)
        self.preloaded_indicators = preloaded

    def get_real_time_data(self, symbol: str, interval: str = '3m', limit: int = 100) -> pd.DataFrame:
        """Get real OHLCV data from Binance Spot with enhanced error handling and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                fetch_limit = limit + self.indicator_history_length + 50
                params = {'symbol': f'{symbol}USDT', 'interval': interval, 'limit': fetch_limit}
                response = self.session.get(f"{self.spot_url}/klines", params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if len(data) < 50:
                    print(f"⚠️ Warning: Insufficient kline data for {symbol} ({interval}). Got {len(data)}.")
                    return pd.DataFrame()

                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Enhanced data validation
                if self._validate_kline_data(df, symbol, interval):
                    return df
                else:
                    print(f"❌ Data validation failed for {symbol} ({interval}) - attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print(f"❌ All retries failed for {symbol} ({interval}). Returning empty DataFrame.")
                        return pd.DataFrame()
                    
            except requests.exceptions.Timeout:
                print(f"❌ Timeout for {symbol} ({interval}) - attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"❌ All retries timed out for {symbol} ({interval})")
                    return pd.DataFrame()
            except Exception as e:
                print(f"❌ Kline data error {symbol} ({interval}) - attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"❌ All retries failed for {symbol} ({interval})")
                    return pd.DataFrame()
        
        return pd.DataFrame()

    def _validate_kline_data(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        """Validate kline data quality with enhanced volume checks"""
        if df.empty:
            print(f"⚠️ Empty DataFrame for {symbol} ({interval})")
            return False
            
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                print(f"⚠️ Invalid price data for {symbol} ({interval}): {col} contains zero/negative values")
                return False
        
        # Check for identical prices (stuck data)
        if df['close'].nunique() < 3:  # Less than 3 unique prices
            print(f"⚠️ Stuck price data for {symbol} ({interval}): only {df['close'].nunique()} unique prices")
            return False
            
        # Enhanced volume validation - check for insufficient volume
        volume_sum = df['volume'].sum()
        volume_mean = df['volume'].mean()
        
        # Check for zero volume
        if volume_sum == 0:
            print(f"⚠️ Zero volume for {symbol} ({interval})")
            return False
            
        # Check for insufficient volume (especially for low-cap coins like ASTR)
        if volume_mean < 1000:  # Minimum average volume threshold
            print(f"⚠️ Insufficient volume for {symbol} ({interval}): avg volume {volume_mean:.0f} < 1000")
            return False
            
        # Check for reasonable price movement
        price_range = df['high'].max() - df['low'].min()
        if price_range == 0:
            print(f"⚠️ No price movement for {symbol} ({interval})")
            return False
            
        return True

    def get_open_interest(self, symbol: str) -> float:
        """Get Latest Open Interest from Binance Futures"""
        try:
            params = {'symbol': f'{symbol}USDT'}
            response = self.session.get(f"{self.futures_url}/openInterest", params=params, timeout=5)
            response.raise_for_status()
            return float(response.json()['openInterest'])
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                print(f"ℹ️ OI not available for {symbol}USDT on Futures.")
            else:
                print(f"❌ OI error for {symbol}: {e}")
            return 0.0

    def get_funding_rate(self, symbol: str) -> float:
        """Get Latest Funding Rate from Binance Futures"""
        try:
            params = {'symbol': f'{symbol}USDT'}
            response = self.session.get(f"{self.futures_url}/premiumIndex", params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list): data = data[0] if data else {}

            rate = data.get('lastFundingRate')
            if rate is not None and rate != '': return float(rate)
            else:
                 # print(f"ℹ️ Using nextFundingRate for {symbol}.")
                 rate = data.get('nextFundingRate')
                 return float(rate) if rate is not None and rate != '' else 0.0
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and (e.response.status_code in [404, 400]):
                 print(f"ℹ️ Funding Rate not available for {symbol}USDT on Futures.")
            else:
                print(f"❌ Funding Rate error for {symbol}: {e}")
            return 0.0

    # --- Indicator Calculation Functions ---
    def calculate_ema_series(self, prices, period): return prices.ewm(span=period, adjust=False).mean()
    def calculate_rsi_series(self, prices, period=14):
        if len(prices) < period + 1: return pd.Series([np.nan] * len(prices))
        delta = prices.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean(); avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100); rsi.loc[avg_gain == 0] = 0
        return rsi
    def calculate_macd_series(self, prices, fast=12, slow=26, signal=9):
        if len(prices) < slow: return pd.Series([np.nan]*len(prices)), pd.Series([np.nan]*len(prices)), pd.Series([np.nan]*len(prices))
        ema_fast = prices.ewm(span=fast, adjust=False).mean(); ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow; macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    def calculate_atr_series(self, df_high, df_low, df_close, period=14):
        if len(df_close) < period + 1: return pd.Series([np.nan] * len(df_close))
        tr0 = abs(df_high - df_low); tr1 = abs(df_high - df_close.shift()); tr2 = abs(df_low - df_close.shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        atr = tr.ewm(com=period - 1, adjust=False).mean()
        return atr



    def _extract_semantic_features(self, prices: pd.Series, period: int = 24) -> Dict[str, Any]:
        """Extract semantic features from price series using numpy"""
        if len(prices) < period:
            return {}
            
        subset = prices.iloc[-period:].values
        
        # 1. Calculate Trend Slope (Linear Regression)
        x = np.arange(len(subset))
        slope, _ = np.polyfit(x, subset, 1)
        slope_pct = (slope / subset[0]) * 100  # Normalize slope as percentage
        
        # 2. Detect Peaks and Valleys (Simple Local Extrema)
        peaks = []
        valleys = []
        for i in range(1, len(subset) - 1):
            if subset[i] > subset[i-1] and subset[i] > subset[i+1]:
                peaks.append(float(subset[i]))
            elif subset[i] < subset[i-1] and subset[i] < subset[i+1]:
                valleys.append(float(subset[i]))
                
        # 3. Determine Volatility State
        std_dev = np.std(subset)
        mean_price = np.mean(subset)
        volatility_ratio = std_dev / mean_price
        
        volatility_state = "STABLE"
        if volatility_ratio > 0.02: volatility_state = "HIGH_VOLATILITY"
        elif volatility_ratio > 0.01: volatility_state = "EXPANDING"
        elif volatility_ratio < 0.005: volatility_state = "COMPRESSED"
        
        # 4. Determine Pattern/Structure
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
            "peaks": peaks[-2:], # Keep last 2 peaks
            "valleys": valleys[-2:], # Keep last 2 valleys
            "volatility_state": volatility_state,
            "structure": structure
        }

    def _generate_smart_sparkline(self, prices: pd.Series, period: int = 24) -> Dict[str, Any]:
        """Generate Smart Sparkline v2.1 with key level, structure, and momentum"""
        if len(prices) < period:
            return {"key_level": None, "structure": "UNCLEAR", "momentum": "STABLE"}
        
        subset = prices.iloc[-period:].values
        current_price = float(subset[-1])
        tolerance_pct = 0.005  # %0.5 tolerance for grouping similar levels
        
        # 1. LOCAL EXTREMA (Tepe ve Dipler) - 2 önceki/sonraki kontrolü
        peaks = []
        valleys = []
        for i in range(2, len(subset) - 2):
            if subset[i] > subset[i-1] and subset[i] > subset[i-2] and \
               subset[i] > subset[i+1] and subset[i] > subset[i+2]:
                peaks.append(float(subset[i]))
            if subset[i] < subset[i-1] and subset[i] < subset[i-2] and \
               subset[i] < subset[i+1] and subset[i] < subset[i+2]:
                valleys.append(float(subset[i]))
        
        # 2. KEY LEVEL TESPİTİ (En yakın support veya resistance)
        key_level = None
        
        # Destek: Fiyatın altındaki en yakın valley
        supports = [v for v in valleys if v < current_price]
        if supports:
            nearest_support = max(supports)
            strength = sum(1 for v in valleys if abs(v - nearest_support) / nearest_support < tolerance_pct)
            distance_pct = (current_price - nearest_support) / current_price * 100
            
            if distance_pct < 2.0:  # %2'den yakınsa önemli
                key_level = {
                    "type": "support",
                    "level": round(nearest_support, 6),
                    "strength": min(strength, 5),
                    "distance_pct": round(distance_pct, 2)
                }
        
        # Direnç: Fiyatın üstündeki en yakın peak (sadece key_level yoksa)
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
                        "distance_pct": round(distance_pct, 2)
                    }
        
        # 3. PRICE STRUCTURE (HH_HL, LH_LL, RANGE, UNCLEAR)
        structure = "UNCLEAR"
        if len(peaks) >= 2 and len(valleys) >= 2:
            last_peaks = peaks[-2:]
            last_valleys = valleys[-2:]
            
            if last_peaks[1] > last_peaks[0] and last_valleys[1] > last_valleys[0]:
                structure = "HH_HL"  # Higher Highs, Higher Lows = Bullish
            elif last_peaks[1] < last_peaks[0] and last_valleys[1] < last_valleys[0]:
                structure = "LH_LL"  # Lower Highs, Lower Lows = Bearish
            else:
                price_range = max(subset) - min(subset)
                if price_range / current_price < 0.015:  # %1.5'den dar
                    structure = "RANGE"
        
        # 4. MOMENTUM (İlk yarı vs Son yarı karşılaştırması)
        mid = len(subset) // 2
        first_half_change = abs(subset[mid] - subset[0]) / subset[0] if subset[0] != 0 else 0
        second_half_change = abs(subset[-1] - subset[mid]) / subset[mid] if subset[mid] != 0 else 0
        
        if first_half_change > 0 and second_half_change > first_half_change * 1.3:
            momentum = "STRENGTHENING"
        elif first_half_change > 0 and second_half_change < first_half_change * 0.7:
            momentum = "WEAKENING"
        else:
            momentum = "STABLE"
        
        # 5. PRICE LOCATION ANALYSIS (v2.2 - Bounce/Pullback Risk Detection)
        # Calculate where current price is within the period's range
        period_high = max(subset)
        period_low = min(subset)
        price_range = period_high - period_low
        
        if price_range > 0:
            # 0 = at period low, 100 = at period high
            percentile = ((current_price - period_low) / price_range) * 100
        else:
            percentile = 50  # No range = middle
        
        # Determine zone
        if percentile <= 10:
            zone = "LOWER_10"  # Bottom 10% - potential bounce zone
        elif percentile >= 90:
            zone = "UPPER_10"  # Top 10% - potential pullback zone
        else:
            zone = "MIDDLE"  # Normal range
        
        price_location = {
            "zone": zone,
            "percentile": round(percentile, 0)
        }
        
        return {
            "key_level": key_level,
            "structure": structure,
            "momentum": momentum,
            "price_location": price_location
        }

    def _calculate_pivots(self, df: pd.DataFrame, periods: int = 24) -> Dict[str, float]:
        """Calculate High/Low pivots over N periods"""
        if len(df) < periods: return {}
        subset = df.iloc[-periods:]
        return {
            'high': float(subset['high'].max()),
            'low': float(subset['low'].min())
        }

    def _generate_tags(self, indicators: Dict[str, Any]) -> List[str]:
        """Generate analytical tags based on indicators"""
        tags = []
        
        # Volatility
        if indicators.get('volume_ratio', 0) > 1.5: tags.append("Vol_High")
        elif indicators.get('volume_ratio', 0) < 0.5: tags.append("Vol_Low")
        
        # Trend (EMA Alignment)
        price = indicators.get('current_price', 0)
        ema20 = indicators.get('ema_20', 0)
        ema50 = indicators.get('ema_50', 0)
        
        if price > ema20 > ema50: tags.append("Trend_Strong_Bull")
        elif price < ema20 < ema50: tags.append("Trend_Strong_Bear")
        elif price > ema20 and price < ema50: tags.append("Trend_Correction_Bull")
        elif price < ema20 and price > ema50: tags.append("Trend_Correction_Bear")
        
        # RSI
        rsi = indicators.get('rsi_14', 50)
        if rsi > 70: tags.append("RSI_Overbought")
        elif rsi < 30: tags.append("RSI_Oversold")
        
        # ATR (Volatility State)
        atr = indicators.get('atr_14', 0)
        if price > 0 and atr / price > 0.02: tags.append("High_Volatility") # >2% ATR
        
        return tags

    def get_technical_indicators(self, coin: str, interval: str) -> Dict[str, Any]:
        """Calculate technical indicators, returning history series"""
        cached = self.preloaded_indicators.get(coin, {}).get(interval)
        if isinstance(cached, dict):
            return copy.deepcopy(cached)

        df = self.get_real_time_data(coin, interval=interval)
        if df.empty or len(df) < 50:
            return {'error': f'Not enough data for {coin} {interval} (got {len(df)})'}

        close_prices = df['close']; current_price = close_prices.iloc[-1]; hist_len = self.indicator_history_length
        indicators = {'current_price': current_price}
        try:
            ema_20_series = self.calculate_ema_series(close_prices, 20); ema_50_series = self.calculate_ema_series(close_prices, 50)
            rsi_14_series = self.calculate_rsi_series(close_prices, 14); macd_line_series, macd_signal_series, macd_hist_series = self.calculate_macd_series(close_prices)
            atr_14_series = self.calculate_atr_series(df['high'], df['low'], df['close'], 14)

            indicators['ema_20'] = ema_20_series.iloc[-1]; indicators['ema_50'] = ema_50_series.iloc[-1]
            indicators['rsi_14'] = rsi_14_series.iloc[-1]; indicators['macd'] = macd_line_series.iloc[-1]
            indicators['macd_signal'] = macd_signal_series.iloc[-1]; indicators['macd_histogram'] = macd_hist_series.iloc[-1]
            indicators['atr_14'] = atr_14_series.iloc[-1] # Keep atr_14 available for AI prompt

            # Use .where(pd.notna, None) to convert NaN to None for JSON
            indicators['ema_20_series'] = ema_20_series.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()
            indicators['rsi_14_series'] = rsi_14_series.iloc[-hist_len:].round(3).where(pd.notna, None).tolist()
            indicators['macd_series'] = macd_line_series.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()

            if interval == '3m':
                 rsi_7_series = self.calculate_rsi_series(close_prices, 7)
                 indicators['rsi_7'] = rsi_7_series.iloc[-1]
                 indicators['rsi_7_series'] = rsi_7_series.iloc[-hist_len:].round(3).where(pd.notna, None).tolist()
            if interval == HTF_INTERVAL:
                 atr_3_series = self.calculate_atr_series(df['high'], df['low'], df['close'], 3)
                 indicators['atr_3'] = atr_3_series.iloc[-1]

            # Volume Analysis: Use last CLOSED candle for consistent ratio
            # iloc[-1] is current incomplete candle. iloc[-2] is last closed candle.
            current_vol = df['volume'].iloc[-1]
            last_closed_vol = df['volume'].iloc[-2]
            
            # Calculate average volume based on LAST 20 CLOSED candles (excluding current partial)
            # We take slice [-21:-1] which gives 20 candles ending at iloc[-2]
            avg_vol_closed = df['volume'].iloc[-21:-1].mean()
            
            indicators['volume'] = current_vol # Keep current volume for AI context
            indicators['last_closed_volume'] = last_closed_vol
            indicators['avg_volume'] = avg_vol_closed if pd.notna(avg_vol_closed) and avg_vol_closed > 0 else 1.0
            
            # Pre-calculate ratio for consistency
            indicators['volume_ratio'] = last_closed_vol / indicators['avg_volume']
            
            # Efficiency Ratio (ER) Calculation for Choppy Regime Detection
            # Using 10 periods (30 mins for 3m interval)
            indicators['efficiency_ratio'] = self.calculate_efficiency_ratio(close_prices, period=10)
            
            indicators['price_series'] = close_prices.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()

            # Enhanced Context Integration (Sparklines, Pivots, Tags)
            # Smart Sparkline v2.1: HTF (1h) gets full data, 15m gets structure+momentum only
            if interval == HTF_INTERVAL:
                indicators['smart_sparkline'] = self._generate_smart_sparkline(close_prices, period=24)
            elif interval == '15m':
                # 15m: structure, momentum, and price_location (no key_level for token efficiency)
                full_sparkline = self._generate_smart_sparkline(close_prices, period=24)
                indicators['smart_sparkline'] = {
                    "structure": full_sparkline.get("structure", "UNCLEAR"),
                    "momentum": full_sparkline.get("momentum", "STABLE"),
                    "price_location": full_sparkline.get("price_location", {"zone": "MIDDLE", "percentile": 50})
                }
            indicators['pivots'] = self._calculate_pivots(df, periods=24)
            indicators['tags'] = self._generate_tags(indicators)

            for key, value in indicators.items():
                if isinstance(value, float) and np.isnan(value): indicators[key] = None
            self.store_preloaded_indicator(coin, interval, indicators)
            return indicators
        except Exception as e:
            print(f"❌ Indicator error {coin} ({interval}): {e}")
            traceback.print_exc()
            return {'current_price': current_price, 'error': str(e)}

    def calculate_efficiency_ratio(self, prices: pd.Series, period: int = 10) -> float:
        """
        Calculate Kaufman Efficiency Ratio (ER) to detect Choppy vs Trending markets.
        ER = Change / Volatility
        Change = |Price(t) - Price(t-n)|
        Volatility = Sum(|Price(i) - Price(i-1)|) for n periods
        
        Returns:
            float: 0.0 to 1.0 (Higher = Trending, Lower = Choppy)
        """
        if len(prices) < period + 1: 
            return 0.5 # Default neutral if insufficient data
        
        # Net change over the period
        change = abs(prices.iloc[-1] - prices.iloc[-period-1])
        
        # Sum of absolute period-to-period changes (Volatility)
        volatility = prices.diff().abs().iloc[-period:].sum()
        
        if volatility == 0: 
            return 1.0 # Theoretical max efficiency (straight line)
            
        return change / volatility

    def get_all_real_prices(self) -> Dict[str, float]:
        """Get real prices for all coins from Spot with enhanced error handling"""
        prices: Dict[str, float] = {}
        symbols = [f"{coin}USDT" for coin in self.available_coins]

        def _assign_price(symbol: str, raw_price: Any):
            coin = symbol.replace("USDT", "")
            try:
                price_val = float(raw_price)
                if price_val <= 0:
                    raise ValueError(f"Non-positive price {price_val}")
                prices[coin] = price_val
            except Exception as e:
                print(f"⚠️ Invalid bulk price for {coin}: {raw_price} ({e}). Using fallback.")
                prices[coin] = self._get_fallback_price(coin)

        # First try batched endpoint (single request, lower latency)
        try:
            response = self.session.get(
                f"{self.spot_url}/ticker/price",
                params={'symbols': json.dumps(symbols, separators=(',', ':'))},
                timeout=3
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                for entry in data:
                    symbol = entry.get('symbol')
                    price_raw = entry.get('price')
                    if symbol and price_raw is not None:
                        _assign_price(symbol, price_raw)
                # Ensure we filled everything; fall back only for missing
                missing = [coin for coin in self.available_coins if coin not in prices]
                if not missing:
                    for coin, val in prices.items():
                        print(f"✅ {coin}: ${val:.4f}")
                    return prices
                else:
                    print(f"⚠️ Bulk price missing for: {', '.join(missing)}. Falling back to individual requests.")
            else:
                print("⚠️ Unexpected bulk ticker response format. Falling back to individual requests.")
        except Exception as e:
            print(f"⚠️ Bulk price fetch failed: {e}. Falling back to individual requests.")

        # Fallback to individual calls (still using session, without artificial delay)
        for coin in self.available_coins:
            try:
                response = self.session.get(
                    f"{self.spot_url}/ticker/price",
                    params={'symbol': f"{coin}USDT"},
                    timeout=3
                )
                response.raise_for_status()
                data = response.json()
                price_val = float(data.get('price', 0))
                if price_val <= 0:
                    raise ValueError(f"Non-positive price {price_val}")
                prices[coin] = price_val
                print(f"✅ {coin}: ${price_val:.4f}")
            except Exception as e:
                print(f"❌ {coin} price error: {e}. Using fallback...")
                prices[coin] = self._get_fallback_price(coin)
                
        return prices

    def _get_fallback_price(self, coin: str) -> float:
        """Get fallback price using multiple methods"""
        # Method 1: Try 1m kline data
        try:
            df = self.get_real_time_data(coin, interval='1m', limit=1)
            if not df.empty and not df['close'].empty:
                price_val = df['close'].iloc[-1]
                if price_val > 0 and pd.notna(price_val):
                    print(f"   Fallback 1m kline: ${price_val:.4f}")
                    return price_val
        except Exception as e:
            print(f"   Fallback 1m failed: {e}")
        
        # Method 2: Try 3m kline data
        try:
            df = self.get_real_time_data(coin, interval='3m', limit=1)
            if not df.empty and not df['close'].empty:
                price_val = df['close'].iloc[-1]
                if price_val > 0 and pd.notna(price_val):
                    print(f"   Fallback 3m kline: ${price_val:.4f}")
                    return price_val
        except Exception as e:
            print(f"   Fallback 3m failed: {e}")
        
        # Method 3: Use cached price from previous cycle
        try:
            from src.utils import safe_file_read
            cached_prices = safe_file_read("data/portfolio_state.json", default_data={})
            if 'positions' in cached_prices:
                for pos_coin, position in cached_prices['positions'].items():
                    if pos_coin == coin and 'current_price' in position:
                        cached_price = position['current_price']
                        if cached_price > 0:
                            print(f"   Fallback cached: ${cached_price:.4f}")
                            return cached_price
        except Exception as e:
            print(f"   Fallback cache failed: {e}")
        
        # Final fallback: return 0 with warning
        print(f"   ⚠️ All fallbacks failed for {coin}. Price set to 0.")
        return 0.0

    def get_market_sentiment(self, coin: str) -> Dict[str, Any]:
        """Get Open Interest and Funding Rate (Nof1ai format)"""
        open_interest = self.get_open_interest(coin)
        funding_rate = self.get_funding_rate(coin)
        
        # Nof1ai format: "Latest: X Average: Y" for Open Interest
        avg_oi = open_interest  # Simplified average calculation
        return {
            'open_interest': open_interest,
            'open_interest_avg': avg_oi,
            'funding_rate': funding_rate
        }

    def detect_trend_reversal_signals(self, coin: str, indicators_3m: Dict[str, Any], indicators_htf: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential trend reversal signals based on multi-timeframe analysis.
        Centralized logic to be used by both PerformanceMonitor and AI Prompt Builder.
        
        Args:
            coin: Coin symbol
            indicators_3m: 3m indicators
            indicators_htf: HTF indicators (1h/4h)
            
        Returns:
            Dictionary containing reversal signals and strength
        """
        signals = []
        
        if not indicators_3m or not indicators_htf:
            return {'signals': [], 'strength': 'NONE'}
            
        # Extract indicators
        price_3m = indicators_3m.get('current_price')
        ema20_3m = indicators_3m.get('ema_20')
        rsi_3m = indicators_3m.get('rsi_14')
        macd_3m = indicators_3m.get('macd')
        macd_signal_3m = indicators_3m.get('macd_signal')
        
        price_htf = indicators_htf.get('current_price')
        ema20_htf = indicators_htf.get('ema_20')
        
        if None in [price_3m, ema20_3m, rsi_3m, macd_3m, macd_signal_3m, price_htf, ema20_htf]:
            return {'signals': [], 'strength': 'NONE'}
            
        # Determine trends with neutral band (matches portfolio_manager.py logic)
        def _determine_trend(price: float, ema20: float) -> str:
            if ema20 == 0:
                return "UNKNOWN"
            delta = (price - ema20) / ema20
            if abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT:
                return "NEUTRAL"
            return "BULLISH" if delta > 0 else "BEARISH"
        
        trend_3m = _determine_trend(price_3m, ema20_3m)
        trend_htf = _determine_trend(price_htf, ema20_htf)
        
        # 1. Trend Conflict (HTF vs 3m)
        if trend_htf != trend_3m:
            signals.append(f"Trend Conflict: HTF {trend_htf} vs 3m {trend_3m}")
            
        # 2. RSI Extremes (Counter to HTF trend)
        # If HTF Bullish, look for Overbought (potential top)
        if trend_htf == "BULLISH" and rsi_3m > Config.RSI_OVERBOUGHT_THRESHOLD:
            signals.append(f"RSI Overbought ({rsi_3m:.1f}) in Bullish Trend")
        # If HTF Bearish, look for Oversold (potential bottom)
        elif trend_htf == "BEARISH" and rsi_3m < Config.RSI_OVERSOLD_THRESHOLD:
            signals.append(f"RSI Oversold ({rsi_3m:.1f}) in Bearish Trend")
            
        # 3. MACD Divergence (Counter to HTF trend)
        # If HTF Bullish, look for Bearish MACD Cross
        if trend_htf == "BULLISH" and macd_3m < macd_signal_3m:
            signals.append("MACD Bearish Cross in Bullish Trend")
        # If HTF Bearish, look for Bullish MACD Cross
        elif trend_htf == "BEARISH" and macd_3m > macd_signal_3m:
            signals.append("MACD Bullish Cross in Bearish Trend")
            
        # Determine Signal Strength
        strength = "NONE"
        if len(signals) >= 3:
            strength = "HIGH_LOSS_RISK"
        elif len(signals) >= 2:
            strength = "MEDIUM_LOSS_RISK"
        elif len(signals) >= 1:
            strength = "LOW_LOSS_RISK"
            
        return {
            'signals': signals,
            'strength': strength,
            'trend_htf': trend_htf,
            'trend_3m': trend_3m
        }
