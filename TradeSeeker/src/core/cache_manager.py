"""
Cache management for performance optimization.
Reduces redundant API calls and calculations.
"""
import time
import json
import hashlib
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class CacheManager:
    """Manages caching for API responses and calculations."""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default TTL
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate a unique cache key based on function name and arguments."""
        key_data = f"{func_name}:{json.dumps(args, sort_keys=True)}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            cache_entry = self.cache[key]
            if time.time() < cache_entry['expires_at']:
                self.hit_count += 1
                return cache_entry['value']
            else:
                # Remove expired entry
                del self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl
        
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
    
    def cached(self, ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Skip caching for certain functions or conditions
                if kwargs.get('skip_cache', False):
                    return func(*args, **kwargs)
                
                key = self._generate_key(func.__name__, *args, **kwargs)
                cached_result = self.get(key)
                
                if cached_result is not None:
                    logging.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result
            return wrapper
        return decorator
    
    def clear_expired(self) -> int:
        """Clear expired cache entries and return number cleared."""
        current_time = time.time()
        expired_keys = [key for key, entry in self.cache.items() 
                       if current_time >= entry['expires_at']]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate_percent': hit_rate,
            'memory_usage_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache in bytes."""
        return sum(len(json.dumps(entry).encode('utf-8')) for entry in self.cache.values())
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    def __init__(self):
        self.cache_manager = CacheManager()
    
    @staticmethod
    def batch_api_calls(api_calls: list, batch_size: int = 10) -> list:
        """Batch API calls to reduce overhead."""
        results = []
        for i in range(0, len(api_calls), batch_size):
            batch = api_calls[i:i + batch_size]
            # In a real implementation, you would make parallel requests here
            batch_results = [call() for call in batch]
            results.extend(batch_results)
        return results
    
    @staticmethod
    def optimize_dataframe_operations(df, operations: list) -> Any:
        """Optimize pandas DataFrame operations."""
        # Apply multiple operations in a single pass when possible
        result = df.copy()
        for operation in operations:
            if operation['type'] == 'filter':
                result = result.query(operation['query'])
            elif operation['type'] == 'transform':
                result[operation['column']] = result[operation['column']].apply(operation['function'])
            elif operation['type'] == 'aggregate':
                result = getattr(result, operation['method'])(**operation.get('kwargs', {}))
        
        return result
    
    def memoize_technical_indicators(self, symbol: str, interval: str, 
                                   calculation_func: callable) -> Any:
        """Memoize technical indicator calculations."""
        cache_key = f"indicators_{symbol}_{interval}"
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        result = calculation_func()
        # Cache technical indicators for shorter time (1 minute)
        self.cache_manager.set(cache_key, result, ttl=60)
        return result


# Global cache instance
cache_manager = CacheManager()
performance_optimizer = PerformanceOptimizer()


def async_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for async functions with retry logic."""
    import asyncio
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logging.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
                        raise last_exception
            raise last_exception
        return wrapper
    return decorator


def time_execution(func):
    """Decorator to measure function execution time."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        if execution_time > 1.0:  # Log only slow operations
            logging.warning(f"{func.__name__} took {execution_time:.2f}s to execute")
        
        return result
    return wrapper


def fetch_all_indicators_parallel(
    market_data_instance,
    available_coins: list,
    htf_interval: str = '1h'
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """
    Fetch all indicators for all coins in parallel.
    
    Args:
        market_data_instance: Instance of RealMarketData with get_technical_indicators method
        available_coins: List of coin symbols to fetch indicators for
        htf_interval: Higher timeframe interval (default: '1h')
    
    Returns:
        Tuple of (all_indicators, all_sentiment) where:
        - all_indicators: {coin: {interval: indicators}}
        - all_sentiment: {coin: sentiment}
    
    Example:
        >>> from cache_manager import fetch_all_indicators_parallel
        >>> indicators, sentiment = fetch_all_indicators_parallel(
        ...     market_data, ['BTC', 'ETH'], '1h'
        ... )
    """
    print("🔄 Fetching all indicators in parallel...")
    start_time = time.time()
    
    all_indicators = {}  # {coin: {interval: indicators}}
    all_sentiment = {}   # {coin: sentiment}
    
    def fetch_indicators_for_coin(coin: str) -> tuple:
        """Fetch all indicators and sentiment for a single coin"""
        try:
            indicators_3m = market_data_instance.get_technical_indicators(coin, '3m')
            indicators_15m = market_data_instance.get_technical_indicators(coin, '15m')
            indicators_htf = market_data_instance.get_technical_indicators(coin, htf_interval)
            sentiment = market_data_instance.get_market_sentiment(coin)
            return (coin, {
                '3m': indicators_3m,
                '15m': indicators_15m,
                htf_interval: indicators_htf
            }, sentiment)
        except Exception as e:
            print(f"⚠️ Error fetching indicators for {coin}: {e}")
            return (coin, {
                '3m': {'error': str(e)},
                '15m': {'error': str(e)},
                htf_interval: {'error': str(e)}
            }, {})
    
    # Fetch all indicators in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_indicators_for_coin, coin): coin 
                  for coin in available_coins}
        
        for future in as_completed(futures):
            coin, indicators, sentiment = future.result()
            all_indicators[coin] = indicators
            all_sentiment[coin] = sentiment
    
    elapsed = time.time() - start_time
    print(f"✅ Fetched all indicators in {elapsed:.2f}s (parallel)")
    
    return all_indicators, all_sentiment


class SmartIndicatorCache:
    """
    Smart TTL cache for technical indicators.
    
    Strategy:
    - 3m candles: NEVER cached (cycle 4min > candle 3min, always fresh data needed)
    - 15m candles: Cached with dynamic TTL (~75% hit rate)
    - HTF candles: Cached with dynamic TTL (~93% hit rate for 1h)
    
    Example:
        >>> cache = SmartIndicatorCache(htf_interval='1h')
        >>> indicators = cache.get_indicators('BTC', '15m', market_data)
        💾 BTC 15m: Cache HIT (or MISS)
    """
    
    def __init__(self, htf_interval: str = '1h'):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.htf_interval = htf_interval
        
        # Only cache 15m and HTF (NOT 3m)
        self.cacheable_intervals = ['15m', htf_interval]
        
        # Stats tracking
        self.hits = 0
        self.misses = 0
        self.api_calls_saved = 0
    
    def get_indicators(
        self, 
        coin: str, 
        interval: str, 
        market_data_instance,
        force_fresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get indicators with smart caching.
        
        Args:
            coin: Coin symbol (e.g., 'BTC')
            interval: Candle interval ('3m', '15m', htf_interval)
            market_data_instance: Instance with get_technical_indicators method
            force_fresh: Skip cache, always fetch fresh (default: False)
        
        Returns:
            Dict with technical indicators
        """
        # 3m NEVER cached (always fresh)
        if interval not in self.cacheable_intervals or force_fresh:
            if interval == '3m':
                # Normal behavior for 3m
                return market_data_instance.get_technical_indicators(coin, interval)
            else:
                # Force fresh for cacheable intervals
                print(f"🌐 {coin} {interval}: Force fresh (cache bypass)")
                data = market_data_instance.get_technical_indicators(coin, interval)
                # Update cache even when forced fresh
                if interval in self.cacheable_intervals:
                    ttl = self._calculate_smart_ttl(interval)
                    self._set_to_cache(coin, interval, data, ttl)
                return data
        
        # Try cache first
        cache_key = self._generate_key(coin, interval)
        cached = self._get_from_cache(cache_key)
        
        if cached:
            self.hits += 1
            self.api_calls_saved += 1
            print(f"💾 {coin} {interval}: Cache HIT")
            return cached
        
        # Cache miss, fetch from API
        self.misses += 1
        print(f"🌐 {coin} {interval}: Cache MISS, fetching from API")
        fresh_data = market_data_instance.get_technical_indicators(coin, interval)
        
        # Cache with smart TTL
        ttl = self._calculate_smart_ttl(interval)
        self._set_to_cache(coin, interval, fresh_data, ttl)
        print(f"💾 {coin} {interval}: Cached (TTL={ttl}s)")
        
        return fresh_data
    
    def _calculate_smart_ttl(self, interval: str) -> int:
        """
        Calculate TTL based on time until next candle close.
        
        Formula: TTL = (time_to_next_candle) * 0.85 (safety margin)
        
        Example:
            15m candle at 00:07 → next close at 00:15 → 8min remaining → TTL=408s
        """
        # Interval to seconds
        interval_map = {
            '15m': 900,   # 15 minutes
            '30m': 1800,  # 30 minutes
            '1h': 3600,   # 1 hour
            '2h': 7200,   # 2 hours
            '4h': 14400   # 4 hours
        }
        
        interval_seconds = interval_map.get(interval, 900)
        current_time = time.time()
        
        # Calculate when current candle started
        candle_start_time = (current_time // interval_seconds) * interval_seconds
        
        # Calculate when it will close
        candle_close_time = candle_start_time + interval_seconds
        
        # Remaining time until close
        remaining_seconds = candle_close_time - current_time
        
        # Apply 85% safety margin (refresh 15% before actual close)
        ttl = int(remaining_seconds * 0.85)
        
        # Minimum 30 seconds (safety guard)
        return max(ttl, 30)
    
    def _generate_key(self, coin: str, interval: str) -> str:
        """Generate cache key: coin_interval"""
        return f"{coin}_{interval}"
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from cache if not expired"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() > entry['expires_at']:
            # Remove expired entry
            del self.cache[key]
            return None
        
        return entry['data']
    
    def _set_to_cache(self, coin: str, interval: str, data: Dict[str, Any], ttl: int):
        """Set data to cache with TTL"""
        key = self._generate_key(coin, interval)
        
        self.cache[key] = {
            'data': data,
            'expires_at': time.time() + ttl,
            'cached_at': time.time(),
            'ttl': ttl
        }
    
    def clear_expired(self) -> int:
        """Clear expired cache entries, return count of cleared entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            print(f"🧹 Cleared {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_pct': round(hit_rate, 2),
            'api_calls_saved': self.api_calls_saved,
            'memory_usage_kb': self._estimate_memory_usage() / 1024
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        return sum(
            len(json.dumps(entry).encode('utf-8')) 
            for entry in self.cache.values()
        )
    
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("💾 SMART CACHE STATISTICS")
        print("="*50)
        print(f"Cache Entries:     {stats['total_entries']}")
        print(f"Hits:              {stats['hits']}")
        print(f"Misses:            {stats['misses']}")
        print(f"Hit Rate:          {stats['hit_rate_pct']}%")
        print(f"API Calls Saved:   {stats['api_calls_saved']}")
        print(f"Memory Usage:      {stats['memory_usage_kb']:.2f} KB")
        print("="*50 + "\n")
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.hits = 0
        self.misses = 0
        self.api_calls_saved = 0
    
    def clear_all(self):
        """Clear all cache and reset stats"""
        self.cache.clear()
        self.reset_stats()
        print("🧹 Cache cleared completely")


# Global cache instance (will be initialized with HTF from config)
_global_cache_instance = None

def get_smart_cache(htf_interval: str = '1h') -> SmartIndicatorCache:
    """Get or create global cache instance"""
    global _global_cache_instance
    if _global_cache_instance is None or _global_cache_instance.htf_interval != htf_interval:
        _global_cache_instance = SmartIndicatorCache(htf_interval)
    return _global_cache_instance


def fetch_all_indicators_with_cache(
    market_data_instance,
    available_coins: list,
    htf_interval: str = '1h',
    use_cache: bool = True
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """
    Fetch all indicators with smart caching support.
    
    Args:
        market_data_instance: Instance of RealMarketData
        available_coins: List of coins to fetch
        htf_interval: Higher timeframe interval (from .env HTF_INTERVAL)
        use_cache: Whether to use cache (default: True)
    
    Returns:
        Tuple of (all_indicators, all_sentiment)
    
    Cache Strategy:
        - 3m: Always fresh (NEVER cached)
        - 15m: Cached with smart TTL (~75% hit rate)
        - HTF: Cached with smart TTL (~93% hit rate for 1h)
    
    Example:
        >>> # With cache (recommended):
        >>> indicators, sentiment = fetch_all_indicators_with_cache(
        ...     market_data, ['BTC', 'ETH'], '1h', use_cache=True
        ... )
        
        >>> # Without cache (force fresh):
        >>> indicators, sentiment = fetch_all_indicators_with_cache(
        ...     market_data, ['BTC', 'ETH'], '1h', use_cache=False
        ... )
    """
    if not use_cache:
        # Fallback to non-cached version
        return fetch_all_indicators_parallel(
            market_data_instance,
            available_coins,
            htf_interval
        )
    
    print("🔄 Fetching indicators with smart cache...")
    start_time = time.time()
    
    # Get global cache instance
    cache = get_smart_cache(htf_interval)
    
    all_indicators = {}
    all_sentiment = {}
    
    def fetch_indicators_for_coin_cached(coin: str) -> tuple:
        """Fetch indicators with cache support"""
        try:
            # 3m: Always fresh (not cached)
            indicators_3m = market_data_instance.get_technical_indicators(coin, '3m')
            
            # 15m: Use cache
            indicators_15m = cache.get_indicators(coin, '15m', market_data_instance)
            
            # HTF: Use cache
            indicators_htf = cache.get_indicators(coin, htf_interval, market_data_instance)
            
            # Sentiment: Always fresh (quick call)
            sentiment = market_data_instance.get_market_sentiment(coin)
            
            return (coin, {
                '3m': indicators_3m,
                '15m': indicators_15m,
                htf_interval: indicators_htf
            }, sentiment)
        except Exception as e:
            print(f"⚠️ Error fetching indicators for {coin}: {e}")
            return (coin, {
                '3m': {'error': str(e)},
                '15m': {'error': str(e)},
                htf_interval: {'error': str(e)}
            }, {})
    
    # Fetch in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_indicators_for_coin_cached, coin): coin 
                  for coin in available_coins}
        
        for future in as_completed(futures):
            coin, indicators, sentiment = future.result()
            all_indicators[coin] = indicators
            all_sentiment[coin] = sentiment
    
    elapsed = time.time() - start_time
    
    # Print stats
    stats = cache.get_stats()
    cache_info = f"Cache: {stats['hits']}/{stats['hits']+stats['misses']} hits ({stats['hit_rate_pct']}%)"
    print(f"✅ Fetched in {elapsed:.2f}s (parallel + cache) | {cache_info}")
    
    # Periodically clear expired entries
    cache.clear_expired()
    
    return all_indicators, all_sentiment
