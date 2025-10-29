"""
Cache management for performance optimization.
Reduces redundant API calls and calculations.
"""
import time
import json
import hashlib
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import logging

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
