"""
Utility functions for the Alpha Arena DeepSeek bot.
Includes rate limiting, retry mechanisms, and common helpers.
"""
import time
import logging
import asyncio
from typing import Any, Callable, Optional
from functools import wraps
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import Config

class RateLimiter:
    """Rate limiting utility for API calls."""
    
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls outside the current period
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.period]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    logging.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    # Update calls after sleep
                    self.calls = [call_time for call_time in self.calls 
                                 if now + sleep_time - call_time < self.period]
            
            self.calls.append(time.time())
            return func(*args, **kwargs)
        return wrapper

class RetryManager:
    """Retry mechanism for API calls with exponential backoff."""
    
    @staticmethod
    def create_session_with_retry() -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=Config.MAX_RETRY_ATTEMPTS,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @staticmethod
    def retry_with_backoff(func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry and exponential backoff."""
        last_exception = None
        
        for attempt in range(Config.MAX_RETRY_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < Config.MAX_RETRY_ATTEMPTS:
                    sleep_time = (2 ** attempt) + (attempt * 0.1)
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"All {Config.MAX_RETRY_ATTEMPTS} attempts failed")
                    raise last_exception
        
        raise last_exception

class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_price_data(prices: dict) -> dict:
        """Validate and clean price data."""
        valid_prices = {}
        for coin, price in prices.items():
            if (isinstance(price, (int, float)) and 
                price > 0 and 
                not (hasattr(float, 'isnan') and float.isnan(price))):
                valid_prices[coin] = price
            else:
                logging.warning(f"Invalid price data for {coin}: {price}")
        
        return valid_prices
    
    @staticmethod
    def validate_trading_decision(decision: dict) -> bool:
        """Validate trading decision structure."""
        required_fields = ['signal']
        valid_signals = ['buy_to_enter', 'sell_to_enter', 'close_position', 'hold']
        
        if not isinstance(decision, dict):
            return False
        
        if 'signal' not in decision:
            return False
        
        if decision['signal'] not in valid_signals:
            logging.warning(f"Invalid signal: {decision['signal']}")
            return False
        
        # Validate entry signals
        if decision['signal'] in ['buy_to_enter', 'sell_to_enter']:
            if 'leverage' not in decision:
                logging.warning("Entry signal missing leverage")
                return False
            
            leverage = decision.get('leverage', 1)
            if not isinstance(leverage, (int, float)) or leverage < 1:
                logging.warning(f"Invalid leverage: {leverage}")
                return False
            
            if leverage > Config.MAX_LEVERAGE:
                logging.warning(f"Leverage {leverage} exceeds maximum {Config.MAX_LEVERAGE}")
                return False
        
        return True

class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start a timer for a specific operation."""
        self.metrics[name] = {'start': time.time()}
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return elapsed time."""
        if name in self.metrics and 'start' in self.metrics[name]:
            elapsed = time.time() - self.metrics[name]['start']
            self.metrics[name]['elapsed'] = elapsed
            self.metrics[name]['end'] = time.time()
            return elapsed
        return 0.0
    
    def get_metrics(self) -> dict:
        """Get all performance metrics."""
        return self.metrics.copy()

# Global instances
rate_limiter = RateLimiter(max_calls=10, period=60)  # 10 calls per minute
performance_monitor = PerformanceMonitor()

def format_num(val, precision=4) -> str:
    """Formats a number, handling None or NaN."""
    if val is None:
        return 'N/A'
    try:
        float_val = float(val)
        if float_val != float_val:  # Check for NaN
            return 'N/A'
        return f"{float_val:.{precision}f}"
    except (ValueError, TypeError):
        return 'N/A'

def safe_file_write(file_path: str, data: Any):
    """Safely write data to a JSON file using file locking."""
    import json
    import os
    import fcntl
    
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name: 
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Critical Error saving file {file_path}: {e}")

def safe_file_read(file_path: str, default_data: Any = None) -> Any:
    """Safely read data from a JSON file using file locking."""
    import json
    import os
    import fcntl
    
    try:
        if not os.path.exists(file_path): 
            return default_data
        with open(file_path, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            content = f.read()
            if not content: 
                return default_data
            data = json.loads(content)
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.info(f"Info reading file {file_path}: {e}. Returning default.")
        return default_data
    except Exception as e:
        logging.error(f"Critical Error reading file {file_path}: {e}")
        return default_data
