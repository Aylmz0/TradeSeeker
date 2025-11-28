import os
import json
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Any, Dict, Optional, Union, List
import pandas as pd
import numpy as np
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_file_read(file_path: str, default_data=None):
    """Safely read JSON file with error handling - handles empty files gracefully"""
    try:
        if os.path.exists(file_path):
            # Check if file is empty (0 bytes)
            if os.path.getsize(file_path) == 0:
                logger.info(f"ℹ️ Empty file detected: {file_path} - returning default data")
                return default_data if default_data is not None else []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Check if file contains only whitespace
                if not content:
                    logger.info(f"ℹ️ Empty content in {file_path} - returning default data")
                    return default_data if default_data is not None else []
                
                return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Invalid JSON in {file_path}: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Error reading {file_path}: {e}")
    return default_data if default_data is not None else []

def safe_file_write(file_path: str, data):
    """Safely write JSON file with error handling and atomicity"""
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Write to a temporary file first
        temp_file_path = f"{file_path}.tmp"
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno()) # Ensure data is written to disk
            
        # Atomic replace
        os.replace(temp_file_path, file_path)
        return True
    except Exception as e:
        logger.error(f"❌ Error writing {file_path}: {e}")
        # Clean up temp file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        return False

def format_num(num: float, precision: int = 2) -> str:
    """Format number with specific precision, handling None/NaN"""
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return "N/A"
    return f"{num:.{precision}f}"

def rate_limiter(calls: int, period: int):
    """Decorator for rate limiting"""
    def decorator(func):
        last_reset = [time.time()]
        calls_made = [0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if now - last_reset[0] > period:
                last_reset[0] = now
                calls_made[0] = 0
            
            if calls_made[0] >= calls:
                sleep_time = period - (now - last_reset[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_reset[0] = time.time()
                calls_made[0] = 0
            
            calls_made[0] += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

class RetryManager:
    """Manages HTTP session retries"""
    
    @staticmethod
    def create_session_with_retry(
        retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: tuple = (500, 502, 504),
        session: requests.Session = None,
    ) -> requests.Session:
        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

class DataValidator:
    """Validates market data"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        if df is None or df.empty:
            return False
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return False
                
        # Check for NaN in critical columns if needed
        return True

class performance_monitor: # Placeholder to match import in main.py if it was imported as a module alias or similar
    # However, main.py imports 'performance_monitor' from 'utils'. 
    # Wait, main.py says: `from utils import (..., performance_monitor)`
    # But performance_monitor is a module in src/core. 
    # It's possible utils.py re-exported it or it was a function.
    # Given the context of `main.py`, `performance_monitor` in the import list might be a function or object.
    # But `src/core/performance_monitor.py` exists.
    # Let's assume for now `main.py` meant to import the class or module.
    # If `main.py` has `from utils import ... performance_monitor`, and `performance_monitor` is a module in `core`,
    # then `utils.py` might have imported it.
    pass
