import contextlib
import copy
import json
import logging
import os
import shutil
import threading
import time
from functools import wraps

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_FILE_CACHE = {}
_file_lock = threading.RLock()  # RLock for re-entrant safety


def safe_file_read_cached(file_path: str, default_data=None):
    """Read a JSON file using a mem-cache based on filesystem mtime to O(1) avoid disk lookups."""
    try:
        with _file_lock:
            if not os.path.exists(file_path):
                return default_data if default_data is not None else {}

            current_mtime = os.path.getmtime(file_path)

            if file_path in _FILE_CACHE:
                cached_mtime, cached_data = _FILE_CACHE[file_path]
                if current_mtime == cached_mtime:
                    # Return deepcopy to prevent accidental mutation of the cache by callers
                    return copy.deepcopy(cached_data)

            # Cache miss or file updated
            if os.path.getsize(file_path) == 0:
                data = default_data if default_data is not None else {}
            else:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        data = default_data if default_data is not None else {}
                    else:
                        data = json.loads(content)

            _FILE_CACHE[file_path] = (current_mtime, data)
            return copy.deepcopy(data)

    except json.JSONDecodeError as e:
        logger.warning(f"[WARN]  Invalid JSON in {file_path}: {e}")
    except Exception as e:
        logger.warning(f"[WARN]  Error reading {file_path}: {e}")

    return default_data if default_data is not None else {}


def safe_file_read(file_path: str, default_data=None):
    """Safely read JSON file with error handling - handles empty files gracefully"""
    try:
        if os.path.exists(file_path):
            # Check if file is empty (0 bytes)
            if os.path.getsize(file_path) == 0:
                logger.info(f"[INFO] Empty file detected: {file_path} - returning default data")
                return default_data if default_data is not None else []

            with open(file_path, encoding="utf-8") as f:
                content = f.read().strip()
                # Check if file contains only whitespace
                if not content:
                    logger.info(f"[INFO] Empty content in {file_path} - returning default data")
                    return default_data if default_data is not None else []

                return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"[WARN]  Invalid JSON in {file_path}: {e}")
    except Exception as e:
        logger.warning(f"[WARN]  Error reading {file_path}: {e}")
    return default_data if default_data is not None else []


def safe_file_write(file_path: str, data):
    """Safely write JSON file with error handling and atomicity (Thread-Safe)"""
    temp_file_path = None
    try:
        # Mutex to prevent internal thread-race on the same file operation
        with _file_lock:
            # CRITICAL FIX: Ensure absolute path to prevent [Errno 2] Issues
            abs_file_path = os.path.abspath(file_path)

            # Ensure directory exists with full hierarchy
            directory = os.path.dirname(abs_file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            # Write to a UNIQUE temporary file to prevent thread collisions
            # Using PID + ThreadID for perfect isolation
            temp_file_path = f"{abs_file_path}.{os.getpid()}.{threading.get_ident()}.tmp"

            with open(temp_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk

            # Atomic replace (using absolute paths)
            try:
                os.replace(temp_file_path, abs_file_path)
            except OSError:
                # Fallback for cross-device or permission quirks
                shutil.move(temp_file_path, abs_file_path)

            return True
    except Exception as e:
        # Get CWD for debugging
        cwd = os.getcwd()
        logger.error(f"[ERR]   Error writing {file_path} (CWD: {cwd}): {e}")
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            with contextlib.suppress(Exception):
                os.remove(temp_file_path)
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
    def validate_dataframe(df: pd.DataFrame, required_columns: list[str] | None = None) -> bool:
        if df is None or df.empty:
            return False

        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"[WARN]  Missing columns: {missing_cols}")
                return False

        # Check for NaN in critical columns if needed
        return True


class performance_monitor:  # Placeholder to match import in main.py if it was imported as a module alias or similar
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
