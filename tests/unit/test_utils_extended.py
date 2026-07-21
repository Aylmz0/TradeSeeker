"""Tests for src/utils.py — extended coverage for utility functions."""

import os
import time

import polars as pl
import pytest
import requests

from src.utils import (
    DataValidator,
    RetryManager,
    cleanup_stale_temp_files,
    rate_limiter,
    safe_file_read_cached,
)


# ---------------------------------------------------------------------------
# rate_limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_rate_limiter_allows_calls(self):
        call_count = 0

        @rate_limiter(calls=5, period=1)
        def inc():
            nonlocal call_count
            call_count += 1

        for _ in range(3):
            inc()

        assert call_count == 3

    def test_rate_limiter_blocks_and_sleeps(self):
        @rate_limiter(calls=2, period=1)
        def do_nothing():
            pass

        # Exhaust the rate limit
        do_nothing()
        do_nothing()

        start = time.time()
        do_nothing()  # This should block then reset
        elapsed = time.time() - start

        assert elapsed >= 0.9


# ---------------------------------------------------------------------------
# RetryManager
# ---------------------------------------------------------------------------


class TestRetryManager:
    def test_create_session_returns_session(self):
        session = RetryManager.create_session_with_retry()
        assert isinstance(session, requests.Session)
        assert "http://" in session.adapters or "https://" in session.adapters

    def test_create_session_configures_retries(self):
        session = RetryManager.create_session_with_retry(retries=5)
        adapter = session.get_adapter("https://example.com")
        assert adapter.max_retries.total == 5


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------


class TestDataValidator:
    def test_validate_dataframe_valid(self, sample_ohlcv):
        assert DataValidator.validate_dataframe(sample_ohlcv) is True

    def test_validate_dataframe_with_required_columns(self, sample_ohlcv):
        assert DataValidator.validate_dataframe(sample_ohlcv, ["open", "close"]) is True

    def test_validate_dataframe_none(self):
        assert DataValidator.validate_dataframe(None) is False

    def test_validate_dataframe_empty(self):
        empty = pl.DataFrame({"a": []})
        assert DataValidator.validate_dataframe(empty) is False

    def test_validate_dataframe_missing_columns(self, sample_ohlcv):
        assert DataValidator.validate_dataframe(sample_ohlcv, ["nonexistent"]) is False


# ---------------------------------------------------------------------------
# cleanup_stale_temp_files
# ---------------------------------------------------------------------------


class TestCleanupStaleTempFiles:
    def test_cleans_temp_files(self, temp_dir):
        tmp1 = os.path.join(temp_dir, "data1.123.456.tmp")
        tmp2 = os.path.join(temp_dir, "data2.789.012.tmp")
        for p in (tmp1, tmp2):
            with open(p, "w") as f:
                f.write("temp")

        # Place .tmp files in cwd (where cleanup scans)
        os.chdir(temp_dir)
        try:
            count = cleanup_stale_temp_files()
        finally:
            os.chdir("/home/yilmaz/projects/TradeSeeker")

        # Both .tmp files should be removed
        assert not os.path.exists(tmp1)
        assert not os.path.exists(tmp2)

    def test_no_temp_files_returns_zero(self, tmp_path):
        os.chdir(tmp_path)
        try:
            count = cleanup_stale_temp_files()
        finally:
            os.chdir("/home/yilmaz/projects/TradeSeeker")
        assert count == 0


# ---------------------------------------------------------------------------
# safe_file_read_cached
# ---------------------------------------------------------------------------


class TestSafeFileReadCached:
    def test_cache_miss_returns_data(self, write_test_json):
        path = write_test_json("test.json", {"key": "value"})
        result = safe_file_read_cached(path)
        assert result == {"key": "value"}

    def test_cache_hit_returns_same_data(self, write_test_json):
        path = write_test_json("cached.json", {"x": 1})
        first = safe_file_read_cached(path)
        second = safe_file_read_cached(path)
        assert first == second
        # Mutating one should not affect the other (deep copy)
        first["x"] = 999
        assert second["x"] == 1

    def test_cache_miss_after_mtime_change(self, write_test_json):
        path = write_test_json("mtime.json", {"v": 1})
        safe_file_read_cached(path)
        # Overwrite file to change mtime
        with open(path, "w") as f:
            import json

            json.dump({"v": 2}, f)
        result = safe_file_read_cached(path)
        assert result == {"v": 2}

    def test_missing_file_returns_default(self):
        result = safe_file_read_cached("/no/such/file.json", default_data="fallback")
        assert result == "fallback"

    def test_missing_file_returns_empty_dict_by_default(self):
        result = safe_file_read_cached("/no/such/file.json")
        assert result == {}

    def test_invalid_json_returns_default(self, temp_dir):
        path = os.path.join(temp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json!!!")
        result = safe_file_read_cached(path, default_data="err")
        assert result == "err"


# ---------------------------------------------------------------------------
# safe_file_read (invalid JSON path — bonus coverage)
# ---------------------------------------------------------------------------


class TestSafeFileReadInvalidJson:
    def test_invalid_json_returns_default(self, temp_dir):
        from src.utils import safe_file_read

        path = os.path.join(temp_dir, "invalid.json")
        with open(path, "w") as f:
            f.write("NOT JSON{{{")
        result = safe_file_read(path, default_data="fallback")
        assert result == "fallback"

    def test_empty_file_returns_default(self, temp_dir):
        from src.utils import safe_file_read

        path = os.path.join(temp_dir, "empty.json")
        with open(path, "w") as f:
            f.write("")
        result = safe_file_read(path, default_data=[])
        assert result == []
