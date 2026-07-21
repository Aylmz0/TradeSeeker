"""Tests for src/utils.py."""

import json
import math
import os

import pytest

from src.utils import format_num, safe_file_read, safe_file_write


class TestSafeFileRead:
    """Tests for safe_file_read."""

    def test_safe_file_read(self, temp_dir, write_test_json):
        """Reads existing JSON file."""
        path = write_test_json("test.json", {"key": "value"})
        result = safe_file_read(path)
        assert result == {"key": "value"}

    def test_safe_file_read_missing(self):
        """Returns default for missing file."""
        result = safe_file_read("/nonexistent/path.json", default_data={"default": True})
        assert result == {"default": True}

    def test_safe_file_read_empty(self, temp_dir):
        """Returns default for empty file."""
        path = os.path.join(temp_dir, "empty.json")
        with open(path, "w") as f:
            f.write("")
        result = safe_file_read(path, default_data=[])
        assert result == []

    def test_safe_file_read_whitespace_only(self, temp_dir):
        """Returns default for whitespace-only content."""
        path = os.path.join(temp_dir, "whitespace.json")
        with open(path, "w") as f:
            f.write("   \n  ")
        result = safe_file_read(path, default_data={"fallback": True})
        assert result == {"fallback": True}


class TestSafeFileWrite:
    """Tests for safe_file_write."""

    def test_safe_file_write(self, temp_dir):
        """Writes and reads back correctly."""
        path = os.path.join(temp_dir, "output.json")
        data = {"coins": ["BTC", "ETH"], "count": 2}
        success = safe_file_write(path, data)
        assert success is True
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_safe_file_write_creates_dirs(self, temp_dir):
        """Creates intermediate directories."""
        path = os.path.join(temp_dir, "sub", "dir", "file.json")
        success = safe_file_write(path, {"nested": True})
        assert success is True
        assert os.path.exists(path)

    def test_safe_file_write_returns_false_on_bad_path(self):
        """Returns False for invalid path."""
        result = safe_file_write("/nonexistent_root_impossible/test.json", {})
        assert result is False


class TestFormatNum:
    """Tests for format_num."""

    def test_format_num(self):
        """Formats numbers correctly."""
        assert format_num(3.14159) == "3.14"
        assert format_num(100.0, precision=0) == "100"
        assert format_num(0.123456, precision=4) == "0.1235"
        assert format_num(None) == "N/A"
        assert format_num(float("nan")) == "N/A"
        assert format_num(0) == "0.00"
