import time
from unittest.mock import MagicMock

from src.core.cache_manager import CacheManager, SmartIndicatorCache


def test_cache_set_get():
    cache = CacheManager(default_ttl=60)
    cache.set("key1", {"price": 2.50})
    assert cache.get("key1") == {"price": 2.50}


def test_cache_expiry():
    cache = CacheManager(default_ttl=1)
    cache.set("key1", "value1", ttl=0)
    time.sleep(0.05)
    assert cache.get("key1") is None


def test_cache_clear_expired():
    cache = CacheManager(default_ttl=1)
    cache.set("expired_key", "old", ttl=0)
    cache.set("valid_key", "new", ttl=60)
    time.sleep(0.05)
    removed = cache.clear_expired()
    assert removed == 1
    assert cache.get("valid_key") == "new"
    assert cache.get("expired_key") is None


def test_cache_stats():
    cache = CacheManager(default_ttl=60)
    cache.set("a", 1)
    cache.get("a")
    cache.get("b")
    stats = cache.get_stats()
    assert stats["total_entries"] == 1
    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 1
    assert stats["hit_rate_percent"] == 50.0


def test_cache_clear_all():
    cache = CacheManager(default_ttl=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.clear_all()
    stats = cache.get_stats()
    assert stats["total_entries"] == 0
    assert stats["hit_count"] == 0
    assert stats["miss_count"] == 0


def test_smart_cache_generate_key():
    sc = SmartIndicatorCache(htf_interval="1h")
    key = sc._generate_key("BTC", "15m")
    assert key == "BTC_15m"


def test_smart_cache_ttl():
    sc = SmartIndicatorCache(htf_interval="1h")
    ttl = sc._calculate_smart_ttl("15m")
    assert 30 <= ttl <= 900
    ttl_h = sc._calculate_smart_ttl("1h")
    assert 30 <= ttl_h <= 3600
