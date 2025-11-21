#!/usr/bin/env python
"""
Test script for Smart Indicator Cache functionality.
Tests cache hit/miss behavior and TTL calculation.
"""

import sys
import time
from cache_manager import SmartIndicatorCache, get_smart_cache

class MockMarketData:
    """Mock market data for testing"""
    def __init__(self):
        self.call_count = {}
    
    def get_technical_indicators(self, coin, interval):
        """Mock method that tracks API calls"""
        key = f"{coin}_{interval}"
        self.call_count[key] = self.call_count.get(key, 0) + 1
        
        print(f"  📡 API CALL #{self.call_count[key]}: {coin} {interval}")
        
        return {
            'price': 42000.0,
            'ema_20': 41500.0,
            'rsi_14': 55.0,
            'volume': 1000000,
            'timestamp': time.time()
        }

def test_cache_strategy():
    """Test caching strategy for different intervals"""
    print("\n" + "="*60)
    print("TEST: Cache Strategy for Different Intervals")
    print("="*60)
    
    cache = SmartIndicatorCache(htf_interval='1h')
    mock_data = MockMarketData()
    
    # Test 1: 3m should NEVER be cached
    print("\n[Test 1] 3m interval (should NEVER cache):")
    for i in range(3):
        print(f"  Request {i+1}:")
        cache.get_indicators('BTC', '3m', mock_data)
    
    expected_3m_calls = 3
    actual_3m_calls = mock_data.call_count.get('BTC_3m', 0)
    assert actual_3m_calls == expected_3m_calls, f"3m should always fetch fresh! Expected {expected_3m_calls}, got {actual_3m_calls}"
    print(f"  ✅ PASS: 3m called API {actual_3m_calls} times (never cached)")
    
    # Test 2: 15m should be cached
    print("\n[Test 2] 15m interval (should cache):")
    for i in range(3):
        print(f"  Request {i+1}:")
        cache.get_indicators('BTC', '15m', mock_data)
    
    expected_15m_calls = 1
    actual_15m_calls = mock_data.call_count.get('BTC_15m', 0)
    assert actual_15m_calls == expected_15m_calls, f"15m should cache! Expected {expected_15m_calls}, got {actual_15m_calls}"
    print(f"  ✅ PASS: 15m called API {actual_15m_calls} time, then 2 cache hits")
    
    # Test 3: 1h should be cached
    print("\n[Test 3] 1h interval (should cache):")
    for i in range(3):
        print(f"  Request {i+1}:")
        cache.get_indicators('BTC', '1h', mock_data)
    
    expected_1h_calls = 1
    actual_1h_calls = mock_data.call_count.get('BTC_1h', 0)
    assert actual_1h_calls == expected_1h_calls, f"1h should cache! Expected {expected_1h_calls}, got {actual_1h_calls}"
    print(f"  ✅ PASS: 1h called API {actual_1h_calls} time, then 2 cache hits")
    
    # Print cache stats
    print("\n" + "-"*60)
    cache.print_stats()

def test_ttl_calculation():
    """Test dynamic TTL calculation"""
    print("\n" + "="*60)
    print("TEST: Dynamic TTL Calculation")
    print("="*60)
    
    cache = SmartIndicatorCache(htf_interval='1h')
    
    # Test TTL for different intervals
    intervals = ['15m', '30m', '1h', '2h']
    
    for interval in intervals:
        ttl = cache._calculate_smart_ttl(interval)
        print(f"  {interval}: TTL = {ttl}s ({ttl/60:.1f} minutes)")
        
        # TTL should be reasonable
        assert ttl >= 30, f"TTL too low for {interval}"
        assert ttl < 7200, f"TTL too high for {interval}"
    
    print("  ✅ PASS: All TTL calculations within reasonable range")

def test_cache_expiration():
    """Test that expired entries are removed"""
    print("\n" + "="*60)
    print("TEST: Cache Expiration")
    print("="*60)
    
    cache = SmartIndicatorCache(htf_interval='1h')
    mock_data = MockMarketData()
    
    # Manually set a cache entry with very short TTL
    cache._set_to_cache('TEST', '15m', {'price': 100}, ttl=1)
    print("  Set cache entry with TTL=1s")
    
    # Should hit cache immediately
    cached = cache._get_from_cache('TEST_15m')
    assert cached is not None, "Cache should hit immediately"
    print("  ✅ Immediate fetch: Cache HIT")
    
    # Wait for expiration
    print("  Waiting 2 seconds for expiration...")
    time.sleep(2)
    
    # Should miss cache after expiration
    cached = cache._get_from_cache('TEST_15m')
    assert cached is None, "Cache should expire after TTL"
    print("  ✅ After 2s: Cache MISS (expired)")
    
    # Test clear_expired
    cache._set_to_cache('EXPIRE1', '15m', {'price': 100}, ttl=1)
    cache._set_to_cache('EXPIRE2', '1h', {'price': 200}, ttl=3600)
    time.sleep(2)
    
    cleared = cache.clear_expired()
    print(f"  ✅ Cleared {cleared} expired entries")

def test_multiple_coins():
    """Test caching with multiple coins"""
    print("\n" + "="*60)
    print("TEST: Multiple Coins")
    print("="*60)
    
    cache = SmartIndicatorCache(htf_interval='1h')
    mock_data = MockMarketData()
    
    coins = ['BTC', 'ETH', 'SOL']
    
    # First request: all should be API calls
    print("\n  Round 1 (cold cache):")
    for coin in coins:
        cache.get_indicators(coin, '15m', mock_data)
    
    # Second request: all should be cache hits
    print("\n  Round 2 (warm cache):")
    for coin in coins:
        cache.get_indicators(coin, '15m', mock_data)
    
    # Verify each coin was called exactly once
    for coin in coins:
        calls = mock_data.call_count.get(f'{coin}_15m', 0)
        assert calls == 1, f"{coin} should be called once, got {calls}"
    
    print(f"  ✅ PASS: All {len(coins)} coins cached correctly")
    
    cache.print_stats()

def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("SMART INDICATOR CACHE - TEST SUITE")
    print("🧪 "*20)
    
    try:
        test_cache_strategy()
        test_ttl_calculation()
        test_cache_expiration()
        test_multiple_coins()
        
        print("\n" + "✅ "*20)
        print("ALL TESTS PASSED!")
        print("✅ "*20 + "\n")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
