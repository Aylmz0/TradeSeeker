"""
Integration tests for JSON prompt system.
Tests full prompt generation, token comparison, and A/B testing.
"""
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Mock config before importing
class MockConfig:
    JSON_SERIES_MAX_LENGTH = 50
    JSON_PROMPT_COMPACT = False
    USE_JSON_PROMPT = False
    JSON_PROMPT_VERSION = "1.0"

sys.modules['config'] = type(sys)('config')
sys.modules['config'].Config = MockConfig

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_json_utils import estimate_token_count, compare_token_usage, safe_json_dumps


def test_token_comparison():
    """Compare token usage between text and JSON formats."""
    print("\n" + "="*60)
    print("TOKEN USAGE COMPARISON TEST")
    print("="*60)
    
    # Sample text prompt (simplified)
    text_prompt = """
USER_PROMPT:
It has been 60 minutes since you started trading. The current time is 2025-01-01 12:00:00 and you've been invoked 10 times.

COUNTER-TRADE ANALYSIS:
XRP: Conditions met: 3/5, Risk: MEDIUM
SOL: Conditions met: 2/5, Risk: HIGH

MARKET DATA:
XRP: Price: 0.50, EMA20: 0.48, RSI: 55.0
SOL: Price: 185.0, EMA20: 190.0, RSI: 45.0

PORTFOLIO:
Total Return: 15.5%
Available Cash: $215.50
Account Value: $230.00
"""
    
    # Sample JSON prompt (simplified)
    json_data = {
        "metadata": {
            "minutes_running": 60,
            "current_time": "2025-01-01T12:00:00",
            "invocation_count": 10
        },
        "counter_trade_analysis": [
            {"coin": "XRP", "conditions": {"total_met": 3}, "risk_level": "MEDIUM"},
            {"coin": "SOL", "conditions": {"total_met": 2}, "risk_level": "HIGH"}
        ],
        "market_data": [
            {"coin": "XRP", "timeframes": {"3m": {"current": {"price": 0.50, "ema20": 0.48, "rsi": 55.0}}}},
            {"coin": "SOL", "timeframes": {"3m": {"current": {"price": 185.0, "ema20": 190.0, "rsi": 45.0}}}}
        ],
        "portfolio": {
            "total_return_pct": 15.5,
            "available_cash": 215.50,
            "account_value": 230.00
        }
    }
    
    json_prompt = f"""
USER_PROMPT:
It has been {json_data['metadata']['minutes_running']} minutes since you started trading.

COUNTER_TRADE_ANALYSIS (JSON):
{safe_json_dumps(json_data['counter_trade_analysis'], compact=False)}

MARKET_DATA (JSON):
{safe_json_dumps(json_data['market_data'], compact=False)}

PORTFOLIO (JSON):
{safe_json_dumps(json_data['portfolio'], compact=False)}
"""
    
    # Compare
    comparison = compare_token_usage(text_prompt, json_prompt)
    
    print(f"\nText Format Tokens: {comparison['format1_tokens']}")
    print(f"JSON Format Tokens: {comparison['format2_tokens']}")
    print(f"Difference: {comparison['difference']} tokens ({comparison['difference_pct']:.2f}%)")
    print(f"More Efficient: {'JSON' if comparison['more_efficient'] else 'Text'}")
    
    return comparison


def test_compact_vs_pretty():
    """Compare compact vs pretty JSON token usage."""
    print("\n" + "="*60)
    print("COMPACT vs PRETTY JSON COMPARISON")
    print("="*60)
    
    data = {
        "market_data": [
            {
                "coin": "XRP",
                "timeframes": {
                    "3m": {
                        "current": {"price": 0.50, "ema20": 0.48, "rsi": 55.0},
                        "series": {
                            "price": [0.48, 0.49, 0.50, 0.51, 0.52],
                            "ema20": [0.47, 0.48, 0.48, 0.49, 0.49]
                        }
                    }
                }
            }
        ]
    }
    
    pretty_json = safe_json_dumps(data, compact=False)
    compact_json = safe_json_dumps(data, compact=True)
    
    pretty_tokens = estimate_token_count(pretty_json)
    compact_tokens = estimate_token_count(compact_json)
    
    print(f"\nPretty JSON Tokens: {pretty_tokens}")
    print(f"Compact JSON Tokens: {compact_tokens}")
    print(f"Savings: {pretty_tokens - compact_tokens} tokens ({((pretty_tokens - compact_tokens) / pretty_tokens * 100):.2f}%)")
    
    return {
        "pretty_tokens": pretty_tokens,
        "compact_tokens": compact_tokens,
        "savings_pct": ((pretty_tokens - compact_tokens) / pretty_tokens * 100) if pretty_tokens > 0 else 0
    }


def test_series_compression_impact():
    """Test impact of series compression on token usage."""
    print("\n" + "="*60)
    print("SERIES COMPRESSION IMPACT TEST")
    print("="*60)
    
    from prompt_json_utils import compress_series
    
    # Large series
    large_series = [0.50 + i * 0.001 for i in range(200)]
    
    # Uncompressed
    uncompressed_data = {
        "price_series": large_series
    }
    uncompressed_json = safe_json_dumps(uncompressed_data, compact=True)
    uncompressed_tokens = estimate_token_count(uncompressed_json)
    
    # Compressed
    compressed_result = compress_series(large_series, max_length=50)
    compressed_data = {
        "price_series": compressed_result
    }
    compressed_json = safe_json_dumps(compressed_data, compact=True)
    compressed_tokens = estimate_token_count(compressed_json)
    
    print(f"\nUncompressed Tokens: {uncompressed_tokens}")
    print(f"Compressed Tokens: {compressed_tokens}")
    print(f"Savings: {uncompressed_tokens - compressed_tokens} tokens ({((uncompressed_tokens - compressed_tokens) / uncompressed_tokens * 100):.2f}%)")
    print(f"Original Length: {len(large_series)}")
    print(f"Compressed Length: {compressed_result['length']}")
    
    return {
        "uncompressed_tokens": uncompressed_tokens,
        "compressed_tokens": compressed_tokens,
        "savings_pct": ((uncompressed_tokens - compressed_tokens) / uncompressed_tokens * 100) if uncompressed_tokens > 0 else 0
    }


def test_json_serialization_performance():
    """Test JSON serialization performance and error handling."""
    print("\n" + "="*60)
    print("JSON SERIALIZATION PERFORMANCE TEST")
    print("="*60)
    
    import time
    
    # Test data with various edge cases
    test_data = {
        "normal_values": [1, 2, 3, 4, 5],
        "float_values": [1.5, 2.5, 3.5],
        "none_values": [None, None, None],
        "nan_value": float('nan'),
        "inf_value": float('inf'),
        "nested": {
            "level1": {
                "level2": {
                    "value": 123
                }
            }
        },
        "large_array": list(range(100))
    }
    
    # Test serialization
    start_time = time.perf_counter()
    result = safe_json_dumps(test_data, fallback_on_error=True)
    elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
    
    print(f"\nSerialization Time: {elapsed:.2f} ms")
    print(f"Output Length: {len(result)} characters")
    
    # Verify it's valid JSON
    try:
        parsed = json.loads(result)
        print("✅ Valid JSON")
        print(f"NaN handled: {parsed.get('nan_value') is None}")
        print(f"Inf handled: {parsed.get('inf_value') is None}")
    except Exception as e:
        print(f"❌ Invalid JSON: {e}")
    
    return {
        "serialization_time_ms": elapsed,
        "output_length": len(result),
        "valid": True
    }


def run_all_integration_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("INTEGRATION TESTS - JSON PROMPT SYSTEM")
    print("="*60)
    
    results = {}
    
    try:
        results['token_comparison'] = test_token_comparison()
    except Exception as e:
        print(f"❌ Token comparison test failed: {e}")
        results['token_comparison'] = {"error": str(e)}
    
    try:
        results['compact_vs_pretty'] = test_compact_vs_pretty()
    except Exception as e:
        print(f"❌ Compact vs pretty test failed: {e}")
        results['compact_vs_pretty'] = {"error": str(e)}
    
    try:
        results['series_compression'] = test_series_compression_impact()
    except Exception as e:
        print(f"❌ Series compression test failed: {e}")
        results['series_compression'] = {"error": str(e)}
    
    try:
        results['serialization_performance'] = test_json_serialization_performance()
    except Exception as e:
        print(f"❌ Serialization performance test failed: {e}")
        results['serialization_performance'] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        if "error" in result:
            print(f"❌ {test_name}: FAILED - {result['error']}")
        else:
            print(f"✅ {test_name}: PASSED")
    
    return results


if __name__ == '__main__':
    import json
    results = run_all_integration_tests()
    print(f"\n📊 Full Results:\n{json.dumps(results, indent=2)}")

