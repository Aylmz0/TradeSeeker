"""
Test suite for JSON prompt system.
Tests JSON builders, utilities, schemas, and integration.
"""
import unittest
import json
import math
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock config before importing modules that depend on it
class MockConfig:
    JSON_SERIES_MAX_LENGTH = 50
    JSON_PROMPT_COMPACT = False

sys.modules['config'] = type(sys)('config')
sys.modules['config'].Config = MockConfig

from prompt_json_builders import (
    build_metadata_json,
    build_counter_trade_json,
    build_trend_reversal_json,
    build_enhanced_context_json,
    build_cooldown_status_json,
    build_position_slot_json,
    build_market_data_json,
    build_portfolio_json,
    build_risk_status_json,
    build_historical_context_json
)
from prompt_json_utils import (
    SafeJSONEncoder,
    safe_json_dumps,
    compress_series,
    format_number_for_json,
    create_json_section,
    estimate_token_count,
    compare_token_usage
)
from prompt_json_schemas import (
    validate_json_against_schema,
    get_counter_trade_schema,
    get_trend_reversal_schema,
    get_enhanced_context_schema,
    get_cooldown_status_schema,
    get_position_slot_schema,
    get_market_data_schema,
    get_portfolio_schema,
    get_risk_status_schema,
    get_historical_context_schema,
    JSON_PROMPT_VERSION
)


class TestJSONUtils(unittest.TestCase):
    """Test JSON utility functions."""
    
    def test_safe_json_encoder_nan_handling(self):
        """Test SafeJSONEncoder handles NaN values."""
        data = {
            "price": float('nan'),
            "volume": 100.5,
            "rsi": None
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertIsNone(parsed["price"])
        self.assertEqual(parsed["volume"], 100.5)
        self.assertIsNone(parsed["rsi"])
    
    def test_safe_json_encoder_inf_handling(self):
        """Test SafeJSONEncoder handles Infinity values."""
        data = {
            "value": float('inf'),
            "negative_inf": float('-inf')
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertIsNone(parsed["value"])
        self.assertIsNone(parsed["negative_inf"])
    
    def test_safe_json_dumps_compact_mode(self):
        """Test compact JSON mode."""
        data = {"key": "value", "number": 123}
        result = safe_json_dumps(data, compact=True)
        # Compact mode should have no spaces
        self.assertNotIn(" ", result)
        # Should still be valid JSON
        parsed = json.loads(result)
        self.assertEqual(parsed["key"], "value")
    
    def test_safe_json_dumps_fallback(self):
        """Test fallback mechanism on serialization error."""
        # Create an object that can't be serialized
        class Unserializable:
            pass
        
        data = {"obj": Unserializable()}
        result = safe_json_dumps(data, fallback_on_error=True)
        # Should return fallback JSON
        parsed = json.loads(result)
        self.assertIn("error", parsed)
    
    def test_compress_series_short(self):
        """Test compression doesn't affect short series."""
        series = [1, 2, 3, 4, 5]
        result = compress_series(series, max_length=10)
        self.assertFalse(result["compressed"])
        self.assertEqual(result["values"], series)
    
    def test_compress_series_long(self):
        """Test compression of long series."""
        series = list(range(100))
        result = compress_series(series, max_length=50, keep_first=5, keep_last=5)
        self.assertTrue(result["compressed"])
        self.assertEqual(len(result["values"]), 10)  # 5 first + 5 last
        self.assertEqual(result["values"][:5], [0, 1, 2, 3, 4])
        self.assertEqual(result["values"][-5:], [95, 96, 97, 98, 99])
        self.assertIn("stats", result)
        self.assertEqual(result["stats"]["original_length"], 100)
    
    def test_format_number_for_json(self):
        """Test number formatting for JSON."""
        self.assertEqual(format_number_for_json(123.456), 123.456)
        self.assertEqual(format_number_for_json(123.0), 123)  # Should be int
        self.assertIsNone(format_number_for_json(float('nan')))
        self.assertIsNone(format_number_for_json(None))
        self.assertIsNone(format_number_for_json("invalid"))
    
    def test_estimate_token_count(self):
        """Test token count estimation."""
        text = "Hello world" * 10  # 110 characters
        tokens = estimate_token_count(text)
        # Rough estimate: 110 / 4 = ~27-28 tokens
        self.assertGreater(tokens, 20)
        self.assertLess(tokens, 35)
    
    def test_compare_token_usage(self):
        """Test token usage comparison."""
        text1 = "Short text"
        text2 = "Much longer text with more words and content"
        result = compare_token_usage(text1, text2)
        self.assertGreater(result["format2_tokens"], result["format1_tokens"])
        self.assertGreater(result["difference"], 0)


class TestJSONBuilders(unittest.TestCase):
    """Test JSON builder functions."""
    
    def test_build_metadata_json(self):
        """Test metadata JSON builder."""
        now = datetime.now()
        result = build_metadata_json(60, now, 10)
        self.assertEqual(result["minutes_running"], 60)
        self.assertEqual(result["invocation_count"], 10)
        self.assertIn("current_time", result)
    
    def test_build_counter_trade_json(self):
        """Test counter-trade analysis JSON builder."""
        all_indicators = {
            "XRP": {
                "3m": {
                    "current_price": 0.5,
                    "ema_20": 0.48,
                    "rsi_14": 30,
                    "volume": 1000,
                    "avg_volume": 500,
                    "macd": 0.01,
                    "macd_signal": 0.005
                },
                "1h": {
                    "current_price": 0.52,
                    "ema_20": 0.50
                }
            }
        }
        result = build_counter_trade_json(
            "dummy_text",
            all_indicators,
            ["XRP"],
            "1h"
        )
        self.assertIsInstance(result, list)
        if result:
            self.assertEqual(result[0]["coin"], "XRP")
            self.assertIn("conditions", result[0])
            self.assertIn("risk_level", result[0])
    
    def test_build_trend_reversal_json(self):
        """Test trend reversal JSON builder."""
        trend_reversal_analysis = {
            "XRP": {
                "loss_risk_signals": [{
                    "type": "LOSS_RISK",
                    "strength": "HIGH_LOSS_RISK",
                    "description": "Test signal"
                }],
                "signal_strength": "HIGH_LOSS_RISK",
                "current_trend_4h": "BULLISH",
                "current_trend_3m": "BEARISH"
            }
        }
        portfolio_positions = {
            "XRP": {
                "direction": "long",
                "entry_time": datetime.now().isoformat()
            }
        }
        result = build_trend_reversal_json(trend_reversal_analysis, portfolio_positions)
        self.assertIsInstance(result, list)
        if result:
            self.assertEqual(result[0]["coin"], "XRP")
            self.assertTrue(result[0]["has_position"])
    
    def test_build_enhanced_context_json(self):
        """Test enhanced context JSON builder."""
        enhanced_context = {
            "position_context": {
                "total_positions": 2,
                "long_positions": 1,
                "short_positions": 1,
                "total_margin_used": 50.0,
                "total_unrealized_pnl": 5.0
            },
            "market_regime": {
                "current_regime": "BULLISH",
                "bullish_count": 4,
                "bearish_count": 1,
                "neutral_count": 1,
                "regime_strength": 0.8
            },
            "performance_insights": {
                "total_return": 10.5,
                "sharpe_ratio": 1.2,
                "win_rate": 60.0
            },
            "directional_feedback": {
                "long": {
                    "total_pnl": 20.0,
                    "trades": 10,
                    "win_rate": 70.0
                },
                "short": {
                    "total_pnl": -5.0,
                    "trades": 5,
                    "win_rate": 40.0
                }
            },
            "risk_context": {
                "total_risk_usd": 50.0,
                "max_risk_allowed": 100.0,
                "risk_utilization_pct": 50.0
            },
            "suggestions": ["Suggestion 1", "Suggestion 2"]
        }
        result = build_enhanced_context_json(enhanced_context)
        self.assertEqual(result["position_context"]["total_positions"], 2)
        self.assertEqual(result["market_regime"]["global_regime"], "BULLISH")
        self.assertIsInstance(result["suggestions"], list)
    
    def test_build_cooldown_status_json(self):
        """Test cooldown status JSON builder."""
        result = build_cooldown_status_json(
            {"long": 2, "short": 0},
            {"XRP": 1},
            0,
            5
        )
        self.assertEqual(result["directional_cooldowns"]["long"], 2)
        self.assertEqual(result["directional_cooldowns"]["short"], 0)
        self.assertEqual(result["coin_cooldowns"]["XRP"], 1)
    
    def test_build_position_slot_json(self):
        """Test position slot JSON builder."""
        portfolio_positions = {
            "XRP": {
                "unrealized_pnl": 10.0,
                "confidence": 0.75
            },
            "SOL": {
                "unrealized_pnl": -5.0,
                "confidence": 0.60
            }
        }
        # Create a mock portfolio object
        class MockPortfolio:
            def __init__(self):
                self.positions = portfolio_positions
        
        portfolio = MockPortfolio()
        result = build_position_slot_json(portfolio.positions, 5)
        self.assertEqual(result["total_open"], 2)
        self.assertEqual(result["max_positions"], 5)
        self.assertEqual(result["available_slots"], 3)
        self.assertIsNotNone(result["weakest_position"])
        self.assertEqual(result["weakest_position"]["coin"], "SOL")
    
    def test_build_market_data_json(self):
        """Test market data JSON builder."""
        indicators_3m = {
            "current_price": 0.5,
            "ema_20": 0.48,
            "rsi_14": 55.0,
            "macd": 0.01,
            "atr_14": 0.02,
            "volume": 1000,
            "price_series": [0.48, 0.49, 0.5],
            "ema_20_series": [0.47, 0.48, 0.48],
            "rsi_14_series": [50, 52, 55],
            "macd_series": [0.005, 0.008, 0.01],
            "atr_14_series": [0.02, 0.02, 0.02],
            "volume_series": [900, 950, 1000]
        }
        indicators_15m = {
            "current_price": 0.51,
            "ema_20": 0.49,
            "rsi_14": 58.0,
            "macd": 0.015,
            "price_series": [0.49, 0.50, 0.51],
            "ema_20_series": [0.48, 0.49, 0.49],
            "rsi_14_series": [55, 56, 58],
            "macd_series": [0.01, 0.012, 0.015]
        }
        indicators_htf = {
            "current_price": 0.52,
            "ema_20": 0.50,
            "rsi_14": 60.0,
            "macd": 0.02,
            "atr_14": 0.025,
            "price_series": [0.50, 0.51, 0.52],
            "ema_20_series": [0.49, 0.50, 0.50],
            "rsi_14_series": [58, 59, 60],
            "macd_series": [0.015, 0.018, 0.02],
            "atr_14_series": [0.024, 0.025, 0.025]
        }
        sentiment = {
            "open_interest": 1000000,
            "funding_rate": 0.0001,
            "funding_rate_24h_avg": 0.00008
        }
        position = {
            "symbol": "XRP",
            "quantity": 100,
            "entry_price": 0.48,
            "current_price": 0.5,
            "liquidation_price": 0.40,
            "unrealized_pnl": 2.0,
            "leverage": 10,
            "confidence": 0.75,
            "risk_usd": 10.0,
            "notional_usd": 50.0,
            "exit_plan": {
                "profit_target": 0.55,
                "stop_loss": 0.45,
                "invalidation_condition": "If price closes below EMA20"
            }
        }
        result = build_market_data_json(
            "XRP",
            "BULLISH",
            sentiment,
            indicators_3m,
            indicators_15m,
            indicators_htf,
            position,
            max_series_length=50
        )
        self.assertEqual(result["coin"], "XRP")
        self.assertEqual(result["market_regime"], "BULLISH")
        self.assertIn("timeframes", result)
        self.assertIn("3m", result["timeframes"])
        self.assertIn("15m", result["timeframes"])
        self.assertIn("htf", result["timeframes"])
        self.assertIsNotNone(result["position"])
        self.assertEqual(result["position"]["symbol"], "XRP")
    
    def test_build_portfolio_json(self):
        """Test portfolio JSON builder."""
        class MockPortfolio:
            def __init__(self):
                self.total_return = 15.5
                self.current_balance = 215.5
                self.total_value = 230.0
                self.sharpe_ratio = 1.5
                self.positions = {
                    "XRP": {
                        "quantity": 100,
                        "entry_price": 0.48,
                        "current_price": 0.5,
                        "unrealized_pnl": 2.0,
                        "leverage": 10,
                        "confidence": 0.75
                    }
                }
        
        portfolio = MockPortfolio()
        result = build_portfolio_json(portfolio)
        self.assertEqual(result["total_return_pct"], 15.5)
        self.assertEqual(result["available_cash"], 215.5)
        self.assertEqual(result["account_value"], 230.0)
        self.assertEqual(len(result["positions"]), 1)
    
    def test_build_risk_status_json(self):
        """Test risk status JSON builder."""
        class MockPortfolio:
            def __init__(self):
                self.positions = {
                    "XRP": {"margin_usd": 20.0},
                    "SOL": {"margin_usd": 30.0}
                }
                self.current_balance = 150.0
        
        portfolio = MockPortfolio()
        result = build_risk_status_json(portfolio, max_positions=5)
        self.assertEqual(result["current_positions_count"], 2)
        self.assertEqual(result["total_margin_used"], 50.0)
        self.assertEqual(result["available_cash"], 150.0)
        self.assertIn("trading_limits", result)
    
    def test_build_historical_context_json(self):
        """Test historical context JSON builder."""
        trading_context = {
            "total_cycles_analyzed": 10,
            "market_behavior": "Bullish bias",
            "recent_decisions": [
                {"coin": "XRP", "signal": "buy_to_enter", "cycle": 8}
            ],
            "performance_trend": "Accumulation phase"
        }
        result = build_historical_context_json(trading_context)
        self.assertEqual(result["total_cycles_analyzed"], 10)
        self.assertEqual(result["market_behavior"], "Bullish bias")
        self.assertEqual(len(result["recent_decisions"]), 1)


class TestJSONSchemas(unittest.TestCase):
    """Test JSON schema validation."""
    
    def test_validate_counter_trade_schema(self):
        """Test counter-trade schema validation."""
        schema = get_counter_trade_schema()
        valid_data = {
            "coin": "XRP",
            "conditions": {
                "condition_1": True,
                "condition_2": False,
                "condition_3": True,
                "condition_4": False,
                "condition_5": True,
                "total_met": 3
            }
        }
        is_valid, error = validate_json_against_schema(valid_data, schema)
        self.assertTrue(is_valid, f"Validation failed: {error}")
    
    def test_validate_enhanced_context_schema(self):
        """Test enhanced context schema validation."""
        schema = get_enhanced_context_schema()
        valid_data = {
            "position_context": {
                "total_positions": 2,
                "long_positions": 1,
                "short_positions": 1,
                "total_margin_used": 50.0,
                "total_unrealized_pnl": 5.0
            },
            "market_regime": {
                "global_regime": "BULLISH",
                "bullish_count": 4,
                "bearish_count": 1,
                "neutral_count": 1
            }
        }
        is_valid, error = validate_json_against_schema(valid_data, schema)
        self.assertTrue(is_valid, f"Validation failed: {error}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data structures."""
        result = build_enhanced_context_json({})
        self.assertIsInstance(result, dict)
        
        result = build_trend_reversal_json({}, {})
        self.assertEqual(result, [])
        
        result = build_counter_trade_json("", {}, [], "1h")
        self.assertEqual(result, [])
    
    def test_nan_none_handling(self):
        """Test NaN and None handling in builders."""
        data = {
            "price": float('nan'),
            "volume": None,
            "rsi": float('inf')
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertIsNone(parsed["price"])
        self.assertIsNone(parsed["volume"])
        self.assertIsNone(parsed["rsi"])
    
    def test_large_series_compression(self):
        """Test compression of very large series."""
        large_series = list(range(1000))
        result = compress_series(large_series, max_length=50)
        self.assertTrue(result["compressed"])
        self.assertEqual(len(result["values"]), 10)  # 5 first + 5 last
        self.assertIn("stats", result)
        self.assertEqual(result["stats"]["original_length"], 1000)
    
    def test_missing_fields_handling(self):
        """Test handling of missing fields."""
        # Test with minimal data
        result = build_enhanced_context_json({
            "position_context": {},
            "market_regime": {}
        })
        self.assertIsInstance(result, dict)
        # Should not raise errors even with missing fields


class TestIntegration(unittest.TestCase):
    """Integration tests for full prompt generation."""
    
    def test_create_json_section(self):
        """Test JSON section creation."""
        data = {"key": "value", "number": 123}
        result = create_json_section("TEST_SECTION", data, compact=False)
        self.assertIn("TEST_SECTION (JSON):", result)
        self.assertIn('"key": "value"', result)
        
        # Test compact mode
        result_compact = create_json_section("TEST_SECTION", data, compact=True)
        self.assertIn("TEST_SECTION (JSON):", result_compact)
        # Compact should have no newlines in JSON part
        json_part = result_compact.split(":\n")[1]
        self.assertNotIn("\n", json_part)
    
    def test_full_prompt_structure(self):
        """Test that all JSON sections can be created together."""
        # This is a smoke test - just verify no errors
        metadata = build_metadata_json(60, datetime.now(), 10)
        cooldown = build_cooldown_status_json({}, {}, 0, 0)
        portfolio = build_portfolio_json(type('obj', (object,), {
            'total_return': 10.0,
            'current_balance': 200.0,
            'total_value': 210.0,
            'sharpe_ratio': 1.0,
            'positions': {}
        })())
        
        # Serialize all
        sections = [
            safe_json_dumps(metadata),
            safe_json_dumps(cooldown),
            safe_json_dumps(portfolio)
        ]
        
        # All should be valid JSON
        for section in sections:
            parsed = json.loads(section)
            self.assertIsInstance(parsed, dict)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

