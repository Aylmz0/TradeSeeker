"""
JSON Schema definitions for AI prompt data structures.
Used for validation and documentation of JSON prompt format.
"""
from typing import Dict, Any, List, Optional
import json

# JSON Prompt Format Version
JSON_PROMPT_VERSION = "1.0"

def get_counter_trade_schema() -> Dict[str, Any]:
    """Schema for counter-trade analysis JSON."""
    return {
        "type": "object",
        "properties": {
            "coin": {"type": "string"},
            "conditions": {
                "type": "object",
                "properties": {
                    "condition_1": {"type": "boolean"},
                    "condition_2": {"type": "boolean"},
                    "condition_3": {"type": "boolean"},
                    "condition_4": {"type": "boolean"},
                    "condition_5": {"type": "boolean"},
                    "total_met": {"type": "integer", "minimum": 0, "maximum": 5}
                },
                "required": ["condition_1", "condition_2", "condition_3", "condition_4", "condition_5", "total_met"]
            }
        },
        "required": ["coin", "conditions"]
    }

def get_trend_reversal_schema() -> Dict[str, Any]:
    """Schema for trend reversal detection JSON."""
    return {
        "type": "object",
        "properties": {
            "coin": {"type": "string"},
            "has_position": {"type": "boolean"},
            "position_direction": {"type": "string", "enum": ["long", "short", None]},
            "position_duration_minutes": {"type": ["number", "null"]},
            "reversal_signals": {
                "type": "object",
                "properties": {
                    "htf_reversal": {"type": "boolean"},
                    "fifteen_m_reversal": {"type": "boolean"},
                    "three_m_reversal": {"type": "boolean"},
                    "strength": {"type": "string", "enum": ["STRONG", "MEDIUM", "INFORMATIONAL", "NONE"]}
                }
            },
            "loss_risk_signal": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW", "NONE"]}
        },
        "required": ["coin", "has_position"]
    }

def get_enhanced_context_schema() -> Dict[str, Any]:
    """Schema for enhanced context JSON."""
    return {
        "type": "object",
        "properties": {
            "position_context": {
                "type": "object",
                "properties": {
                    "total_positions": {"type": "integer"},
                    "long_positions": {"type": "integer"},
                    "short_positions": {"type": "integer"},
                    "total_margin_used": {"type": "number"},
                    "total_unrealized_pnl": {"type": "number"}
                }
            },
            "market_regime": {
                "type": "object",
                "properties": {
                    "global_regime": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                    "bullish_count": {"type": "integer"},
                    "bearish_count": {"type": "integer"},
                    "neutral_count": {"type": "integer"}
                }
            },
            "performance_insights": {
                "type": "object",
                "properties": {
                    "total_return": {"type": "number"},
                    "sharpe_ratio": {"type": ["number", "null"]},
                    "win_rate": {"type": ["number", "null"]}
                }
            },
            "directional_feedback": {
                "type": "object",
                "properties": {
                    "long_performance": {
                        "type": "object",
                        "properties": {
                            "net_pnl": {"type": "number"},
                            "trades": {"type": "integer"},
                            "win_rate": {"type": "number"}
                        }
                    },
                    "short_performance": {
                        "type": "object",
                        "properties": {
                            "net_pnl": {"type": "number"},
                            "trades": {"type": "integer"},
                            "win_rate": {"type": "number"}
                        }
                    }
                }
            },
            "risk_context": {
                "type": "object",
                "properties": {
                    "current_risk_usd": {"type": "number"},
                    "max_risk_allowed": {"type": "number"},
                    "risk_utilization_pct": {"type": "number"}
                }
            },
            "suggestions": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }

def get_cooldown_status_schema() -> Dict[str, Any]:
    """Schema for cooldown status JSON."""
    return {
        "type": "object",
        "properties": {
            "directional_cooldowns": {
                "type": "object",
                "additionalProperties": {
                    "type": "integer"
                }
            },
            "coin_cooldowns": {
                "type": "object",
                "additionalProperties": {
                    "type": "integer"
                }
            },
            "counter_trend_cooldown": {"type": "integer"},
            "relaxed_countertrend_cycles": {"type": "integer"}
        }
    }

def get_position_slot_schema() -> Dict[str, Any]:
    """Schema for position slot status JSON."""
    return {
        "type": "object",
        "properties": {
            "total_open": {"type": "integer"},
            "max_positions": {"type": "integer"},
            "long_slots_used": {"type": "integer"},
            "short_slots_used": {"type": "integer"},
            "available_slots": {"type": "integer"},
            "weakest_position": {
                "type": ["object", "null"],
                "properties": {
                    "coin": {"type": "string"},
                    "unrealized_pnl": {"type": "number"},
                    "confidence": {"type": "number"}
                }
            }
        },
        "required": ["total_open", "max_positions", "available_slots"]
    }

def get_market_data_schema() -> Dict[str, Any]:
    """Schema for market data JSON (per coin)."""
    return {
        "type": "object",
        "properties": {
            "coin": {"type": "string"},
            "market_regime": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
            "sentiment": {
                "type": "object",
                "properties": {
                    "open_interest": {"type": ["number", "null"]},
                    "funding_rate": {"type": ["number", "null"]},
                    "funding_rate_24h_avg": {"type": ["number", "null"]}
                }
            },
            "timeframes": {
                "type": "object",
                "properties": {
                    "3m": {
                        "type": "object",
                        "properties": {
                            "current": {
                                "type": "object",
                                "properties": {
                                    "price": {"type": "number"},
                                    "ema20": {"type": ["number", "null"]},
                                    "rsi": {"type": ["number", "null"]},
                                    "macd": {"type": ["number", "null"]},
                                    "atr": {"type": ["number", "null"]},
                                    "volume": {"type": ["number", "null"]}
                                }
                            },
                            "series": {
                                "type": "object",
                                "properties": {
                                    "price": {"type": "array", "items": {"type": "number"}},
                                    "ema20": {"type": "array", "items": {"type": ["number", "null"]}},
                                    "rsi": {"type": "array", "items": {"type": ["number", "null"]}},
                                    "macd": {"type": "array", "items": {"type": ["number", "null"]}},
                                    "atr": {"type": "array", "items": {"type": ["number", "null"]}},
                                    "volume": {"type": "array", "items": {"type": ["number", "null"]}}
                                }
                            }
                        }
                    },
                    "15m": {
                        "type": "object",
                        "properties": {
                            "current": {
                                "type": "object",
                                "properties": {
                                    "price": {"type": "number"},
                                    "ema20": {"type": ["number", "null"]},
                                    "rsi": {"type": ["number", "null"]},
                                    "macd": {"type": ["number", "null"]}
                                }
                            },
                            "series": {
                                "type": "object",
                                "properties": {
                                    "price": {"type": "array", "items": {"type": "number"}},
                                    "ema20": {"type": "array", "items": {"type": ["number", "null"]}},
                                    "rsi": {"type": "array", "items": {"type": ["number", "null"]}},
                                    "macd": {"type": "array", "items": {"type": ["number", "null"]}}
                                }
                            }
                        }
                    },
                    "htf": {
                        "type": "object",
                        "properties": {
                            "current": {
                                "type": "object",
                                "properties": {
                                    "price": {"type": "number"},
                                    "ema20": {"type": ["number", "null"]},
                                    "rsi": {"type": ["number", "null"]},
                                    "macd": {"type": ["number", "null"]},
                                    "atr": {"type": ["number", "null"]}
                                }
                            },
                            "series": {
                                "type": "object",
                                "properties": {
                                    "price": {"type": "array", "items": {"type": "number"}},
                                    "ema20": {"type": "array", "items": {"type": ["number", "null"]}},
                                    "rsi": {"type": "array", "items": {"type": ["number", "null"]}},
                                    "macd": {"type": "array", "items": {"type": ["number", "null"]}},
                                    "atr": {"type": "array", "items": {"type": ["number", "null"]}}
                                }
                            }
                        }
                    }
                }
            },
            "position": {
                "type": ["object", "null"],
                "properties": {
                    "symbol": {"type": "string"},
                    "quantity": {"type": "number"},
                    "entry_price": {"type": "number"},
                    "current_price": {"type": "number"},
                    "liquidation_price": {"type": "number"},
                    "unrealized_pnl": {"type": "number"},
                    "leverage": {"type": "integer"},
                    "confidence": {"type": "number"},
                    "risk_usd": {"type": ["number", "string"]},
                    "notional_usd": {"type": "number"},
                    "exit_plan": {
                        "type": "object",
                        "properties": {
                            "profit_target": {"type": ["number", "null"]},
                            "stop_loss": {"type": ["number", "null"]},
                            "invalidation_condition": {"type": ["string", "null"]}
                        }
                    }
                }
            }
        },
        "required": ["coin", "market_regime", "timeframes"]
    }

def get_portfolio_schema() -> Dict[str, Any]:
    """Schema for portfolio JSON."""
    return {
        "type": "object",
        "properties": {
            "total_return_pct": {"type": "number"},
            "available_cash": {"type": "number"},
            "account_value": {"type": "number"},
            "sharpe_ratio": {"type": ["number", "null"]},
            "positions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "quantity": {"type": "number"},
                        "entry_price": {"type": "number"},
                        "current_price": {"type": "number"},
                        "unrealized_pnl": {"type": "number"},
                        "leverage": {"type": "integer"},
                        "confidence": {"type": "number"}
                    }
                }
            }
        },
        "required": ["total_return_pct", "available_cash", "account_value"]
    }

def get_risk_status_schema() -> Dict[str, Any]:
    """Schema for risk status JSON."""
    return {
        "type": "object",
        "properties": {
            "current_positions_count": {"type": "integer"},
            "total_margin_used": {"type": "number"},
            "available_cash": {"type": "number"},
            "trading_limits": {
                "type": "object",
                "properties": {
                    "min_position": {"type": "number"},
                    "max_positions": {"type": "integer"},
                    "available_cash_protection": {"type": "number"},
                    "position_sizing_pct": {"type": "number"}
                }
            }
        },
        "required": ["current_positions_count", "total_margin_used", "available_cash"]
    }

def get_historical_context_schema() -> Dict[str, Any]:
    """Schema for historical context JSON."""
    return {
        "type": "object",
        "properties": {
            "total_cycles_analyzed": {"type": "integer"},
            "market_behavior": {"type": "string"},
            "recent_decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cycle": {"type": "integer"},
                        "decisions": {"type": "object"}
                    }
                }
            }
        }
    }

def get_full_prompt_schema() -> Dict[str, Any]:
    """Schema for the complete JSON prompt structure."""
    return {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "metadata": {
                "type": "object",
                "properties": {
                    "minutes_running": {"type": "integer"},
                    "current_time": {"type": "string"},
                    "invocation_count": {"type": "integer"}
                },
                "required": ["minutes_running", "current_time", "invocation_count"]
            },
            "counter_trade_analysis": {
                "type": "array",
                "items": get_counter_trade_schema()
            },
            "trend_reversal_detection": {
                "type": "array",
                "items": get_trend_reversal_schema()
            },
            "enhanced_context": get_enhanced_context_schema(),
            "cooldown_status": get_cooldown_status_schema(),
            "position_slot_status": get_position_slot_schema(),
            "market_data": {
                "type": "array",
                "items": get_market_data_schema()
            },
            "portfolio": get_portfolio_schema(),
            "risk_status": get_risk_status_schema(),
            "historical_context": get_historical_context_schema()
        },
        "required": ["version", "metadata"]
    }

def validate_json_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Simple JSON schema validation.
    Returns (is_valid, error_message).
    Note: This is a basic implementation. For production, consider using jsonschema library.
    """
    try:
        # Basic type checking
        if schema.get("type") == "object":
            if not isinstance(data, dict):
                return False, f"Expected object, got {type(data).__name__}"
            
            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    return False, f"Missing required field: {field}"
            
            # Check properties
            properties = schema.get("properties", {})
            for key, value in data.items():
                if key in properties:
                    prop_schema = properties[key]
                    prop_type = prop_schema.get("type")
                    
                    if prop_type == "array":
                        if not isinstance(value, list):
                            return False, f"Field '{key}' should be array"
                    elif prop_type == "object":
                        if not isinstance(value, dict):
                            return False, f"Field '{key}' should be object"
                    elif prop_type == "string":
                        if not isinstance(value, str):
                            return False, f"Field '{key}' should be string"
                    elif prop_type == "integer":
                        if not isinstance(value, int):
                            return False, f"Field '{key}' should be integer"
                    elif prop_type == "number":
                        if not isinstance(value, (int, float)):
                            return False, f"Field '{key}' should be number"
                    elif isinstance(prop_type, list):  # Union type like ["number", "null"]
                        if value is not None and not any(
                            (t == "number" and isinstance(value, (int, float))) or
                            (t == "string" and isinstance(value, str)) or
                            (t == "integer" and isinstance(value, int)) or
                            (t == "object" and isinstance(value, dict)) or
                            (t == "array" and isinstance(value, list)) or
                            (t == "null" and value is None)
                            for t in prop_type
                        ):
                            return False, f"Field '{key}' has invalid type"
        
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"

