"""
JSON Schema definitions for AI prompt data structures.
Used for validation and documentation of JSON prompt format.
"""

from typing import Any

# JSON Prompt Format Version
JSON_PROMPT_VERSION = "1.0"


def get_counter_trade_schema() -> dict[str, Any]:
    """Schema for counter-trade risk (compact dict per coin)."""
    return {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "risk_level": {
                    "type": "string",
                    "enum": ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "VERY_HIGH_RISK"],
                },
                "alignment_strength": {
                    "type": "string",
                    "enum": ["STRONG", "MEDIUM", "NONE"],
                },
                "conditions_met": {"type": "integer", "minimum": 0, "maximum": 8},
            },
            "required": ["risk_level", "alignment_strength", "conditions_met"],
        },
    }


def get_trend_reversal_schema() -> dict[str, Any]:
    """Schema for trend reversal threats (compact dict per coin)."""
    return {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "strength": {
                    "type": "string",
                    "enum": ["NONE", "WEAK", "MODERATE", "STRONG", "CRITICAL"],
                },
            },
            "required": ["strength"],
        },
    }


def get_state_vector_schema() -> dict[str, Any]:
    """Schema for coin State Vector."""
    return {
        "type": "object",
        "properties": {
            "coin": {"type": "string"},
            "ml_consensus": {
                "type": ["object", "null"],
                "properties": {
                    "probability": {"type": "number"},
                    "signal": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                    "confidence": {"type": "string"},
                },
            },
            "market_context": {
                "type": "object",
                "properties": {
                    "regime": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                    "efficiency_ratio": {"type": ["number", "null"]},
                    "volatility_state": {"type": "string", "enum": ["SQUEEZE", "EXPANDING", "NORMAL"]},
                    "price_location": {"type": "string", "enum": ["UPPER_10", "LOWER_10", "MIDDLE"]},
                },
            },
            "technical_summary": {
                "type": "object",
                "properties": {
                    "trend_alignment": {"type": "string"},
                    "momentum": {"type": "string"},
                    "volume_ratio": {"type": ["number", "null"]},
                    "volume_support": {"type": "string"},
                    "structure_15m": {"type": "string"},
                },
            },
            "key_levels": {
                "type": "object",
                "properties": {
                    "price": {"type": ["number", "null"]},
                    "ema20_htf": {"type": ["number", "null"]},
                    "rsi_15m": {"type": ["number", "null"]},
                    "atr_htf": {"type": ["number", "null"]},
                },
            },
            "risk_profile": {
                "type": "object",
                "properties": {
                    "counter_trade_risk": {"type": "string"},
                    "alignment_strength": {"type": "string"},
                    "reversal_threat": {"type": "string"},
                },
            },
            "sentiment": {
                "type": "object",
                "properties": {
                    "funding_rate": {"type": ["number", "null"]},
                    "open_interest": {"type": ["number", "null"]},
                },
            },
            "position": {"type": ["object", "null"]},
        },
        "required": ["coin", "market_context", "technical_summary", "key_levels", "risk_profile"],
    }



def get_portfolio_schema() -> dict[str, Any]:
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
                        "direction": {"type": "string", "enum": ["long", "short"]},
                        "quantity": {"type": "number"},
                        "entry_price": {"type": "number"},
                        "current_price": {"type": "number"},
                        "unrealized_pnl": {"type": "number"},
                        "leverage": {"type": "integer"},
                        "confidence": {"type": "number"},
                    },
                },
            },
        },
        "required": ["total_return_pct", "available_cash", "account_value"],
    }


def get_risk_status_schema() -> dict[str, Any]:
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
                    "position_sizing_pct": {"type": "number"},
                },
            },
        },
        "required": ["current_positions_count", "total_margin_used", "available_cash"],
    }


def get_historical_context_schema() -> dict[str, Any]:
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
                    "properties": {"cycle": {"type": "integer"}, "decisions": {"type": "object"}},
                },
            },
        },
    }


def get_full_prompt_schema() -> dict[str, Any]:
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
                    "invocation_count": {"type": "integer"},
                },
                "required": ["minutes_running", "current_time", "invocation_count"],
            },
            "cooldown_status": get_cooldown_status_schema(),
            "position_slot_status": get_position_slot_schema(),
            "market_data": {"type": "array", "items": get_state_vector_schema()},
            "portfolio": get_portfolio_schema(),
            "risk_status": get_risk_status_schema(),
            "historical_context": get_historical_context_schema(),
        },
        "required": ["version", "metadata"],
    }


def validate_json_against_schema(
    data: dict[str, Any], schema: dict[str, Any],
) -> tuple[bool, str | None]:
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
                            (t == "number" and isinstance(value, (int, float)))
                            or (t == "string" and isinstance(value, str))
                            or (t == "integer" and isinstance(value, int))
                            or (t == "object" and isinstance(value, dict))
                            or (t == "array" and isinstance(value, list))
                            or (t == "null" and value is None)
                            for t in prop_type
                        ):
                            return False, f"Field '{key}' has invalid type"

        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"
