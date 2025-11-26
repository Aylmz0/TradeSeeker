"""
JSON utility functions for AI prompt generation.
Handles safe JSON serialization, NaN/None handling, and data compression.
"""
import json
import math
from typing import Any, Dict, List, Optional, Union


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NaN, None, and other edge cases.
    Integrates with format_num() for consistent number formatting.
    """
    
    def encode(self, obj: Any) -> str:
        """Encode object to JSON string, handling special cases."""
        return super().encode(self._clean_value(obj))
    
    def _clean_value(self, value: Any) -> Any:
        """Recursively clean values, converting NaN/None to null."""
        if value is None:
            return None
        elif isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        elif isinstance(value, dict):
            return {k: self._clean_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._clean_value(item) for item in value]
        else:
            return value


def safe_json_dumps(
    obj: Any,
    indent: Optional[int] = 2,
    compact: bool = False,
    fallback_on_error: bool = True,
    fallback_value: Optional[str] = None
) -> str:
    """
    Safely serialize object to JSON string with error handling and fallback.
    
    Args:
        obj: Object to serialize
        indent: JSON indentation (None for compact, 2 for pretty)
        compact: If True, use compact JSON (indent=None, no spaces)
        fallback_on_error: If True, return fallback_value on error instead of raising
        fallback_value: Value to return on error if fallback_on_error is True
    
    Returns:
        JSON string representation of obj
    
    Raises:
        json.JSONEncodeError: If serialization fails and fallback_on_error is False
    """
    try:
        # Use compact mode if requested
        if compact:
            indent = None
            separators = (',', ':')
        else:
            separators = (',', ': ') if indent else (',', ':')
        
        # Serialize with custom encoder
        json_str = json.dumps(
            obj,
            cls=SafeJSONEncoder,
            indent=indent,
            separators=separators,
            ensure_ascii=False
        )
        
        return json_str
    
    except (TypeError, ValueError, OverflowError) as e:
        if fallback_on_error:
            if fallback_value is not None:
                return fallback_value
            # Try to create a minimal fallback representation
            try:
                return json.dumps({"error": "Serialization failed", "type": str(type(obj).__name__)})
            except:
                return '{"error": "Serialization failed"}'
        else:
            raise


def compress_series(
    series: List[Any],
    max_length: int = 50,
    keep_first: int = 5,
    keep_last: int = 5
) -> Dict[str, Any]:
    """
    Compress a long series by keeping first N, last N, and summary stats.
    
    Args:
        series: List of values to compress
        max_length: Maximum length before compression is applied
        keep_first: Number of first values to keep
        keep_last: Number of last values to keep
    
    Returns:
        Dict with 'compressed' flag, 'values' (compressed list), and 'stats' (summary)
    """
    if len(series) <= max_length:
        return {
            "compressed": False,
            "values": series,
            "length": len(series)
        }
    
    # Keep first N and last N values
    compressed_values = series[:keep_first] + series[-keep_last:]
    
    # Calculate summary statistics
    numeric_values = [v for v in series if isinstance(v, (int, float)) and not math.isnan(v) if v is not None]
    
    stats = {}
    if numeric_values:
        stats = {
            "original_length": len(series),
            "compressed_length": len(compressed_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": sum(numeric_values) / len(numeric_values) if numeric_values else None,
            "removed_count": len(series) - len(compressed_values)
        }
    
    return {
        "compressed": True,
        "values": compressed_values,
        "stats": stats,
        "length": len(compressed_values),
        "original_length": len(series)
    }


def format_number_for_json(value: Any, precision: int = 4) -> Union[float, int, None]:
    """
    Format number for JSON output.
    Returns raw number (not string) for JSON, handling NaN/None.
    
    Args:
        value: Number to format
        precision: Not used (kept for compatibility), numbers are stored as-is in JSON
    
    Returns:
        Number (int or float) or None if invalid
    """
    if value is None:
        return None
    
    try:
        if isinstance(value, str):
            # Try to parse string
            float_val = float(value)
        else:
            float_val = float(value)
        
        # Check for NaN or Inf
        if math.isnan(float_val) or math.isinf(float_val):
            return None
        
        # Return as int if it's a whole number, otherwise float
        if float_val.is_integer():
            return int(float_val)
        return float_val
    
    except (ValueError, TypeError):
        return None


def create_json_section(
    section_name: str,
    data: Any,
    compact: bool = False,
    description: Optional[str] = None
) -> str:
    """
    Create a formatted JSON section for hybrid prompt.
    
    Args:
        section_name: Name of the JSON section (e.g., "COUNTER_TRADE_ANALYSIS")
        data: Data to serialize
        compact: Use compact JSON format
        description: Optional description text before JSON
    
    Returns:
        Formatted string with description and JSON
    """
    json_str = safe_json_dumps(data, compact=compact)
    
    result = f"{section_name} (JSON):\n{json_str}"
    
    if description:
        result = f"{description}\n\n{result}"
    
    return result


def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count for a text string.
    Uses simple heuristic: ~4 characters per token.
    
    Args:
        text: Text to estimate
    
    Returns:
        Estimated token count
    """
    # Rough estimation: 1 token ≈ 4 characters
    # This is a simple heuristic, actual tokenization varies by model
    return len(text) // 4


def compare_token_usage(text1: str, text2: str) -> Dict[str, Any]:
    """
    Compare token usage between two text strings.
    
    Args:
        text1: First text (e.g., old format)
        text2: Second text (e.g., new JSON format)
    
    Returns:
        Dict with token counts and comparison metrics
    """
    tokens1 = estimate_token_count(text1)
    tokens2 = estimate_token_count(text2)
    
    diff = tokens2 - tokens1
    diff_pct = (diff / tokens1 * 100) if tokens1 > 0 else 0
    
    return {
        "format1_tokens": tokens1,
        "format2_tokens": tokens2,
        "difference": diff,
        "difference_pct": round(diff_pct, 2),
        "more_efficient": tokens2 < tokens1
    }

