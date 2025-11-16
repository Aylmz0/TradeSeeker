"""
A/B Testing script for comparing text vs JSON prompt formats.
This script can be used to compare actual prompt outputs.
"""
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import json

# Mock config
class MockConfig:
    JSON_SERIES_MAX_LENGTH = 50
    JSON_PROMPT_COMPACT = False
    USE_JSON_PROMPT = False
    JSON_PROMPT_VERSION = "1.0"

sys.modules['config'] = type(sys)('config')
sys.modules['config'].Config = MockConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_json_utils import estimate_token_count, compare_token_usage, safe_json_dumps


def compare_prompt_formats(text_prompt: str, json_prompt: str) -> Dict[str, Any]:
    """
    Compare text and JSON prompt formats.
    
    Args:
        text_prompt: Text format prompt
        json_prompt: JSON format prompt
    
    Returns:
        Comparison metrics
    """
    comparison = compare_token_usage(text_prompt, json_prompt)
    
    # Additional metrics
    text_lines = len(text_prompt.split('\n'))
    json_lines = len(json_prompt.split('\n'))
    
    # Count JSON sections
    json_sections = json_prompt.count('(JSON):')
    
    return {
        **comparison,
        "text_lines": text_lines,
        "json_lines": json_lines,
        "json_sections": json_sections,
        "text_length": len(text_prompt),
        "json_length": len(json_prompt)
    }


def analyze_prompt_structure(prompt: str, format_type: str = "text") -> Dict[str, Any]:
    """
    Analyze prompt structure and extract metrics.
    
    Args:
        prompt: Prompt string
        format_type: "text" or "json"
    
    Returns:
        Analysis metrics
    """
    lines = prompt.split('\n')
    
    metrics = {
        "total_lines": len(lines),
        "total_chars": len(prompt),
        "total_tokens": estimate_token_count(prompt),
        "empty_lines": sum(1 for line in lines if not line.strip()),
        "section_headers": sum(1 for line in lines if line.strip().startswith('=') and len(line.strip()) > 10)
    }
    
    if format_type == "json":
        metrics["json_sections"] = prompt.count('(JSON):')
        metrics["json_blocks"] = prompt.count('{')  # Rough estimate
    
    return metrics


def generate_comparison_report(text_prompt: str, json_prompt: str) -> str:
    """
    Generate a detailed comparison report.
    
    Args:
        text_prompt: Text format prompt
        json_prompt: JSON format prompt
    
    Returns:
        Formatted report string
    """
    comparison = compare_prompt_formats(text_prompt, json_prompt)
    text_analysis = analyze_prompt_structure(text_prompt, "text")
    json_analysis = analyze_prompt_structure(json_prompt, "json")
    
    report = f"""
{'='*70}
PROMPT FORMAT COMPARISON REPORT
{'='*70}

TOKEN USAGE:
  Text Format:  {comparison['format1_tokens']:,} tokens
  JSON Format: {comparison['format2_tokens']:,} tokens
  Difference:   {comparison['difference']:,} tokens ({comparison['difference_pct']:.2f}%)
  More Efficient: {'JSON' if comparison['more_efficient'] else 'Text'}

SIZE METRICS:
  Text Length:  {comparison['text_length']:,} characters
  JSON Length:  {comparison['json_length']:,} characters
  Difference:   {abs(comparison['json_length'] - comparison['text_length']):,} characters

STRUCTURE METRICS:
  Text Lines:   {text_analysis['total_lines']:,}
  JSON Lines:   {json_analysis['total_lines']:,}
  JSON Sections: {json_analysis.get('json_sections', 0)}
  Section Headers: Text={text_analysis['section_headers']}, JSON={json_analysis['section_headers']}

RECOMMENDATIONS:
"""
    
    if comparison['more_efficient']:
        report += "  ✅ JSON format is more token-efficient\n"
    else:
        report += "  ⚠️  Text format is more token-efficient\n"
        report += "     Consider using compact JSON mode (JSON_PROMPT_COMPACT=true)\n"
    
    if json_analysis.get('json_sections', 0) > 0:
        report += f"  ✅ JSON format has {json_analysis['json_sections']} structured sections\n"
    
    if comparison['difference_pct'] > 50:
        report += "  ⚠️  Large token difference detected - review data compression\n"
    
    report += "\n" + "="*70 + "\n"
    
    return report


def test_sample_prompts():
    """Test with sample prompts."""
    print("\n" + "="*70)
    print("A/B TESTING - SAMPLE PROMPTS")
    print("="*70)
    
    # Sample text prompt
    text_prompt = """
USER_PROMPT:
It has been 60 minutes since you started trading. The current time is 2025-01-01 12:00:00 and you've been invoked 10 times.

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST

{'='*20} REAL-TIME COUNTER-TRADE ANALYSIS {'='*20}

XRP COUNTER-TRADE ANALYSIS:
  1h Trend: BULLISH | 3m Trend: BEARISH
  Conditions Met: 3/5
  Risk Level: MEDIUM_RISK
  Recommendation: MODERATE COUNTER-TRADE SETUP

SOL COUNTER-TRADE ANALYSIS:
  1h Trend: BEARISH | 3m Trend: BULLISH
  Conditions Met: 2/5
  Risk Level: HIGH_RISK
  Recommendation: WEAK COUNTER-TRADE SETUP

{'='*20} MARKET DATA {'='*20}

--- ALL XRP DATA ---
Market Regime: BULLISH
Current Price: 0.50
EMA20: 0.48
RSI: 55.0
Volume: 1000 vs Avg: 500 (2.0x)

--- ALL SOL DATA ---
Market Regime: BEARISH
Current Price: 185.0
EMA20: 190.0
RSI: 45.0
Volume: 2000 vs Avg: 1500 (1.33x)

{'='*20} PORTFOLIO {'='*20}

Total Return: 15.5%
Available Cash: $215.50
Account Value: $230.00
Sharpe Ratio: 1.5
"""
    
    # Sample JSON prompt
    json_data = {
        "metadata": {
            "minutes_running": 60,
            "current_time": "2025-01-01T12:00:00",
            "invocation_count": 10
        },
        "counter_trade_analysis": [
            {
                "coin": "XRP",
                "htf_trend": "BULLISH",
                "three_m_trend": "BEARISH",
                "conditions": {"total_met": 3},
                "risk_level": "MEDIUM_RISK"
            },
            {
                "coin": "SOL",
                "htf_trend": "BEARISH",
                "three_m_trend": "BULLISH",
                "conditions": {"total_met": 2},
                "risk_level": "HIGH_RISK"
            }
        ],
        "market_data": [
            {
                "coin": "XRP",
                "market_regime": "BULLISH",
                "timeframes": {
                    "3m": {
                        "current": {
                            "price": 0.50,
                            "ema20": 0.48,
                            "rsi": 55.0,
                            "volume": 1000
                        }
                    }
                }
            },
            {
                "coin": "SOL",
                "market_regime": "BEARISH",
                "timeframes": {
                    "3m": {
                        "current": {
                            "price": 185.0,
                            "ema20": 190.0,
                            "rsi": 45.0,
                            "volume": 2000
                        }
                    }
                }
            }
        ],
        "portfolio": {
            "total_return_pct": 15.5,
            "available_cash": 215.50,
            "account_value": 230.00,
            "sharpe_ratio": 1.5
        }
    }
    
    json_prompt = f"""
USER_PROMPT:
It has been {json_data['metadata']['minutes_running']} minutes since you started trading. The current time is {json_data['metadata']['current_time']} and you've been invoked {json_data['metadata']['invocation_count']} times.

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST

{'='*20} REAL-TIME COUNTER-TRADE ANALYSIS {'='*20}

COUNTER_TRADE_ANALYSIS (JSON):
{safe_json_dumps(json_data['counter_trade_analysis'], compact=False)}

{'='*20} MARKET DATA {'='*20}

MARKET_DATA (JSON):
{safe_json_dumps(json_data['market_data'], compact=False)}

{'='*20} PORTFOLIO {'='*20}

PORTFOLIO (JSON):
{safe_json_dumps(json_data['portfolio'], compact=False)}
"""
    
    # Generate report
    report = generate_comparison_report(text_prompt, json_prompt)
    print(report)
    
    # Test with compact mode
    print("\n" + "="*70)
    print("COMPACT MODE COMPARISON")
    print("="*70)
    
    json_prompt_compact = f"""
USER_PROMPT:
It has been {json_data['metadata']['minutes_running']} minutes since you started trading.

COUNTER_TRADE_ANALYSIS (JSON):
{safe_json_dumps(json_data['counter_trade_analysis'], compact=True)}

MARKET_DATA (JSON):
{safe_json_dumps(json_data['market_data'], compact=True)}

PORTFOLIO (JSON):
{safe_json_dumps(json_data['portfolio'], compact=True)}
"""
    
    compact_comparison = compare_token_usage(text_prompt, json_prompt_compact)
    print(f"\nText Format:  {compact_comparison['format1_tokens']:,} tokens")
    print(f"Compact JSON:  {compact_comparison['format2_tokens']:,} tokens")
    print(f"Difference:    {compact_comparison['difference']:,} tokens ({compact_comparison['difference_pct']:.2f}%)")
    print(f"More Efficient: {'Compact JSON' if compact_comparison['more_efficient'] else 'Text'}")


if __name__ == '__main__':
    test_sample_prompts()

