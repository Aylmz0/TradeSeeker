import requests
import json
import time
from typing import Dict, List, Any, Optional
from config.config import Config
from src.utils import RetryManager, safe_file_read

# HTF_INTERVAL used in prompt, we can get it from Config
HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'
HTF_LABEL = HTF_INTERVAL

class DeepSeekAPI:
    """DeepSeek API integration with fully structured JSON prompting (Logic Preservation Edition)"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        self.session = RetryManager.create_session_with_retry()

        if not self.api_key:
            print("❌ DEEPSEEK_API_KEY not found!")
            print("ℹ️  Please check your .env file configuration.")

    def _build_system_prompt(self) -> str:
        """
        Constructs the system prompt as a structured JSON object.
        Preserves 100% of the logic from the original text prompt.
        """
        system_structure = {
            "agent_profile": {
                "role": "Zero-shot systematic trading model",
                "competition": "Alpha Arena",
                "objective": "Maximize PnL via perpetual futures trading",
                "assets": ["XRP", "DOGE", "ASTER", "ADA", "LINK", "SOL"],
                "capital_settings": {
                    "initial_balance": Config.INITIAL_BALANCE,
                    "leverage": "10x (FIXED for all trades)"
                }
            },
            "constraints": {
                "trading_rules": {
                    "max_simultaneous_positions": 5,
                    "same_direction_limit": {
                        "limit": Config.SAME_DIRECTION_LIMIT,
                        "rule": f"If you have {Config.SAME_DIRECTION_LIMIT} LONGs, you CANNOT open another LONG. Same for SHORTs."
                    },
                    "min_confidence": 0.4
                },
                "risk_management": {
                    "risk_reward_ratio": "Maintain at least 1:1.3",
                    "stop_loss_basis": f"{HTF_LABEL} ATR or Swing High/Low",
                    "invalidation_requirement": "Must be explicit (e.g., 'Close below EMA20')"
                }
            },
            "strategy": {
                "philosophy": "Evaluate both LONG and SHORT paths. Bullish regimes support longs; bearish regimes support shorts.",
                "neutral_regime_logic": {
                    "definition": "1h trend ambiguous or conflicts with lower timeframes (e.g. 1h bullish but 15m/3m bearish).",
                    "action": "Take direction with best quantified edge. Both LONG and SHORT are valid."
                },
                "entry_logic": {
                    "trend_following": {
                        "priority": "High",
                        "condition": f"Price aligns with {HTF_LABEL} EMA20 + Volume support",
                        "strong_signal": f"{HTF_LABEL} + 15m + 3m all align in same direction"
                    },
                    "counter_trend": {
                        "priority": "Conditional",
                        "confidence_threshold": 0.65,
                        "strong_setup_confidence": 0.60,
                        "definition": f"Trade direction is OPPOSITE to {HTF_LABEL} trend.",
                        "condition": "15m AND 3m momentum align against 1h structural trend.",
                        "direction_rule": "Counter-trend direction = 15m+3m direction (NOT 1h direction).",
                        "restriction": "If a valid counter-trend signal exists but cannot be executed (e.g. limits), DO NOT open a trend-following trade in the opposite direction."
                    },
                    "volume_rules": {
                         "weakness_warning": "If volume ratio is <= 0.20x average, call out weakness, reduce confidence materially, and consider skipping unless other data overwhelmingly compensates."
                    }
                },
                "exit_logic": {
                    "reversal_warning": {
                        "applicability": "Applies ONLY to EXISTING positions. Do NOT use for entries.",
                        "definition": "Momentum moving AGAINST your current position.",
                        "strong_signal": "15m + 3m BOTH show reversal against position.",
                        "action": "Consider closing if PnL is negative or thesis invalidated."
                    },
                    "reversal_strength_definitions": {
                        "STRONG": f"15m + 3m BOTH show reversal against position (but {HTF_LABEL} doesn't). Consider closing.",
                        "MEDIUM": "Only 3m shows reversal against position. Continue monitoring, don't overreact to noise.",
                        "INFORMATIONAL": "Only 15m shows reversal against position. Prioritize 1h trend."
                    }
                },
                "startup_behavior": {
                    "cycles_1_to_3": "Observe unless an exceptional, well-supported setup appears.",
                    "general": "Avoid impulsive entries immediately after reset. Maintain up to 5 concurrent positions; choose quality over quantity."
                }
            },
            "advanced_playbook": [
                "Apply long and short strategies across all coins; choose the direction that offers the superior quantified edge.",
                "Monitor volume vs. average volume, Open Interest, and Funding to measure conviction.",
                "Employ multi-timeframe technical analysis (EMA, RSI, MACD, ATR, etc.).",
                "Keep take-profit/stop-loss targets responsive (e.g. 2-4% TP, 1-2% SL) when volatility supports it.",
                "Manage exits proactively; do not wait for targets if data invalidates the thesis.",
                "High-confidence setups (0.7-0.8+) justify higher exposure within risk limits.",
                "Consider both trend-following and counter-trend opportunities equally; choose the setup with the best quantified edge.",
                "BE AGGRESSIVE but disciplined - Take calculated risks based on technical analysis."
            ],
            "analysis_process": [
                "1. Check global and per-asset regime data.",
                f"2. Analyze {HTF_LABEL} (1h) indicators for structural trend.",
                "3. Analyze 15m indicators for medium-term momentum.",
                "4. Use 3m indicators for entry/exit timing.",
                "5. Check alignment across all three timeframes.",
                "6. Incorporate Volume, Open Interest, Funding.",
                "7. Decide direction based on strongest quantified edge.",
                "8. Verify constraints (Position Slots) before proposing."
            ],
            "data_protocol": {
                "series_order": "OLDEST -> NEWEST",
                "indicators": ["Price", "EMA", "RSI", "MACD", "Volume", "Open Interest", "Funding"],
                "syntax_requirement": "Compare values explicitly (e.g., 'price=2.5 > EMA=2.4')",
                "authoritative_source": "Treat the supplied data as the authoritative source for every decision."
            },
            "response_schema": {
                "format": "JSON",
                "required_keys": ["CHAIN_OF_THOUGHTS", "DECISIONS"],
                "CHAIN_OF_THOUGHTS": f"String. Analyze {HTF_LABEL}, 15m, and 3m for EACH coin. Justify decisions.",
                "DECISIONS": {
                    "COIN_TICKER": {
                        "signal": "buy_to_enter | sell_to_enter | hold | close_position",
                        "leverage": 10,
                        "confidence": "float (0.0-1.0)",
                        "profit_target": "float",
                        "stop_loss": "float",
                        "invalidation_condition": "string"
                    }
                }
            },
            "few_shot_examples": [
                {
                    "style": "NOF1AI Advanced Style",
                    "input_context": "Market data showing mixed signals...",
                    "output_example": {
                        "CHAIN_OF_THOUGHTS": f"Advanced systematic analysis of all assets using {HTF_LABEL} (1h) trends, 15m momentum confirmation, and 3m entry timing.\n\nXRP: 1h bullish (price > EMA20, RSI 62.5), 15m bullish momentum (price > EMA20, RSI 58), 3m bullish (price > EMA20, RSI 60). All three timeframes aligned bullish with volume confirmation. Open Interest increasing suggests institutional interest. Targeting $0.56 with stop below $0.48. Invalidation if {HTF_LABEL} price closes below EMA20.\n\nSOL: 1h bearish, 15m bearish, 3m bearish. Strong trend-following SHORT setup.\n\nADA: Mixed signals, holding.\n\nDOGE: Bullish trend but overextended, waiting for pullback.\n\nLINK: Low volume, skipping.\n\nASTER: Range bound, no clear edge.",
                        "DECISIONS": {
                            "XRP": {
                                "signal": "buy_to_enter",
                                "leverage": 10,
                                "confidence": 0.75,
                                "profit_target": 0.56,
                                "stop_loss": 0.48,
                                "invalidation_condition": f"If {HTF_LABEL} price closes below {HTF_LABEL} EMA20"
                            },
                            "SOL": {
                                "signal": "sell_to_enter",
                                "leverage": 10,
                                "confidence": 0.75,
                                "profit_target": 185.0,
                                "stop_loss": 198.0,
                                "invalidation_condition": f"If {HTF_LABEL} price closes above {HTF_LABEL} EMA20"
                            },
                            "ADA": { "signal": "hold" },
                            "DOGE": { "signal": "hold" },
                            "LINK": { "signal": "hold" },
                            "ASTER": { "signal": "hold" }
                        }
                    }
                }
            ]
        }
        return json.dumps(system_structure)

    def get_ai_decision(self, prompt: str) -> str:
        """Get trading decision from DeepSeek API using structured JSON prompting"""
        if not self.api_key:
            return self._get_simulation_response(prompt)

        try:
            headers = {
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # User prompt is already a JSON string from main.py
            # We wrap it in a clear instruction
            user_message_content = f"Analyze the following market data JSON and provide decisions based on the system rules:\n\n{prompt}"

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": self._build_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": user_message_content
                    }
                ],
                "temperature": 0.5, # Optimized for JSON consistency
                "max_tokens": 4096,
                "response_format": { "type": "json_object" } # Enforce JSON output
            }

            print("🔄 Sending request to DeepSeek API (JSON Mode)...")
            response = requests.post(self.base_url, json=data, headers=headers, timeout=120)
            response.raise_for_status()

            result = response.json()
            if not result.get('choices') or not result['choices'][0].get('message'):
                raise ValueError("DeepSeek API returned unexpected structure.")
            
            content = result['choices'][0]['message']['content']
            
            # Robust JSON extraction using JSONDecoder
            try:
                # Find the first '{'
                start_index = content.find('{')
                if start_index != -1:
                    # Slice from the first '{' to the end
                    json_candidate = content[start_index:]
                    
                    # Use raw_decode to parse the JSON object and ignore trailing data
                    decoder = json.JSONDecoder()
                    obj, end_index = decoder.raw_decode(json_candidate)
                    
                    # Re-serialize to ensure valid JSON string is returned
                    # This effectively strips all extra text before and after the JSON
                    content = json.dumps(obj, indent=2)
                else:
                    print("⚠️ No JSON object found in response")
                    
            except Exception as e:
                print(f"⚠️ JSON extraction warning: {e}")
                # Fallback: try stripping markdown if extraction failed
                if "```json" in content:
                    content = content.replace("```json", "").replace("```", "")
                elif "```" in content:
                    content = content.replace("```", "")
            
            return content.strip()

        except requests.exceptions.Timeout:
             print("❌ DeepSeek API request timed out.")
             return self._get_error_response("API Timeout")
        except requests.exceptions.RequestException as e:
            print(f"❌ DeepSeek API request failed: {e}")
            return self._get_error_response(f"API Request Failed: {e}")
        except Exception as e:
            print(f"❌ DeepSeek API error: {e}")
            return self._get_error_response(f"General API Error: {e}")

    def _get_simulation_response(self, prompt: str) -> str:
        """Simulation response without API - Returns valid JSON string"""
        print("⚠️  Using simulation mode...")
        simulation_data = {
            "CHAIN_OF_THOUGHTS": f"Simulation Mode: Assuming market pullback. Shorting SOL based on simulated {HTF_LABEL} resistance. Aiming for 1:1.5 R/R using simulated ATR. Holding others.",
            "DECISIONS": {
                "SOL": {
                    "signal": "sell_to_enter",
                    "leverage": 10,
                    "confidence": 0.65,
                    "profit_target": 185.0,
                    "stop_loss": 198.0,
                    "invalidation_condition": "If price closes above 199.0"
                },
                "XRP": { "signal": "hold" },
                "ADA": { "signal": "hold" },
                "DOGE": { "signal": "hold" },
                "ASTER": { "signal": "hold" },
                "LINK": { "signal": "hold" }
            }
        }
        return json.dumps(simulation_data, indent=2)

    def get_cached_decisions(self) -> str:
        """Get cached decisions from recent successful cycles - Returns valid JSON string"""
        try:
            cached_cycles = safe_file_read("data/cycle_history.json", default_data=[])
            if not cached_cycles:
                return self.get_safe_hold_decisions()
            
            for cycle in reversed(cached_cycles[-5:]):  # Last 5 cycles
                decisions = cycle.get('decisions', {})
                if decisions and isinstance(decisions, dict):
                    valid_signals = [d for d in decisions.values() if isinstance(d, dict) and d.get('signal') in ['buy_to_enter', 'sell_to_enter']]
                    if valid_signals:
                        print("🔄 Using cached decisions from recent successful cycle")
                        fallback_response = {
                            "CHAIN_OF_THOUGHTS": "API Error - Using cached decisions from recent successful cycle. Continuing with established strategy.",
                            "DECISIONS": decisions
                        }
                        return json.dumps(fallback_response, indent=2)
            
            return self.get_safe_hold_decisions()
            
        except Exception as e:
            print(f"⚠️ Cache retrieval error: {e}")
            return self.get_safe_hold_decisions()

    def get_safe_hold_decisions(self) -> str:
        """Generate safe hold decisions for all coins - Returns valid JSON string"""
        print("🛡️ Generating safe hold decisions")
        hold_decisions = {}
        for coin in ['XRP', 'DOGE', 'ASTER', 'ADA', 'LINK', 'SOL']:
            hold_decisions[coin] = {"signal": "hold", "justification": "Safe mode: Holding due to API error"}
        
        safe_response = {
            "CHAIN_OF_THOUGHTS": "API Error - Operating in safe mode. Holding all positions/cash to preserve capital.",
            "DECISIONS": hold_decisions
        }
        return json.dumps(safe_response, indent=2)

    def _get_error_response(self, error_message: str) -> str:
        """Enhanced error response with intelligent recovery"""
        print(f"🔧 Enhanced error handling for: {error_message}")
        
        error_type = type(error_message).__name__ if isinstance(error_message, Exception) else str(error_message)
        
        if "Connection" in error_type or "Timeout" in error_type:
            return self.get_cached_decisions()
        
        return self.get_safe_hold_decisions()
