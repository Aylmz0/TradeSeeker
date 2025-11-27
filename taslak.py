import requests
import json
import time
from typing import Dict, List, Any, Optional
from config.config import Config
from src.utils import RetryManager, safe_file_read

# Config'den HTF ayarlarını al
HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'
HTF_LABEL = HTF_INTERVAL

class DeepSeekAPI:
    """DeepSeek API integration with JSON structured prompting and enhanced error handling"""

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
        Sistem talimatlarını yapılandırılmış JSON formatında oluşturur.
        Bu yapı AI'ın kuralları 'kesin kısıtlamalar' olarak algılamasını sağlar.
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
                "max_simultaneous_positions": 5,
                "same_direction_limit": {
                    "limit": Config.SAME_DIRECTION_LIMIT,
                    "rule": f"If you have {Config.SAME_DIRECTION_LIMIT} LONGs, you CANNOT open another LONG. Same for SHORTs."
                },
                "risk_management": {
                    "min_confidence_score": 0.4,
                    "min_risk_reward_ratio": "1:1.3",
                    "stop_loss_basis": f"{HTF_LABEL} ATR or Swing High/Low",
                    "invalidation_requirement": "Must be explicit (e.g., 'Close below EMA20')"
                }
            },
            "strategy_matrix": {
                "trend_following": {
                    "priority": "High",
                    "condition": f"Price aligns with {HTF_LABEL} EMA20 + Volume support",
                    "strong_signal": "1h + 15m + 3m all align in same direction"
                },
                "counter_trend": {
                    "priority": "Conditional",
                    "min_confidence": 0.65,
                    "condition": "15m AND 3m momentum opposes 1h structural trend",
                    "direction_rule": "Trade in direction of 15m+3m (NOT 1h)",
                    "restriction": "Do NOT open trend-following trade if a valid counter-trend signal exists"
                },
                "neutral_regime": {
                    "description": "1h trend ambiguous or conflicts with lower timeframes",
                    "action": "Take direction with best quantified edge (Long or Short)"
                },
                "reversal_exit_logic": {
                    "description": "Applies ONLY to EXISTING positions. Do NOT use for entries.",
                    "strong_warning": "15m + 3m align AGAINST position direction",
                    "action": "Consider closing if PnL is negative or thesis invalidated"
                }
            },
            "data_interpretation": {
                "series_order": "OLDEST -> NEWEST",
                "indicators": ["Price", "EMA", "RSI", "MACD", "Volume", "Open Interest", "Funding"],
                "syntax_requirement": "Compare values explicitly (e.g., 'price=2.5 > EMA=2.4')"
            },
            "response_schema": {
                "format": "JSON",
                "required_keys": ["CHAIN_OF_THOUGHTS", "DECISIONS"],
                "CHAIN_OF_THOUGHTS": "String. Analyze {HTF_LABEL}, 15m, and 3m for EACH coin. Justify decisions.",
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
            }
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
            
            # Kullanıcı prompt'unu da JSON bağlamına alabiliriz (isteğe bağlı ama önerilir)
            # Şimdilik string olarak bırakıyoruz ama açık bir talimat ekliyoruz.
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
                "temperature": 0.5, # JSON tutarlılığı için temperature düşürüldü
                "max_tokens": 4096,
                "response_format": { "type": "json_object" } # DeepSeek JSON modu aktif
            }

            print("🔄 Sending request to DeepSeek API (JSON Mode)...")
            response = requests.post(self.base_url, json=data, headers=headers, timeout=120)
            response.raise_for_status()

            result = response.json()
            if not result.get('choices') or not result['choices'][0].get('message'):
                raise ValueError("DeepSeek API returned unexpected structure.")
            
            return result['choices'][0]['message']['content']

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
        """Simulation response without API"""
        print("⚠️  Using simulation mode...")
        # JSON formatında simülasyon cevabı
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
        """Get cached decisions from recent successful cycles"""
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
        """Generate safe hold decisions for all coins"""
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