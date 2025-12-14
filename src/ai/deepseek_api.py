import json
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI
from config.config import Config
from src.utils import RetryManager, safe_file_read

# HTF_INTERVAL used in prompt, we can get it from Config
HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'
HTF_LABEL = HTF_INTERVAL

class DeepSeekAPI:
    """DeepSeek API integration with fully structured JSON prompting"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-chat"
        
        if not self.api_key:
            print("❌ DEEPSEEK_API_KEY not found!")
            print("ℹ️  Please check your .env file configuration.")
            self.client = None
        else:
            # Set global timeout to 180 seconds to prevent premature 30s timeouts
            self.client = OpenAI(
                api_key=self.api_key, 
                base_url=self.base_url,
                timeout=180.0,
                max_retries=2
            )

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
                "assets": ["XRP", "DOGE", "ASTER", "TRX", "LINK", "SOL"],
                "capital_settings": {
                    "initial_balance": Config.INITIAL_BALANCE
                }
            },
            "constraints": {
                "trading_rules": {
                    "max_simultaneous_positions": 5,
                    "same_direction_limit": {
                        "limit": Config.SAME_DIRECTION_LIMIT,
                        "rule": f"If you have {Config.SAME_DIRECTION_LIMIT} LONGs, you CANNOT open another LONG. Same for SHORTs."
                    },
                    "min_confidence": Config.MIN_CONFIDENCE,
                    "discipline": "SNIPER MODE: Only trade high-probability setups. Holding cash is valid when no clear edge exists."
                },
                "risk_management": {
                    "risk_reward_ratio": "Maintain a positive risk/reward ratio.",
                    "stop_loss_basis": "Logical technical level (e.g., recent support/resistance or ATR-based).",
                    "invalidation_requirement": "Must be explicit and INCLUDE A 0.2% BUFFER to prevent wick-outs (e.g., 'Close if price < EMA20 * 0.998' for LONG, '... > EMA20 * 1.002' for SHORT).",
                    "CRITICAL_WARNING": "System has a HARD MARGIN STOP LOSS. Do NOT rely on wider stops."
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
                        "strong_setup_confidence": 0.70,
                        "definition": f"Trade direction is OPPOSITE to {HTF_LABEL} trend.",
                        "scope": "risk_level applies ONLY when trading AGAINST 1h trend.",
                        "condition": "Evaluate 'risk_level' provided in counter_trade_analysis.",
                        "risk_level_rules": {
                            "LOW_RISK": "STRONG+3 OR MEDIUM+4 conditions. EXECUTE.",
                            "MEDIUM_RISK": "STRONG+1-2 OR MEDIUM+3 conditions. EXECUTE if confidence > 0.65.",
                            "HIGH_RISK": "MEDIUM alignment + <3 conditions. Evaluate carefully, prefer HOLD.",
                            "VERY_HIGH_RISK": "No alignment (15m+3m both follow HTF). Do NOT trade."
                        },
                        "restriction": "Do NOT trade if risk_level is VERY_HIGH_RISK. HIGH_RISK requires extreme caution."
                    },
                    "volume_rules": {
                         "weakness_warning": "If volume ratio is <= 0.30x average, DO NOT TRADE. This is a hard rule (Sniper mode).",
                         "low_volume_caution": "If volume ratio is 0.30-0.60x, reduce confidence significantly."
                    },
                    "momentum_conviction_rule": {
                        "description": "How 15m momentum quality affects entry timing",
                        "STRENGTHENING": "Trend accelerating. Proceed with entry normally.",
                        "STABLE": "Trend steady. Proceed with entry normally.",
                        "WEAKENING": "Trend losing conviction. Wait for momentum stabilization before entering. If 15m WEAKENING and 3m is opposite direction, trend conviction is very weak - prefer HOLD."
                    },
                    "zone_weakening_combined_rule": {
                        "description": "CRITICAL RULE: Zone + WEAKENING combination signals trend exhaustion",
                        "UPPER_10_WEAKENING": {
                            "for_LONG_entry": "DO NOT open LONG. Trend exhausted at highs. Prefer HOLD or evaluate SHORT.",
                            "for_LONG_exit": "If you have LONG: signal close_position immediately. Do NOT wait for stop loss.",
                            "for_SHORT_entry": "GOOD counter-trend opportunity. Proceed with SHORT if conditions align.",
                            "for_SHORT_exit": "SHORT is SAFE at UPPER_10. Continue holding - trend favorably exhausting."
                        },
                        "LOWER_10_WEAKENING": {
                            "for_SHORT_entry": "DO NOT open SHORT. Trend exhausted at lows. Prefer HOLD or evaluate LONG.",
                            "for_SHORT_exit": "If you have SHORT: signal close_position immediately. Do NOT wait for stop loss.",
                            "for_LONG_entry": "GOOD counter-trend opportunity. Proceed with LONG if conditions align.",
                            "for_LONG_exit": "LONG is SAFE at LOWER_10. Continue holding - trend favorably exhausting."
                        }
                    },
                    # DISABLED FOR A/B TESTING (zone+weakening remains active)
                    # "zone_rsi_extreme_rule": {
                    #     "description": "CRITICAL RULE: Zone + RSI extreme combination signals high reversal probability",
                    #     "check_condition": "15m RSI and price_location",
                    #     "LOWER_10_RSI_OVERSOLD": {
                    #         "condition": "price_location = LOWER_10 AND RSI < 30",
                    #         "for_SHORT_entry": "HIGH RISK for SHORT. Bounce probability high. Prefer HOLD or evaluate LONG.",
                    #         "for_SHORT_exit": "If you have SHORT: signal close_position immediately. Bounce imminent.",
                    #         "for_LONG_entry": "GOOD counter-trend opportunity. Proceed with LONG if volume supports.",
                    #         "for_LONG_exit": "LONG is SAFE at oversold. Continue holding."
                    #     },
                    #     "UPPER_10_RSI_OVERBOUGHT": {
                    #         "condition": "price_location = UPPER_10 AND RSI > 70",
                    #         "for_LONG_entry": "HIGH RISK for LONG. Pullback probability high. Prefer HOLD or evaluate SHORT.",
                    #         "for_LONG_exit": "If you have LONG: signal close_position immediately. Pullback imminent.",
                    #         "for_SHORT_entry": "GOOD counter-trend opportunity. Proceed with SHORT if volume supports.",
                    #         "for_SHORT_exit": "SHORT is SAFE at overbought. Continue holding."
                    #     }
                    # }
                },
                "exit_logic": {
                    "reversal_warning": {
                        "applicability": "Applies ONLY to EXISTING positions. Do NOT use for entries.",
                        "definition": "Momentum moving AGAINST your current position.",
                        "strong_signal": "15m + 3m BOTH show reversal against position.",
                        "action": "Consider closing ONLY if PnL is negative or thesis invalidated. Do NOT exit on weak 3m reversals alone."
                    },
                    "reversal_strength_definitions": {
                        "STRONG": f"15m + 3m BOTH show reversal against position (but {HTF_LABEL} doesn't). Consider closing.",
                        "MEDIUM": "Only 15m shows reversal against position. Monitor closely, protect profits if any.",
                        "INFORMATIONAL": "Only 3m shows reversal against position. May be noise, continue watching."
                    },
                    "profit_erosion_rules": {
                        "description": "Rules for protecting profits based on peak_pnl erosion tracking",
                        "fields": {
                            "peak_pnl": "Highest profit reached for this position ($)",
                            "erosion_pct": "How much of peak profit has eroded (%)",
                            "erosion_status": "NONE (<20%), MINOR (20-50%), SIGNIFICANT (50-100%), CRITICAL (>100%)"
                        },
                        "actions": {
                            "NONE": "Normal fluctuation. Continue with existing exit plan.",
                            "MINOR": "Watch closely. Tighten mental stop if reversal signals appear.",
                            "SIGNIFICANT": "Over 50% of peak profit eroded. Close if reversal_strength >= MEDIUM.",
                            "CRITICAL": "Peak profit fully eroded or now losing. Close unless trend still strongly supports position."
                        },
                        "combined_decision": "Combine erosion_status with reversal_strength: SIGNIFICANT/CRITICAL + MEDIUM/STRONG reversal = close position."
                    }
                },
                "startup_behavior": {
                    "cycles_1_to_3": "Observe unless an exceptional, well-supported setup appears. Do NOT cite this rule after cycle 3.",
                    "cycles_4_plus": "Normal trading mode. Apply all rules without startup caution. Trade when conditions are met.",
                    "general": "Avoid impulsive entries immediately after reset. Maintain up to 5 concurrent positions; choose quality over quantity."
                }
            },
            "advanced_playbook": [
                "Apply long and short strategies across all coins; choose the direction that offers the superior quantified edge.",
                "Monitor volume vs. average volume, Open Interest, and Funding to measure conviction.",
                "Employ multi-timeframe technical analysis (EMA, RSI, MACD, ATR, etc.).",
                "Keep take-profit/stop-loss targets responsive and logical.",
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
            "enhanced_context_definitions": {
                "smart_sparkline": {
                    "description": "Price pattern analysis. HTF (1h) includes full data (24h), 15m includes structure+momentum+price_location (6h).",
                    "key_level": "Nearest support or resistance (HTF only). strength = how many times tested (1-5). distance_pct = distance from price.",
                    "structure": {
                        "HH_HL": "Higher Highs + Higher Lows = Bullish structure",
                        "LH_LL": "Lower Highs + Lower Lows = Bearish structure",
                        "RANGE": "Price consolidating, potential breakout",
                        "UNCLEAR": "No clear pattern"
                    },
                    "momentum": "STRENGTHENING (trend accelerating) | STABLE | WEAKENING (trend losing steam)",
                    "price_location": {
                        "description": "Where is the current price within the period's high-low range",
                        "zone": {
                            "LOWER_10": "Price in bottom 10% of period range. When combined with RSI < 25, this indicates the asset may be at a short-term bottom with high bounce probability. Consider reducing SHORT confidence or waiting for confirmation.",
                            "UPPER_10": "Price in top 10% of period range. When combined with RSI > 75, this indicates the asset may be at a short-term top with high pullback probability. Consider reducing LONG confidence or waiting for confirmation.",
                            "MIDDLE": "Price in normal range. No extreme location-based risk."
                        },
                        "percentile": "0-100 scale. 0 = at period low, 100 = at period high.",
                        "guidance": "This is NOT a hard rule. Trend can continue through support/resistance. Use price_location + RSI together to assess risk. If LOWER_10 + RSI<25, mention bounce risk in analysis. If UPPER_10 + RSI>75, mention pullback risk."
                    },
                    "usage": "Use 15m data for shorter-term confirmation. Check price_location on both 1h and 15m when deciding entries."
                },
                "tags": "Analytical labels (e.g., 'Vol_High', 'RSI_Overbought'). Use as confirmation."
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
                    "style": "Advanced Style",
                    "input_context": "Market data showing mixed signals...",
                    "output_example": {
                        "CHAIN_OF_THOUGHTS": f"Advanced systematic analysis of all assets using {HTF_LABEL} (1h) trends, 15m momentum confirmation, and 3m entry timing.\n\nXRP: 1h bullish (price=0.54 > EMA20=0.52, RSI 62.5), 15m bullish (RSI 58), 3m bullish (RSI 60). All three timeframes aligned bullish with volume ratio 1.2x. Price near support@0.52 (tested 3x) adds confluence. Open Interest increasing. Targeting $0.56 with stop below $0.48.\n\nSOL: 1h bearish, 15m bearish, 3m bearish. Volume 0.85x (normal). Strong trend-following SHORT setup.\n\nTRX: 1h bullish, 15m neutral, 3m bearish. Mixed signals, near resistance level. HOLD.\n\nDOGE: 1h bullish but RSI 72 (overbought). Momentum weakening. Waiting for pullback.\n\nLINK: Volume ratio 0.15x (< 0.20 threshold). DO NOT TRADE per hard rule.\n\nASTER: Structure=RANGE (consolidation), no clear directional edge. HOLD.",
                        "DECISIONS": {
                            "XRP": {
                                "signal": "buy_to_enter",
                                "leverage": 10,
                                "confidence": 0.75,
                                "profit_target": 0.56,
                                "stop_loss": 0.48,
                                "invalidation_condition": f"If {HTF_LABEL} price closes below {HTF_LABEL} EMA20 * 0.998"
                            },
                            "SOL": {
                                "signal": "sell_to_enter",
                                "leverage": 10,
                                "confidence": 0.75,
                                "profit_target": 185.0,
                                "stop_loss": 198.0,
                                "invalidation_condition": f"If {HTF_LABEL} price closes above {HTF_LABEL} EMA20 * 1.002"
                            },
                            "TRX": { "signal": "hold" },
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
        if not self.client:
            return self._get_simulation_response(prompt)

        try:
            # User prompt is already a JSON string from main.py
            # We wrap it in a clear instruction
            user_message_content = f"Analyze the following market data JSON and provide decisions based on the system rules:\n\n{prompt}"

            print(f"🔄 Sending request to DeepSeek API (JSON Mode)... Payload Size: {len(prompt)} chars")
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": self._build_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": user_message_content
                    }
                ],
                temperature=0.5,
                max_tokens=4000,
                response_format={ "type": "json_object" },
                stream=True
            )

            print("⏳ Receiving stream...", end="", flush=True)
            collected_content = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    collected_content.append(content_chunk)
                    # Optional: Print a dot every 100 chars to show activity
                    if len(collected_content) % 10 == 0:
                        print(".", end="", flush=True)
            
            print(" ✅")
            content = "".join(collected_content)
            
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

        except Exception as e:
            # Detailed error logging
            error_type = type(e).__name__
            print(f"❌ DeepSeek API error ({error_type}): {e}")
            return self._get_error_response(f"{error_type}: {e}")

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
                "TRX": { "signal": "hold" },
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
        for coin in ['XRP', 'DOGE', 'ASTER', 'TRX', 'LINK', 'SOL']:
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
