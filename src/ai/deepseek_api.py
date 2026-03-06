import json

from openai import OpenAI

from config.config import Config
from src.utils import safe_file_read

# Try to import Gemini SDK
try:
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Try to import Groq SDK
try:
    from groq import Groq
    
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# HTF_INTERVAL used in prompt, we can get it from Config
HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"
HTF_LABEL = HTF_INTERVAL


class DeepSeekAPI:
    """AI API integration with MiMo/Z.AI/DeepSeek support"""

    def __init__(self, api_key: str = None):
        primary_provider = getattr(Config, "PRIMARY_AI_PROVIDER", "openrouter")
        mimo_key = getattr(Config, "MIMO_API_KEY", None)
        zai_key = getattr(Config, "ZAI_API_KEY", None)

        self.gemini_client = None
        self.groq_client = None
        self.client = None

        if primary_provider == "groq" and getattr(Config, "GROQ_API_KEY", None) and GROQ_AVAILABLE:
            self.api_key = Config.GROQ_API_KEY
            self.model = getattr(Config, "GROQ_MODEL", "groq/compound")
            self.provider = "Groq"
            self.thinking_enabled = False
            print(f"[INFO] Using Groq API with model: {self.model}")
            self.groq_client = Groq(api_key=self.api_key)
        elif mimo_key:
            # Use MiMo (Xiaomi AI - fast and free)
            self.api_key = mimo_key
            self.base_url = "https://api.xiaomimimo.com/v1"
            self.model = getattr(Config, "MIMO_MODEL", "mimo-v2-flash")
            self.provider = "MiMo"
            self.thinking_enabled = getattr(Config, "MIMO_THINKING_ENABLED", False)
            print(f"[INFO] Using MiMo API with model: {self.model}")
            self.client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=180.0, max_retries=2,
            )
        elif zai_key:
            # Use Z.AI (GLM models with thinking support)
            self.api_key = zai_key
            self.base_url = "https://api.z.ai/api/paas/v4/"
            self.model = getattr(Config, "ZAI_MODEL", "glm-4.5-flash")
            self.provider = "Z.AI"
            self.thinking_enabled = getattr(Config, "ZAI_THINKING_ENABLED", True)
            print(f"[INFO] Using Z.AI API with model: {self.model} (thinking: {self.thinking_enabled})")
            self.client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=180.0, max_retries=2,
            )
        elif getattr(Config, "OPENROUTER_API_KEY", None):
            # Use OpenRouter (Excellent for cost management)
            self.api_key = Config.OPENROUTER_API_KEY
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = Config.OPENROUTER_MODEL
            self.provider = "OpenRouter"
            self.thinking_enabled = getattr(Config, "OPENROUTER_REASONING_ENABLED", True)
            print(f"[INFO] Using OpenRouter API with model: {self.model} (reasoning: {self.thinking_enabled})")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=180.0,
                max_retries=2,
                default_headers={
                    "HTTP-Referer": "https://github.com/Aylmz0/TradeSeeker",
                    "X-Title": "TradeSeeker AI",
                },
            )
        else:
            # Fallback to DeepSeek
            self.api_key = api_key or Config.DEEPSEEK_API_KEY
            self.base_url = "https://api.deepseek.com"
            self.model = "deepseek-chat"
            self.provider = "DeepSeek"
            self.thinking_enabled = False

            if self.api_key:
                self.client = OpenAI(
                    api_key=self.api_key, base_url=self.base_url, timeout=180.0, max_retries=2,
                )

        if not self.api_key:
            print("[ERR]   No API key found! Set MIMO_API_KEY, ZAI_API_KEY or DEEPSEEK_API_KEY in .env")
            print("[INFO] Please check your .env file configuration.")

    def _build_system_prompt(self) -> str:
        """
        Constructs the system prompt as a structured JSON object.
        Preserves 100% of the logic from the original text prompt.
        """
        system_structure = {
            "agent_profile": {
                "role": "Elite Hybrid Intelligence Orchestrator (LLM + XGBoost)",
                "competition": "Alpha Arena 2026",
                "objective": "Maximize PnL via perpetual futures trading",
                "assets": ["XRP", "DOGE", "ASTER", "TRX", "ETH", "SOL"],
                "capital_settings": {"initial_balance": Config.INITIAL_BALANCE},
            },
            "constraints": {
                "trading_rules": {
                    "max_simultaneous_positions": 5,
                    "same_direction_limit": {
                        "limit": Config.SAME_DIRECTION_LIMIT,
                        "rule": f"If you have {Config.SAME_DIRECTION_LIMIT} LONGs, you CANNOT open another LONG. Same for SHORTs.",
                    },
                    "min_confidence": Config.MIN_CONFIDENCE,
                    "discipline": "SNIPER MODE: Only trade high-probability setups. Holding cash is valid when no clear edge exists.",
                },
                "risk_management": {
                    "risk_reward_ratio": "Maintain a positive risk/reward ratio.",
                    "invalidation_rules": {
                        "requirement": "Every entry MUST have an explicit invalidation_condition.",
                        "by_strategy": {
                            "trend_following": {
                                "basis": "EMA/Structure break",
                                "rule": "Close if price crosses EMA20 (0.2% buffer) OR 15m structure reverses",
                                "example_LONG": "Close if price < EMA20 * 0.998 or 15m structure LH_LL",
                                "example_SHORT": "Close if price > EMA20 * 1.002 or 15m structure HH_HL",
                            },
                            "counter_trend": {
                                "basis": "Swing Level break",
                                "rule": "Close if price breaks recent swing high (SHORT) or swing low (LONG)",
                                "warning": "Do NOT use EMA - price is expected on opposite side",
                                "example_SHORT": "Close if price breaks above recent swing high",
                                "example_LONG": "Close if price breaks below recent swing low",
                            },
                            "risk_management": {
                                "basis": "Erosion + Reversal",
                                "rule": "Close if erosion SIGNIFICANT + reversal MEDIUM or higher",
                                "example": "Profit Erosion > 50% limit breached",
                            },
                        },
                    },
                    "CRITICAL_WARNING": "System handles TP/SL calculation. Your job: Signal, Confidence, Invalidation LOGIC.",
                },
            },
            "strategy": {
                "philosophy": "Evaluate both LONG and SHORT paths. Bullish regimes support longs; bearish regimes support shorts.",
                "neutral_regime_logic": {
                    "definition": "1h trend ambiguous. Note: 3m is for entry TIMING only - do not use 3m to define regime conflict.",
                    "action": "Take direction with best quantified edge. Both LONG and SHORT are valid.",
                },
                "entry_logic": {
                    "trend_following": {
                        "priority": "High",
                        "condition": f"Price aligns with {HTF_LABEL} EMA20 + Volume support",
                        "strong_signal": f"{HTF_LABEL} + 15m + 3m all align in same direction",
                    },
                    "counter_trend": {
                        "priority": "Conditional",
                        "confidence_threshold": 0.65,
                        "strong_setup_confidence": 0.70,
                        "definition": f"Trade direction is OPPOSITE to {HTF_LABEL} trend.",
                        "scope": "risk_level applies ONLY when trading AGAINST 1h trend.",
                        "condition": "Evaluate 'counter_trade_risk' in each coin's risk_profile.",
                        "risk_level_rules": {
                            "LOW_RISK": "STRONG+4 OR MEDIUM+5 conditions. EXECUTE.",
                            "MEDIUM_RISK": "STRONG+3 OR MEDIUM+4 conditions OR NONE+7 conditions. EXECUTE if high confidence.",
                            "HIGH_RISK": "Counter-trend setup is too weak. Do NOT trade. (Wait for better alignment).",
                            "VERY_HIGH_RISK": "No alignment/conditions. Do NOT trade counter-trend. (Trend-Following unaffected).",
                        },
                    },
                    # NOTE: Volume filtering is handled by runtime code - removed from prompt to avoid AI confusion
                    "momentum_conviction_rule": {
                        "description": "How 15m momentum quality affects entry timing",
                        "STRENGTHENING": "Trend accelerating. Proceed with entry normally.",
                        "STABLE": "Trend steady. Proceed with entry normally.",
                        "WEAKENING": "Trend slowing but not reversing. For ENTRY: wait for 3m alignment. For EXIT: WEAKENING alone is NOT an exit signal - require structure reversal confirmation.",
                    },
                    "zone_weakening_combined_rule": {
                        "description": "CRITICAL RULE: Zone + WEAKENING combination signals trend exhaustion",
                        "UPPER_10_WEAKENING": {
                            "for_LONG_entry": "DO NOT open LONG. Trend exhausted at highs. Prefer HOLD or evaluate SHORT.",
                            "for_LONG_exit": "Consider exit. Close if 15m structure shows LH_LL (lower highs, lower lows).",
                            "for_SHORT_entry": "GOOD counter-trend opportunity. Proceed with SHORT if conditions align.",
                            "for_SHORT_exit": "SHORT is SAFE at UPPER_10. Continue holding - trend favorably exhausting.",
                        },
                        "LOWER_10_WEAKENING": {
                            "for_SHORT_entry": "DO NOT open SHORT. Trend exhausted at lows. Prefer HOLD or evaluate LONG.",
                            "for_SHORT_exit": "Consider exit. Close if 15m structure shows HH_HL (higher highs, higher lows).",
                            "for_LONG_entry": "GOOD counter-trend opportunity. Proceed with LONG if conditions align.",
                            "for_LONG_exit": "LONG is SAFE at LOWER_10. Continue holding - trend favorably exhausting.",
                        },
                    },
                    "zone_strengthening_combined_rule": {
                        "description": "Zone + STRENGTHENING = trend accelerating. Favor trend direction.",
                        "UPPER_10_STRENGTHENING": {
                            "for_LONG_entry": "Valid. Trend accelerating up.",
                            "for_LONG_exit": "HOLD. Exit on WEAKENING.",
                            "for_SHORT_entry": "Wait for WEAKENING.",
                            "for_SHORT_exit": "Consider exit. Momentum against.",
                        },
                        "LOWER_10_STRENGTHENING": {
                            "for_SHORT_entry": "Valid. Trend accelerating down.",
                            "for_SHORT_exit": "HOLD. Exit on WEAKENING.",
                            "for_LONG_entry": "Wait for WEAKENING.",
                            "for_LONG_exit": "Consider exit. Momentum against.",
                        },
                    },
                    "volume_rule": {
                        "threshold": "volume_ratio < 0.20 = LOW",
                        "for_entry": "Do NOT enter with LOW volume.",
                        "for_exit": "LOW volume ≠ exit signal.",
                        "labels": {
                            "excellent": "> 2.5x",
                            "good": "> 1.8x",
                            "fair": "> 1.2x",
                            "poor": "< 0.7x",
                        },
                    },
                },
                "exit_logic": {
                    "reversal_warning": {
                        "applicability": "Applies ONLY to EXISTING positions. Do NOT use for entries.",
                        "definition": "Weighted scoring of signals AGAINST your current position.",
                        "score_weights": "HTF(+3), 15m_structure(+3), 15m_momentum(+2), 3m(+1), RSI(+1), MACD(+1)",
                        "action": "Consider closing based on strength level and PnL.",
                    },
                    "reversal_strength_definitions": {
                        "NONE": "No reversal signals (score 0). Continue normally.",
                        "WEAK": "Minor signals (score 1-2). Informational only.",
                        "MODERATE": "Notable signals (score 3-4). Monitor closely.",
                        "STRONG": "Significant signals (score 5-7). Consider exit if PnL negative.",
                        "CRITICAL": "Multiple strong signals (score 8+). Urgent exit review.",
                    },
                    "profit_erosion_rules": {
                        "description": "Rules for protecting profits based on peak_pnl erosion tracking",
                        "fields": {
                            "peak_pnl": "Highest profit reached for this position ($)",
                            "erosion_pct": "How much of peak profit has eroded (%)",
                            "erosion_status": "NONE (<20%), MINOR (20-50%), SIGNIFICANT (50-100%), CRITICAL (>100%)",
                        },
                        "actions": {
                            "NONE": "Normal fluctuation. Continue with existing exit plan.",
                            "MINOR": "Watch closely. Tighten mental stop if reversal signals appear.",
                            "SIGNIFICANT": "Over 50% of peak profit eroded. Close if reversal_strength >= MEDIUM.",
                            "CRITICAL": "Peak profit fully eroded or now losing. Close unless trend still strongly supports position.",
                        },
                        "combined_decision": "Combine erosion_status with reversal_strength: SIGNIFICANT/CRITICAL + MEDIUM/STRONG reversal = close position.",
                    },
                },
                "startup_behavior": {
                    "cycles_1_to_3": "Observe unless an exceptional, well-supported setup appears. Do NOT cite this rule after cycle 3.",
                    "cycles_4_plus": "Normal trading mode. Apply all rules without startup caution. Trade when conditions are met.",
                    "general": "Avoid impulsive entries immediately after reset. Maintain up to 5 concurrent positions; choose quality over quantity.",
                },
            },
            "advanced_playbook": [
                "Apply long and short strategies across all coins; choose the direction that offers the superior quantified edge.",
                "Monitor volume vs. average volume, Open Interest, and Funding to measure conviction.",
                "Employ multi-timeframe technical analysis (EMA, RSI, MACD, ATR, etc.).",
                "Keep valid invalidation conditions (structural breaks).",
                "Manage exits proactively; do not wait for targets if data invalidates the thesis.",
                "High-confidence setups (0.7-0.8+) justify higher exposure within risk limits.",
                "Consider both trend-following and counter-trend opportunities equally; choose the setup with the best quantified edge.",
                "BE AGGRESSIVE but disciplined - Take calculated risks based on technical analysis.",
            ],
            "analysis_process": [
                "1. Check each coin's market_context.regime and risk_profile.",
                f"2. Review technical_summary: trend_alignment, momentum, structure_15m, volume_support.",
                "3. Use key_levels (price, ema20_htf, rsi_15m, atr_htf) for independent reasoning.",
                "4. CROSS-REFERENCE with ml_consensus (XGBoost) if available. If null, rely STRICTLY on technical_summary and key_levels.",
                "5. Evaluate risk_profile: counter_trade_risk + reversal_threat.",
                "6. Check sentiment: funding_rate, open_interest.",
                "7. Decide direction based on strongest quantified edge (AI logic + ML consensus if present).",
                "8. Verify constraints (Position Slots, Cooldowns) before proposing.",
            ],
            "execution_meta": {
                "entry_method": "Smart Limit Orders (Orderbook analyzed at sub-millisecond level)",
                "note": "AI's role is SIGNAL and LOGIC. Invalidation conditions must be technically sound (EMA/Structure).",
            },
            "data_protocol": {
                "format": "State Vectors — each coin has pre-processed labels + key numerical anchors.",
                "authoritative_source": "Treat the supplied data as the authoritative source for every decision.",
            },
            "state_vector_schema": {
                "ml_consensus": {
                    "description": "2026 XGBoost Model probabilities for next 15m-1h direction",
                    "fields": {
                        "BUY": "Probability of price increase (0.0-1.0)",
                        "SELL": "Probability of price decrease (0.0-1.0)",
                        "HOLD": "Probability of ranging/neutral or Model Uncertainty (0.0-1.0)",
                    },
                    "usage": "Use as a 'Statistical Tie-Breaker'. If AI logic and ML agree (e.g., AI=Long + ML_BUY > 0.65), confidence is HIGH. If ML_HOLD > 0.40, the statistical edge is weak; require stronger technical confluence for entry.",
                },
                "market_context": "regime (BULLISH/BEARISH/NEUTRAL), efficiency_ratio, volatility_state (SQUEEZE/EXPANDING/NORMAL), price_location (UPPER_10/LOWER_10/MIDDLE)",
                "technical_summary": "trend_alignment (FULL_BULLISH/FULL_BEARISH/MIXED_BULLISH/MIXED_BEARISH/CONFLICTED), momentum (STRENGTHENING/STABLE/WEAKENING), volume_ratio (numeric), volume_support (EXCELLENT/GOOD/FAIR/POOR/LOW), structure_15m (HH_HL/LH_LL/RANGE/UNCLEAR)",
                "key_levels": "price, ema20_htf, rsi_15m, atr_htf — raw numerical anchors for your independent reasoning and cross-validation of labels.",
                "risk_profile": "counter_trade_risk (LOW_RISK/MEDIUM_RISK/HIGH_RISK/VERY_HIGH_RISK), alignment_strength, reversal_threat (NONE/WEAK/MODERATE/STRONG/CRITICAL)",
                "sentiment": "funding_rate, open_interest",
                "position": "Current position details if exists, including exit_plan and erosion tracking.",
            },
            "response_schema": {
                "format": "JSON",
                "required_keys": ["CHAIN_OF_THOUGHTS", "DECISIONS"],
                "CHAIN_OF_THOUGHTS": f"String. Analyze {HTF_LABEL}, 15m, and 3m for EACH coin. Justify decisions.",
                "DECISIONS": {
                    "COIN_TICKER": {
                        "signal": "buy_to_enter | sell_to_enter | hold | close_position",
                        "strategy": "trend_following | counter_trend | risk_management",
                        "leverage": 10,
                        "confidence": "float (0.0-1.0)",
                        "invalidation_condition": "string",
                    },
                },
            },
            "few_shot_examples": [
                {
                    "style": "Sherlock Holmes Style - High-Density Hybrid Analysis",
                    "input_context": "Market data including technical indicators, ML probabilities, and portfolio state...",
                    "output_example": {
                        "CHAIN_OF_THOUGHTS": "Systematic execution: 1h Trend + 15m Momentum/Structure + 3m Timing + ML Consensus. Logic precedes decision.\\n\\nXRP: 1h Bullish (Price $0.54 > EMA20 $0.52, RSI 62, ADX 32 MODERATE). 15m Bullish (Structure HH_HL, Momentum STRENGTHENING, BB_Width 5.2% expanding). 3m Bullish (RSI 64, Volume 1.8x). ML Consensus: BUY 0.82 (Strong statistical confirmation). Hybrid Alignment: Technical confluence (1h/15m/3m) confirmed by ML edge. Execution: Smart-Limit entry at 15m VWAP. Decision: BUY_TO_ENTER (High Confidence Hybrid Follow).\\n\\nSOL (OPEN LONG): 1h Bearish (ADX 35 STRONG, Bullish Trend Broken). 15m Bearish (Structure LH_LL, Price < EMA20). Position Risk: Erosion SIGNIFICANT (55% profit decay). Reversal Score: STRONG (7/10). ML Consensus: SELL 0.78 (XGBoost confirms trend reversal). Combined Action: Technical invalidation + ML reversal signal = immediate closure. Decision: CLOSE_POSITION (Preserve Capital).\\n\\nTRX: 1h Bullish (Price $0.12 > EMA20 $0.118). 15m Bearish (Extreme dip, RSI 22 OVERSOLD, Price at BB Lower Band). 3m Reversing (MACD Bullish Cross). Context: Buying the dip in a long-term bull regime. ML Consensus: BUY 0.68 (Statistical support for mean reversion). Decision: BUY_TO_ENTER (Trend Following Dip).\\n\\nDOGE: 1h Bullish but Price Location UPPER_10 (Resistance, Sparkline Tag: 'Vol_Low'). 15m Momentum WEAKENING. ML Consensus: HOLD 0.45 (Model uncertainty). Logic: Buying at resistance with low volume and ML uncertainty = Negative Expectancy. Decision: HOLD.\\n\\nASTER: Volume Ratio 0.12x (< 0.20 Hard Filter). BB Squeeze (Volatility 1.2%). Decision: HOLD (No Edge).",
                        "DECISIONS": {
                            "XRP": {
                                "signal": "buy_to_enter",
                                "strategy": "trend_following",
                                "leverage": 10,
                                "confidence": 0.90,
                                "invalidation_condition": "Close if 15m structure breaks to LH_LL or Price < EMA20",
                            },
                            "SOL": {
                                "signal": "close_position",
                                "strategy": "risk_management",
                                "leverage": 10,
                                "confidence": 0.95,
                                "invalidation_condition": "Profit Erosion > 50% limit breached",
                            },
                            "TRX": {
                                "signal": "buy_to_enter",
                                "strategy": "counter_trend",
                                "leverage": 10,
                                "confidence": 0.75,
                                "invalidation_condition": "Close if price breaks below recent swing low",
                            },
                            "DOGE": {"signal": "hold"},
                            "ASTER": {"signal": "hold"},
                            "ETH": {"signal": "hold"},
                        },
                    },
                },
            ],
        }
        return json.dumps(system_structure)

    def get_ai_decision(self, prompt: str) -> str:
        """Get trading decision from AI API using structured JSON prompting"""
        # Check if using Gemini
        if getattr(self, "provider", None) == "Gemini" and getattr(self, "gemini_client", None):
            return self._get_gemini_decision(prompt)

        # Check if using Groq
        if getattr(self, "provider", None) == "Groq" and getattr(self, "groq_client", None):
            return self._get_groq_decision(prompt)

        if not self.client:
            return self._get_simulation_response(prompt)

        try:
            # User prompt is already a JSON string from main.py
            # We wrap it in a clear instruction
            user_message_content = f"Analyze the following market data JSON and provide decisions based on the system rules:\n\n{prompt}"

            print(
                f"[INFO] Sending request to {self.provider} API (JSON Mode)... Payload Size: {len(prompt)} chars",
            )

            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_message_content},
                ],
                "temperature": 0.5,
                "max_tokens": 4000,
                "response_format": {"type": "json_object"},
                "stream": True,
            }

            # Add thinking support for Z.AI
            if self.provider == "Z.AI" and self.thinking_enabled:
                request_params["temperature"] = 1.0  # Required for thinking mode
                request_params["extra_body"] = {"thinking": {"type": "enabled"}}
            
            # Add reasoning support for OpenRouter
            if self.provider == "OpenRouter" and self.thinking_enabled:
                request_params["extra_body"] = {"reasoning": {"enabled": True}}

            stream = self.client.chat.completions.create(**request_params)

            print("[WAIT] Receiving stream...", end="", flush=True)
            collected_content = []
            for chunk in stream:
                # Safe access - check if choices exists and has content
                if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content_chunk = delta.content
                        collected_content.append(content_chunk)
                        # Optional: Print a dot every 10 chunks to show activity
                        if len(collected_content) % 10 == 0:
                            print(".", end="", flush=True)

            print(" [OK]")
            content = "".join(collected_content)

            # Robust JSON extraction using JSONDecoder
            try:
                # Find the first '{'
                start_index = content.find("{")
                if start_index != -1:
                    # Slice from the first '{' to the end
                    json_candidate = content[start_index:]

                    # Use raw_decode to parse the JSON object and ignore trailing data
                    decoder = json.JSONDecoder()
                    obj, end_index = decoder.raw_decode(json_candidate)

                    # Re-serialize to ensure valid JSON string is returned
                    content = json.dumps(obj, indent=2)
                else:
                    print("[WARN]  No JSON object found in response")
                    # Return safe hold response when no valid JSON found
                    return self.get_safe_hold_decisions()

            except Exception as e:
                print(f"[WARN]  JSON extraction warning: {e}")

                # Try to repair common JSON issues
                try:
                    # Fix unterminated strings by finding the last complete JSON structure
                    repaired = content

                    # Count braces to find complete JSON
                    brace_count = 0
                    last_valid_pos = 0
                    in_string = False
                    escape_next = False

                    for i, char in enumerate(repaired):
                        if escape_next:
                            escape_next = False
                            continue
                        if char == "\\":
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        if not in_string:
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    last_valid_pos = i + 1

                    if last_valid_pos > 0:
                        repaired = repaired[:last_valid_pos]
                        obj = json.loads(repaired)
                        print(f"[OK]    JSON repaired successfully (truncated at pos {last_valid_pos})")
                        return json.dumps(obj, indent=2)
                except:
                    pass

                # Fallback: try stripping markdown if extraction failed
                if "```json" in content:
                    content = content.replace("```json", "").replace("```", "")
                elif "```" in content:
                    content = content.replace("```", "")

                # Final validation - if still not valid JSON, return safe hold
                try:
                    json.loads(content)
                except:
                    print("[ERR]   JSON parse failed completely, returning safe HOLD decisions")
                    return self.get_safe_hold_decisions()

            return content.strip()

        except Exception as e:
            # Detailed error logging
            error_type = type(e).__name__
            print(f"[ERR]   API error ({error_type}): {e}")
            return self._get_error_response(f"{error_type}: {e}")

    def _get_gemini_decision(self, prompt: str) -> str:
        """Get trading decision from Gemini API with thinking support"""
        try:
            # Build the full prompt with system context
            system_prompt = self._build_system_prompt()
            user_message = f"Analyze the following market data JSON and provide decisions based on the system rules:\n\n{prompt}"
            full_prompt = f"{system_prompt}\n\n{user_message}\n\nRespond with valid JSON only."

            print(f"[INFO] Sending request to Gemini API... Payload Size: {len(prompt)} chars")

            # Build content
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)],
                ),
            ]

            # Configure thinking level and structured output
            thinking_level = getattr(self, "thinking_level", "HIGH")
            generate_config = types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=types.ThinkingConfig(
                    thinking_level=thinking_level,
                ),
                response_mime_type="application/json",  # Structured JSON output
            )

            print("[WAIT] Receiving stream...", end="", flush=True)
            collected_content = []

            for chunk in self.gemini_client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_config,
            ):
                if chunk.text:
                    collected_content.append(chunk.text)
                    if len(collected_content) % 5 == 0:
                        print(".", end="", flush=True)

            print(" [OK]")
            content = "".join(collected_content)

            # Extract JSON from response
            try:
                start_index = content.find("{")
                if start_index != -1:
                    json_candidate = content[start_index:]
                    decoder = json.JSONDecoder()
                    obj, _ = decoder.raw_decode(json_candidate)
                    content = json.dumps(obj, indent=2)
                else:
                    print("[WARN]  No JSON object found in Gemini response")
                    return self.get_safe_hold_decisions()
            except Exception as e:
                print(f"[WARN]  Gemini JSON extraction warning: {e}")
                # Try to clean up
                if "```json" in content:
                    content = content.replace("```json", "").replace("```", "")
                elif "```" in content:
                    content = content.replace("```", "")

            return content.strip()

        except Exception as e:
            error_type = type(e).__name__
            print(f"[ERR]   Gemini API error ({error_type}): {e}")
            return self._get_error_response(f"{error_type}: {e}")

    def _get_groq_decision(self, prompt: str) -> str:
        """Get trading decision from Groq API with compound tool support"""
        try:
            user_message_content = f"Analyze the following market data JSON and provide decisions based on the system rules:\n\n{prompt}"
            
            print(
                f"[INFO] Sending request to Groq API... Payload Size: {len(prompt)} chars",
            )
            
            # Using specific parameters for groq/compound as requested
            stream = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_message_content},
                ],
                temperature=0.5, # Reduced from 1 to 0.5 for trading stability
                max_completion_tokens=4000,
                top_p=1,
                stream=True,
                stop=None,
                compound_custom={"tools":{"enabled_tools":["web_search","code_interpreter","visit_website"]}}
            )
            
            print("[WAIT] Receiving stream...", end="", flush=True)
            collected_content = []
            for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        collected_content.append(delta.content)
                        if len(collected_content) % 10 == 0:
                            print(".", end="", flush=True)
                            
            print(" [OK]")
            content = "".join(collected_content)
            
            # Extract JSON from response
            try:
                start_index = content.find("{")
                if start_index != -1:
                    json_candidate = content[start_index:]
                    decoder = json.JSONDecoder()
                    obj, _ = decoder.raw_decode(json_candidate)
                    content = json.dumps(obj, indent=2)
                else:
                    print("[WARN]  No JSON object found in Groq response")
                    return self.get_safe_hold_decisions()
            except Exception as e:
                print(f"[WARN]  Groq JSON extraction warning: {e}")
                if "```json" in content:
                    content = content.replace("```json", "").replace("```", "")
                elif "```" in content:
                    content = content.replace("```", "")

            # Strict final parse to guarantee we return JSON
            try:
                json.loads(content)
            except:
                print("[ERR]   JSON parse failed completely for Groq, returning safe HOLD decisions")
                return self.get_safe_hold_decisions()
            
            return content.strip()
            
        except Exception as e:
            error_type = type(e).__name__
            print(f"[ERR]   Groq API error ({error_type}): {e}")
            return self._get_error_response(f"{error_type}: {e}")

    def _get_simulation_response(self, prompt: str) -> str:
        """Simulation response without API - Returns valid JSON string"""
        print("[WARN]  Using simulation mode...")
        simulation_data = {
            "CHAIN_OF_THOUGHTS": f"Simulation Mode: Assuming market pullback. Shorting SOL based on simulated {HTF_LABEL} resistance. Aiming for 1:1.5 R/R using simulated ATR. Holding others.",
            "DECISIONS": {
                "SOL": {
                    "signal": "sell_to_enter",
                    "leverage": 10,
                    "confidence": 0.65,
                    "invalidation_condition": "If price closes above 199.0",
                },
                "XRP": {"signal": "hold"},
                "TRX": {"signal": "hold"},
                "DOGE": {"signal": "hold"},
                "ASTER": {"signal": "hold"},
                "ETH": {"signal": "hold"},
            },
        }
        return json.dumps(simulation_data, indent=2)

    def get_cached_decisions(self) -> str:
        """Get cached decisions from recent successful cycles - Returns valid JSON string"""
        try:
            cached_cycles = safe_file_read("data/cycle_history.json", default_data=[])
            if not cached_cycles:
                return self.get_safe_hold_decisions()

            for cycle in reversed(cached_cycles[-5:]):  # Last 5 cycles
                decisions = cycle.get("decisions", {})
                if decisions and isinstance(decisions, dict):
                    valid_signals = [
                        d
                        for d in decisions.values()
                        if isinstance(d, dict)
                        and d.get("signal") in ["buy_to_enter", "sell_to_enter"]
                    ]
                    if valid_signals:
                        print("[INFO] Using cached decisions from recent successful cycle")
                        fallback_response = {
                            "CHAIN_OF_THOUGHTS": "API Error - Using cached decisions from recent successful cycle. Continuing with established strategy.",
                            "DECISIONS": decisions,
                        }
                        return json.dumps(fallback_response, indent=2)

            return self.get_safe_hold_decisions()

        except Exception as e:
            print(f"[WARN]  Cache retrieval error: {e}")
            return self.get_safe_hold_decisions()

    def get_safe_hold_decisions(self) -> str:
        """Generate safe hold decisions for all coins - Returns valid JSON string"""
        print("[INFO] Generating safe hold decisions")
        hold_decisions = {}
        for coin in ["XRP", "DOGE", "ASTER", "TRX", "ETH", "SOL"]:
            hold_decisions[coin] = {
                "signal": "hold",
                "justification": "Safe mode: Holding due to API error",
            }

        safe_response = {
            "CHAIN_OF_THOUGHTS": "API Error - Operating in safe mode. Holding all positions/cash to preserve capital.",
            "DECISIONS": hold_decisions,
        }
        return json.dumps(safe_response, indent=2)

    def _get_error_response(self, error_message: str) -> str:
        """Enhanced error response with intelligent recovery"""
        print(f"[ERR]   Enhanced error handling for: {error_message}")

        error_type = (
            type(error_message).__name__
            if isinstance(error_message, Exception)
            else str(error_message)
        )

        if "Connection" in error_type or "Timeout" in error_type:
            return self.get_cached_decisions()

        return self.get_safe_hold_decisions()
