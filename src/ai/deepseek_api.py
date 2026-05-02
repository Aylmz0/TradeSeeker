import json
import re

import litellm
from litellm import Router

from config.config import Config
from src.utils import safe_file_read

# HTF_INTERVAL used in prompt, we can get it from Config
HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"
HTF_LABEL = HTF_INTERVAL

import os

# Suppress messy internal LiteLLM terminal logs
os.environ["LITELLM_LOG"] = "ERROR"
litellm.suppress_debug_info = True


class DeepSeekAPI:
    """AI API integration natively powered by LiteLLM Router"""

    def __init__(self, api_key: str | None = None):
        model_list = []
        self.primary_model = None

        self.thinking_enabled = getattr(Config, "OPENROUTER_REASONING_ENABLED", False)

        # 1. OpenRouter (Primary)
        if getattr(Config, "OPENROUTER_API_KEY", None):
            self.primary_model = "openrouter/" + getattr(
                Config, "OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"
            )

            or_params = {
                "model": self.primary_model,
                "api_key": Config.OPENROUTER_API_KEY,
                "api_base": "https://openrouter.ai/api/v1",
                "tpm": 100000,
                "rpm": 100,
            }
            if self.thinking_enabled:
                or_params["extra_body"] = {"reasoning": {"enabled": True}}
            else:
                or_params["response_format"] = {"type": "json_object"}

            model_list.append(
                {
                    "model_name": self.primary_model,
                    "litellm_params": or_params,
                }
            )

            # 1b. OpenRouter Fallback (Secondary OpenRouter model)
            or_fallback_model = "openrouter/" + getattr(
                Config, "OPENROUTER_FALLBACK_MODEL", "google/gemini-2.0-flash-exp:free"
            )
            or_fb_params = {
                "model": or_fallback_model,
                "api_key": Config.OPENROUTER_API_KEY,
                "api_base": "https://openrouter.ai/api/v1",
                "tpm": 100000,
                "rpm": 100,
            }
            if self.thinking_enabled:
                or_fb_params["extra_body"] = {"reasoning": {"enabled": True}}
            else:
                or_fb_params["response_format"] = {"type": "json_object"}

            model_list.append(
                {
                    "model_name": or_fallback_model,
                    "litellm_params": or_fb_params,
                }
            )

        # 2. Groq (Fast Fallback)
        if getattr(Config, "GROQ_API_KEY", None):
            groq_model = "groq/" + getattr(Config, "GROQ_MODEL", "llama-3.3-70b-versatile")
            if not self.primary_model:
                self.primary_model = groq_model
            model_list.append(
                {
                    "model_name": groq_model,
                    "litellm_params": {
                        "model": groq_model,
                        "api_key": Config.GROQ_API_KEY,
                        "tpm": 100000,
                        "rpm": 30,
                        "response_format": {"type": "json_object"},
                    },
                }
            )

        # 3. MiMo (Secondary Fallback)
        if getattr(Config, "MIMO_API_KEY", None):
            mimo_model = "openai/" + getattr(Config, "MIMO_MODEL", "mimo-v2-flash")
            if not self.primary_model:
                self.primary_model = mimo_model
            model_list.append(
                {
                    "model_name": mimo_model,
                    "litellm_params": {
                        "model": mimo_model,
                        "api_key": Config.MIMO_API_KEY,
                        "api_base": "https://api.xiaomimimo.com/v1",
                        "response_format": {"type": "json_object"},
                    },
                }
            )

        # 4. Z.AI
        if getattr(Config, "ZAI_API_KEY", None):
            zai_model = "openai/" + getattr(Config, "ZAI_MODEL", "glm-4.5-flash")
            if not self.primary_model:
                self.primary_model = zai_model
            model_list.append(
                {
                    "model_name": zai_model,
                    "litellm_params": {
                        "model": zai_model,
                        "api_key": Config.ZAI_API_KEY,
                        "api_base": "https://api.z.ai/api/paas/v4/",
                        "response_format": {"type": "json_object"},
                    },
                }
            )

        if not model_list:
            print(
                "[ERR]   No API keys found in .env! Please set OPENROUTER_API_KEY or GROQ_API_KEY."
            )
        else:
            print(f"[INFO] Initializing LiteLLM Router with {len(model_list)} fallback models.")
            self.router = Router(
                model_list=model_list,
                routing_strategy="latency-based-routing",
                allowed_fails=1,
                num_retries=0,  # FORCE 0 RETRIES: Instantly drop to Groq/MiMo on error
                timeout=120.0,  # 2.0 minutes timeout to allow deep reasoning models to finish
            )
            self.invocation_count = 0

    def _build_system_prompt(self) -> str:
        """Constructs the system prompt as a structured JSON object.
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
                    "discipline": "AGGRESSIVE SCALPER: Be AGGRESSIVE in seeking opportunities, but DISCIPLINED in execution. Your mission is to capture frequent, incremental profits while strictly adhering to volume and structure filters. Do NOT sacrifice quality for frequency. If indicators and ML align, EXECUTE decisively. Logic precedes decision.",
                    "commission_guard": f"Only enter if expected profit vs commission ratio exceeds {getattr(Config, 'COMMISSION_GUARD_RATIO', 5.0)}. Aggressive but Disciplined.",
                },
                "data_dictionary": {
                    "price_slope": {
                        "AGGRESSIVE_ASCEND/DESCEND": "High-velocity move. Strong trend conviction.",
                        "MODERATE_ASCEND/DESCEND": "Steady trend. Standard entry conditions apply.",
                        "FLAT": "No clear direction. High risk of choppy/sideways behavior.",
                    },
                    "ema_stretch": {
                        "TIGHT": "Price is hugging the EMA-20. Safe for entry/continuation.",
                        "OVEREXTENDED_UP/DOWN": "Price is significantly far from EMA-20 (Stretch > 1.5%). RISK of sharp mean-reversion. Avoid new entries in stretch direction.",
                    },
                    "rsi_divergence": {
                        "BULLISH_DIVERGENCE": "Price making lower lows but RSI making higher lows. Strong reversal sign for LONG.",
                        "BEARISH_DIVERGENCE": "Price making higher highs but RSI making lower highs. Strong reversal sign for SHORT.",
                    },
                    "volatility_pulse": {
                        "STRETCHING": "Volatility is expanding (short-term > long-term). Expect breakout.",
                        "STAGNANT": "Volatility is drying up. Expect range-bound or squeeze.",
                    },
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
                        "condition": "Evaluate 'counter_trade_risk' in each coin's risk_profile using 12 STRONG conditions: Funding Rate, Volume Surge, RSI Extremes, MACD Divergence, Zone/Exhaustion, VWAP Alignment, Bollinger Position, OBV Divergence, ML Consensus, 15m Structure, 15m Momentum, and 1h Range/Volatility.",
                        "risk_level_rules": {
                            "CT_LOW_RISK": "CT_ALIGNMENT_STRONG+5 OR CT_ALIGNMENT_MEDIUM+7 conditions. (High structural confluence). EXECUTE.",
                            "CT_MEDIUM_RISK": "CT_ALIGNMENT_STRONG+4 OR CT_ALIGNMENT_MEDIUM+6 OR NONE+8 conditions. EXECUTE if momentum confirms.",
                            "CT_HIGH_RISK": "Counter-trend setup is too weak. Do NOT trade. (Wait for better alignment).",
                            "CT_VERY_HIGH_RISK": "No alignment/conditions. Do NOT trade counter-trend. (Trend-Following unaffected).",
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
                    "divergence_execution_rule": {
                        "BEARISH_DIVERGENCE": "If detected at UPPER_10 or with WEAKENING momentum, Treat as a HIGH-PRIORITY exit for LONGs and a STRONG entry signal for SHORTs.",
                        "BULLISH_DIVERGENCE": "If detected at LOWER_10 or with WEAKENING momentum, Treat as a HIGH-PRIORITY exit for SHORTs and a STRONG entry signal for LONGs.",
                    },
                    "ema_stretch_safety_rule": {
                        "OVEREXTENDED_UP": "Strictly block new LONG entries. Favour trailing stops or profit taking.",
                        "OVEREXTENDED_DOWN": "Strictly block new SHORT entries. Favour trailing stops or profit taking.",
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
                        "threshold": "volume_ratio < 0.30 = LOW",
                        "for_entry": "POOR volume is acceptable ONLY with strong structural confirmation (divergence + 15m structure, or counter-trend exhaustion).",
                        "for_exit": "LOW volume ≠ exit signal.",
                        "interpretation": "3m is your SENSOR. 15m is your ADVISOR. 1h is your BOSS. Do not let 3m noise block a structural 1h/15m setup.",
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
                        "score_weights": "HTF(+3), 15m_structure(+5), ML_Consensus(+2), 15m_momentum(+1), RSI(+1), MACD(+1)",
                        "action": "Consider closing based on strength level and PnL. EXIT MANDATORY on CRITICAL.",
                        "discipline_note": "15m Structure is your Master Gate. Avoid closing positions solely on 3m noise or momentum dips if the 15m structure (HH_HL/LH_LL) hasn't reversed.",
                    },
                    "reversal_strength_definitions": {
                        "NONE": "No reversal signals (score 0). Continue normally.",
                        "WEAK": "INFORMATIONAL ONLY. Do NOT exit under any circumstances.",
                        "STRONG": "Consider exit ONLY if: (1) 15m structure reversed (LH_HL ↔ HH_HL), OR (2) erosion > 50%, OR (3) ML reverse signal (>40% opposite). Otherwise HOLD.",
                        "CRITICAL": "Execution mandatory - close immediately. Structural invalidation confirmed.",
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
                            "SIGNIFICANT": "Over 50% of peak profit eroded. Close if reversal_strength is STRONG or CRITICAL.",
                            "CRITICAL": "Peak profit fully eroded or now losing. Close immediately.",
                        },
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
                "1. Check EACH coin's market_context.regime (1h Boss) and risk_profile.",
                "2. Review technical_summary (15m Advisor): momentum, structure_15m.",
                "3. Use 3m (Sensor) ONLY for entry timing via volume surges or RSI extremes. Do NOT block 1h/15m setups if 3m is neutral/noisy. RULE: If 1h Trend + 15m Structure are in FULL alignment, 3m alignment is OPTIONAL.",
                "4. CROSS-REFERENCE with ml_consensus (XGBoost) if available. If null, rely STRICTLY on technical_summary (15m) and key_levels.",
                "5. Evaluate risk_profile: counter_trade_risk + reversal_threat.",
                "6. Check sentiment: funding_rate, open_interest.",
                "7. Decide direction based on strongest quantified edge (1h Direction + 15m Structure).",
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
                    "usage": "Use ML as you would any other indicator (RSI, MACD, Volume). Consider it alongside your technical analysis. When ML and technicals agree, confidence increases. When they disagree, weigh the evidence. Interpretation: HOLD < 40% = Model has directional conviction (BUY+SELL > 60%); check which side dominates. HOLD > 50% = Model is uncertain; proceed with caution. BUY/SELL > 40% = Model favors that direction.",
                },
                "market_context": "regime (TF_STRONG_BULLISH/TF_STABLE_BULLISH/TF_WEAK_BULLISH/TF_STRONG_BEARISH/TF_STABLE_BEARISH/TF_WEAK_BEARISH/TF_NEUTRAL/CHOPPY), efficiency_ratio, volatility_state (SQUEEZE/EXPANDING/NORMAL), price_location (UPPER_10/LOWER_10/MIDDLE). NOTE: CHOPPY (low efficiency) is a high-risk/low-conviction environment—prefer HOLDing or extreme caution.",
                "technical_summary": "momentum (STRENGTHENING/STABLE/WEAKENING), volume_ratio (numeric), volume_support (EXCELLENT/GOOD/FAIR/POOR/LOW), structure_15m (HH_HL/LH_LL/RANGE/UNCLEAR)",
                "key_levels": "price, ema20_htf, rsi_15m, atr_htf — raw numerical anchors for your independent reasoning and cross-validation of labels.",
                "risk_profile": "counter_trade_risk (CT_LOW_RISK/CT_MEDIUM_RISK/CT_HIGH_RISK/CT_VERY_HIGH_RISK), alignment_strength (CT_ALIGNMENT_STRONG/CT_ALIGNMENT_MEDIUM/CT_ALIGNMENT_NONE), reversal_threat (RT_NONE/RT_WEAK/RT_MODERATE/RT_STRONG/RT_CRITICAL)",
                "sentiment": "funding_rate, open_interest",
                "position": "Current position details if exists, including exit_plan and erosion tracking.",
            },
            "response_schema": {
                "format": "JSON",
                "required_keys": ["CHAIN_OF_THOUGHTS", "DECISIONS"],
                "CHAIN_OF_THOUGHTS": f"String. 'High-Density Hybrid Analysis' structure. Analyze {HTF_LABEL}, 15m, and 3m for EACH coin individually. Include Technicals + ML Consensus + Logic for each. Format like: 'COIN: 1h... 15m... Logic... Decision...'",
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
                        "CHAIN_OF_THOUGHTS": "Systematic execution: 1h Trend + 15m Momentum/Structure + 3m Timing + ML Consensus. Logic precedes decision.\\n\\nXRP: Regime TF_STRONG_BULLISH (1h Bullish, Price $0.54 > EMA20 $0.52, RSI 62, ADX 32 MODERATE). 15m Bullish (Structure HH_HL, Momentum STRENGTHENING, BB_Width 5.2% expanding). 3m Bullish (RSI 64, Volume 1.8x). ML Consensus: BUY 38%. Logic: Full technical alignment across all timeframes with expanding volume. ML supports direction. Decision: BUY_TO_ENTER.\\n\\nSOL (OPEN LONG): Regime TF_WEAK_BULLISH (1h Bullish broken). 15m Bearish (Structure LH_LL, Price < EMA20). Position Risk: Erosion SIGNIFICANT (55% profit decay). Reversal Score: RT_STRONG (7/10). ML Consensus: SELL 42%. Logic: Technical invalidation confirmed by 15m structure reversal. Profit erosion above 50%. Decision: CLOSE_POSITION.\\n\\nTRX: Regime TF_STABLE_BULLISH (1h Bullish). 15m Bearish (Extreme dip, RSI 22 OVERSOLD, Price at BB Lower Band). 3m Reversing (MACD Bullish Cross). ML Consensus: BUY 35%. Counter-trade risk: CT_MEDIUM_RISK (oversold RSI + BB lower band + 3m reversal). Logic: Buying the dip in 1h bull regime with structural support. Decision: BUY_TO_ENTER (Counter-Trend Dip).\\n\\nDOGE: Regime TF_STABLE_BULLISH but Price Location UPPER_10 (Resistance). 15m Momentum WEAKENING. ML Consensus: HOLD 38%, BUY+SELL = 62% (directional conviction split). Logic: Buying at resistance with weakening momentum. Wait for pullback or breakout confirmation. Decision: HOLD.\\n\\nASTER: Volume Ratio 0.12x (< 0.30x Hard Block). BB Squeeze (Volatility 1.2%). Decision: HOLD (Volume blocks entry).",
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
        """Get trading decision from AI API natively via LiteLLM Router"""
        if not hasattr(self, "router"):
            return self._get_simulation_response(prompt)

        try:
            self.invocation_count += 1
            user_message_content = f"Analyze the following market data JSON and provide decisions based on the system rules:\n\n{prompt}"

            print(f"[INFO] Sending request to LiteLLM Router... Payload Size: {len(prompt)} chars")

            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_message_content},
            ]

            kwargs = {
                "model": self.primary_model,
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 20000,
                "stream": False,  # Non-streaming provides greater stability against free tier drops
                "timeout": 240,  # 4 minutes: Give reasoning models plenty of time to think
            }

            # Fallbacks are automatically collected from the remaining configured models
            fallbacks = [
                m["model_name"]
                for m in self.router.model_list
                if m["model_name"] != self.primary_model
            ]

            # Execute request through Router
            print(f"[WAIT] Waiting for response from LLM...", end="", flush=True)
            response = self.router.completion(**kwargs, fallbacks=fallbacks)

            print(" [OK]")

            content = response.choices[0].message.content

            # LiteLLM makes token tracking easy
            usage = getattr(response, "usage", None)
            model_used = getattr(response, "model", "unknown")

            # Calculate individual component sizes for transparency
            sys_len = len(messages[0]["content"])
            user_len = len(messages[1]["content"])
            total_chars = sys_len + user_len

            print(" [OK]")
            print("-" * 70)
            print(f"[AI TOKEN BILL - INV #{self.invocation_count}]")
            print("-" * 70)
            if usage:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens

                # Estimated split based on character ratios (since API gives total prompt)
                sys_ratio = sys_len / total_chars if total_chars > 0 else 0.5
                sys_tokens_est = int(prompt_tokens * sys_ratio)
                user_tokens_est = prompt_tokens - sys_tokens_est

                print(f"PROMPT TOKENS: {prompt_tokens:,} (~{total_chars:,} chars)")
                print(f"  - System (Rules): ~{sys_tokens_est:,}")
                print(f"  - Market (Data):  ~{user_tokens_est:,}")
                print(f"COMPLETION TOKENS: {completion_tokens:,} (AI Response)")
                print(f"TOTAL TOKENS: {total_tokens:,}")
            else:
                print(f"[WARN] Usage data missing from API response.")
                print(f"Estimated Total: ~{total_chars // 4:,} Tokens (Based on char count)")

            print(f"MODEL USED: {model_used}")
            print("-" * 70)

            # Capture hidden reasoning if available
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)

            return {"content": content, "reasoning": reasoning}

        except litellm.ContextWindowExceededError as e:
            print(f"[ERR]   Context window exceeded: {e}")
            return self.get_cached_decisions()
        except litellm.RateLimitError as e:
            print(f"[ERR]   Rate Limit hit across all fallbacks: {e}")
            return self.get_cached_decisions()
        except litellm.Timeout as e:
            print(f"[ERR]   Timeout across all fallbacks: {e}")
            return self.get_cached_decisions()
        except Exception as e:
            error_type = type(e).__name__
            print(f"[ERR]   Router API error ({error_type}): {e}")
            return self._get_error_response(f"{error_type}: {e}")

    def _extract_json_from_content(self, content: str) -> str:
        """Robustly extracts and parses JSON from the LLM content block"""
        if not content:
            print("[WARN]  No content returned from API")
            return self.get_safe_hold_decisions()

        try:
            # 1. First Pass: Try raw
            start_index = content.find("{")
            if start_index != -1:
                json_candidate = content[start_index:]
                try:
                    decoder = json.JSONDecoder()
                    obj, _ = decoder.raw_decode(json_candidate)
                    return json.dumps(obj, indent=2)
                except Exception as e:
                    print(f"[WARN]  Strict JSON parse failed ({e}), attempting regex repair...")

                    # 2. Aggressive Repair
                    cleaned = re.sub(r"```json\s*", "", json_candidate)
                    cleaned = re.sub(r"```\s*", "", cleaned)
                    # Fix missing commas
                    cleaned = re.sub(r'(?<=[}\]])\s*(?=[{"\w])', ",", cleaned)
                    # Fix trailing commas
                    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

                    try:
                        obj = json.loads(cleaned)
                        print("[OK]    JSON repaired successfully via regex filtering.")
                        return json.dumps(obj, indent=2)
                    except Exception as inner_e:
                        print(f"[WARN]  Regex repair failed: {inner_e}")

                        # Extract DECISIONS block if everything else is corrupted
                        match = re.search(r'"DECISIONS"\s*:\s*({[^}]+(}[^{}]*)*})', cleaned)
                        if match:
                            try:
                                decisions_str = "{" + match.group(0) + "}"
                                obj = json.loads(decisions_str)
                                print(
                                    "[OK]    JSON repaired successfully by extracting DECISIONS block only."
                                )
                                return json.dumps(obj, indent=2)
                            except Exception:
                                pass
        except Exception as e:
            print(f"[WARN]  JSON extraction warning: {e}")

        print("[ERR]   JSON parse failed completely, returning safe HOLD decisions")
        return self.get_safe_hold_decisions()

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
