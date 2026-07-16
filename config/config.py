"""Configuration management for the Alpha Arena DeepSeek bot.
Provides secure API key handling and application configuration.
"""

import logging
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from src.core import constants
from src.schemas.config import Settings

# Load environment variables
load_dotenv()

# Create validated settings instance
try:
    _settings = Settings()
except ValidationError as e:
    logging.error("Configuration validation failed:")
    for error in e.errors():
        loc = " → ".join(str(x) for x in error["loc"])
        logging.error(f"  {loc}: {error['msg']} (got: {error['input']!r})")
    raise SystemExit(1) from e


class Config:
    """Application configuration with secure API key handling.

    All values are loaded through pydantic-settings with type validation.
    """

    # AI Provider Selection
    PRIMARY_AI_PROVIDER: str = _settings.PRIMARY_AI_PROVIDER

    # Groq Settings
    GROQ_API_KEY: str | None = _settings.GROQ_API_KEY
    GROQ_MODEL: str = _settings.GROQ_MODEL

    # OpenRouter & DeepSeek Keys
    DEEPSEEK_API_KEY: str | None = _settings.DEEPSEEK_API_KEY
    MIMO_API_KEY: str | None = _settings.MIMO_API_KEY
    MIMO_MODEL: str = _settings.MIMO_MODEL
    MIMO_THINKING_ENABLED: bool = _settings.MIMO_THINKING_ENABLED
    ZAI_API_KEY: str | None = _settings.ZAI_API_KEY
    ZAI_MODEL: str = _settings.ZAI_MODEL
    ZAI_THINKING_ENABLED: bool = _settings.ZAI_THINKING_ENABLED
    OPENROUTER_API_KEY: str | None = _settings.OPENROUTER_API_KEY
    OPENROUTER_MODEL: str = _settings.OPENROUTER_MODEL
    OPENROUTER_FALLBACK_MODEL: str = _settings.OPENROUTER_FALLBACK_MODEL
    OPENROUTER_REASONING_ENABLED: bool = _settings.OPENROUTER_REASONING_ENABLED
    BINANCE_API_KEY: str | None = _settings.BINANCE_API_KEY
    BINANCE_SECRET_KEY: str | None = _settings.BINANCE_SECRET_KEY

    # Application Settings
    DEBUG: bool = _settings.DEBUG
    LOG_LEVEL: str = _settings.LOG_LEVEL
    LOG_DIR: str = _settings.LOG_DIR
    MAX_RETRY_ATTEMPTS: int = _settings.MAX_RETRY_ATTEMPTS
    REQUEST_TIMEOUT: int = _settings.REQUEST_TIMEOUT

    # Trading Settings
    INITIAL_BALANCE: float = _settings.INITIAL_BALANCE
    CYCLE_INTERVAL_MINUTES: int = _settings.CYCLE_INTERVAL_MINUTES
    HISTORY_RESET_INTERVAL: int = _settings.HISTORY_RESET_INTERVAL
    MAX_CYCLES: int = _settings.MAX_CYCLES

    # Risk Management
    MAX_LEVERAGE: int = _settings.MAX_LEVERAGE
    MIN_CONFIDENCE: float = _settings.MIN_CONFIDENCE
    CHOPPY_ER_THRESHOLD: float = _settings.CHOPPY_ER_THRESHOLD
    ML_CONFIDENCE_THRESHOLD: float = _settings.ML_CONFIDENCE_THRESHOLD
    MAX_POSITIONS: int = _settings.MAX_POSITIONS
    RISK_PER_TRADE_USD: float = _settings.RISK_PER_TRADE_USD

    # Graduated Stop Loss Limits
    LOSS_MULT_L1: float = _settings.LOSS_MULT_L1
    LOSS_MULT_L2: float = _settings.LOSS_MULT_L2
    LOSS_MULT_L3: float = _settings.LOSS_MULT_L3
    LOSS_MULT_L4: float = _settings.LOSS_MULT_L4
    LOSS_MULT_BASE: float = _settings.LOSS_MULT_BASE

    # Smart Cooldown Settings
    SMART_COOLDOWN_LOSS: int = _settings.SMART_COOLDOWN_LOSS
    SMART_COOLDOWN_WIN: int = _settings.SMART_COOLDOWN_WIN
    MAX_NEW_POSITIONS_PER_CYCLE: int = _settings.MAX_NEW_POSITIONS_PER_CYCLE
    EXTENDED_LOSS_CYCLES: int = _settings.EXTENDED_LOSS_CYCLES
    EXTENDED_PROFIT_CYCLES: int = _settings.EXTENDED_PROFIT_CYCLES

    # Risk Level Configuration
    RISK_LEVEL: str = _settings.RISK_LEVEL
    TRADING_MODE: str = _settings.TRADING_MODE
    BINANCE_TESTNET: bool = _settings.BINANCE_TESTNET
    BINANCE_MARGIN_TYPE: str = _settings.BINANCE_MARGIN_TYPE
    BINANCE_DEFAULT_LEVERAGE: int = _settings.BINANCE_DEFAULT_LEVERAGE
    BINANCE_RECV_WINDOW: int = _settings.BINANCE_RECV_WINDOW

    # Position Limits
    SAME_DIRECTION_LIMIT: int = _settings.SAME_DIRECTION_LIMIT
    DYNAMIC_DIRECTION_LIMIT: int = _settings.DYNAMIC_DIRECTION_LIMIT

    # Market Analysis Thresholds
    GLOBAL_NEUTRAL_STRENGTH_THRESHOLD: float = _settings.GLOBAL_NEUTRAL_STRENGTH_THRESHOLD
    RSI_OVERBOUGHT_THRESHOLD: float = _settings.RSI_OVERBOUGHT_THRESHOLD
    RSI_OVERSOLD_THRESHOLD: float = _settings.RSI_OVERSOLD_THRESHOLD
    EMA_NEUTRAL_BAND_PCT: float = _settings.EMA_NEUTRAL_BAND_PCT
    INTRADAY_NEUTRAL_RSI_LOW: float = _settings.INTRADAY_NEUTRAL_RSI_LOW
    INTRADAY_NEUTRAL_RSI_HIGH: float = _settings.INTRADAY_NEUTRAL_RSI_HIGH
    TREND_LONG_RSI_THRESHOLD: float = _settings.TREND_LONG_RSI_THRESHOLD
    TREND_SHORT_RSI_THRESHOLD: float = _settings.TREND_SHORT_RSI_THRESHOLD

    # Enhanced Trading Settings
    SHORT_ENHANCEMENT_MULTIPLIER: float = _settings.SHORT_ENHANCEMENT_MULTIPLIER
    VOLUME_MINIMUM_THRESHOLD: float = _settings.VOLUME_MINIMUM_THRESHOLD
    SIMULATION_COMMISSION_RATE: float = _settings.SIMULATION_COMMISSION_RATE
    VOLUME_QUALITY_THRESHOLDS: dict = {
        "excellent": 2.5,
        "good": 1.8,
        "fair": 1.2,
        "poor": 0.7,
    }

    DIRECTIONAL_BULLISH_LONG_MULTIPLIER: float = _settings.DIRECTIONAL_BULLISH_LONG_MULTIPLIER
    DIRECTIONAL_BULLISH_SHORT_MULTIPLIER: float = _settings.DIRECTIONAL_BULLISH_SHORT_MULTIPLIER
    DIRECTIONAL_BEARISH_LONG_MULTIPLIER: float = _settings.DIRECTIONAL_BEARISH_LONG_MULTIPLIER
    DIRECTIONAL_BEARISH_SHORT_MULTIPLIER: float = _settings.DIRECTIONAL_BEARISH_SHORT_MULTIPLIER
    DIRECTIONAL_NEUTRAL_MULTIPLIER: float = _settings.DIRECTIONAL_NEUTRAL_MULTIPLIER
    MARKET_REGIME_MULTIPLIERS: dict = {
        "BULLISH": 1.0,
        "BEARISH": 1.0,
        "NEUTRAL": 0.9,
        "CHOPPY": 0.8,
    }

    # Choppy Regime Detection Settings
    CHOPPY_COIN_RATIO_MIN: float = _settings.CHOPPY_COIN_RATIO_MIN
    CHOPPY_LEVERAGE: int = _settings.CHOPPY_LEVERAGE
    CHOPPY_TP_LONG_MULTIPLIER: float = _settings.CHOPPY_TP_LONG_MULTIPLIER
    CHOPPY_TP_SHORT_MULTIPLIER: float = _settings.CHOPPY_TP_SHORT_MULTIPLIER
    CHOPPY_SL_LONG_MULTIPLIER: float = _settings.CHOPPY_SL_LONG_MULTIPLIER
    CHOPPY_SL_SHORT_MULTIPLIER: float = _settings.CHOPPY_SL_SHORT_MULTIPLIER
    CHOPPY_TP_SL_MULTIPLIER: float = _settings.CHOPPY_TP_SL_MULTIPLIER
    CHOPPY_HIGH_ER_EXCEPTION: float = _settings.CHOPPY_HIGH_ER_EXCEPTION

    ATR_TP_MULTIPLIER: float = _settings.ATR_TP_MULTIPLIER
    ATR_SL_MULTIPLIER: float = _settings.ATR_SL_MULTIPLIER

    # Flash Exit Settings
    FLASH_EXIT_ENABLED: bool = _settings.FLASH_EXIT_ENABLED
    FLASH_EXIT_RSI_DELTA_MIN: float = _settings.FLASH_EXIT_RSI_DELTA_MIN
    FLASH_EXIT_RSI_SPIKE_THRESHOLD: float = _settings.FLASH_EXIT_RSI_DELTA_MIN
    FLASH_EXIT_VOLUME_SURGE_MIN: float = _settings.FLASH_EXIT_VOLUME_SURGE_MIN
    FLASH_EXIT_VOLUME_SURGE_THRESHOLD: float = _settings.FLASH_EXIT_VOLUME_SURGE_MIN
    FLASH_EXIT_LOSS_TRIGGER_MULTIPLIER: float = _settings.FLASH_EXIT_LOSS_TRIGGER_MULTIPLIER

    # Dynamic Confidence-Based Position Sizing
    CONFIDENCE_BASED_RISK_PERCENTAGES: dict = {
        "low": (0.10, 0.15),
        "medium": (0.15, 0.20),
        "high": (0.20, 0.25),
    }

    # Minimum Position Size Configuration
    MIN_POSITION_MARGIN_USD: float = _settings.MIN_POSITION_MARGIN_USD
    MIN_POSITION_CLEANUP_THRESHOLD: float = _settings.MIN_POSITION_CLEANUP_THRESHOLD

    # Partial Profit Taking Configuration
    MIN_PARTIAL_PROFIT_MARGIN_REMAINING_USD: float = (
        _settings.MIN_PARTIAL_PROFIT_MARGIN_REMAINING_USD
    )
    MAXIMUM_LIMIT_BALANCE_PCT: float = _settings.MAXIMUM_LIMIT_BALANCE_PCT

    # Exit Plan Defaults
    DEFAULT_STOP_LOSS_PCT: float = _settings.DEFAULT_STOP_LOSS_PCT
    DEFAULT_PROFIT_TARGET_PCT: float = _settings.DEFAULT_PROFIT_TARGET_PCT
    MIN_EXIT_PLAN_OFFSET: float = _settings.MIN_EXIT_PLAN_OFFSET

    # Trailing Stop Configuration
    TRAILING_PROGRESS_TRIGGER: float = _settings.TRAILING_PROGRESS_TRIGGER
    TRAILING_TIME_PROGRESS_FLOOR: float = _settings.TRAILING_TIME_PROGRESS_FLOOR
    TRAILING_TIME_MINUTES: float = _settings.TRAILING_TIME_MINUTES
    TRAILING_ATR_MULTIPLIER: float = _settings.TRAILING_ATR_MULTIPLIER
    TRAILING_FALLBACK_BUFFER_PCT: float = _settings.TRAILING_FALLBACK_BUFFER_PCT
    TRAILING_VOLUME_ABSOLUTE_THRESHOLD: float = _settings.TRAILING_VOLUME_ABSOLUTE_THRESHOLD
    TRAILING_VOLUME_DROP_RATIO: float = _settings.TRAILING_VOLUME_DROP_RATIO
    TRAILING_MIN_IMPROVEMENT_PCT: float = _settings.TRAILING_MIN_IMPROVEMENT_PCT
    TRAILING_PROGRESS_TRIGGER_EXTREME: float = _settings.TRAILING_PROGRESS_TRIGGER_EXTREME

    # Win Streak Cooldown Settings
    WIN_STREAK_COOLDOWN_THRESHOLD: int = _settings.WIN_STREAK_COOLDOWN_THRESHOLD
    WIN_STREAK_COOLDOWN_CYCLES: int = _settings.WIN_STREAK_COOLDOWN_CYCLES

    # Higher Timeframe Configuration
    HTF_INTERVAL: str = _settings.HTF_INTERVAL

    # JSON Prompt Feature Flags
    USE_JSON_PROMPT: bool = _settings.USE_JSON_PROMPT
    JSON_PROMPT_COMPACT: bool = _settings.JSON_PROMPT_COMPACT
    VALIDATE_JSON_PROMPTS: bool = _settings.VALIDATE_JSON_PROMPTS
    JSON_PROMPT_VERSION: str = _settings.JSON_PROMPT_VERSION
    JSON_SERIES_MAX_LENGTH: int = _settings.JSON_SERIES_MAX_LENGTH

    # Smart Indicator Cache Configuration
    USE_SMART_CACHE: bool = _settings.USE_SMART_CACHE
    SMART_CACHE_SAFETY_MARGIN: float = _settings.SMART_CACHE_SAFETY_MARGIN
    SMART_CACHE_STATS_LOGGING: bool = _settings.SMART_CACHE_STATS_LOGGING

    # Performance Monitor Thresholds
    PERFORMANCE_PROFITABILITY_HIGH: float = _settings.PERFORMANCE_PROFITABILITY_HIGH
    PERFORMANCE_PROFITABILITY_LOW: float = _settings.PERFORMANCE_PROFITABILITY_LOW
    PERFORMANCE_PROFIT_FACTOR_LOW: float = _settings.PERFORMANCE_PROFIT_FACTOR_LOW
    PERFORMANCE_PROFIT_FACTOR_HIGH: float = _settings.PERFORMANCE_PROFIT_FACTOR_HIGH
    PERFORMANCE_DECISION_RATE_HIGH: float = _settings.PERFORMANCE_DECISION_RATE_HIGH
    PERFORMANCE_DECISION_RATE_LOW: float = _settings.PERFORMANCE_DECISION_RATE_LOW
    PERFORMANCE_RETURN_HIGH: float = _settings.PERFORMANCE_RETURN_HIGH
    PERFORMANCE_RETURN_LOW: float = _settings.PERFORMANCE_RETURN_LOW
    PERFORMANCE_DRAWDOWN_THRESHOLD: float = _settings.PERFORMANCE_DRAWDOWN_THRESHOLD
    PERFORMANCE_SHARPE_HIGH: float = _settings.PERFORMANCE_SHARPE_HIGH
    PERFORMANCE_SHARPE_LOW: float = _settings.PERFORMANCE_SHARPE_LOW
    PERFORMANCE_PROFIT_FACTOR_CRITICAL: float = _settings.PERFORMANCE_PROFIT_FACTOR_CRITICAL

    # Erosion Rate Configuration
    EROSION_RATE_EXTREME: float = _settings.EROSION_RATE_EXTREME
    EROSION_RATE_NORMAL: float = _settings.EROSION_RATE_NORMAL
    EROSION_MIN_PROFIT_USD: float = _settings.EROSION_MIN_PROFIT_USD

    # Tactical Scout
    SCOUT_MODE_ENABLED: bool = _settings.SCOUT_MODE_ENABLED
    MAX_ALIGNMENT_DELTA_S: int = _settings.MAX_ALIGNMENT_DELTA_S
    WEIGHT_RECALL_TARGET: float = _settings.WEIGHT_RECALL_TARGET
    WEIGHT_PRECISION_LIMIT: float = _settings.WEIGHT_PRECISION_LIMIT
    SHAP_STABILITY_THRESHOLD: float = _settings.SHAP_STABILITY_THRESHOLD
    ADX_TREND_LEVEL: int = _settings.ADX_TREND_LEVEL
    VOLATILITY_LIMIT_PCT: float = _settings.VOLATILITY_LIMIT_PCT
    COMMISSION_GUARD_RATIO: float = _settings.COMMISSION_GUARD_RATIO
    SCOUT_LEVERAGE_MULT: float = _settings.SCOUT_LEVERAGE_MULT
    CANARY_RATIO: float = _settings.CANARY_RATIO
    CANARY_FAIL_DELTA: float = _settings.CANARY_FAIL_DELTA
    REPLAY_SEED: int = _settings.REPLAY_SEED
    REPLAY_CHECKPOINT_CYCLES: int = _settings.REPLAY_CHECKPOINT_CYCLES
    HOLD_THRESHOLD_DEFAULT: float = _settings.HOLD_THRESHOLD_DEFAULT
    HOLD_THRESHOLD_NEUTRAL: float = _settings.HOLD_THRESHOLD_NEUTRAL

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present."""
        errors = []

        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY is required")
        if cls.TRADING_MODE == "live":
            if not cls.BINANCE_API_KEY or not cls.BINANCE_SECRET_KEY:
                errors.append(
                    "BINANCE_API_KEY and BINANCE_SECRET_KEY are required in live trading mode",
                )
            if cls.BINANCE_DEFAULT_LEVERAGE < 1:
                errors.append("BINANCE_DEFAULT_LEVERAGE must be >= 1")
            if cls.BINANCE_MARGIN_TYPE not in ("ISOLATED", "CROSSED"):
                errors.append("BINANCE_MARGIN_TYPE must be either 'ISOLATED' or 'CROSSED'")
            if cls.BINANCE_RECV_WINDOW < constants.MIN_BINANCE_RECV_WINDOW:
                errors.append(
                    f"BINANCE_RECV_WINDOW must be at least {constants.MIN_BINANCE_RECV_WINDOW} ms"
                )
            if cls.BINANCE_DEFAULT_LEVERAGE > cls.MAX_LEVERAGE:
                errors.append("BINANCE_DEFAULT_LEVERAGE cannot exceed MAX_LEVERAGE")

        if cls.INITIAL_BALANCE <= 0:
            errors.append("INITIAL_BALANCE must be positive")

        if cls.CYCLE_INTERVAL_MINUTES < 1:
            errors.append("CYCLE_INTERVAL_MINUTES must be at least 1")

        if cls.MAX_LEVERAGE < 1:
            errors.append("MAX_LEVERAGE must be at least 1")

        if cls.HTF_INTERVAL.lower() not in ("1h", "4h", "2h", "30m"):
            errors.append("HTF_INTERVAL must be one of ['30m', '1h', '2h', '4h']")

        if cls.JSON_SERIES_MAX_LENGTH < constants.MIN_JSON_SERIES_LENGTH:
            errors.append(
                f"JSON_SERIES_MAX_LENGTH must be at least {constants.MIN_JSON_SERIES_LENGTH}"
            )

        if errors:
            logging.error("Configuration validation failed:")
            for error in errors:
                logging.error(f"  - {error}")
            return False

        return True

    @classmethod
    def get_masked_api_key(cls, api_key: str | None) -> str:
        """Return a masked version of the API key for logging."""
        if not api_key:
            return "Not set"
        if len(api_key) <= constants.API_KEY_MIN_LENGTH:
            return "***"
        return f"{api_key[:4]}...{api_key[-4:]}"

    @classmethod
    def log_config_summary(cls):
        """Log a summary of the configuration (without exposing sensitive data)."""
        logging.info("Configuration Summary:")
        logging.info(f"  DEEPSEEK_API_KEY: {cls.get_masked_api_key(cls.DEEPSEEK_API_KEY)}")
        logging.info(f"  BINANCE_API_KEY: {cls.get_masked_api_key(cls.BINANCE_API_KEY)}")
        logging.info(f"  DEBUG: {cls.DEBUG}")
        logging.info(f"  LOG_LEVEL: {cls.LOG_LEVEL}")
        logging.info(f"  INITIAL_BALANCE: ${cls.INITIAL_BALANCE}")
        logging.info(f"  CYCLE_INTERVAL_MINUTES: {cls.CYCLE_INTERVAL_MINUTES}")
        logging.info(f"  MAX_LEVERAGE: {cls.MAX_LEVERAGE}x")
        logging.info(f"  MIN_CONFIDENCE: {cls.MIN_CONFIDENCE}")
        logging.info(f"  RISK_LEVEL: {cls.RISK_LEVEL.upper()}")
        logging.info(f"  TRADING_MODE: {cls.TRADING_MODE.upper()}")
        logging.info(f"  BINANCE_TESTNET: {cls.BINANCE_TESTNET}")
        logging.info(f"  HTF_INTERVAL: {cls.HTF_INTERVAL}")
        if cls.TRADING_MODE == "live":
            logging.info(f"  BINANCE_MARGIN_TYPE: {cls.BINANCE_MARGIN_TYPE}")
            logging.info(f"  BINANCE_DEFAULT_LEVERAGE: {cls.BINANCE_DEFAULT_LEVERAGE}x")
