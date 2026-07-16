"""Pydantic Settings for TradeSeeker — replaces os.getenv() with validated types."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from .env with type validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # --- AI Provider Selection ---
    PRIMARY_AI_PROVIDER: str = Field(default="openrouter")

    # Groq
    GROQ_API_KEY: str | None = Field(default=None)
    GROQ_MODEL: str = Field(default="groq/compound")

    # DeepSeek
    DEEPSEEK_API_KEY: str | None = Field(default=None)

    # Mimo
    MIMO_API_KEY: str | None = Field(default=None)
    MIMO_MODEL: str = Field(default="mimo-v2-flash")
    MIMO_THINKING_ENABLED: bool = Field(default=False)

    # ZAI
    ZAI_API_KEY: str | None = Field(default=None)
    ZAI_MODEL: str = Field(default="glm-4.5-flash")
    ZAI_THINKING_ENABLED: bool = Field(default=True)

    # OpenRouter
    OPENROUTER_API_KEY: str | None = Field(default=None)
    OPENROUTER_MODEL: str = Field(default="openrouter/free")
    OPENROUTER_FALLBACK_MODEL: str = Field(default="google/gemini-2.0-flash-exp:free")
    OPENROUTER_REASONING_ENABLED: bool = Field(default=True)

    # Binance
    BINANCE_API_KEY: str | None = Field(default=None)
    BINANCE_SECRET_KEY: str | None = Field(default=None)

    # --- Application Settings ---
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    LOG_DIR: str = Field(default="data/logs")
    MAX_RETRY_ATTEMPTS: int = Field(default=3, ge=1)
    REQUEST_TIMEOUT: int = Field(default=30, ge=1)

    # --- Trading Settings ---
    INITIAL_BALANCE: float = Field(default=200.0, gt=0)
    CYCLE_INTERVAL_MINUTES: int = Field(default=2, ge=1)
    HISTORY_RESET_INTERVAL: int = Field(default=35, ge=1)
    MAX_CYCLES: int = Field(default=0, ge=0)

    # --- Risk Management ---
    MAX_LEVERAGE: int = Field(default=20, ge=1)
    MIN_CONFIDENCE: float = Field(default=0.60, ge=0, le=1)
    CHOPPY_ER_THRESHOLD: float = Field(default=0.30, ge=0, le=1)
    ML_CONFIDENCE_THRESHOLD: float = Field(default=40.0, ge=0)
    MAX_POSITIONS: int = Field(default=5, ge=1)
    RISK_PER_TRADE_USD: float = Field(default=3.0, gt=0)

    # Graduated Stop Loss
    LOSS_MULT_L1: float = Field(default=0.20, ge=0, le=1)
    LOSS_MULT_L2: float = Field(default=0.15, ge=0, le=1)
    LOSS_MULT_L3: float = Field(default=0.12, ge=0, le=1)
    LOSS_MULT_L4: float = Field(default=0.10, ge=0, le=1)
    LOSS_MULT_BASE: float = Field(default=0.08, ge=0, le=1)

    # Smart Cooldown
    SMART_COOLDOWN_LOSS: int = Field(default=3, ge=0)
    SMART_COOLDOWN_WIN: int = Field(default=2, ge=0)
    MAX_NEW_POSITIONS_PER_CYCLE: int = Field(default=3, ge=1)
    EXTENDED_LOSS_CYCLES: int = Field(default=15, ge=1)
    EXTENDED_PROFIT_CYCLES: int = Field(default=15, ge=1)

    # --- Risk Level ---
    RISK_LEVEL: str = Field(default="medium")
    TRADING_MODE: str = Field(default="simulation")
    BINANCE_TESTNET: bool = Field(default=False)
    BINANCE_MARGIN_TYPE: str = Field(default="ISOLATED")
    BINANCE_DEFAULT_LEVERAGE: int = Field(default=10, ge=1)
    BINANCE_RECV_WINDOW: int = Field(default=5000, ge=1000)

    # Position Limits
    SAME_DIRECTION_LIMIT: int = Field(default=2, ge=1)
    DYNAMIC_DIRECTION_LIMIT: int = Field(default=2, ge=1)

    # --- Market Analysis Thresholds ---
    GLOBAL_NEUTRAL_STRENGTH_THRESHOLD: float = Field(default=0.4, ge=0, le=1)
    RSI_OVERBOUGHT_THRESHOLD: float = Field(default=70.0, ge=0, le=100)
    RSI_OVERSOLD_THRESHOLD: float = Field(default=30.0, ge=0, le=100)
    EMA_NEUTRAL_BAND_PCT: float = Field(default=0.001, ge=0)
    INTRADAY_NEUTRAL_RSI_LOW: float = Field(default=45.0, ge=0, le=100)
    INTRADAY_NEUTRAL_RSI_HIGH: float = Field(default=55.0, ge=0, le=100)
    TREND_LONG_RSI_THRESHOLD: float = Field(default=50.0, ge=0, le=100)
    TREND_SHORT_RSI_THRESHOLD: float = Field(default=50.0, ge=0, le=100)

    # --- Enhanced Trading Settings ---
    SHORT_ENHANCEMENT_MULTIPLIER: float = Field(default=1.15, gt=0)
    VOLUME_MINIMUM_THRESHOLD: float = Field(default=0.30, ge=0)
    SIMULATION_COMMISSION_RATE: float = Field(default=0.0005, ge=0, le=1)
    DIRECTIONAL_BULLISH_LONG_MULTIPLIER: float = Field(default=1.00, gt=0)
    DIRECTIONAL_BULLISH_SHORT_MULTIPLIER: float = Field(default=0.90, gt=0)
    DIRECTIONAL_BEARISH_LONG_MULTIPLIER: float = Field(default=0.90, gt=0)
    DIRECTIONAL_BEARISH_SHORT_MULTIPLIER: float = Field(default=1.00, gt=0)
    DIRECTIONAL_NEUTRAL_MULTIPLIER: float = Field(default=0.95, gt=0)

    # Choppy Regime
    CHOPPY_COIN_RATIO_MIN: float = Field(default=0.5, ge=0, le=1)
    CHOPPY_LEVERAGE: int = Field(default=5, ge=1)
    CHOPPY_TP_LONG_MULTIPLIER: float = Field(default=1.004, gt=0)
    CHOPPY_TP_SHORT_MULTIPLIER: float = Field(default=0.996, gt=0)
    CHOPPY_SL_LONG_MULTIPLIER: float = Field(default=0.994, gt=0)
    CHOPPY_SL_SHORT_MULTIPLIER: float = Field(default=1.006, gt=0)
    CHOPPY_TP_SL_MULTIPLIER: float = Field(default=0.004, ge=0)
    CHOPPY_HIGH_ER_EXCEPTION: float = Field(default=0.45, ge=0, le=1)

    # ATR
    ATR_TP_MULTIPLIER: float = Field(default=2.0, gt=0)
    ATR_SL_MULTIPLIER: float = Field(default=1.8, gt=0)

    # Flash Exit
    FLASH_EXIT_ENABLED: bool = Field(default=True)
    FLASH_EXIT_RSI_DELTA_MIN: float = Field(default=15.0, ge=0)
    FLASH_EXIT_VOLUME_SURGE_MIN: float = Field(default=3.0, ge=0)
    FLASH_EXIT_LOSS_TRIGGER_MULTIPLIER: float = Field(default=1.002, gt=0)

    # Minimum Position Size
    MIN_POSITION_MARGIN_USD: float = Field(default=10.0, gt=0)
    MIN_POSITION_CLEANUP_THRESHOLD: float = Field(default=5.0, gt=0)

    # Partial Profit Taking
    MIN_PARTIAL_PROFIT_MARGIN_REMAINING_USD: float = Field(default=10.0, gt=0)
    MAXIMUM_LIMIT_BALANCE_PCT: float = Field(default=0.08, ge=0, le=1)

    # Exit Plan Defaults
    DEFAULT_STOP_LOSS_PCT: float = Field(default=0.01, ge=0, le=1)
    DEFAULT_PROFIT_TARGET_PCT: float = Field(default=0.015, ge=0, le=1)
    MIN_EXIT_PLAN_OFFSET: float = Field(default=0.0001, ge=0)

    # Trailing Stop
    TRAILING_PROGRESS_TRIGGER: float = Field(default=40.0, ge=0, le=100)
    TRAILING_TIME_PROGRESS_FLOOR: float = Field(default=30.0, ge=0, le=100)
    TRAILING_TIME_MINUTES: float = Field(default=20, ge=0)
    TRAILING_ATR_MULTIPLIER: float = Field(default=1.2, gt=0)
    TRAILING_FALLBACK_BUFFER_PCT: float = Field(default=0.004, ge=0)
    TRAILING_VOLUME_ABSOLUTE_THRESHOLD: float = Field(default=0.2, ge=0)
    TRAILING_VOLUME_DROP_RATIO: float = Field(default=0.5, ge=0, le=1)
    TRAILING_MIN_IMPROVEMENT_PCT: float = Field(default=0.0005, ge=0)
    TRAILING_PROGRESS_TRIGGER_EXTREME: float = Field(default=20.0, ge=0, le=100)

    # Win Streak
    WIN_STREAK_COOLDOWN_THRESHOLD: int = Field(default=2, ge=0)
    WIN_STREAK_COOLDOWN_CYCLES: int = Field(default=1, ge=0)

    # Higher Timeframe
    HTF_INTERVAL: str = Field(default="1h")

    # JSON Prompt
    USE_JSON_PROMPT: bool = Field(default=False)
    JSON_PROMPT_COMPACT: bool = Field(default=False)
    VALIDATE_JSON_PROMPTS: bool = Field(default=False)
    JSON_PROMPT_VERSION: str = Field(default="1.0")
    JSON_SERIES_MAX_LENGTH: int = Field(default=30, ge=10)

    # Smart Cache
    USE_SMART_CACHE: bool = Field(default=True)
    SMART_CACHE_SAFETY_MARGIN: float = Field(default=0.85, ge=0, le=1)
    SMART_CACHE_STATS_LOGGING: bool = Field(default=True)

    # Performance Monitor
    PERFORMANCE_PROFITABILITY_HIGH: float = Field(default=50.0)
    PERFORMANCE_PROFITABILITY_LOW: float = Field(default=40.0)
    PERFORMANCE_PROFIT_FACTOR_LOW: float = Field(default=1.2)
    PERFORMANCE_PROFIT_FACTOR_HIGH: float = Field(default=1.5)
    PERFORMANCE_DECISION_RATE_HIGH: float = Field(default=60.0)
    PERFORMANCE_DECISION_RATE_LOW: float = Field(default=30.0)
    PERFORMANCE_RETURN_HIGH: float = Field(default=5.0)
    PERFORMANCE_RETURN_LOW: float = Field(default=0.0)
    PERFORMANCE_DRAWDOWN_THRESHOLD: float = Field(default=-10.0)
    PERFORMANCE_SHARPE_HIGH: float = Field(default=1.0)
    PERFORMANCE_SHARPE_LOW: float = Field(default=0.0)
    PERFORMANCE_PROFIT_FACTOR_CRITICAL: float = Field(default=0.8)

    # Erosion Rate
    EROSION_RATE_EXTREME: float = Field(default=0.04, ge=0, le=1)
    EROSION_RATE_NORMAL: float = Field(default=0.06, ge=0, le=1)
    EROSION_MIN_PROFIT_USD: float = Field(default=1.00, ge=0)

    # Tactical Scout
    SCOUT_MODE_ENABLED: bool = Field(default=False)
    MAX_ALIGNMENT_DELTA_S: int = Field(default=5, ge=0)
    WEIGHT_RECALL_TARGET: float = Field(default=0.60, ge=0, le=1)
    WEIGHT_PRECISION_LIMIT: float = Field(default=0.30, ge=0, le=1)
    SHAP_STABILITY_THRESHOLD: float = Field(default=0.85, ge=0, le=1)
    ADX_TREND_LEVEL: int = Field(default=25, ge=0)
    VOLATILITY_LIMIT_PCT: float = Field(default=0.02, ge=0)
    COMMISSION_GUARD_RATIO: float = Field(default=5.0, gt=0)
    SCOUT_LEVERAGE_MULT: float = Field(default=0.5, gt=0)
    CANARY_RATIO: float = Field(default=0.10, ge=0, le=1)
    CANARY_FAIL_DELTA: float = Field(default=0.05, ge=0)
    REPLAY_SEED: int = Field(default=42)
    REPLAY_CHECKPOINT_CYCLES: int = Field(default=50, ge=1)
    HOLD_THRESHOLD_DEFAULT: float = Field(default=0.82, ge=0, le=1)
    HOLD_THRESHOLD_NEUTRAL: float = Field(default=0.75, ge=0, le=1)

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls: type["Settings"], v: str) -> str:
        """Validate LOG_LEVEL is a standard Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"LOG_LEVEL must be one of {allowed}"
            raise ValueError(msg)
        return upper

    @field_validator("TRADING_MODE")
    @classmethod
    def validate_trading_mode(cls: type["Settings"], v: str) -> str:
        """Validate TRADING_MODE is either simulation or live."""
        allowed = {"simulation", "live"}
        lower = v.lower()
        if lower not in allowed:
            msg = f"TRADING_MODE must be one of {allowed}"
            raise ValueError(msg)
        return lower

    @field_validator("RISK_LEVEL")
    @classmethod
    def validate_risk_level(cls: type["Settings"], v: str) -> str:
        """Validate RISK_LEVEL is low, medium, or high."""
        allowed = {"low", "medium", "high"}
        lower = v.lower()
        if lower not in allowed:
            msg = f"RISK_LEVEL must be one of {allowed}"
            raise ValueError(msg)
        return lower

    @field_validator("HTF_INTERVAL")
    @classmethod
    def validate_htf_interval(cls: type["Settings"], v: str) -> str:
        """Validate HTF_INTERVAL is a supported candle interval."""
        allowed = {"30m", "1h", "2h", "4h"}
        lower = v.lower()
        if lower not in allowed:
            msg = f"HTF_INTERVAL must be one of {allowed}"
            raise ValueError(msg)
        return lower

    @field_validator("BINANCE_MARGIN_TYPE")
    @classmethod
    def validate_margin_type(cls: type["Settings"], v: str) -> str:
        """Validate BINANCE_MARGIN_TYPE is ISOLATED or CROSSED."""
        allowed = {"ISOLATED", "CROSSED"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"BINANCE_MARGIN_TYPE must be one of {allowed}"
            raise ValueError(msg)
        return upper

    def get_masked_api_key(self: "Settings", api_key: str | None) -> str:
        """Return a masked version of the API key for safe logging."""
        from src.core import constants

        if not api_key:
            return "Not set"
        if len(api_key) <= constants.API_KEY_MIN_LENGTH:
            return "***"
        return f"{api_key[:4]}...{api_key[-4:]}"
