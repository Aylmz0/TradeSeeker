"""
Configuration management for the Alpha Arena DeepSeek bot.
Provides secure API key handling and application configuration.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration with secure API key handling."""
    
    # API Keys
    DEEPSEEK_API_KEY: Optional[str] = os.getenv('DEEPSEEK_API_KEY')
    BINANCE_API_KEY: Optional[str] = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY: Optional[str] = os.getenv('BINANCE_SECRET_KEY')
    
    # Application Settings
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO').upper()
    MAX_RETRY_ATTEMPTS: int = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
    
    # Trading Settings
    INITIAL_BALANCE: float = float(os.getenv('INITIAL_BALANCE', '200.0'))
    CYCLE_INTERVAL_MINUTES: int = int(os.getenv('CYCLE_INTERVAL_MINUTES', '2'))
    HISTORY_RESET_INTERVAL: int = int(os.getenv('HISTORY_RESET_INTERVAL', '35'))
    MAX_CYCLES: int = int(os.getenv('MAX_CYCLES', '0'))  # 0 = unlimited, >0 = stop at this cycle
    
    # Risk Management
    MAX_LEVERAGE: int = int(os.getenv('MAX_LEVERAGE', '20'))  # Nof1ai blog: 20x leverage for medium risk
    MIN_CONFIDENCE: float = float(os.getenv('MIN_CONFIDENCE', '0.4'))  # Nof1ai blog: medium risk
    MAX_POSITIONS: int = int(os.getenv('MAX_POSITIONS', '5'))  # Nof1ai blog: 2-3 positions, but system uses 5
    RISK_PER_TRADE_USD: float = float(os.getenv('RISK_PER_TRADE_USD', '3.0'))  # Fixed dollar risk per trade
    
    # Smart Cooldown Settings
    SMART_COOLDOWN_LOSS: int = int(os.getenv('SMART_COOLDOWN_LOSS', '4'))
    SMART_COOLDOWN_WIN: int = int(os.getenv('SMART_COOLDOWN_WIN', '2'))
    EXTENDED_LOSS_CYCLES: int = int(os.getenv('EXTENDED_LOSS_CYCLES', '15'))  # Close position after this many negative cycles
    
    # Risk Level Configuration
    RISK_LEVEL: str = os.getenv('RISK_LEVEL', 'medium').lower()
    TRADING_MODE: str = os.getenv('TRADING_MODE', 'simulation').lower()  # simulation | live
    BINANCE_TESTNET: bool = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    BINANCE_MARGIN_TYPE: str = os.getenv('BINANCE_MARGIN_TYPE', 'ISOLATED').upper()
    BINANCE_DEFAULT_LEVERAGE: int = int(os.getenv('BINANCE_DEFAULT_LEVERAGE', '10'))
    BINANCE_RECV_WINDOW: int = int(os.getenv('BINANCE_RECV_WINDOW', '5000'))
    
    # Position Limits
    SAME_DIRECTION_LIMIT: int = int(os.getenv('SAME_DIRECTION_LIMIT', '2'))
    DYNAMIC_DIRECTION_LIMIT: int = int(os.getenv('DYNAMIC_DIRECTION_LIMIT', '2'))
    
    # Market Analysis Thresholds
    GLOBAL_NEUTRAL_STRENGTH_THRESHOLD: float = float(os.getenv('GLOBAL_NEUTRAL_STRENGTH_THRESHOLD', '0.4'))
    RSI_OVERBOUGHT_THRESHOLD: float = float(os.getenv('RSI_OVERBOUGHT_THRESHOLD', '70.0'))
    RSI_OVERSOLD_THRESHOLD: float = float(os.getenv('RSI_OVERSOLD_THRESHOLD', '30.0'))
    EMA_NEUTRAL_BAND_PCT: float = float(os.getenv('EMA_NEUTRAL_BAND_PCT', '0.001'))  # 0.1% band around EMA
    INTRADAY_NEUTRAL_RSI_LOW: float = float(os.getenv('INTRADAY_NEUTRAL_RSI_LOW', '45.0'))
    INTRADAY_NEUTRAL_RSI_HIGH: float = float(os.getenv('INTRADAY_NEUTRAL_RSI_HIGH', '55.0'))
    
    # Enhanced Trading Settings
    SHORT_ENHANCEMENT_MULTIPLIER: float = float(os.getenv('SHORT_ENHANCEMENT_MULTIPLIER', '1.15'))  # %15 daha büyük short
    VOLUME_QUALITY_THRESHOLDS: dict = {
        'excellent': 2.5,  # >2.5x average volume
        'good': 1.8,       # >1.8x average volume
        'fair': 1.2,       # >1.2x average volume
        'poor': 0.7        # >0.7x average volume
    }
    
    DIRECTIONAL_BULLISH_LONG_MULTIPLIER: float = float(os.getenv('DIRECTIONAL_BULLISH_LONG_MULTIPLIER', '1.00'))
    DIRECTIONAL_BULLISH_SHORT_MULTIPLIER: float = float(os.getenv('DIRECTIONAL_BULLISH_SHORT_MULTIPLIER', '0.90'))
    DIRECTIONAL_BEARISH_LONG_MULTIPLIER: float = float(os.getenv('DIRECTIONAL_BEARISH_LONG_MULTIPLIER', '0.90'))
    DIRECTIONAL_BEARISH_SHORT_MULTIPLIER: float = float(os.getenv('DIRECTIONAL_BEARISH_SHORT_MULTIPLIER', '1.00'))
    DIRECTIONAL_NEUTRAL_MULTIPLIER: float = float(os.getenv('DIRECTIONAL_NEUTRAL_MULTIPLIER', '0.95'))  # Neutral trend = slight confidence reduction
    MARKET_REGIME_MULTIPLIERS: dict = {
        'BULLISH': 1.0,
        'BEARISH': 1.0,
        'NEUTRAL': 0.9,
        'CHOPPY': 0.8  # Reduced risk for choppy/sideways markets
    }
    
    # Choppy Regime Detection Settings
    CHOPPY_ER_THRESHOLD: float = float(os.getenv('CHOPPY_ER_THRESHOLD', '0.30'))  # ER < 0.30 = Choppy
    CHOPPY_COIN_RATIO_MIN: float = float(os.getenv('CHOPPY_COIN_RATIO_MIN', '0.5'))  # >50% coins choppy
    CHOPPY_LEVERAGE: int = int(os.getenv('CHOPPY_LEVERAGE', '5'))  # Reduced leverage in choppy markets
    CHOPPY_TP_LONG_MULTIPLIER: float = float(os.getenv('CHOPPY_TP_LONG_MULTIPLIER', '1.004'))  # 0.4% TP for longs
    CHOPPY_TP_SHORT_MULTIPLIER: float = float(os.getenv('CHOPPY_TP_SHORT_MULTIPLIER', '0.996'))  # 0.4% TP for shorts
    CHOPPY_SL_LONG_MULTIPLIER: float = float(os.getenv('CHOPPY_SL_LONG_MULTIPLIER', '0.994'))  # 0.6% SL for longs
    CHOPPY_SL_SHORT_MULTIPLIER: float = float(os.getenv('CHOPPY_SL_SHORT_MULTIPLIER', '1.006'))  # 0.6% SL for shorts
    CHOPPY_TP_SL_MULTIPLIER: float = float(os.getenv('CHOPPY_TP_SL_MULTIPLIER', '0.004')) # 0.4% base multiplier
    CHOPPY_HIGH_ER_EXCEPTION: float = float(os.getenv('CHOPPY_HIGH_ER_EXCEPTION', '0.45'))  # ER > 0.45 bypasses global choppy
    
    # Dynamic Volatility Scaling (ATR) Settings
    ATR_TP_MULTIPLIER: float = float(os.getenv('ATR_TP_MULTIPLIER', '2.0'))  # Target = Entry +/- (ATR * 2.0)
    ATR_SL_MULTIPLIER: float = float(os.getenv('ATR_SL_MULTIPLIER', '1.5'))  # Stop = Entry +/- (ATR * 1.5)
    
    # Flash Exit Settings (V-Reversal Protection)
    FLASH_EXIT_ENABLED: bool = os.getenv('FLASH_EXIT_ENABLED', 'true').lower() == 'true'
    FLASH_EXIT_RSI_DELTA_MIN: float = float(os.getenv('FLASH_EXIT_RSI_DELTA_MIN', '15.0'))  # RSI spike threshold
    FLASH_EXIT_VOLUME_SURGE_MIN: float = float(os.getenv('FLASH_EXIT_VOLUME_SURGE_MIN', '3.0'))  # Volume surge multiplier
    FLASH_EXIT_LOSS_TRIGGER_MULTIPLIER: float = float(os.getenv('FLASH_EXIT_LOSS_TRIGGER_MULTIPLIER', '1.002'))  # 0.2% loss trigger
    
    COIN_SPECIFIC_STOP_LOSS_MULTIPLIERS: dict = {
        'SOL': 1.0,   # AI'nın SL değerine saygı duy
        'TRX': 1.0,   # AI'nın SL değerine saygı duy
        'XRP': 1.0,   # AI'nın SL değerine saygı duy
        'LINK': 1.0,  # AI'nın SL değerine saygı duy
        'DOGE': 1.0,  # AI'nın SL değerine saygı duy
        'ASTER': 1.0  # AI'nın SL değerine saygı duy
    }
    
    # Dynamic Confidence-Based Position Sizing
    CONFIDENCE_BASED_RISK_PERCENTAGES: dict = {
        'low': (0.10, 0.15),      # %10-15 risk (max margin'in)
        'medium': (0.15, 0.20),   # %15-20 risk  
        'high': (0.20, 0.25)      # %20-25 risk
    }
    
    # Minimum Position Size Configuration (Margin-based)
    MIN_POSITION_MARGIN_USD: float = float(os.getenv('MIN_POSITION_MARGIN_USD', '10.0'))  # Minimum $8 margin ile pozisyon açılabilir
    MIN_POSITION_CLEANUP_THRESHOLD: float = float(os.getenv('MIN_POSITION_CLEANUP_THRESHOLD', '5.0'))  # $5 altındakileri temizle
    
    # Partial Profit Taking Configuration (Margin-based)
    MIN_PARTIAL_PROFIT_MARGIN_REMAINING_USD: float = float(os.getenv('MIN_PARTIAL_PROFIT_MARGIN_REMAINING_USD', '15.0'))  # Minimum $15 margin kalacak şekilde satış
    MAXIMUM_LIMIT_BALANCE_PCT: float = float(os.getenv('MAXIMUM_LIMIT_BALANCE_PCT', '0.08'))  # %8 of current balance for max limit calculation

    # Exit Plan Defaults
    DEFAULT_STOP_LOSS_PCT: float = float(os.getenv('DEFAULT_STOP_LOSS_PCT', '0.01'))  # 1% default SL buffer
    DEFAULT_PROFIT_TARGET_PCT: float = float(os.getenv('DEFAULT_PROFIT_TARGET_PCT', '0.015'))  # 1.5% default TP buffer
    MIN_EXIT_PLAN_OFFSET: float = float(os.getenv('MIN_EXIT_PLAN_OFFSET', '0.0001'))  # Minimum absolute offset for tiny-priced coins

    # Trailing Stop Configuration
    TRAILING_PROGRESS_TRIGGER: float = float(os.getenv('TRAILING_PROGRESS_TRIGGER', '40.0'))  # % progress towards TP
    TRAILING_TIME_PROGRESS_FLOOR: float = float(os.getenv('TRAILING_TIME_PROGRESS_FLOOR', '30.0'))  # % progress required for time-based trailing
    TRAILING_TIME_MINUTES: float = float(os.getenv('TRAILING_TIME_MINUTES', '20'))  # Minutes in trade before time-based trailing activates
    TRAILING_ATR_MULTIPLIER: float = float(os.getenv('TRAILING_ATR_MULTIPLIER', '1.2'))  # Multiplier for ATR-based buffer
    TRAILING_FALLBACK_BUFFER_PCT: float = float(os.getenv('TRAILING_FALLBACK_BUFFER_PCT', '0.004'))  # 0.4% price buffer fallback
    TRAILING_VOLUME_ABSOLUTE_THRESHOLD: float = float(os.getenv('TRAILING_VOLUME_ABSOLUTE_THRESHOLD', '0.2'))  # Absolute volume ratio floor
    TRAILING_VOLUME_DROP_RATIO: float = float(os.getenv('TRAILING_VOLUME_DROP_RATIO', '0.5'))  # Relative drop vs entry volume ratio
    TRAILING_MIN_IMPROVEMENT_PCT: float = float(os.getenv('TRAILING_MIN_IMPROVEMENT_PCT', '0.0005'))  # Minimum 0.05% improvement before updating stop
    TRAILING_PROGRESS_TRIGGER_EXTREME: float = float(os.getenv('TRAILING_PROGRESS_TRIGGER_EXTREME', '20.0'))  # Lower trigger for LOWER_10/UPPER_10 zones

    # Win Streak Cooldown Settings
    WIN_STREAK_COOLDOWN_THRESHOLD: int = int(os.getenv('WIN_STREAK_COOLDOWN_THRESHOLD', '2'))  # Consecutive wins before cooldown
    WIN_STREAK_COOLDOWN_CYCLES: int = int(os.getenv('WIN_STREAK_COOLDOWN_CYCLES', '1'))  # Cycles to wait after win streak

    # Higher Timeframe Configuration
    HTF_INTERVAL: str = os.getenv('HTF_INTERVAL', '1h').lower()
    
    # JSON Prompt Feature Flags
    USE_JSON_PROMPT: bool = os.getenv('USE_JSON_PROMPT', 'False').lower() == 'true'
    JSON_PROMPT_COMPACT: bool = os.getenv('JSON_PROMPT_COMPACT', 'False').lower() == 'true'  # Compact JSON (indent=None)
    VALIDATE_JSON_PROMPTS: bool = os.getenv('VALIDATE_JSON_PROMPTS', 'False').lower() == 'true'  # Runtime validation
    JSON_PROMPT_VERSION: str = os.getenv('JSON_PROMPT_VERSION', '1.0')  # Format version
    JSON_SERIES_MAX_LENGTH: int = int(os.getenv('JSON_SERIES_MAX_LENGTH', '30'))  # Max series length before compression (optimized for performance)
    
    # Smart Indicator Cache Configuration
    USE_SMART_CACHE: bool = os.getenv('USE_SMART_CACHE', 'true').lower() == 'true'  # Enable smart TTL cache for 15m and HTF
    SMART_CACHE_SAFETY_MARGIN: float = float(os.getenv('SMART_CACHE_SAFETY_MARGIN', '0.85'))  # Safety margin for TTL (85% of candle duration)
    SMART_CACHE_STATS_LOGGING: bool = os.getenv('SMART_CACHE_STATS_LOGGING', 'true').lower() == 'true'  # Print cache stats
    
    # Performance Monitor Thresholds
    PERFORMANCE_PROFITABILITY_HIGH: float = float(os.getenv('PERFORMANCE_PROFITABILITY_HIGH', '50.0'))
    PERFORMANCE_PROFITABILITY_LOW: float = float(os.getenv('PERFORMANCE_PROFITABILITY_LOW', '40.0'))
    PERFORMANCE_PROFIT_FACTOR_LOW: float = float(os.getenv('PERFORMANCE_PROFIT_FACTOR_LOW', '1.2'))
    PERFORMANCE_PROFIT_FACTOR_HIGH: float = float(os.getenv('PERFORMANCE_PROFIT_FACTOR_HIGH', '1.5'))
    PERFORMANCE_DECISION_RATE_HIGH: float = float(os.getenv('PERFORMANCE_DECISION_RATE_HIGH', '60.0'))
    PERFORMANCE_DECISION_RATE_LOW: float = float(os.getenv('PERFORMANCE_DECISION_RATE_LOW', '30.0'))
    PERFORMANCE_RETURN_HIGH: float = float(os.getenv('PERFORMANCE_RETURN_HIGH', '5.0'))
    PERFORMANCE_RETURN_LOW: float = float(os.getenv('PERFORMANCE_RETURN_LOW', '0.0'))
    PERFORMANCE_DRAWDOWN_THRESHOLD: float = float(os.getenv('PERFORMANCE_DRAWDOWN_THRESHOLD', '-10.0'))
    PERFORMANCE_SHARPE_HIGH: float = float(os.getenv('PERFORMANCE_SHARPE_HIGH', '1.0'))
    PERFORMANCE_SHARPE_LOW: float = float(os.getenv('PERFORMANCE_SHARPE_LOW', '0.0'))
    PERFORMANCE_PROFIT_FACTOR_CRITICAL: float = float(os.getenv('PERFORMANCE_PROFIT_FACTOR_CRITICAL', '0.8'))
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present."""
        errors = []
        
        # Check required API keys
        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY is required")
        if cls.TRADING_MODE not in ('simulation', 'live'):
            errors.append("TRADING_MODE must be either 'simulation' or 'live'")
        if cls.TRADING_MODE == 'live':
            if not cls.BINANCE_API_KEY or not cls.BINANCE_SECRET_KEY:
                errors.append("BINANCE_API_KEY and BINANCE_SECRET_KEY are required in live trading mode")
            if cls.BINANCE_DEFAULT_LEVERAGE < 1:
                errors.append("BINANCE_DEFAULT_LEVERAGE must be >= 1")
            if cls.BINANCE_MARGIN_TYPE not in ('ISOLATED', 'CROSSED'):
                errors.append("BINANCE_MARGIN_TYPE must be either 'ISOLATED' or 'CROSSED'")
            if cls.BINANCE_RECV_WINDOW < 1000:
                errors.append("BINANCE_RECV_WINDOW must be at least 1000 ms")
            if cls.BINANCE_DEFAULT_LEVERAGE > cls.MAX_LEVERAGE:
                errors.append("BINANCE_DEFAULT_LEVERAGE cannot exceed MAX_LEVERAGE")
        
        # Validate numeric values
        if cls.INITIAL_BALANCE <= 0:
            errors.append("INITIAL_BALANCE must be positive")
        
        if cls.CYCLE_INTERVAL_MINUTES < 1:
            errors.append("CYCLE_INTERVAL_MINUTES must be at least 1")
        
        if cls.MAX_LEVERAGE < 1:
            errors.append("MAX_LEVERAGE must be at least 1")

        if cls.HTF_INTERVAL.lower() not in ('1h', '4h', '2h', '30m'):
            errors.append("HTF_INTERVAL must be one of ['30m', '1h', '2h', '4h']")
        
        # Validate JSON prompt settings
        if cls.JSON_SERIES_MAX_LENGTH < 10:
            errors.append("JSON_SERIES_MAX_LENGTH must be at least 10")
        
        if errors:
            logging.error("Configuration validation failed:")
            for error in errors:
                logging.error(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def get_masked_api_key(cls, api_key: Optional[str]) -> str:
        """Return a masked version of the API key for logging."""
        if not api_key:
            return "Not set"
        if len(api_key) <= 8:
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
        if cls.TRADING_MODE == 'live':
            logging.info(f"  BINANCE_MARGIN_TYPE: {cls.BINANCE_MARGIN_TYPE}")
            logging.info(f"  BINANCE_DEFAULT_LEVERAGE: {cls.BINANCE_DEFAULT_LEVERAGE}x")

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
