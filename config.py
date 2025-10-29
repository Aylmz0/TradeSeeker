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
    MAX_TRADE_NOTIONAL_USD: float = float(os.getenv('MAX_TRADE_NOTIONAL_USD', '100.0'))
    CYCLE_INTERVAL_MINUTES: int = int(os.getenv('CYCLE_INTERVAL_MINUTES', '2'))
    
    # Risk Management
    MAX_LEVERAGE: int = int(os.getenv('MAX_LEVERAGE', '10'))
    MIN_CONFIDENCE: float = float(os.getenv('MIN_CONFIDENCE', '0.5'))
    MAX_POSITIONS: int = int(os.getenv('MAX_POSITIONS', '3'))
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present."""
        errors = []
        
        # Check required API keys
        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY is required")
        
        # Validate numeric values
        if cls.INITIAL_BALANCE <= 0:
            errors.append("INITIAL_BALANCE must be positive")
        
        if cls.MAX_TRADE_NOTIONAL_USD <= 0:
            errors.append("MAX_TRADE_NOTIONAL_USD must be positive")
        
        if cls.CYCLE_INTERVAL_MINUTES < 1:
            errors.append("CYCLE_INTERVAL_MINUTES must be at least 1")
        
        if cls.MAX_LEVERAGE < 1:
            errors.append("MAX_LEVERAGE must be at least 1")
        
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
        logging.info(f"  MAX_TRADE_NOTIONAL_USD: ${cls.MAX_TRADE_NOTIONAL_USD}")
        logging.info(f"  CYCLE_INTERVAL_MINUTES: {cls.CYCLE_INTERVAL_MINUTES}")
        logging.info(f"  MAX_LEVERAGE: {cls.MAX_LEVERAGE}x")
        logging.info(f"  MIN_CONFIDENCE: {cls.MIN_CONFIDENCE}")

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
