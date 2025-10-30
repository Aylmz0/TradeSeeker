# alpha_arena_deepseek.py
import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings
import traceback # For detailed error logging

# Import new utility modules
from config import Config
from utils import (
    format_num, safe_file_write, safe_file_read, 
    rate_limiter, RetryManager, DataValidator, performance_monitor
)
from backtest import AdvancedRiskManager

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Validate configuration
if not Config.validate_config():
    print("❌ Configuration validation failed. Please check your .env file.")
    exit(1)

Config.log_config_summary()

# --- API Class ---
class DeepSeekAPI:
    """DeepSeek API integration with enhanced error handling and rate limiting"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        self.session = RetryManager.create_session_with_retry()

        if not self.api_key:
            print("❌ DEEPSEEK_API_KEY not found!")
            print("ℹ️  Please check your .env file configuration.")

    def get_ai_decision(self, prompt: str) -> str:
        """Get trading decision from DeepSeek API"""
        if not self.api_key:
            return self._get_simulation_response(prompt)

        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a zero-shot systematic trading model participating in Alpha Arena.
Your goal is to maximize PnL (profit and loss) by trading perpetual futures on 6 assets: XRP, DOGE, ASTR, ADA, LINK, SOL.

You are given $200 starting capital and must process numerical market data to discover alpha.
Your Sharpe ratio is provided to help normalize for risky behavior.

CRITICAL RULES:
- Trade systematically using only the numerical data provided
- Infer market narratives from time-series data, not external news
- Always specify complete exit plans: profit_target, stop_loss, invalidation_condition
- Position sizing is computed by you, conditioned on available cash, leverage, and your internal risk preference
- Use leverage up to 10x for calculated exposure
- Minimum confidence threshold: 0.4
- Maximum positions: 5

RISK MANAGEMENT (NOF1AI MEDIUM RISK PROFILE):
- Portfolio risk limit: 25% ($50 on $200) - Increased for 5 positions
- Position risk limit: 8% ($16 on $200) - Increased for better diversification
- Concentration limit: 35% per position - Reduced for better diversification
- Maximum positions: 5 out of 6 coins
- Aim for Risk/Reward ratio of at least 1:1.3
- Use 4-hour ATR for stop-loss distances
- For invalidation_condition, prioritize sustained breaks of key 4h levels confirmed by multiple candle closes

IMPORTANT STARTUP BEHAVIOR:
- On first cycle (Cycle 1), observe market conditions for at least 2-3 cycles before entering positions
- Do not enter trades immediately on system startup unless there is exceptionally strong confirmation
- Prefer to establish baseline market understanding before taking risk
- You can hold up to 5 positions simultaneously across the 6 available coins
- Focus on the strongest setups across all coins, not just 2-3 favorites
- Avoid XRP bias - do not automatically favor XRP over other coins
- Use variable position sizing based on confidence (30-100 USD margin), not fixed 60 USD

DATA CONTEXT:
- You receive 3m (entry/exit) and 4h (trend) data with historical indicator series
- All price/signal data is ordered: OLDEST → NEWEST
- Use series to understand momentum and trend structure
- Consider Open Interest and Funding Rate for market sentiment

ACTION FORMAT:
Use signals: `buy_to_enter`, `sell_to_enter`, `hold`, `close_position`.
If a position is open, only `hold` or `close_position` are allowed.

Provide response in `CHAIN_OF_THOUGHTS` and `DECISIONS` (JSON) parts.

Example Format (NOF1AI Blog Style):
CHAIN_OF_THOUGHTS
[Systematic analysis of all assets, current positions, and market conditions. Focus on 4h trends, momentum, and risk management. Example: "BTC breaking above consolidation zone with strong momentum. RSI at 62.5 shows room to run, MACD positive at 116.5, price well above EMA20. 4h timeframe showing recovery from oversold (RSI 45.4). Targeting retest of $110k-111k zone. Stop below $106,361 protects against false breakout."]
DECISIONS
{
  "XRP": {
    "signal": "buy_to_enter",
    "leverage": 12,
    "quantity_usd": 80, /* This is MARGIN */
    "confidence": 0.65,
    "profit_target": 0.56,
    "stop_loss": 0.48,
    "risk_usd": 35.0,
    "invalidation_condition": "If 4h price closes below 4h EMA20"
  },
  "SOL": { "signal": "hold" },
  "ADA": { "signal": "close_position", "justification": "Reached profit target." }
}

Remember: You are a systematic trading model. Make principled decisions based on quantitative data."""
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7, "max_tokens": 4096 # Increased max_tokens
            }

            print("🔄 Sending request to DeepSeek API...")
            response = requests.post(self.base_url, json=data, headers=headers, timeout=120) # Increased timeout further
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
        return """
CHAIN_OF_THOUGHTS
Simulation Mode: Assuming market pullback. Shorting SOL based on simulated 4h resistance. Aiming for 1:1.5 R/R using simulated ATR. Holding others.
DECISIONS
{
  "SOL": {
    "signal": "sell_to_enter",
    "leverage": 10,
    "quantity_usd": 30,
    "confidence": 0.65,
    "profit_target": 185.0,
    "stop_loss": 198.0,
    "risk_usd": 15.0,
    "invalidation_condition": "If 4h price closes above 199.0"
  },
  "XRP": { "signal": "hold" },
  "ADA": { "signal": "hold" },
  "DOGE": { "signal": "hold" },
  "ASTR": { "signal": "hold" },
  "LINK": { "signal": "hold" }
}
"""
    def _get_error_response(self, error_message: str) -> str:
        """Generates a standard error response for the AI cycle log"""
        print(f"🔧 Generating error response for AI cycle log.")
        # Generate hold decisions for all available coins
        hold_decisions = {coin: {"signal": "hold", "justification": f"Error during AI analysis: {error_message}"} for coin in RealMarketData().available_coins} # Assuming RealMarketData can be instantiated here or pass coins list

        return f"""
CHAIN_OF_THOUGHTS
Error occurred: {error_message}
Cannot generate trading decisions. Holding all positions/cash.
DECISIONS
{json.dumps(hold_decisions, indent=2)}
"""


# --- Market Data Class ---
class RealMarketData:
    """Real market data from Binance Spot and Futures"""

    def __init__(self):
        self.spot_url = "https://api.binance.com/api/v3"
        self.futures_url = "https://fapi.binance.com/fapi/v1"
        self.available_coins = ['XRP', 'DOGE', 'ASTR', 'ADA', 'LINK', 'SOL'] # SOL Added
        self.indicator_history_length = 10

    def get_real_time_data(self, symbol: str, interval: str = '3m', limit: int = 100) -> pd.DataFrame:
        """Get real OHLCV data from Binance Spot"""
        try:
            fetch_limit = limit + self.indicator_history_length + 50
            params = {'symbol': f'{symbol}USDT', 'interval': interval, 'limit': fetch_limit}
            response = requests.get(f"{self.spot_url}/klines", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if len(data) < 50:
                 print(f"⚠️ Warning: Insufficient kline data for {symbol} ({interval}). Got {len(data)}.")
                 return pd.DataFrame()

            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Kline data error {symbol} ({interval}): {e}")
            return pd.DataFrame()

    def get_open_interest(self, symbol: str) -> float:
        """Get Latest Open Interest from Binance Futures"""
        try:
            params = {'symbol': f'{symbol}USDT'}
            response = requests.get(f"{self.futures_url}/openInterest", params=params, timeout=5)
            response.raise_for_status()
            return float(response.json()['openInterest'])
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                print(f"ℹ️ OI not available for {symbol}USDT on Futures.")
            else:
                print(f"❌ OI error for {symbol}: {e}")
            return 0.0

    def get_funding_rate(self, symbol: str) -> float:
        """Get Latest Funding Rate from Binance Futures"""
        try:
            params = {'symbol': f'{symbol}USDT'}
            response = requests.get(f"{self.futures_url}/premiumIndex", params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list): data = data[0] if data else {}

            rate = data.get('lastFundingRate')
            if rate is not None and rate != '': return float(rate)
            else:
                 # print(f"ℹ️ Using nextFundingRate for {symbol}.") # Less verbose
                 rate = data.get('nextFundingRate')
                 return float(rate) if rate is not None and rate != '' else 0.0
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and (e.response.status_code in [404, 400]):
                 print(f"ℹ️ Funding Rate not available for {symbol}USDT on Futures.")
            else:
                print(f"❌ Funding Rate error for {symbol}: {e}")
            return 0.0

    # --- Indicator Calculation Functions ---
    def calculate_ema_series(self, prices, period): return prices.ewm(span=period, adjust=False).mean()
    def calculate_rsi_series(self, prices, period=14):
        if len(prices) < period + 1: return pd.Series([np.nan] * len(prices))
        delta = prices.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean(); avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100); rsi.loc[avg_gain == 0] = 0
        return rsi
    def calculate_macd_series(self, prices, fast=12, slow=26, signal=9):
        if len(prices) < slow: return pd.Series([np.nan]*len(prices)), pd.Series([np.nan]*len(prices)), pd.Series([np.nan]*len(prices))
        ema_fast = prices.ewm(span=fast, adjust=False).mean(); ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow; macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    def calculate_atr_series(self, df_high, df_low, df_close, period=14):
        if len(df_close) < period + 1: return pd.Series([np.nan] * len(df_close))
        tr0 = abs(df_high - df_low); tr1 = abs(df_high - df_close.shift()); tr2 = abs(df_low - df_close.shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        atr = tr.ewm(com=period - 1, adjust=False).mean()
        return atr

    def get_technical_indicators(self, coin: str, interval: str) -> Dict[str, Any]:
        """Calculate technical indicators, returning history series"""
        df = self.get_real_time_data(coin, interval=interval)
        if df.empty or len(df) < 50:
            return {'error': f'Not enough data for {coin} {interval} (got {len(df)})'}

        close_prices = df['close']; current_price = close_prices.iloc[-1]; hist_len = self.indicator_history_length
        indicators = {'current_price': current_price}
        try:
            ema_20_series = self.calculate_ema_series(close_prices, 20); ema_50_series = self.calculate_ema_series(close_prices, 50)
            rsi_14_series = self.calculate_rsi_series(close_prices, 14); macd_line_series, macd_signal_series, macd_hist_series = self.calculate_macd_series(close_prices)
            atr_14_series = self.calculate_atr_series(df['high'], df['low'], df['close'], 14)

            indicators['ema_20'] = ema_20_series.iloc[-1]; indicators['ema_50'] = ema_50_series.iloc[-1]
            indicators['rsi_14'] = rsi_14_series.iloc[-1]; indicators['macd'] = macd_line_series.iloc[-1]
            indicators['macd_signal'] = macd_signal_series.iloc[-1]; indicators['macd_histogram'] = macd_hist_series.iloc[-1]
            indicators['atr_14'] = atr_14_series.iloc[-1] # Keep atr_14 available for AI prompt

            # Use .where(pd.notna, None) to convert NaN to None for JSON
            indicators['ema_20_series'] = ema_20_series.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()
            indicators['rsi_14_series'] = rsi_14_series.iloc[-hist_len:].round(3).where(pd.notna, None).tolist()
            indicators['macd_series'] = macd_line_series.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()

            if interval == '3m':
                 rsi_7_series = self.calculate_rsi_series(close_prices, 7)
                 indicators['rsi_7'] = rsi_7_series.iloc[-1]
                 indicators['rsi_7_series'] = rsi_7_series.iloc[-hist_len:].round(3).where(pd.notna, None).tolist()
            if interval == '4h':
                 atr_3_series = self.calculate_atr_series(df['high'], df['low'], df['close'], 3)
                 indicators['atr_3'] = atr_3_series.iloc[-1]

            indicators['volume'] = df['volume'].iloc[-1]; avg_vol = df['volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
            indicators['avg_volume'] = avg_vol if pd.notna(avg_vol) else 0.0
            indicators['price_series'] = close_prices.iloc[-hist_len:].round(4).where(pd.notna, None).tolist()

            for key, value in indicators.items():
                if isinstance(value, float) and np.isnan(value): indicators[key] = None
            return indicators
        except Exception as e:
            print(f"❌ Indicator error {coin} ({interval}): {e}")
            traceback.print_exc()
            return {'current_price': current_price, 'error': str(e)}

    def get_all_real_prices(self) -> Dict[str, float]:
        """Get real prices for all coins from Spot"""
        prices = {}
        for coin in self.available_coins:
            try:
                response = requests.get(f"{self.spot_url}/ticker/price?symbol={coin}USDT", timeout=5)
                response.raise_for_status(); data = response.json(); prices[coin] = float(data['price'])
                print(f"✅ {coin}: ${prices[coin]:.4f}")
            except Exception as e:
                print(f"❌ {coin} price error: {e}. Fallback...")
                df = self.get_real_time_data(coin, interval='1m', limit=1)
                price_val = df['close'].iloc[-1] if not df.empty and not df['close'].empty else None
                if price_val is not None and pd.notna(price_val): prices[coin] = price_val
                else: print(f"   Fallback failed for {coin}. Price set to 0."); prices[coin] = 0.0
        return prices

    def get_market_sentiment(self, coin: str) -> Dict[str, Any]:
        """Get Open Interest and Funding Rate (Nof1ai format)"""
        open_interest = self.get_open_interest(coin)
        funding_rate = self.get_funding_rate(coin)
        
        # Nof1ai format: "Latest: X Average: Y" for Open Interest
        avg_oi = open_interest  # Simplified average calculation
        return {
            'open_interest': open_interest,
            'open_interest_avg': avg_oi,
            'funding_rate': funding_rate
        }

# --- Portfolio Manager Class ---
class PortfolioManager:
    """Manages the portfolio state, positions, and history."""

    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        self.state_file = "portfolio_state.json"; self.history_file = "trade_history.json"
        self.override_file = "manual_override.json"; self.cycle_history_file = "cycle_history.json"
        self.max_cycle_history = 50; self.max_trade_notional_usd = Config.MAX_TRADE_NOTIONAL_USD # Use config value
        self.maintenance_margin_rate = 0.01

        self.current_balance = self.initial_balance; self.positions = {}
        self.trade_history = self.load_trade_history() # Load first
        self.load_state() # Loads balance, positions, trade_count
        self.cycle_history = self.load_cycle_history()
        self.risk_manager = AdvancedRiskManager()  # Initialize risk manager

        self.total_value = self.current_balance
        self.total_return = 0.0
        self.start_time = datetime.now()
        self.portfolio_values_history = [self.initial_balance]  # Track portfolio values for Sharpe ratio
        self.sharpe_ratio = 0.0
        self.update_prices({}) # Calculate initial value with loaded positions

    def load_state(self):
        data = safe_file_read(self.state_file, default_data={})
        self.current_balance = data.get('current_balance', self.initial_balance)
        self.positions = data.get('positions', {})
        self.trade_count = data.get('trade_count', len(self.trade_history)) # Initialize from history if not in state
        print(f"✅ Loaded state ({len(self.positions)} positions, {self.trade_count} closed trades)" if data else "ℹ️ No state file found.")

    def save_state(self):
        data = {'current_balance': self.current_balance, 'positions': self.positions, 'total_value': self.total_value, 'total_return': self.total_return, 'initial_balance': self.initial_balance, 'trade_count': self.trade_count, 'last_updated': datetime.now().isoformat(), 'sharpe_ratio': self.sharpe_ratio}
        safe_file_write(self.state_file, data); print(f"✅ Saved state.")

    def load_trade_history(self) -> List[Dict]:
        history = safe_file_read(self.history_file, default_data=[]); print(f"✅ Loaded {len(history)} trades."); return history
    def save_trade_history(self):
        history_to_save = self.trade_history[-25:]; safe_file_write(self.history_file, history_to_save); print(f"✅ Saved {len(history_to_save)} trades.")
    def add_to_history(self, trade: Dict):
        self.trade_history.append(trade); self.trade_count = len(self.trade_history); self.save_trade_history()

    def load_cycle_history(self) -> List[Dict]:
        history = safe_file_read(self.cycle_history_file, default_data=[]); print(f"✅ Loaded {len(history)} cycles."); return history
    def add_to_cycle_history(self, cycle_number: int, prompt: str, thoughts: str, decisions: Dict):
        cycle_data = {'cycle': cycle_number, 'timestamp': datetime.now().isoformat(), 'user_prompt_summary': prompt[:300] + "..." if len(prompt) > 300 else prompt, 'chain_of_thoughts': thoughts, 'decisions': decisions}
        self.cycle_history.append(cycle_data); self.cycle_history = self.cycle_history[-self.max_cycle_history:]
        safe_file_write(self.cycle_history_file, self.cycle_history); print(f"✅ Saved cycle {cycle_number} (Total: {len(self.cycle_history)})")

    def update_prices(self, new_prices: Dict[str, float]):
        """Updates prices and recalculates total value."""
        total_unrealized_pnl = 0.0
        for coin, price in new_prices.items():
            if coin in self.positions and isinstance(price, (int, float)) and price > 0:
                pos = self.positions[coin]; pos['current_price'] = price
                entry = pos['entry_price']; qty = pos['quantity']; direction = pos.get('direction', 'long')
                pnl = (price - entry) * qty if direction == 'long' else (entry - price) * qty
                pos['unrealized_pnl'] = pnl; total_unrealized_pnl += pnl
            elif coin in self.positions: print(f"⚠️ Invalid price for {coin}: {price}. PnL skip.")

        # Calculate total value = cash + sum(margin + pnl for each position)
        current_portfolio_value = self.current_balance
        for pos in self.positions.values():
            current_portfolio_value += pos.get('margin_usd', 0) + pos.get('unrealized_pnl', 0)
        self.total_value = current_portfolio_value

        if self.initial_balance > 0: self.total_return = ((self.total_value - self.initial_balance) / self.initial_balance) * 100
        else: self.total_return = 0.0
        
        # Update portfolio history for Sharpe ratio calculation
        self.portfolio_values_history.append(self.total_value)
        if len(self.portfolio_values_history) > 100:  # Keep last 100 values
            self.portfolio_values_history = self.portfolio_values_history[-100:]
        
        # Calculate Sharpe ratio
        self.sharpe_ratio = self.calculate_sharpe_ratio()
        
        # Save updated state with Sharpe ratio
        self.save_state()
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio based on portfolio value history (Nof1ai blog style)."""
        if len(self.portfolio_values_history) < 2:
            return 0.0
        
        try:
            # Calculate simple returns (percentage changes)
            returns = []
            for i in range(1, len(self.portfolio_values_history)):
                if self.portfolio_values_history[i-1] > 0:
                    ret = (self.portfolio_values_history[i] - self.portfolio_values_history[i-1]) / self.portfolio_values_history[i-1]
                    returns.append(ret)
            
            if len(returns) < 2:
                return 0.0
            
            # Nof1ai style: Simple Sharpe ratio with 0% risk-free rate
            # Daily Sharpe ratio (assuming 2-minute cycles = 720 cycles per day)
            risk_free_rate = 0.0
            
            # Calculate excess returns
            excess_returns = [r - risk_free_rate for r in returns]
            
            # Daily return and volatility
            avg_return = np.mean(excess_returns) * 720  # Daily return (720 cycles per day)
            std_return = np.std(excess_returns) * np.sqrt(720)  # Daily volatility
            
            if std_return == 0:
                return 0.0
            
            # Daily Sharpe ratio
            sharpe = avg_return / std_return
            
            # Return as float (not annualized for simplicity)
            return float(sharpe)
            
        except Exception as e:
            print(f"⚠️ Sharpe ratio calculation error: {e}")
            return 0.0
    
    def calculate_risk_usd(self, entry_price: float, stop_loss: float, quantity: float, direction: str) -> float:
        """Calculate risk in USD for a position."""
        if direction == 'long':
            risk_per_unit = entry_price - stop_loss
        else:
            risk_per_unit = stop_loss - entry_price
        
        return abs(risk_per_unit * quantity)

    def get_manual_override(self) -> Dict:
        """Checks for and deletes the manual override file."""
        override_data = safe_file_read(self.override_file, default_data={})
        if override_data:
            print(f"🔔 MANUAL OVERRIDE DETECTED: {override_data}")
            try: os.remove(self.override_file); print(f"ℹ️ Override file deleted.")
            except OSError as e: print(f"❌ Could not delete override file: {e}")
        return override_data

    def _estimate_liquidation_price(self, entry_price: float, leverage: int, direction: str) -> float:
        """Estimate liquidation price."""
        if leverage <= 1 or entry_price <= 0: return 0.0
        imr = 1.0 / leverage; mmr = self.maintenance_margin_rate; margin_diff = imr - mmr
        if margin_diff <= 0: print(f"⚠️ Liq est. failed: margin diff <= 0 ({margin_diff})."); return 0.0
        liq_price = entry_price * (1 - margin_diff) if direction == 'long' else entry_price * (1 + margin_diff)
        return max(0.0, liq_price)

    # --- NEW: Auto TP/SL Check ---
    def check_and_execute_tp_sl(self, current_prices: Dict[str, float]):
        """Checks if any open position hit TP or SL and closes them automatically."""
        print("🔎 Checking for TP/SL triggers...")
        closed_positions = [] # Keep track of positions closed in this check
        for coin, position in list(self.positions.items()): # Iterate over a copy for safe deletion
            if coin not in current_prices or not isinstance(current_prices[coin], (int, float)) or current_prices[coin] <= 0:
                continue # Skip if price is invalid

            current_price = current_prices[coin]
            exit_plan = position.get('exit_plan', {})
            tp = exit_plan.get('profit_target')
            sl = exit_plan.get('stop_loss')
            direction = position.get('direction', 'long')
            entry_price = position['entry_price']
            margin_used = position.get('margin_usd', position.get('notional_usd', 0) / position.get('leverage', 1))
            quantity = position['quantity']

            close_reason = None

            # Check TP
            # Convert tp/sl to float for safe comparison, handle potential errors
            try: tp = float(tp) if tp is not None else None
            except (ValueError, TypeError): tp = None
            try: sl = float(sl) if sl is not None else None
            except (ValueError, TypeError): sl = None

            if tp is not None:
                if direction == 'long' and current_price >= tp: close_reason = f"Profit Target ({tp}) hit"
                elif direction == 'short' and current_price <= tp: close_reason = f"Profit Target ({tp}) hit"

            # Check SL (only if TP not hit)
            if close_reason is None and sl is not None:
                if direction == 'long' and current_price <= sl: close_reason = f"Stop Loss ({sl}) hit"
                elif direction == 'short' and current_price >= sl: close_reason = f"Stop Loss ({sl}) hit"

            # Execute Close if triggered
            if close_reason:
                print(f"⚡ AUTO-CLOSE {coin} ({direction}): {close_reason} at price ${format_num(current_price, 4)}")

                if direction == 'long': profit = (current_price - entry_price) * quantity
                else: profit = (entry_price - current_price) * quantity

                self.current_balance += (margin_used + profit) # Return margin + PnL

                print(f"   Closed PnL: ${format_num(profit, 2)}")

                history_entry = {
                    "symbol": coin, "direction": direction, "entry_price": entry_price, "exit_price": current_price,
                    "quantity": quantity, "notional_usd": position.get('notional_usd', 'N/A'), "pnl": profit,
                    "entry_time": position['entry_time'], "exit_time": datetime.now().isoformat(),
                    "leverage": position.get('leverage', 'N/A'), "close_reason": close_reason # Add reason
                }
                self.add_to_history(history_entry) # This increments trade_count
                closed_positions.append(coin)
                del self.positions[coin] # Remove from active positions

        if closed_positions:
             print(f"✅ Auto-closed positions: {', '.join(closed_positions)}")
             return True # Indicate that positions were closed
        else:
             print("   No TP/SL triggers found.")
             return False

    def execute_decision(self, decisions: Dict[str, Dict], current_prices: Dict[str, float]):
        """Executes trading decisions from AI, incorporating dynamic sizing."""
        print("\n⚡ EXECUTING AI DECISIONS...")
        if not isinstance(decisions, dict): print(f"❌ Invalid decisions format: {type(decisions)}"); return

        for coin, trade in decisions.items():
            if not isinstance(trade, dict): print(f"⚠️ Invalid trade data for {coin}: {type(trade)}"); continue
            if coin not in current_prices or not isinstance(current_prices[coin], (int, float)) or current_prices[coin] <= 0:
                print(f"⚠️ Skipping {coin}: Invalid price data."); continue

            signal = trade.get('signal'); current_price = current_prices[coin]; position = self.positions.get(coin)

            if signal == 'buy_to_enter' or signal == 'sell_to_enter':
                if position: print(f"⚠️ {signal.upper()} {coin}: Position already open."); continue

                confidence = trade.get('confidence', 0.5) # Default 50% confidence if missing
                leverage = trade.get('leverage', 1)
                # Ensure confidence and leverage are valid numbers
                try: confidence = float(confidence); leverage = int(leverage)
                except (ValueError, TypeError): print(f"⚠️ Invalid confidence ({confidence}) or leverage ({leverage}) for {coin}. Skipping."); continue
                if leverage < 1: leverage = 1
                # Enforce maximum leverage limit of 10x
                if leverage > 10: 
                    print(f"⚠️ Leverage {leverage}x exceeds maximum limit of 10x. Reducing to 10x.")
                    leverage = 10
                if not (0.0 <= confidence <= 1.0): confidence = 0.5 # Clamp confidence to 0.0-1.0

                # --- Advanced Risk Management ---
                stop_loss = trade.get('stop_loss')
                try:
                    stop_loss = float(stop_loss) if stop_loss is not None else None
                except (ValueError, TypeError):
                    stop_loss = None
                
                # Use AI's proposed position size directly (quantity_usd is margin)
                # Convert margin to notional using leverage
                proposed_margin = trade.get('quantity_usd', 0)
                if proposed_margin > 0:
                    calculated_notional_usd = proposed_margin * leverage
                    print(f"   AI proposed sizing: ${calculated_notional_usd:.2f} notional (${proposed_margin:.2f} margin)")
                else:
                    # Fallback to dynamic sizing based on confidence and risk level
                    from config import Config
                    risk_level = Config.RISK_LEVEL.lower()
                    
                    if risk_level == 'low':
                        min_notional = 50.0
                    elif risk_level == 'high':
                        min_notional = 100.0
                    else:  # medium
                        min_notional = 75.0
                        
                    max_notional = self.max_trade_notional_usd
                    calculated_notional_usd = min_notional + (max_notional - min_notional) * confidence
                    calculated_notional_usd = min(calculated_notional_usd, self.max_trade_notional_usd)
                    print(f"   Confidence-based sizing: ${calculated_notional_usd:.2f} notional (min: ${min_notional})")
                
                # Check risk management constraints
                risk_decision = self.risk_manager.should_enter_trade(
                    symbol=coin,
                    current_positions=self.positions,
                    current_prices=current_prices,
                    confidence=confidence,
                    proposed_notional=calculated_notional_usd
                )
                
                if not risk_decision['should_enter']:
                    print(f"⚠️ Risk management blocked trade: {risk_decision['reason']}")
                    continue
                
                if risk_decision['adjusted_notional'] != calculated_notional_usd:
                    print(f"   Risk-adjusted notional: ${risk_decision['adjusted_notional']:.2f}")
                    calculated_notional_usd = risk_decision['adjusted_notional']
                
                notional_usd = calculated_notional_usd
                margin_usd = notional_usd / leverage # Margin required

                if margin_usd <= 0: print(f"⚠️ {signal.upper()} {coin}: Calculated margin is zero/negative. Skipping."); continue
                if margin_usd > self.current_balance: print(f"⚠️ {signal.upper()} {coin}: Not enough cash for margin (${margin_usd:.2f} > ${self.current_balance:.2f})"); continue

                quantity_coin = notional_usd / current_price
                self.current_balance -= margin_usd # Deduct margin

                direction = 'long' if signal == 'buy_to_enter' else 'short'
                estimated_liq_price = self._estimate_liquidation_price(current_price, leverage, direction)

                self.positions[coin] = {
                    'symbol': coin, 'direction': direction, 'quantity': quantity_coin, 'entry_price': current_price,
                    'entry_time': datetime.now().isoformat(), 'current_price': current_price, 'unrealized_pnl': 0.0,
                    'notional_usd': notional_usd, 'margin_usd': margin_usd, 'leverage': leverage,
                    'liquidation_price': estimated_liq_price, 'confidence': confidence,
                    'exit_plan': { 'profit_target': trade.get('profit_target'), 'stop_loss': trade.get('stop_loss'), 'invalidation_condition': trade.get('invalidation_condition') },
                    'risk_usd': trade.get('risk_usd')
                }
                print(f"✅ {signal.upper()}: Opened {direction} {coin} ({format_num(quantity_coin, 4)} @ ${format_num(current_price, 4)} / Notional ${format_num(notional_usd, 2)} / Margin ${format_num(margin_usd, 2)} / Est. Liq: ${format_num(estimated_liq_price, 4)})")

            elif signal == 'close_position':
                if not position: print(f"⚠️ CLOSE {coin}: No position to close."); continue

                sell_quantity = position['quantity']; direction = position.get('direction', 'long')
                entry_price = position['entry_price']
                margin_used = position.get('margin_usd', position.get('notional_usd', 0) / position.get('leverage', 1))

                profit = (current_price - entry_price) * sell_quantity if direction == 'long' else (entry_price - current_price) * sell_quantity
                self.current_balance += (margin_used + profit)

                print(f"✅ CLOSE (AI): Closed {direction} {coin} @ ${format_num(current_price, 4)} (PnL: ${format_num(profit, 2)})")

                history_entry = {
                    "symbol": coin, "direction": direction, "entry_price": entry_price, "exit_price": current_price,
                    "quantity": position['quantity'], "notional_usd": position.get('notional_usd', 'N/A'), "pnl": profit,
                    "entry_time": position['entry_time'], "exit_time": datetime.now().isoformat(),
                    "leverage": position.get('leverage', 'N/A'), "close_reason": f"AI Decision: {trade.get('justification', 'N/A')}"
                }
                self.add_to_history(history_entry)
                del self.positions[coin]

            elif signal == 'hold':
                 print(f"ℹ️ HOLD: {'Holding ' + position.get('direction', 'long') if position else 'Staying cash in'} {coin}.")
            else: print(f"⚠️ Unknown signal '{signal}' for {coin}. Skipping.")


# --- Main Bot Class ---
# format_num is global

class AlphaArenaDeepSeek:
    """Alpha Arena-like DeepSeek integration with auto TP/SL, dynamic sizing, and advanced risk management."""

    def __init__(self, api_key: str = None):
        self.market_data = RealMarketData()
        self.portfolio = PortfolioManager()
        self.deepseek = DeepSeekAPI(api_key)
        self.risk_manager = AdvancedRiskManager()
        self.invocation_count = 0 # Track AI calls since bot start

    def generate_alpha_arena_prompt(self) -> str:
        """Generate prompt with enhanced data, indicator history"""
        current_time = datetime.now(); minutes_running = int((current_time - self.portfolio.start_time).total_seconds() / 60)
        # Use internal invocation counter, don't increment here, do it in run_cycle
        # self.invocation_count += 1

        prompt = f"""
USER_PROMPT:
It has been {minutes_running} minutes since you started trading. The current time is {current_time} and you've been invoked {self.invocation_count} times. Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. Below that is your current account information, value, performance, positions, etc.

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST
Timeframes note: Unless stated otherwise in a section title, intraday series are provided at 3‑minute intervals. If a coin uses a different interval, it is explicitly stated in that coin’s section.
"""

        def format_list(lst, precision=4):
            if not isinstance(lst, list): return []
            return [format_num(x, precision) if x is not None else 'N/A' for x in lst]

        # --- Loop through available coins ---
        for coin in self.market_data.available_coins:
            prompt += f"\n{'='*20} ALL {coin} DATA {'='*20}\n"
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            indicators_4h = self.market_data.get_technical_indicators(coin, '4h')
            sentiment = self.market_data.get_market_sentiment(coin)

            prompt += f"--- Market Sentiment for {coin} Perps ---\n"
            prompt += f"Open Interest: Latest: {format_num(sentiment.get('open_interest', 'N/A'), 2)}\n"
            funding_rate = sentiment.get('funding_rate', 0.0)
            prompt += f"Funding Rate: {format_num(funding_rate, 8)} ({format_num(funding_rate*100, 4)}%)\n\n"

            # --- Inner function to format indicators ---
            def format_indicators(indicators, prefix=""):
                if not isinstance(indicators, dict) or 'error' in indicators:
                     error_msg = indicators.get('error', 'Unknown error') if isinstance(indicators, dict) else 'Invalid indicator data'
                     return f"{prefix}Error fetching indicator data: {error_msg}\n"
                # Format numbers using global helper
                output = f"{prefix}current_price = {format_num(indicators.get('current_price', 'N/A'))}\n"
                output += f"{prefix}Mid prices (last {len(indicators.get('price_series',[]))}): {format_list(indicators.get('price_series',[]))}\n"
                output += f"{prefix}EMA indicators (20‑period): {format_list(indicators.get('ema_20_series',[]))}\n"
                if 'rsi_7_series' in indicators: output += f"{prefix}RSI indicators (7‑Period): {format_list(indicators.get('rsi_7_series',[]), precision=3)}\n"
                output += f"{prefix}RSI indicators (14‑Period): {format_list(indicators.get('rsi_14_series',[]), precision=3)}\n"
                output += f"{prefix}MACD indicators: {format_list(indicators.get('macd_series',[]))}\n"
                atr_3 = indicators.get('atr_3'); atr_14 = indicators.get('atr_14'); atr_str = ""
                if atr_3 is not None and pd.notna(atr_3): atr_str += f"{prefix}3‑Period ATR: {format_num(atr_3)} vs "
                atr_str += f"14‑Period ATR: {format_num(atr_14)}\n"; output += atr_str
                output += f"{prefix}Current Volume: {format_num(indicators.get('volume', 'N/A'), 3)} vs. Average Volume: {format_num(indicators.get('avg_volume', 'N/A'), 3)}\n"
                return output
            # --- End inner function ---

            prompt += "--- Intraday series (3‑minute intervals) ---\n"; prompt += format_indicators(indicators_3m)
            prompt += "\n--- Longer‑term context (4‑hour timeframe) ---\n"; prompt += format_indicators(indicators_4h)

            # --- Add current position details if open ---
            if coin in self.portfolio.positions:
                position = self.portfolio.positions[coin]; prompt += "\n--- CURRENT OPEN POSITION & YOUR PLAN ---\n"
                prompt += f"You have an open {position.get('direction', 'long').upper()} position.\n"; prompt += f"  Symbol: {position.get('symbol', 'N/A')}\n"
                prompt += f"  Quantity: {format_num(position.get('quantity', 0), 6)}\n"; prompt += f"  Entry Price: ${format_num(position.get('entry_price', 0))}\n"
                prompt += f"  Current Price: ${format_num(position.get('current_price', 0))}\n"; prompt += f"  Liquidation Price (Est.): ${format_num(position.get('liquidation_price', 0))}\n"
                prompt += f"  Unrealized PnL: ${format_num(position.get('unrealized_pnl', 0), 2)}\n"; prompt += f"  Leverage: {position.get('leverage', 1)}x\n"
                prompt += f"  Notional Value: ${format_num(position.get('notional_usd', 0), 2)}\n"
                exit_plan = position.get('exit_plan', {}); prompt += f"  YOUR ACTIVE EXIT PLAN:\n"
                prompt += f"    Profit Target: {exit_plan.get('profit_target', 'N/A')}\n"; prompt += f"    Stop Loss: {exit_plan.get('stop_loss', 'N/A')}\n"
                prompt += f"    Invalidation: {exit_plan.get('invalidation_condition', 'N/A')}\n"; prompt += f"  Your Confidence: {position.get('confidence', 'N/A')}\n"
                prompt += f"  Estimated Risk USD: {position.get('risk_usd', 'N/A')}\n"; prompt += "REMINDER: You can only 'hold' or 'close_position'.\n"
        # --- End coin loop ---


        prompt += f"""
{'='*20} HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE {'='*20}

Current Total Return (percent): {format_num(self.portfolio.total_return, 2)}%
Available Cash: {format_num(self.portfolio.current_balance, 2)}
Current Account Value: {format_num(self.portfolio.total_value, 2)}
Sharpe Ratio: {format_num(self.portfolio.sharpe_ratio, 3)}

Current live positions & performance:"""

        if not self.portfolio.positions: 
            prompt += " No open positions. (100% cash)"
        else:
            for coin, pos in self.portfolio.positions.items():
                prompt += f"\n{{'symbol': '{coin}', 'quantity': {format_num(pos.get('quantity', 0), 4)}, 'entry_price': {format_num(pos.get('entry_price', 0))}, 'current_price': {format_num(pos.get('current_price', 0))}, 'liquidation_price': {format_num(pos.get('liquidation_price', 0))}, 'unrealized_pnl': {format_num(pos.get('unrealized_pnl', 0), 2)}, 'leverage': {pos.get('leverage', 1)}, 'exit_plan': {json.dumps(pos.get('exit_plan', {}))}, 'confidence': {pos.get('confidence', 0.5)}, 'risk_usd': {pos.get('risk_usd', 'N/A')}, 'sl_oid': -1, 'tp_oid': -1, 'wait_for_fill': False, 'entry_oid': -1, 'notional_usd': {format_num(pos.get('notional_usd', 0), 2)}}}"
        return prompt

    def parse_ai_response(self, response: str) -> Dict:
        """Parse the AI response"""
        thoughts = "Error: Could not parse AI response."; decisions = {}
        try:
            parts = response.split("DECISIONS", 1)
            if len(parts) == 2: thoughts_raw = parts[0].replace("CHAIN_OF_THOUGHTS", "").strip(); json_part = parts[1].strip(); thoughts = thoughts_raw if thoughts_raw else "No thoughts."
            else: print("⚠️ AI response keywords missing."); json_part = response; thoughts = "Keywords missing."
            start_idx = json_part.find('{'); end_idx = json_part.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = json_part[start_idx:end_idx]
                try:
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, dict): decisions = parsed_json
                    else: print(f"❌ Parsed JSON not dict: {type(parsed_json)}"); thoughts += f"\nError: Parsed JSON not dict."
                except json.JSONDecodeError as json_e: print(f"❌ JSON decode error: {json_e}"); print(f"   Invalid JSON: {json_str}"); thoughts = f"JSON Error: {json_e}\nRaw:\n{response}"
            else: print("⚠️ No valid JSON block found."); thoughts += f"\nError: No JSON block."
        except Exception as e: print(f"❌ General parse error: {e}"); thoughts = f"Parse Error: {e}\nRaw:\n{response}"
        return {"chain_of_thoughts": thoughts, "decisions": decisions}

    def run_trading_cycle(self, cycle_number: int):
        """Run a single trading cycle with auto TP/SL"""
        print(f"\n{'='*80}\n🔄 TRADING CYCLE {cycle_number} | ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}")
        prompt, thoughts, decisions = "N/A", "N/A", {}
        try:
            print("\n📊 FETCHING MARKET DATA...")
            real_prices = self.market_data.get_all_real_prices()
            valid_prices = {k: v for k, v in real_prices.items() if isinstance(v, (int, float)) and v > 0}
            if not valid_prices: raise ValueError("No valid market prices received.")
            self.portfolio.update_prices(valid_prices) # Update PnL before checking TP/SL

            # --- Auto TP/SL Check ---
            positions_closed_by_tp_sl = self.portfolio.check_and_execute_tp_sl(valid_prices)
            # --- End Auto TP/SL Check ---

            manual_override = self.portfolio.get_manual_override()

            if manual_override:
                print("🔔 APPLYING MANUAL OVERRIDE...")
                decisions = manual_override.get('decisions', {}); thoughts = "Manual override."; prompt = "N/A (Manual)"
                print("\n🎯 MANUAL DECISIONS:", json.dumps(decisions, indent=2))
            # Only ask AI if no TP/SL triggered AND no manual override
            elif not positions_closed_by_tp_sl:
                print("\n🤖 GENERATING PROMPT...")
                self.invocation_count += 1 # Increment AI call count
                prompt = self.generate_alpha_arena_prompt()
                print("📋 USER PROMPT (summary): " + prompt[:200] + "...")

                print("\n💭 DEEPSEEK ANALYZING...")
                ai_response = self.deepseek.get_ai_decision(prompt)
                parsed_response = self.parse_ai_response(ai_response)
                thoughts = parsed_response.get("chain_of_thoughts", "Parse Error.")
                decisions = parsed_response.get("decisions", {})

                if not isinstance(decisions, dict):
                    print(f"❌ AI decisions not dict ({type(decisions)}). Resetting."); thoughts += f"\nError: Decisions not dict."; decisions = {}

                print("\n🔍 CHAIN_OF_THOUGHTS:\n", thoughts)
                print("\n🎯 AI TRADING DECISIONS:", json.dumps(decisions, indent=2) if decisions else "{}")
            else:
                 print("ℹ️ Skipping AI analysis due to auto TP/SL closure.")


            # Execute AI decisions only if it's a valid dict and NOT empty AND no manual override was active
            if isinstance(decisions, dict) and decisions and not manual_override:
                # Filter out decisions for coins that might have been closed by TP/SL in the same cycle
                decisions_to_execute = {}
                for coin, trade in decisions.items():
                     if coin in self.portfolio.positions or trade.get('signal') in ['buy_to_enter', 'sell_to_enter']:
                          # Only execute if the position still exists OR it's a new entry signal
                           decisions_to_execute[coin] = trade
                     else:
                           print(f"ℹ️ Filtering AI decision for {coin} as it was likely closed by TP/SL this cycle.")


                # decisions_to_execute = {k: v for k, v in decisions.items() if k not in self.portfolio.positions or self.positions.get(k)} # Old filter might be wrong
                if decisions_to_execute:
                     self.portfolio.execute_decision(decisions_to_execute, valid_prices) # Execute filtered decisions

            # Execute manual override decisions if present
            elif isinstance(decisions, dict) and decisions and manual_override:
                 self.portfolio.execute_decision(decisions, valid_prices)

            elif isinstance(decisions, dict): print("ℹ️ No AI/Manual trading actions to execute this cycle.")

            # Save state and history at the end of the cycle
            self.portfolio.save_state()
            # Log regardless of errors (log contains error info if applicable)
            self.portfolio.add_to_cycle_history(cycle_number, prompt, thoughts, decisions)
            self.show_status()

        except Exception as e:
            print(f"❌ CRITICAL CYCLE ERROR: {e}"); traceback.print_exc()
            try:
                 decisions_log = decisions if isinstance(decisions, dict) else {}
                 self.portfolio.add_to_cycle_history(cycle_number, prompt, f"CRITICAL CYCLE ERROR: {e}\n{traceback.format_exc()}", decisions_log)
            except Exception as log_e: print(f"❌ Failed to save error to cycle history: {log_e}")

    def show_status(self):
        """Show current status in the console"""
        print(f"\n📊 CURRENT STATUS:")
        print(f"💰 Total Value: ${format_num(self.portfolio.total_value, 2)} (Initial: ${format_num(self.portfolio.initial_balance, 2)})")
        print(f"📈 Total Return: {format_num(self.portfolio.total_return, 2)}%")
        print(f"💵 Available Cash: ${format_num(self.portfolio.current_balance, 2)}")
        print(f"🔄 Total Closed Trades: {self.portfolio.trade_count}")
        print(f"\n📦 CURRENT POSITIONS ({len(self.portfolio.positions)} open):")
        if not self.portfolio.positions: print("  No open positions.")
        else:
            for coin, pos in self.portfolio.positions.items():
                pnl = pos.get('unrealized_pnl', 0.0); pnl_sign = "+" if pnl >= 0 else ""
                direction = pos.get('direction', 'long').upper(); leverage = pos.get('leverage', 1)
                notional = pos.get('notional_usd', 0.0); liq = pos.get('liquidation_price', 0.0)
                entry = pos.get('entry_price', 0.0); qty = pos.get('quantity', 0.0)
                print(f"  {coin} ({direction} {leverage}x): {format_num(qty, 4)} units | Notional ${format_num(notional, 2)} | Entry: ${format_num(entry, 4)} | PnL: {pnl_sign}${format_num(pnl, 2)} | Liq Est: ${format_num(liq, 4)}")

    def run_simulation(self, total_duration_hours: int = 168, cycle_interval_minutes: int = 2):
        """Run the simulation"""
        print(f"🚀 ALPHA ARENA - DEEPSEEK INTEGRATION (V{VERSION})")
        print(f"💡 Simulating with ${format_num(self.portfolio.initial_balance, 2)} budget for {total_duration_hours} hours.")
        print(f"   Trading: {', '.join(self.market_data.available_coins)}")
        print(f"   State File: {self.portfolio.state_file}")
        print(f"   Trade History File: {self.portfolio.history_file}")
        print(f"   Cycle History File: {self.portfolio.cycle_history_file}")
        print(f"   Override File Check: {self.portfolio.override_file}")
        print(f"   Cycle Interval: {cycle_interval_minutes} minutes")

        end_time = datetime.now() + timedelta(hours=total_duration_hours)
        start_cycle = len(self.portfolio.cycle_history) + 1
        print(f"   Resuming from Cycle {start_cycle}...")
        self.invocation_count = start_cycle - 1; current_cycle_number = start_cycle - 1

        while datetime.now() < end_time:
            current_cycle_number += 1; cycle_start_time = time.time()
            self.run_trading_cycle(current_cycle_number)
            if datetime.now() >= end_time: break
            elapsed_time = time.time() - cycle_start_time
            sleep_time = max(0, (cycle_interval_minutes * 60) - elapsed_time)
            print(f"\n⏳ Cycle {current_cycle_number} complete in {format_num(elapsed_time,2)}s. Next cycle in {format_num(sleep_time/60, 2)} mins... (Ctrl+C to stop)")
            time.sleep(max(sleep_time, 0.5))

        print(f"\n{'='*80}\n🏁 SIMULATION COMPLETED\n{'='*80}"); self.show_status()

# Define VERSION at the top level
VERSION = "9 - Auto TP/SL, Dynamic Size, Prompt Eng"

def main():
    """Main application entry point"""
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key: print("⚠️ No DEEPSEEK_API_KEY found. Running simulation mode...");
        arena = AlphaArenaDeepSeek(api_key)
        arena.run_simulation(total_duration_hours=168, cycle_interval_minutes=2)
    except KeyboardInterrupt: print("\n⏹️ Program stopped by user.")
    except Exception as e: print(f"\n❌ Unexpected critical error in main: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main()
