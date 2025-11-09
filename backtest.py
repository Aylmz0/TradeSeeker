"""
Backtesting module for the Alpha Arena DeepSeek bot.
Allows testing trading strategies on historical data.
"""
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from config import Config
from utils import format_num, safe_file_write, safe_file_read

class BacktestEngine:
    """Backtesting engine for historical strategy testing."""
    
    def __init__(self, initial_balance: float = 200.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.portfolio_values = []
        self.timestamps = []
        
    def load_historical_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1h') -> pd.DataFrame:
        """
        Load historical price data for backtesting.
        Note: This is a simplified implementation. In production, you would use
        a proper historical data source like Binance API, Yahoo Finance, etc.
        """
        # This is a placeholder implementation
        # In a real implementation, you would fetch historical data from an API
        logging.info(f"Loading historical data for {symbol} from {start_date} to {end_date}")
        
        # For demonstration, create synthetic data
        dates = pd.date_range(start=start_date, end=end_date, freq=interval)
        np.random.seed(42)  # For reproducible results
        
        # Generate synthetic price data
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.normal(1000000, 200000, len(dates))
        })
        
        return df
    
    def execute_trade(self, symbol: str, signal: str, price: float, quantity: float, 
                     leverage: int = 1, timestamp: datetime = None) -> Dict[str, Any]:
        """Execute a trade in the backtest environment."""
        if timestamp is None:
            timestamp = datetime.now()
            
        trade_result = {
            'symbol': symbol,
            'signal': signal,
            'price': price,
            'quantity': quantity,
            'leverage': leverage,
            'timestamp': timestamp.isoformat(),
            'success': False
        }
        
        if signal == 'buy_to_enter' or signal == 'sell_to_enter':
            if symbol in self.positions:
                logging.warning(f"Position already exists for {symbol}")
                return trade_result
            
            notional_usd = quantity * price
            margin_usd = notional_usd / leverage
            
            if margin_usd > self.current_balance:
                logging.warning(f"Insufficient balance for {symbol} trade")
                return trade_result
            
            self.current_balance -= margin_usd
            direction = 'long' if signal == 'buy_to_enter' else 'short'
            
            self.positions[symbol] = {
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'entry_price': price,
                'entry_time': timestamp.isoformat(),
                'notional_usd': notional_usd,
                'margin_usd': margin_usd,
                'leverage': leverage
            }
            
            trade_result['success'] = True
            logging.info(f"Opened {direction} position for {symbol} at ${price:.4f}")
            
        elif signal == 'close_position':
            if symbol not in self.positions:
                logging.warning(f"No position to close for {symbol}")
                return trade_result
            
            position = self.positions[symbol]
            entry_price = position['entry_price']
            direction = position['direction']
            margin_used = position['margin_usd']
            
            if direction == 'long':
                profit = (price - entry_price) * position['quantity']
            else:
                profit = (entry_price - price) * position['quantity']
            
            self.current_balance += (margin_used + profit)
            
            trade_history_entry = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': price,
                'quantity': position['quantity'],
                'notional_usd': position['notional_usd'],
                'pnl': profit,
                'entry_time': position['entry_time'],
                'exit_time': timestamp.isoformat(),
                'leverage': leverage,
                'close_reason': 'Backtest close'
            }
            
            self.trade_history.append(trade_history_entry)
            del self.positions[symbol]
            
            trade_result['success'] = True
            trade_result['pnl'] = profit
            logging.info(f"Closed {direction} position for {symbol}: PnL ${profit:.2f}")
        
        return trade_result
    
    def update_portfolio_value(self, current_prices: Dict[str, float], timestamp: datetime):
        """Update portfolio value based on current prices."""
        total_value = self.current_balance
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                entry_price = position['entry_price']
                quantity = position['quantity']
                direction = position['direction']
                
                if direction == 'long':
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:
                    unrealized_pnl = (entry_price - current_price) * quantity
                
                total_value += position['margin_usd'] + unrealized_pnl
        
        self.portfolio_values.append(total_value)
        self.timestamps.append(timestamp)
        
        return total_value
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics."""
        if not self.portfolio_values:
            return {}
        
        initial_value = self.initial_balance
        final_value = self.portfolio_values[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            returns.append(daily_return)
        
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        peak = self.portfolio_values[0]
        max_drawdown = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Trade statistics
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        metrics = {
            'initial_balance': initial_value,
            'final_balance': final_value,
            'total_return_percent': total_return,
            'total_return_usd': final_value - initial_value,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_percent': max_drawdown * 100,
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_percent': win_rate * 100,
            'avg_win_usd': avg_win,
            'avg_loss_usd': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }
        
        return metrics
    
    def run_backtest(self, strategy_func: callable, symbols: List[str], 
                    start_date: str, end_date: str, interval: str = '1h') -> Dict[str, Any]:
        """
        Run a complete backtest with the given strategy.
        
        Args:
            strategy_func: Function that takes (symbol, data, portfolio_state) and returns trading signals
            symbols: List of symbols to backtest
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1h', '4h', '1d', etc.)
        """
        logging.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Load historical data for all symbols
        historical_data = {}
        for symbol in symbols:
            df = self.load_historical_data(symbol, start_date, end_date, interval)
            historical_data[symbol] = df
        
        # Run backtest for each time period
        for i in range(len(historical_data[symbols[0]])):
            current_timestamp = historical_data[symbols[0]].iloc[i]['timestamp']
            current_prices = {}
            
            # Get current prices for all symbols
            for symbol in symbols:
                current_prices[symbol] = historical_data[symbol].iloc[i]['close']
            
            # Update portfolio value
            self.update_portfolio_value(current_prices, current_timestamp)
            
            # Get trading signals from strategy
            for symbol in symbols:
                data_slice = historical_data[symbol].iloc[:i+1]  # All data up to current point
                portfolio_state = {
                    'current_balance': self.current_balance,
                    'positions': self.positions.copy(),
                    'portfolio_value': self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
                }
                
                signal = strategy_func(symbol, data_slice, portfolio_state)
                
                if signal and signal.get('signal') in ['buy_to_enter', 'sell_to_enter', 'close_position']:
                    self.execute_trade(
                        symbol=symbol,
                        signal=signal['signal'],
                        price=current_prices[symbol],
                        quantity=signal.get('quantity', 1),
                        leverage=signal.get('leverage', 1),
                        timestamp=current_timestamp
                    )
        
        # Calculate final metrics
        metrics = self.calculate_metrics()
        
        logging.info(f"Backtest completed. Total return: {metrics.get('total_return_percent', 0):.2f}%")
        
        return {
            'metrics': metrics,
            'trade_history': self.trade_history,
            'portfolio_values': self.portfolio_values,
            'timestamps': [ts.isoformat() for ts in self.timestamps]
        }


class AdvancedRiskManager:
    """Advanced risk management with portfolio diversification and position sizing."""
    
    def __init__(self, max_portfolio_risk: float = None, max_position_risk: float = None):
        # Set risk parameters based on configuration
        from config import Config
        
        risk_level = Config.RISK_LEVEL.lower()
        
        if risk_level == 'low':
            # Low: Yavaş ve kararlı kazanç - muhafazakar
            self.max_portfolio_risk = 0.015  # 1.5% portfolio risk
            self.max_position_risk = 0.008   # 0.8% position risk
            self.max_position_size_usd = 30.0  # $30 maximum per position
        elif risk_level == 'high':
            # High: Çok riskli - agresif
            self.max_portfolio_risk = 0.15   # 15% portfolio risk
            self.max_position_risk = 0.10    # 10% position risk
            self.max_position_size_usd = 80.0  # $80 maximum per position
        else:  # medium (default)
            # Medium: Nof1ai mantığı - agresif ama kontrollü
            # $50 maximum per position with flexible sizing
            self.max_portfolio_risk = 0.08   # 8% portfolio risk (daha muhafazakar)
            self.max_position_risk = 0.04    # 4% position risk (daha muhafazakar)
            self.max_position_size_usd = 50.0  # $50 maximum per position
        
        # Override with custom values if provided
        if max_portfolio_risk is not None:
            self.max_portfolio_risk = max_portfolio_risk
        if max_position_risk is not None:
            self.max_position_risk = max_position_risk
            
        self.correlation_matrix = {}
        self.volatility_history = {}
        
    def calculate_position_size(self, current_balance: float, entry_price: float, 
                              stop_loss: float, confidence: float = 0.5) -> float:
        """Calculate optimal position size based on risk parameters with $50 maximum limit."""
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
        
        # Risk per trade in USD
        risk_per_trade = current_balance * self.max_position_risk
        
        # Risk per unit (price difference to stop loss)
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            return 0.0
        
        # Base position size
        base_position_size = risk_per_trade / risk_per_unit
        
        # Adjust based on confidence - flexible sizing within limits
        if confidence > 0.7:  # High confidence
            confidence_multiplier = 1.3  # More conservative scaling
        elif confidence > 0.5:  # Medium confidence
            confidence_multiplier = 1.1
        else:  # Low confidence
            confidence_multiplier = 0.7
        
        confidence_adjusted_size = base_position_size * confidence_multiplier
        
        # Ensure position doesn't exceed maximum portfolio risk
        max_position_value = current_balance * self.max_portfolio_risk
        position_value = confidence_adjusted_size * entry_price
        
        if position_value > max_position_value:
            confidence_adjusted_size = max_position_value / entry_price
        
        # Apply dynamic 25% maximum margin limit
        max_margin_usd = current_balance * 0.25  # 25% of current balance
        margin_required = (confidence_adjusted_size * entry_price) / 8  # Assume 8x leverage
        
        if margin_required > max_margin_usd:
            # Scale down to maximum 25% margin
            max_quantity = (max_margin_usd * 8) / entry_price
            confidence_adjusted_size = max_quantity
        
        return max(0.0, confidence_adjusted_size)
    
    def check_portfolio_diversification(self, current_positions: Dict, new_symbol: str, 
                                      max_concentration: float = None) -> bool:
        """Check if adding new position maintains portfolio diversification."""
        if not current_positions:
            return True
        
        total_notional = sum(pos.get('notional_usd', 0) for pos in current_positions.values())
        
        if total_notional <= 0:
            return True
        
        # Set concentration limit based on risk level
        if max_concentration is None:
            from config import Config
            risk_level = Config.RISK_LEVEL.lower()
            
            if risk_level == 'low':
                max_concentration = 1.0  # 100% per position - konsantrasyon limiti kaldırıldı
            elif risk_level == 'high':
                max_concentration = 1.0  # 100% per position - konsantrasyon limiti kaldırıldı
            else:  # medium
                max_concentration = 1.0  # 100% per position - konsantrasyon limiti kaldırıldı
        
        # Check if any single position is too concentrated
        # Use dynamic balance calculation (current_balance + total_margin)
        total_margin = sum(pos.get('margin_usd', 0) for pos in current_positions.values())
        total_balance = 200.0  # Initial balance
        current_available_balance = total_balance - total_margin
        
        # Dynamic total balance = available balance + used margin
        dynamic_total_balance = current_available_balance + total_margin
        
        if dynamic_total_balance <= 0:
            return True
            
        for symbol, position in current_positions.items():
            position_concentration = position.get('margin_usd', 0) / dynamic_total_balance
            # Konsantrasyon limiti kaldırıldı - her zaman True döndür
            # if position_concentration > max_concentration:
            #     logging.warning(f"Position {symbol} exceeds concentration limit: {position_concentration:.2%} > {max_concentration:.0%}")
            #     return False
        
        # Allow multiple positions - only block if we're at maximum positions
        max_positions = 6  # Maximum 6 positions as requested
        if len(current_positions) >= max_positions:
            logging.warning(f"Maximum positions limit reached: {len(current_positions)}/{max_positions}")
            return False
        
        # Allow same symbol positions - don't block if symbol already exists
        # This allows adding to existing positions or opening new positions in same symbol
        if new_symbol in current_positions:
            logging.info(f"Symbol {new_symbol} already in portfolio, allowing additional position")
        
        return True
    
    def calculate_portfolio_risk(self, positions: Dict, current_prices: Dict) -> float:
        """Calculate current portfolio risk level."""
        if not positions:
            return 0.0
        
        total_portfolio_value = sum(pos.get('notional_usd', 0) for pos in positions.values())
        
        if total_portfolio_value <= 0:
            return 0.0
        
        # Simplified risk calculation (in production, use more sophisticated methods)
        portfolio_risk = 0.0
        for symbol, position in positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                entry_price = position.get('entry_price', current_price)
                risk_per_position = abs(current_price - entry_price) / entry_price
                position_weight = position.get('notional_usd', 0) / total_portfolio_value
                portfolio_risk += risk_per_position * position_weight
        
        return portfolio_risk
    
    def should_enter_trade(self, symbol: str, current_positions: Dict, current_prices: Dict, 
                          confidence: float, proposed_notional: float, current_balance: float = 200.0, 
                          ai_risk_usd: float = None) -> Dict[str, Any]:
        """Determine if a trade should be entered based on position limits only."""
        decision = {
            'should_enter': True,
            'reason': 'Position limit check passed',
            'adjusted_notional': proposed_notional
        }
        
        # Only check position limits - total risk limit removed
        if not self.check_portfolio_diversification(current_positions, symbol):
            decision['should_enter'] = False
            decision['reason'] = 'Position limit reached'
            return decision
        
        # **NOTE: AI's risk_usd value is ignored - we use our own confidence-based margin calculation**
        if ai_risk_usd is not None:
            # Log AI's risk_usd for information only, but don't use it for risk management
            decision['reason'] = f'AI risk_usd: ${ai_risk_usd:.2f} (ignored) - Using system margin calculation'
        
        return decision


def sample_strategy(symbol: str, data: pd.DataFrame, portfolio_state: Dict) -> Optional[Dict]:
    """
    Sample trading strategy for backtesting.
    This is a simple moving average crossover strategy.
    """
    if len(data) < 50:
        return None
    
    # Calculate moving averages
    short_ma = data['close'].rolling(window=20).mean().iloc[-1]
    long_ma = data['close'].rolling(window=50).mean().iloc[-1]
    current_price = data['close'].iloc[-1]
    
    # Simple crossover strategy
    if short_ma > long_ma and portfolio_state['current_balance'] > 10:
        # Buy signal
        return {
            'signal': 'buy_to_enter',
            'quantity': 1.0,
            'leverage': 2,
            'confidence': 0.7
        }
    elif short_ma < long_ma and symbol in portfolio_state['positions']:
        # Sell signal
        return {
            'signal': 'close_position',
            'quantity': portfolio_state['positions'][symbol]['quantity'],
            'leverage': 1
        }
    
    return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Run backtest
    engine = BacktestEngine(initial_balance=1000.0)
    result = engine.run_backtest(
        strategy_func=sample_strategy,
        symbols=['BTC', 'ETH'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        interval='1d'
    )
    
    print("Backtest Results:")
    print(json.dumps(result['metrics'], indent=2))
