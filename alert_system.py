"""
Real-time alert system for trading bot.
Monitors price movements, risk limits, and performance metrics.
"""
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    PRICE_MOVEMENT = "price_movement"
    RISK_LIMIT = "risk_limit"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    TRADE_EXECUTION = "trade_execution"

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    type: AlertType
    level: AlertLevel
    title: str
    message: str
    symbol: Optional[str] = None
    timestamp: Optional[datetime] = None
    data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'type': self.type.value,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data or {}
        }

class AlertManager:
    """Manages real-time alerts and notifications."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.max_alerts = 100
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.price_thresholds: Dict[str, Dict[str, float]] = {}
        self.risk_limits: Dict[str, float] = {
            'max_portfolio_risk': 0.02,  # 2%
            'max_drawdown': 0.1,  # 10%
        }
        
    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        return f"alert_{int(time.time())}_{len(self.alerts)}"
    
    def create_alert(self, alert_type: AlertType, level: AlertLevel, 
                    title: str, message: str, symbol: Optional[str] = None,
                    data: Optional[Dict[str, Any]] = None) -> Alert:
        """Create and dispatch a new alert."""
        alert = Alert(
            id=self._generate_alert_id(),
            type=alert_type,
            level=level,
            title=title,
            message=message,
            symbol=symbol,
            data=data
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Dispatch to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
        
        logging.info(f"Alert created: {alert.title} - {alert.message}")
        return alert
    
    def monitor_price_movements(self, symbol: str, current_price: float, 
                               previous_price: float, threshold_percent: float = 2.0) -> bool:
        """Monitor price movements and trigger alerts for significant changes."""
        if previous_price <= 0:
            return False
        
        price_change_percent = abs((current_price - previous_price) / previous_price) * 100
        
        if price_change_percent >= threshold_percent:
            level = AlertLevel.CRITICAL if price_change_percent >= 5.0 else AlertLevel.WARNING
            
            direction = "up" if current_price > previous_price else "down"
            self.create_alert(
                alert_type=AlertType.PRICE_MOVEMENT,
                level=level,
                title=f"Price Movement Alert - {symbol}",
                message=f"{symbol} moved {direction} {price_change_percent:.2f}% from ${previous_price:.4f} to ${current_price:.4f}",
                symbol=symbol,
                data={
                    'current_price': current_price,
                    'previous_price': previous_price,
                    'change_percent': price_change_percent,
                    'direction': direction,
                    'threshold_percent': threshold_percent
                }
            )
            return True
        
        return False
    
    def monitor_risk_limits(self, portfolio_data: Dict[str, Any], 
                           positions: Dict[str, Any], current_prices: Dict[str, float]) -> List[Alert]:
        """Monitor risk limits and trigger alerts."""
        alerts_triggered = []
        
        # Portfolio risk monitoring
        portfolio_risk = self._calculate_portfolio_risk(positions, current_prices)
        if portfolio_risk > self.risk_limits['max_portfolio_risk']:
            alert = self.create_alert(
                alert_type=AlertType.RISK_LIMIT,
                level=AlertLevel.CRITICAL,
                title="Portfolio Risk Limit Exceeded",
                message=f"Portfolio risk ({portfolio_risk:.2%}) exceeds maximum limit ({self.risk_limits['max_portfolio_risk']:.2%})",
                data={
                    'current_risk': portfolio_risk,
                    'risk_limit': self.risk_limits['max_portfolio_risk'],
                    'positions': list(positions.keys())
                }
            )
            alerts_triggered.append(alert)
        
        # Position concentration monitoring - DISABLED
        # Concentration limits have been removed to allow more aggressive position sizing
        
        return alerts_triggered
    
    def monitor_performance(self, portfolio_data: Dict[str, Any]) -> List[Alert]:
        """Monitor performance metrics and trigger alerts."""
        alerts_triggered = []
        
        total_return = portfolio_data.get('total_return', 0)
        current_drawdown = portfolio_data.get('current_drawdown', 0)
        
        # Large PnL movements
        if abs(total_return) >= 5.0:  # 5% move
            direction = "profit" if total_return > 0 else "loss"
            alert = self.create_alert(
                alert_type=AlertType.PERFORMANCE,
                level=AlertLevel.INFO,
                title=f"Significant {direction.capitalize()}",
                message=f"Portfolio {direction}: {total_return:.2f}%",
                data={
                    'return_percent': total_return,
                    'direction': direction
                }
            )
            alerts_triggered.append(alert)
        
        # Drawdown alerts
        if current_drawdown >= self.risk_limits['max_drawdown']:
            alert = self.create_alert(
                alert_type=AlertType.PERFORMANCE,
                level=AlertLevel.CRITICAL,
                title="Maximum Drawdown Alert",
                message=f"Current drawdown ({current_drawdown:.2%}) exceeds maximum limit ({self.risk_limits['max_drawdown']:.2%})",
                data={
                    'current_drawdown': current_drawdown,
                    'drawdown_limit': self.risk_limits['max_drawdown']
                }
            )
            alerts_triggered.append(alert)
        
        return alerts_triggered
    
    def monitor_trade_executions(self, trade_data: Dict[str, Any]) -> Alert:
        """Monitor trade executions and trigger alerts."""
        symbol = trade_data.get('symbol')
        pnl = trade_data.get('pnl', 0)
        direction = trade_data.get('direction', 'unknown')
        
        level = AlertLevel.CRITICAL if abs(pnl) >= 50 else AlertLevel.INFO
        
        return self.create_alert(
            alert_type=AlertType.TRADE_EXECUTION,
            level=level,
            title=f"Trade Executed - {symbol}",
            message=f"{direction.upper()} {symbol}: PnL ${pnl:.2f}",
            symbol=symbol,
            data=trade_data
        )
    
    def _calculate_portfolio_risk(self, positions: Dict[str, Any], 
                                 current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio risk level."""
        if not positions:
            return 0.0
        
        total_notional = sum(pos.get('notional_usd', 0) for pos in positions.values())
        if total_notional <= 0:
            return 0.0
        
        portfolio_risk = 0.0
        for symbol, position in positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                entry_price = position.get('entry_price', current_price)
                risk_per_position = abs(current_price - entry_price) / entry_price
                position_weight = position.get('notional_usd', 0) / total_notional
                portfolio_risk += risk_per_position * position_weight
        
        return portfolio_risk
    
    def _calculate_position_concentration(self, position: Dict[str, Any], 
                                        all_positions: Dict[str, Any]) -> float:
        """Calculate position concentration in portfolio using dynamic margin/balance-based calculation."""
        total_margin = sum(pos.get('margin_usd', 0) for pos in all_positions.values())
        total_balance = 200.0  # Initial balance
        current_available_balance = total_balance - total_margin
        
        # Dynamic total balance = available balance + used margin
        dynamic_total_balance = current_available_balance + total_margin
        
        if dynamic_total_balance <= 0:
            return 0.0
        
        position_margin = position.get('margin_usd', 0)
        return position_margin / dynamic_total_balance
    
    def get_recent_alerts(self, limit: int = 10, alert_type: Optional[AlertType] = None) -> List[Alert]:
        """Get recent alerts, optionally filtered by type."""
        alerts = self.alerts.copy()
        
        if alert_type:
            alerts = [alert for alert in alerts if alert.type == alert_type]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def clear_old_alerts(self, hours: int = 24) -> int:
        """Clear alerts older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        initial_count = len(self.alerts)
        
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
        
        return initial_count - len(self.alerts)


class ConsoleAlertHandler:
    """Simple console-based alert handler."""
    
    def __call__(self, alert: Alert) -> None:
        """Handle alert by printing to console."""
        color_codes = {
            AlertLevel.INFO: '\033[94m',      # Blue
            AlertLevel.WARNING: '\033[93m',   # Yellow
            AlertLevel.CRITICAL: '\033[91m'   # Red
        }
        reset_code = '\033[0m'
        
        color = color_codes.get(alert.level, '\033[0m')
        print(f"{color}[{alert.level.value}] {alert.title}: {alert.message}{reset_code}")


class FileAlertHandler:
    """File-based alert handler for logging."""
    
    def __init__(self, filename: str = "alerts.json"):
        self.filename = filename
    
    def __call__(self, alert: Alert) -> None:
        """Handle alert by writing to file."""
        try:
            with open(self.filename, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
        except Exception as e:
            logging.error(f"Failed to write alert to file: {e}")


# Global alert manager instance
alert_manager = AlertManager()

# Add default handlers
alert_manager.add_handler(ConsoleAlertHandler())
alert_manager.add_handler(FileAlertHandler())
