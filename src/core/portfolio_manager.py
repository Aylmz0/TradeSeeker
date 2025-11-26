import json
import time
import copy
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from config.config import Config
from src.utils import safe_file_read, safe_file_write, format_num
from src.core.backtest import AdvancedRiskManager
from src.core.market_data import RealMarketData

# Define HTF constants
HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'
HTF_LABEL = HTF_INTERVAL

try:
    from src.services.binance import BinanceOrderExecutor
    BINANCE_IMPORT_ERROR = None
except Exception as e:
    BinanceOrderExecutor = None
    BINANCE_IMPORT_ERROR = str(e)

class PortfolioManager:
    """Manages the portfolio state, positions, and history."""

    def __init__(self, initial_balance: float = None):
        # Dinamik initial balance - eğer verilmediyse gerçek balance'dan al veya $200 kullan
        if initial_balance is None:
            # Önce saved state'den dene, yoksa Config'den al
            saved_state = safe_file_read("data/portfolio_state.json", default_data={})
            if 'initial_balance' in saved_state:
                self.initial_balance = saved_state['initial_balance']
            else:
                self.initial_balance = Config.INITIAL_BALANCE
        else:
            self.initial_balance = initial_balance
            
        self.state_file = "data/portfolio_state.json"; self.history_file = "data/trade_history.json"
        self.override_file = "data/manual_override.json"; self.cycle_history_file = "data/cycle_history.json"
        self.max_cycle_history = 50; self.maintenance_margin_rate = 0.01

        self.current_balance = self.initial_balance; self.positions = {}
        self.directional_bias = self._init_directional_bias()
        self.trend_state: Dict[str, Dict[str, Any]] = {}
        self.trend_flip_cooldown = 2
        self.trend_flip_history_window = 5
        # Trend flip cooldown yönetimi PortfolioManager tarafında tutulur.
        self.indicator_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.last_execution_report: Dict[str, Any] = {}
        self.history_reset_interval = getattr(Config, "HISTORY_RESET_INTERVAL", 35)
        self.last_history_reset_cycle = 0
        self.cycles_since_history_reset = 0
        self.trading_mode = getattr(Config, "TRADING_MODE", "simulation")
        self.is_live_trading = self.trading_mode == "live"
        self.order_executor: Optional["BinanceOrderExecutor"] = None
        self.directional_cooldowns: Dict[str, int] = {'long': 0, 'short': 0}
        self.relaxed_countertrend_cycles: int = 0
        self.counter_trend_cooldown: int = 0
        self.counter_trend_consecutive_losses: int = 0
        self.coin_cooldowns: Dict[str, int] = {}  # Coin bazlı cooldown: {coin: cycles_remaining}

        self.current_cycle_number = 0

        self.trade_history = self.load_trade_history() # Load first
        self.load_state() # Loads balance, positions, trade_count
        self.cycle_history = self.load_cycle_history()
        self.risk_manager = AdvancedRiskManager()  # Initialize risk manager
        self.market_data = RealMarketData()  # Initialize market data for counter-trend detection

        # Initialize total_value before _initialize_live_trading (which calls sync_live_account)
        self.total_value = self.current_balance
        self.total_return = 0.0
        self.start_time = datetime.now()
        self.portfolio_values_history = [self.initial_balance]  # Track portfolio values for Sharpe ratio
        self.sharpe_ratio = 0.0

        if self.is_live_trading:
            self._initialize_live_trading()
        elif BINANCE_IMPORT_ERROR:
            print(f"ℹ️ Binance executor unavailable ({BINANCE_IMPORT_ERROR}). Staying in simulation mode.")

        self.update_prices({}, increment_loss_counters=False) # Calculate initial value with loaded positions

    def _init_directional_bias(self) -> Dict[str, Dict[str, Any]]:
        return {
            'long': {
                'rolling': deque(maxlen=20),
                'net_pnl': 0.0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'consecutive_losses': 0,
                'consecutive_wins': 0,
                'caution_active': False,
            'caution_win_progress': 0,
            'loss_streak_loss_usd': 0.0
            },
            'short': {
                'rolling': deque(maxlen=20),
                'net_pnl': 0.0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'consecutive_losses': 0,
                'consecutive_wins': 0,
                'caution_active': False,
            'caution_win_progress': 0,
            'loss_streak_loss_usd': 0.0
            }
        }

    def load_state(self):
        data = safe_file_read(self.state_file, default_data={})
        self.current_balance = data.get('current_balance', self.initial_balance)
        self.positions = data.get('positions', {})
        self.trade_count = data.get('trade_count', len(self.trade_history)) # Initialize from history if not in state
        print(f"✅ Loaded state ({len(self.positions)} positions, {self.trade_count} closed trades)" if data else "ℹ️ No state file found.")

        bias_state = data.get('directional_bias')
        if bias_state:
            self.directional_bias = self._init_directional_bias()
            for side in ('long', 'short'):
                stored = bias_state.get(side, {})
                stats = self.directional_bias[side]
                stats['rolling'].extend(stored.get('rolling', []))
                stats['net_pnl'] = stored.get('net_pnl', 0.0)
                stats['trades'] = stored.get('trades', 0)
                stats['wins'] = stored.get('wins', 0)
                stats['losses'] = stored.get('losses', 0)
                stats['consecutive_losses'] = stored.get('consecutive_losses', 0)
                stats['consecutive_wins'] = stored.get('consecutive_wins', 0)
                stats['caution_active'] = stored.get('caution_active', False)
                stats['caution_win_progress'] = stored.get('caution_win_progress', 0)
                stats['loss_streak_loss_usd'] = stored.get('loss_streak_loss_usd', 0.0)
        self.last_history_reset_cycle = data.get('last_history_reset_cycle', self.last_history_reset_cycle)
        self.cycles_since_history_reset = data.get('cycles_since_history_reset', self.cycles_since_history_reset)
        self.directional_cooldowns = data.get('directional_cooldowns', {'long': 0, 'short': 0})
        self.relaxed_countertrend_cycles = data.get('relaxed_countertrend_cycles', 0)
        self.counter_trend_cooldown = data.get('counter_trend_cooldown', 0)
        self.counter_trend_consecutive_losses = data.get('counter_trend_consecutive_losses', 0)
        self.coin_cooldowns = data.get('coin_cooldowns', {})

    def save_state(self):
        data = {
            'current_balance': self.current_balance,
            'positions': self.positions,
            'total_value': self.total_value,
            'total_return': self.total_return,
            'initial_balance': self.initial_balance,
            'trade_count': self.trade_count,
            'last_updated': datetime.now().isoformat(),
            'sharpe_ratio': self.sharpe_ratio,
            'directional_bias': self._serialize_directional_bias(),
            'last_history_reset_cycle': self.last_history_reset_cycle,
            'cycles_since_history_reset': self.cycles_since_history_reset,
            'directional_cooldowns': self.directional_cooldowns,
            'relaxed_countertrend_cycles': self.relaxed_countertrend_cycles,
            'counter_trend_cooldown': self.counter_trend_cooldown,
            'counter_trend_consecutive_losses': self.counter_trend_consecutive_losses,
            'coin_cooldowns': self.coin_cooldowns
        }
        safe_file_write(self.state_file, data); print(f"✅ Saved state.")

    # --- Live trading helpers -------------------------------------------------
    def _initialize_live_trading(self):
        """Configure Binance executor when live trading mode is enabled."""
        if BinanceOrderExecutor is None:
            print("❌ Live trading requested but Binance executor is unavailable.")
            self.is_live_trading = False
            return
        try:
            self.order_executor = BinanceOrderExecutor(self.market_data.available_coins)
            if not self.order_executor.is_live():
                print("⚠️ Live trading requested but executor initialized in simulation mode. Reverting to paper trading.")
                self.is_live_trading = False
                self.order_executor = None
                return
            print(f"✅ Live trading mode enabled (Binance {'TESTNET' if Config.BINANCE_TESTNET else 'MAINNET'}).")
            self.sync_live_account()
        except BinanceAPIError as exc:
            print(f"❌ Binance setup failed: {exc}. Reverting to simulation mode.")
            self.is_live_trading = False
            self.order_executor = None
        except Exception as exc:
            print(f"❌ Unexpected Binance setup error: {exc}. Reverting to simulation mode.")
            self.is_live_trading = False
            self.order_executor = None

    def _build_default_exit_plan(self, direction: str, entry_price: float) -> Dict[str, float]:
        """Generate a sensible default exit plan when AI data is unavailable."""
        try:
            entry = float(entry_price or 0.0)
        except (TypeError, ValueError):
            entry = 0.0
        if entry <= 0:
            entry = max(Config.MIN_EXIT_PLAN_OFFSET, 1.0)
        stop_pct = max(Config.DEFAULT_STOP_LOSS_PCT, 1e-4)
        tp_pct = max(Config.DEFAULT_PROFIT_TARGET_PCT, stop_pct * 1.5)
        min_offset = max(Config.MIN_EXIT_PLAN_OFFSET, 1e-6)
        stop_offset = max(entry * stop_pct, min_offset)
        tp_offset = max(entry * tp_pct, min_offset)
        if direction == 'short':
            stop_loss = entry + stop_offset
            profit_target = max(entry - tp_offset, 0.0)
        else:
            stop_loss = max(entry - stop_offset, 0.0)
            profit_target = entry + tp_offset
        return {
            'stop_loss': stop_loss,
            'profit_target': profit_target
        }

    def _ensure_exit_plan(self, position: Dict[str, Any], *candidate_plans: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure a position carries a valid exit plan, supplementing with defaults if needed."""
        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price') or position.get('current_price')
        default_plan = self._build_default_exit_plan(direction, entry_price)
        final_plan = default_plan.copy()
        provided_keys = set()
        for plan in candidate_plans:
            if not plan:
                continue
            for key, value in plan.items():
                if value is None:
                    continue
                final_plan[key] = value
                if key in ('stop_loss', 'profit_target') and isinstance(value, (int, float)) and value > 0:
                    provided_keys.add(key)
        missing_required = [key for key in ('stop_loss', 'profit_target') if key not in provided_keys]
        if missing_required:
            symbol = position.get('symbol', 'UNKNOWN')
            print(f"⚠️ Missing {', '.join(missing_required)} for {symbol} - using default exit plan offsets.")
        position['exit_plan'] = final_plan
        return final_plan

    def _merge_live_positions(self, snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Merge Binance snapshot with local runtime metadata (exit plans, confidence, etc.)."""
        merged: Dict[str, Dict[str, Any]] = {}
        existing_positions = self.positions if isinstance(self.positions, dict) else {}
        for coin, snap_pos in snapshot.items():
            previous = existing_positions.get(coin, {})
            merged_pos: Dict[str, Any] = {}
            merged_pos.update(previous)
            merged_pos.update(snap_pos)
            merged_pos['symbol'] = coin

            # Carry forward runtime metadata that Binance snapshot doesn't include
            for key in (
                'confidence',
                'loss_cycle_count',
                'trend_context',
                'trend_alignment',
                'entry_oid',
                'tp_oid',
                'sl_oid',
                'wait_for_fill',
                'entry_volume_ratio',
                'entry_volume',
                'entry_avg_volume',
                'entry_atr_14',
                'trailing'
            ):
                if key in previous and key not in merged_pos:
                    merged_pos[key] = previous[key]

            # Risk USD should reflect current margin if available
            margin_usd = merged_pos.get('margin_usd')
            if isinstance(margin_usd, (int, float)):
                merged_pos['risk_usd'] = margin_usd

            existing_plan = previous.get('exit_plan')
            snapshot_plan = snap_pos.get('exit_plan')
            self._ensure_exit_plan(merged_pos, snapshot_plan, existing_plan)
            merged[coin] = merged_pos
        return merged

    def sync_live_account(self):
        """Refresh balances and open positions from Binance when in live mode."""
        if not self.is_live_trading or not self.order_executor or not self.order_executor.is_live():
            return
        
        overview = None
        try:
            overview = self.order_executor.get_account_overview()
            if overview:
                available = overview.get("availableBalance")
                total_wallet_balance = overview.get("totalWalletBalance")
                wallet_balance = overview.get("walletBalance")
                
                # Debug: Log what we got from Binance
                print(f"🔍 Binance API Response: availableBalance={available}, totalWalletBalance={total_wallet_balance}, walletBalance={wallet_balance}")
                
                # Update available cash balance
                if available is not None and available > 0:
                    old_balance = self.current_balance
                    self.current_balance = float(available)
                    if abs(old_balance - self.current_balance) > 0.01:  # Only log if significant change
                        print(f"💰 Balance updated: ${old_balance:.2f} → ${self.current_balance:.2f}")
                
                # Note: We'll calculate total_value manually after positions are synced
                # Total value = Available cash + Margin used + Unrealized PnL
                # Binance totalWalletBalance is used for validation only
                if total_wallet_balance is not None and total_wallet_balance > 0:
                    print(f"✅ Binance totalWalletBalance: ${total_wallet_balance:.2f} (will validate against calculated value)")
                elif wallet_balance is not None and wallet_balance > 0:
                    print(f"⚠️ totalWalletBalance not available, using walletBalance: ${wallet_balance:.2f}")
                else:
                    print(f"⚠️ Neither totalWalletBalance nor walletBalance available from Binance API")
        except BinanceAPIError as exc:
            print(f"⚠️ Binance balance sync failed: {exc}")
        except Exception as exc:
            print(f"⚠️ Unexpected Binance balance sync error: {exc}")

        try:
            snapshot = self.order_executor.get_positions_snapshot()
            if isinstance(snapshot, dict):
                self.positions = self._merge_live_positions(snapshot)
                
                # Calculate total margin used from all open positions
                # For cross margin, margin_usd might be 0, so calculate from notional/leverage
                total_margin_used = 0.0
                for pos in self.positions.values():
                    margin = pos.get('margin_usd', 0.0)
                    if margin <= 0:
                        # Calculate margin from notional and leverage (for cross margin)
                        notional = pos.get('notional_usd', 0.0)
                        leverage = pos.get('leverage', 1)
                        if notional > 0 and leverage > 0:
                            margin = notional / leverage
                    if isinstance(margin, (int, float)) and margin > 0:
                        total_margin_used += margin
                
                old_total = self.total_value
                
                # Calculate total unrealized PnL
                total_unrealized_pnl = 0.0
                for pos in self.positions.values():
                    pnl = pos.get('unrealized_pnl', 0.0)
                    if isinstance(pnl, (int, float)):
                        total_unrealized_pnl += pnl
                
                # Total value = Available cash + Margin used + Unrealized PnL
                # This is the correct formula: what you have available + what's locked in positions + unrealized gains/losses
                self.total_value = self.current_balance + total_margin_used + total_unrealized_pnl
                
                if abs(old_total - self.total_value) > 0.01:
                    print(f"📊 Total value updated: ${old_total:.2f} → ${self.total_value:.2f}")
                    print(f"   (Available cash: ${self.current_balance:.2f} + Margin used: ${total_margin_used:.2f} + Unrealized PnL: ${total_unrealized_pnl:.2f})")
                    print(f"   ℹ️ Unrealized PnL from Binance (includes funding fees)")
                    
                    # Debug: Also show what Binance says for comparison
                    if overview:
                        total_wb = overview.get("totalWalletBalance")
                        wallet_b = overview.get("walletBalance")
                        if total_wb:
                            wallet_b_str = f"${wallet_b:.2f}" if wallet_b else "N/A"
                            print(f"   (Binance totalWalletBalance: ${total_wb:.2f}, walletBalance: {wallet_b_str})")
                            # Validate our calculation against Binance
                            diff = abs(self.total_value - total_wb)
                            if diff > 0.10:  # More than 10 cents difference
                                print(f"   ⚠️ Warning: Calculated total_value differs from Binance totalWalletBalance by ${diff:.2f}")
            else:
                self.positions = {}
                # No positions, total value = totalWalletBalance (or available cash if not available)
                if overview and overview.get("totalWalletBalance") and overview.get("totalWalletBalance") > 0:
                    self.total_value = float(overview["totalWalletBalance"])
                else:
                    self.total_value = self.current_balance
        except BinanceAPIError as exc:
            print(f"⚠️ Binance position sync failed: {exc}")
        except Exception as exc:
            print(f"⚠️ Unexpected Binance position sync error: {exc}")

    @staticmethod
    def _calculate_realized_pnl(entry_price: float, exit_price: float, quantity: float, direction: str) -> float:
        if quantity <= 0 or entry_price <= 0 or exit_price <= 0:
            return 0.0
        if direction == 'long':
            return (exit_price - entry_price) * quantity
        return (entry_price - exit_price) * quantity

    def execute_live_entry(
        self,
        coin: str,
        direction: str,
        quantity: float,
        leverage: int,
        current_price: float,
        notional_usd: float,
        confidence: float,
        margin_usd: Optional[float] = None,
        stop_loss: Optional[float] = None,
        profit_target: Optional[float] = None,
        invalidation: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        try:
            order = self.order_executor.place_market_order(
                coin=coin,
                direction=direction,
                quantity=quantity,
                leverage=leverage,
                price_reference=current_price,
                reduce_only=False,
            )
            executed_qty = float(order.get("executedQty", 0.0))
            avg_price = float(order.get("avgPriceComputed", order.get("avgPrice", 0.0)))
            self.sync_live_account()
            position = self.positions.get(coin, {})
            
            # Use provided margin_usd if available, otherwise calculate from position or notional
            if margin_usd is None or margin_usd <= 0:
                if position and position.get('margin_usd', 0) > 0:
                    margin_usd = position.get('margin_usd')
                else:
                    # Calculate from executed notional (more accurate than initial notional)
                    executed_notional = executed_qty * avg_price
                    margin_usd = executed_notional / max(leverage, 1)
            
            notional_runtime = position.get('notional_usd') if position else executed_qty * avg_price
            if position:
                position['confidence'] = confidence
                # Ensure margin_usd is saved to position for later use in TP/SL checks
                if 'margin_usd' not in position or position.get('margin_usd', 0) <= 0:
                    position['margin_usd'] = margin_usd
                    print(f"💾 Saved margin_usd=${margin_usd:.2f} to position for {coin}")
                exit_plan = position.setdefault('exit_plan', {})
                if stop_loss is not None:
                    exit_plan['stop_loss'] = stop_loss
                if profit_target is not None:
                    exit_plan['profit_target'] = profit_target
                if invalidation is not None:
                    exit_plan['invalidation_condition'] = invalidation
                position['risk_usd'] = position.get('margin_usd', margin_usd)
                self._ensure_exit_plan(position, exit_plan)
                
                # Calculate and save kademeli stop loss to exit_plan (for 30-second monitoring)
                # No Binance orders - all TP/SL decisions made by 30-second monitoring loop (like simulation mode)
                if executed_qty > 0:
                    # Calculate margin-based stop loss using kademeli loss cutting
                    loss_multiplier = 0.03  # Default: %3 for margin >= 50
                    if margin_usd < 30:
                        loss_multiplier = 0.07  # %7 for margin < 30
                    elif margin_usd < 40:
                        loss_multiplier = 0.05  # %5 for margin 30-40
                    elif margin_usd < 50:
                        loss_multiplier = 0.05  # %5 for margin 40-50
                    else:
                        loss_multiplier = 0.03  # %3 for margin >= 50
                    
                    loss_threshold_usd = margin_usd * loss_multiplier
                    
                    # Calculate stop loss price from loss threshold
                    if direction == 'long':
                        margin_based_stop_loss = avg_price - (loss_threshold_usd / executed_qty)
                    else:  # short
                        margin_based_stop_loss = avg_price + (loss_threshold_usd / executed_qty)
                    
                    # Use the tighter stop loss (closer to entry price = more conservative)
                    # For long: higher stop_loss is tighter (closer to entry)
                    # For short: lower stop_loss is tighter (closer to entry)
                    if stop_loss is not None and stop_loss > 0:
                        if direction == 'long':
                            # For long: use the higher stop loss (more conservative)
                            final_stop_loss = max(stop_loss, margin_based_stop_loss)
                        else:  # short
                            # For short: use the lower stop loss (more conservative)
                            final_stop_loss = min(stop_loss, margin_based_stop_loss)
                    else:
                        final_stop_loss = margin_based_stop_loss
                    
                    # Final validation: ensure stop loss direction is correct
                    if direction == 'long':
                        if final_stop_loss >= avg_price:
                            # Stop loss cannot be at or above entry for long - recalculate from loss_threshold
                            final_stop_loss = avg_price - (loss_threshold_usd / executed_qty)
                            print(f"⚠️ Final validation: Stop loss for {coin} LONG was invalid (>= entry), recalculated to ${format_num(final_stop_loss, 4)}")
                    else:  # short
                        if final_stop_loss <= avg_price:
                            # Stop loss cannot be at or below entry for short - recalculate from loss_threshold
                            final_stop_loss = avg_price + (loss_threshold_usd / executed_qty)
                            print(f"⚠️ Final validation: Stop loss for {coin} SHORT was invalid (<= entry), recalculated to ${format_num(final_stop_loss, 4)}")
                    
                    # Save kademeli stop loss to exit_plan (will be checked by 30-second monitoring)
                    if final_stop_loss > 0:
                        exit_plan['stop_loss'] = final_stop_loss
                        print(f"💾 Kademeli stop loss saved for {coin}: ${format_num(final_stop_loss, 4)} (${loss_threshold_usd:.2f} loss limit, {loss_multiplier*100:.1f}% of ${margin_usd:.2f} margin) - will be monitored by 30s loop")
            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "margin_usd": margin_usd,
                "notional_usd": notional_runtime,
            }
        except BinanceAPIError as exc:
            print(f"❌ Binance entry order failed for {coin}: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            print(f"❌ Unexpected Binance entry error for {coin}: {exc}")
            return {"success": False, "error": str(exc)}

    def execute_live_close(
        self,
        coin: str,
        position: Dict[str, Any],
        current_price: float,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        if not position:
            return {"success": False, "error": "no_position"}

        direction = position.get('direction', 'long')
        quantity = float(position.get('quantity', 0.0) or 0.0)
        if quantity <= 0:
            return {"success": False, "error": "invalid_quantity"}

        try:
            # No Binance TP/SL orders to cancel - all managed by 30-second monitoring
            order = self.order_executor.close_position(
                coin=coin,
                direction=direction,
                quantity=quantity,
                price_reference=current_price,
            )
            executed_qty = float(order.get("executedQty", 0.0))
            avg_price = float(order.get("avgPriceComputed", order.get("avgPrice", 0.0)))
            pnl = self._calculate_realized_pnl(position.get("entry_price", 0.0), avg_price, executed_qty, direction)
            self.sync_live_account()
            history_entry = {
                "symbol": coin,
                "direction": direction,
                "entry_price": position.get("entry_price"),
                "exit_price": avg_price,
                "quantity": executed_qty,
                "notional_usd": executed_qty * avg_price,
                "pnl": pnl,
                "entry_time": position.get("entry_time"),
                "exit_time": datetime.now().isoformat(),
                "leverage": position.get("leverage"),
                "close_reason": reason or "live_close",
                "exchange_order_id": order.get("orderId"),
            }
            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "pnl": pnl,
                "history_entry": history_entry,
            }
        except BinanceAPIError as exc:
            print(f"❌ Binance close order failed for {coin}: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            print(f"❌ Unexpected Binance close error for {coin}: {exc}")
            return {"success": False, "error": str(exc)}

    def execute_live_partial_close(
        self,
        coin: str,
        position: Dict[str, Any],
        close_percent: float,
        current_price: float,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        if not position:
            return {"success": False, "error": "no_position"}

        direction = position.get('direction', 'long')
        quantity = float(position.get('quantity', 0.0) or 0.0)
        if quantity <= 0:
            return {"success": False, "error": "invalid_quantity"}
        close_percent = max(0.0, min(close_percent, 1.0))
        requested_qty = quantity * close_percent
        if requested_qty <= 0:
            return {"success": False, "error": "zero_quantity"}

        try:
            order = self.order_executor.close_position(
                coin=coin,
                direction=direction,
                quantity=requested_qty,
                price_reference=current_price,
            )
            executed_qty = float(order.get("executedQty", 0.0))
            avg_price = float(order.get("avgPriceComputed", order.get("avgPrice", 0.0)))
            if executed_qty <= 0:
                return {"success": False, "error": "no_fill"}
            pnl = self._calculate_realized_pnl(position.get("entry_price", 0.0), avg_price, executed_qty, direction)
            # Sync account balance after partial close to get updated balance
            self.sync_live_account()
            history_entry = {
                "symbol": coin,
                "direction": direction,
                "entry_price": position.get("entry_price"),
                "exit_price": avg_price,
                "quantity": executed_qty,
                "notional_usd": executed_qty * avg_price,
                "pnl": pnl,
                "entry_time": position.get("entry_time"),
                "exit_time": datetime.now().isoformat(),
                "leverage": position.get("leverage"),
                "close_reason": reason or "live_partial_close",
                "exchange_order_id": order.get("orderId"),
            }
            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "pnl": pnl,
                "history_entry": history_entry,
            }
        except BinanceAPIError as exc:
            print(f"❌ Binance partial close failed for {coin}: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            print(f"❌ Unexpected Binance partial close error for {coin}: {exc}")
            return {"success": False, "error": str(exc)}

    def _backup_historical_files(self, cycle_number: int) -> Optional[str]:
        """Create a timestamped backup for historical JSON files before wiping them."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join("history_backups", f"{timestamp}_cycle_{cycle_number}")
        files_to_backup = [
            (self.history_file, []),
            (self.cycle_history_file, []),
            ("data/performance_history.json", []),
            ("data/performance_report.json", [])  # Changed from {} to [] - now array format
        ]

        try:
            os.makedirs(backup_dir, exist_ok=True)
            backed_up = []

            for file_path, default in files_to_backup:
                data = safe_file_read(file_path, default)
                # Skip writing files that never existed and match the empty default
                if data == default and not os.path.exists(file_path):
                    continue
                target_path = os.path.join(backup_dir, os.path.basename(file_path))
                safe_file_write(target_path, data)
                
                # Calculate items count for metadata
                items_count = None
                if isinstance(data, list):
                    items_count = len(data)
                elif isinstance(data, dict):
                    # For dict, count keys (but performance_report.json should be array now)
                    items_count = len(data)
                
                backed_up.append({
                    "file": file_path,
                    "items": items_count
                })

            metadata = {
                "cycle_number": cycle_number,
                "backed_up_at": datetime.now().isoformat(),
                "files": backed_up
            }
            safe_file_write(os.path.join(backup_dir, "metadata.json"), metadata)
            print(f"💾 History backup created at {backup_dir}")
            return backup_dir
        except Exception as e:
            print(f"⚠️ History backup failed: {e}")
            return None

    def reset_historical_data(self, cycle_number: int):
        """Clear historical logs to prevent long-term bias while keeping live positions."""
        self._backup_historical_files(cycle_number)
        print(f"🧹 HISTORY RESET: Clearing logs at cycle {cycle_number}")
        self.trade_history = []
        self.trade_count = 0
        self.directional_bias = self._init_directional_bias()
        self.trend_state = {}
        self.cycle_history = []
        safe_file_write(self.history_file, [])
        safe_file_write(self.cycle_history_file, [])
        safe_file_write("data/performance_history.json", [])
        # Preserve existing performance reports, just add a reset marker
        existing_reports = safe_file_read("data/performance_report.json", [])
        if isinstance(existing_reports, dict):
            # Old format - convert to array
            if "reset_reason" not in existing_reports:
                existing_reports = [existing_reports]
            else:
                existing_reports = []
        elif not isinstance(existing_reports, list):
            existing_reports = []
        
        # Add reset marker
        reset_marker = {
            "reset_reason": "periodic_bias_control",
            "reset_at_cycle": cycle_number,
            "timestamp": datetime.now().isoformat()
        }
        existing_reports.append(reset_marker)
        
        # Keep only last 50 entries
        if len(existing_reports) > 50:
            existing_reports = existing_reports[-50:]
        
        safe_file_write("data/performance_report.json", existing_reports)
        self.portfolio_values_history = [self.total_value]
        for pos in self.positions.values():
            pos['loss_cycle_count'] = 0
        self.last_history_reset_cycle = cycle_number
        self.cycles_since_history_reset = 0
        self.directional_cooldowns = {'long': 0, 'short': 0}
        self.coin_cooldowns = {}  # Coin bazlı cooldown'ları da sıfırla
        self.counter_trend_cooldown = 0
        self.counter_trend_consecutive_losses = 0
        self.relaxed_countertrend_cycles = 0
        self.save_state()
        print("✅ History reset complete.")

    def _serialize_directional_bias(self) -> Dict[str, Dict[str, Any]]:
        serialized = {}
        for side, stats in self.directional_bias.items():
            serialized[side] = {
                'rolling': list(stats['rolling']),
                'net_pnl': stats['net_pnl'],
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'consecutive_losses': stats['consecutive_losses'],
                'consecutive_wins': stats.get('consecutive_wins', 0),
                'caution_active': stats.get('caution_active', False),
                'caution_win_progress': stats.get('caution_win_progress', 0),
                'loss_streak_loss_usd': stats.get('loss_streak_loss_usd', 0.0)
            }
        return serialized

    def load_trade_history(self) -> List[Dict]:
        history = safe_file_read(self.history_file, default_data=[]); print(f"✅ Loaded {len(history)} trades."); return history
    def save_trade_history(self):
        history_to_save = self.trade_history[-100:]; safe_file_write(self.history_file, history_to_save); print(f"✅ Saved {len(history_to_save)} trades.")
    def add_to_history(self, trade: Dict):
        self.trade_history.append(trade)
        self.trade_count = len(self.trade_history)
        self.save_trade_history()
        self.update_directional_bias(trade)
        self.save_state()

    def update_directional_bias(self, trade: Dict):
        direction = trade.get('direction')
        if direction not in ('long', 'short'):
            return
        stats = self.directional_bias[direction]
        pnl = float(trade.get('pnl', 0.0) or 0.0)
        print(f"📊 update_directional_bias called: {direction.upper()} trade, PnL=${pnl:.2f}")
        stats['rolling'].append(pnl)
        stats['net_pnl'] += pnl
        stats['trades'] += 1
        
        # Check if this is a counter-trend trade
        trend_alignment = trade.get('trend_alignment', 'unknown')
        is_counter_trend = (trend_alignment == 'counter_trend')
        
        if pnl > 0:
            stats['wins'] += 1
            stats['consecutive_losses'] = 0
            stats['consecutive_wins'] = stats.get('consecutive_wins', 0) + 1
            if stats.get('caution_active'):
                stats['caution_win_progress'] = stats.get('caution_win_progress', 0) + 1
                if stats['caution_win_progress'] >= 3:
                    stats['caution_active'] = False
                    stats['caution_win_progress'] = 0
            # Reset counter-trend consecutive losses on win
            if is_counter_trend:
                self.counter_trend_consecutive_losses = 0
        elif pnl < 0:
            stats['losses'] += 1
            stats['consecutive_losses'] += 1
            stats['consecutive_wins'] = 0
            stats['caution_win_progress'] = 0
            # loss_streak_loss_usd'yi güncelle - cooldown aktif olsa bile takip etmeye devam et
            # Cooldown sırasında da yeni kayıplar gelebilir ve toplam $5'i geçebilir
            current_loss_streak = stats.get('loss_streak_loss_usd', 0.0)
            stats['loss_streak_loss_usd'] = current_loss_streak + abs(pnl)
            
            # Coin bazlı cooldown: Zararla kapandığında o coin için 1 cycle cooldown
            coin_symbol = trade.get('symbol', '').upper()
            if coin_symbol:
                self.coin_cooldowns[coin_symbol] = 1
                print(f"🛡️ Coin cooldown ACTIVATED for {coin_symbol}: 1 cycle (loss: ${pnl:.2f})")
            
            if stats['consecutive_losses'] >= 3:
                stats['caution_active'] = True
                stats['caution_win_progress'] = 0
            # Cooldown: 3 consecutive losses OR $5 total loss → 3 cycle cooldown (sabit)
            # Cooldown aktif olsa bile, yeni kayıplar geldiğinde tekrar kontrol et ve gerekirse yeniden aktif et
            loss_streak_usd = stats.get('loss_streak_loss_usd', 0.0)
            consecutive = stats['consecutive_losses']
            should_activate = consecutive >= 3 or loss_streak_usd >= 5.0
            
            print(f"🔍 Cooldown check for {direction.upper()}: consecutive_losses={consecutive}, loss_streak_usd=${loss_streak_usd:.2f}, should_activate={should_activate}")
            
            if should_activate:
                self._activate_directional_cooldown(direction, 3)
                print(f"🛡️ Directional cooldown ACTIVATED for {direction.upper()}: consecutive_losses={consecutive}, loss_streak_usd=${loss_streak_usd:.2f}")
                # loss_streak_loss_usd'yi sıfırlama - bir sonraki zarar için takip etmeye devam et
                # Sadece cooldown bittiğinde sıfırlanacak
            
            # Counter-trend cooldown: 2 consecutive counter-trend losses
            if is_counter_trend:
                self.counter_trend_consecutive_losses += 1
                if self.counter_trend_consecutive_losses >= 2:
                    self.counter_trend_cooldown = 3
                    self.counter_trend_consecutive_losses = 0
                    print(f"🛡️ Counter-trend cooldown activated: 2 consecutive counter-trend losses (3 cycles cooldown).")
        else:
            stats['consecutive_losses'] = 0
            stats['consecutive_wins'] = 0
            stats['caution_win_progress'] = 0
            stats['loss_streak_loss_usd'] = 0.0
            # Reset counter-trend consecutive losses on breakeven
            if is_counter_trend:
                self.counter_trend_consecutive_losses = 0

    def count_positions_by_direction(self) -> Dict[str, int]:
        counts = {'long': 0, 'short': 0}
        for pos in self.positions.values():
            direction = pos.get('direction')
            if direction in counts:
                counts[direction] += 1
        return counts

    def _activate_directional_cooldown(self, direction: str, cycles: int = 3):
        if direction not in ('long', 'short'):
            return
        current = self.directional_cooldowns.get(direction, 0)
        # Mevcut cooldown'dan daha uzun bir süre varsa, onu kullan
        self.directional_cooldowns[direction] = max(current, cycles)
        self.relaxed_countertrend_cycles = max(self.relaxed_countertrend_cycles, 3)
        print(f"🛡️ Directional cooldown activated for {direction.upper()} trades (3 cycles). Counter-trend restrictions relaxed for 3 cycles.")

    def tick_cooldowns(self):
        print(f"⏱️ tick_cooldowns called. Current cooldowns: {self.directional_cooldowns}, Coin cooldowns: {self.coin_cooldowns}")
        for direction in ('long', 'short'):
            cycles = self.directional_cooldowns.get(direction, 0)
            if cycles > 0:
                self.directional_cooldowns[direction] = cycles - 1
                print(f"⏱️ {direction.upper()} cooldown: {cycles} → {self.directional_cooldowns[direction]} cycles remaining")
                if self.directional_cooldowns[direction] == 0:
                    # Cooldown bittiğinde loss_streak_loss_usd ve consecutive_losses'i sıfırla
                    # Çünkü cooldown bir "reset" dönemi - yeni bir başlangıç yapıyoruz
                    if direction in self.directional_bias:
                        self.directional_bias[direction]['loss_streak_loss_usd'] = 0.0
                        self.directional_bias[direction]['consecutive_losses'] = 0
                    print(f"✅ Directional cooldown cleared for {direction.upper()} trades. Loss streak reset.")
        
        # Coin bazlı cooldown'ları azalt
        coins_to_remove = []
        for coin, cycles in self.coin_cooldowns.items():
            if cycles > 0:
                self.coin_cooldowns[coin] = cycles - 1
                print(f"⏱️ {coin} coin cooldown: {cycles} → {self.coin_cooldowns[coin]} cycles remaining")
                if self.coin_cooldowns[coin] == 0:
                    coins_to_remove.append(coin)
                    print(f"✅ Coin cooldown cleared for {coin}.")
        
        # Sıfırlanan coin cooldown'larını temizle
        for coin in coins_to_remove:
            del self.coin_cooldowns[coin]
        
        if self.relaxed_countertrend_cycles > 0:
            self.relaxed_countertrend_cycles -= 1
            if self.relaxed_countertrend_cycles == 0:
                print(f"✅ Relaxed counter-trend mode cleared.")
        if self.counter_trend_cooldown > 0:
            self.counter_trend_cooldown -= 1
            if self.counter_trend_cooldown == 0:
                print(f"✅ Counter-trend cooldown cleared.")

    def get_trend_following_strength(self, coin: str, signal: str) -> Dict[str, Any]:
        """
        Hibrit yaklaşım: Trend-following gücünü 15m dahil olarak belirler
        (Confidence ve margin ayarlaması YOK, sadece bilgilendirme amaçlı)
        
        Returns:
            {
                'strength': 'STRONG' | 'MEDIUM' | 'WEAK' | None,
                'alignment_info': str,
                'trends': {
                    '1h': str,
                    '15m': str,
                    '3m': str
                }
            }
        
        Mantık:
        - STRONG: 1h+15m+3m hepsi aynı yönde
        - MEDIUM: 1h+15m aynı yönde (3m farklı) VEYA 1h+3m aynı yönde (15m farklı)
        - WEAK: Sadece 1h aynı yönde (15m ve 3m farklı)
        """
        try:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            indicators_15m = self.market_data.get_technical_indicators(coin, '15m')
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            
            if 'error' in indicators_htf or 'error' in indicators_15m or 'error' in indicators_3m:
                return None
            
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            price_15m = indicators_15m.get('current_price')
            ema20_15m = indicators_15m.get('ema_20')
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            
            # Validasyon
            if not all(isinstance(x, (int, float)) for x in [price_htf, ema20_htf, price_15m, ema20_15m, price_3m, ema20_3m]):
                return None
            
            # Trend yönleri
            trend_1h = "BULLISH" if price_htf > ema20_htf else "BEARISH"
            trend_15m = "BULLISH" if price_15m > ema20_15m else "BEARISH"
            trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
            
            # Sinyal yönü
            signal_direction = "BULLISH" if signal == 'buy_to_enter' else "BEARISH"
            
            # Counter-trend kontrolü: 1h ile sinyal zıt ise None döndür
            if trend_1h != signal_direction:
                return None
            
            # Trend-following güç seviyesi belirleme
            # Mantık: 1h+15m+3m = STRONG, 1h+15m = MEDIUM, 1h+3m = MEDIUM, sadece 1h = WEAK
            if trend_1h == trend_15m == trend_3m == signal_direction:
                # STRONG: 1h + 15m + 3m hepsi aynı yönde
                return {
                    'strength': 'STRONG',
                    'alignment_info': f"Perfect alignment: {HTF_LABEL}+15m+3m all {signal_direction}",
                    'trends': {
                        '1h': trend_1h,
                        '15m': trend_15m,
                        '3m': trend_3m
                    }
                }
            elif trend_1h == trend_15m == signal_direction:
                # MEDIUM_15: 1h + 15m aynı (3m farklı)
                return {
                    'strength': 'MEDIUM_15',
                    'alignment_info': f"Moderate: {HTF_LABEL}+15m {signal_direction} (3m {trend_3m})",
                    'trends': {
                        '1h': trend_1h,
                        '15m': trend_15m,
                        '3m': trend_3m
                    }
                }
            elif trend_1h == trend_3m == signal_direction:
                # MEDIUM_3: 1h + 3m aynı (15m farklı)
                return {
                    'strength': 'MEDIUM_3',
                    'alignment_info': f"Moderate: {HTF_LABEL}+3m {signal_direction} (15m {trend_15m})",
                    'trends': {
                        '1h': trend_1h,
                        '15m': trend_15m,
                        '3m': trend_3m
                    }
                }
            else:
                # WEAK: Sadece 1h aynı (15m ve 3m farklı)
                return {
                    'strength': 'WEAK',
                    'alignment_info': f"Weak: Only {HTF_LABEL} {signal_direction} (15m {trend_15m}, 3m {trend_3m})",
                    'trends': {
                        '1h': trend_1h,
                        '15m': trend_15m,
                        '3m': trend_3m
                    }
                }
                
        except Exception as e:
            print(f"⚠️ Trend-following strength detection error for {coin}: {e}")
            return None

    def apply_directional_bias(self, signal: str, confidence: float, bias_metrics: Dict[str, Dict[str, Any]], current_trend: str) -> float:
        side = 'long' if signal == 'buy_to_enter' else 'short'
        stats = bias_metrics.get(side)
        if not stats:
            return confidence

        original_confidence = confidence
        adjusted_confidence = confidence

        # Hafifletilmiş caution penalty: 0.8 yerine 0.95
        if stats.get('caution_active'):
            adjusted_confidence = max(adjusted_confidence * 0.95, adjusted_confidence - 0.03)

        trend_lower = current_trend.lower() if isinstance(current_trend, str) else 'unknown'

        if trend_lower == 'neutral':
            adjusted_confidence *= Config.DIRECTIONAL_NEUTRAL_MULTIPLIER
        elif trend_lower == 'bullish':
            if side == 'long':
                adjusted_confidence *= Config.DIRECTIONAL_BULLISH_LONG_MULTIPLIER
            elif side == 'short':
                adjusted_confidence *= Config.DIRECTIONAL_BULLISH_SHORT_MULTIPLIER
        elif trend_lower == 'bearish':
            if side == 'long':
                adjusted_confidence *= Config.DIRECTIONAL_BEARISH_LONG_MULTIPLIER
            elif side == 'short':
                adjusted_confidence *= Config.DIRECTIONAL_BEARISH_SHORT_MULTIPLIER

        # Hafifletilmiş rolling avg penalty: 0.93 yerine 0.97
        rolling_avg = stats.get('rolling_avg', 0.0)
        if rolling_avg < 0:
            adjusted_confidence = max(adjusted_confidence * 0.97, adjusted_confidence - 0.02)

        # Minimum confidence floor: Orijinal değerin %90'ı altına düşmesin
        min_floor = original_confidence * 0.90
        adjusted_confidence = max(adjusted_confidence, min_floor, Config.MIN_CONFIDENCE)

        return adjusted_confidence

    def get_directional_bias_metrics(self) -> Dict[str, Dict[str, Any]]:
        metrics = {}
        for side, stats in self.directional_bias.items():
            rolling_list = list(stats['rolling'])
            rolling_sum = sum(rolling_list)
            rolling_avg = (rolling_sum / len(rolling_list)) if rolling_list else 0.0
            
            # Calculate win rate based on profit/loss amounts (not trade counts)
            # Win Rate = Total Profit / (|Total Profit| + |Total Loss|) * 100
            total_profit = sum(pnl for pnl in rolling_list if pnl > 0)
            total_loss = abs(sum(pnl for pnl in rolling_list if pnl < 0))
            
            if total_profit + total_loss > 0:
                win_rate = (total_profit / (total_profit + total_loss)) * 100
            else:
                win_rate = 0.0
            
            metrics[side] = {
                'net_pnl': stats['net_pnl'],
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': win_rate,  # Added win_rate based on profit/loss amounts
                'rolling_sum': rolling_sum,
                'rolling_avg': rolling_avg,
                'consecutive_losses': stats['consecutive_losses'],
                'consecutive_wins': stats.get('consecutive_wins', 0),
                'caution_active': stats.get('caution_active', False),
                'caution_win_progress': stats.get('caution_win_progress', 0)
            }
        return metrics

    def update_trend_state(
        self,
        coin: str,
        indicators_htf: Dict[str, Any],
        indicators_3m: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        price_htf = indicators_htf.get('current_price')
        ema20_htf = indicators_htf.get('ema_20')

        if not isinstance(price_htf, (int, float)) or not isinstance(ema20_htf, (int, float)) or ema20_htf == 0:
            return {'trend': 'unknown', 'recent_flip': False, 'last_flip_cycle': None}

        delta = (price_htf - ema20_htf) / ema20_htf
        price_neutral = abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT
        current_trend = 'neutral' if price_neutral else ('bullish' if delta > 0 else 'bearish')

        if indicators_3m and isinstance(indicators_3m, dict) and 'error' not in indicators_3m:
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20', price_3m)
            rsi_3m = indicators_3m.get('rsi_14', indicators_3m.get('rsi_7', 50))

            if isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)) and isinstance(rsi_3m, (int, float)):
                intraday_trend = 'bullish' if price_3m >= ema20_3m else 'bearish'
                if current_trend == 'bearish' and intraday_trend == 'bullish' and rsi_3m >= Config.INTRADAY_NEUTRAL_RSI_HIGH:
                    current_trend = 'neutral'
                elif current_trend == 'bullish' and intraday_trend == 'bearish' and rsi_3m <= Config.INTRADAY_NEUTRAL_RSI_LOW:
                    current_trend = 'neutral'

                if current_trend == 'neutral':
                    if price_htf <= ema20_htf and price_3m <= ema20_3m and rsi_3m <= Config.TREND_SHORT_RSI_THRESHOLD:
                        current_trend = 'bearish'
                    elif price_htf >= ema20_htf and price_3m >= ema20_3m and rsi_3m >= Config.TREND_LONG_RSI_THRESHOLD:
                        current_trend = 'bullish'

        record = self.trend_state.get(coin, {'trend': current_trend, 'last_flip_cycle': self.current_cycle_number, 'last_flip_direction': current_trend})
        previous_trend = record.get('trend', current_trend)
        recent_flip = False

        if previous_trend != current_trend:
            record['trend'] = current_trend
            if current_trend != 'neutral':
                record['last_flip_cycle'] = self.current_cycle_number
                record['last_flip_direction'] = current_trend
                recent_flip = True
        else:
            last_flip_cycle = record.get('last_flip_cycle', self.current_cycle_number)
            if current_trend != 'neutral' and self.current_cycle_number - last_flip_cycle <= self.trend_flip_cooldown:
                recent_flip = True

        record['last_seen_cycle'] = self.current_cycle_number
        self.trend_state[coin] = record
        return {
            'trend': current_trend,
            'recent_flip': recent_flip,
            'last_flip_cycle': record.get('last_flip_cycle'),
            'last_flip_direction': record.get('last_flip_direction')
        }

    def get_recent_trend_flip_summary(self) -> List[str]:
        summaries = []
        guard_window = self.trend_flip_cooldown
        history_window = max(guard_window, getattr(self, 'trend_flip_history_window', guard_window))
        entries: List[Tuple[int, str]] = []
        for coin, record in self.trend_state.items():
            last_flip_cycle = record.get('last_flip_cycle')
            if last_flip_cycle is None:
                continue
            cycles_ago = self.current_cycle_number - last_flip_cycle
            if cycles_ago < 0 or cycles_ago > history_window:
                continue
            trend_label = record.get('trend', 'unknown').upper()
            if cycles_ago <= guard_window:
                status = "GUARD"
            else:
                status = "RECENT"
            if cycles_ago == 0:
                cycles_text = "current cycle"
            elif cycles_ago == 1:
                cycles_text = "1 cycle ago"
            else:
                cycles_text = f"{cycles_ago} cycles ago"
            direction_note = record.get('last_flip_direction', trend_label)
            entries.append((cycles_ago, f"{coin}: {direction_note} since cycle {last_flip_cycle} ({status}, {cycles_text})"))
        entries.sort(key=lambda x: x[0])
        summaries = [text for _, text in entries]
        return summaries

    def load_cycle_history(self) -> List[Dict]:
        history = safe_file_read(self.cycle_history_file, default_data=[]); print(f"✅ Loaded {len(history)} cycles."); return history
    def add_to_cycle_history(
        self,
        cycle_number: int,
        prompt: str,
        thoughts: str,
        decisions: Dict,
        status: str = "ai_decision",
        metadata: Optional[Dict[str, Any]] = None
    ):
        prompt_summary = "N/A"
        if isinstance(prompt, str) and prompt not in (None, "N/A"):
            # For JSON prompts, try to extract a meaningful summary
            # Check for all JSON sections
            json_sections = [
                ("COUNTER_TRADE_ANALYSIS (JSON):", "Counter-trade analysis"),
                ("TREND_REVERSAL_DATA (JSON):", "Trend reversal"),
                ("ENHANCED_CONTEXT (JSON):", "Enhanced context"),
                ("DIRECTIONAL_BIAS (JSON):", "Directional bias"),
                ("COOLDOWN_STATUS (JSON):", "Cooldown status"),
                ("TREND_FLIP_GUARD (JSON):", "Trend flip guard"),
                ("POSITION_SLOTS (JSON):", "Position slots"),
                ("MARKET_DATA (JSON):", "Market data"),
                ("HISTORICAL_CONTEXT (JSON):", "Historical context"),
                ("RISK_STATUS (JSON):", "Risk status"),
                ("PORTFOLIO (JSON):", "Portfolio")
            ]
            
            found_sections = [name for marker, name in json_sections if marker in prompt]
            
            if found_sections:
                # JSON format prompt - create a structured summary
                try:
                    section_count = len(found_sections)
                    if section_count <= 3:
                        summary_text = ", ".join(found_sections)
                    else:
                        summary_text = f"{', '.join(found_sections[:3])} + {section_count - 3} more"
                    prompt_summary = f"JSON Format ({section_count} sections): {summary_text} | " + prompt[:200] + "..."
                except:
                    prompt_summary = prompt[:300] + "..." if len(prompt) > 300 else prompt
            else:
                # Text format prompt - use original truncation
                prompt_summary = prompt[:300] + "..." if len(prompt) > 300 else prompt
        
        # Add cooldown information to cycle data
        cooldown_info = {
            'directional_cooldowns': dict(self.directional_cooldowns),
            'relaxed_countertrend_cycles': self.relaxed_countertrend_cycles,
            'counter_trend_cooldown': self.counter_trend_cooldown,
            'coin_cooldowns': dict(self.coin_cooldowns)
        }
        
        cycle_data = {
            'cycle': cycle_number,
            'timestamp': datetime.now().isoformat(),
            'user_prompt_summary': prompt_summary,
            'chain_of_thoughts': thoughts,
            'decisions': decisions,
            'status': status,
            'cooldown_status': cooldown_info  # Always include cooldown status
        }
        if metadata:
            cycle_data['metadata'] = metadata
        self.cycle_history.append(cycle_data); self.cycle_history = self.cycle_history[-self.max_cycle_history:]
        safe_file_write(self.cycle_history_file, self.cycle_history); print(f"✅ Saved cycle {cycle_number} (Total: {len(self.cycle_history)})")
    def update_prices(self, new_prices: Dict[str, float], increment_loss_counters: bool = True):
        """Updates prices and recalculates total value."""
        total_unrealized_pnl = 0.0
        for coin, price in new_prices.items():
            if coin in self.positions and isinstance(price, (int, float)) and price > 0:
                pos = self.positions[coin]
                
                # Update current_price (use Spot price, but in live mode markPrice from Binance is preferred)
                # For live mode, current_price should already be set from sync_live_account() (markPrice)
                # We update it with Spot price only if it's significantly different (fallback)
                if self.is_live_trading:
                    # In live mode, prefer keeping Binance markPrice if available
                    existing_price = pos.get('current_price', 0)
                    if existing_price > 0:
                        # Keep Binance markPrice, but update if Spot price is significantly different (>0.1%)
                        price_diff_pct = abs(price - existing_price) / existing_price if existing_price > 0 else 1.0
                        if price_diff_pct > 0.001:  # More than 0.1% difference
                            # Use Spot price as fallback if markPrice seems stale
                            pos['current_price'] = price
                        # else: keep existing markPrice
                    else:
                        # No existing price, use Spot price
                        pos['current_price'] = price
                else:
                    # Simulation mode: always use Spot price
                    pos['current_price'] = price
                
                # CRITICAL FIX: In live mode, preserve Binance unrealized_pnl (includes funding fees)
                # In simulation mode, calculate manually
                if self.is_live_trading:
                    # Live mode: Keep Binance unrealized_pnl if available (includes funding fees, commissions, etc.)
                    existing_pnl = pos.get('unrealized_pnl', 0.0)
                    if isinstance(existing_pnl, (int, float)) and existing_pnl != 0.0:
                        # Use Binance value (more accurate, includes funding fees)
                        pnl = existing_pnl
                        pos['unrealized_pnl'] = pnl
                    else:
                        # Fallback: calculate manually if Binance value not available
                        entry = pos['entry_price']
                        qty = pos['quantity']
                        direction = pos.get('direction', 'long')
                        pnl = (price - entry) * qty if direction == 'long' else (entry - price) * qty
                        pos['unrealized_pnl'] = pnl
                else:
                    # Simulation mode: calculate manually
                    entry = pos['entry_price']
                    qty = pos['quantity']
                    direction = pos.get('direction', 'long')
                    pnl = (price - entry) * qty if direction == 'long' else (entry - price) * qty
                    pos['unrealized_pnl'] = pnl
                
                total_unrealized_pnl += pos.get('unrealized_pnl', 0.0)
                
                if increment_loss_counters:
                    direction = pos.get('direction', 'unknown')
                    pnl_for_counter = pos.get('unrealized_pnl', 0.0)
                    if pnl_for_counter <= 0:
                        pos['loss_cycle_count'] = pos.get('loss_cycle_count', 0) + 1
                        new_count = pos['loss_cycle_count']
                        if new_count in (5, 8, 10):
                            print(f"⏳ LOSS CYCLE WATCH: {coin} {direction} negative for {new_count} cycles (PnL ${pnl_for_counter:.2f}).")
                    else:
                        pos['loss_cycle_count'] = 0
            elif coin in self.positions: print(f"⚠️ Invalid price for {coin}: {price}. PnL skip.")

        # Calculate total value correctly
        # In live mode, prefer syncing from Binance (done in sync_live_account)
        # In simulation mode or when Binance data unavailable, calculate manually
        if not self.positions:
            # No positions, total value = available cash (or totalWalletBalance if available in live mode)
            if self.is_live_trading and self.order_executor and self.order_executor.is_live():
                # Try to get totalWalletBalance from Binance
                try:
                    overview = self.order_executor.get_account_overview()
                    if overview and overview.get("totalWalletBalance") and overview.get("totalWalletBalance") > 0:
                        self.total_value = float(overview["totalWalletBalance"])
                    else:
                        self.total_value = self.current_balance
                except:
                    self.total_value = self.current_balance
            else:
                self.total_value = self.current_balance
        else:
            # With positions: Calculate margin used (for cross margin, calculate from notional/leverage)
            # OPTIMIZATION: In live mode, total_value was already calculated in sync_live_account()
            # Only recalculate in simulation mode or if sync_live_account() wasn't called
            if self.is_live_trading and self.order_executor and self.order_executor.is_live():
                # In live mode, total_value should already be set by sync_live_account()
                # But we recalculate here to ensure consistency after price updates
                # Use the unrealized_pnl values we just updated (which preserve Binance values)
                total_margin_used = 0.0
                total_unrealized_pnl = 0.0
                
                for pos in self.positions.values():
                    # Get unrealized PnL (should be Binance value in live mode)
                    pnl = pos.get('unrealized_pnl', 0.0)
                    if isinstance(pnl, (int, float)):
                        total_unrealized_pnl += pnl
                    
                    # Get margin (for cross margin, margin_usd might be 0, so calculate from notional/leverage)
                    margin = pos.get('margin_usd', 0.0)
                    if margin <= 0:
                        notional = pos.get('notional_usd', 0.0)
                        leverage = pos.get('leverage', 1)
                        if notional > 0 and leverage > 0:
                            margin = notional / leverage
                    if isinstance(margin, (int, float)) and margin > 0:
                        total_margin_used += margin
                
                # Total value = Available cash + Margin used + Unrealized PnL
                # In live mode, unrealized_pnl should be from Binance (includes funding fees)
                self.total_value = self.current_balance + total_margin_used + total_unrealized_pnl
            else:
                # Simulation mode: calculate manually
                total_margin_used = 0.0
                total_unrealized_pnl = 0.0
                
                for pos in self.positions.values():
                    # Get unrealized PnL (manually calculated in simulation mode)
                    pnl = pos.get('unrealized_pnl', 0.0)
                    if isinstance(pnl, (int, float)):
                        total_unrealized_pnl += pnl
                    
                    # Get margin (for cross margin, margin_usd might be 0, so calculate from notional/leverage)
                    margin = pos.get('margin_usd', 0.0)
                    if margin <= 0:
                        notional = pos.get('notional_usd', 0.0)
                        leverage = pos.get('leverage', 1)
                        if notional > 0 and leverage > 0:
                            margin = notional / leverage
                    if isinstance(margin, (int, float)) and margin > 0:
                        total_margin_used += margin
                
                # Total value = Available cash + Margin used + Unrealized PnL
                self.total_value = self.current_balance + total_margin_used + total_unrealized_pnl

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

    # --- NEW: Enhanced Auto TP/SL Check with Advanced Exit Strategies ---
    def check_and_execute_tp_sl(self, current_prices: Dict[str, float]):
        """Checks if any open position hit TP or SL and closes them automatically with enhanced exit strategies.
        
        This function is called every 30 seconds by the monitoring loop:
        - All TP/SL decisions are made by this monitoring (like simulation mode)
        - No Binance TP/SL orders - all managed by this loop
        - Kademeli margin-based stop loss is checked and positions are closed accordingly
        """
        # Enhanced exit strategy control - check if enabled
        if hasattr(self, 'bot') and not self.bot.enhanced_exit_enabled:
            print("⏸️ Enhanced exit strategy paused during cycle")
            return False
            
        # All TP/SL decisions made by 30-second monitoring (like simulation mode)
        # No Binance TP/SL orders - all managed by monitoring loop
        print(f"🔎 Checking for TP/SL triggers (30-second monitoring mode)")
        
        closed_positions = [] # Keep track of positions closed in this check
        updated_stops = [] # Track positions with updated trailing stops
        state_changed = False
        
        for coin, position in list(self.positions.items()): # Iterate over a copy for safe deletion
            if coin not in current_prices or not isinstance(current_prices[coin], (int, float)) or current_prices[coin] <= 0:
                continue # Skip if price is invalid

            current_price = current_prices[coin]
            exit_plan = position.get('exit_plan', {})
            tp = exit_plan.get('profit_target')
            sl = exit_plan.get('stop_loss')
            direction = position.get('direction', 'long')
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            # Calculate margin_used properly - try multiple fallback methods
            margin_used = position.get('margin_usd')
            if margin_used is None or margin_used <= 0:
                # Fallback 1: Calculate from notional and leverage
                notional = position.get('notional_usd', 0)
                leverage = position.get('leverage', 1)
                if notional > 0 and leverage > 0:
                    margin_used = notional / leverage
                else:
                    # Fallback 2: Calculate from entry_price and quantity
                    if entry_price > 0 and quantity > 0:
                        notional = entry_price * quantity
                        leverage = position.get('leverage', 10)
                        margin_used = notional / leverage
                    else:
                        margin_used = 0
            
            # Debug log if margin_used is still 0
            if margin_used <= 0:
                print(f"⚠️ Warning: margin_used is 0 for {coin}. Position data: margin_usd={position.get('margin_usd')}, notional={position.get('notional_usd')}, leverage={position.get('leverage')}, entry={entry_price}, qty={quantity}")

            close_reason = None

            # Check TP
            # Convert tp/sl to float for safe comparison, handle potential errors
            try: tp = float(tp) if tp is not None else None
            except (ValueError, TypeError): tp = None
            try: sl = float(sl) if sl is not None else None
            except (ValueError, TypeError): sl = None

            # Enhanced exit strategy check - REAL-TIME ENTEGRASYON
            exit_decision = self.enhanced_exit_strategy(position, current_price)
            
            # Handle enhanced exit strategy signals - ANINDA İŞLEME
            if exit_decision['action'] == 'close_position':
                # Enhanced exit strategy wants to close the position completely
                close_reason = exit_decision['reason']
                print(f"⚡ ENHANCED EXIT CLOSE {coin} ({direction}): {close_reason} at price ${format_num(current_price, 4)}")
                state_changed = True
            elif exit_decision['action'] == 'partial_close':
                # Partial profit taking - ANINDA İŞLEME
                close_percent = exit_decision['percent']
                if self.is_live_trading:
                    live_result = self.execute_live_partial_close(
                        coin=coin,
                        position=position,
                        close_percent=close_percent,
                        current_price=current_price,
                        reason=exit_decision['reason']
                    )
                    if not live_result.get('success'):
                        print(f"🚫 Live partial close failed for {coin}: {live_result.get('error', 'unknown_error')}")
                        continue
                    history_entry = live_result.get('history_entry')
                    if history_entry:
                        self.add_to_history(history_entry)
                    print(f"⚡ PARTIAL CLOSE {coin} ({direction}) [LIVE]: {exit_decision['reason']} ({close_percent*100:.0f}% / PnL ${format_num(live_result.get('pnl', 0), 2)})")
                    state_changed = True
                    # Sync account balance after partial close in live mode
                    try:
                        self.sync_live_account()
                        print(f"✅ Account balance synced after partial close of {coin}")
                    except Exception as sync_exc:
                        print(f"⚠️ Failed to sync account after partial close: {sync_exc}")
                    continue

                close_quantity = quantity * close_percent
                
                if direction == 'long': profit = (current_price - entry_price) * close_quantity
                else: profit = (entry_price - current_price) * close_quantity
                
                # Update position quantity
                position['quantity'] = quantity * (1 - close_percent)
                position['margin_usd'] = margin_used * (1 - close_percent)
                position['notional_usd'] = position['notional_usd'] * (1 - close_percent)
                
                # Add profit to balance
                self.current_balance += (margin_used * close_percent + profit)
                
                print(f"⚡ PARTIAL CLOSE {coin} ({direction}): {exit_decision['reason']} - Closed {close_percent*100}% at price ${format_num(current_price, 4)}")
                print(f"   Partial PnL: ${format_num(profit, 2)}")
                
                history_entry = {
                    "symbol": coin, "direction": direction, "entry_price": entry_price, "exit_price": current_price,
                    "quantity": close_quantity, "notional_usd": position.get('notional_usd', 'N/A') * close_percent, 
                    "pnl": profit, "entry_time": position['entry_time'], "exit_time": datetime.now().isoformat(),
                    "leverage": position.get('leverage', 'N/A'), "close_reason": exit_decision['reason']
                }
                self.add_to_history(history_entry)
                state_changed = True
                continue  # Continue with remaining position
            
            elif exit_decision['action'] == 'update_stop':
                # Update trailing stop - ANINDA GÜNCELLEME
                updated_stops.append(coin)
                new_stop = exit_decision['new_stop']
                exit_plan['stop_loss'] = new_stop
                print(f"📈 TRAILING STOP UPDATE {coin}: New stop at ${format_num(new_stop, 4)}")
                
                # No Binance orders - stop loss updated in exit_plan, will be monitored by 30-second loop
                
                state_changed = True
                continue
            
            # Traditional TP/SL checks (only if no enhanced exit triggered)
            if close_reason is None and tp is not None:
                if direction == 'long' and current_price >= tp: close_reason = f"Profit Target ({tp}) hit"
                elif direction == 'short' and current_price <= tp: close_reason = f"Profit Target ({tp}) hit"

            # Check SL (only if TP not hit)
            # First check exit_plan stop_loss, then fallback to margin-based kademeli stop loss
            if close_reason is None:
                # Check exit_plan stop_loss first
                if sl is not None:
                    if direction == 'long' and current_price <= sl: 
                        close_reason = f"Stop Loss ({sl}) hit"
                    elif direction == 'short' and current_price >= sl: 
                        close_reason = f"Stop Loss ({sl}) hit"
                
                # If no exit_plan stop_loss or it didn't trigger, check margin-based kademeli stop loss
                # Only check if margin_used is valid (> 0)
                if close_reason is None and quantity > 0 and margin_used > 0:
                    # Calculate margin-based stop loss using kademeli loss cutting (same as entry)
                    loss_multiplier = 0.03  # Default: %3 for margin >= 50
                    if margin_used < 30:
                        loss_multiplier = 0.07  # %7 for margin < 30
                    elif margin_used < 40:
                        loss_multiplier = 0.05  # %5 for margin 30-40
                    elif margin_used < 50:
                        loss_multiplier = 0.05  # %5 for margin 40-50
                    else:
                        loss_multiplier = 0.03  # %3 for margin >= 50
                    
                    loss_threshold_usd = margin_used * loss_multiplier
                    
                    # Only proceed if loss_threshold_usd is valid
                    if loss_threshold_usd > 0:
                        # Calculate stop loss price from loss threshold
                        if direction == 'long':
                            margin_based_stop_loss = entry_price - (loss_threshold_usd / quantity)
                        else:  # short
                            margin_based_stop_loss = entry_price + (loss_threshold_usd / quantity)
                        
                        # Kademeli stop loss is calculated correctly based on margin and loss_multiplier
                        # No minimum distance adjustment needed
                        
                        # Check if current price hit margin-based stop loss
                        if direction == 'long' and current_price <= margin_based_stop_loss:
                            close_reason = f"Margin-based Stop Loss ({format_num(margin_based_stop_loss, 4)}) hit (${loss_threshold_usd:.2f} loss limit, {loss_multiplier*100:.1f}% of ${margin_used:.2f} margin)"
                        elif direction == 'short' and current_price >= margin_based_stop_loss:
                            close_reason = f"Margin-based Stop Loss ({format_num(margin_based_stop_loss, 4)}) hit (${loss_threshold_usd:.2f} loss limit, {loss_multiplier*100:.1f}% of ${margin_used:.2f} margin)"

            # Execute Close if triggered
            if close_reason:
                print(f"⚡ AUTO-CLOSE {coin} ({direction}): {close_reason} at price ${format_num(current_price, 4)}")

                if self.is_live_trading:
                    print(f"🔄 Executing LIVE close on Binance for {coin}...")
                    live_result = self.execute_live_close(
                        coin=coin,
                        position=position,
                        current_price=current_price,
                        reason=close_reason
                    )
                    if not live_result.get('success'):
                        print(f"🚫 Live auto-close failed for {coin}: {live_result.get('error', 'unknown_error')}")
                        continue
                    
                    # Log Binance order details
                    order_id = live_result.get('order', {}).get('orderId')
                    executed_qty = live_result.get('executed_qty', 0)
                    avg_price = live_result.get('avg_price', 0)
                    print(f"✅ Binance CLOSE order executed for {coin}: orderId={order_id}, qty={format_num(executed_qty, 4)}, avgPrice=${format_num(avg_price, 4)}")
                    
                    history_entry = live_result.get('history_entry')
                    if history_entry:
                        self.add_to_history(history_entry)
                    print(f"   Live Closed PnL: ${format_num(live_result.get('pnl', 0), 2)}")
                    closed_positions.append(coin)
                    state_changed = True
                    # Sync account balance after closing position in live mode
                    try:
                        self.sync_live_account()
                        print(f"✅ Account balance synced after closing {coin}")
                    except Exception as sync_exc:
                        print(f"⚠️ Failed to sync account after close: {sync_exc}")
                    continue

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
                state_changed = True

        if closed_positions:
             print(f"✅ Auto-closed positions: {', '.join(closed_positions)}")
        if updated_stops:
             print(f"📈 Updated trailing stops: {', '.join(updated_stops)}")

        if state_changed:
            self.save_state()
             
        return len(closed_positions) > 0  # Indicate if any positions were closed

    def calculate_dynamic_position_size(self, coin: str, confidence: float, market_regime: str, trend_strength: int) -> float:
        """Calculate dynamic position size based on multiple factors"""
        base_risk = 25.0  # Reduced maximum risk to $25
        
        # Confidence factor
        confidence_multiplier = confidence
        
        # Market regime factor
        if market_regime == "BULLISH":
            regime_multiplier = 1.2
        elif market_regime == "BEARISH":
            regime_multiplier = 0.8
        else:
            regime_multiplier = 1.0
        
        # Trend strength factor
        trend_multiplier = 1.0 + (trend_strength * 0.1)
        
        # Volume consideration
        try:
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            volume = indicators_3m.get('volume', 0)
            avg_volume = indicators_3m.get('avg_volume', 0)
            
            # Volume multiplier: higher volume = higher confidence
            if volume > avg_volume * 2:
                volume_multiplier = 1.2
            elif volume > avg_volume:
                volume_multiplier = 1.1
            else:
                volume_multiplier = 0.8  # Penalize low volume
        except:
            volume_multiplier = 1.0
        
        # Dynamic risk calculation
        dynamic_risk = base_risk * confidence_multiplier * regime_multiplier * trend_multiplier * volume_multiplier
        
        # Maximum risk limit
        return min(dynamic_risk, 25.0)

    def get_profit_levels_by_notional(self, notional_usd: float) -> Dict[str, float]:
        """Get dynamic profit levels based on notional size"""
        if notional_usd < 200:
            # Small positions: aggressive profit taking
            return {
                'level1': 0.006,  # %0.7
                'level2': 0.007,  # %0.9
                'level3': 0.08,  # %1.1
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        elif notional_usd < 300:
            # Medium positions: balanced profit taking
            return {
                'level1': 0.005,  # %0.7
                'level2': 0.006,  # %0.9
                'level3': 0.007,  # %1.1
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        elif notional_usd < 400:
            # Large positions: conservative profit taking
            return {
                'level1': 0.004,  # %0.6
                'level2': 0.005,  # %0.8
                'level3': 0.006,  # %1.0
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        elif notional_usd < 500:
            # xLarge positions: conservative profit taking
            return {
                'level1': 0.003,  # %0.5
                'level2': 0.004,  # %0.7
                'level3': 0.005,  # %0.9
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        elif notional_usd < 600:
            # xxLarge positions: conservative profit taking
            return {
                'level1': 0.002,  # %0.
                'level2': 0.003,  # %0.6
                'level3': 0.004,  # %0.8
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }
        else:
            # Very large positions: very conservative profit taking
            return {
                'level1': 0.002,  # %0.3
                'level2': 0.004,  # %0.5
                'level3': 0.006,  # %0.7
                'take1': 0.25,    # %25 profit al
                'take2': 0.50,    # %50 profit al
                'take3': 0.75     # %75 profit al
            }

    def get_dynamic_stop_loss_percentage(self, total_portfolio_value: float) -> float:
        """Get dynamic stop-loss percentage based on portfolio value"""
        if total_portfolio_value < 300:
            return 0.07  # %1.0
        elif total_portfolio_value < 400:
            return 0.006 # %0.8
        elif total_portfolio_value < 500:
            return 0.005 # %0.7
        else:
            return 0.004 # %0.5

    def enhanced_exit_strategy(self, position: Dict, current_price: float) -> Dict[str, Any]:
        """Enhanced exit strategy with dynamic profit taking and KADEMELİ loss cutting"""
        entry_price = position.get('entry_price')
        if entry_price is None:
            entry_price = position.get('current_price', 0)
            position['entry_price'] = entry_price
        direction = position.get('direction', 'long')
        exit_plan = self._ensure_exit_plan(position, position.get('exit_plan'))
        stop_loss = exit_plan.get('stop_loss')
        profit_target = exit_plan.get('profit_target')
        notional_usd = position.get('notional_usd', 0)
        
        exit_decision = {"action": "hold", "reason": "No exit trigger"}
        
        current_margin = position.get('margin_usd', 0)
        margin_used = position.get('margin_usd', position.get('notional_usd', 0) / max(position.get('leverage', 1), 1))
        loss_cycle_count = position.get('loss_cycle_count', 0)
        unrealized_pnl = position.get('unrealized_pnl', 0)
        if loss_cycle_count >= 10 and unrealized_pnl <= 0:
            reason = f"Position negative for {loss_cycle_count} cycles"
            print(f"⏳ Extended loss exit: {position['symbol']} {direction} closed ({reason}).")
            return {"action": "close_position", "reason": reason}
        
        # --- KADEMELİ LOSS CUTTING MEKANİZMASI (Margin tabanlı) ---
        loss_multiplier = 0.03  # Default: %3 for margin >= 50
        if margin_used < 30:
            loss_multiplier = 0.07  # %7 for margin < 30
        elif margin_used < 40:
            loss_multiplier = 0.05  # %5 for margin 30-40
        elif margin_used < 50:
            loss_multiplier = 0.05  # %5 for margin 40-50
        else:
            loss_multiplier = 0.03  # %3 for margin >= 50

        loss_threshold_usd = margin_used * loss_multiplier
        
        if direction == 'long':
            unrealized_loss_usd = max(0.0, (entry_price - current_price) * position['quantity'])
        else:
            unrealized_loss_usd = max(0.0, (current_price - entry_price) * position['quantity'])

        if unrealized_loss_usd >= loss_threshold_usd and loss_threshold_usd > 0:
            print(f"🛑 KADEMELİ LOSS CUTTING: {direction} {position['symbol']} ${unrealized_loss_usd:.2f} zarar (eşik: ${loss_threshold_usd:.2f}). Pozisyon kapatılıyor.")
            return {"action": "close_position", "reason": f"Margin-based loss cut ${unrealized_loss_usd:.2f} ≥ ${loss_threshold_usd:.2f}"}
        
        # Get dynamic profit levels based on notional size
        profit_levels = self.get_profit_levels_by_notional(notional_usd)
        level1 = profit_levels['level1']
        level2 = profit_levels['level2']
        level3 = profit_levels['level3']
        take1 = profit_levels['take1']
        take2 = profit_levels['take2']
        take3 = profit_levels['take3']
        
        print(f"📊 Dynamic profit levels for ${notional_usd:.2f} notional: {level1*100:.1f}%/{level2*100:.1f}%/{level3*100:.1f}%")
        
        if direction == 'long':
            unrealized_pnl_usd = max(0.0, (current_price - entry_price) * position['quantity'])
            unrealized_pnl_percent = (unrealized_pnl_usd / notional_usd) if notional_usd else 0.0
            
            # Dynamic Profit Taking Levels based on notional size
            if unrealized_pnl_percent >= level3:  # Level 3 profit - take 75%
                take_profit_percent = take3
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level3*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            elif unrealized_pnl_percent >= level2:  # Level 2 profit - take 50%
                take_profit_percent = take2
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level2*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            elif unrealized_pnl_percent >= level1:  # Level 1 profit - take 25%
                take_profit_percent = take1
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level1*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            
            trailing_action = self._evaluate_trailing_stop(
                position=position,
                current_price=current_price,
                profit_target=profit_target,
                direction=direction,
                entry_price=entry_price,
                unrealized_pnl_percent=unrealized_pnl_percent,
                profit_levels=profit_levels
            )
            if trailing_action:
                return trailing_action
        
        elif direction == 'short':
            unrealized_pnl_usd = max(0.0, (entry_price - current_price) * position['quantity'])
            unrealized_pnl_percent = (unrealized_pnl_usd / notional_usd) if notional_usd else 0.0
            
            # Dynamic Profit Taking Levels for shorts based on notional size
            if unrealized_pnl_percent >= level3:  # Level 3 profit - take 75%
                take_profit_percent = take3
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level3*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            elif unrealized_pnl_percent >= level2:  # Level 2 profit - take 50%
                take_profit_percent = take2
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level2*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            elif unrealized_pnl_percent >= level1:  # Level 1 profit - take 25%
                take_profit_percent = take1
                adjusted_percent, force_close, reason = self._adjust_partial_sale_for_max_limit(position, take_profit_percent)
                if force_close:
                    return {"action": "close_position", "reason": reason or "Maximum limit reached during profit taking"}
                if adjusted_percent > 0:
                    return {"action": "partial_close", "percent": adjusted_percent, "reason": f"Profit taking at {level1*100:.1f}% gain ({adjusted_percent*100:.0f}%)"}
            
            trailing_action = self._evaluate_trailing_stop(
                position=position,
                current_price=current_price,
                profit_target=profit_target,
                direction=direction,
                entry_price=entry_price,
                unrealized_pnl_percent=unrealized_pnl_percent,
                profit_levels=profit_levels
            )
            if trailing_action:
                return trailing_action
        
        return exit_decision

    def _evaluate_trailing_stop(
        self,
        position: Dict[str, Any],
        current_price: float,
        profit_target: Optional[float],
        direction: str,
        entry_price: float,
        unrealized_pnl_percent: float,
        profit_levels: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate advanced trailing stop conditions based on progress, time, volume and ATR."""
        if unrealized_pnl_percent <= 0 or not isinstance(current_price, (int, float)) or current_price <= 0:
            return None

        symbol = position.get('symbol')
        exit_plan = position.get('exit_plan') or {}

        level1_threshold = 0.0
        if isinstance(profit_levels, dict):
            try:
                level1_threshold = float(profit_levels.get('level1', 0.0) or 0.0)
            except (TypeError, ValueError):
                level1_threshold = 0.0
        if unrealized_pnl_percent < max(level1_threshold * 0.5, 0.0):
            return None

        existing_stop = exit_plan.get('stop_loss')
        try:
            existing_stop = float(existing_stop) if existing_stop is not None else None
        except (TypeError, ValueError):
            existing_stop = None

        # Calculate progress toward profit target (in %)
        progress_pct = 0.0
        progress_valid = False
        if isinstance(profit_target, (int, float)) and profit_target > 0 and profit_target != entry_price:
            if direction == 'long':
                denominator = profit_target - entry_price
                if denominator > 0:
                    progress_pct = ((current_price - entry_price) / denominator) * 100
                    progress_valid = True
            elif direction == 'short':
                denominator = entry_price - profit_target
                if denominator > 0:
                    progress_pct = ((entry_price - current_price) / denominator) * 100
                    progress_valid = True
        progress_pct = max(0.0, min(progress_pct, 200.0))

        pnl_percent = max(0.0, unrealized_pnl_percent * 100.0)
        progress_score = progress_pct if progress_valid else pnl_percent

        # Time in trade (minutes)
        time_in_trade = 0.0
        entry_time_str = position.get('entry_time')
        if entry_time_str:
            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                time_in_trade = max(0.0, (datetime.now() - entry_time).total_seconds() / 60.0)
            except Exception:
                time_in_trade = 0.0

        progress_triggered = progress_score >= Config.TRAILING_PROGRESS_TRIGGER
        time_triggered = (
            time_in_trade >= Config.TRAILING_TIME_MINUTES
            and progress_score >= Config.TRAILING_TIME_PROGRESS_FLOOR
        )
        if not (progress_triggered or time_triggered):
            return None

        # Fetch current 3m indicators for ATR & volume context
        current_volume_ratio = None
        atr_value = None
        try:
            indicators_3m = self.market_data.get_technical_indicators(symbol, '3m') if self.market_data else {}
        except Exception as exc:
            print(f"⚠️ Trailing stop indicator fetch failed for {symbol}: {exc}")
            indicators_3m = {}

        if isinstance(indicators_3m, dict):
            volume_now = indicators_3m.get('volume')
            avg_volume_now = indicators_3m.get('avg_volume')
            if isinstance(volume_now, (int, float)) and isinstance(avg_volume_now, (int, float)) and avg_volume_now > 0:
                current_volume_ratio = volume_now / avg_volume_now
            atr_value = indicators_3m.get('atr_14')

        if not isinstance(atr_value, (int, float)) or atr_value <= 0:
            atr_value = position.get('entry_atr_14')
        if not isinstance(atr_value, (int, float)) or atr_value <= 0:
            atr_value = current_price * Config.TRAILING_FALLBACK_BUFFER_PCT

        entry_volume_ratio = position.get('entry_volume_ratio')
        volume_drop_triggered = False
        if isinstance(current_volume_ratio, (int, float)):
            if current_volume_ratio <= Config.TRAILING_VOLUME_ABSOLUTE_THRESHOLD:
                volume_drop_triggered = True
            elif isinstance(entry_volume_ratio, (int, float)) and entry_volume_ratio > 0:
                if current_volume_ratio <= entry_volume_ratio * Config.TRAILING_VOLUME_DROP_RATIO:
                    volume_drop_triggered = True

        min_improvement_abs = max(
            current_price * Config.TRAILING_MIN_IMPROVEMENT_PCT,
            max(Config.MIN_EXIT_PLAN_OFFSET, 1e-7)
        )
        atr_buffer = max(atr_value * Config.TRAILING_ATR_MULTIPLIER, min_improvement_abs)

        reason_tokens: List[str] = []
        if progress_triggered:
            reason_tokens.append(f"progress {progress_score:.1f}%")
        if time_triggered:
            reason_tokens.append(f"time {time_in_trade:.1f}m")
        if volume_drop_triggered and isinstance(current_volume_ratio, (int, float)):
            reason_tokens.append(f"volume {current_volume_ratio:.2f}x")

        if not reason_tokens:
            reason_tokens.append("trailing criteria met")

        new_stop: Optional[float] = None
        if direction == 'long':
            baseline_stop = current_price - atr_buffer
            baseline_stop = min(baseline_stop, current_price - min_improvement_abs)
            if progress_triggered:
                baseline_stop = max(baseline_stop, entry_price + min_improvement_abs)
            elif time_triggered:
                baseline_stop = max(baseline_stop, entry_price + Config.MIN_EXIT_PLAN_OFFSET)

            if existing_stop is not None:
                baseline_stop = max(baseline_stop, existing_stop + min_improvement_abs)

            if baseline_stop >= current_price:
                baseline_stop = current_price - min_improvement_abs

            if existing_stop is not None and baseline_stop <= existing_stop + min_improvement_abs:
                return None

            if baseline_stop <= 0:
                return None

            new_stop = baseline_stop
        else:  # short
            baseline_stop = current_price + atr_buffer
            baseline_stop = max(baseline_stop, current_price + min_improvement_abs)
            if progress_triggered:
                baseline_stop = min(baseline_stop, entry_price - min_improvement_abs)
            elif time_triggered:
                baseline_stop = min(baseline_stop, entry_price - Config.MIN_EXIT_PLAN_OFFSET)

            if existing_stop is not None:
                baseline_stop = min(baseline_stop, existing_stop - min_improvement_abs)

            if baseline_stop <= current_price:
                baseline_stop = current_price + min_improvement_abs

            if progress_triggered and baseline_stop > entry_price - min_improvement_abs:
                baseline_stop = entry_price - min_improvement_abs

            if baseline_stop <= current_price:
                return None

            if existing_stop is not None and (existing_stop - baseline_stop) <= min_improvement_abs:
                return None

            new_stop = baseline_stop

        if new_stop is None:
            return None

        new_stop = round(max(0.0, new_stop), 6)
        if direction == 'long' and new_stop >= current_price:
            return None
        if direction == 'short' and new_stop <= current_price:
            return None

        # Persist updated stop and trailing metadata
        exit_plan['stop_loss'] = new_stop
        position['exit_plan'] = exit_plan

        trailing_meta = position.setdefault('trailing', {})
        trailing_meta.update({
            'active': True,
            'last_update_cycle': getattr(self, 'current_cycle_number', None),
            'last_reason': ", ".join(reason_tokens),
            'last_stop': new_stop,
            'progress_percent': round(progress_score, 2),
            'time_in_trade_min': round(time_in_trade, 2)
        })
        if isinstance(current_volume_ratio, (int, float)):
            trailing_meta['last_volume_ratio'] = round(current_volume_ratio, 4)

        reason = f"Trailing stop tightened ({', '.join(reason_tokens)})"
        return {"action": "update_stop", "new_stop": new_stop, "reason": reason}

    def _execute_new_positions_only(
        self,
        decisions: Dict,
        valid_prices: Dict,
        cycle_number: int,
        indicator_cache: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Execute only new position entries after AI close_position signal"""
        print("🔄 Executing new positions only (after close_position signal)")
        
        # KADEMELİ POZİSYON SİSTEMİ: Cycle bazlı pozisyon limiti
        max_positions_for_cycle = self.get_max_positions_for_cycle(cycle_number)
        current_positions = len(self.positions)
        
        decisions_to_execute = {}
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                continue
                
            signal = trade.get('signal')
            if signal in ['buy_to_enter', 'sell_to_enter']:
                # Apply kademeli position limit
                if current_positions >= max_positions_for_cycle:
                    print(f"⚠️ KADEMELİ POZİSYON LİMİTİ (Cycle {cycle_number}): Max {max_positions_for_cycle} positions allowed. Skipping {coin} entry.")
                    continue
                current_positions += 1
                
                decisions_to_execute[coin] = trade

        if decisions_to_execute:
            self.execute_decision(decisions_to_execute, valid_prices, indicator_cache=indicator_cache)

    def get_max_positions_for_cycle(self, cycle_number: int) -> int:
        """Cycle bazlı maximum pozisyon limiti - Kademeli artış sistemi, MAX_POSITIONS ile sınırlı"""
        from config.config import Config
        max_allowed = Config.MAX_POSITIONS
        
        if cycle_number == 1:
            return min(1, max_allowed)  # Cycle 1: max 1 pozisyon (veya MAX_POSITIONS)
        elif cycle_number == 2:
            return min(2, max_allowed)  # Cycle 2: max 2 pozisyon (veya MAX_POSITIONS)
        elif cycle_number == 3:
            return min(3, max_allowed)  # Cycle 3: max 3 pozisyon (veya MAX_POSITIONS)
        elif cycle_number == 4:
            return min(4, max_allowed)  # Cycle 4: max 4 pozisyon (veya MAX_POSITIONS)
        else:
            return max_allowed  # Cycle 5+: MAX_POSITIONS değerini kullan

    def _get_indicator_snapshot(
        self,
        coin: str,
        indicator_cache: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Fetch indicators for 3m and higher timeframe from cache if available, otherwise from market data."""
        cache_source = indicator_cache if indicator_cache is not None else getattr(self, 'indicator_cache', {})
        cached_entry = cache_source.get(coin) if isinstance(cache_source, dict) else None

        indicators_3m = None
        indicators_htf = None

        if isinstance(cached_entry, dict):
            indicators_3m = copy.deepcopy(cached_entry.get('3m'))
            cached_htf = cached_entry.get(HTF_INTERVAL)
            if cached_htf is None and HTF_INTERVAL != '4h':
                cached_htf = cached_entry.get('4h')  # backward compatibility
            indicators_htf = copy.deepcopy(cached_htf)

        if not isinstance(indicators_3m, dict) or 'error' in indicators_3m:
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
        if not isinstance(indicators_htf, dict) or 'error' in indicators_htf:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)

        return indicators_3m, indicators_htf

    def _execute_normal_decisions(
        self,
        decisions: Dict,
        valid_prices: Dict,
        cycle_number: int,
        positions_closed_by_tp_sl: bool,
        indicator_cache: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Execute normal AI decisions with partial profit active"""
        print("🔄 Executing normal AI decisions (partial profit active)")
        
        # KADEMELİ POZİSYON SİSTEMİ: Cycle bazlı pozisyon limiti
        max_positions_for_cycle = self.get_max_positions_for_cycle(cycle_number)
        current_positions = len(self.positions)
        
        decisions_to_execute = {}
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                continue
                
            signal = trade.get('signal')
            if signal in ['buy_to_enter', 'sell_to_enter']:
                # Apply kademeli position limit
                if current_positions >= max_positions_for_cycle:
                    print(f"⚠️ KADEMELİ POZİSYON LİMİTİ (Cycle {cycle_number}): Max {max_positions_for_cycle} positions allowed. Skipping {coin} entry.")
                    continue
                current_positions += 1
                
                decisions_to_execute[coin] = trade
            else:
                # Execute all other decisions (hold, close_position)
                decisions_to_execute[coin] = trade

        if decisions_to_execute:
            self.execute_decision(decisions_to_execute, valid_prices, indicator_cache=indicator_cache)

    def _calculate_maximum_limit(self) -> float:
        """Calculate maximum limit: $15 fixed OR 15% of available cash, whichever is larger"""
        max_from_percentage = self.current_balance * 0.15
        max_limit = max(15.0, max_from_percentage)
        print(f"📊 Maximum limit: ${max_limit:.2f} (${15.0} fixed vs ${max_from_percentage:.2f} 15% of ${self.current_balance:.2f} available cash)")
        return max_limit


    def _adjust_partial_sale_for_max_limit(self, position: Dict, proposed_percent: float) -> Tuple[float, bool, Optional[str]]:
        """Adjust partial sale percentage to ensure position doesn't go below maximum limit"""
        current_margin = position.get('margin_usd', 0)
        
        # Calculate maximum limit: $15 fixed OR 15% of available cash, whichever is larger
        max_limit = self._calculate_maximum_limit()
        
        if current_margin <= max_limit:
            # Position already at or below maximum limit, don't sell - close completely
            print(f"🛑 Partial sale blocked: Position margin ${current_margin:.2f} <= maximum limit ${max_limit:.2f}. Position will be closed.")
            return 0.0, True, f"Position margin ${current_margin:.2f} <= maximum limit ${max_limit:.2f}"
        
        # Calculate remaining margin after proposed sale
        remaining_after_proposed = current_margin * (1 - proposed_percent)
        
        if remaining_after_proposed >= max_limit:
            # Proposed sale keeps us above maximum limit, use as-is
            return proposed_percent, False, None
        else:
            # Adjust sale to leave exactly max_limit margin
            adjusted_sale_amount = current_margin - max_limit
            adjusted_percent = adjusted_sale_amount / current_margin
            
            print(f"📊 Adjusted partial sale: {proposed_percent*100:.0f}% → {adjusted_percent*100:.0f}% to maintain ${max_limit:.2f} maximum limit")
            return adjusted_percent, False, None

    def _adjust_partial_sale_for_min_limit(self, position: Dict, proposed_percent: float) -> float:
        """Adjust partial sale percentage to ensure minimum limit remains after sale"""
        current_margin = position.get('margin_usd', 0)
        
        # Calculate dynamic minimum limit: $15 fixed OR 10% of available cash, whichever is larger
        min_remaining = self._calculate_dynamic_minimum_limit()
        
        if current_margin <= min_remaining:
            # Position already at or below minimum, don't sell
            print(f"🛑 Partial sale blocked: Position margin ${current_margin:.2f} <= minimum limit ${min_remaining:.2f}")
            return 0.0
        
        # Calculate remaining margin after proposed sale
        remaining_after_proposed = current_margin * (1 - proposed_percent)
        
        if remaining_after_proposed >= min_remaining:
            # Proposed sale keeps us above minimum, use as-is
            return proposed_percent
        else:
            # Adjust sale to leave exactly min_remaining margin
            adjusted_sale_amount = current_margin - min_remaining
            adjusted_percent = adjusted_sale_amount / current_margin
            
            print(f"📊 Adjusted partial sale: {proposed_percent*100:.0f}% → {adjusted_percent*100:.0f}% to maintain ${min_remaining:.2f} minimum limit")
            return adjusted_percent

    def _is_counter_trend_trade(self, coin: str, signal: str, indicators_3m: Dict, indicators_htf: Dict) -> bool:
        """Check if trade is counter-trend based on higher timeframe trend vs 15m+3m signal"""
        try:
            if 'error' in indicators_3m or 'error' in indicators_htf:
                return False
            
            # Get 15m indicators for counter-trend validation
            indicators_15m = self.market_data.get_technical_indicators(coin, '15m')
            if 'error' in indicators_15m:
                # Fallback: use only 3m if 15m unavailable
                indicators_15m = None
            
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            
            # Determine higher timeframe trend direction
            trend_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"
            
            # Determine 3m trend direction
            trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
            
            # Determine 15m trend direction (if available)
            trend_15m = None
            if indicators_15m:
                price_15m = indicators_15m.get('current_price')
                ema20_15m = indicators_15m.get('ema_20')
                if isinstance(price_15m, (int, float)) and isinstance(ema20_15m, (int, float)):
                    trend_15m = "BULLISH" if price_15m > ema20_15m else "BEARISH"
            
            # Determine signal direction
            signal_direction = "BULLISH" if signal == 'buy_to_enter' else "BEARISH"
            
            # Check if trade is counter-trend (signal vs higher timeframe trend)
            is_counter_trend = False
            if signal == 'buy_to_enter' and trend_htf == "BEARISH":
                is_counter_trend = True  # Long against bearish higher timeframe trend
            elif signal == 'sell_to_enter' and trend_htf == "BULLISH":
                is_counter_trend = True  # Short against bullish higher timeframe trend
            
            # Counter-trend STRONG: 15m + 3m both align with signal direction (against 1h)
            if is_counter_trend and trend_15m and trend_15m == trend_3m == signal_direction:
                # STRONG counter-trend: 15m + 3m both support the counter-trend signal
                return True
            elif is_counter_trend:
                # Counter-trend but not STRONG (15m or 3m doesn't align)
                return True  # Still counter-trend, just not STRONG
            
            return False
            
        except Exception as e:
            print(f"⚠️ Counter-trend detection error for {coin}: {e}")
            return False

    def apply_market_regime_adjustment(self, confidence: float, signal: str, market_regime: str) -> float:
        """Apply market regime based confidence adjustment (hafifletilmiş - 0.7 yerine 0.92)"""
        original_confidence = confidence
        if market_regime == "BEARISH" and signal == "buy_to_enter":
            # Long in bearish market - counter-trade
            # Hafifletilmiş: 0.7 yerine 0.92 (sadece %8 düşüş)
            adjusted_confidence = max(confidence * 0.92, confidence - 0.05, original_confidence * 0.90)
            print(f"📊 Market regime adjustment: BEARISH market, LONG signal → confidence {confidence:.2f} → {adjusted_confidence:.2f}")
            return adjusted_confidence
        elif market_regime == "BULLISH" and signal == "sell_to_enter":
            # Short in bullish market - counter-trade
            # Hafifletilmiş: 0.7 yerine 0.92 (sadece %8 düşüş)
            adjusted_confidence = max(confidence * 0.92, confidence - 0.05, original_confidence * 0.90)
            print(f"📊 Market regime adjustment: BULLISH market, SHORT signal → confidence {confidence:.2f} → {adjusted_confidence:.2f}")
            return adjusted_confidence
        else:
            # Trend-following trade - no adjustment
            return confidence

    def validate_counter_trade(self, coin: str, signal: str, indicators_3m: Dict, indicators_htf: Dict) -> Dict[str, Any]:
        """Validate counter-trade with multiple technical conditions (15m + 3m alignment for STRONG)"""
        try:
            if 'error' in indicators_3m or 'error' in indicators_htf:
                return {"valid": False, "reason": "Indicator data error"}
            
            # Get 15m indicators for counter-trend validation
            indicators_15m = self.market_data.get_technical_indicators(coin, '15m')
            has_15m = indicators_15m and 'error' not in indicators_15m
            
            conditions_met: List[str] = []
            conditions_met_count = 0
            total_conditions_available = 6 if has_15m else 5  # 15m+3m alignment is a condition
            score = 0.0
            score_breakdown: List[str] = []
            relaxed_cycles = getattr(self, 'relaxed_countertrend_cycles', 0)
            relax_mode_active = relaxed_cycles > 0
            
            def _register(condition: bool, description: str, weight: float) -> None:
                nonlocal conditions_met_count, score
                if condition:
                    conditions_met_count += 1
                    conditions_met.append(description)
                    score += weight
                    score_breakdown.append(f"{description} (+{weight:.1f})")

            # Condition 1: 15m + 3m alignment (STRONG counter-trend requirement)
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            if has_15m:
                price_15m = indicators_15m.get('current_price')
                ema20_15m = indicators_15m.get('ema_20')
                if isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)) and \
                   isinstance(price_15m, (int, float)) and isinstance(ema20_15m, (int, float)):
                    trend_15m = "BULLISH" if price_15m > ema20_15m else "BEARISH"
                    trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
                    signal_direction = "BULLISH" if signal == 'buy_to_enter' else "BEARISH"
                    
                    # STRONG counter-trend: 15m + 3m both align with signal direction
                    if trend_15m == trend_3m == signal_direction:
                        _register(True, "STRONG: 15m+3m alignment with counter-trend signal", 2.0)
                    # MEDIUM counter-trend: 15m VEYA 3m tek başına align with signal direction
                    elif trend_15m == signal_direction:
                        _register(True, "MEDIUM: 15m alignment with counter-trend signal", 1.2)
                    elif trend_3m == signal_direction:
                        _register(True, "MEDIUM: 3m alignment with counter-trend signal", 1.2)
                    else:
                        score_breakdown.append(f"15m+3m alignment: 15m={trend_15m}, 3m={trend_3m}, signal={signal_direction} (no boost)")
            
            # Condition 2: 3m momentum supportive of the counter direction (if 15m not available)
            if not has_15m and isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)):
                if signal == 'buy_to_enter':
                    _register(price_3m > ema20_3m, "MEDIUM: 3m momentum supportive (price > EMA20)", 1.2)
                elif signal == 'sell_to_enter':
                    _register(price_3m < ema20_3m, "MEDIUM: 3m momentum supportive (price < EMA20)", 1.2)
            
            # Condition 3: Volume confirmation (>1.5x average)
            current_volume = indicators_3m.get('volume', 0)
            avg_volume = indicators_3m.get('avg_volume', 1)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            if volume_ratio > 1.5:
                _register(True, f"Volume {volume_ratio:.1f}x average", 1.1)
            elif volume_ratio > 1.0:
                _register(True, f"Volume {volume_ratio:.1f}x average", 0.9)
            elif volume_ratio > 0.8:
                _register(True, f"Volume {volume_ratio:.1f}x average", 0.8)
            else:
                score_breakdown.append(f"Volume {volume_ratio:.2f}x average (no boost)")
            
            # Condition 4: Extreme RSI
            rsi_3m = indicators_3m.get('rsi_14', 50)
            if signal == 'buy_to_enter':
                _register(rsi_3m < 35, f"Extreme RSI ({rsi_3m:.1f})", 1.0)
            elif signal == 'sell_to_enter':
                _register(rsi_3m > 65, f"Extreme RSI ({rsi_3m:.1f})", 1.0)
            
            # Condition 5: Price close to EMA20 (< 1%)
            if isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)) and price_3m:
                price_ema_distance = abs(price_3m - ema20_3m) / price_3m * 100
                ema_distance_threshold = 1.8 if not relax_mode_active else 2.5
                _register(price_ema_distance < ema_distance_threshold, f"Price within {ema_distance_threshold:.1f}% of EMA20 ({price_ema_distance:.2f}%)", 0.6)
            
            # Condition 6: MACD divergence supportive
            macd_3m = indicators_3m.get('macd', 0)
            macd_signal_3m = indicators_3m.get('macd_signal', 0)
            if signal == 'buy_to_enter':
                _register(macd_3m > macd_signal_3m, "MACD divergence supportive", 0.8)
            elif signal == 'sell_to_enter':
                _register(macd_3m < macd_signal_3m, "MACD divergence supportive", 0.8)

            # Volume safety rail: reject extremely illiquid environments
            if volume_ratio is not None:
                volume_safety_floor = 0.15
                if relax_mode_active:
                    volume_safety_floor = 0.10
                if volume_ratio < volume_safety_floor:
                    return {
                        "valid": False,
                        "conditions_met": conditions_met,
                        "total_conditions": conditions_met_count,
                        "conditions_required": total_conditions_available,
                        "score": score,
                        "score_threshold": None,
                        "score_breakdown": score_breakdown,
                        "reason": (
                            f"Volume ratio {volume_ratio:.2f}x is below {volume_safety_floor:.2f}x minimum for counter-trend entries"
                            if not relax_mode_active else
                            f"Volume ratio {volume_ratio:.2f}x is below relaxed minimum {volume_safety_floor:.2f}x while cooldown active"
                        )
                    }

            # Dynamic score threshold based on liquidity context
            score_threshold = 2.3
            if volume_ratio is not None:
                if volume_ratio >= 1.5:
                    score_threshold = 2.0
                elif volume_ratio >= 1.0:
                    score_threshold = 2.1
                elif volume_ratio >= 0.8:
                    score_threshold = 2.2
                else:
                    score_threshold = 2.3
            if relax_mode_active:
                score_threshold = max(1.8, score_threshold - 0.3)

            valid = score >= score_threshold
            return {
                "valid": valid,
                "conditions_met": conditions_met,
                "total_conditions": conditions_met_count,
                "conditions_required": total_conditions_available,
                "score": round(score, 2),
                "score_threshold": round(score_threshold, 2),
                "score_breakdown": score_breakdown,
                "reason": (
                    f"Counter-trend score {score:.2f} ≥ threshold {score_threshold:.2f} ({conditions_met_count}/{total_conditions_available} signals)"
                    if valid else
                    f"Counter-trend score {score:.2f} < threshold {score_threshold:.2f} ({conditions_met_count}/{total_conditions_available} signals)"
                ),
                "relax_mode_active": relax_mode_active,
                "relaxed_cycles_remaining": relaxed_cycles if relax_mode_active else 0
            }
        except Exception as e:
            print(f"⚠️ Counter-trade validation error for {coin}: {e}")
            return {"valid": False, "reason": f"Validation error: {str(e)}"}

    def calculate_dynamic_risk(self, market_regime: str, confidence: float) -> float:
        """Calculate dynamic risk based on market regime and confidence"""
        base_risk = 50.0  # $50 base risk
        
        # Market regime adjustment
        if "BEARISH" in market_regime:
            base_risk *= 0.8  # Bearish market: reduce risk to $40
        elif "BULLISH" in market_regime:
            base_risk *= 1.2  # Bullish market: increase risk to $60
        
        # Confidence adjustment
        if confidence >= 0.7:
            base_risk *= 1.1  # High confidence: +10%
        elif confidence <= 0.5:
            base_risk *= 0.9  # Low confidence: -10%
            
        return min(base_risk, 60.0)  # Cap at $60 maximum risk


    def calculate_confidence_based_margin(self, confidence: float, available_cash: float) -> float:
        """Calculate margin based on confidence level and available cash (new simplified formula)"""
        # Max margin = 40% of available cash × confidence
        margin = available_cash * 0.40 * confidence
        
        # Apply minimum margin limit ($10)
        margin = max(margin, Config.MIN_POSITION_MARGIN_USD)
        
        print(f"📊 Confidence-based margin: ${margin:.2f} (confidence: {confidence:.2f}, available cash: ${available_cash:.2f})")
        return margin

    def get_volume_threshold(self, market_regime: str, signal: str) -> float:
        """Get volume threshold based on market regime and signal type"""
        if signal == "buy_to_enter":
            if "BULLISH" in market_regime:
                return 0.6  # Bullish market: LONG >60% volume
            else:
                return 0.8  # Other markets: LONG >80% volume
        elif signal == "sell_to_enter":
            if "BEARISH" in market_regime:
                return 0.3  # Bearish market: SHORT >30% volume
            else:
                return 0.4  # Other markets: SHORT >40% volume
        return 0.8  # Default threshold

    def calculate_volume_quality_score(
        self,
        coin: str,
        indicators_3m: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate volume quality score (0-100) based on Config thresholds"""
        try:
            if indicators_3m is None or not isinstance(indicators_3m, dict):
                indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            if 'error' in indicators_3m:
                return 0.0
            
            current_volume = indicators_3m.get('volume', 0)
            avg_volume = indicators_3m.get('avg_volume', 0)
            
            if avg_volume <= 0:
                return 0.0
            
            volume_ratio = current_volume / avg_volume
            
            # Calculate score based on Config thresholds
            if volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['excellent']:
                return 90.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['good']:
                return 75.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['fair']:
                return 60.0
            elif volume_ratio >= Config.VOLUME_QUALITY_THRESHOLDS['poor']:
                return 40.0
            else:
                return 20.0
                
        except Exception as e:
            print(f"⚠️ Volume quality score calculation error for {coin}: {e}")
            return 0.0

    def detect_market_regime_overall(self) -> str:
        """Detect overall market regime across all coins"""
        try:
            # Use existing market_data instance
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            for coin in self.market_data.available_coins:
                indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
                if 'error' in indicators_htf:
                    continue
                
                price = indicators_htf.get('current_price')
                ema20 = indicators_htf.get('ema_20')
                
                if not isinstance(price, (int, float)) or not isinstance(ema20, (int, float)) or ema20 == 0:
                    continue

                delta = (price - ema20) / ema20
                if abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT:
                    neutral_count += 1
                elif price > ema20:
                    bullish_count += 1
                else:
                    bearish_count += 1
            
            if bullish_count >= 4:
                return "BULLISH"
            if bearish_count >= 4:
                return "BEARISH"
            if neutral_count >= 4:
                return "NEUTRAL"

            if bullish_count > bearish_count:
                return "BULLISH" if bullish_count >= 3 else "NEUTRAL"
            if bearish_count > bullish_count:
                return "BEARISH" if bearish_count >= 3 else "NEUTRAL"
                return "NEUTRAL"
                
        except Exception as e:
            print(f"⚠️ Market regime detection error: {e}")
            return "NEUTRAL"

    def should_enhance_short_sizing(self, coin: str) -> bool:
        """Check if short position should be enhanced (%15 daha büyük)"""
        try:
            # Use existing market_data instance
            indicators_3m = self.market_data.get_technical_indicators(coin, '3m')
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            
            if 'error' in indicators_3m or 'error' in indicators_htf:
                return False
            
            # Enhanced short conditions:
            # 1. 3m RSI > 70 (aşırı alım)
            rsi_3m = indicators_3m.get('rsi_14', 50)
            # 2. Volume > 1.5x average
            volume_ratio = indicators_3m.get('volume', 0) / indicators_3m.get('avg_volume', 1)
            # 3. Higher timeframe trend bearish
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            trend_bearish = price_htf < ema20_htf
            
            # All conditions must be met
            return rsi_3m > 70 and volume_ratio > 1.5 and trend_bearish
            
        except Exception as e:
            print(f"⚠️ Enhanced short sizing check error for {coin}: {e}")
            return False

    def execute_decision(
        self,
        decisions: Dict[str, Dict],
        current_prices: Dict[str, float],
        indicator_cache: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Executes trading decisions from AI, incorporating dynamic sizing and enhanced features."""
        print("\n⚡ EXECUTING AI DECISIONS...")
        if not isinstance(decisions, dict): print(f"❌ Invalid decisions format: {type(decisions)}"); return

        # Import Config inside the function to avoid scope issues
        from config.config import Config

        bias_metrics = getattr(self, 'latest_bias_metrics', self.get_directional_bias_metrics())
        execution_report = {
            'executed': [],
            'blocked': [],
            'skipped': [],
            'holds': [],
            'notes': [],
            'timestamp': datetime.now().isoformat()
        }
        live_trading = getattr(self, 'is_live_trading', False)
        if live_trading:
            self.sync_live_account()

        for coin, trade in decisions.items():
            if not isinstance(trade, dict): print(f"⚠️ Invalid trade data for {coin}: {type(trade)}"); continue
            if coin not in current_prices or not isinstance(current_prices[coin], (int, float)) or current_prices[coin] <= 0:
                print(f"⚠️ Skipping {coin}: Invalid price data."); continue

            signal = trade.get('signal'); current_price = current_prices[coin]; position = self.positions.get(coin)

            if signal == 'buy_to_enter' or signal == 'sell_to_enter':
                if position:
                    print(f"⚠️ {signal.upper()} {coin}: Position already open.")
                    execution_report['skipped'].append({'coin': coin, 'reason': 'position_exists', 'signal': signal})
                    trade['runtime_decision'] = 'skipped_existing_position'
                    continue

                confidence = trade.get('confidence', 0.5) # Default 50% confidence if missing
                leverage = trade.get('leverage')
                if leverage in (None, "", 0):
                    leverage = 8
                # Ensure confidence and leverage are valid numbers
                try:
                    confidence = float(confidence)
                    leverage = int(leverage)
                except (ValueError, TypeError): print(f"⚠️ Invalid confidence ({confidence}) or leverage ({leverage}) for {coin}. Skipping."); continue
                if leverage < 1:
                    leverage = 1
                # Enforce maximum leverage limit from config
                if leverage > Config.MAX_LEVERAGE: 
                    print(f"⚠️ Leverage {leverage}x exceeds maximum limit of {Config.MAX_LEVERAGE}x. Reducing to {Config.MAX_LEVERAGE}x.")
                    leverage = Config.MAX_LEVERAGE
                # Clamp leverage into [8, 10] operational band for new entries
                if signal in ['buy_to_enter', 'sell_to_enter']:
                    if leverage < 8:
                        print(f"ℹ️ Adjusting leverage from {leverage}x to minimum operational level 8x for {coin}.")
                        leverage = 8
                    elif leverage > 10:
                        print(f"ℹ️ Adjusting leverage from {leverage}x to maximum operational level 10x for {coin}.")
                        leverage = 10
                if not (0.0 <= confidence <= 1.0): confidence = 0.5 # Clamp confidence to 0.0-1.0
                
                # 2. Market Regime Position Sizing
                market_regime = self.detect_market_regime_overall()
                market_regime_multiplier = Config.MARKET_REGIME_MULTIPLIERS.get(market_regime, 1.0)
                partial_margin_factor = 1.0

                direction = 'long' if signal == 'buy_to_enter' else 'short'
                dominant_direction = None
                if market_regime == 'BULLISH':
                    dominant_direction = 'long'
                elif market_regime == 'BEARISH':
                    dominant_direction = 'short'

                # Coin bazlı cooldown kontrolü (öncelikli - zararlı trade'den sonra aynı coin'i engelle)
                coin_cooldowns = self.coin_cooldowns
                coin_upper = coin.upper()
                coin_cooldown_remaining = coin_cooldowns.get(coin_upper, 0)
                if coin_cooldown_remaining > 0:
                    print(f"⏸️ Coin cooldown active: Blocking {coin} entry ({coin_cooldown_remaining} cycles remaining - previous loss).")
                    execution_report['blocked'].append({
                        'coin': coin,
                        'reason': 'coin_cooldown',
                        'cooldown_remaining': coin_cooldown_remaining
                    })
                    trade['runtime_decision'] = 'blocked_coin_cooldown'
                    continue

                # Cooldown kontrolü: PortfolioManager'dan cooldown durumunu al
                cooldowns = self.directional_cooldowns
                cooldown_remaining = cooldowns.get(direction, 0)
                print(f"🔍 Cooldown check for {coin} {direction.upper()}: cooldown_remaining={cooldown_remaining}, cooldowns={cooldowns}")
                if cooldown_remaining > 0:
                    print(f"⏸️ Directional cooldown active: Blocking {direction.upper()} entry for {coin} ({cooldown_remaining} cycles remaining).")
                    execution_report['blocked'].append({
                        'coin': coin,
                        'reason': 'directional_cooldown',
                        'direction': direction,
                        'cooldown_remaining': cooldown_remaining
                    })
                    trade['runtime_decision'] = 'blocked_directional_cooldown'
                    continue

                # Check SAME_DIRECTION_LIMIT for ALL directions (not just dominant)
                directional_counts = self.count_positions_by_direction()
                current_same_direction = directional_counts.get(direction, 0)
                if current_same_direction >= Config.SAME_DIRECTION_LIMIT:
                    print(
                        f"🚫 SAME-DIRECTION LIMIT: {coin} {signal} blocked. "
                        f"{current_same_direction}/{Config.SAME_DIRECTION_LIMIT} {direction.upper()} positions already open."
                    )
                    execution_report['blocked'].append({
                        'coin': coin,
                        'reason': 'same_direction_limit',
                        'direction': direction,
                        'current': current_same_direction,
                        'limit': Config.SAME_DIRECTION_LIMIT
                    })
                    trade['runtime_decision'] = 'blocked_same_direction_limit'
                    continue
                
                # 3. Enhanced Short Sizing (increase by 15% when criteria met)
                if signal == 'sell_to_enter':
                    # Check enhanced short conditions
                    if self.should_enhance_short_sizing(coin):
                        print(f"📈 ENHANCED SHORT: increasing {coin} short exposure by 15% based on conditions")
                        if 'quantity_usd' in trade:
                            trade['quantity_usd'] *= Config.SHORT_ENHANCEMENT_MULTIPLIER
                
                # 4. Coin-specific dynamic stop-loss adjustment
                stop_loss = trade.get('stop_loss')
                try:
                    stop_loss = float(stop_loss) if stop_loss is not None else None
                except (ValueError, TypeError):
                    stop_loss = None
                
                # Apply dynamic stop-loss multiplier
                if stop_loss is not None:
                    stop_loss_multiplier = Config.COIN_SPECIFIC_STOP_LOSS_MULTIPLIERS.get(coin, 1.0)
                    if signal == 'buy_to_enter':
                        stop_loss = current_price - ((current_price - stop_loss) * stop_loss_multiplier)
                    else:  # sell_to_enter
                        stop_loss = current_price + ((stop_loss - current_price) * stop_loss_multiplier)
                    print(f"📊 Dynamic Stop-Loss: applied {stop_loss_multiplier}x multiplier for {coin}")
                
                # 5. Counter-Trend detection (validate only for counter trades) & indicator alignment
                # Use cached indicators when available to keep prompt/execution consistent
                current_trend = 'unknown'
                volume_ratio = None
                trend_classification = 'unknown'
                try:
                    indicators_3m, indicators_htf = self._get_indicator_snapshot(coin, indicator_cache)
                    if ('error' in indicators_3m) or ('error' in indicators_htf):
                        print(f"⚠️ Indicator fetch error for {coin}: {indicators_3m.get('error', '') or indicators_htf.get('error', '')}")
                        execution_report['blocked'].append({'coin': coin, 'reason': 'indicator_error'})
                        trade['runtime_decision'] = 'blocked_indicator_error'
                        continue
                    # Volume quality scoring using the same data the AI saw
                    volume_quality_score = self.calculate_volume_quality_score(coin, indicators_3m=indicators_3m)
                    confidence = min(1.0, confidence + (volume_quality_score / 1000))
                    trade['volume_quality_score'] = volume_quality_score

                    current_volume = indicators_3m.get('volume', 0)
                    avg_volume = indicators_3m.get('avg_volume', 1)
                    volume_ratio = current_volume / avg_volume if avg_volume and avg_volume > 0 else 0.0
                    trade['volume_ratio_runtime'] = round(volume_ratio, 4)
                    relax_cycles_global = getattr(self, 'relaxed_countertrend_cycles', 0)
                    relax_mode_active_global = relax_cycles_global > 0
                    if volume_ratio is not None:
                        low_volume_threshold = 0.20 if not relax_mode_active_global else 0.15
                        if volume_ratio < low_volume_threshold:
                            if relax_mode_active_global:
                                print(f"⚡ Relaxed mode: skipping low-volume penalty for {coin} (ratio {volume_ratio:.2f}x).")
                            else:
                                original_confidence = confidence
                                # Hafifletilmiş penalty: 0.8 yerine 0.92 (sadece %8 düşüş)
                                confidence = max(confidence * 0.92, confidence - 0.05)
                                # Minimum confidence floor: AI'nın verdiği değerin %85'i altına düşmesin
                                min_floor = original_confidence * 0.85
                                confidence = max(confidence, min_floor, Config.MIN_CONFIDENCE)
                                print(f"🥶 LOW VOLUME PENALTY: {coin} volume ratio {volume_ratio:.2f}x. Confidence {original_confidence:.2f} → {confidence:.2f}")
                                if confidence < Config.MIN_CONFIDENCE:
                                    print(f"🚫 Low volume block: {coin} confidence {confidence:.2f} below minimum after penalty.")
                                    execution_report['blocked'].append({'coin': coin, 'reason': 'low_volume', 'volume_ratio': volume_ratio, 'confidence': confidence})
                                    trade['runtime_decision'] = 'blocked_low_volume'
                                    continue
                                trade['confidence'] = confidence
                    trend_info = self.update_trend_state(coin, indicators_htf, indicators_3m)
                    current_trend = trend_info.get('trend', 'unknown')
                    flip_cycle = trend_info.get('last_flip_cycle')
                    last_flip_direction = trend_info.get('last_flip_direction')
                    guard_cycles_since_flip = None
                    guard_window = self.trend_flip_cooldown
                    if isinstance(flip_cycle, int):
                        guard_cycles_since_flip = max(0, self.current_cycle_number - flip_cycle)
                    trade['trend_runtime'] = current_trend

                    pre_bias_confidence = confidence
                    confidence = self.apply_directional_bias(signal, confidence, bias_metrics, current_trend)
                    if confidence != pre_bias_confidence:
                        print(f"🧭 Directional bias adjustment: {coin} {signal} confidence {pre_bias_confidence:.2f} → {confidence:.2f}")
                        trade['confidence'] = confidence
                    is_counter_trend = self._is_counter_trend_trade(coin, signal, indicators_3m, indicators_htf)
                    trend_classification = 'counter_trend' if is_counter_trend else 'trend_following'
                    trade['trend_alignment'] = trend_classification
                    snapshot_parts = []
                    price_htf = indicators_htf.get('current_price')
                    ema20_htf = indicators_htf.get('ema_20')
                    price_3m = indicators_3m.get('current_price')
                    ema20_3m = indicators_3m.get('ema_20')
                    def _fmt(val):
                        return f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
                    comparison_htf = "?" if not isinstance(price_htf, (int, float)) or not isinstance(ema20_htf, (int, float)) else (">" if price_htf > ema20_htf else "<" if price_htf < ema20_htf else "=")
                    comparison_3m = "?" if not isinstance(price_3m, (int, float)) or not isinstance(ema20_3m, (int, float)) else (">" if price_3m > ema20_3m else "<" if price_3m < ema20_3m else "=")
                    snapshot_parts.append(f"{HTF_LABEL} price={_fmt(price_htf)} {comparison_htf} EMA20={_fmt(ema20_htf)}")
                    snapshot_parts.append(f"3m price={_fmt(price_3m)} {comparison_3m} EMA20={_fmt(ema20_3m)}")
                    snapshot_parts.append(f"volume_ratio={volume_ratio:.2f}x")
                    snapshot_parts.append(f"counter_trend={is_counter_trend}")
                    snapshot_parts.append(f"trend_state={current_trend.upper()}")
                    print(f"🧾 EXECUTION SNAPSHOT {coin}: " + " | ".join(snapshot_parts))

                    direction = 'long' if signal == 'buy_to_enter' else 'short'
                    
                    if is_counter_trend:
                        # Check counter-trend cooldown
                        counter_trend_cooldown = self.counter_trend_cooldown
                        if counter_trend_cooldown > 0:
                            print(f"🚫 Counter-trend cooldown active: Blocking {coin} {signal} ({counter_trend_cooldown} cycles remaining).")
                            execution_report['blocked'].append({'coin': coin, 'reason': 'counter_trend_cooldown', 'classification': trend_classification})
                            trade['runtime_decision'] = 'blocked_counter_trend_cooldown'
                            continue
                        
                        relaxed_countertrend = self.relaxed_countertrend_cycles > 0
                        if relaxed_countertrend:
                            remaining_relax = self.relaxed_countertrend_cycles
                            print(f"⚡ RELAXED COUNTER-TREND MODE: {coin} - skipping flip guard & validation ({remaining_relax} cycles remaining).")
                        else:
                            guard_active = guard_cycles_since_flip is not None and guard_cycles_since_flip <= guard_window
                            if guard_active:
                                # FIXED: Confidence threshold DECREASES over time (stricter → more relaxed)
                                if guard_cycles_since_flip == 0:
                                    min_conf = 0.70  # Strictest: same cycle as flip
                                    if confidence < min_conf:
                                        print(f"🚫 Flip guard confidence floor: {coin} {signal} confidence {confidence:.2f} < {min_conf:.2f} in same cycle after flip.")
                                        execution_report['blocked'].append({'coin': coin, 'reason': 'trend_flip_guard_confidence', 'classification': trend_classification})
                                        trade['runtime_decision'] = 'blocked_trend_flip_confidence'
                                        continue
                                    partial_margin_factor = min(partial_margin_factor, 0.5)
                                    print(f"⏳ Trend flip guard: {coin} sizing capped at 50% in same-cycle counter-trend attempt (confidence {confidence:.2f}).")
                                elif guard_cycles_since_flip == 1:
                                    min_conf = 0.65  # More relaxed: one cycle after flip
                                    if confidence < min_conf:
                                        print(f"🚫 Flip guard confidence floor: {coin} {signal} confidence {confidence:.2f} < {min_conf:.2f} one cycle after flip.")
                                        execution_report['blocked'].append({'coin': coin, 'reason': 'trend_flip_guard_confidence', 'classification': trend_classification})
                                        trade['runtime_decision'] = 'blocked_trend_flip_confidence'
                                        continue
                                    partial_margin_factor = min(partial_margin_factor, 0.7)
                                    print(f"⏳ Trend flip guard (counter-trend): {coin} sizing capped at 70% one cycle after flip.")
                                elif guard_cycles_since_flip == 2:
                                    min_conf = 0.60  # Most relaxed: two cycles after flip
                                    if confidence < min_conf:
                                        print(f"🚫 Flip guard confidence floor: {coin} {signal} confidence {confidence:.2f} < {min_conf:.2f} two cycles after flip.")
                                        execution_report['blocked'].append({'coin': coin, 'reason': 'trend_flip_guard_confidence', 'classification': trend_classification})
                                        trade['runtime_decision'] = 'blocked_trend_flip_confidence'
                                        continue
                                    partial_margin_factor = min(partial_margin_factor, 0.9)
                                    print(f"⏳ Trend flip guard (counter-trend): {coin} sizing capped at 90% two cycles after flip.")
                            counter_confidence_floor = 0.65 if not relaxed_countertrend else 0.60
                            if confidence < counter_confidence_floor:
                                if relaxed_countertrend:
                                    print(f"⚠️ Relaxed mode: counter-trend confidence {confidence:.2f} below {counter_confidence_floor:.2f}, but allowing due to cooldown.")
                                else:
                                    print(f"⚠️ WARNING: Counter-trend confidence {confidence:.2f} below recommended {counter_confidence_floor:.2f} - proceeding with AI decision")
                            print(f"⚠️ COUNTER-TREND DETECTED: {coin} - respecting AI decision with additional validation")
                            
                            # Perform validation for counter-trend trades only
                            validation_result = self.validate_counter_trade(coin, signal, indicators_3m, indicators_htf)
                            
                            if validation_result['valid']:
                                print(f"✅ COUNTER-TRADE STRONG: {validation_result['reason']}")
                                print(f"   Conditions met: {validation_result.get('conditions_met', [])}")
                            else:
                                print(f"⚠️ COUNTER-TRADE WEAK: {validation_result['reason']}")
                                print(f"   Conditions met: {validation_result.get('conditions_met', [])}")
                                print(f"⚠️ WARNING: Counter-trend validation shows weak conditions - proceeding with AI decision")
                    else:
                        guard_active = guard_cycles_since_flip is not None and guard_cycles_since_flip <= guard_window
                        if guard_active and last_flip_direction:
                            is_trend_direction = ((last_flip_direction == 'bullish' and direction == 'long') or (last_flip_direction == 'bearish' and direction == 'short'))
                            if is_trend_direction:
                                original_conf = confidence
                                if guard_cycles_since_flip == 0:
                                    # Hafifletilmiş: 0.90 yerine 0.97
                                    confidence = max(confidence * 0.97, confidence - 0.02, original_conf * 0.95)
                                    partial_margin_factor = min(partial_margin_factor, 0.7)
                                    print(f"⏳ Trend flip guard (trend-following): {coin} confidence {original_conf:.2f} → {confidence:.2f} & sizing 50% immediately after flip.")
                                elif guard_cycles_since_flip == 1:
                                    # Hafifletilmiş: 0.95 yerine 0.98
                                    confidence = max(confidence * 0.98, confidence - 0.01, original_conf * 0.97)
                                    partial_margin_factor = min(partial_margin_factor, 0.8)
                                    print(f"⏳ Trend flip guard (trend-following): {coin} confidence {original_conf:.2f} → {confidence:.2f} & sizing 70% one cycle after flip.")
                                elif guard_cycles_since_flip == 2:
                                    # Hafifletilmiş: 0.98 yerine 0.99
                                    confidence = max(confidence * 0.99, original_conf * 0.98)
                                    partial_margin_factor = min(partial_margin_factor, 0.90)
                                    print(f"⏳ Trend flip guard (trend-following): {coin} confidence {original_conf:.2f} → {confidence:.2f} & sizing 85% two cycles after flip.")
                                trade['confidence'] = confidence
                        # Trend-following trade path - Hibrit yaklaşım (15m dahil)
                        trend_strength_result = self.get_trend_following_strength(coin, signal)

                        if trend_strength_result and trend_strength_result['strength']:
                            # 15m dahil multi-timeframe analizi başarılı
                            strength = trend_strength_result['strength']
                            alignment_info = trend_strength_result['alignment_info']
                            trends = trend_strength_result['trends']
                            
                            # Mevcut trend_aligned kontrolü (geriye dönük uyumluluk için)
                            price_htf_follow = indicators_htf.get('current_price')
                            ema20_htf_follow = indicators_htf.get('ema_20')
                            ema20_3m = indicators_3m.get('ema_20')
                            price_3m = indicators_3m.get('current_price')
                            trend_aligned = False
                            if isinstance(price_htf_follow, (int, float)) and isinstance(ema20_htf_follow, (int, float)) \
                                    and isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)):
                                if signal == 'buy_to_enter' and price_htf_follow >= ema20_htf_follow and price_3m >= ema20_3m:
                                    trend_aligned = True
                                elif signal == 'sell_to_enter' and price_htf_follow <= ema20_htf_follow and price_3m <= ema20_3m:
                                    trend_aligned = True
                            
                            # Logging - Güç seviyesi ve 15m bilgisi ile
                            ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else "n/a"
                            print(f"✅ TREND-FOLLOWING ({strength}): {coin} {alignment_info}")
                            print(f"   Timeframes: 1h={trends['1h']}, 15m={trends['15m']}, 3m={trends['3m']} | Volume: {ratio_str}x")
                            
                            # Mevcut volume kontrolü (değişiklik yok)
                            if trend_aligned:
                                if volume_ratio is not None and volume_ratio >= 0.5:
                                    if volume_ratio < 0.8:
                                        partial_margin_factor = 0.5
                                        print(f"🧪 Low-volume trend-following: using 50% margin for {coin} (volume ratio {volume_ratio:.2f})")
                        else:
                            # Fallback: Eski mantık (hata durumunda veya 15m verisi yoksa)
                            price_htf_follow = indicators_htf.get('current_price')
                            ema20_htf_follow = indicators_htf.get('ema_20')
                            ema20_3m = indicators_3m.get('ema_20')
                            price_3m = indicators_3m.get('current_price')
                            trend_aligned = False
                            if isinstance(price_htf_follow, (int, float)) and isinstance(ema20_htf_follow, (int, float)) \
                                    and isinstance(price_3m, (int, float)) and isinstance(ema20_3m, (int, float)):
                                if signal == 'buy_to_enter' and price_htf_follow >= ema20_htf_follow and price_3m >= ema20_3m:
                                    trend_aligned = True
                                elif signal == 'sell_to_enter' and price_htf_follow <= ema20_htf_follow and price_3m <= ema20_3m:
                                    trend_aligned = True
                            if trend_aligned:
                                if volume_ratio is not None and volume_ratio >= 0.5:
                                    if volume_ratio < 0.8:
                                        partial_margin_factor = 0.5
                                        print(f"🧪 Low-volume trend-following: using 50% margin for {coin} (volume ratio {volume_ratio:.2f})")
                                ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else "n/a"
                                print(f"✅ TREND-FOLLOWING: {coin} aligns with {HTF_LABEL} trend direction (volume ratio {ratio_str})")
                            else:
                                print(f"✅ TREND-FOLLOWING: {coin} aligns with {HTF_LABEL} trend direction")
                            
                except Exception as e:
                    print(f"⚠️ Counter-trend detection failed for {coin}: {e}")
                    # Detection hatasında trade'e izin ver
                    execution_report['notes'].append({'coin': coin, 'warning': str(e)})
                
                # Use dynamic confidence-based margin calculation instead of AI's quantity_usd
                # This ensures position sizing is ratio-based and dynamic
                calculated_margin = self.calculate_confidence_based_margin(confidence, self.current_balance)
                
                # Apply market regime multiplier
                calculated_margin *= market_regime_multiplier
                if partial_margin_factor < 1.0:
                    standard_margin = calculated_margin
                    reduced_margin = standard_margin * partial_margin_factor
                    print(f"📉 Applying partial margin ({partial_margin_factor*100:.0f}%): ${standard_margin:.2f} → ${reduced_margin:.2f}")
                    calculated_margin = max(reduced_margin, Config.MIN_POSITION_MARGIN_USD)
                
                # MINIMUM $10 COIN MIKTARI KONTROLÜ
                if calculated_margin < Config.MIN_POSITION_MARGIN_USD:
                    print(f"ℹ️ Calculated margin ${calculated_margin:.2f} below minimum ${Config.MIN_POSITION_MARGIN_USD:.2f}. Using minimum margin instead.")
                    calculated_margin = Config.MIN_POSITION_MARGIN_USD
                
                # AVAILABLE CASH KORUMA KONTROLÜ
                min_available_cash = self.current_balance * 0.10
                if (self.current_balance - calculated_margin) < min_available_cash:
                    print(f"⚠️ Trade would reduce available cash below minimum ${min_available_cash:.2f}. Trade blocked.")
                    execution_report['blocked'].append({'coin': coin, 'reason': 'available_cash_guard', 'calculated_margin': calculated_margin})
                    trade['runtime_decision'] = 'blocked_cash_guard'
                    continue
                
                # Convert margin to notional using leverage
                calculated_notional_usd = calculated_margin * leverage
                print(f"   Dynamic confidence-based sizing: ${calculated_notional_usd:.2f} notional (${calculated_margin:.2f} margin)")
                
                # Check risk management constraints with new simplified system
                # Minimum $10 ve available cash koruma zaten kontrol edildi
                # Risk manager artık sadece position limit kontrolü yapacak
                risk_decision = self.risk_manager.should_enter_trade(
                    symbol=coin,
                    current_positions=self.positions,
                    current_prices=current_prices,
                    confidence=confidence,
                    proposed_notional=calculated_notional_usd,
                    current_balance=self.current_balance
                )
                
                if not risk_decision['should_enter']:
                    print(f"⚠️ Risk management blocked trade: {risk_decision['reason']}")
                    execution_report['blocked'].append({'coin': coin, 'reason': f"risk_manager:{risk_decision['reason']}"})
                    trade['runtime_decision'] = 'blocked_risk_manager'
                    continue
                
                notional_usd = calculated_notional_usd
                margin_usd = notional_usd / leverage # Margin required

                if margin_usd <= 0: print(f"⚠️ {signal.upper()} {coin}: Calculated margin is zero/negative. Skipping."); continue
                if margin_usd > self.current_balance: print(f"⚠️ {signal.upper()} {coin}: Not enough cash for margin (${margin_usd:.2f} > ${self.current_balance:.2f})"); continue

                quantity_coin = notional_usd / current_price
                
                if live_trading:
                    live_result = self.execute_live_entry(
                        coin=coin,
                        direction=direction,
                        quantity=quantity_coin,
                        leverage=leverage,
                        current_price=current_price,
                        notional_usd=notional_usd,
                        confidence=confidence,
                        margin_usd=margin_usd,  # Pass calculated margin_usd
                        stop_loss=stop_loss,
                        profit_target=trade.get('profit_target'),
                        invalidation=trade.get('invalidation_condition')
                    )
                    if not live_result.get('success'):
                        error_msg = live_result.get('error', 'unknown_error')
                        print(f"🚫 LIVE ORDER FAILED: {coin} {signal} ({error_msg})")
                        execution_report['blocked'].append({
                            'coin': coin,
                            'reason': 'live_order_failed',
                            'error': error_msg
                        })
                        trade['runtime_decision'] = 'blocked_live_order'
                        continue

                    execution_report['executed'].append({
                        'coin': coin,
                        'signal': signal,
                        'confidence': confidence,
                        'classification': trend_classification,
                        'volume_ratio': volume_ratio,
                        'margin_usd': live_result.get('margin_usd'),
                        'mode': 'live',
                        'order_id': live_result.get('order', {}).get('orderId')
                    })
                    trade['runtime_decision'] = 'executed_live'
                    continue

                self.current_balance -= margin_usd # Deduct margin (simulation)

                direction = 'long' if signal == 'buy_to_enter' else 'short'
                estimated_liq_price = self._estimate_liquidation_price(current_price, leverage, direction)
                
                self.positions[coin] = {
                    'symbol': coin,
                    'direction': direction,
                    'quantity': quantity_coin,
                    'entry_price': current_price,
                    'entry_time': datetime.now().isoformat(),
                    'current_price': current_price,
                    'unrealized_pnl': 0.0,
                    'notional_usd': notional_usd,
                    'margin_usd': margin_usd,
                    'leverage': leverage,
                    'liquidation_price': estimated_liq_price,
                    'confidence': confidence,
                    'exit_plan': {
                        'profit_target': trade.get('profit_target'),
                        'stop_loss': stop_loss,
                        'invalidation_condition': trade.get('invalidation_condition')
                    },
                    'risk_usd': margin_usd,
                    'loss_cycle_count': 0,
                    'entry_volume': indicators_3m.get('volume') if isinstance(indicators_3m, dict) else None,
                    'entry_avg_volume': indicators_3m.get('avg_volume') if isinstance(indicators_3m, dict) else None,
                    'entry_volume_ratio': volume_ratio,
                    'entry_atr_14': indicators_3m.get('atr_14') if isinstance(indicators_3m, dict) else None,
                    'trend_alignment': trend_classification,
                    'trend_context': {
                        'trend_at_entry': current_trend,
                        'alignment': trend_classification,
                        'cycle': self.current_cycle_number
                    },
                    'trailing': {},
                    'sl_oid': -1,
                    'tp_oid': -1,
                    'entry_oid': -1,
                    'wait_for_fill': False
                }
                print(f"✅ {signal.upper()}: Opened {direction} {coin} ({format_num(quantity_coin, 4)} @ ${format_num(current_price, 4)} / Notional ${format_num(notional_usd, 2)} / Margin ${format_num(margin_usd, 2)} / Est. Liq: ${format_num(estimated_liq_price, 4)})")
                execution_report['executed'].append({
                    'coin': coin,
                    'signal': signal,
                    'confidence': confidence,
                    'classification': trend_classification,
                    'volume_ratio': volume_ratio,
                    'margin_usd': margin_usd
                })
                trade['runtime_decision'] = 'executed'

            elif signal == 'close_position':
                if not position:
                    print(f"⚠️ CLOSE {coin}: No position to close.")
                    execution_report['skipped'].append({'coin': coin, 'reason': 'no_position_to_close'})
                    trade['runtime_decision'] = 'skipped_no_position'
                    continue

                if live_trading:
                    live_result = self.execute_live_close(
                        coin=coin,
                        position=position,
                        current_price=current_price,
                        reason=trade.get('justification')
                    )
                    if not live_result.get('success'):
                        error_msg = live_result.get('error', 'unknown_error')
                        print(f"🚫 LIVE CLOSE FAILED: {coin} ({error_msg})")
                        execution_report['blocked'].append({
                            'coin': coin,
                            'reason': 'live_close_failed',
                            'error': error_msg
                        })
                        trade['runtime_decision'] = 'blocked_live_close'
                    else:
                        history_entry = live_result.get('history_entry')
                        if history_entry:
                            self.add_to_history(history_entry)
                        execution_report['executed'].append({
                            'coin': coin,
                            'signal': 'close_position',
                            'pnl': live_result.get('pnl'),
                            'direction': position.get('direction'),
                            'mode': 'live',
                            'order_id': live_result.get('order', {}).get('orderId')
                        })
                        trade['runtime_decision'] = 'executed_live'
                    continue

                sell_quantity = position['quantity']; direction = position.get('direction', 'long')
                entry_price = position['entry_price']
                margin_used = position.get('margin_usd', position.get('notional_usd', 0) / position.get('leverage', 1))

                profit = (current_price - entry_price) * sell_quantity if direction == 'long' else (entry_price - current_price) * sell_quantity
                if not live_trading:
                    self.current_balance += (margin_used + profit)

                print(f"✅ CLOSE (AI): Closed {direction} {coin} @ ${format_num(current_price, 4)} (PnL: ${format_num(profit, 2)})")
                execution_report['executed'].append({'coin': coin, 'signal': 'close_position', 'pnl': profit, 'direction': direction})
                trade['runtime_decision'] = 'executed'

                history_entry = {
                    "symbol": coin, "direction": direction, "entry_price": entry_price, "exit_price": current_price,
                    "quantity": position['quantity'], "notional_usd": position.get('notional_usd', 'N/A'), "pnl": profit,
                    "entry_time": position['entry_time'], "exit_time": datetime.now().isoformat(),
                    "leverage": position.get('leverage', 'N/A'), "close_reason": f"AI Decision: {trade.get('justification', 'N/A')}"
                }
                self.add_to_history(history_entry)
                del self.positions[coin]

            elif signal == 'hold':
                # For hold signals, just log the decision - no action needed
                if position:
                    print(f"ℹ️ HOLD: Holding {position.get('direction', 'long')} {coin} position.")
                else:
                    print(f"ℹ️ HOLD: Staying cash in {coin}.")
                execution_report['holds'].append({'coin': coin, 'has_position': bool(position)})
                trade['runtime_decision'] = 'hold'
            else: print(f"⚠️ Unknown signal '{signal}' for {coin}. Skipping.")

        self.last_execution_report = execution_report
