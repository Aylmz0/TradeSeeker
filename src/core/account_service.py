from datetime import datetime, timezone
from typing import Any

from loguru import logger

from config.config import Config
from src.core import constants
from src.core.data_engine import DataEngine
from src.schemas.position import Position
from src.schemas.trade import TradeHistoryEntry
from src.utils import format_num


try:
    from src.services.binance import BinanceAPIError, BinanceOrderExecutor

    BINANCE_IMPORT_ERROR = None
except Exception as e:
    BinanceOrderExecutor = None
    BINANCE_IMPORT_ERROR = str(e)

    class BinanceAPIError(Exception):
        pass


HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"
HTF_LABEL = HTF_INTERVAL


class AccountService:
    _warned_missing_exit_plan: set[str] = set()

    def __init__(self, portfolio_manager):
        """Initialize AccountService and configure live trading if enabled.

        Args:
            portfolio_manager: PortfolioManager instance for position and balance management.
        """
        self.pm = portfolio_manager
        self.is_live_trading = getattr(Config, "TRADING_MODE", "simulation") == "live"
        self.order_executor = None

        if self.is_live_trading:
            self._initialize_live_trading()
        elif BINANCE_IMPORT_ERROR:
            logger.info(
                "Binance executor unavailable ({}). Staying in simulation mode.",
                BINANCE_IMPORT_ERROR,
            )

        # Propagate state to PortfolioManager
        self.pm.is_live_trading = self.is_live_trading
        self.pm.order_executor = self.order_executor

    def _initialize_live_trading(self):
        """Configure Binance executor when live trading mode is enabled.

        Falls back to simulation mode if the executor is unavailable or
        fails to connect.
        """
        if BinanceOrderExecutor is None:
            logger.error("Live trading requested but Binance executor is unavailable.")
            self.is_live_trading = False
            return
        try:
            self.order_executor = BinanceOrderExecutor(self.pm.market_data.available_coins)
            if not self.order_executor.is_live():
                logger.warning(
                    "Live trading requested but executor initialized in simulation mode. Reverting to paper trading."
                )
                self.is_live_trading = False
                self.order_executor = None
                return
            logger.success(
                "Live trading mode enabled (Binance {}).",
                "TESTNET" if Config.BINANCE_TESTNET else "MAINNET",
            )
            self.sync_live_account()
        except BinanceAPIError as exc:
            logger.error("Binance setup failed: {}. Reverting to simulation mode.", exc)
            self.is_live_trading = False
            self.order_executor = None
        except Exception as exc:
            logger.error("Unexpected Binance setup error: {}. Reverting to simulation mode.", exc)
            self.is_live_trading = False
            self.order_executor = None

        # Ensure sync back to PM after initialization attempts
        self.pm.is_live_trading = self.is_live_trading
        self.pm.order_executor = self.order_executor

    def _build_default_exit_plan(self, direction: str, entry_price: float) -> dict[str, float]:
        """Generate a default exit plan with stop loss and profit target.

        Args:
            direction: Trade direction, either "long" or "short".
            entry_price: Entry price of the position.

        Returns:
            Dictionary with "stop_loss" and "profit_target" values.
        """
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
        if direction == "short":
            stop_loss = entry + stop_offset
            profit_target = max(entry - tp_offset, 0.0)
        else:
            stop_loss = max(entry - stop_offset, 0.0)
            profit_target = entry + tp_offset
        return {"stop_loss": stop_loss, "profit_target": profit_target}

    def _ensure_exit_plan(
        self,
        direction: str,
        entry_price: float,
        symbol: str,
        *candidate_plans: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Ensure a position carries a valid exit plan, supplementing with defaults if needed.

        Args:
            direction: Trade direction, either "long" or "short".
            entry_price: Entry price used to calculate default offsets.
            symbol: Coin symbol for warning messages.
            *candidate_plans: Optional exit plans to merge, in priority order.

        Returns:
            Merged exit plan with at least "stop_loss" and "profit_target" keys.
        """
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
                if (
                    key in ("stop_loss", "profit_target")
                    and isinstance(value, (int, float))
                    and value > 0
                ):
                    provided_keys.add(key)
        missing_required = [
            key for key in ("stop_loss", "profit_target") if key not in provided_keys
        ]
        if missing_required:
            warn_key = f"{symbol}:{','.join(missing_required)}"
            if warn_key not in self._warned_missing_exit_plan:
                logger.warning(
                    "Missing {} for {} - using default exit plan offsets.",
                    ", ".join(missing_required),
                    symbol,
                )
                self._warned_missing_exit_plan.add(warn_key)
        return final_plan

    def _merge_live_positions(
        self,
        snapshot: dict[str, dict[str, Any]],
    ) -> dict[str, Position]:
        """Merge Binance snapshot with local runtime metadata.

        Combines exchange position data with locally maintained state such as
        exit plans, confidence scores, and trailing stop information.

        Args:
            snapshot: Position data from Binance keyed by coin symbol.

        Returns:
            Dictionary of merged Position objects keyed by coin symbol.
        """
        merged: dict[str, Position] = {}
        existing_positions = self.pm.positions
        for coin, snap_pos in snapshot.items():
            previous = existing_positions.get(coin)
            if isinstance(previous, Position):
                previous_dict = previous.model_dump()
            elif isinstance(previous, dict):
                previous_dict = previous
            else:
                previous_dict = {}
            merged_pos: dict[str, Any] = {}
            merged_pos.update(previous_dict)
            merged_pos.update(snap_pos)
            merged_pos["symbol"] = coin

            # Carry forward runtime metadata that Binance snapshot doesn't include
            for key in (
                "confidence",
                "loss_cycle_count",
                "profit_cycle_count",
                "trend_context",
                "trend_alignment",
                "entry_oid",
                "tp_oid",
                "sl_oid",
                "wait_for_fill",
                "entry_volume_ratio",
                "entry_volume",
                "entry_avg_volume",
                "entry_atr_14",
                "trailing",
            ):
                if key in previous_dict and key not in merged_pos:
                    merged_pos[key] = previous_dict[key]

            # Risk USD is set at entry time in _execute_order_payload
            # Do not overwrite with margin_usd here

            existing_plan = previous_dict.get("exit_plan")
            snapshot_plan = snap_pos.get("exit_plan")
            final_plan = self._ensure_exit_plan(
                merged_pos.get("direction", "long"),
                merged_pos.get("entry_price") or merged_pos.get("current_price") or 0.0,
                coin,
                snapshot_plan,
                existing_plan,
            )
            merged_pos["exit_plan"] = final_plan
            merged[coin] = Position(**merged_pos)
        return merged

    def sync_live_account(self):
        """Refresh balances and open positions from Binance when in live mode.

        Updates current_balance, total_value, and positions from the exchange.
        Only operates when live trading is active and the executor is available.
        """
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
                logger.debug(
                    "Binance API Response: availableBalance={}, totalWalletBalance={}, walletBalance={}",
                    available,
                    total_wallet_balance,
                    wallet_balance,
                )

                # Update available cash balance
                if available is not None and available > 0:
                    old_balance = self.pm.current_balance
                    self.pm.current_balance = float(available)
                    if (
                        abs(old_balance - self.pm.current_balance)
                        > constants.BALANCE_UPDATE_THRESHOLD
                    ):  # Only log if significant change
                        logger.info(
                            "Balance updated: ${:.2f} → ${:.2f}",
                            old_balance,
                            self.pm.current_balance,
                        )

                # Note: We'll calculate total_value manually after positions are synced
                # Total value = Available cash + Margin used + Unrealized PnL
                # Binance totalWalletBalance is used for validation only
                if total_wallet_balance is not None and total_wallet_balance > 0:
                    logger.debug(
                        "Binance totalWalletBalance: ${:.2f} (will validate against calculated value)",
                        total_wallet_balance,
                    )
                elif wallet_balance is not None and wallet_balance > 0:
                    logger.warning(
                        "totalWalletBalance not available, using walletBalance: ${:.2f}",
                        wallet_balance,
                    )
                else:
                    logger.warning(
                        "Neither totalWalletBalance nor walletBalance available from Binance API"
                    )
        except BinanceAPIError as exc:
            logger.warning("Binance balance sync failed: {}", exc)
        except Exception as exc:
            logger.warning("Unexpected Binance balance sync error: {}", exc)

        try:
            snapshot = self.order_executor.get_positions_snapshot()
            if isinstance(snapshot, dict):
                # FIX: Replace positions dict under lock to prevent concurrent reads from seeing
                # a half-updated reference during the swap.
                with self.pm._lock:
                    self.pm.positions = self._merge_live_positions(snapshot)

                # Update erosion tracking for each position using existing method
                for coin, pos in self.pm.positions.items():
                    self.pm._update_peak_pnl_tracking(coin, pos)

                # Calculate total margin used from all open positions
                # For cross margin, margin_usd might be 0, so calculate from notional/leverage
                total_margin_used = 0.0
                for pos in self.pm.positions.values():
                    margin = pos.margin_usd
                    if margin <= 0:
                        # Calculate margin from notional and leverage (for cross margin)
                        notional = pos.notional_usd
                        leverage = pos.leverage
                        if notional > 0 and leverage > 0:
                            margin = notional / leverage
                    if isinstance(margin, (int, float)) and margin > 0:
                        total_margin_used += margin

                old_total = self.pm.total_value

                # Calculate total unrealized PnL
                total_unrealized_pnl = 0.0
                for pos in self.pm.positions.values():
                    pnl = pos.unrealized_pnl
                    if isinstance(pnl, (int, float)):
                        total_unrealized_pnl += pnl

                # Total value = Available cash + Margin used + Unrealized PnL
                # This is the correct formula: what you have available + what's locked in positions + unrealized gains/losses
                self.pm.total_value = (
                    self.pm.current_balance + total_margin_used + total_unrealized_pnl
                )

                if abs(old_total - self.pm.total_value) > constants.BALANCE_UPDATE_THRESHOLD:
                    logger.info(
                        "Total value updated: ${:.2f} → ${:.2f} (Available cash: ${:.2f} + Margin used: ${:.2f} + Unrealized PnL: ${:.2f})",
                        old_total,
                        self.pm.total_value,
                        self.pm.current_balance,
                        total_margin_used,
                        total_unrealized_pnl,
                    )

                    # Debug: Also show what Binance says for comparison
                    if overview:
                        total_wb = overview.get("totalWalletBalance")
                        wallet_b = overview.get("walletBalance")
                        if total_wb:
                            wallet_b_str = f"${wallet_b:.2f}" if wallet_b else "N/A"
                            logger.debug(
                                "Binance totalWalletBalance: ${:.2f}, walletBalance: {}",
                                total_wb,
                                wallet_b_str,
                            )
                            # Validate our calculation against Binance
                            diff = abs(self.pm.total_value - total_wb)
                            if (
                                diff > constants.VALUE_SYNC_THRESHOLD
                            ):  # More than 10 cents difference
                                logger.warning(
                                    "Calculated total_value differs from Binance totalWalletBalance by ${:.2f}",
                                    diff,
                                )
            else:
                self.pm.positions = {}
                # No positions, total value = totalWalletBalance (or available cash if not available)
                if (
                    overview
                    and overview.get("totalWalletBalance")
                    and float(overview["totalWalletBalance"]) > 0
                ):
                    self.pm.total_value = float(overview["totalWalletBalance"])
                else:
                    self.pm.total_value = self.pm.current_balance
        except BinanceAPIError as exc:
            logger.warning("Binance position sync failed: {}", exc)
        except Exception as exc:
            logger.warning("Unexpected Binance position sync error: {}", exc)

    @staticmethod
    def _calculate_realized_pnl(
        entry_price: float,
        exit_price: float,
        quantity: float,
        direction: str,
        include_commission: bool = True,
    ) -> float:
        """Calculate realized PnL with optional commission deduction.

        Args:
            entry_price: Price at which the position was opened.
            exit_price: Price at which the position was closed.
            quantity: Number of units traded.
            direction: Trade direction, either "long" or "short".
            include_commission: Whether to deduct round-trip simulation commission.

        Returns:
            Realized profit or loss in USD.
        """
        if quantity <= 0 or entry_price <= 0 or exit_price <= 0:
            return 0.0

        # Calculate raw PnL
        if direction == "long":
            raw_pnl = (exit_price - entry_price) * quantity
        else:
            raw_pnl = (entry_price - exit_price) * quantity

        # Deduct commission for simulation realism (round-trip: entry + exit)
        if include_commission:
            notional = (entry_price + exit_price) / 2 * quantity  # Average notional
            commission = (
                notional * Config.SIMULATION_COMMISSION_RATE * 2
            )  # Round-trip (entry + exit)
            raw_pnl -= commission

        return raw_pnl

    def execute_live_close(
        self,
        coin: str,
        position: Position,
        current_price: float,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Close a position on Binance in live trading mode.

        Args:
            coin: Coin symbol to close.
            position: Position object with entry details.
            current_price: Current market price for order reference.
            reason: Optional reason for closing the position.

        Returns:
            Dictionary with success status, order details, executed quantity,
            average price, PnL, and history entry.
        """
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        if not position:
            return {"success": False, "error": "no_position"}

        direction = position.direction
        quantity = float(position.quantity or 0.0)
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
            pnl = self._calculate_realized_pnl(
                position.entry_price,
                avg_price,
                executed_qty,
                direction,
                include_commission=False,
            )  # Live: Binance already deducted
            self.sync_live_account()
            history_entry = {
                "symbol": coin,
                "direction": direction,
                "entry_price": position.entry_price,
                "exit_price": avg_price,
                "quantity": executed_qty,
                "notional_usd": executed_qty * avg_price,
                "pnl": pnl,
                "entry_time": position.entry_time,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "leverage": position.leverage,
                "close_reason": reason or "live_close",
                "exchange_order_id": order.get("orderId"),
            }
            # Decision Feedback Hook: Log CLOSED trade to SQLite
            try:
                DataEngine().log_decision_close(
                    coin=coin,
                    exit_price=avg_price,
                    pnl_result=pnl,
                )
            except Exception as e:
                # FIX: Log the error instead of silently swallowing
                logger.warning("Failed to log decision close: {}", e)

            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "pnl": pnl,
                "history_entry": history_entry,
            }
        except BinanceAPIError as exc:
            logger.error("Binance close order failed for {}: {}", coin, exc)
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.error("Unexpected Binance close error for {}: {}", coin, exc)
            return {"success": False, "error": str(exc)}

    def execute_live_partial_close(
        self,
        coin: str,
        position: Position,
        close_percent: float,
        current_price: float,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Partially close a position on Binance in live trading mode.

        Args:
            coin: Coin symbol to partially close.
            position: Position object with entry details.
            close_percent: Fraction of the position to close (0.0 to 1.0).
            current_price: Current market price for order reference.
            reason: Optional reason for the partial close.

        Returns:
            Dictionary with success status, order details, executed quantity,
            average price, PnL, and history entry.
        """
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        if not position:
            return {"success": False, "error": "no_position"}

        direction = position.direction
        quantity = float(position.quantity or 0.0)
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
            pnl = self._calculate_realized_pnl(
                position.entry_price,
                avg_price,
                executed_qty,
                direction,
                include_commission=False,
            )  # Live: Binance already deducted
            # Sync account balance after partial close to get updated balance
            self.sync_live_account()
            history_entry = {
                "symbol": coin,
                "direction": direction,
                "entry_price": position.entry_price,
                "exit_price": avg_price,
                "quantity": executed_qty,
                "notional_usd": executed_qty * avg_price,
                "pnl": pnl,
                "entry_time": position.entry_time,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "leverage": position.leverage,
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
            logger.error("Binance partial close failed for {}: {}", coin, exc)
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.error("Unexpected Binance partial close error for {}: {}", coin, exc)
            return {"success": False, "error": str(exc)}

    def close_position(
        self,
        coin: str,
        current_price: float,
        reason: str = "Manual Close",
    ) -> dict[str, Any]:
        """Close a position in paper trading mode (simulation).
        Includes commission deduction for realism.

        Args:
        ----
            coin: Coin symbol
            current_price: Current market price
            reason: Reason for closing

        Returns:
        -------
            Dict with success status and PnL

        """
        # FIX: Hold lock across the entire close operation to prevent double-close
        # from TP/SL thread racing with the cycle thread.
        with self.pm._lock:
            if coin not in self.pm.positions:
                return {"success": False, "error": "no_position"}

            position = self.pm.positions[coin]
            direction = position.direction
            entry_price = position.entry_price
            quantity = position.quantity
            margin_used = position.margin_usd

            if quantity <= 0 or entry_price <= 0:
                return {"success": False, "error": "invalid_position_data"}

            # Calculate PnL
            if direction == "long":
                profit = (current_price - entry_price) * quantity
            else:
                profit = (entry_price - current_price) * quantity

            # Deduct commission for simulation realism (round-trip: entry + exit)
            notional = (entry_price + current_price) / 2 * quantity
            commission = notional * Config.SIMULATION_COMMISSION_RATE * 2
            profit -= commission

            # Update balance
            self.pm.current_balance += margin_used + profit

            # Create history entry
            history_entry = {
                "symbol": coin,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": current_price,
                "quantity": quantity,
                "notional_usd": position.notional_usd,
                "pnl": profit,
                "entry_time": position.entry_time,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "leverage": position.leverage,
                "close_reason": reason,
            }

            # Remove from active positions (under same lock)
            del self.pm.positions[coin]

        # History/bias update outside lock (add_to_history acquires its own lock)
        self.pm.add_to_history(history_entry)

        logger.info(
            "PAPER CLOSE: {} {} @ ${} (PnL: ${}, Commission: ${})",
            direction,
            coin,
            format_num(current_price, 4),
            format_num(profit, 2),
            format_num(commission, 3),
        )

        self.pm.save_state()

        return {"success": True, "pnl": profit, "commission": commission}

    def check_and_execute_tp_sl(self, current_prices: dict[str, float]):
        """Check all open positions for TP/SL triggers and execute closes.

        Runs every 30 seconds via the monitoring loop. Evaluates enhanced exit
        strategies, traditional TP/SL levels, and graduated margin-based stop
        losses. Closes positions in both simulation and live modes.

        Args:
            current_prices: Current market prices keyed by coin symbol.

        Returns:
            True if any positions were closed, False otherwise.
        """
        # Enhanced exit strategy control - check if enabled
        if hasattr(self, "bot") and not self.pm.bot.enhanced_exit_enabled:
            logger.info("Enhanced exit strategy paused during cycle")
            return False

        # All TP/SL decisions made by 30-second monitoring (like simulation mode)
        # No Binance TP/SL orders - all managed by monitoring loop
        if self.pm.positions:
            logger.debug("TP/SL check: {} positions", len(self.pm.positions))

        closed_positions = []  # Keep track of positions closed in this check
        updated_stops = []  # Track positions with updated trailing stops
        state_changed = False

        # FIX: Thread-safe position iteration with lock
        with self.pm._lock:
            positions_snapshot = list(self.pm.positions.items())

        for coin, position in positions_snapshot:
            # Check if position still exists (might have been closed by another thread)
            if coin not in self.pm.positions:
                continue
            if (
                coin not in current_prices
                or not isinstance(current_prices[coin], (int, float))
                or current_prices[coin] <= 0
            ):
                continue  # Skip if price is invalid

            current_price = current_prices[coin]

            # Update erosion tracking (captures intraday peaks)
            self.pm._update_peak_pnl_tracking(coin, position)

            exit_plan = position.exit_plan
            tp = exit_plan.profit_target
            sl = exit_plan.stop_loss
            direction = position.direction
            # FIX: KeyError protection - use .get() with fallback to current_price
            entry_price = position.entry_price or position.current_price or 0.0
            quantity = position.quantity

            # Calculate margin_used properly - try multiple fallback methods
            margin_used = position.margin_usd
            if margin_used is None or margin_used <= 0:
                # Fallback 1: Calculate from notional and leverage
                notional = position.notional_usd
                leverage = position.leverage
                if notional > 0 and leverage > 0:
                    margin_used = notional / leverage
                # Fallback 2: Calculate from entry_price and quantity
                elif entry_price > 0 and quantity > 0:
                    notional = entry_price * quantity
                    leverage = position.leverage
                    # FIX: Leverage zero protection
                    margin_used = notional / leverage if leverage and leverage > 0 else 0
                else:
                    margin_used = 0

            # Debug log if margin_used is still 0
            if margin_used <= 0:
                logger.warning(
                    "margin_used is 0 for {}. Position data: margin_usd={}, notional={}, leverage={}, entry={}, qty={}",
                    coin,
                    position.margin_usd,
                    position.notional_usd,
                    position.leverage,
                    entry_price,
                    quantity,
                )

            close_reason = None

            # Check TP
            # Convert tp/sl to float for safe comparison, handle potential errors
            try:
                tp = float(tp) if tp is not None else None
            except (ValueError, TypeError):
                tp = None
            try:
                sl = float(sl) if sl is not None else None
            except (ValueError, TypeError):
                sl = None

            # Enhanced exit strategy check - REAL-TIME ENTEGRASYON
            exit_decision, position = self.enhanced_exit_strategy(position, current_price, coin)

            # Handle enhanced exit strategy signals - ANINDA İŞLEME
            if exit_decision["action"] == "close_position":
                # Enhanced exit strategy wants to close the position completely
                close_reason = exit_decision["reason"]
                logger.warning(
                    "ENHANCED EXIT CLOSE {} ({}): {} at price ${}",
                    coin,
                    direction,
                    close_reason,
                    format_num(current_price, 4),
                )
                state_changed = True
            elif exit_decision["action"] == "partial_close":
                # Partial profit taking - ANINDA İŞLEME
                close_percent = exit_decision["percent"]
                if self.is_live_trading:
                    live_result = self.execute_live_partial_close(
                        coin=coin,
                        position=position,
                        close_percent=close_percent,
                        current_price=current_price,
                        reason=exit_decision["reason"],
                    )
                    if not live_result.get("success"):
                        logger.error(
                            "Live partial close failed for {}: {}",
                            coin,
                            live_result.get("error", "unknown_error"),
                        )
                        continue
                    history_entry = live_result.get("history_entry")
                    if history_entry:
                        self.pm.add_to_history(history_entry)
                    logger.warning(
                        "PARTIAL CLOSE {} ({}) [LIVE]: {} ({:.0f}% / PnL ${})",
                        coin,
                        direction,
                        exit_decision["reason"],
                        close_percent * 100,
                        format_num(live_result.get("pnl", 0), 2),
                    )
                    # BUG FIX: Adjust peak_pnl proportionally after partial close
                    # Without this, erosion tracking would falsely alarm (peak $3 -> current $1.5 = 50% erosion)
                    if position.peak_pnl > 0:
                        old_peak = position.peak_pnl
                        position = position.model_copy(
                            update={"peak_pnl": position.peak_pnl * (1 - close_percent)}
                        )
                        self.pm.positions[coin] = position
                        logger.debug(
                            "peak_pnl adjusted: ${} → ${} (after {:.0f}% close)",
                            format_num(old_peak, 2),
                            format_num(position.peak_pnl, 2),
                            close_percent * 100,
                        )
                    state_changed = True
                    # Sync account balance after partial close in live mode
                    try:
                        self.sync_live_account()
                        logger.debug("Account balance synced after partial close of {}", coin)
                    except Exception as sync_exc:
                        logger.warning("Failed to sync account after partial close: {}", sync_exc)
                    continue

                # FIX: Hold lock for position mutation + balance update to prevent
                # concurrent reads from seeing partially updated position data.
                with self.pm._lock:
                    # 1. Capture current state BEFORE mutation
                    current_qty = float(position.quantity)
                    current_margin = float(position.margin_usd)
                    current_notional = float(position.notional_usd)
                    close_quantity = current_qty * close_percent

                    # 2. Calculate PnL and Commission
                    if direction == "long":
                        profit = (current_price - entry_price) * close_quantity
                    else:
                        profit = (entry_price - current_price) * close_quantity

                    # Round-trip commission for this portion
                    notional_portion_cost = entry_price * close_quantity
                    notional_portion_exit = current_price * close_quantity
                    avg_notional = (notional_portion_cost + notional_portion_exit) / 2
                    commission = avg_notional * Config.SIMULATION_COMMISSION_RATE * 2
                    profit -= commission

                    # 3. Prepare History Entry BEFORE updating position (using original notional)
                    history_entry = {
                        "symbol": coin,
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "quantity": close_quantity,
                        "notional_usd": notional_portion_cost,  # Use cost-based notional for accuracy
                        "pnl": profit,
                        "entry_time": position.entry_time,
                        "exit_time": datetime.now(timezone.utc).isoformat(),
                        "leverage": position.leverage,
                        "close_reason": exit_decision["reason"],
                    }

                    # 4. Update position state (MUTATION) — under lock
                    new_quantity = current_qty - close_quantity
                    new_margin = current_margin * (1 - close_percent)
                    new_notional = current_notional * (1 - close_percent)

                    # Adjust peak_pnl proportionally
                    peak_update: dict[str, Any] = {}
                    if position.peak_pnl > 0:
                        old_peak = position.peak_pnl
                        new_peak = position.peak_pnl * (1 - close_percent)
                        peak_update = {"peak_pnl": new_peak}
                        logger.debug(
                            "peak_pnl adjusted: ${} → ${}",
                            format_num(old_peak, 2),
                            format_num(new_peak, 2),
                        )

                    position = position.model_copy(
                        update={
                            "quantity": new_quantity,
                            "margin_usd": new_margin,
                            "notional_usd": new_notional,
                            **peak_update,
                        }
                    )
                    self.pm.positions[coin] = position

                    # 5. Update global balance — under lock
                    self.pm.current_balance += (current_margin * close_percent) + profit

                self.pm.add_to_history(history_entry)

                logger.warning(
                    "PARTIAL CLOSE {} ({}): {} - Closed {:.0f}% at ${}, PnL: ${}",
                    coin,
                    direction,
                    exit_decision["reason"],
                    close_percent * 100,
                    format_num(current_price, 4),
                    format_num(profit, 2),
                )
                state_changed = True
                continue  # Continue with remaining position

            elif exit_decision["action"] == "update_stop":
                # Update trailing stop - ANINDA GÜNCELLEME
                updated_stops.append(coin)
                new_stop = exit_decision["new_stop"]
                position = position.model_copy(
                    update={
                        "exit_plan": position.exit_plan.model_copy(update={"stop_loss": new_stop})
                    }
                )
                self.pm.positions[coin] = position
                logger.info(
                    "TRAILING STOP UPDATE {}: New stop at ${}", coin, format_num(new_stop, 4)
                )

                # No Binance orders - stop loss updated in exit_plan, will be monitored by 30-second loop

                state_changed = True
                continue

            # Traditional TP/SL checks (only if no enhanced exit triggered)
            if close_reason is None and tp is not None:
                if (direction == "long" and current_price >= tp) or (
                    direction == "short" and current_price <= tp
                ):
                    close_reason = f"Profit Target ({tp}) hit"

            # Check SL (only if TP not hit)
            # First check exit_plan stop_loss, then fallback to margin-based kademeli stop loss
            if close_reason is None:
                # Check exit_plan stop_loss first
                if sl is not None:
                    if (direction == "long" and current_price <= sl) or (
                        direction == "short" and current_price >= sl
                    ):
                        close_reason = f"Stop Loss ({sl}) hit"

                # If no exit_plan stop_loss or it didn't trigger, check margin-based kademeli stop loss
                # Only check if margin_used is valid (> 0)
                if close_reason is None and quantity > 0 and margin_used > 0:
                    # Calculate margin-based stop loss using graduated loss cutting (same as entry)
                    loss_multiplier = self.pm.get_graduated_loss_multiplier(margin_used)

                    loss_threshold_usd = margin_used * loss_multiplier

                    # Only proceed if loss_threshold_usd is valid
                    if loss_threshold_usd > 0:
                        # Calculate stop loss price from loss threshold
                        if direction == "long":
                            margin_based_stop_loss = entry_price - (loss_threshold_usd / quantity)
                        else:  # short
                            margin_based_stop_loss = entry_price + (loss_threshold_usd / quantity)

                        # Kademeli stop loss is calculated correctly based on margin and loss_multiplier
                        # No minimum distance adjustment needed

                        # Check if current price hit margin-based stop loss
                        if (direction == "long" and current_price <= margin_based_stop_loss) or (
                            direction == "short" and current_price >= margin_based_stop_loss
                        ):
                            close_reason = f"Margin-based Stop Loss ({format_num(margin_based_stop_loss, 4)}) hit (${loss_threshold_usd:.2f} loss limit, {loss_multiplier * 100:.1f}% of ${margin_used:.2f} margin)"

            # Execute Close if triggered
            if close_reason:
                logger.warning(
                    "AUTO-CLOSE {} ({}): {} at price ${}",
                    coin,
                    direction,
                    close_reason,
                    format_num(current_price, 4),
                )

                if self.is_live_trading:
                    logger.info("Executing LIVE close on Binance for {}...", coin)
                    live_result = self.execute_live_close(
                        coin=coin,
                        position=position,
                        current_price=current_price,
                        reason=close_reason,
                    )
                    if not live_result.get("success"):
                        logger.error(
                            "Live auto-close failed for {}: {}",
                            coin,
                            live_result.get("error", "unknown_error"),
                        )
                        continue

                    # Log Binance order details
                    order_id = live_result.get("order", {}).get("orderId")
                    executed_qty = live_result.get("executed_qty", 0)
                    avg_price = live_result.get("avg_price", 0)
                    logger.success(
                        "Binance CLOSE order executed for {}: orderId={}, qty={}, avgPrice=${}",
                        coin,
                        order_id,
                        format_num(executed_qty, 4),
                        format_num(avg_price, 4),
                    )

                    history_entry = live_result.get("history_entry")
                    if history_entry:
                        self.pm.add_to_history(history_entry)
                    logger.info("Live Closed PnL: ${}", format_num(live_result.get("pnl", 0), 2))
                    closed_positions.append(coin)
                    state_changed = True
                    # Sync account balance after closing position in live mode
                    try:
                        self.sync_live_account()
                        logger.debug("Account balance synced after closing {}", coin)
                    except Exception as sync_exc:
                        logger.warning("Failed to sync account after close: {}", sync_exc)
                    continue

                # FIX: Hold lock for balance update + position deletion to prevent
                # double-close from concurrent TP/SL checks.
                with self.pm._lock:
                    if direction == "long":
                        profit = (current_price - entry_price) * quantity
                    else:
                        profit = (entry_price - current_price) * quantity

                    # Deduct commission for simulation realism (round-trip)
                    notional_closed = ((entry_price + current_price) / 2) * quantity
                    commission = (
                        notional_closed * Config.SIMULATION_COMMISSION_RATE * 2
                    )  # Round-trip
                    profit -= commission

                    self.pm.current_balance += (
                        margin_used + profit
                    )  # Return margin + PnL (commission already deducted)

                    logger.info("Closed PnL: ${}", format_num(profit, 2))

                    history_entry = {
                        "symbol": coin,
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "quantity": quantity,
                        "notional_usd": position.notional_usd,
                        "pnl": profit,
                        "entry_time": position.entry_time,
                        "exit_time": datetime.now(timezone.utc).isoformat(),
                        "leverage": position.leverage,
                        "close_reason": close_reason,  # Add reason
                    }
                    # Remove from active positions while lock is held
                    if coin in self.pm.positions:
                        del self.pm.positions[coin]  # Remove from active positions
                    closed_positions.append(coin)
                    state_changed = True

                self.pm.add_to_history(history_entry)  # This increments trade_count

        if closed_positions:
            logger.success("Auto-closed positions: {}", ", ".join(closed_positions))
        if updated_stops:
            logger.success("Updated trailing stops: {}", ", ".join(updated_stops))

        if state_changed:
            self.pm.save_state()

        return len(closed_positions) > 0  # Indicate if any positions were closed

    def get_dynamic_exit_tiers(self, notional_usd: float) -> dict[str, float]:
        """Get dynamic exit tiers based on position notional size.

        Smaller positions use more aggressive thresholds, while larger
        positions use more conservative ones.

        Args:
            notional_usd: Total notional value of the position in USD.

        Returns:
            Dictionary with level1-3 thresholds and take1-3 percentages.
        """
        if notional_usd < constants.NOTIONAL_TIER_1:
            # Small positions: aggressive
            return {
                "level1": 0.008,
                "level2": 0.009,
                "level3": 0.01,
                "take1": 0.35,
                "take2": 0.60,
                "take3": 0.75,
            }
        if notional_usd < constants.NOTIONAL_TIER_2:
            # Medium positions: balanced
            return {
                "level1": 0.007,
                "level2": 0.008,
                "level3": 0.009,
                "take1": 0.30,
                "take2": 0.55,
                "take3": 0.75,
            }
        if notional_usd < constants.NOTIONAL_TIER_3:
            return {
                "level1": 0.006,
                "level2": 0.007,
                "level3": 0.008,
                "take1": 0.25,
                "take2": 0.50,
                "take3": 0.75,
            }
        if notional_usd < constants.NOTIONAL_TIER_4:
            # Large positions: conservative
            return {
                "level1": 0.005,
                "level2": 0.006,
                "level3": 0.007,
                "take1": 0.20,
                "take2": 0.45,
                "take3": 0.75,
            }
        if notional_usd < constants.NOTIONAL_TIER_5:
            return {
                "level1": 0.004,
                "level2": 0.005,
                "level3": 0.006,
                "take1": 0.18,
                "take2": 0.40,
                "take3": 0.75,
            }
        if notional_usd < constants.NOTIONAL_TIER_6:
            return {
                "level1": 0.003,
                "level2": 0.004,
                "level3": 0.005,
                "take1": 0.15,
                "take2": 0.35,
                "take3": 0.75,
            }
        # Very large positions: very conservative
        return {
            "level1": 0.002,
            "level2": 0.003,
            "level3": 0.004,
            "take1": 0.25,
            "take2": 0.50,
            "take3": 0.75,
        }

    def _process_partial_exit_logic(
        self, position: Position, pnl_pct: float, tiers: dict
    ) -> tuple[dict[str, Any], Position] | None:
        """Process partial exit logic for profit taking or loss mitigation.

        Evaluates exit tiers from highest to lowest and triggers a partial
        close or full close if a hard limit is reached.

        Args:
            position: Current position state.
            pnl_pct: Unrealized PnL as a decimal fraction (e.g. 0.05 for 5%).
            tiers: Exit tier thresholds and take percentages.

        Returns:
            Tuple of (exit decision dict, updated position) or None if no
            tier was triggered.
        """
        abs_pnl = abs(pnl_pct)
        is_profit = pnl_pct > 0

        # Determine labels
        type_label = "Profit taking" if is_profit else "Loss mitigation"

        # Check tiers from highest (Level 3) to lowest (Level 1)
        for i in range(3, 0, -1):
            level_key = f"level{i}"
            take_key = f"take{i}"

            if abs_pnl >= tiers[level_key]:
                # Avoid repeat execution for the same level in the same direction (especially for loss mitigation)
                executed_flag = f"{type_label.lower().replace(' ', '_')}_L{i}_active"
                if position.partial_exit_flags.get(executed_flag, False):
                    continue

                take_percent = tiers[take_key]
                adjusted_percent, force_close, reason = self.pm._adjust_partial_sale(
                    position, take_percent
                )

                if force_close:
                    return {
                        "action": "close_position",
                        "reason": reason or f"Hard limit reached during {type_label}",
                    }, position

                if adjusted_percent > 0:
                    position = position.model_copy(
                        update={
                            "partial_exit_flags": {
                                **position.partial_exit_flags,
                                executed_flag: True,
                            }
                        }
                    )
                    return {
                        "action": "partial_close",
                        "percent": adjusted_percent,
                        "reason": f"{type_label} at {tiers[level_key] * 100:.1f}% {'gain' if is_profit else 'loss'} ({adjusted_percent * 100:.0f}%)",
                    }, position
        return None

    def enhanced_exit_strategy(
        self, position: Position, current_price: float, coin: str
    ) -> tuple[dict[str, Any], Position]:
        """Evaluate enhanced exit strategy with dynamic profit taking and loss cutting.

        Combines extended cycle exits, hard stop loss, partial exit tiers,
        graduated loss cutting, and trailing stop logic.

        Args:
            position: Current position state.
            current_price: Current market price.
            coin: Coin symbol.

        Returns:
            Tuple of (exit decision dict, updated position).
        """
        entry_price = position.entry_price
        if entry_price == 0:
            entry_price = position.current_price or 0
            position = position.model_copy(update={"entry_price": entry_price})
            self.pm.positions[coin] = position
        direction = position.direction
        exit_plan = self._ensure_exit_plan(
            position.direction,
            position.entry_price or position.current_price,
            position.symbol,
            position.exit_plan.model_dump(),
        )
        exit_plan.get("stop_loss")
        profit_target = exit_plan.get("profit_target")
        notional_usd = position.notional_usd

        exit_decision = {"action": "hold", "reason": "No exit trigger"}

        position.margin_usd
        margin_used = position.margin_usd or (position.notional_usd / max(position.leverage, 1))
        loss_cycle_count = position.loss_cycle_count
        profit_cycle_count = position.profit_cycle_count
        unrealized_pnl = position.unrealized_pnl

        # Extended loss exit - close after N negative cycles
        if loss_cycle_count >= Config.EXTENDED_LOSS_CYCLES and unrealized_pnl <= 0:
            reason = f"Position negative for {loss_cycle_count} cycles"
            logger.warning(
                "Extended loss exit: {} {} closed ({}).", position.symbol, direction, reason
            )
            return {"action": "close_position", "reason": reason}, position

        # Extended profit exit - take profit after N positive cycles
        if profit_cycle_count >= Config.EXTENDED_PROFIT_CYCLES and unrealized_pnl > 0:
            reason = f"Taking profit after {profit_cycle_count} profitable cycles (PnL ${unrealized_pnl:.2f})"
            logger.success(
                "Extended profit exit: {} {} closed ({}).", position.symbol, direction, reason
            )
            return {"action": "close_position", "reason": reason}, position

        # --- 1. HARD STOP LOSS ENFORCEMENT ---
        primary_sl = exit_plan.get("stop_loss")
        if primary_sl and isinstance(primary_sl, (int, float)) and primary_sl > 0:
            sl_hit = False
            if direction == "long" and current_price <= primary_sl:
                sl_hit = True
            elif direction == "short" and current_price >= primary_sl:
                sl_hit = True

            if sl_hit:
                reason = f"Stop Loss (${primary_sl:.6f}) hit at ${current_price:.6f}"
                logger.warning(
                    "STOP LOSS HIT: {} {} closed at ${:.6f} ({})",
                    position.symbol,
                    direction,
                    current_price,
                    reason,
                )
                return {"action": "close_position", "reason": reason}, position

        # Get dynamic exit tiers based on notional size
        exit_tiers = self.get_dynamic_exit_tiers(notional_usd)

        # Calculate PnL percent
        unrealized_pnl_percent = (unrealized_pnl / notional_usd) if notional_usd else 0.0

        # --- 2. CENTRAL PARTIAL EXIT ENGINE (Evaluated BEFORE Graduated Loss Cut) ---
        partial_exit_decision = self._process_partial_exit_logic(
            position, unrealized_pnl_percent, exit_tiers
        )
        if partial_exit_decision:
            return partial_exit_decision

        # --- 3. GRADUATED LOSS CUTTING MECHANISM (Margin-based - Final Defense) ---
        loss_multiplier = self.pm.get_graduated_loss_multiplier(margin_used)
        loss_threshold_usd = margin_used * loss_multiplier

        if direction == "long":
            unrealized_loss_usd = max(0.0, (entry_price - current_price) * position.quantity)
        else:
            unrealized_loss_usd = max(0.0, (current_price - entry_price) * position.quantity)

        if unrealized_loss_usd >= loss_threshold_usd > 0:
            logger.warning(
                "GRADUATED LOSS CUTTING: {} {} ${:.2f} loss (threshold: ${:.2f}). Closing position.",
                direction,
                position.symbol,
                unrealized_loss_usd,
                loss_threshold_usd,
            )
            return {
                "action": "close_position",
                "reason": f"Margin-based loss cut ${unrealized_loss_usd:.2f} >= ${loss_threshold_usd:.2f}",
            }, position

        # FIRST: Always evaluate and update trailing stop when in profit
        if unrealized_pnl_percent > 0:
            trailing_action, position = self._evaluate_trailing_stop(
                position=position,
                current_price=current_price,
                profit_target=profit_target,
                direction=direction,
                entry_price=entry_price,
                unrealized_pnl_percent=unrealized_pnl_percent,
                profit_levels=exit_tiers,
                coin=coin,
            )
        else:
            trailing_action = None

        # If no partial exit but trailing stop was updated, return trailing action
        if trailing_action:
            return trailing_action, position

        return exit_decision, position

    def _evaluate_trailing_stop(
        self,
        position: Position,
        current_price: float,
        profit_target: float | None,
        direction: str,
        entry_price: float,
        unrealized_pnl_percent: float,
        profit_levels: dict[str, float],
        coin: str,
    ) -> tuple[dict[str, Any] | None, Position]:
        """Evaluate advanced trailing stop conditions.

        Considers progress toward profit target, time in trade, volume drops,
        ATR-based buffers, and overbought protection to tighten the stop loss.

        Args:
            position: Current position state.
            current_price: Current market price.
            profit_target: Target profit price from exit plan, or None.
            direction: Trade direction, either "long" or "short".
            entry_price: Entry price of the position.
            unrealized_pnl_percent: Unrealized PnL as a decimal fraction.
            profit_levels: Exit tier thresholds and take percentages.
            coin: Coin symbol.

        Returns:
            Tuple of (trailing stop update dict or None, updated position).
        """
        if (
            unrealized_pnl_percent <= 0
            or not isinstance(current_price, (int, float))
            or current_price <= 0
        ):
            return None, position

        symbol = position.symbol
        exit_plan = position.exit_plan

        level1_threshold = 0.0
        if isinstance(profit_levels, dict):
            try:
                level1_threshold = float(profit_levels.get("level1", 0.0) or 0.0)
            except (TypeError, ValueError):
                level1_threshold = 0.0
        if unrealized_pnl_percent < max(level1_threshold * 0.5, 0.0):
            return None, position

        existing_stop = exit_plan.stop_loss
        try:
            existing_stop = float(existing_stop) if existing_stop is not None else None
        except (TypeError, ValueError):
            existing_stop = None

        # Calculate progress toward profit target (in %)
        progress_pct = 0.0
        progress_valid = False
        if (
            isinstance(profit_target, (int, float))
            and profit_target > 0
            and profit_target != entry_price
        ):
            if direction == "long":
                denominator = profit_target - entry_price
                if denominator > 0:
                    progress_pct = ((current_price - entry_price) / denominator) * 100
                    progress_valid = True
            elif direction == "short":
                denominator = entry_price - profit_target
                if denominator > 0:
                    progress_pct = ((entry_price - current_price) / denominator) * 100
                    progress_valid = True
        progress_pct = max(0.0, min(progress_pct, 200.0))

        pnl_percent = max(0.0, unrealized_pnl_percent * 100.0)
        progress_score = progress_pct if progress_valid else pnl_percent

        # Time in trade (minutes)
        time_in_trade = 0.0
        entry_time_str = position.entry_time
        if entry_time_str:
            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                time_in_trade = max(
                    0.0, (datetime.now(timezone.utc) - entry_time).total_seconds() / 60.0
                )
            except Exception:
                time_in_trade = 0.0

        # Dynamic trailing trigger based on price location
        # In extreme zones (LOWER_10/UPPER_10), use lower threshold for earlier trailing stop activation
        effective_progress_trigger = Config.TRAILING_PROGRESS_TRIGGER
        extreme_zone_active = False
        try:
            indicators_htf_early = (
                self.pm.market_data.get_technical_indicators(symbol, HTF_INTERVAL)
                if self.pm.market_data
                else {}
            )
            if isinstance(indicators_htf_early, dict) and "error" not in indicators_htf_early:
                sparkline_early = indicators_htf_early.get("smart_sparkline", {})
                price_loc_early = (
                    sparkline_early.get("price_location", {})
                    if isinstance(sparkline_early, dict)
                    else {}
                )
                zone_early = price_loc_early.get("zone", "MIDDLE")
                if zone_early in ["LOWER_10", "UPPER_10"]:
                    effective_progress_trigger = Config.TRAILING_PROGRESS_TRIGGER_EXTREME
                    extreme_zone_active = True
        except Exception as e:
            # FIX: Log the error instead of silently swallowing
            logger.warning("Trailing stop progress calculation failed: {}", e)

        progress_triggered = progress_score >= effective_progress_trigger
        time_triggered = (
            time_in_trade >= Config.TRAILING_TIME_MINUTES
            and progress_score >= Config.TRAILING_TIME_PROGRESS_FLOOR
        )
        if not (progress_triggered or time_triggered):
            return None, position

        # Fetch current 3m indicators for ATR & volume context
        current_volume_ratio = None
        atr_value = None
        try:
            indicators_3m = (
                self.pm.market_data.get_technical_indicators(symbol, "3m")
                if self.pm.market_data
                else {}
            )
        except Exception as exc:
            logger.warning("Trailing stop indicator fetch failed for {}: {}", symbol, exc)
            indicators_3m = {}

        if isinstance(indicators_3m, dict):
            volume_now = indicators_3m.get("volume")
            avg_volume_now = indicators_3m.get("avg_volume")
            if (
                isinstance(volume_now, (int, float))
                and isinstance(avg_volume_now, (int, float))
                and avg_volume_now > 0
            ):
                current_volume_ratio = volume_now / avg_volume_now
            atr_value = indicators_3m.get("atr_14")

        if not isinstance(atr_value, (int, float)) or atr_value <= 0:
            atr_value = position.entry_atr_14
        if not isinstance(atr_value, (int, float)) or atr_value <= 0:
            atr_value = current_price * Config.TRAILING_FALLBACK_BUFFER_PCT

        entry_volume_ratio = position.entry_volume_ratio
        volume_drop_triggered = False
        if isinstance(current_volume_ratio, (int, float)):
            if current_volume_ratio <= Config.TRAILING_VOLUME_ABSOLUTE_THRESHOLD:
                volume_drop_triggered = True
            elif isinstance(entry_volume_ratio, (int, float)) and entry_volume_ratio > 0:
                if current_volume_ratio <= entry_volume_ratio * Config.TRAILING_VOLUME_DROP_RATIO:
                    volume_drop_triggered = True

        min_improvement_abs = max(
            current_price * Config.TRAILING_MIN_IMPROVEMENT_PCT, Config.MIN_EXIT_PLAN_OFFSET, 1e-07
        )
        atr_buffer = max(atr_value * Config.TRAILING_ATR_MULTIPLIER, min_improvement_abs)

        # UPPER_10 + Overbought Profit Protection
        # Tighten trailing stop if price is in the top 10% zone AND RSI > 70
        overbought_protect_active = False
        try:
            indicators_htf = (
                self.pm.market_data.get_technical_indicators(symbol, HTF_INTERVAL)
                if self.pm.market_data
                else {}
            )
            if isinstance(indicators_htf, dict) and "error" not in indicators_htf:
                rsi_htf = indicators_htf.get("rsi_13", 50)
                sparkline = indicators_htf.get("smart_sparkline", {})
                price_loc = (
                    sparkline.get("price_location", {}) if isinstance(sparkline, dict) else {}
                )
                zone = price_loc.get("zone", "MIDDLE")

                # Halve buffer in UPPER_10 + RSI > 70 condition
                if (
                    zone == "UPPER_10"
                    and isinstance(rsi_htf, (int, float))
                    and rsi_htf > constants.RSI_HTF_OVERBOUGHT
                ):
                    atr_buffer = atr_buffer * 0.5
                    overbought_protect_active = True
                    logger.info(
                        "OVERBOUGHT PROTECT: {} zone={} RSI={:.1f} -> Buffer halved",
                        symbol,
                        zone,
                        rsi_htf,
                    )
        except Exception as e:
            # FIX: Log the error instead of silently swallowing
            logger.warning("Overbought protection calculation failed: {}", e)

        reason_tokens: list[str] = []
        if progress_triggered:
            reason_tokens.append(f"progress {progress_score:.1f}%")
        if extreme_zone_active:
            reason_tokens.append(f"extreme_zone (trigger {effective_progress_trigger:.0f}%)")
        if time_triggered:
            reason_tokens.append(f"time {time_in_trade:.1f}m")
        if volume_drop_triggered and isinstance(current_volume_ratio, (int, float)):
            reason_tokens.append(f"volume {current_volume_ratio:.2f}x")
        if overbought_protect_active:
            reason_tokens.append("overbought_protect")

        if not reason_tokens:
            reason_tokens.append("trailing criteria met")

        new_stop: float | None = None
        if direction == "long":
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
                return None, position

            if baseline_stop <= 0:
                return None, position

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
                return None, position

            if existing_stop is not None and (existing_stop - baseline_stop) <= min_improvement_abs:
                return None, position

            new_stop = baseline_stop

        if new_stop is None:
            return None, position

        new_stop = round(max(0.0, new_stop), 6)
        if direction == "long" and new_stop >= current_price:
            return None, position
        if direction == "short" and new_stop <= current_price:
            return None, position

        # Persist updated stop and trailing metadata
        new_exit_plan = position.exit_plan.model_copy(update={"stop_loss": new_stop})
        trailing_update: dict[str, Any] = {
            "active": True,
            "last_update_cycle": getattr(self, "current_cycle_number", None),
            "last_reason": ", ".join(reason_tokens) if reason_tokens else None,
            "last_stop": new_stop,
            "progress_percent": round(progress_score, 2),
            "time_in_trade_min": round(time_in_trade, 2),
        }
        new_trailing = position.trailing.model_copy(update=trailing_update)
        if isinstance(current_volume_ratio, (int, float)):
            new_trailing = new_trailing.model_copy(
                update={"last_volume_ratio": round(current_volume_ratio, 4)}
            )
        position = position.model_copy(
            update={"exit_plan": new_exit_plan, "trailing": new_trailing}
        )
        self.pm.positions[coin] = position

        reason = f"Trailing stop tightened ({', '.join(reason_tokens)})"
        return {"action": "update_stop", "new_stop": new_stop, "reason": reason}, position

    def _execute_new_positions_only(
        self,
        decisions: dict,
        valid_prices: dict,
        cycle_number: int,
        indicator_cache: dict[str, dict[str, Any]] | None = None,
    ):
        """Execute only new position entries after an AI close_position signal.

        Filters decisions to entry signals only and respects the graduated
        position limit for the current cycle.

        Args:
            decisions: Trade decisions from the AI engine keyed by coin.
            valid_prices: Current valid prices keyed by coin.
            cycle_number: Current trading cycle number.
            indicator_cache: Optional pre-fetched indicator data.
        """
        logger.info("Executing new positions only (after close_position signal)")

        # KADEMELİ POZİSYON SİSTEMİ: Cycle bazlı pozisyon limiti
        max_positions_for_cycle = self.pm.get_max_positions_for_cycle(cycle_number)
        current_positions = len(self.pm.positions)

        decisions_to_execute = {}
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                continue

            signal = trade.get("signal")
            if signal in ["buy_to_enter", "sell_to_enter"]:
                # Apply kademeli position limit
                if current_positions >= max_positions_for_cycle:
                    logger.warning(
                        "KADEMELİ POZİSYON LİMİTİ (Cycle {}): Max {} positions allowed. Skipping {} entry.",
                        cycle_number,
                        max_positions_for_cycle,
                        coin,
                    )
                    continue
                current_positions += 1

                decisions_to_execute[coin] = trade

        if decisions_to_execute:
            self.pm.execute_decision(
                decisions_to_execute,
                valid_prices,
                indicator_cache=indicator_cache,
            )
