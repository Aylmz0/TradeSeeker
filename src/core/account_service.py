import copy
from datetime import datetime
from typing import Any
from config.config import Config
from src.utils import format_num
from src.core.data_engine import DataEngine

try:
    from src.services.binance import BinanceOrderExecutor, BinanceAPIError
    BINANCE_IMPORT_ERROR = None
except Exception as e:
    BinanceOrderExecutor = None
    BINANCE_IMPORT_ERROR = str(e)
    class BinanceAPIError(Exception): pass

HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'
HTF_LABEL = HTF_INTERVAL

class AccountService:
    def __init__(self, portfolio_manager):
        self.pm = portfolio_manager
        self.is_live_trading = getattr(Config, 'TRADING_MODE', 'simulation') == 'live'
        self.order_executor = None
        if self.is_live_trading:
            self._initialize_live_trading()
        elif BINANCE_IMPORT_ERROR:
            print(f'[INFO] Binance executor unavailable ({BINANCE_IMPORT_ERROR}). Staying in simulation mode.')

    def _initialize_live_trading(self):
        """Configure Binance executor when live trading mode is enabled."""
        if BinanceOrderExecutor is None:
            print("[ERROR] Live trading requested but Binance executor is unavailable.")
            self.is_live_trading = False
            return
        try:
            self.order_executor = BinanceOrderExecutor(self.pm.market_data.available_coins)
            if not self.order_executor.is_live():
                print(
                    "[WARNING] Live trading requested but executor initialized in simulation mode. Reverting to paper trading."
                )
                self.is_live_trading = False
                self.order_executor = None
                return
            print(
                f"[SUCCESS] Live trading mode enabled (Binance {'TESTNET' if Config.BINANCE_TESTNET else 'MAINNET'})."
            )
            self.sync_live_account()
        except BinanceAPIError as exc:
            print(f"[ERROR] Binance setup failed: {exc}. Reverting to simulation mode.")
            self.is_live_trading = False
            self.order_executor = None
        except Exception as exc:
            print(f"[ERROR] Unexpected Binance setup error: {exc}. Reverting to simulation mode.")
            self.is_live_trading = False
            self.order_executor = None

    def _build_default_exit_plan(self, direction: str, entry_price: float) -> dict[str, float]:
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
        if direction == "short":
            stop_loss = entry + stop_offset
            profit_target = max(entry - tp_offset, 0.0)
        else:
            stop_loss = max(entry - stop_offset, 0.0)
            profit_target = entry + tp_offset
        return {"stop_loss": stop_loss, "profit_target": profit_target}

    def _ensure_exit_plan(
        self, position: dict[str, Any], *candidate_plans: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Ensure a position carries a valid exit plan, supplementing with defaults if needed."""
        direction = position.get("direction", "long")
        entry_price = position.get("entry_price") or position.get("current_price")
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
            symbol = position.get("symbol", "UNKNOWN")
            print(
                f"[WARNING] Missing {', '.join(missing_required)} for {symbol} - using default exit plan offsets."
            )
        position["exit_plan"] = final_plan
        return final_plan

    def _merge_live_positions(
        self, snapshot: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Merge Binance snapshot with local runtime metadata (exit plans, confidence, etc.)."""
        merged: dict[str, dict[str, Any]] = {}
        existing_positions = self.pm.positions if isinstance(self.pm.positions, dict) else {}
        for coin, snap_pos in snapshot.items():
            previous = existing_positions.get(coin, {})
            merged_pos: dict[str, Any] = {}
            merged_pos.update(previous)
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
                if key in previous and key not in merged_pos:
                    merged_pos[key] = previous[key]

            # Risk USD should reflect current margin if available
            margin_usd = merged_pos.get("margin_usd")
            if isinstance(margin_usd, (int, float)):
                merged_pos["risk_usd"] = margin_usd

            existing_plan = previous.get("exit_plan")
            snapshot_plan = snap_pos.get("exit_plan")
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
                print(
                    f"[DEBUG] Binance API Response: availableBalance={available}, totalWalletBalance={total_wallet_balance}, walletBalance={wallet_balance}"
                )

                # Update available cash balance
                if available is not None and available > 0:
                    old_balance = self.pm.current_balance
                    self.pm.current_balance = float(available)
                    if (
                        abs(old_balance - self.pm.current_balance) > 0.01
                    ):  # Only log if significant change
                        print(
                            f"[INFO] Balance updated: ${old_balance:.2f} → ${self.pm.current_balance:.2f}"
                        )

                # Note: We'll calculate total_value manually after positions are synced
                # Total value = Available cash + Margin used + Unrealized PnL
                # Binance totalWalletBalance is used for validation only
                if total_wallet_balance is not None and total_wallet_balance > 0:
                    print(
                        f"[SUCCESS] Binance totalWalletBalance: ${total_wallet_balance:.2f} (will validate against calculated value)"
                    )
                elif wallet_balance is not None and wallet_balance > 0:
                    print(
                        f"[WARNING] totalWalletBalance not available, using walletBalance: ${wallet_balance:.2f}"
                    )
                else:
                    print(
                        "[WARNING] Neither totalWalletBalance nor walletBalance available from Binance API"
                    )
        except BinanceAPIError as exc:
            print(f"[WARNING] Binance balance sync failed: {exc}")
        except Exception as exc:
            print(f"[WARNING] Unexpected Binance balance sync error: {exc}")

        try:
            snapshot = self.order_executor.get_positions_snapshot()
            if isinstance(snapshot, dict):
                self.pm.positions = self._merge_live_positions(snapshot)

                # Update erosion tracking for each position using existing method
                for coin, pos in self.pm.positions.items():
                    self.pm._update_peak_pnl_tracking(coin, pos)

                # Calculate total margin used from all open positions
                # For cross margin, margin_usd might be 0, so calculate from notional/leverage
                total_margin_used = 0.0
                for pos in self.pm.positions.values():
                    margin = pos.get("margin_usd", 0.0)
                    if margin <= 0:
                        # Calculate margin from notional and leverage (for cross margin)
                        notional = pos.get("notional_usd", 0.0)
                        leverage = pos.get("leverage", 1)
                        if notional > 0 and leverage > 0:
                            margin = notional / leverage
                    if isinstance(margin, (int, float)) and margin > 0:
                        total_margin_used += margin

                old_total = self.pm.total_value

                # Calculate total unrealized PnL
                total_unrealized_pnl = 0.0
                for pos in self.pm.positions.values():
                    pnl = pos.get("unrealized_pnl", 0.0)
                    if isinstance(pnl, (int, float)):
                        total_unrealized_pnl += pnl

                # Total value = Available cash + Margin used + Unrealized PnL
                # This is the correct formula: what you have available + what's locked in positions + unrealized gains/losses
                self.pm.total_value = self.pm.current_balance + total_margin_used + total_unrealized_pnl

                if abs(old_total - self.pm.total_value) > 0.01:
                    print(f"[STATS] Total value updated: ${old_total:.2f} → ${self.pm.total_value:.2f}")
                    print(
                        f"   (Available cash: ${self.pm.current_balance:.2f} + Margin used: ${total_margin_used:.2f} + Unrealized PnL: ${total_unrealized_pnl:.2f})"
                    )
                    print("   [INFO] Unrealized PnL from Binance (includes funding fees)")

                    # Debug: Also show what Binance says for comparison
                    if overview:
                        total_wb = overview.get("totalWalletBalance")
                        wallet_b = overview.get("walletBalance")
                        if total_wb:
                            wallet_b_str = f"${wallet_b:.2f}" if wallet_b else "N/A"
                            print(
                                f"   (Binance totalWalletBalance: ${total_wb:.2f}, walletBalance: {wallet_b_str})"
                            )
                            # Validate our calculation against Binance
                            diff = abs(self.pm.total_value - total_wb)
                            if diff > 0.10:  # More than 10 cents difference
                                print(
                                    f"   [WARNING] Warning: Calculated total_value differs from Binance totalWalletBalance by ${diff:.2f}"
                                )
            else:
                self.pm.positions = {}
                # No positions, total value = totalWalletBalance (or available cash if not available)
                if (
                    overview
                    and overview.get("totalWalletBalance")
                    and overview.get("totalWalletBalance") > 0
                ):
                    self.pm.total_value = float(overview["totalWalletBalance"])
                else:
                    self.pm.total_value = self.pm.current_balance
        except BinanceAPIError as exc:
            print(f"[WARNING] Binance position sync failed: {exc}")
        except Exception as exc:
            print(f"[WARNING] Unexpected Binance position sync error: {exc}")

    @staticmethod
    def _calculate_realized_pnl(
        entry_price: float,
        exit_price: float,
        quantity: float,
        direction: str,
        include_commission: bool = True,
    ) -> float:
        """Calculate realized PnL with optional commission deduction for simulation realism.

        Commission is applied as round-trip (entry + exit) = 2 * SIMULATION_COMMISSION_RATE
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

    def execute_live_entry(
        self,
        coin: str,
        direction: str,
        quantity: float,
        leverage: int,
        current_price: float,
        notional_usd: float,
        confidence: float,
        margin_usd: float | None = None,
        stop_loss: float | None = None,
        profit_target: float | None = None,
        invalidation: str | None = None,
    ) -> dict[str, Any]:
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
            
            # API Consistency Buffer (Ghost Position Ping)
            for attempt in range(5):
                self.sync_live_account()
                if coin in self.pm.positions:
                    break
                import time
                print(f"[INFO] Resolving Binance replication lag for {coin}... (attempt {attempt+1}/5)")
                time.sleep(1)
                
            position = self.pm.positions.get(coin, {})

            # Use provided margin_usd if available, otherwise calculate from position or notional
            if margin_usd is None or margin_usd <= 0:
                if position and position.get("margin_usd", 0) > 0:
                    margin_usd = position.get("margin_usd")
                else:
                    # Calculate from executed notional (more accurate than initial notional)
                    executed_notional = executed_qty * avg_price
                    margin_usd = executed_notional / max(leverage, 1)

            notional_runtime = (
                position.get("notional_usd") if position else executed_qty * avg_price
            )
            if position:
                position["confidence"] = confidence
                # Ensure margin_usd is saved to position for later use in TP/SL checks
                if "margin_usd" not in position or position.get("margin_usd", 0) <= 0:
                    position["margin_usd"] = margin_usd
                    print(f"[INFO] Saved margin_usd=${margin_usd:.2f} to position for {coin}")
                exit_plan = position.setdefault("exit_plan", {})
                if stop_loss is not None:
                    exit_plan["stop_loss"] = stop_loss
                if profit_target is not None:
                    exit_plan["profit_target"] = profit_target
                if invalidation is not None:
                    exit_plan["invalidation_condition"] = invalidation
                position["risk_usd"] = position.get("margin_usd", margin_usd)
                self._ensure_exit_plan(position, exit_plan)

                # Calculate and save ATR-based stop loss to exit_plan (for 30-second monitoring)
                # Backend Authority: AI's stop_loss suggestion is IGNORED - system uses ATR only
                if executed_qty > 0:
                    # Fetch HTF (1h) indicators for ATR value
                    try:
                        from src.core.market_data import RealMarketData

                        market_data = RealMarketData()
                        indicators_htf = market_data.get_technical_indicators(coin, HTF_INTERVAL)
                        atr_value = (
                            indicators_htf.get("atr_14")
                            if isinstance(indicators_htf, dict)
                            else None
                        )
                    except Exception as e:
                        print(f"[WARNING] ATR fetch failed for {coin}: {e}")
                        atr_value = None

                    # Fallback: If ATR unavailable, use 2% of price
                    if not atr_value or atr_value <= 0:
                        atr_value = avg_price * 0.02
                        print(f"[WARNING] ATR fallback for {coin}: Using 2% of price = ${atr_value:.4f}")

                    # Calculate stop distance using Config multiplier
                    sl_distance = atr_value * Config.ATR_SL_MULTIPLIER
                    tp_distance = atr_value * Config.ATR_TP_MULTIPLIER

                    # Calculate stop loss and profit target based on direction
                    if direction == "long":
                        final_stop_loss = avg_price - sl_distance
                        final_profit_target = avg_price + tp_distance
                    else:  # short
                        final_stop_loss = avg_price + sl_distance
                        final_profit_target = avg_price - tp_distance

                    # Final validation: ensure stop loss direction is correct
                    if direction == "long":
                        if final_stop_loss >= avg_price:
                            final_stop_loss = avg_price - (avg_price * 0.02)
                            print(
                                f"[WARNING] Final validation: Stop loss for {coin} LONG was invalid (>= entry), recalculated to ${format_num(final_stop_loss, 4)}"
                            )
                    elif final_stop_loss <= avg_price:
                        final_stop_loss = avg_price + (avg_price * 0.02)
                        print(
                            f"[WARNING] Final validation: Stop loss for {coin} SHORT was invalid (<= entry), recalculated to ${format_num(final_stop_loss, 4)}"
                        )

                    # Save ATR-based stop loss and profit target to exit_plan
                    if final_stop_loss > 0:
                        exit_plan["stop_loss"] = final_stop_loss
                        exit_plan["profit_target"] = final_profit_target
                        print(
                            f"[INFO] ATR-based SL/TP saved for {coin}: SL=${format_num(final_stop_loss, 4)}, TP=${format_num(final_profit_target, 4)} (ATR={atr_value:.4f} x {Config.ATR_SL_MULTIPLIER}/{Config.ATR_TP_MULTIPLIER}) - Backend Authority"
                        )
            # Decision Feedback Hook: Log OPEN trade to SQLite
            try:
                DataEngine().log_decision_open(
                    coin=coin,
                    direction=direction,
                    ai_confidence=confidence,
                    ml_probability=0.0,  # Populated by caller if available
                    entry_price=avg_price
                )
            except Exception:
                pass  # Never crash the trade flow for logging

            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "margin_usd": margin_usd,
                "notional_usd": notional_runtime,
            }
        except BinanceAPIError as exc:
            print(f"[ERROR] Binance entry order failed for {coin}: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            print(f"[ERROR] Unexpected Binance entry error for {coin}: {exc}")
            return {"success": False, "error": str(exc)}

    def execute_live_close(
        self,
        coin: str,
        position: dict[str, Any],
        current_price: float,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        if not position:
            return {"success": False, "error": "no_position"}

        direction = position.get("direction", "long")
        quantity = float(position.get("quantity", 0.0) or 0.0)
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
                position.get("entry_price", 0.0),
                avg_price,
                executed_qty,
                direction,
                include_commission=False,
            )  # Live: Binance already deducted
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
            # Decision Feedback Hook: Log CLOSED trade to SQLite
            try:
                DataEngine().log_decision_close(
                    coin=coin,
                    exit_price=avg_price,
                    pnl_result=pnl
                )
            except Exception:
                pass  # Never crash the trade flow for logging

            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
                "pnl": pnl,
                "history_entry": history_entry,
            }
        except BinanceAPIError as exc:
            print(f"[ERROR] Binance close order failed for {coin}: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            print(f"[ERROR] Unexpected Binance close error for {coin}: {exc}")
            return {"success": False, "error": str(exc)}

    def execute_live_partial_close(
        self,
        coin: str,
        position: dict[str, Any],
        close_percent: float,
        current_price: float,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        if not position:
            return {"success": False, "error": "no_position"}

        direction = position.get("direction", "long")
        quantity = float(position.get("quantity", 0.0) or 0.0)
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
                position.get("entry_price", 0.0),
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
            print(f"[ERROR] Binance partial close failed for {coin}: {exc}")
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            print(f"[ERROR] Unexpected Binance partial close error for {coin}: {exc}")
            return {"success": False, "error": str(exc)}

    def close_position(
        self, coin: str, current_price: float, reason: str = "Manual Close"
    ) -> dict[str, Any]:
        """
        Close a position in paper trading mode (simulation).
        Includes commission deduction for realism.

        Args:
            coin: Coin symbol
            current_price: Current market price
            reason: Reason for closing

        Returns:
            Dict with success status and PnL
        """
        if coin not in self.pm.positions:
            return {"success": False, "error": "no_position"}

        position = self.pm.positions[coin]
        direction = position.get("direction", "long")
        entry_price = position.get("entry_price", 0)
        quantity = position.get("quantity", 0)
        margin_used = position.get("margin_usd", 0)

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
            "notional_usd": position.get("notional_usd", 0),
            "pnl": profit,
            "entry_time": position.get("entry_time", datetime.now().isoformat()),
            "exit_time": datetime.now().isoformat(),
            "leverage": position.get("leverage", 10),
            "close_reason": reason,
        }

        self.pm.add_to_history(history_entry)

        # Remove from active positions
        del self.pm.positions[coin]

        print(
            f"[SUCCESS] PAPER CLOSE: {direction} {coin} @ ${format_num(current_price, 4)} (PnL: ${format_num(profit, 2)}, Commission: ${format_num(commission, 3)})"
        )

        self.pm.save_state()

        return {"success": True, "pnl": profit, "commission": commission}

    def check_and_execute_tp_sl(self, current_prices: dict[str, float]):
        """Checks if any open position hit TP or SL and closes them automatically with enhanced exit strategies.

        This function is called every 30 seconds by the monitoring loop:
        - All TP/SL decisions are made by this monitoring (like simulation mode)
        - No Binance TP/SL orders - all managed by this loop
        - Kademeli margin-based stop loss is checked and positions are closed accordingly
        """
        # Enhanced exit strategy control - check if enabled
        if hasattr(self, "bot") and not self.pm.bot.enhanced_exit_enabled:
            print("⏸️ Enhanced exit strategy paused during cycle")
            return False

        # All TP/SL decisions made by 30-second monitoring (like simulation mode)
        # No Binance TP/SL orders - all managed by monitoring loop
        print("[INFO] Checking for TP/SL triggers (30-second monitoring mode)")

        closed_positions = []  # Keep track of positions closed in this check
        updated_stops = []  # Track positions with updated trailing stops
        state_changed = False

        for coin, position in list(self.pm.positions.items()):  # Iterate over a copy for safe deletion
            if (
                coin not in current_prices
                or not isinstance(current_prices[coin], (int, float))
                or current_prices[coin] <= 0
            ):
                continue  # Skip if price is invalid

            current_price = current_prices[coin]

            # Update erosion tracking (captures intraday peaks)
            self.pm._update_peak_pnl_tracking(coin, position)

            exit_plan = position.get("exit_plan", {})
            tp = exit_plan.get("profit_target")
            sl = exit_plan.get("stop_loss")
            direction = position.get("direction", "long")
            entry_price = position["entry_price"]
            quantity = position["quantity"]

            # Calculate margin_used properly - try multiple fallback methods
            margin_used = position.get("margin_usd")
            if margin_used is None or margin_used <= 0:
                # Fallback 1: Calculate from notional and leverage
                notional = position.get("notional_usd", 0)
                leverage = position.get("leverage", 1)
                if notional > 0 and leverage > 0:
                    margin_used = notional / leverage
                # Fallback 2: Calculate from entry_price and quantity
                elif entry_price > 0 and quantity > 0:
                    notional = entry_price * quantity
                    leverage = position.get("leverage", 10)
                    margin_used = notional / leverage
                else:
                    margin_used = 0

            # Debug log if margin_used is still 0
            if margin_used <= 0:
                print(
                    f"⚠️ Warning: margin_used is 0 for {coin}. Position data: margin_usd={position.get('margin_usd')}, notional={position.get('notional_usd')}, leverage={position.get('leverage')}, entry={entry_price}, qty={quantity}"
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
            exit_decision = self.enhanced_exit_strategy(position, current_price)

            # Handle enhanced exit strategy signals - ANINDA İŞLEME
            if exit_decision["action"] == "close_position":
                # Enhanced exit strategy wants to close the position completely
                close_reason = exit_decision["reason"]
                print(
                    f"[ALERT] ENHANCED EXIT CLOSE {coin} ({direction}): {close_reason} at price ${format_num(current_price, 4)}"
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
                        print(
                            f"🚫 Live partial close failed for {coin}: {live_result.get('error', 'unknown_error')}"
                        )
                        continue
                    history_entry = live_result.get("history_entry")
                    if history_entry:
                        self.pm.add_to_history(history_entry)
                    print(
                        f"[ALERT] PARTIAL CLOSE {coin} ({direction}) [LIVE]: {exit_decision['reason']} ({close_percent * 100:.0f}% / PnL ${format_num(live_result.get('pnl', 0), 2)})"
                    )
                    # BUG FIX: Adjust peak_pnl proportionally after partial close
                    # Without this, erosion tracking would falsely alarm (peak $3 -> current $1.5 = 50% erosion)
                    if "peak_pnl" in position and position["peak_pnl"] > 0:
                        old_peak = position["peak_pnl"]
                        position["peak_pnl"] = position["peak_pnl"] * (1 - close_percent)
                        print(
                            f"   [STATS] peak_pnl adjusted: ${format_num(old_peak, 2)} → ${format_num(position['peak_pnl'], 2)} (after {close_percent * 100:.0f}% close)"
                        )
                    state_changed = True
                    # Sync account balance after partial close in live mode
                    try:
                        self.sync_live_account()
                        print(f"✅ Account balance synced after partial close of {coin}")
                    except Exception as sync_exc:
                        print(f"⚠️ Failed to sync account after partial close: {sync_exc}")
                    continue

                close_quantity = quantity * close_percent

                if direction == "long":
                    profit = (current_price - entry_price) * close_quantity
                else:
                    profit = (entry_price - current_price) * close_quantity

                # Deduct commission for simulation realism (entry commission for this portion + exit commission)
                notional_closed = ((entry_price + current_price) / 2) * close_quantity
                commission = notional_closed * Config.SIMULATION_COMMISSION_RATE * 2  # Round-trip
                profit -= commission

                # Update position quantity
                position["quantity"] = quantity * (1 - close_percent)
                position["margin_usd"] = margin_used * (1 - close_percent)
                position["notional_usd"] = position["notional_usd"] * (1 - close_percent)
                # BUG FIX: Adjust peak_pnl proportionally after partial close
                if "peak_pnl" in position and position["peak_pnl"] > 0:
                    old_peak = position["peak_pnl"]
                    position["peak_pnl"] = position["peak_pnl"] * (1 - close_percent)
                    print(
                        f"   📊 peak_pnl adjusted: ${format_num(old_peak, 2)} → ${format_num(position['peak_pnl'], 2)} (after {close_percent * 100:.0f}% close)"
                    )

                # Add profit to balance
                self.pm.current_balance += margin_used * close_percent + profit

                print(
                    f"[ALERT] PARTIAL CLOSE {coin} ({direction}): {exit_decision['reason']} - Closed {close_percent * 100}% at price ${format_num(current_price, 4)}"
                )
                print(f"   Partial PnL: ${format_num(profit, 2)}")

                history_entry = {
                    "symbol": coin,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "quantity": close_quantity,
                    "notional_usd": position.get("notional_usd", "N/A") * close_percent,
                    "pnl": profit,
                    "entry_time": position["entry_time"],
                    "exit_time": datetime.now().isoformat(),
                    "leverage": position.get("leverage", "N/A"),
                    "close_reason": exit_decision["reason"],
                }
                self.pm.add_to_history(history_entry)
                state_changed = True
                continue  # Continue with remaining position

            elif exit_decision["action"] == "update_stop":
                # Update trailing stop - ANINDA GÜNCELLEME
                updated_stops.append(coin)
                new_stop = exit_decision["new_stop"]
                exit_plan["stop_loss"] = new_stop
                print(f"[INFO] TRAILING STOP UPDATE {coin}: New stop at ${format_num(new_stop, 4)}")

                # No Binance orders - stop loss updated in exit_plan, will be monitored by 30-second loop

                state_changed = True
                continue

            # Traditional TP/SL checks (only if no enhanced exit triggered)
            if close_reason is None and tp is not None:
                if (
                    direction == "long"
                    and current_price >= tp
                    or direction == "short"
                    and current_price <= tp
                ):
                    close_reason = f"Profit Target ({tp}) hit"

            # Check SL (only if TP not hit)
            # First check exit_plan stop_loss, then fallback to margin-based kademeli stop loss
            if close_reason is None:
                # Check exit_plan stop_loss first
                if sl is not None:
                    if (
                        direction == "long"
                        and current_price <= sl
                        or direction == "short"
                        and current_price >= sl
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
                        if (
                            direction == "long"
                            and current_price <= margin_based_stop_loss
                            or direction == "short"
                            and current_price >= margin_based_stop_loss
                        ):
                            close_reason = f"Margin-based Stop Loss ({format_num(margin_based_stop_loss, 4)}) hit (${loss_threshold_usd:.2f} loss limit, {loss_multiplier * 100:.1f}% of ${margin_used:.2f} margin)"

            # Execute Close if triggered
            if close_reason:
                print(
                    f"[ALERT] AUTO-CLOSE {coin} ({direction}): {close_reason} at price ${format_num(current_price, 4)}"
                )

                if self.is_live_trading:
                    print(f"🔄 Executing LIVE close on Binance for {coin}...")
                    live_result = self.execute_live_close(
                        coin=coin,
                        position=position,
                        current_price=current_price,
                        reason=close_reason,
                    )
                    if not live_result.get("success"):
                        print(
                            f"🚫 Live auto-close failed for {coin}: {live_result.get('error', 'unknown_error')}"
                        )
                        continue

                    # Log Binance order details
                    order_id = live_result.get("order", {}).get("orderId")
                    executed_qty = live_result.get("executed_qty", 0)
                    avg_price = live_result.get("avg_price", 0)
                    print(
                        f"[SUCCESS] Binance CLOSE order executed for {coin}: orderId={order_id}, qty={format_num(executed_qty, 4)}, avgPrice=${format_num(avg_price, 4)}"
                    )

                    history_entry = live_result.get("history_entry")
                    if history_entry:
                        self.pm.add_to_history(history_entry)
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

                if direction == "long":
                    profit = (current_price - entry_price) * quantity
                else:
                    profit = (entry_price - current_price) * quantity

                # Deduct commission for simulation realism (round-trip)
                notional_closed = ((entry_price + current_price) / 2) * quantity
                commission = notional_closed * Config.SIMULATION_COMMISSION_RATE * 2  # Round-trip
                profit -= commission

                self.pm.current_balance += (
                    margin_used + profit
                )  # Return margin + PnL (commission already deducted)

                print(f"   Closed PnL: ${format_num(profit, 2)}")

                history_entry = {
                    "symbol": coin,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "quantity": quantity,
                    "notional_usd": position.get("notional_usd", "N/A"),
                    "pnl": profit,
                    "entry_time": position["entry_time"],
                    "exit_time": datetime.now().isoformat(),
                    "leverage": position.get("leverage", "N/A"),
                    "close_reason": close_reason,  # Add reason
                }
                self.pm.add_to_history(history_entry)  # This increments trade_count
                closed_positions.append(coin)
                del self.pm.positions[coin]  # Remove from active positions
                state_changed = True

        if closed_positions:
            print(f"✅ Auto-closed positions: {', '.join(closed_positions)}")
        if updated_stops:
            print(f"📈 Updated trailing stops: {', '.join(updated_stops)}")

        if state_changed:
            self.pm.save_state()

        return len(closed_positions) > 0  # Indicate if any positions were closed

    def get_profit_levels_by_notional(self, notional_usd: float) -> dict[str, float]:
        """Get dynamic profit levels based on notional size"""
        if notional_usd < 100:
            # Small positions: aggressive profit taking
            return {
                "level1": 0.008,  # %0.7
                "level2": 0.009,  # %0.9
                "level3": 0.01,  # %1.1
                "take1": 0.25,  # 25% profit take
                "take2": 0.50,  # 50% profit take
                "take3": 0.75,  # 75% profit take
            }
        elif notional_usd < 200:
            # Medium positions: balanced profit taking
            return {
                "level1": 0.007,  # %0.7
                "level2": 0.008,  # %0.9
                "level3": 0.009,  # %1.1
                "take1": 0.25,  # 25% profit take
                "take2": 0.50,  # 50% profit take
                "take3": 0.75,  # 75% profit take
            }
        elif notional_usd < 300:
            # Medium positions: balanced profit taking
            return {
                "level1": 0.006,  # %0.7
                "level2": 0.007,  # %0.9
                "level3": 0.008,  # %1.1
                "take1": 0.25,  # 25% profit take
                "take2": 0.50,  # 50% profit take
                "take3": 0.75,  # 75% profit take
            }
        elif notional_usd < 400:
            # Large positions: conservative profit taking
            return {
                "level1": 0.005,  # %0.6
                "level2": 0.006,  # %0.8
                "level3": 0.007,  # %1.0
                "take1": 0.25,  # 25% profit take
                "take2": 0.50,  # 50% profit take
                "take3": 0.75,  # 75% profit take
            }
        elif notional_usd < 500:
            # xLarge positions: conservative profit taking
            return {
                "level1": 0.004,  # %0.5
                "level2": 0.005,  # %0.7
                "level3": 0.006,  # %0.9
                "take1": 0.25,  # 25% profit take
                "take2": 0.50,  # 50% profit take
                "take3": 0.75,  # 75% profit take
            }
        elif notional_usd < 600:
            # xxLarge positions: conservative profit taking
            return {
                "level1": 0.003,  # %0.
                "level2": 0.004,  # %0.6
                "level3": 0.005,  # %0.8
                "take1": 0.25,  # 25% profit take
                "take2": 0.50,  # 50% profit take
                "take3": 0.75,  # 75% profit take
            }
        else:
            # Very large positions: very conservative profit taking
            return {
                "level1": 0.002,  # %0.3
                "level2": 0.003,  # %0.5
                "level3": 0.004,  # %0.7
                "take1": 0.25,  # 25% profit take
                "take2": 0.50,  # 50% profit take
                "take3": 0.75,  # 75% profit take
            }

    def enhanced_exit_strategy(self, position: dict, current_price: float) -> dict[str, Any]:
        """Enhanced exit strategy with dynamic profit taking and KADEMELİ loss cutting"""
        entry_price = position.get("entry_price")
        if entry_price is None:
            entry_price = position.get("current_price", 0)
            position["entry_price"] = entry_price
        direction = position.get("direction", "long")
        exit_plan = self._ensure_exit_plan(position, position.get("exit_plan"))
        stop_loss = exit_plan.get("stop_loss")
        profit_target = exit_plan.get("profit_target")
        notional_usd = position.get("notional_usd", 0)

        exit_decision = {"action": "hold", "reason": "No exit trigger"}

        current_margin = position.get("margin_usd", 0)
        margin_used = position.get(
            "margin_usd", position.get("notional_usd", 0) / max(position.get("leverage", 1), 1)
        )
        loss_cycle_count = position.get("loss_cycle_count", 0)
        profit_cycle_count = position.get("profit_cycle_count", 0)
        unrealized_pnl = position.get("unrealized_pnl", 0)

        # Extended loss exit - close after N negative cycles
        if loss_cycle_count >= Config.EXTENDED_LOSS_CYCLES and unrealized_pnl <= 0:
            reason = f"Position negative for {loss_cycle_count} cycles"
            print(f"⏳ Extended loss exit: {position['symbol']} {direction} closed ({reason}).")
            return {"action": "close_position", "reason": reason}

        # Extended profit exit - take profit after N positive cycles
        if profit_cycle_count >= Config.EXTENDED_PROFIT_CYCLES and unrealized_pnl > 0:
            reason = f"Taking profit after {profit_cycle_count} profitable cycles (PnL ${unrealized_pnl:.2f})"
            print(f"💰 Extended profit exit: {position['symbol']} {direction} closed ({reason}).")
            return {"action": "close_position", "reason": reason}

        # --- GRADUATED LOSS CUTTING MECHANISM (Margin-based) ---
        # Relaxed for Volatility Sizing: Acts as "Disaster Stop" only.
        # Primary risk is controlled by Position Sizing ($3 risk).
        loss_multiplier = self.pm.get_graduated_loss_multiplier(margin_used)

        loss_threshold_usd = margin_used * loss_multiplier

        if direction == "long":
            unrealized_loss_usd = max(0.0, (entry_price - current_price) * position["quantity"])
        else:
            unrealized_loss_usd = max(0.0, (current_price - entry_price) * position["quantity"])

        if unrealized_loss_usd >= loss_threshold_usd > 0:
            print(
                f"[ALERT] GRADUATED LOSS CUTTING: {direction} {position['symbol']} ${unrealized_loss_usd:.2f} loss (threshold: ${loss_threshold_usd:.2f}). Closing position."
            )
            return {
                "action": "close_position",
                "reason": f"Margin-based loss cut ${unrealized_loss_usd:.2f} >= ${loss_threshold_usd:.2f}",
            }

        # Get dynamic profit levels based on notional size
        profit_levels = self.get_profit_levels_by_notional(notional_usd)
        level1 = profit_levels["level1"]
        level2 = profit_levels["level2"]
        level3 = profit_levels["level3"]
        take1 = profit_levels["take1"]
        take2 = profit_levels["take2"]
        take3 = profit_levels["take3"]

        print(
            f"[STATS] Dynamic profit levels for ${notional_usd:.2f} notional: {level1 * 100:.1f}%/{level2 * 100:.1f}%/{level3 * 100:.1f}%"
        )

        if direction == "long":
            unrealized_pnl_usd = max(0.0, (current_price - entry_price) * position["quantity"])
            unrealized_pnl_percent = (unrealized_pnl_usd / notional_usd) if notional_usd else 0.0

            # FIRST: Always evaluate and update trailing stop when in profit
            # This ensures stop loss is tightened even when partial take happens
            trailing_action = self._evaluate_trailing_stop(
                position=position,
                current_price=current_price,
                profit_target=profit_target,
                direction=direction,
                entry_price=entry_price,
                unrealized_pnl_percent=unrealized_pnl_percent,
                profit_levels=profit_levels,
            )
            # If trailing stop updated, the new stop is already saved in position['exit_plan']
            # We don't return here - continue to check partial take

            # Dynamic Profit Taking Levels based on notional size
            if unrealized_pnl_percent >= level3:  # Level 3 profit - take 75%
                take_profit_percent = take3
                adjusted_percent, force_close, reason = self.pm._adjust_partial_sale_for_max_limit(
                    position, take_profit_percent
                )
                if force_close:
                    return {
                        "action": "close_position",
                        "reason": reason or "Maximum limit reached during profit taking",
                    }
                if adjusted_percent > 0:
                    return {
                        "action": "partial_close",
                        "percent": adjusted_percent,
                        "reason": f"Profit taking at {level3 * 100:.1f}% gain ({adjusted_percent * 100:.0f}%)",
                    }
            elif unrealized_pnl_percent >= level2:  # Level 2 profit - take 50%
                take_profit_percent = take2
                adjusted_percent, force_close, reason = self.pm._adjust_partial_sale_for_max_limit(
                    position, take_profit_percent
                )
                if force_close:
                    return {
                        "action": "close_position",
                        "reason": reason or "Maximum limit reached during profit taking",
                    }
                if adjusted_percent > 0:
                    return {
                        "action": "partial_close",
                        "percent": adjusted_percent,
                        "reason": f"Profit taking at {level2 * 100:.1f}% gain ({adjusted_percent * 100:.0f}%)",
                    }
            elif unrealized_pnl_percent >= level1:  # Level 1 profit - take 25%
                take_profit_percent = take1
                adjusted_percent, force_close, reason = self.pm._adjust_partial_sale_for_max_limit(
                    position, take_profit_percent
                )
                if force_close:
                    return {
                        "action": "close_position",
                        "reason": reason or "Maximum limit reached during profit taking",
                    }
                if adjusted_percent > 0:
                    return {
                        "action": "partial_close",
                        "percent": adjusted_percent,
                        "reason": f"Profit taking at {level1 * 100:.1f}% gain ({adjusted_percent * 100:.0f}%)",
                    }

            # If trailing stop was updated but no partial take, return the trailing action
            if trailing_action:
                return trailing_action

        elif direction == "short":
            unrealized_pnl_usd = max(0.0, (entry_price - current_price) * position["quantity"])
            unrealized_pnl_percent = (unrealized_pnl_usd / notional_usd) if notional_usd else 0.0

            # FIRST: Always evaluate and update trailing stop when in profit
            # This ensures stop loss is tightened even when partial take happens
            trailing_action = self._evaluate_trailing_stop(
                position=position,
                current_price=current_price,
                profit_target=profit_target,
                direction=direction,
                entry_price=entry_price,
                unrealized_pnl_percent=unrealized_pnl_percent,
                profit_levels=profit_levels,
            )
            # If trailing stop updated, the new stop is already saved in position['exit_plan']
            # We don't return here - continue to check partial take

            # Dynamic Profit Taking Levels for shorts based on notional size
            if unrealized_pnl_percent >= level3:  # Level 3 profit - take 75%
                take_profit_percent = take3
                adjusted_percent, force_close, reason = self.pm._adjust_partial_sale_for_max_limit(
                    position, take_profit_percent
                )
                if force_close:
                    return {
                        "action": "close_position",
                        "reason": reason or "Maximum limit reached during profit taking",
                    }
                if adjusted_percent > 0:
                    return {
                        "action": "partial_close",
                        "percent": adjusted_percent,
                        "reason": f"Profit taking at {level3 * 100:.1f}% gain ({adjusted_percent * 100:.0f}%)",
                    }
            elif unrealized_pnl_percent >= level2:  # Level 2 profit - take 50%
                take_profit_percent = take2
                adjusted_percent, force_close, reason = self.pm._adjust_partial_sale_for_max_limit(
                    position, take_profit_percent
                )
                if force_close:
                    return {
                        "action": "close_position",
                        "reason": reason or "Maximum limit reached during profit taking",
                    }
                if adjusted_percent > 0:
                    return {
                        "action": "partial_close",
                        "percent": adjusted_percent,
                        "reason": f"Profit taking at {level2 * 100:.1f}% gain ({adjusted_percent * 100:.0f}%)",
                    }
            elif unrealized_pnl_percent >= level1:  # Level 1 profit - take 25%
                take_profit_percent = take1
                adjusted_percent, force_close, reason = self.pm._adjust_partial_sale_for_max_limit(
                    position, take_profit_percent
                )
                if force_close:
                    return {
                        "action": "close_position",
                        "reason": reason or "Maximum limit reached during profit taking",
                    }
                if adjusted_percent > 0:
                    return {
                        "action": "partial_close",
                        "percent": adjusted_percent,
                        "reason": f"Profit taking at {level1 * 100:.1f}% gain ({adjusted_percent * 100:.0f}%)",
                    }

            # If trailing stop was updated but no partial take, return the trailing action
            if trailing_action:
                return trailing_action

        return exit_decision

    def _evaluate_trailing_stop(
        self,
        position: dict[str, Any],
        current_price: float,
        profit_target: float | None,
        direction: str,
        entry_price: float,
        unrealized_pnl_percent: float,
        profit_levels: dict[str, float],
    ) -> dict[str, Any] | None:
        """Evaluate advanced trailing stop conditions based on progress, time, volume and ATR."""
        if (
            unrealized_pnl_percent <= 0
            or not isinstance(current_price, (int, float))
            or current_price <= 0
        ):
            return None

        symbol = position.get("symbol")
        exit_plan = position.get("exit_plan") or {}

        level1_threshold = 0.0
        if isinstance(profit_levels, dict):
            try:
                level1_threshold = float(profit_levels.get("level1", 0.0) or 0.0)
            except (TypeError, ValueError):
                level1_threshold = 0.0
        if unrealized_pnl_percent < max(level1_threshold * 0.5, 0.0):
            return None

        existing_stop = exit_plan.get("stop_loss")
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
        entry_time_str = position.get("entry_time")
        if entry_time_str:
            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                time_in_trade = max(0.0, (datetime.now() - entry_time).total_seconds() / 60.0)
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
        except Exception:
            pass  # Use default trigger

        progress_triggered = progress_score >= effective_progress_trigger
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
            indicators_3m = (
                self.pm.market_data.get_technical_indicators(symbol, "3m") if self.pm.market_data else {}
            )
        except Exception as exc:
            print(f"⚠️ Trailing stop indicator fetch failed for {symbol}: {exc}")
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
            atr_value = position.get("entry_atr_14")
        if not isinstance(atr_value, (int, float)) or atr_value <= 0:
            atr_value = current_price * Config.TRAILING_FALLBACK_BUFFER_PCT

        entry_volume_ratio = position.get("entry_volume_ratio")
        volume_drop_triggered = False
        if isinstance(current_volume_ratio, (int, float)):
            if current_volume_ratio <= Config.TRAILING_VOLUME_ABSOLUTE_THRESHOLD:
                volume_drop_triggered = True
            elif isinstance(entry_volume_ratio, (int, float)) and entry_volume_ratio > 0:
                if current_volume_ratio <= entry_volume_ratio * Config.TRAILING_VOLUME_DROP_RATIO:
                    volume_drop_triggered = True

        min_improvement_abs = max(
            current_price * Config.TRAILING_MIN_IMPROVEMENT_PCT,
            max(Config.MIN_EXIT_PLAN_OFFSET, 1e-7),
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
                rsi_htf = indicators_htf.get("rsi_14", 50)
                sparkline = indicators_htf.get("smart_sparkline", {})
                price_loc = (
                    sparkline.get("price_location", {}) if isinstance(sparkline, dict) else {}
                )
                zone = price_loc.get("zone", "MIDDLE")

                # Halve buffer in UPPER_10 + RSI > 70 condition
                if zone == "UPPER_10" and isinstance(rsi_htf, (int, float)) and rsi_htf > 70:
                    atr_buffer = atr_buffer * 0.5
                    overbought_protect_active = True
                    print(
                        f"[PROTECTION] OVERBOUGHT PROTECT: {symbol} zone={zone} RSI={rsi_htf:.1f} -> Buffer halved"
                    )
        except Exception:
            pass  # Silently continue without overbought protection

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
        if direction == "long" and new_stop >= current_price:
            return None
        if direction == "short" and new_stop <= current_price:
            return None

        # Persist updated stop and trailing metadata
        exit_plan["stop_loss"] = new_stop
        position["exit_plan"] = exit_plan

        trailing_meta = position.setdefault("trailing", {})
        trailing_meta.update(
            {
                "active": True,
                "last_update_cycle": getattr(self, "current_cycle_number", None),
                "last_reason": ", ".join(reason_tokens),
                "last_stop": new_stop,
                "progress_percent": round(progress_score, 2),
                "time_in_trade_min": round(time_in_trade, 2),
            }
        )
        if isinstance(current_volume_ratio, (int, float)):
            trailing_meta["last_volume_ratio"] = round(current_volume_ratio, 4)

        reason = f"Trailing stop tightened ({', '.join(reason_tokens)})"
        return {"action": "update_stop", "new_stop": new_stop, "reason": reason}

    def _execute_new_positions_only(
        self,
        decisions: dict,
        valid_prices: dict,
        cycle_number: int,
        indicator_cache: dict[str, dict[str, Any]] | None = None,
    ):
        """Execute only new position entries after AI close_position signal"""
        print("[INFO] Executing new positions only (after close_position signal)")

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
                    print(
                        f"⚠️ KADEMELİ POZİSYON LİMİTİ (Cycle {cycle_number}): Max {max_positions_for_cycle} positions allowed. Skipping {coin} entry."
                    )
                    continue
                current_positions += 1

                decisions_to_execute[coin] = trade

        if decisions_to_execute:
            self.pm.execute_decision(
                decisions_to_execute, valid_prices, indicator_cache=indicator_cache
            )

