"""Trade execution engine for TradeSeeker.

Extracted from PortfolioManager to handle order execution, position sizing,
and margin calculations independently.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from config.config import Config
from src.core import constants
from src.schemas.position import Position

if TYPE_CHECKING:
    from src.core.market_data import RealMarketData


class TradeExecutor:
    """Handles order execution (live and simulation), position sizing, and margin calculations."""

    def __init__(
        self,
        order_executor: Any,
        is_live_trading: bool,
    ) -> None:
        """Initialize the TradeExecutor.

        Args:
            order_executor: The order executor instance for live trading.
            is_live_trading: Whether live trading is enabled.
        """
        self.order_executor = order_executor
        self.is_live_trading = is_live_trading

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
    ) -> dict[str, Any]:
        """Execute a live entry order via the order executor.

        Args:
            coin: The cryptocurrency symbol.
            direction: Trade direction ('long' or 'short').
            quantity: Number of coins to trade.
            leverage: Leverage multiplier.
            current_price: Current market price for order reference.
            notional_usd: Total notional value in USD.
            confidence: Trade confidence level.
            margin_usd: Optional margin amount in USD.

        Returns:
            Dictionary with 'success' status, order details, executed quantity, and average price.
        """
        if not self.is_live_trading or not self.order_executor:
            return {"success": False, "error": "live_trading_disabled"}
        try:
            order = self.order_executor.place_smart_limit_order(
                coin=coin,
                direction=direction,
                quantity=quantity,
                leverage=leverage,
                price_reference=current_price,
                reduce_only=False,
                timeout_seconds=5.0,
            )
            executed_qty = float(order.get("executedQty", 0.0))
            avg_price = float(order.get("avgPriceComputed", order.get("avgPrice", 0.0)))
            return {
                "success": True,
                "order": order,
                "executed_qty": executed_qty,
                "avg_price": avg_price,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def calculate_confidence_based_margin(
        self,
        margin_ctx: dict[str, Any],
        market_data: RealMarketData | None = None,
    ) -> float:
        """Calculate margin based on volatility sizing (fixed risk) and confidence.

        Args:
            margin_ctx: Context containing confidence, balance, and price data.
            market_data: Optional market data instance for coin lookup.

        Returns:
            The calculated target margin in USD.
        """
        confidence = margin_ctx["confidence"]
        available_cash = margin_ctx["balance"]
        entry_price = margin_ctx.get("entry_price")
        stop_loss = margin_ctx.get("stop_loss")
        leverage = margin_ctx.get("leverage", 10)
        log_func = margin_ctx.get("log_func")
        # 1. Default fallback (Old method) if data is missing
        if entry_price is None or stop_loss is None or entry_price <= 0:
            margin = available_cash * constants.MAX_MARGIN_CASH_RATIO * confidence
            margin = max(margin, Config.MIN_POSITION_MARGIN_USD)
            logger.info(
                "Standard Sizing (Missing Data): ${:.2f} (conf: {:.2f})", margin, confidence
            )
            return margin

        # 2. Calculate Stop Distance %
        dist_pct = abs(entry_price - stop_loss) / entry_price

        # Safety: Avoid division by zero or extremely small stops (<0.2%)
        dist_pct = max(
            dist_pct, constants.MIN_STOP_DISTANCE_PCT
        )  # Min 0.2% stop distance assumption

        # 3. Calculate Base Position Size (Notional) for Fixed Risk
        # Risk = Notional * Dist_Pct  =>  Notional = Risk / Dist_Pct
        risk_amount = Config.RISK_PER_TRADE_USD
        base_notional = risk_amount / dist_pct

        # 4. Apply Confidence Scaling
        target_notional = base_notional * confidence

        # 5. Convert to Margin
        target_margin = target_notional / leverage

        # 6. Apply Limits
        # Cap margin at 40% of available cash (Safety ceiling)
        max_margin_cash = available_cash * constants.MAX_MARGIN_CASH_RATIO
        if target_margin > max_margin_cash:
            coin_label = "unknown"
            if (
                market_data
                and hasattr(market_data, "available_coins")
                and market_data.available_coins
            ):
                coin_label = market_data.available_coins[0]
            if log_func:
                log_func(
                    "sizing",
                    f"[WARN]  Volatility sizing capped by cash limit: ${target_margin:.2f} -> ${max_margin_cash:.2f}",
                    {
                        "coin": coin_label,
                        "target_margin": target_margin,
                        "max_margin": max_margin_cash,
                    },
                )
            else:
                logger.warning(
                    "Volatility sizing capped by cash limit: ${:.2f} -> ${:.2f}",
                    target_margin,
                    max_margin_cash,
                )
            target_margin = max_margin_cash

        # Note: Scout Sizing multipliers are now handled centrally in the main loop
        # via Config.MARKET_REGIME_MULTIPLIERS to avoid double multiplication.

        # Apply minimum margin ($10)
        target_margin = max(target_margin, Config.MIN_POSITION_MARGIN_USD)

        msg = f"[INFO]  Volatility Sizing: Risk ${risk_amount} | Stop {dist_pct * 100:.2f}% | Base Notional ${base_notional:.1f} | Conf {confidence:.2f} -> Margin ${target_margin:.2f}"
        if log_func:
            log_func("sizing", msg)
        else:
            logger.info("{}", msg)
        return target_margin

    def _estimate_liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        direction: str,
    ) -> float:
        """Estimate the liquidation price for a position.

        Args:
            entry_price: Price at which position was entered.
            leverage: Leverage used for the position.
            direction: Trade direction ('long' or 'short').

        Returns:
            The estimated liquidation price.
        """
        if leverage <= 1 or entry_price <= 0:
            return 0.0
        imr = 1.0 / leverage
        mmr = constants.MAINTENANCE_MARGIN_RATE
        margin_diff = imr - mmr
        if margin_diff <= 0:
            logger.warning("Liq est. failed: margin diff <= 0 ({}).", margin_diff)
            return 0.0
        liq_price = (
            entry_price * (1 - margin_diff)
            if direction == "long"
            else entry_price * (1 + margin_diff)
        )
        return max(0.0, liq_price)

    def _execute_order_payload(
        self,
        signal_ctx: dict[str, Any],
        calculated_margin: float,
        atr_stop_loss: float,
        current_balance: float,
        lock: threading.RLock,
        current_cycle_number: int,
        market_data: RealMarketData | None = None,
    ) -> bool | Position:
        """Execute the final order payload using live or simulated execution.

        Args:
            signal_ctx: Context for the trade signal.
            calculated_margin: The final margin to use.
            atr_stop_loss: The stop loss price.
            current_balance: Current portfolio balance for margin checks.
            lock: Thread lock for atomic position updates.
            current_cycle_number: The current cycle number.
            market_data: Optional market data instance for coin lookup.

        Returns:
            True for live execution success, Position for simulated execution, or False on failure.
        """
        from src.utils import format_num

        coin = signal_ctx["coin"]
        signal = signal_ctx["signal"]
        current_price = signal_ctx["current_price"]
        leverage = signal_ctx["leverage"]
        confidence = signal_ctx["confidence"]
        trade = signal_ctx["trade"]
        execution_report = signal_ctx["report"]
        direction = signal_ctx["direction"]
        classification = signal_ctx.get("classification", "unknown")
        indicators_3m = signal_ctx.get("indicators_3m", {})

        notional_usd = calculated_margin * leverage
        margin_usd = notional_usd / leverage
        if margin_usd <= 0:
            logger.warning(
                "{} {}: Calculated margin is zero/negative. Skipping.", signal.upper(), coin
            )
            return False
        if margin_usd > current_balance:
            logger.warning(
                "{} {}: Not enough cash for margin (${:.2f} > ${:.2f})",
                signal.upper(),
                coin,
                margin_usd,
                current_balance,
            )
            return False

        quantity_coin = notional_usd / current_price

        # Fixed percentage-based profit target (avoids N/A in UI)
        tp_pct = getattr(Config, "DEFAULT_PROFIT_TARGET_PCT", 0.015)
        default_profit_target = (
            current_price * (1 + tp_pct) if direction == "long" else current_price * (1 - tp_pct)
        )

        # Live or Simulated
        if self.is_live_trading:
            live_result = self.execute_live_entry(
                coin=coin,
                direction=direction,
                quantity=quantity_coin,
                leverage=leverage,
                current_price=current_price,
                notional_usd=notional_usd,
                confidence=confidence,
                margin_usd=margin_usd,
            )
            if not live_result.get("success"):
                err_msg = live_result.get("error", "unknown_error")
                logger.warning("LIVE ORDER FAILED: {} {} ({})", coin, signal, err_msg)
                execution_report["blocked"].append(
                    {"coin": coin, "reason": "live_order_failed", "error": err_msg},
                )
                trade["runtime_decision"] = "blocked_live_order"
                return False

            execution_report["executed"].append(
                {
                    "coin": coin,
                    "signal": signal,
                    "confidence": confidence,
                    "classification": classification,
                    "margin_usd": live_result.get("margin_usd"),
                    "mode": "live",
                    "order_id": live_result.get("order", {}).get("orderId"),
                },
            )
            trade["runtime_decision"] = "executed_live"
            return True

        # Simulated — atomically create position under lock
        with lock:
            est_liq = self._estimate_liquidation_price(current_price, leverage, direction)
            position_data = {
                "symbol": coin,
                "direction": direction,
                "quantity": quantity_coin,
                "entry_price": current_price,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "current_price": current_price,
                "unrealized_pnl": 0.0,
                "notional_usd": notional_usd,
                "margin_usd": margin_usd,
                "leverage": leverage,
                "liquidation_price": est_liq,
                "confidence": confidence,
                "exit_plan": {
                    "profit_target": default_profit_target,
                    "stop_loss": atr_stop_loss,
                    "invalidation_condition": trade.get("invalidation_condition"),
                },
                "risk_usd": notional_usd * (abs(current_price - atr_stop_loss) / current_price)
                if current_price
                else 0,
                "loss_cycle_count": 0,
                "entry_volume": indicators_3m.get("volume"),
                "entry_avg_volume": indicators_3m.get("avg_volume"),
                "entry_volume_ratio": indicators_3m.get("volume_ratio"),
                "entry_atr_14": indicators_3m.get("atr_14"),
                "trend_alignment": classification,
                "trend_context": {
                    "trend_at_entry": trade.get("trend_runtime", "unknown"),
                    "alignment": classification,
                    "cycle": current_cycle_number,
                },
                "trailing": {},
                "sl_oid": -1,
                "tp_oid": -1,
                "entry_oid": -1,
                "wait_for_fill": False,
            }
            position = Position.model_validate(position_data)
        logger.info(
            "{}: Opened {} {} ({} @ ${} / Notional ${} / Margin ${})",
            signal.upper(),
            direction,
            coin,
            format_num(quantity_coin, 4),
            format_num(current_price, 4),
            format_num(notional_usd, 2),
            format_num(margin_usd, 2),
        )
        execution_report["executed"].append(
            {
                "coin": coin,
                "signal": signal,
                "confidence": confidence,
                "classification": classification,
                "volume_ratio": indicators_3m.get("volume_ratio")
                if isinstance(indicators_3m, dict)
                else None,
                "margin_usd": margin_usd,
            },
        )
        trade["runtime_decision"] = "executed"
        return position
