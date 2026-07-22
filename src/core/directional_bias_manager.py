"""Directional bias management for TradeSeeker.

Manages directional bias metrics, cooldowns, and confidence adjustments
based on historical trade performance by direction.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from loguru import logger

from config.config import Config
from src.core import constants
from src.core.cooldown_manager import CooldownManager

if TYPE_CHECKING:
    from src.schemas.position import Position


class DirectionalBiasManager:
    """Manages directional bias metrics, cooldowns, and confidence adjustments."""

    def __init__(self, cooldown_manager: CooldownManager | None = None) -> None:
        """Initialize the DirectionalBiasManager.

        Args:
            cooldown_manager: Optional shared CooldownManager instance. If None, creates a new one.
        """
        self.directional_bias: dict[str, dict[str, Any]] = self._init_directional_bias()
        self.counter_trend_consecutive_losses: int = 0
        self.cooldown_manager = (
            cooldown_manager if cooldown_manager is not None else CooldownManager()
        )

    @property
    def directional_cooldowns(self) -> dict[str, int]:
        """Return directional cooldowns (delegated to CooldownManager)."""
        return self.cooldown_manager.directional_cooldowns

    @directional_cooldowns.setter
    def directional_cooldowns(self, value: dict[str, int]) -> None:
        """Set directional cooldowns (delegated to CooldownManager)."""
        self.cooldown_manager.directional_cooldowns = value

    @property
    def relaxed_countertrend_cycles(self) -> int:
        """Return relaxed counter-trend cycles (delegated to CooldownManager)."""
        return self.cooldown_manager.relaxed_countertrend_cycles

    @relaxed_countertrend_cycles.setter
    def relaxed_countertrend_cycles(self, value: int) -> None:
        """Set relaxed counter-trend cycles (delegated to CooldownManager)."""
        self.cooldown_manager.relaxed_countertrend_cycles = value

    @property
    def counter_trend_cooldown(self) -> int:
        """Return counter-trend cooldown (delegated to CooldownManager)."""
        return self.cooldown_manager.counter_trend_cooldown

    @counter_trend_cooldown.setter
    def counter_trend_cooldown(self, value: int) -> None:
        """Set counter-trend cooldown (delegated to CooldownManager)."""
        self.cooldown_manager.counter_trend_cooldown = value

    @property
    def coin_cooldowns(self) -> dict[str, int]:
        """Return coin cooldowns (delegated to CooldownManager)."""
        return self.cooldown_manager.coin_cooldowns

    @coin_cooldowns.setter
    def coin_cooldowns(self, value: dict[str, int]) -> None:
        """Set coin cooldowns (delegated to CooldownManager)."""
        self.cooldown_manager.coin_cooldowns = value

    def _init_directional_bias(self) -> dict[str, dict[str, Any]]:
        """Initialize directional bias metrics for long and short trade tracking.

        Returns:
            Dictionary containing initialized stats for both 'long' and 'short' directions.
        """
        return {
            "long": {
                "rolling": deque(maxlen=constants.ROLLING_BIAS_WINDOW),
                "net_pnl": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "consecutive_losses": 0,
                "consecutive_wins": 0,
                "caution_active": False,
                "caution_win_progress": 0,
                "loss_streak_loss_usd": 0.0,
            },
            "short": {
                "rolling": deque(maxlen=constants.ROLLING_BIAS_WINDOW),
                "net_pnl": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "consecutive_losses": 0,
                "consecutive_wins": 0,
                "caution_active": False,
                "caution_win_progress": 0,
                "loss_streak_loss_usd": 0.0,
            },
        }

    def reset(self) -> None:
        """Reset all directional bias state to initial values."""
        self.directional_bias = self._init_directional_bias()
        self.counter_trend_consecutive_losses = 0
        self.cooldown_manager.reset()

    def load_state(self, data: dict[str, Any]) -> None:
        """Load directional bias state from saved data.

        Args:
            data: Dictionary containing saved directional bias state.
        """
        bias_state = data.get("directional_bias")
        if bias_state:
            self.directional_bias = self._init_directional_bias()
            for side in ("long", "short"):
                stored = bias_state.get(side, {})
                stats = self.directional_bias[side]
                stats["rolling"].extend(stored.get("rolling", []))
                stats["net_pnl"] = stored.get("net_pnl", 0.0)
                stats["trades"] = stored.get("trades", 0)
                stats["wins"] = stored.get("wins", 0)
                stats["losses"] = stored.get("losses", 0)
                stats["consecutive_losses"] = stored.get("consecutive_losses", 0)
                stats["consecutive_wins"] = stored.get("consecutive_wins", 0)
                stats["caution_active"] = stored.get("caution_active", False)
                stats["caution_win_progress"] = stored.get("caution_win_progress", 0)
                stats["loss_streak_loss_usd"] = stored.get("loss_streak_loss_usd", 0.0)

        self.counter_trend_consecutive_losses = data.get("counter_trend_consecutive_losses", 0)
        self.cooldown_manager.load_state(data)

    def serialize(self) -> dict[str, dict[str, Any]]:
        """Serialize directional bias metrics for state saving.

        Returns:
            Dictionary containing bias stats simplified for JSON serialization.
        """
        serialized: dict[str, dict[str, Any]] = {}
        for side, stats in self.directional_bias.items():
            serialized[side] = {
                "rolling": list(stats["rolling"]),
                "net_pnl": stats["net_pnl"],
                "trades": stats["trades"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "consecutive_losses": stats["consecutive_losses"],
                "consecutive_wins": stats.get("consecutive_wins", 0),
                "caution_active": stats.get("caution_active", False),
                "caution_win_progress": stats.get("caution_win_progress", 0),
                "loss_streak_loss_usd": stats.get("loss_streak_loss_usd", 0.0),
            }
        return serialized

    def get_bias_state_dict(self) -> dict[str, Any]:
        """Get the full bias state as a dictionary for persistence.

        Returns:
            Dictionary with directional_bias, cooldowns, and counter-trend state.
        """
        return {
            "directional_bias": self.serialize(),
            "counter_trend_consecutive_losses": self.counter_trend_consecutive_losses,
            **self.cooldown_manager.get_state_dict(),
        }

    def update_directional_bias(
        self,
        trade: dict[str, Any],
        counter_trend_consecutive_losses: int | None = None,
    ) -> int | None:
        """Update directional bias statistics based on a completed trade's outcome.

        Args:
            trade: Completed trade record containing direction, pnl, symbol, and trend_alignment.
            counter_trend_consecutive_losses: Optional reference to external counter-trend losses.

        Returns:
            Updated counter_trend_consecutive_losses if provided, else None.
        """
        direction = trade.get("direction")
        if direction not in ("long", "short"):
            return counter_trend_consecutive_losses

        stats = self.directional_bias[direction]
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        logger.info("update_directional_bias called: {} trade, PnL=${:.2f}", direction.upper(), pnl)
        stats["rolling"].append(pnl)
        stats["net_pnl"] += pnl
        stats["trades"] += 1

        if pnl > 0:
            self._handle_win_bias(direction, stats, pnl, trade)
        elif pnl < 0:
            counter_trend_consecutive_losses = self._handle_loss_bias(
                direction, stats, pnl, trade, counter_trend_consecutive_losses
            )
        else:
            # pnl == 0 case (breakeven)
            stats["consecutive_losses"] = 0
            stats["consecutive_wins"] = 0
            stats["caution_win_progress"] = 0
            stats["loss_streak_loss_usd"] = 0.0
            if (
                trade.get("trend_alignment") == "counter_trend"
                and counter_trend_consecutive_losses is not None
            ):
                counter_trend_consecutive_losses = 0

        return counter_trend_consecutive_losses

    def _handle_win_bias(
        self, direction: str, stats: dict[str, Any], pnl: float, trade: dict[str, Any]
    ) -> None:
        """Process directional bias updates for a winning trade outcome.

        Args:
            direction: The trade direction ('long' or 'short').
            stats: Dictionary of directional bias statistics to update.
            pnl: The profit/loss amount in USD.
            trade: The completed trade record.
        """
        stats["wins"] += 1
        stats["consecutive_losses"] = 0
        stats["consecutive_wins"] = stats.get("consecutive_wins", 0) + 1

        # Win Streak Cooldown
        if stats["consecutive_wins"] >= Config.WIN_STREAK_COOLDOWN_THRESHOLD:
            self.cooldown_manager.activate_directional_cooldown(
                direction, Config.WIN_STREAK_COOLDOWN_CYCLES
            )
            logger.warning(
                "Win streak cooldown for {}: {} wins", direction.upper(), stats["consecutive_wins"]
            )

        # Smart Cooldown (WIN)
        coin_symbol = trade.get("symbol", "").upper()
        if coin_symbol:
            self.cooldown_manager.activate_coin_cooldown(coin_symbol, Config.SMART_COOLDOWN_WIN)

        if stats.get("caution_active"):
            stats["caution_win_progress"] = stats.get("caution_win_progress", 0) + 1
            if stats["caution_win_progress"] >= constants.CAUTION_WIN_PROGRESS_THRESHOLD:
                stats["caution_active"] = False
                stats["caution_win_progress"] = 0

    def _handle_loss_bias(
        self,
        direction: str,
        stats: dict[str, Any],
        pnl: float,
        trade: dict[str, Any],
        counter_trend_consecutive_losses: int | None = None,
    ) -> int | None:
        """Process directional bias updates for a losing trade outcome.

        Args:
            direction: The trade direction ('long' or 'short').
            stats: Dictionary of directional bias statistics to update.
            pnl: The profit/loss amount in USD (negative for losses).
            trade: The completed trade record.
            counter_trend_consecutive_losses: Optional reference to external counter-trend losses.

        Returns:
            Updated counter_trend_consecutive_losses if provided, else None.
        """
        stats["losses"] += 1
        stats["consecutive_losses"] += 1
        stats["consecutive_wins"] = 0
        stats["caution_win_progress"] = 0

        # Track loss streak USD
        current_loss_streak = stats.get("loss_streak_loss_usd", 0.0)
        new_streak = current_loss_streak + abs(pnl)
        stats["loss_streak_loss_usd"] = min(new_streak, constants.LOSS_STREAK_MAX_CAP_USD)

        # Smart Cooldown (LOSS)
        coin_symbol = trade.get("symbol", "").upper()
        if coin_symbol:
            self.cooldown_manager.activate_coin_cooldown(coin_symbol, Config.SMART_COOLDOWN_LOSS)

        if stats["consecutive_losses"] >= constants.REVERSAL_SCORE_WEAK:
            stats["caution_active"] = True
            stats["caution_win_progress"] = 0

        # Cooldown: consecutive losses OR total loss -> default cooldown
        loss_streak_usd = stats.get("loss_streak_loss_usd", 0.0)
        consecutive = stats["consecutive_losses"]
        should_activate = (
            consecutive >= constants.REVERSAL_SCORE_STRONG
            or loss_streak_usd >= constants.LOSS_STREAK_USD_THRESHOLD
        )

        if should_activate:
            self.cooldown_manager.activate_directional_cooldown(
                direction, constants.DEFAULT_COOLDOWN_CYCLES
            )
            logger.warning(
                "Directional cooldown ACTIVATED for {}: consecutive_losses={}, loss_streak_usd=${:.2f}",
                direction.upper(),
                consecutive,
                loss_streak_usd,
            )

        # Counter-trend cooldown: consecutive counter-trend losses
        if trade.get("trend_alignment") == "counter_trend":
            if counter_trend_consecutive_losses is not None:
                counter_trend_consecutive_losses += 1
                if counter_trend_consecutive_losses >= constants.COUNTER_TREND_LOSS_THRESHOLD:
                    self.cooldown_manager.counter_trend_cooldown = constants.DEFAULT_COOLDOWN_CYCLES
                    counter_trend_consecutive_losses = 0

        return counter_trend_consecutive_losses

    def _activate_directional_cooldown(self, direction: str, cycles: int = 3) -> None:
        """Activate a cooldown for a specific trade direction.

        Args:
            direction: The trade direction ('long' or 'short') to cool down.
            cycles: Number of cycles the cooldown should remain active.
        """
        self.cooldown_manager.activate_directional_cooldown(direction, cycles)

    def tick_cooldowns(self) -> None:
        """Decrement all active cooldown timers by one cycle."""
        self.cooldown_manager.tick_cooldowns()

    def _tick_directional_cooldowns(self) -> None:
        """Decrement directional cooldown counters and reset loss streaks on expiry."""
        self.cooldown_manager._tick_directional_cooldowns()
        for direction in ("long", "short"):
            if (
                self.directional_cooldowns.get(direction, 0) == 0
                and direction in self.directional_bias
            ):
                self.directional_bias[direction]["loss_streak_loss_usd"] = 0.0
                self.directional_bias[direction]["consecutive_losses"] = 0
                logger.info(
                    "Directional cooldown cleared for {} trades. Loss streak reset.",
                    direction.upper(),
                )

    def _tick_coin_cooldowns(self) -> None:
        """Decrement per-coin cooldown counters and remove expired entries."""
        self.cooldown_manager._tick_coin_cooldowns()

    def count_positions_by_direction(self, positions: dict[str, Position]) -> dict[str, int]:
        """Count currently open positions categorized by trade direction.

        Args:
            positions: Dictionary of current positions.

        Returns:
            Dictionary containing counts for 'long' and 'short' positions.
        """
        counts = {"long": 0, "short": 0}
        for pos in positions.values():
            direction = pos.direction
            if direction in counts:
                counts[direction] += 1
        return counts

    def get_directional_bias_metrics(self) -> dict[str, dict[str, Any]]:
        """Calculate and retrieve holistic directional bias metrics.

        Returns:
            Dictionary with 'long' and 'short' performance metrics.
        """
        metrics = {}
        for side, stats in self.directional_bias.items():
            rolling_list = list(stats["rolling"])
            rolling_sum = sum(rolling_list)
            # FIX: Zero division protection
            rolling_avg = (
                (rolling_sum / len(rolling_list)) if rolling_list and len(rolling_list) > 0 else 0.0
            )

            # Calculate profitability index based on profit/loss amounts (not trade counts)
            total_profit = sum(pnl for pnl in rolling_list if pnl > 0)
            total_loss = abs(sum(pnl for pnl in rolling_list if pnl < 0))

            if total_profit + total_loss > 0:
                profitability_index = (total_profit / (total_profit + total_loss)) * 100
            else:
                profitability_index = 0.0

            metrics[side] = {
                "net_pnl": stats["net_pnl"],
                "trades": stats["trades"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "profitability_index": profitability_index,
                "rolling_sum": rolling_sum,
                "rolling_avg": rolling_avg,
                "consecutive_losses": stats["consecutive_losses"],
                "consecutive_wins": stats.get("consecutive_wins", 0),
                "caution_active": stats.get("caution_active", False),
                "caution_win_progress": stats.get("caution_win_progress", 0),
            }
        return metrics

    def apply_directional_bias(
        self,
        bias_ctx: dict[str, Any],
    ) -> float:
        """Apply directional bias metrics to adjust confidence.

        Args:
            bias_ctx: Context containing 'signal', 'confidence', 'bias_metrics',
                and 'current_trend'.

        Returns:
            The adjusted confidence value.
        """
        signal = bias_ctx["signal"]
        confidence = bias_ctx["confidence"]
        bias_metrics = bias_ctx["bias_metrics"]
        current_trend = bias_ctx["current_trend"]

        side = "long" if signal == "buy_to_enter" else "short"
        stats = bias_metrics.get(side, {})
        if not stats:
            return confidence

        original_confidence = confidence

        # 1. Base Penalty (Caution & Rolling Avg)
        confidence = self._apply_bias_penalty(confidence, stats)

        # 2. Trend & Side Multipliers
        confidence = self._apply_trend_bias_multipliers(confidence, side, current_trend)

        # 3. ML Consensus Boost/Penalty
        ml_consensus = bias_ctx.get("ml_consensus")
        if ml_consensus:
            confidence = self._apply_ml_consensus_boost(confidence, side, ml_consensus)

        # Minimum confidence floor
        min_floor = original_confidence * constants.CONFIDENCE_FLOOR_MULTIPLIER
        return max(confidence, min_floor, Config.MIN_CONFIDENCE)

    def _apply_bias_penalty(self, confidence: float, stats: dict[str, Any]) -> float:
        """Apply caution and negative rolling average penalties.

        Args:
            confidence: Base confidence value.
            stats: Dictionary of directional bias statistics.

        Returns:
            The penalized confidence value.
        """
        if stats.get("caution_active"):
            confidence = max(
                confidence * constants.CAUTION_PENALTY_MULTIPLIER,
                confidence - constants.CAUTION_ABSOLUTE_REDUCTION,
            )

        rolling_avg = stats.get("rolling_avg", 0.0)
        if rolling_avg < 0:
            confidence = max(
                confidence * constants.NEGATIVE_ROLLING_AVG_PENALTY_MULTIPLIER,
                confidence - constants.NEGATIVE_ROLLING_AVG_REDUCTION,
            )
        return confidence

    def _apply_trend_bias_multipliers(
        self, confidence: float, side: str, current_trend: str
    ) -> float:
        """Apply multipliers based on trend alignment and signal side.

        Args:
            confidence: Base confidence value.
            side: The trade side ('long' or 'short').
            current_trend: The detected high-level trend direction.

        Returns:
            The trend-adjusted confidence value.
        """
        trend_lower = current_trend.lower() if isinstance(current_trend, str) else "unknown"

        # 1. Trend Following (TF) Strength Selection
        strength_multiplier = 1.0
        if "tf_strong" in trend_lower:
            strength_multiplier = constants.CONFIDENCE_TF_STRONG_BOOST
        elif "tf_stable" in trend_lower:
            strength_multiplier = 1.00
        elif "tf_weak" in trend_lower:
            strength_multiplier = constants.CONFIDENCE_TF_WEAK_PENALTY
        elif "choppy" in trend_lower:
            strength_multiplier = constants.CONFIDENCE_CHOPPY_PENALTY

        # 2. Directional Alignment Multipliers
        if "neutral" in trend_lower:
            confidence *= Config.DIRECTIONAL_NEUTRAL_MULTIPLIER
        elif "bullish" in trend_lower:
            if side == "long":
                confidence *= Config.DIRECTIONAL_BULLISH_LONG_MULTIPLIER * strength_multiplier
            elif side == "short":
                confidence *= Config.DIRECTIONAL_BULLISH_SHORT_MULTIPLIER * (
                    1 / strength_multiplier
                )
        elif "bearish" in trend_lower:
            if side == "long":
                confidence *= Config.DIRECTIONAL_BEARISH_LONG_MULTIPLIER * (1 / strength_multiplier)
            elif side == "short":
                confidence *= Config.DIRECTIONAL_BEARISH_SHORT_MULTIPLIER * strength_multiplier

        return confidence

    def _apply_ml_consensus_boost(
        self, confidence: float, side: str, ml_consensus: dict[str, float]
    ) -> float:
        """Apply confidence multipliers based on ML prediction consensus.

        Args:
            confidence: Current confidence level.
            side: The trade side ('long' or 'short').
            ml_consensus: Dictionary with 'BUY', 'SELL', 'HOLD' probability values.

        Returns:
            The adjusted confidence value after ML consensus multipliers.
        """
        ml_buy = ml_consensus.get("BUY", 0.0)
        ml_sell = ml_consensus.get("SELL", 0.0)
        ml_hold = ml_consensus.get("HOLD", 0.0)

        # 1. HOLD Priority: If ML is indecisive (>50% HOLD), apply penalty regardless of side
        if ml_hold > 0.50:
            confidence *= constants.CONFIDENCE_ML_HOLD_PENALTY
            return confidence

        # 2. Directional Alignment
        if ml_buy > 0.50:
            if side == "long":
                confidence *= constants.CONFIDENCE_ML_SUPPORTS_BOOST
            else:
                confidence *= constants.CONFIDENCE_ML_CONTRADICTS_PENALTY
        elif ml_sell > 0.50:
            if side == "short":
                confidence *= constants.CONFIDENCE_ML_SUPPORTS_BOOST
            else:
                confidence *= constants.CONFIDENCE_ML_CONTRADICTS_PENALTY

        return confidence
