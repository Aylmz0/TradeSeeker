"""Cooldown management for TradeSeeker.

Handles directional cooldowns, coin-specific cooldowns, counter-trend cooldowns,
and the tick logic that decrements all active cooldown timers each cycle.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.core import constants


class CooldownManager:
    """Manages all cooldown state and tick logic for the portfolio."""

    def __init__(self) -> None:
        """Initialize the CooldownManager."""
        self.directional_cooldowns: dict[str, int] = {"long": 0, "short": 0}
        self.coin_cooldowns: dict[str, int] = {}
        self.counter_trend_cooldown: int = 0
        self.relaxed_countertrend_cycles: int = 0

    def reset(self) -> None:
        """Reset all cooldown state to initial values."""
        self.directional_cooldowns = {"long": 0, "short": 0}
        self.coin_cooldowns = {}
        self.counter_trend_cooldown = 0
        self.relaxed_countertrend_cycles = 0

    def load_state(self, data: dict[str, Any]) -> None:
        """Load cooldown state from saved data.

        Args:
            data: Dictionary containing saved cooldown state.
        """
        self.directional_cooldowns = data.get("directional_cooldowns", {"long": 0, "short": 0})
        self.relaxed_countertrend_cycles = data.get("relaxed_countertrend_cycles", 0)
        self.counter_trend_cooldown = data.get("counter_trend_cooldown", 0)
        self.coin_cooldowns = data.get("coin_cooldowns", {})

    def get_state_dict(self) -> dict[str, Any]:
        """Get cooldown state as a dictionary for persistence.

        Returns:
            Dictionary with directional cooldowns, coin cooldowns, counter-trend cooldown,
            and relaxed cycle count.
        """
        return {
            "directional_cooldowns": self.directional_cooldowns,
            "relaxed_countertrend_cycles": self.relaxed_countertrend_cycles,
            "counter_trend_cooldown": self.counter_trend_cooldown,
            "coin_cooldowns": self.coin_cooldowns,
        }

    def activate_directional_cooldown(self, direction: str, cycles: int = 3) -> None:
        """Activate a cooldown for a specific trade direction.

        Args:
            direction: The trade direction ('long' or 'short') to cool down.
            cycles: Number of cycles the cooldown should remain active.
        """
        if direction not in ("long", "short"):
            return
        current = self.directional_cooldowns.get(direction, 0)
        new_cooldown = max(current, cycles)
        self.directional_cooldowns[direction] = min(
            new_cooldown, constants.MAX_DIRECTIONAL_COOLDOWN
        )
        self.relaxed_countertrend_cycles = max(
            self.relaxed_countertrend_cycles,
            constants.MIN_RELAXED_CYCLES,
        )
        logger.warning(
            "Directional cooldown activated for {} trades ({} cycles). Counter-trend restrictions relaxed for {} cycles.",
            direction.upper(),
            constants.MIN_RELAXED_CYCLES,
            constants.MIN_RELAXED_CYCLES,
        )

    def activate_coin_cooldown(self, coin_symbol: str, cycles: int) -> None:
        """Activate a cooldown for a specific coin.

        Args:
            coin_symbol: The coin symbol (e.g., 'BTCUSDT') to cool down.
            cycles: Number of cycles the cooldown should remain active.
        """
        if not coin_symbol:
            return
        self.coin_cooldowns[coin_symbol] = cycles
        logger.warning(
            "Coin cooldown ACTIVATED for {}: {} cycles",
            coin_symbol,
            cycles,
        )

    def tick_cooldowns(self) -> None:
        """Decrement all active cooldown timers by one cycle."""
        logger.debug(
            "tick_cooldowns called. Current cooldowns: {}, Coin cooldowns: {}",
            self.directional_cooldowns,
            self.coin_cooldowns,
        )
        self._tick_directional_cooldowns()
        self._tick_coin_cooldowns()

        if self.relaxed_countertrend_cycles > 0:
            self.relaxed_countertrend_cycles -= 1
            if self.relaxed_countertrend_cycles == 0:
                logger.info("Relaxed counter-trend validation mode EXPIRED.")

    def _tick_directional_cooldowns(self) -> None:
        """Decrement directional cooldown counters and reset loss streaks on expiry."""
        for direction in ("long", "short"):
            cycles = self.directional_cooldowns.get(direction, 0)
            if cycles > 0:
                self.directional_cooldowns[direction] = cycles - 1
                logger.debug(
                    "{} cooldown: {} -> {} cycles remaining",
                    direction.upper(),
                    cycles,
                    self.directional_cooldowns[direction],
                )
                if self.directional_cooldowns[direction] == 0:
                    logger.info(
                        "Directional cooldown cleared for {} trades.",
                        direction.upper(),
                    )

    def _tick_coin_cooldowns(self) -> None:
        """Decrement per-coin cooldown counters and remove expired entries."""
        to_remove = []
        for coin, cycles in self.coin_cooldowns.items():
            if cycles > 0:
                self.coin_cooldowns[coin] = cycles - 1
                logger.debug(
                    "{} coin cooldown: {} -> {} cycles remaining",
                    coin,
                    cycles,
                    self.coin_cooldowns[coin],
                )
                if self.coin_cooldowns[coin] == 0:
                    to_remove.append(coin)
                    logger.info("Coin cooldown cleared for {}.", coin)

        for coin in to_remove:
            del self.coin_cooldowns[coin]
        if self.counter_trend_cooldown > 0:
            self.counter_trend_cooldown -= 1
            if self.counter_trend_cooldown == 0:
                logger.info("Counter-trend cooldown cleared.")
