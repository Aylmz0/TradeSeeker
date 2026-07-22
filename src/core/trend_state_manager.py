"""Trend state tracking, flip guards, and intraday trend logic for TradeSeeker.

Extracted from PortfolioManager to encapsulate all trend-related state management
including per-coin trend tracking, trend flip detection, and flip guard enforcement.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from config.config import Config
from src.core import constants
from src.core.indicators import determine_trend


class TrendStateManager:
    """Manages trend state tracking, flip guards, and intraday trend logic.

    This class is self-contained and does not depend on PortfolioManager.
    It tracks per-coin trend state internally and enforces flip guard rules.
    """

    def __init__(self) -> None:
        """Initialize the TrendStateManager with default settings."""
        self.trend_state: dict[str, dict[str, Any]] = {}
        self.trend_flip_cooldown: int = constants.TREND_FLIP_COOLDOWN_DEFAULT
        self.trend_flip_history_window: int = constants.TREND_FLIP_HISTORY_WINDOW
        self.current_cycle_number: int = 0

    def reset(self) -> None:
        """Reset all trend state."""
        self.trend_state = {}

    def set_cycle(self, cycle_number: int) -> None:
        """Update the current cycle number for flip tracking.

        Args:
            cycle_number: The current cycle sequence number.
        """
        self.current_cycle_number = cycle_number

    # --- Trend Direction ---

    def calculate_trend_direction(self, price: float, ema20: float) -> str:
        """Determine trend direction relative to EMA with a neutral band.

        Args:
            price: Current asset price.
            ema20: The EMA 20 value.

        Returns:
            String representing the trend direction ('BULLISH', 'BEARISH', or 'NEUTRAL').
        """
        return determine_trend(price, ema20)

    # --- Intraday Trend ---

    def calculate_trend_with_intraday(
        self,
        price_htf: float,
        ema20_htf: float,
        current_trend: str,
        indicators_15m: dict[str, Any] | None,
    ) -> str:
        """Apply intraday RSI and EMA filters to the high-level trend.

        Uses 15m RSI (primary for 1h target) instead of 3m to avoid noise.

        Args:
            price_htf: The HTF current price.
            ema20_htf: The HTF EMA 20 value.
            current_trend: The initial detected trend.
            indicators_15m: Technical indicators for the 15m timeframe.

        Returns:
            The adjusted trend direction as a string.
        """
        if not (
            indicators_15m and isinstance(indicators_15m, dict) and "error" not in indicators_15m
        ):
            return current_trend

        p15m = indicators_15m.get("current_price")
        e15m = indicators_15m.get("ema_20", p15m)
        r15m = indicators_15m.get("rsi_13", 50)

        if not all(isinstance(x, (int, float)) for x in [p15m, e15m, r15m]):
            return current_trend

        # Neutralization logic (15m)
        intra_bull = p15m >= e15m
        if (
            current_trend == "bearish" and intra_bull and r15m >= Config.INTRADAY_NEUTRAL_RSI_HIGH
        ) or (
            current_trend == "bullish"
            and not intra_bull
            and r15m <= Config.INTRADAY_NEUTRAL_RSI_LOW
        ):
            current_trend = "neutral"

        # Strong trend recovery logic (15m)
        if current_trend == "neutral":
            if price_htf <= ema20_htf and p15m <= e15m and r15m <= Config.TREND_SHORT_RSI_THRESHOLD:
                current_trend = "bearish"
            elif (
                price_htf >= ema20_htf and p15m >= e15m and r15m >= Config.TREND_LONG_RSI_THRESHOLD
            ):
                current_trend = "bullish"

        return current_trend

    # --- Counter-Trend Alignment ---

    def evaluate_counter_trend_alignment(
        self,
        signal: str,
        sig_dir: str,
        t_htf: str,
        t_15m: str | None,
        t_3m: str,
    ) -> bool:
        """Evaluate if a counter-trend signal has sufficient timeframe alignment.

        Args:
            signal: The original signal string.
            sig_dir: The normalized signal direction.
            t_htf: The HTF trend direction.
            t_15m: The 15m trend direction (if available).
            t_3m: The 3m trend direction.

        Returns:
            True if alignment is sufficient for entry, False otherwise.
        """
        is_ct = False
        if t_htf == "NEUTRAL":
            is_ct = False
        elif (signal == "buy_to_enter" and t_htf == "BEARISH") or (
            signal == "sell_to_enter" and t_htf == "BULLISH"
        ):
            is_ct = True

        if is_ct:
            # Counter-trend STRONG: 15m + 3m both align with signal direction (against 1h)
            if t_15m and t_15m == t_3m == sig_dir:
                return True
            # For counter-trend, we return True if it's CT at all (as per original logic)
            return True

        return False

    # --- Trend State Update ---

    def update_trend_state(
        self,
        coin: str,
        indicators_htf: dict[str, Any],
        indicators_15m: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Calculate and update the trend state for a given coin.

        Args:
            coin: The cryptocurrency symbol.
            indicators_htf: HTF technical indicators.
            indicators_15m: Optional 15m technical indicators for intraday adjustment.

        Returns:
            Dictionary containing the updated trend record.
        """
        price_htf = indicators_htf.get("current_price")
        ema20_htf = indicators_htf.get("ema_20")

        if (
            not isinstance(price_htf, (int, float))
            or not isinstance(ema20_htf, (int, float))
            or ema20_htf == 0
        ):
            return {"trend": "unknown", "recent_flip": False, "last_flip_cycle": None}

        delta = (price_htf - ema20_htf) / ema20_htf
        is_neutral = abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT
        current_trend = "neutral" if is_neutral else ("bullish" if delta > 0 else "bearish")

        # Intraday adjustment (15m RSI — primary for 1h target)
        current_trend = self.calculate_trend_with_intraday(
            price_htf, ema20_htf, current_trend, indicators_15m
        )

        return self.update_trend_record(coin, current_trend)

    def update_trend_record(self, coin: str, current_trend: str) -> dict[str, Any]:
        """Update the trend state record and detect trend flips.

        Args:
            coin: The cryptocurrency symbol.
            current_trend: The latest calculated trend direction.

        Returns:
            The updated trend record dictionary for the coin.
        """
        record = self.trend_state.get(
            coin,
            {
                "trend": current_trend,
                "last_flip_cycle": self.current_cycle_number,
                "last_flip_direction": current_trend,
            },
        )
        prev_trend = record.get("trend", current_trend)
        recent_flip = False

        if prev_trend != current_trend:
            record["trend"] = current_trend
            if current_trend != "neutral":
                record["last_flip_cycle"] = self.current_cycle_number
                record["last_flip_direction"] = current_trend
                recent_flip = True
        else:
            last = record.get("last_flip_cycle", self.current_cycle_number)
            if (
                current_trend != "neutral"
                and (self.current_cycle_number - last) <= constants.TREND_FLIP_COOLDOWN_DEFAULT
            ):
                recent_flip = True

        record["last_seen_cycle"] = self.current_cycle_number
        self.trend_state[coin] = record
        return {
            "trend": current_trend,
            "recent_flip": recent_flip,
            "last_flip_cycle": record.get("last_flip_cycle"),
            "last_flip_direction": record.get("last_flip_direction"),
        }

    # --- Trend Flip Summary ---

    def get_recent_trend_flip_summary(self) -> list[str]:
        """Get a summary of recent trend flips within the guard and history windows.

        Returns:
            List of summary strings describing recent trend changes.
        """
        guard_window: int = self.trend_flip_cooldown
        history_window: int = max(
            guard_window, getattr(self, "trend_flip_history_window", guard_window)
        )
        entries: list[tuple[int, str]] = []
        for coin, record in self.trend_state.items():
            last_flip_cycle: int | None = record.get("last_flip_cycle")
            if last_flip_cycle is None:
                continue
            cycles_ago: int = self.current_cycle_number - last_flip_cycle
            if cycles_ago < 0 or cycles_ago > history_window:
                continue
            trend_label: str = record.get("trend", "unknown").upper()
            status: str = "GUARD" if cycles_ago <= guard_window else "RECENT"
            cycles_text: str
            if cycles_ago == 0:
                cycles_text = "current cycle"
            elif cycles_ago == 1:
                cycles_text = "1 cycle ago"
            else:
                cycles_text = f"{cycles_ago} cycles ago"
            direction_note: str = record.get("last_flip_direction", trend_label)
            entries.append(
                (
                    cycles_ago,
                    f"{coin}: {direction_note} since cycle {last_flip_cycle} ({status}, {cycles_text})",
                ),
            )
        entries.sort(key=lambda x: x[0])
        return [text for _, text in entries]

    # --- Trend Flip Guard ---

    def evaluate_trend_flip_guard(
        self,
        guard_ctx: dict[str, Any],
    ) -> tuple[bool, float, float]:
        """Orchestrate trend flip guard logic based on trade classification.

        Args:
            guard_ctx: Context containing trend flip and trade data.

        Returns:
            A tuple of (proceed_boolean, confidence, partial_margin_factor).
        """
        is_counter_trend = guard_ctx["is_counter_trend"]
        if is_counter_trend:
            return self._apply_counter_trend_flip_logic(guard_ctx)
        confidence, partial_margin_factor = self._apply_trend_following_flip_logic(guard_ctx)
        return True, confidence, partial_margin_factor

    def _apply_counter_trend_flip_logic(
        self,
        guard_ctx: dict[str, Any],
    ) -> tuple[bool, float, float]:
        """Apply specialized guard logic for counter-trend trades.

        Args:
            guard_ctx: Context containing trend flip and trade data.

        Returns:
            A tuple of (proceed_boolean, confidence, partial_margin_factor).
        """
        coin = guard_ctx["coin"]
        signal = guard_ctx["signal"]
        confidence = guard_ctx["confidence"]
        partial_margin_factor = guard_ctx["partial_margin_factor"]
        relax_mode_active = guard_ctx["relax_mode_active"]
        execution_report = guard_ctx["report"]
        trade = guard_ctx["trade"]
        classification = "counter_trend"

        counter_trend_cooldown = guard_ctx.get("counter_trend_cooldown", 0)
        if counter_trend_cooldown > 0:
            logger.info(
                "Counter-trend cooldown active: Blocking {} {} ({} cycles remaining).",
                coin,
                signal,
                counter_trend_cooldown,
            )
            execution_report["blocked"].append(
                {"coin": coin, "reason": "counter_trend_cooldown", "classification": classification}
            )
            trade["runtime_decision"] = "blocked_counter_trend_cooldown"
            return False, confidence, partial_margin_factor

        if not relax_mode_active:
            proceed, confidence, partial_margin_factor = self._check_flip_guard_confidence(
                guard_ctx, confidence, partial_margin_factor
            )
            if not proceed:
                return False, confidence, partial_margin_factor

        return True, confidence, partial_margin_factor

    def _check_flip_guard_confidence(
        self,
        guard_ctx: dict[str, Any],
        confidence: float,
        partial_margin_factor: float,
    ) -> tuple[bool, float, float]:
        """Check confidence floor and cap sizing after a trend flip.

        Args:
            guard_ctx: Context for the flip guard.
            confidence: Current confidence level.
            partial_margin_factor: Current margin scaling factor.

        Returns:
            A tuple of (proceed_boolean, updated_confidence, updated_factor).
        """
        coin = guard_ctx["coin"]
        signal = guard_ctx["signal"]
        guard_cycles_since_flip = guard_ctx["guard_cycles_since_flip"]
        execution_report = guard_ctx["report"]
        trade = guard_ctx["trade"]
        log_func = guard_ctx["log_func"]
        classification = "counter_trend"
        guard_window = self.trend_flip_cooldown

        if not (guard_cycles_since_flip is not None and guard_cycles_since_flip <= guard_window):
            return True, confidence, partial_margin_factor

        # Cycle-specific thresholds
        thresholds = {
            0: (0.60, 0.8, "same cycle"),
            1: (0.55, 0.9, "one cycle"),
            constants.TREND_FLIP_COOLDOWN_DEFAULT: (
                0.50,
                1.0,
                f"{constants.TREND_FLIP_COOLDOWN_DEFAULT} cycles",
            ),
        }

        if guard_cycles_since_flip in thresholds:
            min_c, cap, desc = thresholds[guard_cycles_since_flip]
            if confidence < min_c:
                log_func(
                    "flip_guard",
                    f"[BLOCK] Flip guard confidence floor: {coin} {signal} confidence {confidence:.2f} < {min_c:.2f} {desc} after flip.",
                    {
                        "coin": coin,
                        "signal": signal,
                        "confidence": confidence,
                        "min_required": min_c,
                        "cycles_since_flip": guard_cycles_since_flip,
                    },
                )
                execution_report["blocked"].append(
                    {
                        "coin": coin,
                        "reason": "trend_flip_guard_confidence",
                        "classification": classification,
                    }
                )
                trade["runtime_decision"] = "blocked_trend_flip_confidence"
                return False, confidence, partial_margin_factor
            partial_margin_factor = min(partial_margin_factor, cap)
            if cap < 1.0:
                log_func(
                    "sizing",
                    f"[WATCH] Trend flip guard ({classification}): {coin} sizing capped at {cap * 100:.0f}% {desc} after flip.",
                )

        return True, confidence, partial_margin_factor

    def _apply_trend_following_flip_logic(
        self,
        guard_ctx: dict[str, Any],
    ) -> tuple[float, float]:
        """Apply specialized guard logic for trend-following flip scenarios.

        Args:
            guard_ctx: Context containing trend flip and trade data.

        Returns:
            A tuple of (adjusted_confidence, adjusted_margin_factor).
        """
        coin = guard_ctx["coin"]
        direction = guard_ctx["direction"]
        confidence = guard_ctx["confidence"]
        partial_margin_factor = guard_ctx["partial_margin_factor"]
        guard_cycles_since_flip = guard_ctx.get("guard_cycles_since_flip")
        last_flip_direction = guard_ctx.get("last_flip_direction")
        trade = guard_ctx["trade"]
        log_func = guard_ctx["log_func"]
        guard_window = self.trend_flip_cooldown
        if (
            guard_cycles_since_flip is not None
            and guard_cycles_since_flip <= guard_window
            and last_flip_direction
        ):
            is_trend_direction = (last_flip_direction == "bullish" and direction == "long") or (
                last_flip_direction == "bearish" and direction == "short"
            )
            if is_trend_direction:
                orig_conf = confidence
                if guard_cycles_since_flip == 0:
                    confidence = max(
                        orig_conf * constants.FLIP_GUARD_CYCLE_0_MULTIPLIER,
                        orig_conf - constants.FLIP_GUARD_CYCLE_0_REDUCTION,
                        orig_conf * constants.FLIP_GUARD_CYCLE_0_FLOOR,
                    )
                    partial_margin_factor = min(
                        partial_margin_factor, constants.FLIP_GUARD_CYCLE_0_MARGIN
                    )
                    note = f"flip_guard_cycle_0({orig_conf:.2f}->{confidence:.2f})"
                    log_func(
                        "sizing",
                        f"[WATCH] Trend flip guard (trend-following): {coin} confidence {orig_conf:.2f} → {confidence:.2f} & sizing 50% immediately after flip.",
                    )
                elif guard_cycles_since_flip == 1:
                    confidence = max(
                        orig_conf * constants.FLIP_GUARD_CYCLE_1_MULTIPLIER,
                        orig_conf - constants.FLIP_GUARD_CYCLE_1_REDUCTION,
                        orig_conf * constants.FLIP_GUARD_CYCLE_1_FLOOR,
                    )
                    partial_margin_factor = min(
                        partial_margin_factor, constants.FLIP_GUARD_CYCLE_1_MARGIN
                    )
                    note = f"flip_guard_cycle_1({orig_conf:.2f}->{confidence:.2f})"
                    log_func(
                        "sizing",
                        f"[WATCH] Trend flip guard (trend-following): {coin} confidence {orig_conf:.2f} → {confidence:.2f} & sizing 70% {constants.ALIGNMENT_STRENGTH_L1} cycle after flip.",
                    )
                elif guard_cycles_since_flip == constants.TREND_FLIP_COOLDOWN_DEFAULT:
                    confidence = max(
                        orig_conf * constants.FLIP_GUARD_CYCLE_2_MULTIPLIER,
                        orig_conf * constants.FLIP_GUARD_CYCLE_2_FLOOR,
                    )
                    partial_margin_factor = min(
                        partial_margin_factor, constants.FLIP_GUARD_CYCLE_2_MARGIN
                    )
                    note = f"flip_guard_cycle_{constants.TREND_FLIP_COOLDOWN_DEFAULT}({orig_conf:.2f}->{confidence:.2f})"
                    log_func(
                        "sizing",
                        f"[WATCH] Trend flip guard (trend-following): {coin} confidence {orig_conf:.2f} → {confidence:.2f} & sizing 85% {constants.TREND_FLIP_COOLDOWN_DEFAULT} cycles after flip.",
                    )
                else:
                    note = None

                if note:
                    execution_report = guard_ctx.get("report", {})
                    if execution_report is not None:
                        if "debug_logs" not in execution_report:
                            execution_report["debug_logs"] = []
                        execution_report["debug_logs"].append(
                            {
                                "coin": coin,
                                "type": "flip_guard_adjustment",
                                "before": orig_conf,
                                "after": confidence,
                                "note": note,
                            }
                        )

        trade["confidence"] = confidence
        return confidence, partial_margin_factor
