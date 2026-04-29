"""Portfolio management and trade execution engine for TradeSeeker.

This module contains the PortfolioManager class which handles balance tracking,
position management, trade history, and directional bias logic.
"""

from __future__ import annotations

import copy
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

from config.config import Config
from src.core import constants
from src.core.backtest import AdvancedRiskManager
from src.core.market_data import RealMarketData
from src.core.regime_detector import RegimeDetector
from src.utils import format_num, safe_file_read, safe_file_read_cached, safe_file_write


# Define HTF constants
HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"
HTF_LABEL = HTF_INTERVAL

try:
    from src.services.binance import BinanceOrderExecutor

    BINANCE_IMPORT_ERROR = None
except Exception as e:
    BinanceOrderExecutor = None
    BINANCE_IMPORT_ERROR = str(e)


class PortfolioManager:
    """Manages the portfolio state, positions, and history.

    This class provides a comprehensive system for tracking balance, managing active positions,
    handling trade history, and applying directional bias based on performance.
    """

    # Class attributes with type hints
    initial_balance: float
    state_file: str
    history_file: str
    full_history_file: str
    override_file: str
    cycle_history_file: str
    max_cycle_history: int
    maintenance_margin_rate: float
    current_balance: float
    positions: dict[str, Any]
    directional_bias: dict[str, dict[str, Any]]
    trend_state: dict[str, dict[str, Any]]
    trend_flip_cooldown: int
    trend_flip_history_window: int
    indicator_cache: dict[str, dict[str, dict[str, Any]]]
    last_execution_report: dict[str, Any]
    history_reset_interval: int
    last_history_reset_cycle: int
    cycles_since_history_reset: int
    directional_cooldowns: dict[str, int]
    relaxed_countertrend_cycles: int
    counter_trend_cooldown: int
    counter_trend_consecutive_losses: int
    coin_cooldowns: dict[str, int]
    current_cycle_number: int
    trade_count: int
    total_value: float
    total_return: float
    start_time: datetime
    portfolio_values_history: list[float]
    sharpe_ratio: float
    is_live_trading: bool
    order_executor: Any

    def __init__(self, initial_balance: float | None = None) -> None:
        """Initialize the PortfolioManager with optional initial balance.

        Args:
        ----
            initial_balance: Starting balance for the portfolio. If None,
                loads from saved state or Config.

        """
        # Dynamic initial balance - if not provided, take from actual balance or use $200
        if initial_balance is None:
            # First try from saved state, otherwise take from Config
            saved_state: dict[str, Any] = safe_file_read_cached(
                "data/portfolio_state.json", default_data={}
            )
            self.initial_balance = saved_state.get("initial_balance", Config.INITIAL_BALANCE)
        else:
            self.initial_balance = initial_balance

        self.state_file = constants.PORTFOLIO_STATE_FILE
        self.history_file = constants.TRADE_HISTORY_FILE
        self.full_history_file = constants.FULL_TRADE_HISTORY_FILE
        self.override_file = "data/manual_override.json"
        self.cycle_history_file = "data/cycle_history.json"
        self.max_cycle_history = constants.MAX_CYCLE_HISTORY
        self.maintenance_margin_rate = constants.MAINTENANCE_MARGIN_RATE

        self.current_balance = self.initial_balance
        self.positions = {}
        self._lock = threading.RLock()  # RLock for re-entrant safety
        self.directional_bias = self._init_directional_bias()
        self.trend_state = {}
        self.trend_flip_cooldown = constants.TREND_FLIP_COOLDOWN_DEFAULT
        self.trend_flip_history_window = constants.TREND_FLIP_HISTORY_WINDOW
        # Trend flip cooldown management is kept on the PortfolioManager side.
        self.indicator_cache = {}
        self.last_execution_report = {}
        self.history_reset_interval = getattr(Config, "HISTORY_RESET_INTERVAL", 35)
        self.last_history_reset_cycle = 0
        self.cycles_since_history_reset = 0

        self.directional_cooldowns = {"long": 0, "short": 0}
        self.relaxed_countertrend_cycles = 0
        self.counter_trend_cooldown = 0
        self.counter_trend_consecutive_losses = 0
        self.coin_cooldowns = {}  # Coin-based cooldown: {coin: cycles_remaining}

        self.current_cycle_number = 0

        self.trade_history: list[dict[str, Any]] = self.load_trade_history()  # Load first
        self._ensure_full_history_exists()  # Migrate existing history if needed
        self.load_state()  # Loads balance, positions, trade_count
        self.cycle_history: list[dict[str, Any]] = self.load_cycle_history()
        self.risk_manager = AdvancedRiskManager()  # Initialize risk manager
        self.market_data = RealMarketData()  # Initialize market data for counter-trend detection

        # Fixed: Initialize live trading attributes used in update_prices()
        self.is_live_trading = False
        self.order_executor = None

        # Initialize total_value before _initialize_live_trading (which calls sync_live_account)
        self.total_value = self.current_balance
        self.total_return = 0.0
        self.start_time = datetime.now(timezone.utc)
        self.portfolio_values_history = [
            self.initial_balance,
        ]  # Track portfolio values for Sharpe ratio
        self.sharpe_ratio = 0.0

        self.update_prices(
            {},
            increment_loss_counters=False,
        )  # Calculate initial value with loaded positions

    def _ensure_full_history_exists(self) -> None:
        """Ensure full trade history exists, copying from active history if needed."""
        if not Path(self.full_history_file).exists():
            print(f"[INFO] Creating {self.full_history_file} from existing trade history...")
            safe_file_write(self.full_history_file, self.trade_history)

    def _init_directional_bias(self) -> dict[str, dict[str, Any]]:
        """Initialize directional bias metrics for long and short trade tracking.

        Returns
        -------
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

    def load_state(self) -> None:
        """Load portfolio state from local JSON file.

        Restores balance, active positions, trade counts, and directional bias metrics.
        """
        data: dict[str, Any] = safe_file_read_cached(self.state_file, default_data={})
        self.current_balance = data.get("current_balance", self.initial_balance)
        self.positions = data.get("positions", {})
        self.trade_count = data.get(
            "trade_count",
            len(self.trade_history),
        )  # Initialize from history if not in state
        print(
            f"[OK]    Loaded state ({len(self.positions)} positions, {self.trade_count} closed trades)"
            if data
            else "[INFO] No state file found.",
        )

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
        self.last_history_reset_cycle = data.get(
            "last_history_reset_cycle",
            self.last_history_reset_cycle,
        )
        self.cycles_since_history_reset = data.get(
            "cycles_since_history_reset",
            self.cycles_since_history_reset,
        )
        self.directional_cooldowns = data.get("directional_cooldowns", {"long": 0, "short": 0})
        self.relaxed_countertrend_cycles = data.get("relaxed_countertrend_cycles", 0)
        self.counter_trend_cooldown = data.get("counter_trend_cooldown", 0)
        self.counter_trend_consecutive_losses = data.get("counter_trend_consecutive_losses", 0)
        self.coin_cooldowns = data.get("coin_cooldowns", {})

    def save_state(self) -> None:
        """Save current portfolio state to local JSON file.

        Persists balance, positions, bias metrics, and cycle information with thread safety.
        """
        with self._lock:
            data: dict[str, Any] = {
                "current_balance": self.current_balance,
                "positions": self.positions,
                "total_value": self.total_value,
                "total_return": self.total_return,
                "initial_balance": self.initial_balance,
                "trade_count": self.trade_count,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "sharpe_ratio": self.sharpe_ratio,
                "directional_bias": self._serialize_directional_bias(),
                "last_history_reset_cycle": self.last_history_reset_cycle,
                "cycles_since_history_reset": self.cycles_since_history_reset,
                "directional_cooldowns": self.directional_cooldowns,
                "relaxed_countertrend_cycles": self.relaxed_countertrend_cycles,
                "counter_trend_cooldown": self.counter_trend_cooldown,
                "counter_trend_consecutive_losses": self.counter_trend_consecutive_losses,
                "coin_cooldowns": self.coin_cooldowns,
            }
            safe_file_write(self.state_file, data)
            # print("[OK]    Saved state.") # Silenced for less console noise

    # --- Live trading helpers -------------------------------------------------
    def _backup_historical_files(self, cycle_number: int) -> str | None:
        """Create a timestamped backup for historical JSON files before wiping them.

        Args:
        ----
            cycle_number: The current cycle number for grouping the backup.

        Returns:
        -------
            Path to the backup directory if successful, None otherwise.

        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("data/backups") / f"{timestamp}_cycle_{cycle_number}"
        files_to_backup = [
            (self.history_file, []),
            (self.full_history_file, []),
            (self.cycle_history_file, []),
            ("data/performance_history.json", []),
            ("data/performance_report.json", []),
        ]

        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            backed_up = []

            for file_path, default in files_to_backup:
                path_obj = Path(file_path)
                data = safe_file_read(str(path_obj), default)
                # Skip writing files that never existed and match the empty default
                if data == default and not path_obj.exists():
                    continue

                target_path = backup_dir / path_obj.name
                safe_file_write(str(target_path), data)

                # Calculate items count for metadata
                items_count = None
                if isinstance(data, (list, dict)):
                    items_count = len(data)

                backed_up.append({"file": file_path, "items": items_count})

            metadata = {
                "cycle_number": cycle_number,
                "backed_up_at": datetime.now(timezone.utc).isoformat(),
                "files": backed_up,
            }
            safe_file_write(str(backup_dir / "metadata.json"), metadata)
            print(f"[OK]    History backup created at {backup_dir}")
            return str(backup_dir)
        except Exception as e:
            print(f"[WARN]  History backup failed: {e}")
            return None

    def reset_historical_data(self, cycle_number: int) -> None:
        """Clear historical logs to prevent long-term bias while keeping live positions.

        Args:
        ----
            cycle_number: The current cycle number at which reset occurs.

        """
        # FIX: Keep lock for entire state modification to prevent race conditions
        with self._lock:
            self._backup_historical_files(cycle_number)
            print(f"[CLEANUP] HISTORY RESET: Clearing logs at cycle {cycle_number}")
            self.trade_history = []
            self.trade_count = 0
            self.directional_bias = self._init_directional_bias()
            self.trend_state = {}
            self.cycle_history = []
        # File writes outside lock - they're thread-safe themselves
        safe_file_write(self.history_file, [])
        safe_file_write(self.cycle_history_file, [])
        safe_file_write("data/performance_history.json", [])
        # Preserve existing performance reports, just add a reset marker
        existing_reports = safe_file_read("data/performance_report.json", [])
        if isinstance(existing_reports, dict):
            # Old format - convert to array
            existing_reports = [existing_reports] if "reset_reason" not in existing_reports else []
        elif not isinstance(existing_reports, list):
            existing_reports = []

        # Add reset marker
        reset_marker = {
            "reset_reason": "periodic_bias_control",
            "reset_at_cycle": cycle_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        existing_reports.append(reset_marker)

        # Keep only last MAX_REPORT_ENTRIES entries
        if len(existing_reports) > constants.MAX_REPORT_ENTRIES:
            existing_reports = existing_reports[-constants.MAX_REPORT_ENTRIES :]

        safe_file_write("data/performance_report.json", existing_reports)
        self.portfolio_values_history = [self.total_value]
        for pos in self.positions.values():
            pos["loss_cycle_count"] = 0
            pos["profit_cycle_count"] = 0
        self.last_history_reset_cycle = cycle_number
        self.cycles_since_history_reset = 0
        self.directional_cooldowns = {"long": 0, "short": 0}
        self.coin_cooldowns = {}  # Also reset coin-based cooldowns
        self.counter_trend_cooldown = 0
        self.counter_trend_consecutive_losses = 0
        self.relaxed_countertrend_cycles = 0
        self.save_state()
        print("[OK]    History reset complete.")

    def _serialize_directional_bias(self) -> dict[str, dict[str, Any]]:
        """Serialize directional bias metrics for state saving.

        Returns
        -------
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

    def load_trade_history(self) -> list[dict[str, Any]]:
        """Load trade history from the local JSON file.

        Returns
        -------
            List of trade records.

        """
        history: list[dict[str, Any]] = safe_file_read(self.history_file, default_data=[])
        print(f"[OK]    Loaded {len(history)} trades.")
        return history

    def load_cycle_history(self) -> list[dict[str, Any]]:
        """Load cycle-by-cycle performance history records.

        Returns
        -------
            List of cycle history contexts.

        """
        history: list[dict[str, Any]] = safe_file_read(self.cycle_history_file, default_data=[])
        return history

    def save_trade_history(self) -> None:
        """Save the most recent trade history to the active history file."""
        with self._lock:
            history_to_save: list[dict[str, Any]] = self.trade_history[
                -constants.MAX_TRADE_HISTORY_DISPLAY :
            ]
            safe_file_write(self.history_file, history_to_save)
            print(f"[OK]    Saved {len(history_to_save)} trades.")

    def add_to_history(self, trade: dict[str, Any]) -> None:
        """Add a completed trade to both active and full history files.

        Args:
        ----
            trade: The trade record to append.

        """
        # CRITICAL: Keep lock for ALL updates (History + Bias + State) to ensure atomicity
        with self._lock:
            # 1. Update memory state & active history
            self.trade_history.append(trade)
            self.trade_count = len(self.trade_history)
            self.save_trade_history()  # Internal logic uses _lock if we add it

            # 2. Update full history file
            full_history = safe_file_read(self.full_history_file, [])

            # Robustness: If read failed but file is not empty, retry once
            if not full_history:
                path_obj = Path(self.full_history_file)
                if path_obj.exists() and path_obj.stat().st_size > 0:
                    time.sleep(0.1)
                    full_history = safe_file_read(self.full_history_file, [])

            full_history.append(trade)

            if not safe_file_write(self.full_history_file, full_history):
                print(f"[ERR]   Failed to save full history for {trade.get('symbol')}")

            # 3. Update directional bias & persistence (Must be INSIDE lock)
            self.update_directional_bias(trade)
            self.save_state()  # This also uses _lock (RLock handles it)

    def update_directional_bias(self, trade: dict[str, Any]) -> None:
        """Update directional bias based on trade outcome and logic."""
        direction = trade.get("direction")
        if direction not in ("long", "short"):
            return
        stats = self.directional_bias[direction]
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        print(f"[INFO]  update_directional_bias called: {direction.upper()} trade, PnL=${pnl:.2f}")
        stats["rolling"].append(pnl)
        stats["net_pnl"] += pnl
        stats["trades"] += 1

        if pnl > 0:
            self._handle_win_bias(direction, stats, pnl, trade)
        elif pnl < 0:
            self._handle_loss_bias(direction, stats, pnl, trade)
        else:
            # pnl == 0 case (breakeven)
            stats["consecutive_losses"] = 0
            stats["consecutive_wins"] = 0
            stats["caution_win_progress"] = 0
            stats["loss_streak_loss_usd"] = 0.0
            if trade.get("trend_alignment") == "counter_trend":
                self.counter_trend_consecutive_losses = 0

    def _handle_win_bias(
        self, direction: str, stats: dict[str, Any], pnl: float, trade: dict[str, Any]
    ) -> None:
        """Process logic for a winning trade."""
        from config.config import Config

        stats["wins"] += 1
        stats["consecutive_losses"] = 0
        stats["consecutive_wins"] = stats.get("consecutive_wins", 0) + 1

        # Win Streak Cooldown
        if stats["consecutive_wins"] >= Config.WIN_STREAK_COOLDOWN_THRESHOLD:
            self._activate_directional_cooldown(direction, Config.WIN_STREAK_COOLDOWN_CYCLES)
            print(
                f"[WARN]  Win streak cooldown for {direction.upper()}: {stats['consecutive_wins']} wins"
            )

        # Smart Cooldown (WIN)
        coin_symbol = trade.get("symbol", "").upper()
        if coin_symbol:
            self.coin_cooldowns[coin_symbol] = Config.SMART_COOLDOWN_WIN
            print(
                f"[WARN]  Smart Cooldown (WIN) ACTIVATED for {coin_symbol}: {Config.SMART_COOLDOWN_WIN} cycles"
            )

        if stats.get("caution_active"):
            stats["caution_win_progress"] = stats.get("caution_win_progress", 0) + 1
            if stats["caution_win_progress"] >= constants.CAUTION_WIN_PROGRESS_THRESHOLD:
                stats["caution_active"] = False
                stats["caution_win_progress"] = 0

        # Reset counter-trend consecutive losses on win
        if trade.get("trend_alignment") == "counter_trend":
            self.counter_trend_consecutive_losses = 0

    def _handle_loss_bias(
        self, direction: str, stats: dict[str, Any], pnl: float, trade: dict[str, Any]
    ) -> None:
        """Process logic for a losing trade."""
        from config.config import Config

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
            self.coin_cooldowns[coin_symbol] = Config.SMART_COOLDOWN_LOSS
            print(
                f"[WARN]  Smart Cooldown (LOSS) ACTIVATED for {coin_symbol}: {Config.SMART_COOLDOWN_LOSS} cycles"
            )

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
            self._activate_directional_cooldown(direction, constants.DEFAULT_COOLDOWN_CYCLES)
            print(
                f"[WARN]  Directional cooldown ACTIVATED for {direction.upper()}: consecutive_losses={consecutive}, loss_streak_usd=${loss_streak_usd:.2f}",
            )

        # Counter-trend cooldown: consecutive counter-trend losses
        if trade.get("trend_alignment") == "counter_trend":
            self.counter_trend_consecutive_losses += 1
            if self.counter_trend_consecutive_losses >= constants.COUNTER_TREND_LOSS_THRESHOLD:
                self.counter_trend_cooldown = constants.DEFAULT_COOLDOWN_CYCLES
                self.counter_trend_consecutive_losses = 0

    def count_positions_by_direction(self) -> dict[str, int]:
        """Count currently open positions categorized by trade direction.

        Returns
        -------
            Dictionary containing counts for 'long' and 'short' positions.

        """
        counts = {"long": 0, "short": 0}
        for pos in self.positions.values():
            direction = pos.get("direction")
            if direction in counts:
                counts[direction] += 1
        return counts

    def _activate_directional_cooldown(self, direction: str, cycles: int = 3) -> None:
        """Activate a cooldown for a specific trade direction.

        Args:
        ----
            direction: The trade direction ('long' or 'short') to cool down.
            cycles: Number of cycles the cooldown should remain active.

        """
        if direction not in ("long", "short"):
            return
        current = self.directional_cooldowns.get(direction, 0)
        # If there is a longer duration than currently exists, use it
        new_cooldown = max(current, cycles)
        # FIX: Cooldown overflow protection - cap at MAX_DIRECTIONAL_COOLDOWN cycles
        self.directional_cooldowns[direction] = min(
            new_cooldown, constants.MAX_DIRECTIONAL_COOLDOWN
        )
        self.relaxed_countertrend_cycles = max(
            self.relaxed_countertrend_cycles,
            constants.MIN_RELAXED_CYCLES,
        )
        print(
            f"[WARN]  Directional cooldown activated for {direction.upper()} trades ({constants.MIN_RELAXED_CYCLES} cycles). Counter-trend restrictions relaxed for {constants.MIN_RELAXED_CYCLES} cycles.",
        )

    def tick_cooldowns(self) -> None:
        """Decrement all active cooldown timers."""
        print(
            f"[TIME] tick_cooldowns called. Current cooldowns: {self.directional_cooldowns}, Coin cooldowns: {self.coin_cooldowns}"
        )

        # 1. Directional Cooldowns
        self._tick_directional_cooldowns()

        # 2. Coin-based Cooldowns
        self._tick_coin_cooldowns()

        # 3. Relaxed Mode Cooldown
        if self.relaxed_countertrend_cycles > 0:
            self.relaxed_countertrend_cycles -= 1
            if self.relaxed_countertrend_cycles == 0:
                print("[OK]    Relaxed counter-trend validation mode EXPIRED.")

    def _tick_directional_cooldowns(self) -> None:
        """Decrement long/short directional cooldowns."""
        for direction in ("long", "short"):
            cycles = self.directional_cooldowns.get(direction, 0)
            if cycles > 0:
                self.directional_cooldowns[direction] = cycles - 1
                print(
                    f"[TIME] {direction.upper()} cooldown: {cycles} -> {self.directional_cooldowns[direction]} cycles remaining"
                )
                if self.directional_cooldowns[direction] == 0:
                    if direction in self.directional_bias:
                        self.directional_bias[direction]["loss_streak_loss_usd"] = 0.0
                        self.directional_bias[direction]["consecutive_losses"] = 0
                    print(
                        f"[OK]    Directional cooldown cleared for {direction.upper()} trades. Loss streak reset."
                    )

    def _tick_coin_cooldowns(self) -> None:
        """Decrement individual coin cooldowns."""
        to_remove = []
        for coin, cycles in self.coin_cooldowns.items():
            if cycles > 0:
                self.coin_cooldowns[coin] = cycles - 1
                print(
                    f"[TIME] {coin} coin cooldown: {cycles} -> {self.coin_cooldowns[coin]} cycles remaining"
                )
                if self.coin_cooldowns[coin] == 0:
                    to_remove.append(coin)
                    print(f"[OK]    Coin cooldown cleared for {coin}.")

        for coin in to_remove:
            del self.coin_cooldowns[coin]
        if self.counter_trend_cooldown > 0:
            self.counter_trend_cooldown -= 1
            if self.counter_trend_cooldown == 0:
                print("[OK]    Counter-trend cooldown cleared.")

    def get_trend_following_strength(self, coin: str, signal: str) -> dict[str, Any] | None:
        """Determine trend-following strength including 1h, 15m, and 3m alignment.

        Args:
        ----
            coin: The cryptocurrency symbol.
            signal: The trade signal type ('buy_to_enter' or 'sell_to_enter').

        Returns:
        -------
            Dictionary with 'strength', 'alignment_info', and 'trends' if valid, else None.

        """
        result: dict[str, Any] | None = None
        try:
            inds_htf: dict[str, Any] = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            inds_15m: dict[str, Any] = self.market_data.get_technical_indicators(coin, "15m")
            inds_3m: dict[str, Any] = self.market_data.get_technical_indicators(coin, "3m")

            if all("error" not in i for i in (inds_htf, inds_15m, inds_3m)):
                p1h, e1h = inds_htf.get("current_price"), inds_htf.get("ema_20")
                p15, e15 = inds_15m.get("current_price"), inds_15m.get("ema_20")
                p3m, e3m = inds_3m.get("current_price"), inds_3m.get("ema_20")

                if all(isinstance(x, (int, float)) for x in [p1h, e1h, p15, e15, p3m, e3m]):
                    t1h = "BULLISH" if p1h > e1h else "BEARISH"
                    t15 = "BULLISH" if p15 > e15 else "BEARISH"
                    t3m = "BULLISH" if p3m > e3m else "BEARISH"
                    sig_dir = "BULLISH" if signal == "buy_to_enter" else "BEARISH"

                    if t1h == sig_dir:
                        trends = {"1h": t1h, "15m": t15, "3m": t3m}
                        if t15 == sig_dir and t3m == sig_dir:
                            result = {
                                "strength": "STRONG",
                                "alignment_info": f"Perfect alignment: {HTF_LABEL}+15m+3m all {sig_dir}",
                                "trends": trends,
                            }
                        elif t15 == sig_dir:
                            result = {
                                "strength": "MEDIUM_15",
                                "alignment_info": f"Moderate: {HTF_LABEL}+15m {sig_dir} (3m {t3m})",
                                "trends": trends,
                            }
                        elif t3m == sig_dir:
                            result = {
                                "strength": "MEDIUM_3",
                                "alignment_info": f"Moderate: {HTF_LABEL}+3m {sig_dir} (15m {t15})",
                                "trends": trends,
                            }
                        else:
                            result = {
                                "strength": "WEAK",
                                "alignment_info": f"Weak: Only {HTF_LABEL} {sig_dir} (15m {t15}, 3m {t3m})",
                                "trends": trends,
                            }
        except Exception as e:
            print(f"[WARN]  Trend-following strength detection error for {coin}: {e}")
        return result

    def apply_directional_bias(
        self,
        bias_ctx: dict[str, Any],
    ) -> float:
        """Apply directional bias metrics to adjust confidence.

        Args:
        ----
            bias_ctx: Context containing 'signal', 'confidence', 'bias_metrics',
                and 'current_trend'.

        Returns:
        -------
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

        # Minimum confidence floor
        min_floor = original_confidence * constants.CONFIDENCE_FLOOR_MULTIPLIER
        return max(confidence, min_floor, Config.MIN_CONFIDENCE)

    def _apply_bias_penalty(self, confidence: float, stats: dict[str, Any]) -> float:
        """Apply caution and negative rolling average penalties.

        Args:
        ----
            confidence: Base confidence value.
            stats: Dictionary of directional bias statistics.

        Returns:
        -------
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
        ----
            confidence: Base confidence value.
            side: The trade side ('long' or 'short').
            current_trend: The detected high-level trend direction.

        Returns:
        -------
            The trend-adjusted confidence value.

        """
        from config.config import Config

        trend_lower = current_trend.lower() if isinstance(current_trend, str) else "unknown"

        if trend_lower == "neutral":
            confidence *= Config.DIRECTIONAL_NEUTRAL_MULTIPLIER
        elif trend_lower == "bullish":
            if side == "long":
                confidence *= Config.DIRECTIONAL_BULLISH_LONG_MULTIPLIER
            elif side == "short":
                confidence *= Config.DIRECTIONAL_BULLISH_SHORT_MULTIPLIER
        elif trend_lower == "bearish":
            if side == "long":
                confidence *= Config.DIRECTIONAL_BEARISH_LONG_MULTIPLIER
            elif side == "short":
                confidence *= Config.DIRECTIONAL_BEARISH_SHORT_MULTIPLIER

        return confidence

    def get_directional_bias_metrics(self) -> dict[str, dict[str, Any]]:
        """Calculate and retrieve holistic directional bias metrics.

        Returns
        -------
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
            # Profitability Index = Total Profit / (|Total Profit| + |Total Loss|) * 100
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
                "profitability_index": profitability_index,  # Added profitability_index based on profit/loss amounts
                "rolling_sum": rolling_sum,
                "rolling_avg": rolling_avg,
                "consecutive_losses": stats["consecutive_losses"],
                "consecutive_wins": stats.get("consecutive_wins", 0),
                "caution_active": stats.get("caution_active", False),
                "caution_win_progress": stats.get("caution_win_progress", 0),
            }
        return metrics

    def update_trend_state(
        self,
        coin: str,
        indicators_htf: dict[str, Any],
        indicators_3m: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Calculate and update the trend state for a given coin.

        Args:
        ----
            coin: The cryptocurrency symbol.
            indicators_htf: HTF technical indicators.
            indicators_3m: Optional 3m technical indicators for intraday adjustment.

        Returns:
        -------
            Dictionary containing the updated trend record.

        """
        from config.config import Config

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

        # Intraday adjustment
        current_trend = self._calculate_trend_with_intraday(
            price_htf, ema20_htf, current_trend, indicators_3m
        )

        return self._update_trend_record(coin, current_trend)

    def _calculate_trend_with_intraday(
        self,
        price_htf: float,
        ema20_htf: float,
        current_trend: str,
        indicators_3m: dict[str, Any] | None,
    ) -> str:
        """Apply intraday RSI and EMA filters to the high-level trend.

        Args:
        ----
            price_htf: The HTF current price.
            ema20_htf: The HTF EMA 20 value.
            current_trend: The initial detected trend.
            indicators_3m: Technical indicators for the 3m timeframe.

        Returns:
        -------
            The adjusted trend direction as a string.

        """
        from config.config import Config

        if not (indicators_3m and isinstance(indicators_3m, dict) and "error" not in indicators_3m):
            return current_trend

        p3m = indicators_3m.get("current_price")
        e3m = indicators_3m.get("ema_20", p3m)
        r3m = indicators_3m.get("rsi_14", indicators_3m.get("rsi_7", 50))

        if not all(isinstance(x, (int, float)) for x in [p3m, e3m, r3m]):
            return current_trend

        # Neutralization logic
        intra_bull = p3m >= e3m
        if (
            current_trend == "bearish" and intra_bull and r3m >= Config.INTRADAY_NEUTRAL_RSI_HIGH
        ) or (
            current_trend == "bullish" and not intra_bull and r3m <= Config.INTRADAY_NEUTRAL_RSI_LOW
        ):
            current_trend = "neutral"

        # Strong trend recovery logic
        if current_trend == "neutral":
            if price_htf <= ema20_htf and p3m <= e3m and r3m <= Config.TREND_SHORT_RSI_THRESHOLD:
                current_trend = "bearish"
            elif price_htf >= ema20_htf and p3m >= e3m and r3m >= Config.TREND_LONG_RSI_THRESHOLD:
                current_trend = "bullish"

        return current_trend

    def _update_trend_record(self, coin: str, current_trend: str) -> dict[str, Any]:
        """Update the trend state record and detect trend flips.

        Args:
        ----
            coin: The cryptocurrency symbol.
            current_trend: The latest calculated trend direction.

        Returns:
        -------
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

    def get_recent_trend_flip_summary(self) -> list[str]:
        """Get a summary of recent trend flips within the guard and history windows.

        Returns
        -------
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

    def load_cycle_history(self) -> list[dict[str, Any]]:
        """Load cycle history records from the local JSON file.

        Returns
        -------
            List of cycle history records.

        """
        history = safe_file_read(self.cycle_history_file, default_data=[])
        print(f"[OK]    Loaded {len(history)} cycles.")
        return history

    def add_to_cycle_history(self, history_ctx: dict[str, Any]) -> None:
        """Add the current state and AI decisions to the cycle history log.

        Args:
        ----
            history_ctx: Context containing cycle_number, prompt, thoughts, and decisions.

        """
        cycle_number = history_ctx["cycle_number"]
        prompt = history_ctx["prompt"]
        thoughts = history_ctx["thoughts"]
        decisions = history_ctx["decisions"]
        status = history_ctx.get("status", "ai_decision")
        metadata = history_ctx.get("metadata")
        with self._lock:
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
                ("PORTFOLIO (JSON):", "Portfolio"),
            ]

            found_sections = [name for marker, name in json_sections if marker in prompt]

            if found_sections:
                # JSON format prompt - create a structured summary
                try:
                    section_count = len(found_sections)
                    if section_count <= constants.PROMPT_SUMMARY_MAX_SECTIONS:
                        summary_text = ", ".join(found_sections)
                    else:
                        summary_text = f"{', '.join(found_sections[: constants.PROMPT_SUMMARY_MAX_SECTIONS])} + {section_count - constants.PROMPT_SUMMARY_MAX_SECTIONS} more"
                    prompt_summary = (
                        f"JSON Format ({section_count} sections): {summary_text} | "
                        + prompt[:200]
                        + "..."
                    )
                except Exception:
                    # FIX: Replace bare except with specific exception handling
                    prompt_summary = (
                        prompt[: constants.PROMPT_SUMMARY_MAX_SECTIONS] + "..."
                        if len(prompt) > constants.PROMPT_SUMMARY_TRUNCATE
                        else prompt
                    )
            else:
                # Text format prompt - use original truncation
                prompt_summary = (
                    prompt[: constants.PROMPT_SUMMARY_TRUNCATE] + "..."
                    if len(prompt) > constants.PROMPT_SUMMARY_TRUNCATE
                    else prompt
                )

        # Add cooldown information to cycle data
        cooldown_info = {
            "directional_cooldowns": dict(self.directional_cooldowns),
            "relaxed_countertrend_cycles": self.relaxed_countertrend_cycles,
            "counter_trend_cooldown": self.counter_trend_cooldown,
            "coin_cooldowns": dict(self.coin_cooldowns),
        }

        cycle_data = {
            "cycle": cycle_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_prompt_summary": prompt_summary,
            "chain_of_thoughts": thoughts,
            "decisions": decisions,
            "status": status,
            "cooldown_status": cooldown_info,  # Always include cooldown status
        }
        if metadata:
            cycle_data["metadata"] = metadata
        self.cycle_history.append(cycle_data)
        self.cycle_history = self.cycle_history[-constants.MAX_CYCLE_HISTORY :]
        safe_file_write(self.cycle_history_file, self.cycle_history)
        print(f"[OK]    Saved cycle {cycle_number} (Total: {len(self.cycle_history)})")

    def _update_peak_pnl_tracking(self, coin: str, pos: dict[str, Any]) -> None:
        """Track peak PnL and calculate profit erosion for a single position.

        Args:
        ----
            coin: The cryptocurrency symbol.
            pos: The position dictionary to update.

        """
        current_pnl: float = pos.get("unrealized_pnl", 0.0)
        if "peak_pnl" not in pos:
            pos["peak_pnl"] = 0.0
            pos["peak_pnl_cycle"] = None
        peak_pnl: float = pos.get("peak_pnl", 0.0)
        if current_pnl > peak_pnl and current_pnl > 0:
            pos["peak_pnl"] = current_pnl
            pos["peak_pnl_cycle"] = getattr(self, "current_cycle_number", None)
            peak_pnl = current_pnl

        # Erosion sadece KARDAYKEN hesaplanır (zararda erosion yok)
        if current_pnl > 0 and peak_pnl > 0:
            erosion_from_peak = peak_pnl - current_pnl
            erosion_pct = (erosion_from_peak / peak_pnl * 100) if peak_pnl > 0 else 0.0
        else:
            erosion_from_peak = 0.0
            erosion_pct = 0.0

        pos["erosion_from_peak"] = round(erosion_from_peak, 4)
        pos["erosion_pct"] = round(erosion_pct, 2)
        zone: str = self._get_erosion_zone(coin, pos)
        self._apply_erosion_status(pos, erosion_pct, zone, peak_pnl)

    def _get_erosion_zone(self, coin: str, pos: dict[str, Any]) -> str:
        """Determine price location zone for erosion sensitivity analysis.

        Args:
        ----
            coin: The cryptocurrency symbol.
            pos: The position dictionary.

        Returns:
        -------
            String representing the sparkline zone (e.g., 'UPPER_10', 'MIDDLE').

        """
        try:
            symbol = pos.get("symbol", "")
            coin_name = symbol.replace("USDT", "") if symbol else coin
            from config.config import Config

            htf_interval = getattr(Config, "HTF_INTERVAL", "1h")
            if hasattr(self, "indicator_cache") and self.indicator_cache:
                indicators_htf = self.indicator_cache.get_indicators(
                    coin_name, htf_interval, self.market_data
                )
                if indicators_htf and "smart_sparkline" in indicators_htf:
                    return (
                        indicators_htf["smart_sparkline"]
                        .get("price_location", {})
                        .get("zone", "MIDDLE")
                    )
        except Exception:
            pass
        return "MIDDLE"

    def _apply_erosion_status(
        self, pos: dict[str, Any], erosion_pct: float, zone: str, peak_pnl: float
    ) -> None:
        """Apply erosion status based on thresholds and market zone.

        Args:
        ----
            pos: The position dictionary to update.
            erosion_pct: Percentage of profit eroded from peak.
            zone: Current price location zone.
            peak_pnl: Highest PnL reached for this position.

        """
        from config.config import Config

        erosion_rate = (
            Config.EROSION_RATE_EXTREME
            if zone in ("UPPER_10", "LOWER_10")
            else Config.EROSION_RATE_NORMAL
        )
        margin_usd = pos.get("margin_usd", 0.0)
        if isinstance(margin_usd, str):
            margin_usd = 0.0
        min_meaningful = max(margin_usd * erosion_rate, Config.EROSION_MIN_PROFIT_USD)
        status = "NONE"
        if peak_pnl > 0 and peak_pnl >= min_meaningful:
            if erosion_pct >= constants.EROSION_THRESHOLD_CRITICAL:
                status = "CRITICAL"
            elif erosion_pct >= constants.EROSION_THRESHOLD_SIGNIFICANT:
                status = "SIGNIFICANT"
            elif erosion_pct >= constants.EROSION_THRESHOLD_MINOR:
                status = "MINOR"
        pos["erosion_status"] = status

    def update_prices(
        self, new_prices: dict[str, float], increment_loss_counters: bool = True
    ) -> None:
        """Update portfolio prices and recalculate total valuation and return.

        Args:
        ----
            new_prices: Dictionary mapping coin symbols to current market prices.
            increment_loss_counters: Whether to increment cycle counters for
                positions in loss.

        """
        # FIX: Keep lock for entire position modification to prevent race conditions
        with self._lock:
            total_unrealized_pnl = 0.0
            for coin, price in new_prices.items():
                if coin in self.positions and isinstance(price, (int, float)) and price > 0:
                    pos = self.positions[coin]
                    self._sync_runtime_price(pos, price)

                    pnl = self._update_position_pnl(pos, price)
                    pos["unrealized_pnl"] = pnl
                    total_unrealized_pnl += pnl

                    # Track peak PnL and calculate erosion
                    self._update_peak_pnl_tracking(coin, pos)

                    if increment_loss_counters:
                        self._update_price_counters(coin, pos, pnl)
                elif coin in self.positions:
                    print(f"[WARN]  Invalid price for {coin}: {price}. PnL skip.")

        self._calculate_total_portfolio_value()

        if self.initial_balance > 0:
            self.total_return = (
                (self.total_value - self.initial_balance) / self.initial_balance
            ) * 100
        else:
            self.total_return = 0.0

        # Update portfolio history for Sharpe ratio calculation
        self.portfolio_values_history.append(self.total_value)
        if (
            len(self.portfolio_values_history) > constants.PORTFOLIO_HISTORY_MAX_ENTRIES
        ):  # Keep last 5000 values (approx 1 week)
            self.portfolio_values_history = self.portfolio_values_history[
                -constants.PORTFOLIO_HISTORY_MAX_ENTRIES :
            ]

        # Calculate Sharpe ratio
        self.sharpe_ratio = self.calculate_sharpe_ratio()

        # Save updated state with Sharpe ratio
        self.save_state()

    def _sync_runtime_price(self, pos: dict[str, Any], price: float) -> None:
        """Sync position price with market price, preferring mark price in live mode.

        Args:
        ----
            pos: The position dictionary to update.
            price: Current spot market price.

        """
        if self.is_live_trading:
            # In live mode, prefer keeping Binance markPrice if available
            existing_price = pos.get("current_price", 0)
            if existing_price > 0:
                # Keep Binance markPrice, but update if Spot price is significantly different (>0.1%)
                price_diff_pct = abs(price - existing_price) / existing_price
                if price_diff_pct > constants.EMA_BAND_SENSITIVITY:
                    # Use Spot price as fallback if markPrice seems stale
                    pos["current_price"] = price
            else:
                # No existing price, use Spot price
                pos["current_price"] = price
        else:
            # Simulation mode: always use Spot price
            pos["current_price"] = price

    def _update_price_counters(self, coin: str, pos: dict[str, Any], pnl: float) -> None:
        """Update loss and profit cycle counters for a specific position.

        Args:
        ----
            coin: The cryptocurrency symbol.
            pos: The position dictionary to update.
            pnl: Current unrealized PnL in USD.

        """
        direction = pos.get("direction", "unknown")
        if pnl <= 0:
            pos["loss_cycle_count"] = pos.get("loss_cycle_count", 0) + 1
            pos["profit_cycle_count"] = 0  # Reset profit counter when negative
            new_count = pos["loss_cycle_count"]
            if new_count in constants.WATCH_CYCLES_LIST:
                print(
                    f"[WATCH] LOSS CYCLE WATCH: {coin} {direction} negative for {new_count} cycles (PnL ${pnl:.2f}).",
                )
        else:
            pos["loss_cycle_count"] = 0
            pos["profit_cycle_count"] = pos.get("profit_cycle_count", 0) + 1
            new_profit_count = pos["profit_cycle_count"]
            if new_profit_count in constants.PROFIT_WATCH_CYCLES_LIST:
                print(
                    f"[WATCH] PROFIT CYCLE WATCH: {coin} {direction} profitable for {new_profit_count} cycles (PnL ${pnl:.2f}).",
                )

    def _calculate_total_portfolio_value(self) -> None:
        """Calculate total portfolio value based on balance, margin, and PnL."""
        if not self.positions:
            self._calculate_empty_portfolio_value()
            return

        total_margin_used = 0.0
        total_unrealized_pnl = 0.0

        for pos in self.positions.values():
            pnl = pos.get("unrealized_pnl", 0.0)
            if isinstance(pnl, (int, float)):
                total_unrealized_pnl += pnl

            # Calculate margin (for cross margin, handle potentially missing margin_usd)
            margin = pos.get("margin_usd", 0.0)
            if margin <= 0:
                notional = pos.get("notional_usd", 0.0)
                leverage = pos.get("leverage", 1)
                if notional > 0 and leverage > 0:
                    margin = notional / leverage
            if isinstance(margin, (int, float)) and margin > 0:
                total_margin_used += margin

        self.total_value = self.current_balance + total_margin_used + total_unrealized_pnl

    def _calculate_empty_portfolio_value(self) -> None:
        """Calculate total value when no positions are active."""
        if self.is_live_trading and self.order_executor and self.order_executor.is_live():
            try:
                overview = self.order_executor.get_account_overview()
                if overview and overview.get("totalWalletBalance", 0) > 0:
                    self.total_value = float(overview["totalWalletBalance"])
                else:
                    self.total_value = self.current_balance
            except Exception as e:
                print(f"Error updating balance from overview: {e}")
                self.total_value = self.current_balance
        else:
            self.total_value = self.current_balance

        if self.initial_balance > 0:
            self.total_return = (
                (self.total_value - self.initial_balance) / self.initial_balance
            ) * 100
        else:
            self.total_return = 0.0

        # Update portfolio history for Sharpe ratio calculation
        self.portfolio_values_history.append(self.total_value)
        if (
            len(self.portfolio_values_history) > constants.PORTFOLIO_HISTORY_MAX_ENTRIES
        ):  # Keep last 5000 values (approx 1 week)
            self.portfolio_values_history = self.portfolio_values_history[
                -constants.PORTFOLIO_HISTORY_MAX_ENTRIES :
            ]

        # Calculate Sharpe ratio
        self.sharpe_ratio = self.calculate_sharpe_ratio()

        # Save updated state with Sharpe ratio
        self.save_state()

    def calculate_sharpe_ratio(self) -> float:
        """Calculate the annualized Sharpe ratio based on portfolio value history.

        Returns
        -------
            The calculated Sharpe ratio, or 0.0 if insufficient history.

        """
        if len(self.portfolio_values_history) < constants.MIN_HISTORY_FOR_ANALYSIS:
            return 0.0

        try:
            # Calculate simple returns (percentage changes)
            returns = []
            for i in range(1, len(self.portfolio_values_history)):
                if self.portfolio_values_history[i - 1] > 0:
                    ret = (
                        self.portfolio_values_history[i] - self.portfolio_values_history[i - 1]
                    ) / self.portfolio_values_history[i - 1]
                    returns.append(ret)

            if len(returns) < constants.MIN_HISTORY_FOR_ANALYSIS:
                return 0.0

            # Nof1ai style: Simple Sharpe ratio with 0% risk-free rate
            # Daily Sharpe ratio (assuming 2-minute cycles = 720 cycles per day)
            risk_free_rate = 0.0

            # Calculate excess returns
            excess_returns = [r - risk_free_rate for r in returns]

            # Daily return and volatility
            avg_return_per_cycle = np.mean(excess_returns)
            std_return_per_cycle = np.std(excess_returns)

            # Annualize metrics
            # Assuming 2-minute cycles -> 720 cycles/day
            cycles_per_day = 720

            avg_daily_return = avg_return_per_cycle * cycles_per_day
            std_daily_return = std_return_per_cycle * np.sqrt(cycles_per_day)

            if std_daily_return == 0:
                return 0.0

            # Daily Sharpe Ratio
            daily_sharpe = avg_daily_return / std_daily_return

            # Annualized Sharpe Ratio (Daily Sharpe * sqrt(365))
            annualized_sharpe = daily_sharpe * np.sqrt(365)

            return float(annualized_sharpe)

        except Exception as e:
            print(f"[WARN]  Sharpe ratio calculation error: {e}")
            return 0.0

    def get_manual_override(self) -> dict[str, Any]:
        """Check for and remove the manual override file.

        Returns
        -------
            Dictionary containing override commands if found, else empty.

        """
        path_obj = Path(self.override_file)
        override_data = safe_file_read_cached(str(path_obj), default_data={})
        if override_data:
            print(f"[ALERT] MANUAL OVERRIDE DETECTED: {override_data}")
            try:
                path_obj.unlink(missing_ok=True)
                print("[INFO]  Override file deleted.")
            except OSError as e:
                print(f"[WARN]  Could not delete override file: {e}")
        return override_data

    def _estimate_liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        direction: str,
    ) -> float:
        """Estimate the liquidation price for a position.

        Args:
        ----
            entry_price: Price at which position was entered.
            leverage: Leverage used for the position.
            direction: Trade direction ('long' or 'short').

        Returns:
        -------
            The estimated liquidation price.

        """
        if leverage <= 1 or entry_price <= 0:
            return 0.0
        imr = 1.0 / leverage
        mmr = self.maintenance_margin_rate
        margin_diff = imr - mmr
        if margin_diff <= 0:
            print(f"[WARN]  Liq est. failed: margin diff <= 0 ({margin_diff}).")
            return 0.0
        liq_price = (
            entry_price * (1 - margin_diff)
            if direction == "long"
            else entry_price * (1 + margin_diff)
        )
        return max(0.0, liq_price)

    # --- NEW: Enhanced Auto TP/SL Check with Advanced Exit Strategies ---
    def calculate_dynamic_position_size(
        self,
        coin: str,
        confidence: float,
        market_regime: str,
        trend_strength: int,
    ) -> float:
        """Calculate dynamic position size based on confidence, regime, and trend.

        Args:
        ----
            coin: The cryptocurrency symbol.
            confidence: Normalized confidence score.
            market_regime: Detected market regime string.
            trend_strength: Calculated strength of the trend.

        Returns:
        -------
            The calculated dynamic risk amount in USD.

        """
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
            indicators_3m = self.market_data.get_technical_indicators(coin, "3m")
            volume = indicators_3m.get("volume", 0)
            avg_volume = indicators_3m.get("avg_volume", 0)

            # Volume multiplier: higher volume = higher confidence
            if volume > avg_volume * 2:
                volume_multiplier = 1.2
            elif volume > avg_volume:
                volume_multiplier = 1.1
            else:
                volume_multiplier = 0.8  # Penalize low volume
        except Exception:
            # FIX: Replace bare except with specific exception handling
            volume_multiplier = 1.0

        # Dynamic risk calculation
        dynamic_risk = (
            base_risk
            * confidence_multiplier
            * regime_multiplier
            * trend_multiplier
            * volume_multiplier
        )

        # Maximum risk limit
        return min(dynamic_risk, 25.0)

    def get_max_positions_for_cycle(self, cycle_number: int) -> int:
        """Calculate gradual maximum position limit based on cycle number.

        Args:
        ----
            cycle_number: The current cycle sequence number.

        Returns:
        -------
            The maximum number of allowed concurrent positions.

        """
        from config.config import Config

        max_allowed = Config.MAX_POSITIONS

        if cycle_number == constants.ALIGNMENT_STRENGTH_L1:
            return min(1, max_allowed)  # Cycle 1: max 1 position (or MAX_POSITIONS)
        if cycle_number == constants.ALIGNMENT_STRENGTH_L2:
            return min(2, max_allowed)  # Cycle 2: max 2 positions (or MAX_POSITIONS)
        if cycle_number == constants.ALIGNMENT_STRENGTH_L3:
            return min(3, max_allowed)  # Cycle 3: max 3 positions (or MAX_POSITIONS)
        if cycle_number == constants.ALIGNMENT_STRENGTH_L4:
            return min(4, max_allowed)  # Cycle 4: max 4 positions (or MAX_POSITIONS)
        return max_allowed  # Cycle 5+: use MAX_POSITIONS value

    def _get_indicator_snapshot(
        self,
        coin: str,
        indicator_cache: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Fetch indicators for 3m and HTF from cache or market data.

        Args:
        ----
            coin: The cryptocurrency symbol.
            indicator_cache: Optional cache to pull from.

        Returns:
        -------
            A tuple of (indicators_3m, indicators_htf) dictionaries.

        """
        cache_source = (
            indicator_cache if indicator_cache is not None else getattr(self, "indicator_cache", {})
        )
        cached_entry = cache_source.get(coin) if isinstance(cache_source, dict) else None

        indicators_3m = None
        indicators_htf = None

        if isinstance(cached_entry, dict):
            indicators_3m = copy.deepcopy(cached_entry.get("3m"))
            cached_htf = cached_entry.get(HTF_INTERVAL)
            if cached_htf is None and HTF_INTERVAL != "4h":
                cached_htf = cached_entry.get("4h")  # backward compatibility
            indicators_htf = copy.deepcopy(cached_htf)

        if not isinstance(indicators_3m, dict) or "error" in indicators_3m:
            indicators_3m = self.market_data.get_technical_indicators(coin, "3m")
        if not isinstance(indicators_htf, dict) or "error" in indicators_htf:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)

        return indicators_3m, indicators_htf

    def _execute_normal_decisions(
        self,
        decisions: dict[str, Any],
        valid_prices: dict[str, float],
        cycle_number: int,
        positions_closed_by_tp_sl: bool,
        indicator_cache: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Execute normal AI decisions with gradual position limit enforcement.

        Args:
        ----
            decisions: Dictionary of coin-to-trade decisions.
            valid_prices: Dictionary of current validated prices.
            cycle_number: The current cycle sequence number.
            positions_closed_by_tp_sl: Whether any positions were closed by TP/SL
                flags during price update.
            indicator_cache: Optional cache for technical indicators.

        """
        # print("[INFO] Executing normal AI decisions (partial profit active)")

        # KADEMELI POZISYON SISTEMI: Cycle bazli pozisyon limiti
        max_positions_for_cycle = self.get_max_positions_for_cycle(cycle_number)
        current_positions = len(self.positions)

        decisions_to_execute = {}
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                continue

            signal = trade.get("signal")
            # Process based on signal type
            if signal in ["buy_to_enter", "sell_to_enter"]:
                # Apply kademeli position limit
                if current_positions >= max_positions_for_cycle:
                    print(
                        f"[WARN]  KADEMELİ POZİSYON LİMİTİ (Cycle {cycle_number}): Max {max_positions_for_cycle} positions allowed. Skipping {coin} entry.",
                    )
                    continue
                current_positions += 1

                decisions_to_execute[coin] = trade
            else:
                # Execute all other decisions (hold, close_position)
                decisions_to_execute[coin] = trade

        if decisions_to_execute:
            self.execute_decision(
                decisions_to_execute,
                valid_prices,
                indicator_cache=indicator_cache,
            )

    def _calculate_maximum_limit(self) -> float:
        """Calculate the maximum margin limit for partial sales.

        Returns
        -------
            The calculated maximum limit in USD.

        """
        max_from_percentage: float = self.current_balance * Config.MAXIMUM_LIMIT_BALANCE_PCT
        min_limit: float = Config.MIN_PARTIAL_PROFIT_MARGIN_REMAINING_USD
        max_limit: float = max(min_limit, max_from_percentage)
        print(
            f"[INFO]  Maximum limit: ${max_limit:.2f} (${min_limit} fixed vs ${max_from_percentage:.2f} {Config.MAXIMUM_LIMIT_BALANCE_PCT * 100:.1f}% of ${self.current_balance:.2f} available cash)",
        )
        return max_limit

    def _calculate_dynamic_minimum_limit(self) -> float:
        """Calculate the dynamic minimum margin limit for partial sales.

        Returns
        -------
            The calculated minimum limit in USD.

        """
        # Note: 10% is used for 'minimum' to be more conservative after partial sales
        min_from_percentage: float = self.current_balance * 0.10
        min_limit: float = Config.MIN_PARTIAL_PROFIT_MARGIN_REMAINING_USD
        return max(min_limit, min_from_percentage)

    def _adjust_partial_sale_for_max_limit(
        self,
        position: dict[str, Any],
        proposed_percent: float,
    ) -> tuple[float, bool, str | None]:
        """Adjust partial sale percentage to respect the maximum limit.

        Args:
        ----
            position: The position dictionary to evaluate.
            proposed_percent: The original proposed sale percentage (0.0 to 1.0).

        Returns:
        -------
            A tuple of (adjusted_percent, should_close_entirely, block_reason).

        """
        current_margin = position.get("margin_usd", 0)
        if current_margin <= 0:
            # Fallback: Calculate from notional/leverage if margin_usd is missing/zero
            notional = position.get("notional_usd", 0)
            leverage = position.get("leverage", 1)
            if notional > 0 and leverage > 0:
                current_margin = notional / leverage
            elif position.get("entry_price", 0) > 0 and position.get("quantity", 0) > 0:
                current_margin = (position["entry_price"] * position["quantity"]) / position.get(
                    "leverage",
                    10,
                )

        # Calculate maximum limit: $10 fixed OR 15% of available cash, whichever is larger (from Config)
        max_limit = self._calculate_maximum_limit()

        if current_margin <= max_limit:
            # Position already at or below maximum limit, don't sell - close completely
            print(
                f"🛑 Partial sale blocked: Position margin ${current_margin:.2f} <= maximum limit ${max_limit:.2f}. Position will be closed.",
            )
            return (
                0.0,
                True,
                f"Position margin ${current_margin:.2f} <= maximum limit ${max_limit:.2f}",
            )

        # Calculate remaining margin after proposed sale
        remaining_after_proposed = current_margin * (1 - proposed_percent)

        if remaining_after_proposed >= max_limit:
            # Proposed sale keeps us above maximum limit, use as-is
            return proposed_percent, False, None
        # Adjust sale to leave exactly max_limit margin
        adjusted_sale_amount = current_margin - max_limit
        # FIX: Division by zero protection
        adjusted_percent = adjusted_sale_amount / current_margin if current_margin > 0 else 0.0

        print(
            f"[INFO]  Adjusted partial sale: {proposed_percent * 100:.0f}% → {adjusted_percent * 100:.0f}% to maintain ${max_limit:.2f} maximum limit",
        )
        return adjusted_percent, False, None

    def _adjust_partial_sale_for_min_limit(
        self, position: dict[str, Any], proposed_percent: float
    ) -> float:
        """Adjust partial sale percentage to respect the minimum margin limit.

        Args:
        ----
            position: The position dictionary.
            proposed_percent: The original proposed sale percentage.

        Returns:
        -------
            The adjusted sale percentage.

        """
        current_margin = position.get("margin_usd", 0)
        if current_margin <= 0:
            # Fallback: Calculate from notional/leverage if margin_usd is missing/zero
            notional = position.get("notional_usd", 0)
            leverage = position.get("leverage", 1)
            if notional > 0 and leverage > 0:
                current_margin = notional / leverage
            elif position.get("entry_price", 0) > 0 and position.get("quantity", 0) > 0:
                current_margin = (position["entry_price"] * position["quantity"]) / position.get(
                    "leverage",
                    10,
                )

        # Calculate dynamic minimum limit: $10 fixed OR 10% of available cash, whichever is larger (from Config)
        min_remaining = self._calculate_dynamic_minimum_limit()

        if current_margin <= min_remaining:
            # Position already at or below minimum, don't sell
            print(
                f"🛑 Partial sale blocked: Position margin ${current_margin:.2f} <= minimum limit ${min_remaining:.2f}",
            )
            return 0.0

        # Calculate remaining margin after proposed sale
        remaining_after_proposed = current_margin * (1 - proposed_percent)

        if remaining_after_proposed >= min_remaining:
            # Proposed sale keeps us above minimum, use as-is
            return proposed_percent
        # Adjust sale to leave exactly min_remaining margin
        adjusted_sale_amount = current_margin - min_remaining
        # FIX: Division by zero protection
        adjusted_percent = adjusted_sale_amount / current_margin if current_margin > 0 else 0.0

        print(
            f"[INFO]  Adjusted partial sale: {proposed_percent * 100:.0f}% → {adjusted_percent * 100:.0f}% to maintain ${min_remaining:.2f} minimum limit",
        )
        return adjusted_percent

    def _is_counter_trend_trade(
        self,
        coin: str,
        signal: str,
        indicators_3m: dict[str, Any],
        indicators_htf: dict[str, Any],
    ) -> bool:
        """Determine if a prospective trade aligns with or counters the HTF trend.

        Args:
        ----
            coin: The cryptocurrency symbol.
            signal: The trade signal type.
            indicators_3m: 3m technical indicators.
            indicators_htf: HTF technical indicators.

        Returns:
        -------
            True if the trade is considered counter-trend, False otherwise.

        """
        try:
            if "error" in indicators_3m or "error" in indicators_htf:
                return False

            indicators_15m = self.market_data.get_technical_indicators(coin, "15m")
            if "error" in indicators_15m:
                indicators_15m = None

            t_htf = self._calculate_trend_direction(
                indicators_htf.get("current_price"), indicators_htf.get("ema_20")
            )
            t_3m = self._calculate_trend_direction(
                indicators_3m.get("current_price"), indicators_3m.get("ema_20")
            )
            t_15m = None
            if indicators_15m:
                t_15m = self._calculate_trend_direction(
                    indicators_15m.get("current_price"), indicators_15m.get("ema_20")
                )

            sig_dir = "BULLISH" if signal == "buy_to_enter" else "BEARISH"
            return self._evaluate_counter_trend_alignment(signal, sig_dir, t_htf, t_15m, t_3m)

        except Exception as e:
            print(f"[WARN]  Counter-trend detection error for {coin}: {e}")
            return False

    def _calculate_trend_direction(self, price: float, ema20: float) -> str:
        """Determine trend direction relative to EMA with a neutral band.

        Args:
        ----
            price: Current asset price.
            ema20: The EMA 20 value.

        Returns:
        -------
            String representing the trend direction ('BULLISH', 'BEARISH', or 'NEUTRAL').

        """
        from config.config import Config

        if not isinstance(price, (int, float)) or not isinstance(ema20, (int, float)) or ema20 == 0:
            return "NEUTRAL"
        delta = (price - ema20) / ema20
        if abs(delta) <= Config.EMA_NEUTRAL_BAND_PCT:
            return "NEUTRAL"
        return "BULLISH" if delta > 0 else "BEARISH"

    def _evaluate_counter_trend_alignment(
        self,
        signal: str,
        sig_dir: str,
        t_htf: str,
        t_15m: str | None,
        t_3m: str,
    ) -> bool:
        """Evaluate if a counter-trend signal has sufficient timeframe alignment.

        Args:
        ----
            signal: The original signal string.
            sig_dir: The normalized signal direction.
            t_htf: The HTF trend direction.
            t_15m: The 15m trend direction (if available).
            t_3m: The 3m trend direction.

        Returns:
        -------
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

    def calculate_confidence_based_margin(
        self,
        margin_ctx: dict[str, Any],
    ) -> float:
        """Calculate margin based on volatility sizing (fixed risk) and confidence.

        Args:
        ----
            margin_ctx: Context containing confidence, balance, and price data.

        Returns:
        -------
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
            margin = available_cash * 0.40 * confidence
            margin = max(margin, Config.MIN_POSITION_MARGIN_USD)
            print(f"[INFO]  Standard Sizing (Missing Data): ${margin:.2f} (conf: {confidence:.2f})")
            return margin

        # 2. Calculate Stop Distance %
        dist_pct = abs(entry_price - stop_loss) / entry_price

        # Safety: Avoid division by zero or extremely small stops (<0.2%)
        dist_pct = max(dist_pct, 0.002)  # Min 0.2% stop distance assumption

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
        max_margin_cash = available_cash * 0.40
        if target_margin > max_margin_cash:
            if log_func:
                log_func(
                    "sizing",
                    f"[WARN]  Volatility sizing capped by cash limit: ${target_margin:.2f} -> ${max_margin_cash:.2f}",
                    {
                        "coin": self.market_data.available_coins[0]
                        if hasattr(self, "market_data")
                        else "unknown",
                        "target_margin": target_margin,
                        "max_margin": max_margin_cash,
                    },
                )
            else:
                print(
                    f"[WARN]  Volatility sizing capped by cash limit: ${target_margin:.2f} -> ${max_margin_cash:.2f}"
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
            print(msg)
        return target_margin

    def get_graduated_loss_multiplier(self, margin_usd: float) -> float:
        """Calculate the loss multiplier based on position margin size.

        Args:
        ----
            margin_usd: The position margin in USD.

        Returns:
        -------
            The graduated loss multiplier percentage.

        """
        from config.config import Config

        if margin_usd < constants.MARGIN_TIER_20:
            return Config.LOSS_MULT_L1  # %20 for margin < 20 (Allows tight stops)
        if margin_usd < constants.MARGIN_TIER_30:
            return Config.LOSS_MULT_L2  # %15 for margin 20-30
        if margin_usd < constants.MARGIN_TIER_40:
            return Config.LOSS_MULT_L3  # %12 for margin 30-40
        if margin_usd < constants.MARGIN_TIER_50:
            return Config.LOSS_MULT_L4  # %10 for margin 40-50
        return Config.LOSS_MULT_BASE  # %8 for margin >= 50

    def calculate_volume_quality_score(
        self,
        coin: str,
        indicators_3m: dict[str, Any] | None = None,
    ) -> float:
        """Calculate volume quality score based on current vs average volume.

        Args:
        ----
            coin: The cryptocurrency symbol.
            indicators_3m: Optional technical indicators for 3m timeframe.

        Returns:
        -------
            A score from 0.0 to 100.0 representing volume quality.

        """
        score = 0.0
        try:
            if indicators_3m is None or not isinstance(indicators_3m, dict):
                indicators_3m = self.market_data.get_technical_indicators(coin, "3m")
            if "error" not in indicators_3m:
                v_rat = indicators_3m.get("volume_ratio")
                if v_rat is None:
                    curr, avg = indicators_3m.get("volume", 0), indicators_3m.get("avg_volume", 0)
                    v_rat = curr / avg if avg > 0 else 0.0
                # Check thresholds
                t = Config.VOLUME_QUALITY_THRESHOLDS
                if v_rat >= t["excellent"]:
                    score = 90.0
                elif v_rat >= t["good"]:
                    score = 75.0
                elif v_rat >= t["fair"]:
                    score = 60.0
                elif v_rat >= t["poor"]:
                    score = 40.0
                else:
                    score = 20.0
        except Exception as e:
            print(f"[WARN]  Volume quality score calculation error for {coin}: {e}")
        return score

    def check_flash_exit_conditions(self, coin: str, position: dict[str, Any]) -> bool:
        """Check for flash exit conditions including price, RSI, and volume surge.

        Args:
        ----
            coin: The cryptocurrency symbol.
            position: The position dictionary to evaluate.

        Returns:
        -------
            True if flash exit conditions are met, False otherwise.

        """
        should_flash_exit = False
        if not Config.FLASH_EXIT_ENABLED:
            return should_flash_exit

        try:
            inds = self.market_data.get_technical_indicators(coin, "3m")
            if "error" not in inds:
                direction = position.get("direction")
                entry = position.get("entry_price")
                price = inds.get("current_price")
                # 1. Price Check
                trig = Config.FLASH_EXIT_LOSS_TRIGGER_MULTIPLIER
                p_ok = (direction == "short" and price > entry * trig) or (
                    direction == "long" and price < entry * (2 - trig)
                )
                if p_ok:
                    # 2. RSI Spike Check
                    rsi_c = inds.get("rsi_14", 50)
                    rsi_s = inds.get("rsi_14_series", [])
                    if len(rsi_s) >= constants.MAX_POSITIONS_DIVERSITY:
                        rsi_p = rsi_s[-constants.MAX_POSITIONS_DIVERSITY]
                        rsi_diff = abs(rsi_c - rsi_p)
                        if rsi_diff >= Config.FLASH_EXIT_RSI_SPIKE_THRESHOLD:
                            # 3. Volume Check
                            v_rat = inds.get("volume_ratio", 1.0)
                            if v_rat >= Config.FLASH_EXIT_VOLUME_SURGE_THRESHOLD:
                                should_flash_exit = True
        except Exception as e:
            print(f"[WARN] Flash exit check error for {coin}: {e}")
        return should_flash_exit

    def get_effective_same_direction_limit(self) -> int:
        """Calculate the effective same direction position limit.

        Returns
        -------
            The maximum number of allowed concurrent positions in the same direction.

        """
        # FIX: Use getattr for safe config access
        return getattr(Config, "SAME_DIRECTION_LIMIT", 2)

    def validate_exit_signal(
        self, coin: str, position: dict[str, Any], indicators_3m: dict[str, Any]
    ) -> bool:
        """Validate if an exit signal should be executed based on PnL and technicals.

        Args:
        ----
            coin: The cryptocurrency symbol.
            position: The position dictionary.
            indicators_3m: 3m technical indicators.

        Returns:
        -------
            True if exit is validated, False to block the exit.

        """
        try:
            # 1. PnL Check (Profit Protection / Stop Loss)
            current_price = indicators_3m.get("current_price")
            entry_price = position.get("entry_price")
            quantity = position.get("quantity")
            direction = position.get("direction")

            if not all(isinstance(x, (int, float)) for x in [current_price, entry_price, quantity]):
                return True  # Default to allow exit if data missing

            pnl_usd = (
                (current_price - entry_price) * quantity
                if direction == "long"
                else (entry_price - current_price) * quantity
            )
            margin = position.get("margin_usd", 1.0)
            pnl_pct = (pnl_usd / margin) * 100 if margin > 0 else 0

            # Allow exit if PnL is good (>2%) or bad (<-1.5%)
            if pnl_pct > constants.PNL_PCT_EXIT_PROFIT or pnl_pct < constants.PNL_PCT_EXIT_LOSS:
                print(f"[OK]    Exit validated by PnL: {pnl_pct:.2f}%")
                return True

            # 2. 15m Confirmation
            indicators_15m = self.market_data.get_technical_indicators(coin, "15m")
            if "error" not in indicators_15m:
                price_15m = indicators_15m.get("current_price")
                ema20_15m = indicators_15m.get("ema_20")
                # Check if 15m trend opposes position
                is_15m_reversal = (direction == "long" and price_15m < ema20_15m) or (
                    direction == "short" and price_15m > ema20_15m
                )
                if is_15m_reversal:
                    print("[OK]    Exit validated by 15m reversal")
                    return True

            # 3. Strong 3m Reversal (Check MACD/RSI intensity)
            # This is a heuristic since we don't have explicit "strength" flag from AI here easily
            # We assume if AI called close, it saw a reversal. We check if it's "strong".
            # Strong = Price crossed EMA20 AND (RSI extreme or MACD cross)
            price_3m = indicators_3m.get("current_price")
            ema20_3m = indicators_3m.get("ema_20")
            rsi_3m = indicators_3m.get("rsi_14", 50)

            is_price_crossed = (direction == "long" and price_3m < ema20_3m) or (
                direction == "short" and price_3m > ema20_3m
            )

            is_rsi_extreme = (direction == "long" and rsi_3m > constants.RSI_OVERBOUGHT) or (
                direction == "short" and rsi_3m < constants.RSI_OVERSOLD
            )  # Overbought/sold reversal

            if is_price_crossed and is_rsi_extreme:
                print("[OK]    Exit validated by Strong 3m Reversal (Price+RSI)")
                return True

            print(
                f"[BLOCK] Exit blocked: Weak 3m reversal without confirmation (PnL: {pnl_pct:.2f}%)"
            )
            return False
        except Exception as e:
            print(f"[WARN]  Exit validation error: {e}")
            return True  # Fail safe: allow exit

    def _verify_technical_reversal(self, coin: str, direction: str) -> bool:
        """Verify if technical indicators support a reversal claim to prevent premature exit.

        Args:
        ----
            coin: The cryptocurrency symbol.
            direction: The position direction ('long' or 'short').

        Returns:
        -------
            True if technical indicators support the reversal, False otherwise.

        """
        try:
            indicators_3m = self.market_data.get_technical_indicators(coin, "3m")
            if "error" in indicators_3m:
                return True  # Fail-safe: allow exit if data is missing

            price_3m = indicators_3m.get("current_price")
            ema20_3m = indicators_3m.get("ema_20")

            if not price_3m or not ema20_3m:
                return True

            # Threshold for strict validation (0.1% buffer)
            if direction == "long":
                # Long is still valid if price is not significantly below EMA20
                is_invalid = price_3m < (ema20_3m * 0.999)
            else:
                # Short is still valid if price is not significantly above EMA20
                is_invalid = price_3m > (ema20_3m * 1.001)

            return is_invalid
        except Exception as e:
            print(f"[WARN] Error in _verify_technical_reversal: {e}")
            return True

    def _get_signal_priority(self, item: tuple[str, Any]) -> tuple[int, float]:
        """Determine the execution priority for a trade signal based on confidence.

        Args:
        ----
            item: A tuple of (coin, trade_decision).

        Returns:
        -------
            A tuple of (priority_index, negative_confidence) for sorting.

        """
        _coin, trade = item
        if not isinstance(trade, dict):
            return (2, 0)  # Invalid trades last

        signal = trade.get("signal", "")
        confidence = trade.get("confidence", 0)

        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0

        if signal in ["buy_to_enter", "sell_to_enter"]:
            return (0, -confidence)  # Entry signals first, sorted by confidence desc
        if signal == "close_position":
            return (1, 0)  # Close signals after entries
        return (2, 0)  # Hold signals last

    def _prepare_execution_context(self) -> dict[str, Any]:
        """Initialize the execution context including regime detection and report setup.

        Returns
        -------
            Dictionary containing 'report', 'market_regime', and 'regime_strength'.

        """
        if self.is_live_trading:
            self.sync_live_account()

        # Phase 7: Centralized Regime Detection
        coin_indicators = {}
        for coin in self.market_data.available_coins:
            indicators_htf = self.market_data.get_technical_indicators(coin, HTF_INTERVAL)
            if "error" not in indicators_htf:
                coin_indicators[coin] = indicators_htf

        market_regime = RegimeDetector.detect_overall_regime(coin_indicators)
        regime_strength = RegimeDetector.calculate_regime_strength(coin_indicators)
        print(f"[CYCLE] Market Regime: {market_regime} | Strength: {regime_strength:.2f}")

        return {
            "report": {
                "executed": [],
                "blocked": [],
                "skipped": [],
                "holds": [],
                "notes": [],
                "debug_logs": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "market_regime": market_regime,
            "regime_strength": regime_strength,
        }

    def _handle_exit_signal_logic(
        self,
        coin: str,
        trade: dict[str, Any],
        current_price: float,
        position: dict[str, Any] | None,
        execution_report: dict[str, Any],
    ) -> None:
        """Process an exit signal including verification and execution steps.

        Args:
        ----
            coin: The cryptocurrency symbol.
            trade: The trade decision from AI.
            current_price: The current market price.
            position: The existing position dictionary, if any.
            execution_report: The report dictionary to update.

        """
        if not position:
            print(f"[WARN]  CLOSE {coin}: No position to close.")
            execution_report["skipped"].append(
                {"coin": coin, "reason": "no_position_to_close"},
            )
            trade["runtime_decision"] = "skipped_no_position"
            return

        justification = trade.get("justification", "")
        if "reversal" in justification or "invalidation" in justification:
            # Treat Take Profit, Stop Loss or PnL justifications as hard exits
            is_pnl_exit = any(
                word in justification.lower() for word in ["take profit", "stop loss", "pnl"]
            )

            if not is_pnl_exit:
                # 5.1 Surgical Invalidation Check
                direction = position.get("direction", "long")
                strong_reversal = self._verify_technical_reversal(coin, direction)

                if not strong_reversal:
                    print(
                        f"[BLOCK] Exit Desensitization: {coin} close blocked. Technicals still support {direction.upper()}."
                    )
                    execution_report["holds"].append(
                        {
                            "coin": coin,
                            "reason": "exit_desensitization_blocked",
                            "justification": justification,
                        },
                    )
                    trade["runtime_decision"] = "blocked_exit_desensitization"
                    return

        if self.is_live_trading:
            live_result = self.execute_live_close(
                coin=coin,
                position=position,
                current_price=current_price,
                reason=trade.get("justification"),
            )
            if not live_result.get("success"):
                error_msg = live_result.get("error", "unknown_error")
                print(f"[BLOCK] LIVE CLOSE FAILED: {coin} ({error_msg})")
                execution_report["blocked"].append(
                    {"coin": coin, "reason": "live_close_failed", "error": error_msg},
                )
                trade["runtime_decision"] = "blocked_live_close"
            else:
                history_entry = live_result.get("history_entry")
                if history_entry:
                    self.add_to_history(history_entry)
                execution_report["executed"].append(
                    {
                        "coin": coin,
                        "signal": "close_position",
                        "pnl": live_result.get("pnl"),
                        "direction": position.get("direction"),
                        "mode": "live",
                        "order_id": live_result.get("order", {}).get("orderId"),
                    },
                )
                trade["runtime_decision"] = "executed_live"
            return

        sell_quantity = position["quantity"]
        direction = position.get("direction", "long")
        entry_price = position["entry_price"]
        margin_used = position.get(
            "margin_usd",
            position.get("notional_usd", 0) / position.get("leverage", 1),
        )

        profit = (
            (current_price - entry_price) * sell_quantity
            if direction == "long"
            else (entry_price - current_price) * sell_quantity
        )
        if not self.is_live_trading:
            self.current_balance += margin_used + profit

        print(
            f"[OK]    CLOSE (AI): Closed {direction} {coin} @ ${format_num(current_price, 4)} (PnL: ${format_num(profit, 2)})",
        )
        execution_report["executed"].append(
            {
                "coin": coin,
                "signal": "close_position",
                "pnl": profit,
                "direction": direction,
            },
        )
        trade["runtime_decision"] = "executed"

        history_entry = {
            "symbol": coin,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": current_price,
            "quantity": position["quantity"],
            "notional_usd": position.get("notional_usd", "N/A"),
            "pnl": profit,
            "entry_time": position["entry_time"],
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "leverage": position.get("leverage", "N/A"),
            "close_reason": f"AI Decision: {trade.get('justification', 'N/A')}",
        }
        self.add_to_history(history_entry)
        del self.positions[coin]

    def _check_entry_preconditions(
        self,
        signal_ctx: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate if a new entry signal should proceed based on limits and state.

        Args:
        ----
            signal_ctx: Context containing coin, trade, and execution state.

        Returns:
        -------
            Dictionary with 'proceed', 'confidence', and 'leverage'.

        """
        # 1. Basic checks (Existing position & Same-cycle limit)
        if not self._check_basic_preconditions(signal_ctx):
            return {"proceed": False}

        # 2. Cooldowns (Coin & Directional)
        if not self._check_cooldown_preconditions(signal_ctx):
            return {"proceed": False}

        # 3. Diversity (Slot Limits)
        if not self._check_diversity_preconditions(signal_ctx):
            return {"proceed": False}

        # 4. Confidence & Leverage Validation
        return self._validate_and_clamp_leverage(signal_ctx)

    def _check_basic_preconditions(self, signal_ctx: dict[str, Any]) -> bool:
        """Check for existing positions and same-cycle entry limits.

        Args:
        ----
            signal_ctx: Context containing coin, trade, and execution state.

        Returns:
        -------
            True if basic preconditions are met, False otherwise.

        """
        from config.config import Config

        coin = signal_ctx["coin"]
        signal = signal_ctx["signal"]
        position = signal_ctx.get("position")
        new_positions_this_cycle = signal_ctx["new_positions_this_cycle"]
        execution_report = signal_ctx["report"]
        trade = signal_ctx["trade"]

        if position:
            print(f"[WARN]  {signal.upper()} {coin}: Position already open.")
            execution_report["skipped"].append(
                {"coin": coin, "reason": "position_exists", "signal": signal},
            )
            trade["runtime_decision"] = "skipped_existing_position"
            return False

        if new_positions_this_cycle >= Config.MAX_NEW_POSITIONS_PER_CYCLE:
            signal_ctx["log_func"](
                "block",
                f"[BLOCK] Same-cycle limit ({Config.MAX_NEW_POSITIONS_PER_CYCLE}): Blocking {coin} {signal}",
                {
                    "coin": coin,
                    "signal": signal,
                    "limit": Config.MAX_NEW_POSITIONS_PER_CYCLE,
                    "opened": new_positions_this_cycle,
                },
            )
            execution_report["blocked"].append(
                {"coin": coin, "reason": "same_cycle_limit", "signal": signal},
            )
            trade["runtime_decision"] = "blocked_same_cycle_limit"
            return False

        return True

    def _check_cooldown_preconditions(self, signal_ctx: dict[str, Any]) -> bool:
        """Check for active coin or directional cooldown restrictions.

        Args:
        ----
            signal_ctx: Context containing coin, trade, and execution state.

        Returns:
        -------
            True if no cooldowns are active, False otherwise.

        """
        coin = signal_ctx["coin"]
        direction = signal_ctx["direction"]
        execution_report = signal_ctx["report"]
        trade = signal_ctx["trade"]

        coin_cooldowns = self.coin_cooldowns
        coin_upper = coin.upper()
        coin_rem = coin_cooldowns.get(coin_upper, 0)
        if coin_rem > 0:
            print(
                f"[PAUSED] Coin cooldown active: Blocking {coin} entry ({coin_rem} cycles remaining - previous loss)."
            )
            execution_report["blocked"].append(
                {"coin": coin, "reason": "coin_cooldown", "cooldown_remaining": coin_rem}
            )
            trade["runtime_decision"] = "blocked_coin_cooldown"
            return False

        directional_cooldowns = self.directional_cooldowns
        dir_rem = directional_cooldowns.get(direction, 0)
        if dir_rem > 0:
            print(
                f"[PAUSED] Directional cooldown active: Blocking {direction.upper()} entry for {coin} ({dir_rem} cycles remaining)."
            )
            execution_report["blocked"].append(
                {
                    "coin": coin,
                    "reason": "directional_cooldown",
                    "direction": direction,
                    "cooldown_remaining": dir_rem,
                }
            )
            trade["runtime_decision"] = "blocked_directional_cooldown"
            return False

        return True

    def _check_diversity_preconditions(self, signal_ctx: dict[str, Any]) -> bool:
        """Check for directional slot limits and regime-based capacity.

        Args:
        ----
            signal_ctx: Context containing coin, trade, and execution state.

        Returns:
        -------
            True if diversity limits allow the trade, False otherwise.

        """
        from src.core import constants

        from config.config import Config

        coin = signal_ctx["coin"]
        signal = signal_ctx["signal"]
        direction = signal_ctx["direction"]
        regime_strength = signal_ctx["regime_strength"]
        execution_report = signal_ctx["report"]
        trade = signal_ctx["trade"]

        directional_counts = self.count_positions_by_direction()
        current_same_direction = directional_counts.get(direction, 0)
        effective_limit = Config.SAME_DIRECTION_LIMIT
        if regime_strength > constants.REGIME_STRENGTH_THRESHOLD:
            effective_limit = Config.DYNAMIC_DIRECTION_LIMIT
            print(
                f"🌊 Dynamic Slot Limit Active: Strength {regime_strength:.2f} > 0.7 -> Limit increased to {effective_limit}"
            )

        if current_same_direction >= effective_limit:
            print(
                f"[BLOCK] SAME-DIRECTION LIMIT: {coin} {signal} blocked. {current_same_direction}/{effective_limit} {direction.upper()} positions already open."
            )
            execution_report["blocked"].append(
                {
                    "coin": coin,
                    "reason": "same_direction_limit",
                    "direction": direction,
                    "current": current_same_direction,
                    "limit": effective_limit,
                }
            )
            trade["runtime_decision"] = "blocked_same_direction_limit"
            return False

        return True

    def _validate_and_clamp_leverage(self, signal_ctx: dict[str, Any]) -> dict[str, Any]:
        """Validate and clamp confidence and leverage values to operational limits.

        Args:
        ----
            signal_ctx: Context containing coin, trade, and execution state.

        Returns:
        -------
            Dictionary with 'proceed', 'confidence', and 'leverage'.

        """
        from src.core import constants

        from config.config import Config

        coin = signal_ctx["coin"]
        trade = signal_ctx["trade"]

        confidence = trade.get("confidence", 0.5)
        leverage = trade.get("leverage")
        if leverage in (None, "", 0):
            leverage = 8

        try:
            confidence = float(confidence)
            leverage = int(leverage)
        except (ValueError, TypeError):
            print(
                f"[WARN]  Invalid confidence ({confidence}) or leverage ({leverage}) for {coin}. Skipping."
            )
            return {"proceed": False}

        leverage = max(leverage, 1)

        # Enforce limits
        if leverage > Config.MAX_LEVERAGE:
            print(
                f"[WARN]  Leverage {leverage}x exceeds maximum limit of {Config.MAX_LEVERAGE}x. Reducing to {Config.MAX_LEVERAGE}x."
            )
            leverage = Config.MAX_LEVERAGE

        # Operational Band
        if leverage < constants.LEVERAGE_MIN_OP:
            print(
                f"[INFO]  Adjusting leverage from {leverage}x to minimum operational level {constants.LEVERAGE_MIN_OP}x for {coin}."
            )
            leverage = constants.LEVERAGE_MIN_OP
        elif leverage > constants.LEVERAGE_MAX_OP:
            print(
                f"[INFO]  Adjusting leverage from {leverage}x to maximum operational level {constants.LEVERAGE_MAX_OP}x for {coin}."
            )
            leverage = constants.LEVERAGE_MAX_OP

        if not (0 <= confidence <= 1):
            confidence = 0.5

        return {"proceed": True, "confidence": confidence, "leverage": leverage}

    def _apply_technical_confidence_adjustments(
        self,
        signal_ctx: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply commission guard and technical indicator adjustments to confidence.

        Args:
        ----
            signal_ctx: Context containing coin, trade, and market indicators.

        Returns:
        -------
            Dictionary with 'proceed' status and the adjusted 'confidence'.

        """
        coin = signal_ctx["coin"]
        signal = signal_ctx["signal"]
        confidence = signal_ctx.get("confidence", 0.5)
        indicators_htf = signal_ctx.get("indicators_htf", {})
        indicators_3m = signal_ctx.get("indicators_3m", {})
        execution_report = signal_ctx["report"]
        trade = signal_ctx["trade"]
        log_func = signal_ctx["log_func"]
        current_price = signal_ctx["current_price"]
        # Import Config and constants inside to avoid scoping issues
        from config.config import Config

        # 1. Commission Guard (v1.2)
        try:
            atr_val = indicators_htf.get("atr_14", 0)
            if atr_val > 0:
                tp_mult = getattr(Config, "ATR_PROFIT_MULTIPLIER", 3.0)
                expected_profit_usd_per_coin = atr_val * tp_mult
                expected_profit_pct = expected_profit_usd_per_coin / current_price

                comm_rate = getattr(Config, "SIMULATION_COMMISSION_RATE", 0.001)
                round_trip_comm_pct = comm_rate * 2

                profit_comm_ratio = expected_profit_pct / round_trip_comm_pct
                guard_ratio = getattr(Config, "COMMISSION_GUARD_RATIO", 5.0)

                if profit_comm_ratio < guard_ratio:
                    log_func(
                        "trade",
                        f"[BLOCK] COMMISSION GUARD: {coin} Ratio {profit_comm_ratio:.2f} < {guard_ratio:.2f}. Trade not economically viable.",
                        {"coin": coin, "ratio": profit_comm_ratio},
                    )
                    execution_report["blocked"].append(
                        {
                            "coin": coin,
                            "reason": "commission_guard",
                            "ratio": profit_comm_ratio,
                        },
                    )
                    trade["runtime_decision"] = "blocked_commission_guard"
                    return {"proceed": False}

                log_func(
                    "trade",
                    f"[OK]    COMMISSION GUARD: {coin} Ratio {profit_comm_ratio:.2f} passed threshold {guard_ratio}.",
                )
        except Exception as e:
            log_func("trade", f"[WARN]  Commission guard check failed for {coin}: {e}")

        # 2. Momentum and Price Location Confidence Adjustments
        try:
            indicators_15m = (
                self.market_data.get_technical_indicators(coin, "15m") if self.market_data else {}
            )
            if isinstance(indicators_15m, dict) and "error" not in indicators_15m:
                direction = "long" if signal == "buy_to_enter" else "short"
                confidence, adjustments = self._get_momentum_and_indicator_adjustments(
                    coin, direction, confidence, indicators_15m, indicators_3m
                )

                if adjustments:
                    log_func(
                        "confidence",
                        f"[INFO] Confidence adjusted for {coin}: {' '.join(adjustments)} -> {confidence:.2f}",
                        {
                            "coin": coin,
                            "adjustments": adjustments,
                            "final_confidence": confidence,
                        },
                    )
        except (KeyError, AttributeError, ValueError):
            pass

        return {"proceed": True, "confidence": confidence}

    def _get_momentum_and_indicator_adjustments(
        self,
        coin: str,
        direction: str,
        confidence: float,
        indicators_15m: dict[str, Any],
        indicators_3m: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Calculate combined technical adjustments for trade confidence.

        Args:
        ----
            coin: The cryptocurrency symbol.
            direction: Trade direction ('long' or 'short').
            confidence: Base confidence level.
            indicators_15m: 15m technical indicators.
            indicators_3m: 3m technical indicators.

        Returns:
        -------
            A tuple of (adjusted_confidence, list_of_adjustment_notes).

        """
        adjustments = []

        # 1. Momentum & Price Location (15m)
        confidence, adjustments = self._apply_momentum_price_adjustments(
            direction, confidence, indicators_15m, adjustments
        )

        # 2. Volume & Trend Filters (3m/15m)
        confidence, adjustments = self._apply_trend_volume_adjustments(
            direction, confidence, indicators_15m, indicators_3m, adjustments
        )

        return confidence, adjustments

    def _apply_momentum_price_adjustments(
        self,
        direction: str,
        confidence: float,
        indicators_15m: dict[str, Any],
        adjustments: list[str],
    ) -> tuple[float, list[str]]:
        """Handle momentum, price zone, RSI, and VWAP adjustments to confidence.

        Args:
        ----
            direction: Trade direction ('long' or 'short').
            confidence: Current confidence level.
            indicators_15m: 15m technical indicators.
            adjustments: List of existing adjustment notes to append to.

        Returns:
        -------
            A tuple of (adjusted_confidence, updated_adjustments).

        """
        sparkline = indicators_15m.get("smart_sparkline", {})
        mom = sparkline.get("momentum", "STABLE") if isinstance(sparkline, dict) else "STABLE"
        price_loc = sparkline.get("price_location", {}) if isinstance(sparkline, dict) else {}
        zone = price_loc.get("zone", "MIDDLE")
        rsi = indicators_15m.get("rsi_14", 50)
        vwap_rel = indicators_15m.get("price_vs_vwap", "UNKNOWN")

        # 1. Momentum & RSI Zone
        mom_ctx = {
            "direction": direction,
            "mom": mom,
            "zone": zone,
            "rsi": rsi,
        }
        confidence, adjustments = self._apply_momentum_and_rsi_zone_adjustments(
            confidence, mom_ctx, adjustments
        )

        # 2. VWAP
        return self._apply_vwap_adjustments(direction, confidence, vwap_rel, adjustments)

    def _apply_momentum_and_rsi_zone_adjustments(
        self,
        confidence: float,
        mom_ctx: dict[str, Any],
        adjustments: list[str],
    ) -> tuple[float, list[str]]:
        """Handle momentum and RSI zone specific adjustments to confidence.

        Args:
        ----
            confidence: Current confidence level.
            mom_ctx: Momentum context containing direction, mom, zone, and rsi.
            adjustments: List of existing adjustment notes.

        Returns:
        -------
            A tuple of (adjusted_confidence, updated_adjustments).

        """
        direction = mom_ctx["direction"]
        mom = mom_ctx["mom"]
        zone = mom_ctx["zone"]
        rsi = mom_ctx["rsi"]
        if mom == "WEAKENING":
            confidence *= 0.90
            adjustments.append("momentum_weak(-10%)")
        elif mom == "STRENGTHENING":
            confidence *= 1.10
            adjustments.append("momentum_strong(+10%)")

        if zone == "LOWER_10" and rsi < constants.RSI_OVERSOLD and direction == "short":
            confidence *= 0.90
            adjustments.append(f"lower10_rsi{rsi:.0f}(-10%)")
        elif zone == "UPPER_10" and rsi > constants.RSI_OVERBOUGHT and direction == "long":
            confidence *= 0.90
            adjustments.append(f"upper10_rsi{rsi:.0f}(-10%)")

        if mom == "WEAKENING":
            if zone == "UPPER_10" and direction == "long":
                confidence *= 0.90
                adjustments.append("upper10_weak(-10%)")
            elif zone == "LOWER_10" and direction == "short":
                confidence *= 0.90
                adjustments.append("lower10_weak(-10%)")
        return confidence, adjustments

    def _apply_vwap_adjustments(
        self,
        direction: str,
        confidence: float,
        vwap_rel: str,
        adjustments: list[str],
    ) -> tuple[float, list[str]]:
        """Apply VWAP relationship adjustments to trade confidence.

        Args:
        ----
            direction: Trade direction ('long' or 'short').
            confidence: Current confidence level.
            vwap_rel: Price relationship to VWAP ('ABOVE' or 'BELOW').
            adjustments: List of existing adjustment notes.

        Returns:
        -------
            A tuple of (adjusted_confidence, updated_adjustments).

        """
        if direction == "long":
            if vwap_rel == "BELOW":
                confidence *= 0.95
                adjustments.append("vwap_below_long(-5%)")
            elif vwap_rel == "ABOVE":
                confidence *= 1.05
                adjustments.append("vwap_above_long(+5%)")
        elif direction == "short":
            if vwap_rel == "ABOVE":
                confidence *= 0.95
                adjustments.append("vwap_above_short(-5%)")
            elif vwap_rel == "BELOW":
                confidence *= 1.05
                adjustments.append("vwap_below_short(+5%)")
        return confidence, adjustments

    def _apply_trend_volume_adjustments(
        self,
        direction: str,
        confidence: float,
        indicators_15m: dict[str, Any],
        indicators_3m: dict[str, Any],
        adjustments: list[str],
    ) -> tuple[float, list[str]]:
        """Apply volume, ADX, and indicator alignment adjustments to confidence.

        Args:
        ----
            direction: Trade direction ('long' or 'short').
            confidence: Current confidence level.
            indicators_15m: 15m technical indicators.
            indicators_3m: 3m technical indicators.
            adjustments: List of existing adjustment notes.

        Returns:
        -------
            A tuple of (adjusted_confidence, updated_adjustments).

        """
        # 1. Volume & ADX
        confidence, adjustments = self._apply_volume_and_adx_adjustments(
            confidence, indicators_15m, indicators_3m, adjustments
        )

        # 2. Indicators alignment (BB, OBV, ST, Zone)
        return self._apply_indicator_alignment_adjustments(
            direction, confidence, indicators_15m, adjustments
        )

    def _apply_volume_and_adx_adjustments(
        self,
        confidence: float,
        indicators_15m: dict[str, Any],
        indicators_3m: dict[str, Any],
        adjustments: list[str],
    ) -> tuple[float, list[str]]:
        """Apply volume and ADX trend strength adjustments to confidence.

        Args:
        ----
            confidence: Current confidence level.
            indicators_15m: 15m technical indicators.
            indicators_3m: 3m technical indicators.
            adjustments: List of existing adjustment notes.

        Returns:
        -------
            A tuple of (adjusted_confidence, updated_adjustments).

        """
        # Volume (3m)
        v_rat = indicators_3m.get("volume_ratio", 1.0) if indicators_3m else 1.0
        if v_rat < constants.VOL_RATIO_WEAK:
            confidence *= 0.95
            adjustments.append(f"volume_weak({v_rat:.2f})(-5%)")
        elif v_rat > constants.VOL_CONF_CRITICAL:
            confidence *= 1.05
            adjustments.append(f"volume_strong({v_rat:.2f})(+5%)")

        # ADX (15m)
        t_str = indicators_15m.get("trend_strength_adx", "MODERATE")
        adx_v = indicators_15m.get("adx", 25)
        if t_str == "NO_TREND":
            confidence *= 0.80
            adjustments.append(f"adx_no_trend({adx_v:.0f})(-20%)")
        elif t_str == "WEAK":
            confidence *= 0.90
            adjustments.append(f"adx_weak({adx_v:.0f})(-10%)")
        elif t_str == "STRONG":
            confidence *= 1.05
            adjustments.append(f"adx_strong({adx_v:.0f})(+5%)")

        return confidence, adjustments

    def _apply_indicator_alignment_adjustments(
        self,
        direction: str,
        confidence: float,
        indicators_15m: dict[str, Any],
        adjustments: list[str],
    ) -> tuple[float, list[str]]:
        """Apply Bollinger, OBV, and SuperTrend alignment adjustments.

        Args:
        ----
            direction: Trade direction ('long' or 'short').
            confidence: Current confidence level.
            indicators_15m: 15m technical indicators.
            adjustments: List of existing adjustment notes.

        Returns:
        -------
            A tuple of (adjusted_confidence, updated_adjustments).

        """
        # Bollinger
        bb_sig = indicators_15m.get("bb_signal", "NORMAL")
        bb_sqz = indicators_15m.get("bb_squeeze", False)
        if bb_sig == "OVERBOUGHT" and direction == "long":
            confidence *= 0.90
            adjustments.append("bb_overbought_long(-10%)")
        elif bb_sig == "OVERSOLD" and direction == "short":
            confidence *= 0.90
            adjustments.append("bb_oversold_short(-10%)")
        if bb_sqz:
            confidence *= 0.95
            adjustments.append("bb_squeeze(-5%)")

        # OBV Divergence
        obv_div = indicators_15m.get("obv_divergence", "NONE")
        if obv_div == "BEARISH" and direction == "long":
            confidence *= 0.85
            adjustments.append("obv_bearish_div(-15%)")
        elif obv_div == "BULLISH" and direction == "short":
            confidence *= 0.85
            adjustments.append("obv_bullish_div(-15%)")

        # SuperTrend
        st_dir = indicators_15m.get("supertrend_direction", "UP")
        conf_st = (direction == "long" and st_dir == "UP") or (
            direction == "short" and st_dir == "DOWN"
        )
        against_st = (direction == "long" and st_dir == "DOWN") or (
            direction == "short" and st_dir == "UP"
        )
        if conf_st:
            confidence *= 1.05
            adjustments.append("st_confirms(+5%)")
        elif against_st:
            confidence *= 0.95
            adjustments.append("st_against(-5%)")

        # Zone Alignment Reward
        price_loc = indicators_15m.get("smart_sparkline", {}).get("price_location", {})
        zone = price_loc.get("zone", "MIDDLE")
        if direction == "long" and zone == "LOWER_10":
            confidence *= 1.05
            adjustments.append("zone_aligned_long(+5%)")
        elif direction == "short" and zone == "UPPER_10":
            confidence *= 1.05
            adjustments.append("zone_aligned_short(+5%)")

        return confidence, adjustments

    def _analyze_entry_trend_and_bias(
        self,
        signal_ctx: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze trend alignment, apply directional bias, and apply flip guard logic.

        Args:
        ----
            signal_ctx: Context for the trade signal.

        Returns:
        -------
            Dictionary with 'proceed' status and updated signal data.

        """
        coin = signal_ctx["coin"]
        signal = signal_ctx["signal"]
        indicator_cache = signal_ctx.get("indicator_cache")
        execution_report = signal_ctx["report"]
        trade = signal_ctx["trade"]
        direction = "long" if signal == "buy_to_enter" else "short"

        # 1. Snapshot
        inds_3m, inds_htf = self._get_indicator_snapshot(coin, indicator_cache)

        # Pull 15m explicitly to form the hybrid volume filter
        inds_15m = indicator_cache.get(coin, {}).get("15m") if indicator_cache else None
        if not isinstance(inds_15m, dict) or "error" in inds_15m:
            inds_15m = self.market_data.get_technical_indicators(coin, "15m")

        if ("error" in inds_3m) or ("error" in inds_htf) or ("error" in inds_15m):
            execution_report["blocked"].append({"coin": coin, "reason": "indicator_error"})
            trade["runtime_decision"] = "blocked_indicator_error"
            return {"proceed": False}

        # 2. Volume Analysis (Hybrid 3m/15m)
        proceed, conf = self._handle_volume_analysis_and_penalty(signal_ctx, inds_3m, inds_15m)
        signal_ctx["confidence"] = conf
        if not proceed:
            return {"proceed": False}

        # 3. Update Trend
        trend_info = self.update_trend_state(coin, inds_htf, inds_3m)
        curr_trend = trend_info.get("trend", "unknown")
        trade["trend_runtime"] = curr_trend

        # 4. Classification & Bias
        is_ct = self._is_counter_trend_trade(coin, signal, inds_3m, inds_htf)
        classification = "counter_trend" if is_ct else "trend_following"
        # Clash/Bias refinement
        conf = self._handle_trend_clash_and_bias(signal_ctx, curr_trend)
        trade["classification"] = classification
        trade["trend_alignment"] = classification

        # 5. Informational logging
        if not is_ct:
            self._log_trend_alignment_info(coin, signal, inds_htf, inds_3m, signal_ctx["log_func"])

        self._log_execution_snapshot(signal_ctx, inds_htf, inds_3m, curr_trend, is_ct)

        # 6. Trend Flip Guard Context
        guard_ctx = {
            "coin": coin,
            "signal": signal,
            "direction": direction,
            "confidence": conf,
            "partial_margin_factor": 1.0,
            "is_counter_trend": is_ct,
            "guard_cycles_since_flip": max(
                0, self.current_cycle_number - trend_info.get("last_flip_cycle", 0)
            )
            if isinstance(trend_info.get("last_flip_cycle"), int)
            else None,
            "last_flip_direction": trend_info.get("last_flip_direction"),
            "relax_mode_active": getattr(self, "relaxed_countertrend_cycles", 0) > 0,
            "report": execution_report,
            "trade": trade,
            "log_func": signal_ctx["log_func"],
        }
        proceed, final_conf, final_pmf = self._handle_trend_flip_guard(guard_ctx)
        return {
            "proceed": proceed,
            "confidence": final_conf,
            "partial_margin_factor": final_pmf,
            "is_counter_trend": is_ct,
            "classification": classification,
        }

    def _log_trend_alignment_info(
        self,
        coin: str,
        signal: str,
        inds_htf: dict[str, Any],
        inds_3m: dict[str, Any],
        log_func: Callable[[str, str, dict[str, Any] | None], None],
    ) -> None:
        """Log detailed trend alignment information and fallbacks.

        Args:
        ----
            coin: The cryptocurrency symbol.
            signal: The original signal string.
            inds_htf: HTF technical indicators.
            inds_3m: 3m technical indicators.
            log_func: Logging function to use.

        """
        strength_result = self.get_trend_following_strength(coin, signal)
        if strength_result and strength_result.get("strength"):
            log_func(
                "trend",
                f"[OK] TREND-FOLLOWING ({strength_result['strength']}): {coin} {strength_result['alignment_info']}",
                {"coin": coin, "strength": strength_result["strength"]},
            )
        else:
            # Fallback old-style check
            p1h, e1h = inds_htf.get("current_price"), inds_htf.get("ema_20")
            p3m, e3m = inds_3m.get("current_price"), inds_3m.get("ema_20")
            aligned = False
            if all(isinstance(x, (int, float)) for x in [p1h, e1h, p3m, e3m]):
                aligned = (signal == "buy_to_enter" and p1h >= e1h and p3m >= e3m) or (
                    signal == "sell_to_enter" and p1h <= e1h and p3m <= e3m
                )
            if aligned:
                print(f"[OK] TREND-FOLLOWING: {coin} aligns with trend direction")
            else:
                print(f"[OK] TREND-FOLLOWING: {coin} aligns with trend direction (fallback)")

    def _handle_volume_analysis_and_penalty(
        self,
        signal_ctx: dict[str, Any],
        indicators_3m_snap: dict[str, Any],
        indicators_15m_snap: dict[str, Any] | None = None,
    ) -> tuple[bool, float]:
        """Perform volume quality analysis and apply confidence penalties if weak.

        Args:
        ----
            signal_ctx: Context for the trade signal.
            indicators_3m_snap: Snapshot of 3m technical indicators.

        Returns:
        -------
            A tuple of (proceed_boolean, final_confidence_float).

        """
        """Handles volume quality scoring and filters."""
        from config.config import Config

        coin = signal_ctx["coin"]
        confidence = signal_ctx["confidence"]
        execution_report = signal_ctx["report"]
        trade = signal_ctx["trade"]
        log_func = signal_ctx["log_func"]

        # Volume quality scoring
        volume_quality_score = self.calculate_volume_quality_score(
            coin, indicators_3m=indicators_3m_snap
        )
        confidence = min(1.0, confidence + (volume_quality_score / 1000))
        trade["volume_quality_score"] = volume_quality_score

        current_volume = indicators_3m_snap.get("volume", 0)
        avg_volume = indicators_3m_snap.get("avg_volume", 1)
        volume_ratio_3m = indicators_3m_snap.get(
            "volume_ratio", (current_volume / avg_volume if avg_volume > 0 else 0.0)
        )
        trade["volume_ratio_runtime"] = round(volume_ratio_3m, 4)

        volume_ratio_15m = 0.0
        if indicators_15m_snap:
            vol_15m = indicators_15m_snap.get("volume", 0)
            avg_vol_15m = indicators_15m_snap.get("avg_volume", 1)
            volume_ratio_15m = indicators_15m_snap.get(
                "volume_ratio", (vol_15m / avg_vol_15m if avg_vol_15m > 0 else 0.0)
            )
            trade["volume_ratio_15m_runtime"] = round(volume_ratio_15m, 4)

        # Average Volume Filter (HYBRID: 3m & 15m Average)
        volume_threshold = Config.VOLUME_MINIMUM_THRESHOLD
        avg_volume_ratio = (volume_ratio_3m + volume_ratio_15m) / 2
        has_existing_position = coin in self.positions

        if avg_volume_ratio < volume_threshold and not has_existing_position:
            log_func(
                "block",
                f"[BLOCK] AVERAGE VOLUME FILTER: {coin} Avg ratio {avg_volume_ratio:.2f} (3m:{volume_ratio_3m:.2f}, 15m:{volume_ratio_15m:.2f}) < {volume_threshold}. Trade blocked.",
                {
                    "coin": coin,
                    "avg_volume_ratio": avg_volume_ratio,
                    "volume_ratio_3m": volume_ratio_3m,
                    "volume_ratio_15m": volume_ratio_15m,
                },
            )
            execution_report["blocked"].append(
                {
                    "coin": coin,
                    "reason": "average_volume_filter",
                    "avg_ratio": avg_volume_ratio,
                    "volume_ratio_3m": volume_ratio_3m,
                    "volume_ratio_15m": volume_ratio_15m,
                }
            )
            trade["runtime_decision"] = "blocked_hybrid_volume_filter"
            return False, confidence

        # Low Volume Penalty (Applies only if BOTH are below threshold)
        relax_mode_active = getattr(self, "relaxed_countertrend_cycles", 0) > 0
        low_vol_threshold = 0.20 if not relax_mode_active else 0.15

        if (
            volume_ratio_3m < low_vol_threshold
            and volume_ratio_15m < low_vol_threshold
            and not relax_mode_active
        ):
            original_conf = confidence
            confidence = max(confidence * 0.92, confidence - 0.05)
            min_floor = original_conf * 0.85
            confidence = max(confidence, min_floor, Config.MIN_CONFIDENCE)
            log_func(
                "volume",
                f"[INFO] LOW VOLUME PENALTY (HYBRID): {coin} 3m ratio {volume_ratio_3m:.2f}x, 15m ratio {volume_ratio_15m:.2f}x. Confidence {original_conf:.2f} -> {confidence:.2f}",
                {
                    "coin": coin,
                    "volume_ratio_3m": volume_ratio_3m,
                    "volume_ratio_15m": volume_ratio_15m,
                    "original_confidence": original_conf,
                    "adjusted_confidence": confidence,
                },
            )

            if confidence < Config.MIN_CONFIDENCE:
                execution_report["blocked"].append(
                    {
                        "coin": coin,
                        "reason": "low_volume_hybrid",
                        "volume_ratio_3m": volume_ratio_3m,
                        "volume_ratio_15m": volume_ratio_15m,
                        "confidence": confidence,
                    }
                )
                trade["runtime_decision"] = "blocked_low_volume_hybrid"
                return False, confidence

        trade["confidence"] = confidence
        return True, confidence

    def _handle_trend_clash_and_bias(
        self,
        signal_ctx: dict[str, Any],
        current_trend: str,
    ) -> float:
        """Handle detection of AI/Runtime trend clashes and apply directional bias.

        Args:
        ----
            signal_ctx: Context for the trade signal.
            current_trend: The current detected runtime trend.

        Returns:
        -------
            The adjusted confidence level.

        """
        coin = signal_ctx["coin"]
        signal = signal_ctx["signal"]
        direction = signal_ctx["direction"]
        confidence = signal_ctx["confidence"]
        bias_metrics = signal_ctx["bias_metrics"]
        trade = signal_ctx["trade"]
        log_func = signal_ctx["log_func"]
        # AI/Runtime Trend Clash
        ai_runtime_clash = (current_trend == "BULLISH" and direction == "short") or (
            current_trend == "BEARISH" and direction == "long"
        )
        if ai_runtime_clash:
            trade["forced_min_margin"] = True
            trade["forced_min_margin_reason"] = "ai_runtime_trend_clash"
            original_conf = confidence
            confidence *= 0.85
            log_func(
                "sizing",
                f"[WARN] AI/Runtime TREND CLASH: {coin} {direction.upper()} vs {current_trend} trend → MIN_MARGIN penalty",
                {
                    "coin": coin,
                    "direction": direction,
                    "runtime_trend": current_trend,
                    "original_conf": original_conf,
                    "new_conf": confidence,
                },
            )

        # Directional Bias
        pre_bias_conf = confidence
        bias_ctx = {
            "signal": signal,
            "confidence": confidence,
            "bias_metrics": bias_metrics,
            "current_trend": current_trend,
        }
        confidence = self.apply_directional_bias(bias_ctx)
        if confidence != pre_bias_conf:
            log_func(
                "confidence",
                f"[INFO] Directional bias adjustment: {coin} {signal} {pre_bias_conf:.2f} -> {confidence:.2f}",
                {
                    "coin": coin,
                    "signal": signal,
                    "original": pre_bias_conf,
                    "adjusted": confidence,
                },
            )

        trade["confidence"] = confidence
        return confidence

    def _log_execution_snapshot(
        self,
        signal_ctx: dict[str, Any],
        indicators_htf_snap: dict[str, Any],
        indicators_3m_snap: dict[str, Any],
        current_trend: str,
        is_counter_trend: bool,
    ) -> None:
        """Log a detailed snapshot of market conditions at the time of execution.

        Args:
        ----
            signal_ctx: Context for the trade signal.
            indicators_htf_snap: Snapshot of HTF technical indicators.
            indicators_3m_snap: Snapshot of 3m technical indicators.
            current_trend: The current detected runtime trend.
            is_counter_trend: Boolean indicating if trade is counter-trend.

        """
        coin = signal_ctx["coin"]
        trade = signal_ctx["trade"]
        execution_report = signal_ctx["report"]
        log_func = signal_ctx["log_func"]
        signal = signal_ctx["signal"]
        volume_ratio = trade.get("volume_ratio_runtime", 0)
        price_htf, ema20_htf = (
            indicators_htf_snap.get("current_price"),
            indicators_htf_snap.get("ema_20"),
        )
        price_3m, ema20_3m = (
            indicators_3m_snap.get("current_price"),
            indicators_3m_snap.get("ema_20"),
        )

        def _fmt(val: float | int | None) -> str:
            return f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"

        comp_htf = ">" if (price_htf or 0) > (ema20_htf or 0) else "<"
        comp_3m = ">" if (price_3m or 0) > (ema20_3m or 0) else "<"

        print(
            f"🧾 EXECUTION SNAPSHOT {coin}: {HTF_LABEL} price={_fmt(price_htf)} {comp_htf} EMA20={_fmt(ema20_htf)} | 3m price={_fmt(price_3m)} {comp_3m} EMA20={_fmt(ema20_3m)} | volume_ratio={volume_ratio:.2f}x | counter_trend={is_counter_trend} | trend_state={current_trend.upper()}"
        )

        # Inconsistency check
        signal = "buy_to_enter" if trade.get("signal") == "buy_to_enter" else "sell_to_enter"
        runtime_counter_trend = (
            signal == "buy_to_enter" and current_trend.lower() == "bearish"
        ) or (signal == "sell_to_enter" and current_trend.lower() == "bullish")

        if is_counter_trend != runtime_counter_trend:
            log_func(
                "trend",
                f"[WARN] TREND INCONSISTENCY {coin}: EMA counter={is_counter_trend}, Runtime counter={runtime_counter_trend}",
                {
                    "coin": coin,
                    "signal": signal,
                    "ema_counter_trend": is_counter_trend,
                    "runtime_counter_trend": runtime_counter_trend,
                    "current_trend": current_trend,
                },
            )
            trade["forced_min_margin"] = True
            execution_report["notes"].append(
                {
                    "coin": coin,
                    "note": "trend_mismatch_min_margin",
                    "classification": trade.get("classification"),
                    "ema_counter_trend": is_counter_trend,
                    "runtime_counter_trend": runtime_counter_trend,
                },
            )

    def _handle_trend_flip_guard(
        self,
        guard_ctx: dict[str, Any],
    ) -> tuple[bool, float, float]:
        """Orchestrate trend flip guard logic based on trade classification.

        Args:
        ----
            guard_ctx: Context containing trend flip and trade data.

        Returns:
        -------
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
        ----
            guard_ctx: Context containing trend flip and trade data.

        Returns:
        -------
            A tuple of (proceed_boolean, confidence, partial_margin_factor).

        """
        coin = guard_ctx["coin"]
        signal = guard_ctx["signal"]
        confidence = guard_ctx["confidence"]
        partial_margin_factor = guard_ctx["partial_margin_factor"]
        confidence = guard_ctx["confidence"]
        partial_margin_factor = guard_ctx["partial_margin_factor"]
        relax_mode_active = guard_ctx["relax_mode_active"]
        execution_report = guard_ctx["report"]
        trade = guard_ctx["trade"]
        classification = "counter_trend"

        counter_trend_cooldown = self.counter_trend_cooldown
        if counter_trend_cooldown > 0:
            print(
                f"[BLOCK] Counter-trend cooldown active: Blocking {coin} {signal} ({counter_trend_cooldown} cycles remaining)."
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
        ----
            guard_ctx: Context for the flip guard.
            confidence: Current confidence level.
            partial_margin_factor: Current margin scaling factor.

        Returns:
        -------
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
        ----
            guard_ctx: Context containing trend flip and trade data.

        Returns:
        -------
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
                    confidence = max(orig_conf * 0.97, orig_conf - 0.02, orig_conf * 0.95)
                    partial_margin_factor = min(partial_margin_factor, 0.7)
                    log_func(
                        "sizing",
                        f"[WATCH] Trend flip guard (trend-following): {coin} confidence {orig_conf:.2f} → {confidence:.2f} & sizing 50% immediately after flip.",
                    )
                elif guard_cycles_since_flip == 1:
                    confidence = max(orig_conf * 0.98, orig_conf - 0.01, orig_conf * 0.97)
                    partial_margin_factor = min(partial_margin_factor, 0.8)
                    log_func(
                        "sizing",
                        f"[WATCH] Trend flip guard (trend-following): {coin} confidence {orig_conf:.2f} → {confidence:.2f} & sizing 70% {constants.ALIGNMENT_STRENGTH_L1} cycle after flip.",
                    )
                elif guard_cycles_since_flip == constants.TREND_FLIP_COOLDOWN_DEFAULT:
                    confidence = max(orig_conf * 0.99, orig_conf * 0.98)
                    partial_margin_factor = min(partial_margin_factor, 0.90)
                    log_func(
                        "sizing",
                        f"[WATCH] Trend flip guard (trend-following): {coin} confidence {orig_conf:.2f} → {confidence:.2f} & sizing 85% {constants.TREND_FLIP_COOLDOWN_DEFAULT} cycles after flip.",
                    )

        trade["confidence"] = confidence
        return confidence, partial_margin_factor

    def _finalize_entry_sizing_and_execution(
        self,
        signal_ctx: dict[str, Any],
    ) -> bool:
        """Finalize position sizing and trigger the order execution process.

        Args:
        ----
            signal_ctx: Context for the trade signal.

        Returns:
        -------
            True if execution was successfully attempted.

        """
        coin = signal_ctx["coin"]
        signal = signal_ctx["signal"]
        current_price = signal_ctx["current_price"]
        indicators_htf = signal_ctx.get("indicators_htf", {})

        # 1. ATR-based Stop Loss Calculation
        atr_stop_loss = self._calculate_atr_stop_loss(signal, coin, current_price, indicators_htf)

        # 2. & 3. Sizing & Multipliers
        calculated_margin = self._calculate_final_margin(signal_ctx, atr_stop_loss)

        # 4. Safety & Risk Validation
        if not self._validate_risk_and_cash(signal_ctx, calculated_margin):
            return False

        # 5. Final Execution
        return self._execute_order_payload(signal_ctx, calculated_margin, atr_stop_loss)

    def _calculate_atr_stop_loss(
        self,
        signal: str,
        coin: str,
        current_price: float,
        indicators_htf: dict[str, Any],
    ) -> float:
        """Calculate an ATR-based stop loss price for a new entry.

        Args:
        ----
            signal: The entry signal type.
            coin: The cryptocurrency symbol.
            current_price: Current market entry price.
            indicators_htf: HTF technical indicators.

        Returns:
        -------
            The calculated stop loss price.

        """
        from config.config import Config

        try:
            atr_value = indicators_htf.get("atr_14") if isinstance(indicators_htf, dict) else None
            if atr_value and atr_value > 0:
                sl_distance = atr_value * Config.ATR_SL_MULTIPLIER
                atr_stop_loss = (
                    (current_price - sl_distance)
                    if signal == "buy_to_enter"
                    else (current_price + sl_distance)
                )
                print(
                    f"[INFO]  Pre-calculated ATR SL for sizing: {coin} ATR={atr_value:.6f} x {Config.ATR_SL_MULTIPLIER} -> SL=${atr_stop_loss:.6f}",
                )
                return atr_stop_loss
            fallback_dist = current_price * 0.02
            atr_stop_loss = (
                (current_price - fallback_dist)
                if signal == "buy_to_enter"
                else (current_price + fallback_dist)
            )
            print(f"[WARN]  ATR unavailable for {coin}, using 2% fallback SL: ${atr_stop_loss:.6f}")
            return atr_stop_loss
        except Exception as e:
            print(f"[WARN]  ATR SL calculation failed for {coin}: {e}")
            fallback_dist = current_price * 0.02
            return (
                (current_price - fallback_dist)
                if signal == "buy_to_enter"
                else (current_price + fallback_dist)
            )

    def _calculate_final_margin(self, signal_ctx: dict[str, Any], atr_stop_loss: float) -> float:
        """Calculate and adjust the final margin for a trade based on risk and context.

        Args:
        ----
            signal_ctx: Context for the trade signal.
            atr_stop_loss: The pre-calculated stop loss price.

        Returns:
        -------
            The final margin amount in USD.

        """
        from config.config import Config

        confidence = signal_ctx["confidence"]
        current_price = signal_ctx["current_price"]
        leverage = signal_ctx["leverage"]
        market_regime = signal_ctx["market_regime"]
        log_func = signal_ctx["log_func"]
        partial_margin_factor = signal_ctx.get("partial_margin_factor", 1.0)
        trade = signal_ctx["trade"]
        coin = signal_ctx["coin"]

        margin_ctx = {
            "confidence": confidence,
            "balance": self.current_balance,
            "entry_price": current_price,
            "stop_loss": atr_stop_loss,
            "leverage": leverage,
            "market_regime": market_regime,
            "log_func": log_func,
        }
        calculated_margin = self.calculate_confidence_based_margin(margin_ctx)

        # Market Regime & Scout Multipliers
        regime_mult = Config.MARKET_REGIME_MULTIPLIERS.get(market_regime, 1.0)
        current_mult = regime_mult
        if Config.SCOUT_MODE_ENABLED and market_regime == "NEUTRAL":
            current_mult *= getattr(Config, "SCOUT_LEVERAGE_MULT", 0.5)

        calculated_margin *= current_mult

        # Partial Margin & Overrides
        if partial_margin_factor < 1.0:
            reduced_margin = calculated_margin * partial_margin_factor
            print(
                f"[WATCH] Applying partial margin ({partial_margin_factor * 100:.0f}%): ${calculated_margin:.2f} → ${reduced_margin:.2f}",
            )
            calculated_margin = max(reduced_margin, Config.MIN_POSITION_MARGIN_USD)

        if trade.get("forced_min_margin"):
            orig_margin = calculated_margin
            calculated_margin = Config.MIN_POSITION_MARGIN_USD
            log_func(
                "sizing",
                f"[WARN]  TREND MISMATCH: Using MIN_MARGIN ${Config.MIN_POSITION_MARGIN_USD:.0f} (was ${orig_margin:.2f})",
                {
                    "coin": coin,
                    "original_margin": orig_margin,
                    "reduced_margin": calculated_margin,
                    "reason": "trend_mismatch_min_margin",
                },
            )

        # Safety Check: Min Margin
        if calculated_margin < Config.MIN_POSITION_MARGIN_USD:
            print(
                f"[INFO]  Calculated margin ${calculated_margin:.2f} below minimum ${Config.MIN_POSITION_MARGIN_USD:.2f}. Using minimum.",
            )
            calculated_margin = Config.MIN_POSITION_MARGIN_USD

        return calculated_margin

    def _validate_risk_and_cash(self, signal_ctx: dict[str, Any], calculated_margin: float) -> bool:
        """Validate the trade against available cash and risk management constraints.

        Args:
        ----
            signal_ctx: Context for the trade signal.
            calculated_margin: The proposed margin amount.

        Returns:
        -------
            True if the trade passes risk calibration, False otherwise.

        """
        from config.config import Config

        coin = signal_ctx["coin"]
        current_price = signal_ctx["current_price"]
        leverage = signal_ctx["leverage"]
        confidence = signal_ctx["confidence"]
        trade = signal_ctx["trade"]
        execution_report = signal_ctx["report"]

        min_cash_guard = self.current_balance * 0.10
        if (self.current_balance - calculated_margin) < min_cash_guard:
            print(
                f"[WARN]  Trade would reduce available cash below minimum ${min_cash_guard:.2f}. Trade blocked.",
            )
            execution_report["blocked"].append(
                {
                    "coin": coin,
                    "reason": "available_cash_guard",
                    "calculated_margin": calculated_margin,
                },
            )
            trade["runtime_decision"] = "blocked_cash_guard"
            return False

        # Risk Manager Validation
        notional_usd = calculated_margin * leverage
        risk_decision = self.risk_manager.should_enter_trade(
            symbol=coin,
            current_positions=self.positions,
            current_prices={coin: current_price},
            confidence=confidence,
            proposed_notional=notional_usd,
            current_balance=self.current_balance,
        )

        if not risk_decision["should_enter"]:
            print(f"[WARN]  Risk management blocked trade: {risk_decision['reason']}")
            execution_report["blocked"].append(
                {"coin": coin, "reason": f"risk_manager:{risk_decision['reason']}"},
            )
            trade["runtime_decision"] = "blocked_risk_manager"
            return False

        return True

    def _execute_order_payload(
        self,
        signal_ctx: dict[str, Any],
        calculated_margin: float,
        atr_stop_loss: float,
    ) -> bool:
        """Execute the final order payload using live or simulated execution.

        Args:
        ----
            signal_ctx: Context for the trade signal.
            calculated_margin: The final margin to use.
            atr_stop_loss: The stop loss price.

        Returns:
        -------
            True if the order execution was successful.

        """
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
            print(f"[WARN]  {signal.upper()} {coin}: Calculated margin is zero/negative. Skipping.")
            return False
        if margin_usd > self.current_balance:
            print(
                f"[WARN]  {signal.upper()} {coin}: Not enough cash for margin (${margin_usd:.2f} > ${self.current_balance:.2f})",
            )
            return False

        quantity_coin = notional_usd / current_price

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
                stop_loss=atr_stop_loss,
                profit_target=trade.get("profit_target"),
                invalidation=trade.get("invalidation_condition"),
            )
            if not live_result.get("success"):
                err_msg = live_result.get("error", "unknown_error")
                print(f"[BLOCK] LIVE ORDER FAILED: {coin} {signal} ({err_msg})")
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

        # Simulated
        self.current_balance -= margin_usd
        est_liq = self._estimate_liquidation_price(current_price, leverage, direction)
        self.positions[coin] = {
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
                "profit_target": trade.get("profit_target"),
                "stop_loss": atr_stop_loss,
                "invalidation_condition": trade.get("invalidation_condition"),
            },
            "risk_usd": margin_usd,
            "loss_cycle_count": 0,
            "entry_volume": indicators_3m.get("volume"),
            "entry_avg_volume": indicators_3m.get("avg_volume"),
            "entry_volume_ratio": indicators_3m.get("volume_ratio"),
            "entry_atr_14": indicators_3m.get("atr_14"),
            "trend_alignment": classification,
            "trend_context": {
                "trend_at_entry": trade.get("trend_runtime", "unknown"),
                "alignment": classification,
                "cycle": self.current_cycle_number,
            },
            "trailing": {},
            "sl_oid": -1,
            "tp_oid": -1,
            "entry_oid": -1,
            "wait_for_fill": False,
        }
        print(
            f"[OK]    {signal.upper()}: Opened {direction} {coin} ({format_num(quantity_coin, 4)} @ ${format_num(current_price, 4)} / Notional ${format_num(notional_usd, 2)} / Margin ${format_num(margin_usd, 2)})",
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
        return True

    def execute_decision(
        self,
        decisions: dict[str, dict[str, Any]],
        current_prices: dict[str, float],
        indicator_cache: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Execute trading decisions from AI with dynamic sizing and slot management.

        Args:
        ----
            decisions: Dictionary of coin-to-decision data.
            current_prices: Current market price dictionary.
            indicator_cache: Optional technical indicator cache.

        """
        if not isinstance(decisions, dict):
            print(f"[ERR]   Invalid decisions format: {type(decisions)}")
            return

        batch_ctx = self._prepare_decision_batch_context(decisions)

        for coin, trade in batch_ctx["sorted_decisions"]:
            self._process_single_decision(coin, trade, current_prices, indicator_cache, batch_ctx)

        self.last_execution_report = batch_ctx["report"]

    def _prepare_decision_batch_context(
        self, decisions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Prepare the context and report for a batch of trading decisions.

        Args:
        ----
            decisions: The dictionary of raw AI decisions.

        Returns:
        -------
            Dictionary containing sorted decisions, report, and metrics.

        """
        from config.config import Config

        bias_metrics = getattr(self, "latest_bias_metrics", self.get_directional_bias_metrics())
        context = self._prepare_execution_context()
        report = context["report"]

        # Define logging helper
        def log_func(category: str, message: str, details: dict | None = None) -> None:
            print(message)
            report["debug_logs"].append(
                {
                    "category": category,
                    "message": message,
                    "details": details or {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        sorted_decisions = sorted(decisions.items(), key=self._get_signal_priority)
        entry_signals = [
            f"{c}({t.get('confidence', 0):.2f})"
            for c, t in sorted_decisions
            if t.get("signal") in ["buy_to_enter", "sell_to_enter"]
        ]
        if entry_signals:
            print(f"[INFO]  Signal priority order: {entry_signals}")

        return {
            "report": report,
            "market_regime": context["market_regime"],
            "regime_strength": context["regime_strength"],
            "bias_metrics": bias_metrics,
            "sorted_decisions": sorted_decisions,
            "new_positions_this_cycle": 0,
            "log_func": log_func,
            "Config": Config,
        }

    def _process_single_decision(
        self,
        coin: str,
        trade: dict[str, Any],
        current_prices: dict[str, float],
        indicator_cache: object,
        batch_ctx: dict[str, Any],
    ) -> None:
        """Process a single AI trading decision through the execution pipeline.

        Args:
        ----
            coin: The cryptocurrency symbol.
            trade: The decision data for the coin.
            current_prices: Dictionary of current market prices.
            indicator_cache: Technical indicator cache handler.
            batch_ctx: Shared execution context for the cycle.

        """
        if not isinstance(trade, dict):
            print(f"[WARN]  Invalid trade data for {coin}: {type(trade)}")
            return

        price = current_prices.get(coin)
        if price is None or not isinstance(price, (int, float)) or price <= 0:
            print(f"[WARN]  Skipping {coin}: Invalid price data.")
            return

        signal = trade.get("signal")
        position = self.positions.get(coin)

        if signal in {"buy_to_enter", "sell_to_enter"}:
            entry_ctx = {
                "coin": coin,
                "trade": trade,
                "price": price,
                "position": position,
                "indicator_cache": indicator_cache,
                "batch_ctx": batch_ctx,
            }
            self._handle_entry_decision(entry_ctx)
        elif signal == "close_position":
            self._handle_exit_signal_logic(coin, trade, price, position, batch_ctx["report"])
        elif signal == "hold":
            self._handle_hold_signal(coin, position, batch_ctx["report"], trade)
        else:
            print(f"[WARN]  Unknown signal '{signal}' for {coin}. Skipping.")

    def _handle_entry_decision(
        self,
        entry_ctx: dict[str, Any],
    ) -> None:
        """Process an entry (buy/sell) decision including technical scaling.

        Args:
        ----
            entry_ctx: Context for the entry attempt.

        """
        coin = entry_ctx["coin"]
        trade = entry_ctx["trade"]
        price = entry_ctx["price"]
        position = entry_ctx["position"]
        indicator_cache = entry_ctx["indicator_cache"]
        batch_ctx = entry_ctx["batch_ctx"]
        direction = "long" if trade.get("signal") == "buy_to_enter" else "short"

        # Initialize indicators
        indicators_htf = (
            indicator_cache.get_indicators(coin, HTF_INTERVAL, self.market_data)
            if indicator_cache
            else {}
        )
        indicators_3m = (
            indicator_cache.get_indicators(coin, "3m", self.market_data) if indicator_cache else {}
        )

        signal_ctx = {
            "coin": coin,
            "signal": trade.get("signal"),
            "direction": direction,
            "trade": trade,
            "position": position,
            "new_positions_this_cycle": batch_ctx["new_positions_this_cycle"],
            "regime_strength": batch_ctx["regime_strength"],
            "market_regime": batch_ctx["market_regime"],
            "report": batch_ctx["report"],
            "log_func": batch_ctx["log_func"],
            "current_price": price,
            "indicator_cache": indicator_cache,
            "bias_metrics": batch_ctx["bias_metrics"],
            "indicators_htf": indicators_htf,
            "indicators_3m": indicators_3m,
        }

        # 1. Preconditions
        res = self._check_entry_preconditions(signal_ctx)
        if not res["proceed"]:
            return

        signal_ctx["confidence"] = res["confidence"]
        signal_ctx["leverage"] = res["leverage"]

        # 2. & 3. Technical Confidence Adjustments
        res_tech = self._apply_technical_confidence_adjustments(signal_ctx)
        if not res_tech["proceed"]:
            return
        signal_ctx["confidence"] = res_tech["confidence"]

        # 5. Trend, Bias & Flip Guard Analysis
        res_ana = self._analyze_entry_trend_and_bias(signal_ctx)
        if not res_ana["proceed"]:
            return

        signal_ctx["confidence"] = res_ana["confidence"]
        signal_ctx["partial_margin_factor"] = res_ana["partial_margin_factor"]
        signal_ctx["is_counter_trend"] = res_ana["is_counter_trend"]
        signal_ctx["classification"] = res_ana["classification"]

        # 6. Sizing & Final Execution
        if self._finalize_entry_sizing_and_execution(signal_ctx):
            batch_ctx["new_positions_this_cycle"] += 1

    def _handle_hold_signal(
        self,
        coin: str,
        position: dict[str, Any] | None,
        report: dict[str, Any],
        trade: dict[str, Any],
    ) -> None:
        """Handle a 'hold' signal for a coin by logging the decision.

        Args:
        ----
            coin: The cryptocurrency symbol.
            position: The existing position, if any.
            report: The execution report to update.
            trade: The trade decision dictionary.

        """
        if position:
            print(f"[INFO]  HOLD: Holding {position.get('direction', 'long')} {coin} position.")
        else:
            print(f"[INFO]  HOLD: Staying cash in {coin}.")
        report["holds"].append({"coin": coin, "has_position": bool(position)})
        trade["runtime_decision"] = "hold"

    def _update_position_pnl(self, pos: dict[str, Any], price: float) -> float:
        """Calculate unrealized PnL for a position, honoring live values if available.

        Args:
        ----
            pos: The position dictionary.
            price: Current market price.

        Returns:
        -------
            The calculated unrealized PnL in USD.

        """
        if self.is_live_trading:
            # Live mode: Keep Binance unrealized_pnl if available (includes funding fees)
            existing_pnl = pos.get("unrealized_pnl", 0.0)
            if isinstance(existing_pnl, (int, float)) and existing_pnl != 0:
                return existing_pnl

            # Fallback for live mode
            entry = pos["entry_price"]
            qty = pos["quantity"]
            direction = pos.get("direction", "long")
            return (price - entry) * qty if direction == "long" else (entry - price) * qty
        # Simulation mode: calculate manually
        entry = pos["entry_price"]
        qty = pos["quantity"]
        direction = pos.get("direction", "long")
        return (price - entry) * qty if direction == "long" else (entry - price) * qty
