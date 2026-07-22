"""Portfolio state management and persistence for TradeSeeker.

This module contains the PortfolioStateManager class which handles state persistence,
trade history management, and performance analytics.
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from loguru import logger

from src.core import constants
from src.utils import safe_file_read, safe_file_read_cached, safe_file_write


class PortfolioStateManager:
    """Manages portfolio state persistence, trade history, and performance analytics.

    This class provides thread-safe state management with callback hooks for
    directional bias updates when trades are added to history.
    """

    def __init__(
        self,
        state_file: str,
        history_file: str,
        full_history_file: str,
        cycle_history_file: str,
        initial_balance: float,
        max_cycle_history: int,
    ) -> None:
        """Initialize the PortfolioStateManager.

        Args:
            state_file: Path to the portfolio state JSON file.
            history_file: Path to the active trade history JSON file.
            full_history_file: Path to the full trade history JSON file.
            cycle_history_file: Path to the cycle history JSON file.
            initial_balance: Starting balance for the portfolio.
            max_cycle_history: Maximum number of cycle history entries to retain.
        """
        self.state_file = state_file
        self.history_file = history_file
        self.full_history_file = full_history_file
        self.cycle_history_file = cycle_history_file
        self.initial_balance = initial_balance
        self.max_cycle_history = max_cycle_history
        self._lock = threading.RLock()

    def load_state(self) -> dict[str, Any]:
        """Load portfolio state from local JSON file.

        Returns:
            Dictionary containing the saved portfolio state.
        """
        data: dict[str, Any] = safe_file_read_cached(self.state_file, default_data={})
        logger.info(
            "Loaded state ({} positions, {} closed trades)" if data else "No state file found.",
            len(data.get("positions", {})),
            data.get("trade_count", 0),
        )
        return data

    def save_state(self, data: dict[str, Any]) -> None:
        """Save portfolio state to local JSON file.

        Args:
            data: The state dictionary to persist.
        """
        with self._lock:
            data["last_updated"] = datetime.now(timezone.utc).isoformat()
            safe_file_write(self.state_file, data)

    def load_trade_history(self) -> list[dict[str, Any]]:
        """Load trade history from the local JSON file.

        Returns:
            List of trade records.
        """
        history: list[dict[str, Any]] = safe_file_read(self.history_file, default_data=[])
        logger.info("Loaded {} trades.", len(history))
        return history

    def save_trade_history(self, trade_history: list[dict[str, Any]]) -> None:
        """Save the most recent trade history to the active history file.

        Args:
            trade_history: The full trade history list to save.
        """
        with self._lock:
            history_to_save: list[dict[str, Any]] = trade_history[
                -constants.MAX_TRADE_HISTORY_DISPLAY :
            ]
            safe_file_write(self.history_file, history_to_save)
            logger.info("Saved {} trades.", len(history_to_save))

    def ensure_full_history_exists(self, trade_history: list[dict[str, Any]]) -> None:
        """Ensure full trade history exists, copying from active history if needed.

        Args:
            trade_history: The current trade history to copy if file is missing.
        """
        if not Path(self.full_history_file).exists():
            logger.info("Creating {} from existing trade history...", self.full_history_file)
            safe_file_write(self.full_history_file, trade_history)

    def add_to_history(
        self,
        trade: dict[str, Any],
        trade_history: list[dict[str, Any]],
        on_trade_added: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Add a completed trade to both active and full history files.

        Args:
            trade: The trade record to append.
            trade_history: The current trade history list (will be mutated).
            on_trade_added: Optional callback invoked after the trade is appended.
                Intended for triggering directional bias updates and state saves.

        Returns:
            The updated trade_history list.
        """
        with self._lock:
            trade_history.append(trade)
            self.save_trade_history(trade_history)

            full_history = safe_file_read(self.full_history_file, [])
            if not full_history:
                path_obj = Path(self.full_history_file)
                if path_obj.exists() and path_obj.stat().st_size > 0:
                    time.sleep(0.1)
                    full_history = safe_file_read(self.full_history_file, [])

            full_history.append(trade)
            if not safe_file_write(self.full_history_file, full_history):
                logger.error("Failed to save full history for {}", trade.get("symbol"))

            if on_trade_added is not None:
                on_trade_added(trade)

        return trade_history

    def load_cycle_history(self) -> list[dict[str, Any]]:
        """Load cycle history records from the local JSON file.

        Returns:
            List of cycle history records.
        """
        history = safe_file_read(self.cycle_history_file, default_data=[])
        logger.info("Loaded {} cycles.", len(history))
        return history

    def save_cycle_history(self, cycle_history: list[dict[str, Any]]) -> None:
        """Save cycle history to the local JSON file.

        Args:
            cycle_history: The cycle history list to persist.
        """
        safe_file_write(self.cycle_history_file, cycle_history)

    def add_to_cycle_history(
        self,
        cycle_data: dict[str, Any],
        cycle_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add a cycle record to history, enforcing the max cycle limit.

        Args:
            cycle_data: The cycle record to append.
            cycle_history: The current cycle history list (will be mutated).

        Returns:
            The updated cycle history list.
        """
        with self._lock:
            cycle_history.append(cycle_data)
            cycle_history = cycle_history[-self.max_cycle_history :]
        self.save_cycle_history(cycle_history)
        logger.info("Saved cycle {} (Total: {})", cycle_data.get("cycle"), len(cycle_history))
        return cycle_history

    # --- Performance Analytics ---

    def get_performance_summary(self, trade_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a performance summary from trade history.

        Args:
            trade_history: List of completed trade records.

        Returns:
            Dictionary containing total trades, wins, losses, win rate, net PnL,
            average win, average loss, and profit factor.
        """
        if not trade_history:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "profit_factor": 0.0,
            }

        total = len(trade_history)
        wins = [t for t in trade_history if float(t.get("pnl", 0) or 0) > 0]
        losses = [t for t in trade_history if float(t.get("pnl", 0) or 0) < 0]
        breakeven = total - len(wins) - len(losses)

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total * 100) if total > 0 else 0.0

        net_pnl = sum(float(t.get("pnl", 0) or 0) for t in trade_history)

        total_wins = sum(float(t.get("pnl", 0) or 0) for t in wins)
        total_losses = abs(sum(float(t.get("pnl", 0) or 0) for t in losses))

        avg_win = (total_wins / win_count) if win_count > 0 else 0.0
        avg_loss = (total_losses / loss_count) if loss_count > 0 else 0.0

        profit_factor = (total_wins / total_losses) if total_losses > 0 else float("inf")

        return {
            "total_trades": total,
            "wins": win_count,
            "losses": loss_count,
            "breakeven": breakeven,
            "win_rate": round(win_rate, 2),
            "net_pnl": round(net_pnl, 2),
            "total_wins": round(total_wins, 2),
            "total_losses": round(total_losses, 2),
            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
        }

    def get_win_rate(self, trade_history: list[dict[str, Any]]) -> float:
        """Calculate the win rate percentage from trade history.

        Args:
            trade_history: List of completed trade records.

        Returns:
            Win rate as a percentage (0.0 to 100.0).
        """
        if not trade_history:
            return 0.0

        total = len(trade_history)
        wins = sum(1 for t in trade_history if float(t.get("pnl", 0) or 0) > 0)
        return round((wins / total) * 100, 2) if total > 0 else 0.0

    def get_net_pnl(self, trade_history: list[dict[str, Any]]) -> float:
        """Calculate the net profit/loss from trade history.

        Args:
            trade_history: List of completed trade records.

        Returns:
            Net PnL in USD.
        """
        return round(sum(float(t.get("pnl", 0) or 0) for t in trade_history), 2)

    def backup_historical_files(self, cycle_number: int) -> str | None:
        """Create a timestamped backup for historical JSON files before wiping them.

        Args:
            cycle_number: The current cycle number for grouping the backup.

        Returns:
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
            ("data/reset_log.json", []),
        ]

        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            backed_up = []

            for file_path, default in files_to_backup:
                path_obj = Path(file_path)
                data = safe_file_read(str(path_obj), default)
                if data == default and not path_obj.exists():
                    continue

                target_path = backup_dir / path_obj.name
                safe_file_write(str(target_path), data)

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
            logger.info("History backup created at {}", backup_dir)
            return str(backup_dir)
        except Exception as e:
            logger.warning("History backup failed: {}", e)
            return None
