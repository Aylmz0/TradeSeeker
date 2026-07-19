# alpha_arena_deepseek.py
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger


# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.config import Config
from src.ai.deepseek_api import DeepSeekAPI
from src.schemas.position import Position
from src.core import constants
from src.core.account_service import AccountService
from src.core.ai_service import AIService
from src.core.backtest import AdvancedRiskManager
from src.core.data_engine import DataEngine
from src.core.log_config import setup_logging
from src.core.market_data import RealMarketData
from src.core.portfolio_manager import PortfolioManager
from src.core.strategy_analyzer import StrategyAnalyzer
from src.utils import (
    cleanup_stale_temp_files,
    format_num,
    safe_file_read,
    safe_file_read_cached,
    safe_file_write,
)


# Define constants
HTF_INTERVAL = getattr(Config, "HTF_INTERVAL", "1h") or "1h"
HTF_LABEL = HTF_INTERVAL


class AlphaArenaDeepSeek:
    """Alpha Arena-like DeepSeek integration with auto TP/SL, dynamic sizing, and advanced risk management."""

    def __init__(self, api_key: str | None = None):
        self.market_data = RealMarketData()
        self.strategy_analyzer = StrategyAnalyzer(self.market_data)
        self.portfolio = PortfolioManager()
        self.ai_service = AIService(self.portfolio, self.market_data, self.strategy_analyzer)
        self.account_service = AccountService(self.portfolio)
        self.deepseek = DeepSeekAPI(api_key)
        self.risk_manager = AdvancedRiskManager()
        self.data_engine = DataEngine()
        self.invocation_count = 0  # Track AI calls since bot start
        self.tp_sl_timer = None
        self.is_running = False
        self.enhanced_exit_enabled = True  # Enhanced exit strategy control flag
        self.cycle_active = False  # Track whether a trading cycle is executing
        self.current_cycle_number = 0
        # Trend flip cooldown management is handled in PortfolioManager.
        self.latest_indicator_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self.history_reset_interval = Config.HISTORY_RESET_INTERVAL
        self.auto_train_cycle_count = 0  # Track cycles for 3h automated retrain

    def _apply_directional_capacity_filter(
        self,
        decisions: dict[str, dict],
    ) -> tuple[dict[str, dict], bool]:
        """Convert entry signals to hold when directional capacity is full."""
        if not isinstance(decisions, dict):
            return decisions, False

        directional_counts = self.portfolio.count_positions_by_direction()
        limit = Config.SAME_DIRECTION_LIMIT
        blocked = {
            "long": directional_counts.get("long", 0) >= limit,
            "short": directional_counts.get("short", 0) >= limit,
        }
        cooldowns = self.portfolio.directional_cooldowns
        if (
            not blocked["long"]
            and not blocked["short"]
            and cooldowns.get("long", 0) == 0
            and cooldowns.get("short", 0) == 0
        ):
            return decisions, False

        filtered: dict[str, dict] = {}
        changed = False
        for coin, trade in decisions.items():
            if not isinstance(trade, dict):
                filtered[coin] = trade
                continue

            signal = trade.get("signal")
            direction = None
            if signal == "buy_to_enter":
                direction = "long"
            elif signal == "sell_to_enter":
                direction = "short"

            if direction and cooldowns.get(direction, 0) > 0:
                changed = True
                remaining = cooldowns.get(direction, 0)
                filtered[coin] = {
                    "signal": "hold",
                    "justification": f"Directional cooldown active ({remaining} cycles remaining)",
                }
                logger.info(
                    "Directional cooldown: Blocking {} entry for {} ({} cycles remaining)",
                    direction.upper(),
                    coin,
                    remaining,
                )
                continue

            if direction and blocked.get(direction):
                changed = True
                filtered[coin] = {
                    "signal": "hold",
                    "justification": f"{direction.upper()} capacity full ({directional_counts.get(direction, 0)}/{limit}); evaluate exits or opposite-side setups.",
                }
            else:
                filtered[coin] = trade

        return filtered, changed

    def maybe_reset_history(self, cycle_number: int):
        """Reset historical logs at configured intervals to prevent long-term bias."""
        interval = getattr(self, "history_reset_interval", 35)
        if interval <= 0:
            return
        cycles_elapsed = getattr(self.portfolio, "cycles_since_history_reset", 0)
        if cycles_elapsed >= interval:
            logger.info(
                "Bias control: {} cycles since last reset (interval {}). Resetting history.",
                cycles_elapsed,
                interval,
            )
            self.portfolio.reset_historical_data(cycle_number)
            self.invocation_count = 0

    def calculate_optimal_cycle_frequency(self) -> int:
        """Calculate optimal cycle frequency based on user configuration.
        Strictly uses CYCLE_INTERVAL_MINUTES from .env as requested by the user,
        disabling the previous dynamic ATR-based volatility logic.
        """
        try:
            return Config.CYCLE_INTERVAL_MINUTES * 60  # Convert to seconds
        except Exception as e:
            logger.warning("Cycle frequency calculation error: {}", e)
            return constants.DEFAULT_CYCLE_FREQUENCY_S  # Default fallback

    def track_performance_metrics(self, cycle_number: int):
        """Record basic performance metrics for each cycle"""
        try:
            metrics = {
                "cycle": cycle_number,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_value": self.portfolio.total_value,
                "total_return": self.portfolio.total_return,
                "sharpe_ratio": self.portfolio.sharpe_ratio,
                "open_positions": len(self.portfolio.positions),
                "available_cash": self.portfolio.current_balance,
                "total_trades": self.portfolio.trade_count,
            }

            # Save to performance history file
            performance_history = safe_file_read("data/performance_history.json", [])
            performance_history.append(metrics)
            safe_file_write(
                "data/performance_history.json",
                performance_history[-constants.MAX_PERFORMANCE_HISTORY :],
            )  # Last N cycles

        except Exception as e:
            logger.warning("Performance tracking error: {}", e)

    def should_run_performance_analysis(self, cycle_number: int) -> bool:
        """Run analysis every 10 cycles or in critical situations"""
        # Every N cycles
        if cycle_number % constants.PERFORMANCE_ANALYSIS_INTERVAL == 0:
            return True

        # During large PnL changes
        if abs(self.portfolio.total_return) > constants.CRITICAL_RETURN_THRESHOLD:
            return True

        # When too many positions are open
        return len(self.portfolio.positions) >= constants.MAX_POSITIONS_ALERT_THRESHOLD

    def run_trading_cycle(self, cycle_number: int, cancel_event: threading.Event | None = None):
        """Run a single trading cycle with auto TP/SL and enhanced features"""
        log = logger.bind(cycle=cycle_number, phase="trading_cycle")
        log.info(
            "==== CYCLE {} | {} {}",
            cycle_number,
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "=" * 50,
        )

        # Check cancellation (watchdog timeout)
        if cancel_event and cancel_event.is_set():
            logger.warning("Cycle {} cancelled before start.", cycle_number)
            return

        # Check bot control at cycle start
        control = self._read_bot_control()
        if control.get("status") == "paused":
            logger.info("Cycle {} SKIPPED - Bot is PAUSED", cycle_number)
            return
        if control.get("status") == "stopped":
            logger.info("Cycle {} STOPPED - Bot STOP command received", cycle_number)
            return

        self.current_cycle_number = cycle_number
        self.portfolio.current_cycle_number = cycle_number
        self.ai_service.current_cycle_number = cycle_number
        self.ai_service.invocation_count = self.invocation_count  # Sync invocation count
        # [SUCCESS] FIX: tick_cooldowns() must be called AFTER the prompt is generated
        # Because cooldown values are needed while generating the prompt
        # tick_cooldowns() decrements cooldowns, so it must be called AFTER the prompt
        self.market_data.clear_preloaded_indicators()
        self.portfolio.cycles_since_history_reset += 1
        self.maybe_reset_history(cycle_number)
        self.latest_bias_metrics = self.portfolio.get_directional_bias_metrics()
        self.portfolio.latest_bias_metrics = self.latest_bias_metrics
        prompt, thoughts, decisions = "N/A", "N/A", {}
        self.cycle_active = True
        cycle_timing: dict[str, float] = {}
        try:
            # Enhanced exit strategy control - pause during cycle
            # Enhanced exit strategy paused silently during cycle
            self.enhanced_exit_enabled = False

            # Track performance metrics every cycle
            self.track_performance_metrics(cycle_number)

            # Run performance analysis every 10 cycles or on critical conditions
            if self.should_run_performance_analysis(cycle_number):
                logger.info("Performance analysis (Cycle {})", cycle_number)
                from src.core.performance_monitor import PerformanceMonitor

                monitor = PerformanceMonitor()
                report = monitor.analyze_performance(last_n_cycles=10)
                monitor.print_performance_summary(report)

            # PHASE 27: AUTOMATED ML TRAINING
            self.auto_train_cycle_count += 1
            if self.auto_train_cycle_count >= constants.AUTO_TRAIN_CYCLE_INTERVAL:
                logger.info("Triggering 3-hour automated model retraining...")
                try:
                    import subprocess

                    # Run train_model.py in the background
                    venv_py = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
                    if not os.path.exists(venv_py):
                        venv_py = "python3"

                    cmd = [venv_py, os.path.join(PROJECT_ROOT, "scripts", "train_model.py")]
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logger.info(
                        "Training started in background. Hot-reload will trigger on completion."
                    )
                    self.auto_train_cycle_count = 0  # Reset counter
                except Exception as aml_err:
                    logger.warning("Failed to trigger automated training: {}", aml_err)

            logger.info("Fetching market data...")
            md_start = time.perf_counter()
            real_prices = self.market_data.get_all_real_prices()
            valid_prices = {
                k: v for k, v in real_prices.items() if isinstance(v, (int, float)) and v > 0
            }
            cycle_timing["market_data_ms"] = round((time.perf_counter() - md_start) * 1000, 2)
            if not valid_prices:
                raise ValueError("No valid market prices received.")

            # Check bot control before live account sync (can be slow)
            control = self._read_bot_control()
            if control.get("status") == "paused":
                logger.info("Cycle {} PAUSED before account sync - stopping cycle", cycle_number)
                self.cycle_active = False
                return
            if control.get("status") == "stopped":
                logger.info("Cycle {} STOPPED - Bot STOP command received", cycle_number)
                self.cycle_active = False
                return
            if self.portfolio.is_live_trading:
                self.account_service.sync_live_account()
            self.portfolio.update_prices(
                valid_prices,
                increment_loss_counters=True,
            )  # Update PnL before checking TP/SL

            # --- Auto TP/SL Check & Flash Exit Check (Atomic) ---
            # FIX: Race condition protection - use lock, combine checks to avoid double-close
            positions_closed_by_tp_sl = 0
            flash_exits_triggered = False

            with self.portfolio._lock:
                # Take a snapshot of current positions to iterate safely
                positions_snapshot = list(self.portfolio.positions.items())

                # First pass: Check TP/SL for all positions
                # Note: check_and_execute_tp_sl needs valid_prices dict
                # We call it without lock internally it also uses lock when modifying positions
                # But here we want to avoid concurrent modifications

            # Call TP/SL check outside lock to avoid holding lock during Binance API calls
            # The function internally uses _lock when modifying positions
            positions_closed_by_tp_sl = self.account_service.check_and_execute_tp_sl(valid_prices)

            # Second pass: Flash Exit Check - use fresh snapshot AFTER TP/SL modifications
            # Take another snapshot since TP/SL may have closed positions
            with self.portfolio._lock:
                current_positions = list(self.portfolio.positions.items())

            for coin, position in current_positions:
                # Skip if position was already closed by TP/SL in this cycle
                if coin not in self.portfolio.positions:
                    continue

                if coin in valid_prices:
                    if self.portfolio.check_flash_exit_conditions(coin, position):
                        logger.warning("EXECUTING FLASH EXIT for {}...", coin)
                        current_price = valid_prices[coin]

                        # Close position immediately
                        if self.portfolio.is_live_trading:
                            result = self.account_service.execute_live_close(
                                coin,
                                position,
                                current_price,
                                reason="Flash Exit (V-Reversal)",
                            )
                            if result.get("success"):
                                flash_exits_triggered = True
                                # Double-check position still exists before deletion
                                with self.portfolio._lock:
                                    if coin in self.portfolio.positions:
                                        del self.portfolio.positions[coin]
                        else:
                            # Paper trading close
                            self.account_service.close_position(
                                coin,
                                current_price,
                                reason="Flash Exit (V-Reversal)",
                            )
                            flash_exits_triggered = True

            if flash_exits_triggered:
                logger.info("Flash Exits triggered. Continuing cycle...")
            # --- End Flash Exit Check ---

            manual_override = self.portfolio.get_manual_override()
            auto_exit_triggered = bool(positions_closed_by_tp_sl)

            if manual_override:
                logger.info("APPLYING MANUAL OVERRIDE...")
                decisions = manual_override.get("decisions", {})
                thoughts = "Manual override."
                prompt = "N/A (Manual)"
                logger.info("MANUAL DECISIONS:\n{}", json.dumps(decisions, indent=2))
            # Only ask AI if no TP/SL triggered AND no manual override
            else:
                if auto_exit_triggered:
                    logger.info(
                        "Auto TP/SL/extended exit triggered earlier this cycle - proceeding with AI analysis.",
                    )
                # Check bot control before AI call (can be slow in live mode)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    logger.info("Cycle {} PAUSED before AI call - stopping cycle", cycle_number)
                    self.cycle_active = False
                    return
                if control.get("status") == "stopped":
                    logger.info("Cycle {} STOPPED - Bot STOP command received", cycle_number)
                    self.cycle_active = False
                    return
                # ZOMBIE ABORT: Check if watchdog fired while we were waiting
                if cancel_event and cancel_event.is_set():
                    logger.warning("Cycle {} cancelled before AI call.", cycle_number)
                    return

                ai_timer_start = time.perf_counter()
                logger.info("Generating prompt...")
                self.invocation_count += 1  # Increment AI call count
                # Use JSON prompt if enabled, with fallback to text format
                prompt = None
                prompt_format_used = "text"
                json_serialization_error = None

                if Config.USE_JSON_PROMPT:
                    try:
                        prompt = self.ai_service.generate_alpha_arena_prompt_json()
                        prompt_format_used = "json"
                        logger.info("JSON prompt format v{}", Config.JSON_PROMPT_VERSION)
                    except Exception as e:
                        json_serialization_error = str(e)
                        logger.warning("JSON prompt failed: {} -- falling back to text", e)
                        prompt = self.ai_service.generate_alpha_arena_prompt()
                        prompt_format_used = "json_fallback"
                else:
                    prompt = self.ai_service.generate_alpha_arena_prompt()
                logger.debug("Prompt size: {} chars, format: {}", len(prompt), prompt_format_used)

                # PHASE 10: AUTOMATED DATA COLLECTION (DEPRECATED - Moved to Bulk Sync in train_model.py)
                """
                try:
                    latest_indicators = getattr(self.ai_service, "latest_indicators", {})
                    for coin in self.market_data.available_coins:
                        # 1. Log Raw OHLCV Data (Market Data Table)
                        for interval in ["3m", "15m", HTF_INTERVAL]:
                            df_raw = self.market_data.get_cached_raw_dataframe(coin, interval)
                            if df_raw is not None and not df_raw.is_empty():
                                self.data_engine.log_market_data(df_raw, coin, interval)

                        # 2. Log Indicator Snapshots (Features Table)
                        if coin in latest_indicators and "15m" in latest_indicators[coin]:
                            self.data_engine.log_cycle_features(
                                coin, "15m", latest_indicators[coin]["15m"]
                            )
                except Exception as log_err:
                        logger.warning("Database logging failed: {}", log_err)
                """

                # Check bot control before AI API call (can be slow in live mode)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    logger.info("Cycle {} PAUSED before AI API call - stopping cycle", cycle_number)
                    self.cycle_active = False
                    return
                if control.get("status") == "stopped":
                    logger.info("Cycle {} STOPPED - Bot STOP command received", cycle_number)
                    self.cycle_active = False
                    return

                logger.info("Sending to API...")
                ai_response = self.deepseek.get_ai_decision(prompt)

                # ZOMBIE THREAD CHECK: If this thread woke up late and the main loop moved on, ABORT.
                if self.current_cycle_number != cycle_number:
                    logger.warning(
                        "Cycle {} woke up late! (Main loop is at {}). Aborting trades to prevent corruption.",
                        cycle_number,
                        self.current_cycle_number,
                    )
                    # Do NOT touch self.cycle_active here, as the new cycle might be using it.
                    return

                # Check bot control after AI API call (may have taken time in live mode)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    logger.info("Cycle {} PAUSED after AI call - stopping cycle", cycle_number)
                    self.cycle_active = False
                    return
                if control.get("status") == "stopped":
                    logger.info("Cycle {} STOPPED - Bot STOP command received", cycle_number)
                    self.cycle_active = False
                    return

                parsed_result = self.ai_service.parse_ai_response(ai_response)
                match parsed_result:
                    case (parsed_response, None):
                        thoughts = parsed_response.get("chain_of_thoughts", "Parse Error.")
                        decisions = parsed_response.get("decisions", {})
                    case (None, error_msg):
                        logger.error("AI response parse failed: {}", error_msg)
                        thoughts = error_msg
                        decisions = {}

                if auto_exit_triggered and isinstance(thoughts, str):
                    thoughts += "\n[Auto Exit Note: TP/SL or extended-loss closure executed before this analysis]"
                cycle_timing["ai_ms"] = round((time.perf_counter() - ai_timer_start) * 1000, 2)

                if not isinstance(decisions, dict):
                    logger.error("AI decisions not dict ({}). Resetting.", type(decisions))
                    thoughts += "\nError: Decisions not dict."
                    decisions = {}

                if isinstance(thoughts, str):
                    thoughts = thoughts.replace("\\n", "\n")
                logger.debug("REASONING: {} chars, {} decisions", len(thoughts), len(decisions))
                logger.bind(kind="ai_reasoning").debug("REASONING FULL:\n{}", thoughts)
                logger.info(
                    "DECISIONS:\n{}",
                    json.dumps(decisions, indent=2) if decisions else "{}",
                )

                # GRADUAL POSITION SYSTEM: Cycle-based position limit
                max_positions_for_cycle = self.ai_service.get_max_positions_for_cycle(cycle_number)
                current_positions = len(self.portfolio.positions)

                if current_positions >= max_positions_for_cycle:
                    logger.warning(
                        "Position limit reached: max {} for cycle {}",
                        max_positions_for_cycle,
                        cycle_number,
                    )
                    # If position limit is reached, convert new entry signals to hold
                    filtered_decisions = {}
                    for coin, trade in decisions.items():
                        if isinstance(trade, dict):
                            signal = trade.get("signal")
                            if signal in ["buy_to_enter", "sell_to_enter"]:
                                logger.info(
                                    "{} {} -> HOLD (limit)",
                                    coin,
                                    signal,
                                )
                                filtered_decisions[coin] = {
                                    "signal": "hold",
                                    "justification": f"Position limit reached - Cycle {cycle_number} (max {max_positions_for_cycle} positions)",
                                }
                            else:
                                filtered_decisions[coin] = trade
                        else:
                            filtered_decisions[coin] = trade
                    decisions = filtered_decisions
                    thoughts += f"\n[Position Limit: Cycle {cycle_number} - Max {max_positions_for_cycle} positions allowed]"
                decisions, directional_filtered = self._apply_directional_capacity_filter(decisions)
                if directional_filtered:
                    thoughts += "\n[Directional Capacity: Entry signals converted to HOLD until opposite-side exposure is reduced]"
            # Execute AI decisions only if it's a valid dict and NOT empty AND no manual override was active
            execution_elapsed = None
            if isinstance(decisions, dict) and decisions and not manual_override:
                # Check bot control before execution (live mode can be slow)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    logger.info("Cycle {} PAUSED before execution - stopping cycle", cycle_number)
                    self.cycle_active = False
                    return
                if control.get("status") == "stopped":
                    logger.info("Cycle {} STOPPED - Bot STOP command received", cycle_number)
                    self.cycle_active = False
                    return
                # ZOMBIE ABORT: Check if watchdog fired before we start executing trades
                if cancel_event and cancel_event.is_set():
                    logger.warning("Cycle {} cancelled before execution.", cycle_number)
                    return

                exec_start = time.perf_counter()
                # AI PRIORITY SYSTEM: If "close_position" signal exists, the position is closed
                has_close_position_signal = any(
                    trade.get("signal") == "close_position"
                    for trade in decisions.values()
                    if isinstance(trade, dict)
                )

                close_execution_report = {
                    "executed": [],
                    "blocked": [],
                    "skipped": [],
                    "holds": [],
                    "notes": [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                if has_close_position_signal:
                    logger.info("CLOSE_POSITION signal -- executing closes")
                    # Close only the coins with close_position signal
                    for coin, trade in decisions.items():
                        if not isinstance(trade, dict):
                            continue
                        if (
                            trade.get("signal") == "close_position"
                            and coin in self.portfolio.positions
                        ):
                            if coin in valid_prices:
                                position = self.portfolio.positions[coin]
                                current_price = valid_prices[coin]
                                direction = position.direction
                                entry_price = position.entry_price
                                quantity = position.quantity
                                margin_used = position.margin_usd

                                # AI Panic Exit Preventer (Minimum Hold Shield)
                                entry_time_str = position.entry_time
                                if entry_time_str:
                                    try:
                                        entry_dt = datetime.fromisoformat(entry_time_str)
                                        held_minutes = max(
                                            0,
                                            (datetime.now(timezone.utc) - entry_dt).total_seconds()
                                            / 60,
                                        )
                                        if held_minutes < 12.0:
                                            logger.info(
                                                "Panic exit prevented for {} (Held: {}m). Minimum hold is 12m.",
                                                coin,
                                                int(held_minutes),
                                            )
                                            close_execution_report["blocked"].append(
                                                {
                                                    "coin": coin,
                                                    "reason": "minimum_hold_prevented",
                                                    "held_minutes": int(held_minutes),
                                                }
                                            )
                                            continue
                                    except (ValueError, TypeError, AttributeError):
                                        pass

                                # AI Panic Exit Validator — blocks weak exits
                                # between -1.5% and +2% without technical confirmation
                                try:
                                    inds_3m = self.market_data.get_technical_indicators(coin, "3m")
                                    if "error" not in inds_3m:
                                        exit_valid = self.portfolio.validate_exit_signal(
                                            coin, position, inds_3m
                                        )
                                        if not exit_valid:
                                            logger.info(
                                                "AI close blocked — weak exit for {} (PnL: ${:.2f})",
                                                coin,
                                                position.unrealized_pnl,
                                            )
                                            close_execution_report["blocked"].append(
                                                {
                                                    "coin": coin,
                                                    "reason": "weak_exit_validation_failed",
                                                    "unrealized_pnl": position.unrealized_pnl,
                                                }
                                            )
                                            continue
                                except Exception as e:
                                    logger.debug("Exit validation skipped for {}: {}", coin, e)

                                if self.portfolio.is_live_trading:
                                    live_result = self.account_service.execute_live_close(
                                        coin=coin,
                                        position=position,
                                        current_price=current_price,
                                        reason="AI close_position signal",
                                    )
                                    if not live_result.get("success"):
                                        error_msg = live_result.get("error", "unknown_error")
                                        logger.error("Live close failed: {} ({})", coin, error_msg)
                                        close_execution_report["blocked"].append(
                                            {
                                                "coin": coin,
                                                "reason": "live_close_failed",
                                                "error": error_msg,
                                            },
                                        )
                                    else:
                                        history_entry = live_result.get("history_entry")
                                        if history_entry:
                                            self.portfolio.add_to_history(history_entry)
                                        close_execution_report["executed"].append(
                                            {
                                                "coin": coin,
                                                "signal": "close_position",
                                                "pnl": live_result.get("pnl"),
                                                "direction": direction,
                                                "price": current_price,
                                                "mode": "live",
                                            },
                                        )
                                        logger.info(
                                            "Closed {} {} @ ${} (PnL: ${}) [LIVE]",
                                            direction,
                                            coin,
                                            format_num(current_price, 4),
                                            format_num(live_result.get("pnl", 0), 2),
                                        )
                                    continue

                                # FIX: Hold lock for balance update + position deletion to prevent
                                # race with TP/SL thread closing the same position.
                                with self.portfolio._lock:
                                    if coin not in self.portfolio.positions:
                                        close_execution_report["skipped"].append(
                                            {"coin": coin, "reason": "position_already_closed"}
                                        )
                                        continue
                                    if direction == "long":
                                        profit = (current_price - entry_price) * quantity
                                    else:
                                        profit = (entry_price - current_price) * quantity

                                    # Deduct commission for simulation realism (round-trip: entry + exit)
                                    notional = (entry_price + current_price) / 2 * quantity
                                    commission = (
                                        notional
                                        * Config.SIMULATION_COMMISSION_RATE
                                        * constants.COMMISSION_ROUND_TRIP_MULTIPLIER
                                    )
                                    profit -= commission

                                    self.portfolio.current_balance += margin_used + profit

                                    logger.info(
                                        "Closed {} {} @ ${} (PnL: ${})",
                                        direction,
                                        coin,
                                        format_num(current_price, 4),
                                        format_num(profit, 2),
                                    )

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
                                        "close_reason": "AI close_position signal",
                                    }
                                    del self.portfolio.positions[coin]

                                self.portfolio.add_to_history(history_entry)

                                close_execution_report["executed"].append(
                                    {
                                        "coin": coin,
                                        "signal": "close_position",
                                        "pnl": profit,
                                        "direction": direction,
                                        "price": current_price,
                                    },
                                )
                            else:
                                close_execution_report["skipped"].append(
                                    {"coin": coin, "reason": "no_price_data_for_close"},
                                )
                        elif isinstance(trade, dict):
                            close_execution_report["holds"].append(
                                {"coin": coin, "has_position": coin in self.portfolio.positions},
                            )

                    # Combine with normal execution report for logging
                    previous_report = getattr(self.portfolio, "last_execution_report", {})
                    merged_report = {
                        "executed": (
                            previous_report.get("executed", []) + close_execution_report["executed"]
                        ),
                        "blocked": (
                            previous_report.get("blocked", []) + close_execution_report["blocked"]
                        ),
                        "skipped": (
                            previous_report.get("skipped", []) + close_execution_report["skipped"]
                        ),
                        "holds": close_execution_report["holds"]
                        or previous_report.get("holds", []),
                        "notes": previous_report.get("notes", []) + close_execution_report["notes"],
                        "timestamp": close_execution_report["timestamp"],
                    }
                    self.portfolio.last_execution_report = merged_report

                    # Process other AI decisions (new positions only)
                    self.account_service._execute_new_positions_only(
                        decisions,
                        valid_prices,
                        cycle_number,
                        indicator_cache=self.latest_indicator_cache,
                    )
                else:
                    # Normal decision processing (partial profit active)
                    self.portfolio._execute_normal_decisions(
                        decisions,
                        valid_prices,
                        cycle_number,
                        positions_closed_by_tp_sl,
                        indicator_cache=self.latest_indicator_cache,
                    )
                execution_elapsed = time.perf_counter() - exec_start

            # Execute manual override decisions if present
            elif isinstance(decisions, dict) and decisions and manual_override:
                exec_start = time.perf_counter()
                self.portfolio.execute_decision(
                    decisions,
                    valid_prices,
                    indicator_cache=self.latest_indicator_cache,
                )
                execution_elapsed = time.perf_counter() - exec_start

            elif isinstance(decisions, dict):
                logger.info("No trading actions this cycle.")

            if execution_elapsed is not None:
                cycle_timing["execution_ms"] = round(execution_elapsed * 1000, 2)

            # Save state and history at the end of the cycle
            self.portfolio.save_state()
            # Log regardless of errors (log contains error info if applicable)
            execution_report = getattr(self.portfolio, "last_execution_report", {})
            if manual_override:
                cycle_status = "manual_override"
            elif positions_closed_by_tp_sl and not decisions:
                cycle_status = "tp_sl_only"
            elif isinstance(decisions, dict) and decisions:
                cycle_status = "ai_decision"
            else:
                cycle_status = "idle"

            cycle_metadata: dict[str, Any] = {
                "positions_closed_by_tp_sl": bool(positions_closed_by_tp_sl),
                "manual_override": bool(manual_override),
                "cooldown_status": {
                    "directional_cooldowns": dict(self.portfolio.directional_cooldowns),
                    "relaxed_countertrend_cycles": self.portfolio.relaxed_countertrend_cycles,
                    "counter_trend_cooldown": self.portfolio.counter_trend_cooldown,
                },
                "prompt_format": prompt_format_used if "prompt_format_used" in locals() else "text",
                "json_serialization_error": json_serialization_error
                if "json_serialization_error" in locals()
                else None,
            }
            if execution_report:
                cycle_metadata["execution_report"] = execution_report
            if cycle_timing:
                cycle_metadata["performance"] = cycle_timing
                timing_summary = []
                if "market_data_ms" in cycle_timing:
                    timing_summary.append(f"market {cycle_timing['market_data_ms']:.2f}ms")
                if "ai_ms" in cycle_timing:
                    timing_summary.append(f"ai {cycle_timing['ai_ms']:.2f}ms")
                if "execution_ms" in cycle_timing:
                    timing_summary.append(f"exec {cycle_timing['execution_ms']:.2f}ms")
                if timing_summary:
                    logger.info("Timers: {}", " | ".join(timing_summary))

            self.portfolio.add_to_cycle_history(
                {
                    "cycle_number": cycle_number,
                    "prompt": prompt,
                    "thoughts": thoughts,
                    "decisions": decisions,
                    "status": cycle_status,
                    "metadata": cycle_metadata or None,
                }
            )

            # tick_cooldowns() MUST be called AFTER the prompt is generated
            # because the values are needed during prompt generation.
            # tick_cooldowns() decrements them, so call it AFTER.
            if hasattr(self.portfolio, "tick_cooldowns"):
                self.portfolio.tick_cooldowns()

            # Enhanced exit strategy control - re-enable after cycle completion
            # Enhanced exit strategy re-enabled silently after cycle completion
            self.show_status()

        except Exception as e:
            logger.error("CRITICAL CYCLE ERROR: {}", e)
            traceback.print_exc()
            try:
                decisions_log = decisions if isinstance(decisions, dict) else {}
                self.portfolio.add_to_cycle_history(
                    {
                        "cycle_number": cycle_number,
                        "prompt": prompt,
                        "thoughts": f"CRITICAL CYCLE ERROR: {e}\n{traceback.format_exc()}",
                        "decisions": decisions_log,
                        "status": "error",
                        "metadata": {"exception": str(e)},
                    }
                )
            except Exception as log_e:
                logger.error("Failed to save error to cycle history: {}", log_e)
        finally:
            self.cycle_active = False
            self.enhanced_exit_enabled = True
            # Check bot control after exception - if paused/stopped, don't continue
            try:
                control = self._read_bot_control()
                if control.get("status") == "stopped":
                    logger.info(
                        "Cycle {} exception handler: Bot STOP command received",
                        cycle_number,
                    )
                    raise SystemExit("Bot stopped by user command")
            except SystemExit:
                raise
            except Exception as control_e:
                logger.warning("Failed to check bot control after exception: {}", control_e)

    def show_status(self):
        """Show current status in the console"""
        logger.info("--- STATUS ---")
        with self.portfolio._lock:
            balance = self.portfolio.current_balance
            total_value = self.portfolio.total_value
            total_return = self.portfolio.total_return
            trade_count = self.portfolio.trade_count
            positions_snapshot = list(self.portfolio.positions.items())
        logger.info(
            "Value: ${} | Return: {}% | Cash: ${} | Trades: {}",
            format_num(total_value, 2),
            format_num(total_return, 2),
            format_num(balance, 2),
            trade_count,
        )
        logger.info("Positions ({} open):", len(positions_snapshot))
        if not positions_snapshot:
            logger.info("  No open positions.")
        else:
            for coin, pos in positions_snapshot:
                pnl = pos.unrealized_pnl
                pnl_sign = "+" if pnl >= 0 else ""
                direction = pos.direction.upper()
                leverage = pos.leverage
                notional = pos.notional_usd
                liq = pos.liquidation_price
                entry = pos.entry_price
                qty = pos.quantity
                logger.info(
                    "  {} ({} {}x): {} units | Notional ${} | Entry: ${} | PnL: {}${} | Liq Est: ${}",
                    coin,
                    direction,
                    leverage,
                    format_num(qty, 4),
                    format_num(notional, 2),
                    format_num(entry, 4),
                    pnl_sign,
                    format_num(pnl, 2),
                    format_num(liq, 4),
                )

    def start_tp_sl_monitoring(self):
        """Start TP/SL monitoring timer that runs every 1 minute"""
        if self.tp_sl_timer and self.tp_sl_timer.is_alive():
            logger.info("TP/SL monitoring already running")
            return

        self.is_running = True
        self.tp_sl_timer = threading.Thread(target=self._tp_sl_monitoring_loop, daemon=True)
        self.tp_sl_timer.start()
        logger.info("TP/SL monitoring started (30s interval)")

    def stop_tp_sl_monitoring(self):
        """Stop TP/SL monitoring timer"""
        self.is_running = False
        if self.tp_sl_timer and self.tp_sl_timer.is_alive():
            self.tp_sl_timer.join(timeout=5)
            logger.info("TP/SL monitoring stopped")
        else:
            logger.info("TP/SL monitoring was not running")

    def _tp_sl_monitoring_loop(self):
        """Background thread that checks TP/SL every 30 seconds"""
        # TP/SL monitoring loop started silently
        while self.is_running:
            try:
                # Check bot control file for pause/stop command
                control = self._read_bot_control()
                if control.get("status") == "stopped":
                    logger.info(
                        "TP/SL monitoring: STOP command received. Stopping monitoring loop..."
                    )
                    self.is_running = False
                    break
                if control.get("status") == "paused":
                    logger.info("TP/SL monitoring: Bot is PAUSED. Waiting for resume...")
                    # Wait in smaller intervals to check for resume with safety timeout
                    pause_timeout = 300  # 5 minutes max in paused state
                    pause_iterations = 0
                    while pause_iterations < pause_timeout // 10:
                        time.sleep(10)
                        pause_iterations += 1
                        try:
                            control = self._read_bot_control()
                        except Exception:
                            # FIX: Handle corrupted control file gracefully
                            logger.warning("Failed to read control file, retrying...")
                            continue
                        if control.get("status") == "running":
                            logger.info("TP/SL monitoring: Bot RESUMED. Continuing monitoring...")
                            break
                        if control.get("status") == "stopped":
                            logger.info(
                                "TP/SL monitoring: STOP command received. Stopping monitoring loop..."
                            )
                            self.is_running = False
                            break
                    else:
                        # Timeout reached - auto-resume to prevent hanging
                        logger.info("Pause timeout reached, auto-resuming...")
                    if control.get("status") == "stopped":
                        break
                    continue

                # Enhanced exit strategy control - check if enabled
                if getattr(self, "cycle_active", False):
                    # Trading cycle active; wait until it completes
                    for _ in range(5):
                        if not self.is_running:
                            break
                        # Check bot control during wait
                        control = self._read_bot_control()
                        if control.get("status") == "stopped":
                            self.is_running = False
                            break
                        if not getattr(self, "cycle_active", False):
                            break
                        time.sleep(1)
                    if control.get("status") == "stopped":
                        break
                    continue

                if not self.enhanced_exit_enabled:
                    # Enhanced exit strategy paused - TP/SL monitoring waiting silently
                    # Wait 10 seconds and check again
                    for _ in range(10):
                        if not self.is_running:
                            break
                        # Check bot control during wait
                        control = self._read_bot_control()
                        if control.get("status") == "stopped":
                            self.is_running = False
                            break
                        time.sleep(1)
                    if control.get("status") == "stopped":
                        break
                    continue

                # Get current prices
                real_prices = self.market_data.get_all_real_prices()
                valid_prices = {
                    k: v for k, v in real_prices.items() if isinstance(v, (int, float)) and v > 0
                }

                if valid_prices:
                    # Update portfolio prices
                    self.portfolio.update_prices(valid_prices, increment_loss_counters=False)

                    # Flash Exit Check (V-Reversal) - now runs every 20 seconds
                    # FIX: Thread-safe access to positions with lock
                    if Config.FLASH_EXIT_ENABLED:
                        with self.portfolio._lock:
                            positions_snapshot = list(self.portfolio.positions.items())

                        flash_exits_triggered = False
                        for coin, position in positions_snapshot:
                            if self.portfolio.check_flash_exit_conditions(coin, position):
                                current_price = valid_prices.get(coin)
                                if current_price:
                                    if self.portfolio.is_live_trading:
                                        live_result = self.account_service.execute_live_close(
                                            coin=coin,
                                            position=position,
                                            current_price=current_price,
                                            reason="Flash Exit (V-Reversal) - 20s monitor",
                                        )
                                        if live_result.get("success"):
                                            history_entry = live_result.get("history_entry")
                                            if history_entry:
                                                self.portfolio.add_to_history(history_entry)
                                            with self.portfolio._lock:
                                                if coin in self.portfolio.positions:
                                                    del self.portfolio.positions[coin]
                                            flash_exits_triggered = True
                                    else:
                                        # Paper trading close
                                        self.account_service.close_position(
                                            coin,
                                            current_price,
                                            reason="Flash Exit (V-Reversal) - 20s monitor",
                                        )
                                        flash_exits_triggered = True
                        if flash_exits_triggered:
                            logger.warning("Flash exit: V-Reversal detected and closed")

                    # Run TP/SL check - all decisions made by 20-second monitoring (like simulation mode)
                    # No Binance TP/SL orders - all managed by monitoring loop
                    positions_closed = self.account_service.check_and_execute_tp_sl(valid_prices)

                    if positions_closed:
                        logger.info("20-SECOND TP/SL CHECK: Positions closed")
                    elif self.portfolio.positions:
                        logger.debug(
                            "20-SECOND TP/SL CHECK: {} positions monitored, no trigger",
                            len(self.portfolio.positions),
                        )
                else:
                    logger.warning("TP/SL monitoring: No valid prices")

                # Check bot control before sleep
                control = self._read_bot_control()
                if control.get("status") == "stopped":
                    logger.info(
                        "TP/SL monitoring: STOP command received. Stopping monitoring loop..."
                    )
                    self.is_running = False
                    break

            except Exception as e:
                logger.error("TP/SL monitoring error: {}", e)
                # Check bot control after exception
                try:
                    control = self._read_bot_control()
                    if control.get("status") == "stopped":
                        logger.info("TP/SL monitoring: STOP command received after exception")
                        self.is_running = False
                        break
                except Exception as control_e:
                    logger.warning(
                        "Failed to check bot control after TP/SL exception: {}", control_e
                    )

            # Wait 30 seconds before next check
            if self.is_running:
                time.sleep(20)

    def run_simulation(self, total_duration_hours: int = 168, cycle_interval_minutes: int = 2):
        """Run the simulation with dynamic cycle frequency and TP/SL monitoring"""
        logger.info("ALPHA ARENA v{}", VERSION)
        logger.info(
            "Budget: ${} | Duration: {}h | Coins: {}",
            format_num(self.portfolio.initial_balance, 2),
            total_duration_hours,
            ", ".join(self.market_data.available_coins),
        )
        logger.info("Dynamic cycle: 2-4 min | TP/SL monitor: 30s interval")

        # Start TP/SL monitoring
        self.start_tp_sl_monitoring()

        end_time = datetime.now(timezone.utc) + timedelta(hours=total_duration_hours)
        # Calculate correct cycle number: reset_cycle + cycles_since_reset
        last_reset = getattr(self.portfolio, "last_history_reset_cycle", 0) or 0
        cycles_since_reset = len(self.portfolio.cycle_history)
        start_cycle = last_reset + cycles_since_reset + 1
        logger.info("Resuming from cycle {}", start_cycle)
        self.invocation_count = start_cycle - 1
        current_cycle_number = start_cycle - 1

        # Initialize bot control file
        self._write_bot_control(
            {"status": "running", "last_updated": datetime.now(timezone.utc).isoformat()}
        )

        try:
            while datetime.now(timezone.utc) < end_time:
                # Check bot control file for pause/stop command BEFORE starting cycle
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    logger.info(
                        "Bot is PAUSED. Waiting for resume command... (checking every 10 seconds)"
                    )
                    # Wait in smaller intervals to check for resume with safety timeout
                    pause_timeout = 300  # 5 minutes max in paused state
                    pause_iterations = 0
                    while pause_iterations < pause_timeout // 10:
                        time.sleep(10)
                        pause_iterations += 1
                        try:
                            control = self._read_bot_control()
                        except Exception:
                            logger.warning("Failed to read control file, retrying...")
                            continue
                        if control.get("status") == "running":
                            logger.info("Bot RESUMED. Continuing trading cycles...")
                            break
                        if control.get("status") == "stopped":
                            logger.info("Bot STOP command received. Stopping gracefully...")
                            break
                    else:
                        logger.info("Pause timeout reached, auto-resuming...")
                    if control.get("status") == "stopped":
                        break
                    continue
                if control.get("status") == "stopped":
                    logger.info("Bot STOP command received. Stopping gracefully...")
                    break

                current_cycle_number += 1
                self.current_cycle_number = current_cycle_number
                cycle_start_time = time.time()

                # Check MAX_CYCLES limit - auto-stop at configured cycle number
                if Config.MAX_CYCLES > 0 and current_cycle_number > Config.MAX_CYCLES:
                    logger.info(
                        "MAX_CYCLES limit reached ({}). Auto-stopping bot...", Config.MAX_CYCLES
                    )
                    break

                # Calculate dynamic cycle frequency
                dynamic_cycle_interval = self.calculate_optimal_cycle_frequency()
                logger.info(
                    "Cycle interval: {}s ({:.1f} min)",
                    dynamic_cycle_interval,
                    dynamic_cycle_interval / 60,
                )

                # --- CYCLE WATCHDOG (Hard Kill via Daemon Thread) ---
                # Use a native daemon thread. Unlike ThreadPoolExecutor, daemon threads
                # do NOT block the Python interpreter from exiting (Ctrl+C works perfectly).
                import threading

                watchdog_timeout = max(dynamic_cycle_interval - 10, 30)  # Min 30s for safety

                class CycleTask:
                    def __init__(self, cancel_event: threading.Event):
                        self.error = None
                        self.cancel_event = cancel_event

                    def run(self, instance, cycle_num):
                        try:
                            instance.run_trading_cycle(cycle_num, cancel_event=self.cancel_event)
                        except Exception as e:
                            self.error = e

                cancel_event = threading.Event()
                task = CycleTask(cancel_event)
                cycle_thread = threading.Thread(
                    target=task.run, args=(self, current_cycle_number), daemon=True
                )
                cycle_thread.start()
                cycle_thread.join(timeout=watchdog_timeout)

                if cycle_thread.is_alive():
                    # Thread is still running after timeout -> HUNG
                    logger.warning(
                        "Cycle {} timed out after {}s!", current_cycle_number, watchdog_timeout
                    )
                    logger.warning("Force-closing cycle and moving to next interval...")
                    # Signal the abandoned thread to stop at next checkpoint
                    cancel_event.set()
                    # Cleanup to allow next cycle to resume cleanly
                    self.cycle_active = False
                    self.enhanced_exit_enabled = True
                    # The thread is daemonized, so we just abandon it. It will die with the process.
                elif task.error:
                    # Thread finished with an error
                    logger.error(
                        "Cycle {} failed with internal error: {}", current_cycle_number, task.error
                    )
                # --- END WATCHDOG ---

                if datetime.now(timezone.utc) >= end_time:
                    break
                elapsed_time = time.time() - cycle_start_time
                sleep_time = max(0, dynamic_cycle_interval - elapsed_time)
                logger.info(
                    "Cycle {} done in {}s. Next in {} min.",
                    current_cycle_number,
                    format_num(elapsed_time, 2),
                    format_num(sleep_time / 60, 2),
                )
                time.sleep(max(sleep_time, 0.5))

                # Check bot control file AFTER sleep (before next cycle)
                control = self._read_bot_control()
                if control.get("status") == "paused":
                    logger.info(
                        "Bot is PAUSED. Waiting for resume command... (checking every 10 seconds)"
                    )
                    # Wait in smaller intervals to check for resume with safety timeout
                    pause_timeout = 300  # 5 minutes max in paused state
                    pause_iterations = 0
                    while pause_iterations < pause_timeout // 10:
                        time.sleep(10)
                        pause_iterations += 1
                        try:
                            control = self._read_bot_control()
                        except Exception:
                            # FIX: Handle corrupted control file gracefully
                            logger.warning("Failed to read control file, retrying...")
                            continue
                        if control.get("status") == "running":
                            logger.info("Bot RESUMED. Continuing trading cycles...")
                            break
                        if control.get("status") == "stopped":
                            logger.info("Bot STOP command received. Stopping gracefully...")
                            break
                    else:
                        logger.info("Pause timeout reached, auto-resuming...")
                    if control.get("status") == "stopped":
                        break
                    continue
                if control.get("status") == "stopped":
                    logger.info("Bot STOP command received. Stopping gracefully...")
                    break

        except KeyboardInterrupt:
            logger.info("Program stopped by user.")
        finally:
            # Stop TP/SL monitoring
            self.stop_tp_sl_monitoring()

        logger.info("=" * 60)
        logger.info("SIMULATION COMPLETED")
        logger.info("=" * 60)
        self.show_status()

    def _adjust_partial_sale_for_min_limit(
        self, position: Position, proposed_percent: float
    ) -> float:
        """Adjust partial sale percentage to ensure minimum limit remains after sale"""
        current_margin = position.margin_usd

        # Calculate dynamic minimum limit: $15 fixed OR 10% of available cash, whichever is larger
        min_remaining = self.portfolio._calculate_dynamic_minimum_limit()

        if current_margin <= min_remaining:
            # Position already at or below minimum, don't sell
            logger.info(
                "Partial sale blocked: Position margin ${:.2f} <= minimum limit ${:.2f}",
                current_margin,
                min_remaining,
            )
            return 0.0

        # Calculate remaining margin after proposed sale
        remaining_after_proposed = current_margin * (1 - proposed_percent)

        if remaining_after_proposed >= min_remaining:
            # Proposed sale keeps us above minimum, use as-is
            return proposed_percent
        # Adjust sale to leave exactly min_remaining margin
        adjusted_sale_amount = current_margin - min_remaining
        adjusted_percent = adjusted_sale_amount / current_margin

        logger.info(
            "Adjusted partial sale: {:.0f}% -> {:.0f}% to maintain ${:.2f} minimum limit",
            proposed_percent * 100,
            adjusted_percent * 100,
            min_remaining,
        )
        return adjusted_percent

    def update_trend_state(
        self,
        coin: str,
        indicators_htf: dict[str, Any],
        indicators_15m: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Delegate trend state updates to PortfolioManager for backward compatibility."""
        return self.portfolio.update_trend_state(coin, indicators_htf, indicators_15m)

    def get_recent_trend_flip_summary(self) -> list[str]:
        """Expose portfolio trend flip summary for existing integrations."""
        return self.portfolio.get_recent_trend_flip_summary()

    def count_positions_by_direction(self) -> dict[str, int]:
        return self.portfolio.count_positions_by_direction()

    def _read_bot_control(self) -> dict[str, Any]:
        """Read bot control file to check for pause/stop commands using memcache."""
        try:
            return safe_file_read_cached(
                "data/bot_control.json",
                {"status": "running", "last_updated": datetime.now(timezone.utc).isoformat()},
            )
        except Exception as e:
            logger.warning("Failed to read bot_control.json: {}", e)
            # Return default running state if file read fails (fail-safe)
            return {"status": "running", "last_updated": datetime.now(timezone.utc).isoformat()}

    def _write_bot_control(self, data: dict[str, Any]):
        """Write bot control file."""
        safe_file_write("data/bot_control.json", data)

    def apply_directional_bias(
        self,
        signal: str,
        confidence: float,
        bias_metrics: dict[str, dict[str, Any]],
        current_trend: str,
    ) -> float:
        return self.portfolio.apply_directional_bias(
            {
                "signal": signal,
                "confidence": confidence,
                "bias_metrics": bias_metrics,
                "current_trend": current_trend,
            },
        )

    def get_directional_bias_metrics(self) -> dict[str, dict[str, Any]]:
        """Proxy to portfolio directional bias metrics."""
        return self.portfolio.get_directional_bias_metrics()

    def add_to_cycle_history(
        self,
        cycle_number: int,
        prompt: str,
        thoughts: str,
        decisions: dict,
        status: str = "ai_decision",
        metadata: dict[str, Any] | None = None,
    ):
        return self.portfolio.add_to_cycle_history(
            {
                "cycle_number": cycle_number,
                "prompt": prompt,
                "thoughts": thoughts,
                "decisions": decisions,
                "status": status,
                "metadata": metadata,
            }
        )


# Define VERSION at the top level
VERSION = "9 - Auto TP/SL, Dynamic Size, Prompt Eng"


def main():
    """Main application entry point"""
    setup_logging(log_level=Config.LOG_LEVEL, log_dir=Config.LOG_DIR)
    try:
        cleanup_stale_temp_files()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("No DEEPSEEK_API_KEY found. Running simulation mode...")
        arena = AlphaArenaDeepSeek(api_key)
        arena.run_simulation(total_duration_hours=168, cycle_interval_minutes=2)
    except KeyboardInterrupt:
        logger.info("Program stopped by user.")
    except Exception as e:
        logger.critical("Unexpected critical error in main: {}", e)
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
