"""Alpha Arena System Reset Script

Clears all runtime data for a fresh start.
Preserves: market_data.db, ML models, .env, config files.
"""

import glob
import os
import shutil

from loguru import logger

# Runtime data files to delete
TARGET_FILES = [
    "data/portfolio_state.json",
    "data/trade_history.json",
    "data/full_trade_history.json",
    "data/cycle_history.json",
    "data/ml_predictions.jsonl",
    "data/performance_report.json",
    "data/performance_history.json",
    "data/bot_control.json",
    "data/reset_log.json",
    "data/llm_benchmark_test.json",
]

# Folders to clear (delete contents, recreate empty)
TARGET_FOLDERS = [
    "data/backups",
    "history_backups",
    "data/logs",
]


def reset_system() -> None:
    logger.info("=== Alpha Arena System Reset ===")

    deleted = 0
    failed = 0

    # 1. Delete specific files
    for f in TARGET_FILES:
        if os.path.exists(f):
            try:
                os.remove(f)
                logger.info("Deleted: {}", f)
                deleted += 1
            except Exception as e:
                logger.error("Failed to delete {}: {}", f, e)
                failed += 1

    # 2. Delete backup files (performance_report.backup_*)
    for backup in glob.glob("data/performance_report.backup_*"):
        try:
            os.remove(backup)
            logger.info("Deleted backup: {}", backup)
            deleted += 1
        except Exception as e:
            logger.error("Failed to delete {}: {}", backup, e)
            failed += 1

    # 3. Clear folders (delete + recreate)
    for folder in TARGET_FOLDERS:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                os.makedirs(folder, exist_ok=True)
                logger.info("Cleared folder: {}", folder)
                deleted += 1
            except Exception as e:
                logger.error("Failed to clear folder {}: {}", folder, e)
                failed += 1

    # 4. Safe zone — preserved files
    logger.info("Preserved: data/market_data.db, models/*.xgb, .env, config/")

    if failed == 0:
        logger.success("System reset complete. {} items cleared. Ready for fresh start.", deleted)
    else:
        logger.warning("Reset done with {} failures. {} items cleared.", failed, deleted)


if __name__ == "__main__":
    confirm = input("This will reset all trade history and portfolio state. Are you sure? (y/n): ")
    if confirm.strip().lower() == "y":
        reset_system()
    else:
        logger.info("Reset cancelled.")
