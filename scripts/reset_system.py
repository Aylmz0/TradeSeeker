import os
import shutil
import glob

# Alpha Arena Reset Script
# This script clears all runtime data for a fresh start.

target_files = [
    "data/portfolio_state.json",
    "data/trade_history.json",
    "data/full_trade_history.json",
    "data/cycle_history.json",
    "data/ml_predictions.jsonl",
    "data/performance_report.json",
    "data/performance_history.json",
    "data/bot_control.json"
]

def reset_system():
    print("--- Alpha Arena System Reset ---")
    
    # 1. Delete specific trackable JSON files
    for f in target_files:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"[OK] Deleted: {f}")
            except Exception as e:
                print(f"[ERR] Failed to delete {f}: {e}")
    
    # 2. Clear backups
    if os.path.exists("data/backups"):
        try:
            shutil.rmtree("data/backups")
            os.makedirs("data/backups")
            print("[OK] Cleared Backups")
        except Exception as e:
            print(f"[ERR] Failed to clear backups: {e}")

    # 3. Market data (Optional - keeping by default but user can uncomment)
    # db_path = "data/market_data.db"
    # if os.path.exists(db_path):
    #     os.remove(db_path)
    #     print(f"[OK] Deleted Market Database: {db_path}")

    print("\n[SUCCESS] System is clean. Ready for a fresh start.")
    print("Next: Run 'python3 src/main.py' to begin first cycle.")

if __name__ == "__main__":
    confirm = input("This will reset all trade history and portfolio state. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        reset_system()
    else:
        print("Reset cancelled.")
