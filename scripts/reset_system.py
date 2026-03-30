import os
import shutil


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
    "data/bot_control.json",
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

    # 2. Clear backups and history_backups
    for folder in ["data/backups", "history_backups"]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                os.makedirs(folder, exist_ok=True)
                print(f"[OK] Cleared folder: {folder}")
            except Exception as e:
                print(f"[ERR] Failed to clear folder {folder}: {e}")

    # 3. KORUNAN VERİLER (Safe Zone - Temiz başlangıçta silinmeyenler)
    print("\n[SAFE] Aşağıdaki veriler korundu (SILINMEDI):")
    print("       - data/market_data.db (Toplanan kline / market verileri)")
    print("       - models/seeker_v1.xgb (Eğitilmiş yapay zeka/ML modeli)")
    print("       - .env & config dosyaları")

    print("\n[SUCCESS] System is clean. Ready for a fresh start.")
    print("Next: Run 'python3 src/main.py' to begin first cycle.")


if __name__ == "__main__":
    confirm = input("This will reset all trade history and portfolio state. Are you sure? (y/n): ")
    if confirm.lower() == "y":
        reset_system()
    else:
        print("Reset cancelled.")
