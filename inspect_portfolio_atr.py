import json
import os

def inspect_portfolio():
    file_path = 'data/portfolio_state.json'
    if not os.path.exists(file_path):
        print("❌ Portfolio state file not found.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        positions = data.get('positions', {})
        if not positions:
            print("ℹ️ No active positions.")
            return

        print(f"📊 Analyzing {len(positions)} Active Positions:\n")
        
        for coin, pos in positions.items():
            entry = pos.get('entry_price')
            exit_plan = pos.get('exit_plan', {})
            tp = exit_plan.get('profit_target')
            sl = exit_plan.get('stop_loss')
            
            if entry and tp and sl:
                tp_dist = abs(tp - entry)
                sl_dist = abs(sl - entry)
                tp_pct = (tp_dist / entry) * 100
                sl_pct = (sl_dist / entry) * 100
                
                # Estimated ATR (TP / 2.0)
                est_atr = tp_dist / 2.0
                
                print(f"   Coin: {coin}")
                print(f"     Entry: ${entry:.4f}")
                print(f"     TP: ${tp:.4f} ({tp_pct:.2f}%)")
                print(f"     SL: ${sl:.4f} ({sl_pct:.2f}%)")
                print(f"     Est. ATR: ${est_atr:.4f}")
                
                if tp_pct < 0.6:
                    print("     -> Status: VERY TIGHT (Low Volatility/Choppy)")
                elif tp_pct > 1.5:
                    print("     -> Status: WIDE (High Volatility)")
                else:
                    print("     -> Status: NORMAL")
                print("")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_portfolio()
