import json
import os

def verify_atr_behavior():
    file_path = 'data/cycle_history.json'
    if not os.path.exists(file_path):
        print("❌ Cycle history file not found.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            cycles = data
        else:
            cycles = data.get('cycles', [])

        if not cycles:
            print("ℹ️ No cycles found.")
            return

        last_cycles = cycles[-5:] # Check last 5 cycles
        print(f"📊 Analyzing ATR Behavior in last {len(last_cycles)} cycles:\n")

        for cycle in last_cycles:
            cycle_num = cycle.get('cycle', '?')
            print(f"--- Cycle {cycle_num} ---")
            
            decisions = cycle.get('decisions', {})
            for coin, decision in decisions.items():
                signal = decision.get('signal')
                if signal in ['buy_to_enter', 'sell_to_enter']:
                    entry = decision.get('entry_price')
                    tp = decision.get('profit_target')
                    sl = decision.get('stop_loss')
                    
                    if entry and tp and sl:
                        tp_dist = abs(tp - entry)
                        sl_dist = abs(sl - entry)
                        tp_pct = (tp_dist / entry) * 100
                        sl_pct = (sl_dist / entry) * 100
                        
                        # Reverse engineer ATR roughly (TP = ATR * 2.0)
                        estimated_atr = tp_dist / 2.0
                        
                        print(f"   Coin: {coin}")
                        print(f"     Entry: ${entry:.4f}")
                        print(f"     TP: {tp_pct:.2f}% (Dist: ${tp_dist:.4f})")
                        print(f"     SL: {sl_pct:.2f}% (Dist: ${sl_dist:.4f})")
                        print(f"     Est. ATR: ${estimated_atr:.4f}")
                        
                        if tp_pct < 0.5:
                            print("     -> Status: TIGHT (Choppy/Low Volatility)")
                        elif tp_pct > 1.5:
                            print("     -> Status: WIDE (Trending/High Volatility)")
                        else:
                            print("     -> Status: NORMAL")
                        print("")
                        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify_atr_behavior()
