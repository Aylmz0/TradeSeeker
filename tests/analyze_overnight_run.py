import json
import os
from datetime import datetime
import glob

def analyze_overnight_run():
    base_path = r"c:\Users\yilmaz\Desktop\TradeSeeker-main"
    data_path = os.path.join(base_path, "data")
    
    trade_file = os.path.join(data_path, "trade_history.json")
    cycle_file = os.path.join(data_path, "cycle_history.json")
    perf_file = os.path.join(data_path, "performance_report.json")
    
    print(f"Analyzing Overnight Run Data...\n")
    
    # 1. Performance Report Overview
    try:
        if os.path.exists(perf_file):
            with open(perf_file, 'r') as f:
                perf = json.load(f)
            print(f"--- Performance Report Summary ---")
            print(f"Total Return: {perf.get('total_return_pct', 0):.2f}%")
            print(f"Win Rate: {perf.get('win_rate', 0):.2f}%")
            print(f"Total Trades: {perf.get('total_trades', 0)}")
            print(f"Profit Factor: {perf.get('profit_factor', 0)}")
            print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0)}")
            print("-" * 30 + "\n")
    except Exception as e:
        print(f"Error reading performance_report.json: {e}")

    # 2. Trade History Analysis
    try:
        if os.path.exists(trade_file):
            with open(trade_file, 'r') as f:
                trades = json.load(f)
            
            print(f"--- Trade History Analysis ({len(trades)} trades) ---")
            total_pnl = 0
            for trade in trades:
                pnl = trade.get('pnl', 0)
                total_pnl += pnl
                symbol = trade.get('symbol')
                direction = trade.get('direction')
                entry_time = trade.get('entry_time')
                exit_time = trade.get('exit_time')
                reason = trade.get('close_reason', 'Unknown')
                
                status = "WIN" if pnl > 0 else "LOSS"
                print(f"[{status}] {symbol} {direction} | PnL: ${pnl:.2f} | Reason: {reason}")
                print(f"      Entry: {entry_time} | Exit: {exit_time}")
            
            print(f"\nTotal Realized PnL: ${total_pnl:.2f}")
            print("-" * 30 + "\n")
    except Exception as e:
        print(f"Error reading trade_history.json: {e}")

    # 3. Cycle History Analysis (AI Reasoning)
    try:
        if os.path.exists(cycle_file):
            with open(cycle_file, 'r') as f:
                cycles = json.load(f)
            
            print(f"--- Cycle History Analysis (Last 10 Cycles) ---")
            cycles.sort(key=lambda x: x.get('cycle', 0))
            recent_cycles = cycles[-10:]
            
            for cycle in recent_cycles:
                c_num = cycle.get('cycle')
                timestamp = cycle.get('timestamp')
                thoughts = cycle.get('chain_of_thoughts', 'No thoughts recorded')
                decisions = cycle.get('decisions', {})
                
                print(f"Cycle {c_num} ({timestamp}):")
                print(f"Thoughts: {thoughts[:300]}...") # First 300 chars
                
                # Check for active decisions
                has_action = False
                for coin, decision in decisions.items():
                    signal = decision.get('signal')
                    if signal not in ['hold', 'wait']:
                        print(f"  -> ACTION: {coin} - {signal} (Conf: {decision.get('confidence')})")
                        print(f"     Reasoning: {decision.get('invalidation_condition', 'N/A')}")
                        has_action = True
                
                if not has_action:
                    print("  -> No new actions (HOLD/WAIT)")
                print("-" * 30)
    except Exception as e:
        print(f"Error reading cycle_history.json: {e}")

if __name__ == "__main__":
    import sys
    # Redirect stdout to a file
    with open('overnight_analysis_report.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        analyze_overnight_run()
