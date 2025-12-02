import json
import os
from datetime import datetime

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_timestamp(ts_str):
    try:
        return datetime.fromisoformat(ts_str)
    except:
        return None

def main():
    trades = load_json('data/full_trade_history.json')
    cycles = load_json('data/cycle_history.json')

    if not trades or not cycles:
        print("Failed to load data.")
        return

    output_lines = []

    # Filter for recent losses (last 20 trades, PnL < 0)
    recent_trades = trades[-20:]
    losing_trades = [t for t in recent_trades if t.get('pnl', 0) < 0]

    print(f"Found {len(losing_trades)} losing trades in the last 20 trades.")

    for trade in losing_trades:
        coin = trade.get('symbol')
        exit_time_str = trade.get('exit_time')
        exit_reason = trade.get('close_reason')
        pnl = trade.get('pnl')
        
        output_lines.append(f"\n--------------------------------------------------")
        output_lines.append(f"Analyzing Loss: {coin} | PnL: ${pnl:.2f} | Reason: {exit_reason}")
        output_lines.append(f"Exit Time: {exit_time_str}")

        if not exit_time_str:
            continue

        exit_dt = parse_timestamp(exit_time_str)
        if not exit_dt:
            continue

        # Find the cycle just before the exit
        relevant_cycle = None
        min_diff = float('inf')

        for cycle in cycles:
            cycle_ts_str = cycle.get('timestamp')
            if not cycle_ts_str:
                continue
            
            cycle_dt = parse_timestamp(cycle_ts_str)
            if not cycle_dt:
                continue

            # We want the cycle closest to exit time but BEFORE it (or very slightly after if processing lag)
            # Actually, if AI closed it, the cycle timestamp might be slightly before the trade exit timestamp.
            diff = (exit_dt - cycle_dt).total_seconds()
            
            # Look for cycles within 5 minutes before exit
            if 0 <= diff < 300: 
                if diff < min_diff:
                    min_diff = diff
                    relevant_cycle = cycle

        if relevant_cycle:
            output_lines.append(f"Found relevant cycle: {relevant_cycle.get('cycle')} (Timestamp: {relevant_cycle.get('timestamp')})")
            thoughts = relevant_cycle.get('chain_of_thoughts', '')
            
            # Extract thoughts for this coin
            # Thoughts are usually formatted as "COIN: analysis..."
            # We can try to split by coin name or newlines.
            
            coin_thought_start = thoughts.find(f"{coin}:")
            if coin_thought_start != -1:
                # Find end of this coin's section (next coin or double newline)
                rest = thoughts[coin_thought_start:]
                # Simple heuristic: split by double newline or next coin name
                # But next coin name is hard to predict.
                # Let's just print the first 500 chars after finding the coin
                output_lines.append(f"AI Reasoning for {coin}:")
                output_lines.append(rest[:600] + "...")
            else:
                output_lines.append(f"Could not find specific reasoning for {coin} in thoughts.")
                output_lines.append(f"Full thoughts snippet: {thoughts[:200]}")
        else:
            output_lines.append("No matching cycle found within 5 minutes before exit.")

    with open('loss_analysis_output.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print("Analysis complete. Output written to loss_analysis_output.txt")

if __name__ == "__main__":
    main()
