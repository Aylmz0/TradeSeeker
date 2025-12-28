
import json
import os

def extract_all_trade_logs():
    input_path = r'c:\Users\yilmaz\Desktop\TradeSeeker-main\data\cycle_history.json'
    output_path = r'c:\Users\yilmaz\Desktop\TradeSeeker-main\all_trades_log.txt'
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading json: {e}")
        return
        
    cycles = data if isinstance(data, list) else data.get('cycles', [])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        count = 0
        for cycle in cycles:
            decisions = cycle.get('decisions', {})
            # Look for any decision that is NOT 'hold' or 'wait'
            active_trades = []
            for coin, dec in decisions.items():
                signal = dec.get('signal', '').lower()
                if signal not in ['hold', 'wait', 'monitor']:
                    active_trades.append((coin, dec))
            
            if active_trades:
                count += 1
                f.write(f"\n{'='*50}\n")
                f.write(f"CYCLE {cycle.get('cycle')} - {cycle.get('timestamp')}\n")
                f.write(f"{'='*50}\n")
                
                for coin, dec in active_trades:
                    f.write(f"ACTION on {coin}: {dec.get('signal')}\n")
                    f.write(f"Decision Details: {json.dumps(dec, indent=2)}\n")
                
                f.write(f"\n--- CHAIN OF THOUGHTS ---\n")
                f.write(cycle.get('chain_of_thoughts', 'No CoT provided.'))
                f.write("\n\n")
    
    print(f"Successfully extracted {count} cycles with active trades to: {output_path}")

if __name__ == "__main__":
    extract_all_trade_logs()
