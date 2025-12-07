import json

data = json.load(open('data/cycle_history.json'))

# Find all trade signals
trades = []
blocked = []
for c in data:
    for coin, d in c.get('decisions', {}).items():
        if isinstance(d, dict):
            signal = d.get('signal')
            runtime = d.get('runtime_decision')
            if signal in ['buy_to_enter', 'sell_to_enter']:
                info = {
                    'cycle': c['cycle'],
                    'coin': coin,
                    'signal': signal,
                    'runtime': runtime,
                    'confidence': d.get('confidence'),
                    'volume_ratio': d.get('volume_ratio_runtime')
                }
                if 'blocked' in str(runtime):
                    blocked.append(info)
                else:
                    trades.append(info)

output = []
output.append(f"\n=== TRADE SIGNALS SUMMARY ===")
output.append(f"Total entry signals: {len(trades) + len(blocked)}")
output.append(f"Executed: {len(trades)}")
output.append(f"Blocked: {len(blocked)}")

output.append(f"\n=== EXECUTED TRADES ===")
for t in trades:
    output.append(f"Cycle {t['cycle']}: {t['coin']} {t['signal']} -> {t['runtime']} (conf={t['confidence']})")

output.append(f"\n=== BLOCKED TRADES ===")
for b in blocked:
    output.append(f"Cycle {b['cycle']}: {b['coin']} {b['signal']} -> {b['runtime']} (vol={b['volume_ratio']})")

# Count holds per cycle
hold_counts = [len([1 for d in c.get('decisions',{}).values() if isinstance(d,dict) and d.get('signal')=='hold']) for c in data]
output.append(f"\n=== HOLD ANALYSIS ===")
output.append(f"Total cycles: {len(data)}")
output.append(f"Average holds per cycle: {sum(hold_counts)/len(hold_counts):.1f}")

# Position count over time
output.append(f"\n=== LAST 10 CYCLES METADATA ===")
for c in data[-10:]:
    meta = c.get('metadata', {})
    exec_report = meta.get('execution_report', {})
    output.append(f"Cycle {c['cycle']}: executed={len(exec_report.get('executed',[]))}, blocked={len(exec_report.get('blocked',[]))}, holds={len(exec_report.get('holds',[]))}")

result = '\n'.join(output)
with open('cycle_analysis_result.txt', 'w') as f:
    f.write(result)
print(result)

