import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Backup cycle history
bc = json.load(open('history_backups/20251207_182927_cycle_35/cycle_history.json', encoding='utf-8'))
cc = json.load(open('data/cycle_history.json', encoding='utf-8'))
th = json.load(open('history_backups/20251207_182927_cycle_35/trade_history.json', encoding='utf-8'))
th2 = json.load(open('data/full_trade_history.json', encoding='utf-8'))

all_cycles = bc + cc
all_trades = th + th2

print('=== COUNTER-TREND ANALIZI ===')

# Counter-trend trade'leri bul
counter_trades = []
for cycle in all_cycles:
    for coin, decision in cycle.get('decisions', {}).items():
        if isinstance(decision, dict):
            trend = decision.get('trend_alignment', '')
            if 'counter' in str(trend).lower():
                counter_trades.append({
                    'cycle': cycle.get('cycle'),
                    'coin': coin,
                    'signal': decision.get('signal'),
                    'confidence': decision.get('confidence'),
                    'blocked': decision.get('runtime_decision', '')
                })

print(f'Toplam counter-trend karar: {len(counter_trades)}')

blocked = [t for t in counter_trades if 'blocked' in str(t.get('blocked', '')).lower()]
executed = [t for t in counter_trades if t not in blocked]

print(f'Bloklanan: {len(blocked)}')
print(f'Izin verilen: {len(executed)}')

print()
print('BLOKLANAN:')
for t in blocked[:8]:
    print(f"  C{t['cycle']}: {t['coin']} {t.get('signal','')} reason={t['blocked']}")

print()
print('IZIN VERILEN:')
for t in executed[:8]:
    print(f"  C{t['cycle']}: {t['coin']} {t.get('signal','')} conf={t.get('confidence','')}")

# UPPER_10 analizi
print()
print('=== UPPER_10 ANALIZI ===')
upper10_count = 0
for cycle in all_cycles:
    cot = cycle.get('chain_of_thoughts', '')
    if 'UPPER_10' in cot:
        upper10_count += 1

print(f'UPPER_10 gorulen cycle: {upper10_count}')

# Trade sonuclari
print()
print('=== TRADE SONUCLARI ===')
wins = [t for t in all_trades if t.get('pnl', 0) > 0]
losses = [t for t in all_trades if t.get('pnl', 0) <= 0]
print(f'Kazanan: {len(wins)}')
print(f'Kaybeden: {len(losses)}')
total_pnl = sum(t.get('pnl', 0) for t in all_trades)
print(f'Toplam PnL: ${total_pnl:.2f}')
