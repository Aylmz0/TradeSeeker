"""
55 Cycle Veri Analizi - price_location ve UPPER_10/LOWER_10 Pattern Analizi
"""
import json
from collections import defaultdict

# Load data
with open('data/cycle_history.json', 'r') as f:
    cycles = json.load(f)

with open('data/full_trade_history.json', 'r') as f:
    trades = json.load(f)

with open('data/portfolio_state.json', 'r') as f:
    portfolio = json.load(f)

print("=" * 80)
print("55 CYCLE VERİ ANALİZİ - price_location Pattern Analizi")
print("=" * 80)

# 1. Trade Özeti
print("\n## 1. KAPANAN POZİSYONLAR")
print("-" * 40)
total_pnl = 0
wins = 0
losses = 0
for t in trades:
    pnl = t['pnl']
    total_pnl += pnl
    if pnl > 0:
        wins += 1
    else:
        losses += 1
    print(f"  {t['symbol']} {t['direction'].upper()}: ${pnl:+.2f} | {t['close_reason'][:50]}")

print(f"\n  Toplam: {len(trades)} trade | Win: {wins} | Loss: {losses} | Net PnL: ${total_pnl:+.2f}")

# 2. Mevcut Pozisyonlar
print("\n## 2. MEVCUT POZİSYONLAR")
print("-" * 40)
for symbol, pos in portfolio['positions'].items():
    print(f"  {symbol} {pos['direction'].upper()}: ${pos['unrealized_pnl']:+.2f}")
    print(f"    Peak PnL: ${pos['peak_pnl']:.2f} | Erosion: {pos['erosion_pct']:.1f}%")

# 3. UPPER_10 Pattern Analizi
print("\n## 3. UPPER_10 + OVERBOUGHT PATTERN ANALİZİ")
print("-" * 40)

upper_10_mentions = []
for cycle in cycles:
    cot = cycle.get('chain_of_thoughts', '')
    cycle_num = cycle.get('cycle', 0)
    
    # Find UPPER_10 mentions
    if 'UPPER_10' in cot or 'upper_10' in cot.lower():
        # Check for overbought
        overbought = 'overbought' in cot.lower() or 'RSI >70' in cot or 'RSI 7' in cot
        pullback_risk = 'pullback risk' in cot.lower()
        
        # Get position PnL if any
        decisions = cycle.get('decisions', {})
        position_info = []
        for coin, dec in decisions.items():
            if 'unrealized_pnl' in dec:
                position_info.append(f"{coin}: ${dec['unrealized_pnl']:.2f}")
        
        upper_10_mentions.append({
            'cycle': cycle_num,
            'overbought': overbought,
            'pullback_risk_noted': pullback_risk,
            'positions': position_info
        })

print(f"  UPPER_10 mention count: {len(upper_10_mentions)}")
print("\n  Sample UPPER_10 occurrences:")
for m in upper_10_mentions[:5]:
    print(f"    Cycle {m['cycle']}: Overbought={m['overbought']}, Pullback Risk Noted={m['pullback_risk_noted']}")
    if m['positions']:
        print(f"      Positions: {', '.join(m['positions'])}")

# 4. LOWER_10 Pattern Analizi
print("\n## 4. LOWER_10 + OVERSOLD PATTERN ANALİZİ")
print("-" * 40)

lower_10_mentions = []
for cycle in cycles:
    cot = cycle.get('chain_of_thoughts', '')
    cycle_num = cycle.get('cycle', 0)
    
    if 'LOWER_10' in cot or 'lower_10' in cot.lower():
        oversold = 'oversold' in cot.lower() or 'RSI <30' in cot or 'bounce' in cot.lower()
        
        lower_10_mentions.append({
            'cycle': cycle_num,
            'oversold': oversold
        })

print(f"  LOWER_10 mention count: {len(lower_10_mentions)}")
for m in lower_10_mentions[:5]:
    print(f"    Cycle {m['cycle']}: Oversold/Bounce mentioned={m['oversold']}")

# 5. XRP ve DOGE Peak to Current Erosion
print("\n## 5. XRP VE DOGE KÂR ERİME ANALİZİ")
print("-" * 40)

xrp = portfolio['positions'].get('XRP', {})
doge = portfolio['positions'].get('DOGE', {})

print(f"  XRP:")
print(f"    Peak PnL: ${xrp.get('peak_pnl', 0):.2f}")
print(f"    Current PnL: ${xrp.get('unrealized_pnl', 0):.2f}")
print(f"    Erosion: ${xrp.get('erosion_from_peak', 0):.2f} ({xrp.get('erosion_pct', 0):.1f}%)")

print(f"\n  DOGE:")
print(f"    Peak PnL: ${doge.get('peak_pnl', 0):.2f}")
print(f"    Current PnL: ${doge.get('unrealized_pnl', 0):.2f}")
print(f"    Erosion: ${doge.get('erosion_from_peak', 0):.2f} ({doge.get('erosion_pct', 0):.1f}%)")

# 6. Özet ve Öneriler
print("\n## 6. OZET VE ONERILER")
print("-" * 40)
print(f"""
  [DATA] Toplam Veri:
     - {len(cycles)} cycle analiz edildi
     - {len(trades)} trade kapandi
     - {len(upper_10_mentions)} kez UPPER_10 tespit edildi
     - {len(lower_10_mentions)} kez LOWER_10 tespit edildi

  [PROBLEM] Problem:
     - XRP ve DOGE UPPER_10 + overbought durumundayken HOLD edildi
     - Peak'ten ${xrp.get('erosion_from_peak', 0) + doge.get('erosion_from_peak', 0):.2f} eridi
     - AI "pullback risk" uyarisi verdi ama aksiyon almadi

  [ONERI] Oneriler:
     1. Iyilestirme 1: LOWER_10 + RSI<30 -> counter-trade risk dusur
     2. Iyilestirme 2: UPPER_10 + RSI>70 -> trailing stop sikilastir veya kismi kar al
""")

print("=" * 80)
