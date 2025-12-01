import json
from collections import defaultdict

try:
    with open('data/full_trade_history.json', 'r') as f:
        trades = json.load(f)
    
    total_trades = len(trades)
    total_pnl = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    
    print(f'Total Trades: {total_trades}')
    print(f'Total PnL: {total_pnl:.2f}')
    print(f'Win Rate: {win_rate:.1f}%')
    
    print('\n--- By Coin ---')
    coin_pnl = defaultdict(float)
    coin_counts = defaultdict(int)
    coin_wins = defaultdict(int)
    
    for t in trades:
        coin_pnl[t['symbol']] += t['pnl']
        coin_counts[t['symbol']] += 1
        if t['pnl'] > 0:
            coin_wins[t['symbol']] += 1
            
    # Sort by PnL
    sorted_coins = sorted(coin_pnl.items(), key=lambda x: x[1])
    
    for coin, pnl in sorted_coins:
        count = coin_counts[coin]
        wr = (coin_wins[coin] / count * 100)
        print(f'{coin}: Count={count}, PnL={pnl:.2f}, WinRate={wr:.1f}%')
        
    print('\n--- By Close Reason ---')
    reasons = defaultdict(int)
    for t in trades:
        reason = t.get('close_reason', 'Unknown')
        # Simplify reason for grouping
        if 'Margin-based Stop Loss' in reason:
            reason = 'Margin Stop Loss'
        elif 'Profit taking' in reason:
            reason = 'Profit Taking'
        elif 'AI close_position' in reason:
            reason = 'AI Close'
        elif 'Position margin' in reason and 'maximum limit' in reason:
            reason = 'Max Margin Limit (Partial)'
        elif 'Profit Target' in reason:
            reason = 'Profit Target Hit'
        reasons[reason] += 1
        
    for r, c in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
        print(f'{r}: {c}')
        
    print('\n--- Top 5 Losses ---')
    sorted_trades = sorted(trades, key=lambda x: x['pnl'])
    for t in sorted_trades[:5]:
        print(f"{t['symbol']} {t['direction']} PnL: {t['pnl']:.2f} Reason: {t.get('close_reason')}")

    print('\n--- Top 5 Wins ---')
    for t in sorted_trades[-5:]:
        print(f"{t['symbol']} {t['direction']} PnL: {t['pnl']:.2f} Reason: {t.get('close_reason')}")

except Exception as e:
    print(e)
