# Operations

## Running the Bot

### Simulation Mode (Default)
```bash
python3 src/main.py
```
- Uses simulated trades
- No real money at risk
- Data collected for ML training

### Live Mode
```bash
# 1. Configure .env
TRADING_MODE=live
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret

# 2. Start
python3 src/main.py
```

### Admin Dashboard
```bash
# Flask admin server (separate process)
python3 src/web/admin_server_flask.py
```

## 35-Cycle Reset

Every `HISTORY_RESET_INTERVAL` cycles (default: 35):

1. Current data backed up to `data/backups/`
2. Trade history cleared
3. Cycle history cleared
4. Performance history cleared
5. `full_trade_history.json` preserved

**Why**: Prevents long-term bias from accumulating data.

## Data Files

| File | Location | Purpose |
|------|----------|---------|
| `portfolio_state.json` | `data/` | Current balance, positions, PnL |
| `trade_history.json` | `data/` | Recent trade history |
| `cycle_history.json` | `data/` | Cycle-by-cycle records |
| `performance_history.json` | `data/` | Performance metrics |
| `full_trade_history.json` | `data/` | Complete trade archive (survives reset) |
| `backups/` | `data/backups/` | Periodic backups |

## Troubleshooting

### Bot won't start
- Check `.env` has valid API keys
- Verify `TRADING_MODE` is `simulation` or `live`
- Check `LOG_LEVEL` is a valid Python logging level

### No trades executing
- Check `MIN_CONFIDENCE` threshold (default: 0.60)
- Check `CHOPPY_ER_THRESHOLD` (default: 0.30) — market may be choppy
- Check `MAX_POSITIONS` limit
- Check `VOLUME_MINIMUM_THRESHOLD` (default: 0.30x)

### ML not working
- Run `python3 scripts/train_model.py` to train XGBoost model
- Bot degrades gracefully to AI-only mode if ML unavailable

### API errors
- Check Binance API keys and permissions
- Verify `BINANCE_TESTNET` setting
- Check `MAX_RETRY_ATTEMPTS` and `REQUEST_TIMEOUT`

### High memory usage
- Check `USE_SMART_CACHE` setting
- Clear cache: `CacheManager.clear_all()`
- Check `JSON_SERIES_MAX_LENGTH` (default: 30)

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_model.py` | Train XGBoost model |
| `scripts/generate_performance_report.py` | Generate performance report |
| `scripts/forensic_audit.py` | Audit trading history |
| `scripts/llm_benchmark.py` | Benchmark LLM decisions |
| `scripts/replay_trade.py` | Replay historical trades |
| `scripts/backtest_runner.py` | Run backtests |
| `scripts/debug_brain.py` | Debug AI decisions |
| `scripts/system_forensic_audit.py` | System-level audit |
