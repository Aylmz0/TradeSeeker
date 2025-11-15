import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import requests

def safe_file_read(file_path: str, default_data=None):
    """Safely read JSON file with error handling - handles empty files gracefully"""
    try:
        if os.path.exists(file_path):
            # Check if file is empty (0 bytes)
            if os.path.getsize(file_path) == 0:
                print(f"ℹ️ Empty file detected: {file_path} - returning default data")
                return default_data if default_data is not None else []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Check if file contains only whitespace
                if not content:
                    print(f"ℹ️ Empty content in {file_path} - returning default data")
                    return default_data if default_data is not None else []
                
                return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"⚠️ Invalid JSON in {file_path}: {e}")
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
    return default_data if default_data is not None else []

def safe_file_write(file_path: str, data):
    """Safely write JSON file with error handling"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Error writing {file_path}: {e}")
        return False

class PerformanceMonitor:
    """Performance monitoring system for Alpha Arena DeepSeek"""
    
    def __init__(self):
        self.cycle_history_file = "cycle_history.json"
        self.trade_history_file = "trade_history.json"
        self.portfolio_state_file = "portfolio_state.json"
        self.performance_file = "performance_report.json"
        
    def analyze_performance(self, last_n_cycles: int = 10) -> Dict[str, Any]:
        """Analyze performance for the last N cycles"""
        try:
            # Load data
            cycles = safe_file_read(self.cycle_history_file, [])
            trades = safe_file_read(self.trade_history_file, [])
            portfolio = safe_file_read(self.portfolio_state_file, {})
            
            if not cycles:
                return {"info": "No trading data available yet. Run Alpha Arena DeepSeek to generate performance data."}
            
            # Get last N cycles
            recent_cycles = cycles[-last_n_cycles:] if len(cycles) > last_n_cycles else cycles
            
            # Calculate basic metrics
            total_cycles = len(cycles)
            recent_cycle_count = len(recent_cycles)
            
            # Count trading decisions
            total_decisions = 0
            total_entries = 0
            total_holds = 0
            total_closes = 0
            
            for cycle in recent_cycles:
                decisions = cycle.get('decisions', {})
                if isinstance(decisions, dict):
                    total_decisions += len(decisions)
                    for coin, trade in decisions.items():
                        if isinstance(trade, dict):
                            signal = trade.get('signal', '')
                            if signal == 'buy_to_enter' or signal == 'sell_to_enter':
                                total_entries += 1
                            elif signal == 'hold':
                                total_holds += 1
                            elif signal == 'close_position':
                                total_closes += 1
            
            # Analyze trade performance
            if trades:
                # Calculate win rate based on profit/loss amounts (not trade counts)
                winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
                break_even_trades = [t for t in trades if t.get('pnl', 0) == 0]
                
                # Calculate total profit and loss amounts
                total_profit = sum(t.get('pnl', 0) for t in winning_trades)
                total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
                
                # Win rate = Total Profit / (|Total Profit| + |Total Loss|) * 100
                # This gives a more accurate picture than trade count ratio
                if total_profit + total_loss > 0:
                    win_rate = (total_profit / (total_profit + total_loss)) * 100
                else:
                    win_rate = 0
                
                # Calculate average PnL
                total_pnl = sum(t.get('pnl', 0) for t in trades)
                avg_pnl = total_pnl / len(trades) if trades else 0
                
                # Calculate profit factor
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                # Calculate largest win/loss
                largest_win = max(t.get('pnl', 0) for t in trades) if trades else 0
                largest_loss = min(t.get('pnl', 0) for t in trades) if trades else 0
                
            else:
                win_rate = 0
                avg_pnl = 0
                profit_factor = 0
                largest_win = 0
                largest_loss = 0
                winning_trades = []
                losing_trades = []
                break_even_trades = []
            
            # Portfolio performance
            initial_balance = portfolio.get('initial_balance', 200.0)
            current_balance = portfolio.get('current_balance', 0.0)
            total_value = portfolio.get('total_value', 0.0)
            total_return = portfolio.get('total_return', 0.0)
            sharpe_ratio = portfolio.get('sharpe_ratio', 0.0)
            
            # Position analysis
            positions = portfolio.get('positions', {})
            open_positions_count = len(positions)
            
            # Coin performance analysis
            coin_performance = {}
            for trade in trades:
                coin = trade.get('symbol')
                if coin:
                    if coin not in coin_performance:
                        coin_performance[coin] = {'trades': 0, 'total_pnl': 0, 'wins': 0, 'losses': 0}
                    
                    coin_performance[coin]['trades'] += 1
                    coin_performance[coin]['total_pnl'] += trade.get('pnl', 0)
                    
                    if trade.get('pnl', 0) > 0:
                        coin_performance[coin]['wins'] += 1
                    elif trade.get('pnl', 0) < 0:
                        coin_performance[coin]['losses'] += 1
            
            # Calculate coin win rates (based on profit/loss amounts, not trade counts)
            for coin, stats in coin_performance.items():
                # Calculate profit and loss amounts for this coin
                coin_trades = [t for t in trades if t.get('symbol') == coin]
                coin_profit = sum(t.get('pnl', 0) for t in coin_trades if t.get('pnl', 0) > 0)
                coin_loss = abs(sum(t.get('pnl', 0) for t in coin_trades if t.get('pnl', 0) < 0))
                
                # Win rate = Total Profit / (|Total Profit| + |Total Loss|) * 100
                if coin_profit + coin_loss > 0:
                    stats['win_rate'] = (coin_profit / (coin_profit + coin_loss)) * 100
                else:
                    stats['win_rate'] = 0
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            # Compile performance report
            performance_report = {
                "analysis_period": f"Last {recent_cycle_count} cycles (Total: {total_cycles})",
                "timestamp": datetime.now().isoformat(),
                
                # Trading Activity
                "trading_activity": {
                    "total_decisions": total_decisions,
                    "entry_signals": total_entries,
                    "hold_signals": total_holds,
                    "close_signals": total_closes,
                    "decision_rate": (total_entries / total_decisions * 100) if total_decisions > 0 else 0
                },
                
                # Trade Performance
                "trade_performance": {
                    "total_trades": len(trades),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "break_even_trades": len(break_even_trades),
                    "win_rate": round(win_rate, 2),
                    "total_pnl": round(total_pnl, 2),
                    "average_pnl": round(avg_pnl, 2),
                    "profit_factor": round(profit_factor, 2),
                    "largest_win": round(largest_win, 2),
                    "largest_loss": round(largest_loss, 2)
                },
                
                # Portfolio Performance
                "portfolio_performance": {
                    "initial_balance": initial_balance,
                    "current_balance": current_balance,
                    "total_value": total_value,
                    "total_return": round(total_return, 2),
                    "sharpe_ratio": round(sharpe_ratio, 3),
                    "open_positions": open_positions_count
                },
                
                # Coin Performance
                "coin_performance": coin_performance,
                
                # Recommendations
                "recommendations": self._generate_recommendations(
                    win_rate, profit_factor, coin_performance, open_positions_count
                )
            }
            
            # Save performance report - append to reports array
            existing_reports = safe_file_read(self.performance_file, [])
            
            # If existing_reports is a dict (old format), convert to array
            if isinstance(existing_reports, dict):
                # Check if it's a reset message or old single report
                if "reset_reason" in existing_reports:
                    # It's a reset message, start fresh
                    reports_array = []
                else:
                    # It's an old single report, convert to array
                    reports_array = [existing_reports]
            elif isinstance(existing_reports, list):
                # Already an array
                reports_array = existing_reports
            else:
                # Invalid format, start fresh
                reports_array = []
            
            # Add new report to array
            reports_array.append(performance_report)
            
            # Keep only last 50 reports to prevent file from growing too large
            if len(reports_array) > 50:
                reports_array = reports_array[-50:]
            
            # Save updated reports array
            safe_file_write(self.performance_file, reports_array)
            
            return performance_report
            
        except Exception as e:
            print(f"❌ Performance analysis error: {e}")
            return {"error": f"Performance analysis failed: {str(e)}"}
    
    def _generate_recommendations(self, win_rate: float, profit_factor: float, 
                                coin_performance: Dict, open_positions: int) -> List[str]:
        """Generate honest performance-based recommendations in English"""
        recommendations = []
        
        # Get current portfolio data for dynamic values
        portfolio = safe_file_read(self.portfolio_state_file, {})
        current_balance = portfolio.get('current_balance', 0.0)
        initial_balance = portfolio.get('initial_balance', 200.0)
        total_return = portfolio.get('total_return', 0.0)
        
        # Dynamic cash balance warning
        if current_balance < initial_balance * 0.5:
            recommendations.append(f"[INFO] Cash balance ${current_balance:.2f} vs ${initial_balance:.2f} initial; liquidity below 50% of baseline")
        
        # 3 positions threshold (more conservative)
        if open_positions >= 3:
            recommendations.append(f"[INFO] Position count {open_positions}; exceeds reference threshold (≥3)")
        
        # Performance feedback
        if total_return < 5:
            recommendations.append(f"[INFO] Recorded return {total_return:.2f}% within analysis window; below growth target")
        
        # Coin performance - simple and clear
        if coin_performance:
            unprofitable_coins = [coin for coin, stats in coin_performance.items() 
                                if stats.get('total_pnl', 0) < 0]
            if unprofitable_coins:
                recommendations.append(f"[INFO] Coins with cumulative negative PnL: {', '.join(unprofitable_coins)}")
        
        # General recommendation if no specific issues
        if not recommendations:
            recommendations.append("[INFO] Performance metrics stable; no notable anomalies detected")
        
        return recommendations
    
    def print_performance_summary(self, report: Dict):
        """Print a formatted performance summary"""
        if "error" in report:
            print(f"❌ Performance analysis failed: {report['error']}")
            return
        
        if "info" in report:
            print(f"ℹ️ {report['info']}")
            return
        
        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE REPORT - {report.get('analysis_period', 'N/A')}")
        print(f"{'='*60}")
        
        # Trading Activity
        activity = report.get('trading_activity', {})
        print(f"\n🎯 TRADING ACTIVITY:")
        print(f"   Total Decisions: {activity.get('total_decisions', 0)}")
        print(f"   Entry Signals: {activity.get('entry_signals', 0)}")
        print(f"   Hold Signals: {activity.get('hold_signals', 0)}")
        print(f"   Decision Rate: {activity.get('decision_rate', 0):.1f}%")
        
        # Trade Performance
        trade_perf = report.get('trade_performance', {})
        print(f"\n💰 TRADE PERFORMANCE:")
        print(f"   Total Trades: {trade_perf.get('total_trades', 0)}")
        print(f"   Win Rate: {trade_perf.get('win_rate', 0):.1f}%")
        print(f"   Total PnL: ${trade_perf.get('total_pnl', 0):.2f}")
        print(f"   Avg PnL/Trade: ${trade_perf.get('average_pnl', 0):.2f}")
        print(f"   Profit Factor: {trade_perf.get('profit_factor', 0):.2f}")
        
        # Portfolio Performance
        portfolio_perf = report.get('portfolio_performance', {})
        print(f"\n📈 PORTFOLIO PERFORMANCE:")
        print(f"   Total Return: {portfolio_perf.get('total_return', 0):.2f}%")
        print(f"   Sharpe Ratio: {portfolio_perf.get('sharpe_ratio', 0):.3f}")
        print(f"   Open Positions: {portfolio_perf.get('open_positions', 0)}")
        
        # Coin Performance
        coin_perf = report.get('coin_performance', {})
        if coin_perf:
            print(f"\n🪙 COIN PERFORMANCE:")
            for coin, stats in coin_perf.items():
                win_rate = stats.get('win_rate', 0)
                total_pnl = stats.get('total_pnl', 0)
                trades = stats.get('trades', 0)
                print(f"   {coin}: {trades} trades, {win_rate:.1f}% win rate, PnL: ${total_pnl:.2f}")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        # Adaptive strategy suggestions
        adaptive_suggestions = self._generate_adaptive_suggestions(report)
        if adaptive_suggestions:
            print(f"\n🎯 ADAPTIVE STRATEGY SUGGESTIONS:")
            for suggestion in adaptive_suggestions:
                print(f"   • {suggestion}")
        
        print(f"\n📄 Full report saved to: {self.performance_file}")
        print(f"{'='*60}")

    def _generate_adaptive_suggestions(self, report: Dict) -> List[str]:
        """Generate adaptive strategy suggestions based on performance patterns"""
        suggestions = []
        
        # Get performance metrics
        trade_perf = report.get('trade_performance', {})
        portfolio_perf = report.get('portfolio_performance', {})
        activity = report.get('trading_activity', {})
        coin_perf = report.get('coin_performance', {})
        
        win_rate = trade_perf.get('win_rate', 0)
        profit_factor = trade_perf.get('profit_factor', 0)
        total_return = portfolio_perf.get('total_return', 0)
        sharpe_ratio = portfolio_perf.get('sharpe_ratio', 0)
        decision_rate = activity.get('decision_rate', 0)
        open_positions = portfolio_perf.get('open_positions', 0)
        
        # Strategy suggestions based on performance patterns
        
        # High win rate but low profit factor (small wins, big losses)
        if win_rate > 50 and profit_factor < 1.2:
            suggestions.append("[INFO] Win rate >50% alongside profit factor <1.2; average gains are relatively small versus losses")
        
        # Low win rate but positive profit factor (big wins, small losses)
        elif win_rate < 40 and profit_factor > 1.5:
            suggestions.append("[INFO] Win rate <40% with profit factor >1.5; outsized winners offset low hit rate")
        
        # High decision rate but poor performance
        if decision_rate > 60 and total_return < 0:
            suggestions.append("[INFO] Decision rate above 60% coincides with negative total return in the sample window")
        
        # Low decision rate with good performance
        elif decision_rate < 30 and total_return > 5:
            suggestions.append("[INFO] Decision rate below 30% with positive return observed; selective participation linked to gains")
        
        # Sharpe ratio analysis
        if sharpe_ratio < 0:
            suggestions.append("[INFO] Sharpe ratio <0; risk-adjusted performance trails baseline")
        elif sharpe_ratio > 1.0:
            suggestions.append("[INFO] Sharpe ratio >1.0; risk-adjusted performance classified as strong")
        
        # Position management suggestions
        if open_positions >= 4 and total_return < 0:
            suggestions.append("[INFO] Open positions ≥4 while total return is negative; exposure concentration elevated")
        elif open_positions <= 2 and total_return > 0:
            suggestions.append("[INFO] Open positions ≤2 with positive total return; lean positioning correlated with gains")
        
        # Coin-specific suggestions
        if coin_perf:
            # Find best and worst performing coins
            coin_pnls = [(coin, stats.get('total_pnl', 0)) for coin, stats in coin_perf.items()]
            coin_pnls.sort(key=lambda x: x[1], reverse=True)
            
            if coin_pnls:
                best_coin, best_pnl = coin_pnls[0]
                worst_coin, worst_pnl = coin_pnls[-1]
                
                if best_pnl > 0:
                    suggestions.append(f"[INFO] Strong performer: {best_coin} (${best_pnl:.2f}); positive contribution noted")
                
                if worst_pnl < -10:  # Significant loss
                    suggestions.append(f"[INFO] Weak performer: {worst_coin} (${worst_pnl:.2f}); drawdown exceeds -$10 threshold")
        
        # Market regime suggestions
        if total_return > 10:
            suggestions.append("[INFO] Portfolio return above 10%; market conditions appear supportive during analysis")
        elif total_return < -5:
            suggestions.append("[INFO] Portfolio return below -5%; drawdown environment observed")
        
        # Risk management suggestions
        if profit_factor < 0.8:
            suggestions.append("[INFO] Profit factor <0.8; indicates unfavorable reward-to-risk ratio")
        
        return suggestions

    def detect_trend_reversal_signals(self, coin: str, indicators_cache: Dict[str, Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Detect trend reversal signals for a specific coin (Information Only - Decision Remains With AI)
        
        Args:
            coin: Coin symbol to analyze
            indicators_cache: Optional pre-fetched indicators dict {coin: {interval: indicators}}
                            If provided, uses cached data instead of fetching new data
        """
        try:
            # Get HTF_INTERVAL from alpha_arena_deepseek
            from alpha_arena_deepseek import HTF_INTERVAL
            
            # Use cached indicators if provided (OPTIMIZATION: No re-fetch)
            if indicators_cache and coin in indicators_cache:
                coin_indicators = indicators_cache[coin]
                indicators_3m = coin_indicators.get('3m', {})
                indicators_htf = coin_indicators.get(HTF_INTERVAL, {})
            else:
                # Fallback: fetch indicators (legacy behavior)
                from alpha_arena_deepseek import RealMarketData
                market_data = RealMarketData()
                indicators_3m = market_data.get_technical_indicators(coin, '3m')
                indicators_htf = market_data.get_technical_indicators(coin, HTF_INTERVAL)
            
            if 'error' in indicators_3m or 'error' in indicators_htf:
                return {
                    "coin": coin,
                    "reversal_detected": False,
                    "signal_strength": "NO_DATA",
                    "details": "Unable to fetch indicator data",
                    "recommendation": "No trend reversal analysis available"
                }
            
            # Extract key indicators
            price_htf = indicators_htf.get('current_price')
            ema20_htf = indicators_htf.get('ema_20')
            price_3m = indicators_3m.get('current_price')
            ema20_3m = indicators_3m.get('ema_20')
            rsi_3m = indicators_3m.get('rsi_14', 50)
            rsi_htf = indicators_htf.get('rsi_14', 50)
            volume_3m = indicators_3m.get('volume', 0)
            avg_volume_3m = indicators_3m.get('avg_volume', 1)
            macd_3m = indicators_3m.get('macd', 0)
            macd_signal_3m = indicators_3m.get('macd_signal', 0)
            macd_htf = indicators_htf.get('macd', 0)
            macd_signal_htf = indicators_htf.get('macd_signal', 0)
            
            # Determine current trend direction
            trend_htf = "BULLISH" if price_htf > ema20_htf else "BEARISH"
            trend_3m = "BULLISH" if price_3m > ema20_3m else "BEARISH"
            
            # Trend reversal detection criteria
            reversal_signals = 0
            total_signals = 4
            signal_details = []
            
            # 1. Multi-timeframe EMA kırılma analizi
            htf_label = HTF_INTERVAL.upper()
            if trend_htf != trend_3m:
                reversal_signals += 1
                signal_details.append(f"Multi-timeframe EMA divergence: {htf_label} {trend_htf} vs 3m {trend_3m}")
            
            # 2. RSI momentum shift detection
            rsi_shift_3m = "BULLISH" if rsi_3m > 50 else "BEARISH"
            rsi_shift_htf = "BULLISH" if rsi_htf > 50 else "BEARISH"
            
            if rsi_shift_3m != trend_3m or rsi_shift_htf != trend_htf:
                reversal_signals += 1
                signal_details.append(f"RSI momentum shift: {htf_label} RSI {rsi_htf:.1f} ({rsi_shift_htf}), 3m RSI {rsi_3m:.1f} ({rsi_shift_3m})")
            
            # 3. Volume-price divergence
            volume_ratio = volume_3m / avg_volume_3m if avg_volume_3m > 0 else 0
            volume_divergence = False
            
            if trend_3m == "BULLISH" and volume_ratio < 0.8:
                volume_divergence = True
                signal_details.append(f"Volume divergence: Bullish trend but low volume ({volume_ratio:.1f}x)")
            elif trend_3m == "BEARISH" and volume_ratio < 0.8:
                volume_divergence = True
                signal_details.append(f"Volume divergence: Bearish trend but low volume ({volume_ratio:.1f}x)")
            
            if volume_divergence:
                reversal_signals += 1
            
            # 4. MACD signal line crossover
            macd_divergence_3m = (macd_3m > macd_signal_3m and trend_3m == "BEARISH") or (macd_3m < macd_signal_3m and trend_3m == "BULLISH")
            macd_divergence_htf = (macd_htf > macd_signal_htf and trend_htf == "BEARISH") or (macd_htf < macd_signal_htf and trend_htf == "BULLISH")
            
            if macd_divergence_3m or macd_divergence_htf:
                reversal_signals += 1
                signal_details.append("MACD signal line divergence detected")
            
            # Determine signal strength
            if reversal_signals >= 3:
                signal_strength = "HIGH_LOSS_RISK"
                recommendation = f"[INFO] Trend break detected for {coin}; loss-risk classification = HIGH"
            elif reversal_signals >= 2:
                signal_strength = "MEDIUM_LOSS_RISK"
                recommendation = f"[INFO] Trend weakening signals observed for {coin}; loss-risk classification = MEDIUM"
            elif reversal_signals >= 1:
                signal_strength = "LOW_LOSS_RISK"
                recommendation = f"[INFO] Minor divergence present for {coin}; loss-risk classification = LOW"
            else:
                signal_strength = "NO_LOSS_RISK"
                recommendation = f"[INFO] No loss-risk signals detected for {coin}; trend classification intact"
            
            return {
                "coin": coin,
                "reversal_detected": reversal_signals > 0,
                "signal_strength": signal_strength,
                "signals_detected": reversal_signals,
                "total_signals": total_signals,
                f"current_trend_{HTF_INTERVAL}": trend_htf,
                "current_trend_3m": trend_3m,
                "signal_details": signal_details,
                "recommendation": recommendation,
                "note": "INFORMATION ONLY - Final decision remains with AI"
            }
            
        except Exception as e:
            print(f"⚠️ Trend reversal detection error for {coin}: {e}")
            return {
                "coin": coin,
                "reversal_detected": False,
                "signal_strength": "ERROR",
                "details": f"Detection error: {str(e)}",
                "recommendation": "Unable to analyze trend reversal"
            }

    def detect_trend_reversal_for_all_coins(self, coins: List[str], indicators_cache: Dict[str, Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Detect trend break signals for all specified coins (Loss Risk Information Only)
        
        Args:
            coins: List of coin symbols to analyze
            indicators_cache: Optional pre-fetched indicators dict {coin: {interval: indicators}}
                            If provided, uses cached data instead of fetching new data (OPTIMIZATION)
        """
        try:
            reversal_results = {}
            high_risk_coins = []
            medium_risk_coins = []
            low_risk_coins = []
            no_risk_coins = []
            
            print(f"🔍 Analyzing trend break signals for {len(coins)} coins...")
            
            for coin in coins:
                reversal_result = self.detect_trend_reversal_signals(coin, indicators_cache=indicators_cache)
                reversal_results[coin] = reversal_result
                
                # Categorize coins by signal strength
                signal_strength = reversal_result.get('signal_strength', 'NO_LOSS_RISK')
                if signal_strength == "HIGH_LOSS_RISK":
                    high_risk_coins.append(coin)
                elif signal_strength == "MEDIUM_LOSS_RISK":
                    medium_risk_coins.append(coin)
                elif signal_strength == "LOW_LOSS_RISK":
                    low_risk_coins.append(coin)
                else:
                    no_risk_coins.append(coin)
            
            # Generate summary
            total_coins = len(coins)
            coins_with_risk = len(high_risk_coins) + len(medium_risk_coins) + len(low_risk_coins)
            
            summary = {
                "total_coins_analyzed": total_coins,
                "coins_with_loss_risk": coins_with_risk,
                "high_loss_risk_coins": high_risk_coins,
                "medium_loss_risk_coins": medium_risk_coins,
                "low_loss_risk_coins": low_risk_coins,
                "no_loss_risk_coins": no_risk_coins,
                "loss_risk_percentage": (coins_with_risk / total_coins * 100) if total_coins > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate recommendations based on loss risk patterns
            recommendations = self._generate_reversal_recommendations(summary, reversal_results)
            summary["recommendations"] = recommendations
            
            # Create loss risk signals for AI prompt format
            loss_risk_signals = {}
            for coin, result in reversal_results.items():
                signals = []
                if result.get('reversal_detected', False):
                    signals.append({
                        'type': 'LOSS_RISK',
                        'strength': result.get('signal_strength', 'UNKNOWN'),
                        'description': f"{coin}: {result.get('recommendation', 'No description')}"
                    })
                loss_risk_signals[coin] = {
                    'loss_risk_signals': signals,
                    'current_trend_4h': result.get('current_trend_4h', 'UNKNOWN'),
                    'current_trend_3m': result.get('current_trend_3m', 'UNKNOWN'),
                    'signal_strength': result.get('signal_strength', 'NO_LOSS_RISK')
                }
            
            return loss_risk_signals
            
        except Exception as e:
            print(f"❌ Error in trend break analysis for all coins: {e}")
            return {
                "error": f"Trend break analysis failed: {str(e)}"
            }

    def _generate_reversal_recommendations(self, summary: Dict, reversal_results: Dict) -> List[str]:
        """Generate recommendations based on trend reversal analysis"""
        recommendations = []
        
        strong_count = len(summary.get('strong_reversal_coins', []))
        moderate_count = len(summary.get('moderate_reversal_coins', []))
        weak_count = len(summary.get('weak_reversal_coins', []))
        reversal_percentage = summary.get('reversal_percentage', 0)
        
        # Market-wide reversal signals
        if strong_count >= 3:
            recommendations.append("[INFO] Multiple strong reversal signals detected across assets")
        elif strong_count >= 2:
            recommendations.append("[INFO] Several strong reversal signals present across coverage set")
        
        if reversal_percentage > 50:
            recommendations.append("[INFO] Reversal signal percentage above 50%; elevated probability of directional shifts")
        elif reversal_percentage > 30:
            recommendations.append("[INFO] Reversal signal percentage above 30%; moderate reversal frequency observed")
        
        # Specific coin recommendations
        strong_coins = summary.get('strong_reversal_coins', [])
        if strong_coins:
            recommendations.append(f"[INFO] Strong reversal readings detected in: {', '.join(strong_coins)}")
        
        # Risk management recommendations
        if strong_count > 0:
            recommendations.append("[INFO] Reversal signals flagged; protective level proximity worth monitoring")
        
        # General market sentiment
        if strong_count == 0 and moderate_count == 0:
            recommendations.append("[INFO] No significant reversal signals detected; prevailing trends classified as intact")
        
        return recommendations

# Main function for standalone usage
def main():
    """Standalone performance analysis"""
    monitor = PerformanceMonitor()
    print("🔍 Analyzing trading performance...")
    report = monitor.analyze_performance(last_n_cycles=10)
    monitor.print_performance_summary(report)

if __name__ == "__main__":
    main()
