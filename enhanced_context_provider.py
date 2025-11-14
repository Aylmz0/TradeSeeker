import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from config import Config

HTF_INTERVAL = getattr(Config, 'HTF_INTERVAL', '1h') or '1h'
HTF_LABEL = HTF_INTERVAL
class EnhancedContextProvider:
    """
    Provides enhanced context to help AI make better decisions
    NO manipulation - only better data and suggestions
    """
    
    def __init__(self):
        self.cycle_history_file = "cycle_history.json"
        self.trade_history_file = "trade_history.json"
        self.portfolio_state_file = "portfolio_state.json"
        self.performance_file = "performance_report.json"
        
    def safe_file_read(self, file_path: str, default_data=None):
        """Safely read JSON file with error handling"""
        try:
            if os.path.exists(file_path):
                if os.path.getsize(file_path) == 0:
                    return default_data if default_data is not None else []
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        return default_data if default_data is not None else []
                    
                    return json.loads(content)
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
        return default_data if default_data is not None else []
    
    def get_enhanced_position_context(self, portfolio_state: Dict) -> Dict[str, Any]:
        """Provides enhanced context for current positions"""
        positions = portfolio_state.get('positions', {})
        enhanced_context = {}
        
        for symbol, position in positions.items():
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', 0)
            unrealized_pnl = position.get('unrealized_pnl', 0)
            profit_target = position.get('exit_plan', {}).get('profit_target', 0)
            stop_loss = position.get('exit_plan', {}).get('stop_loss', 0)
            direction = (position.get('direction') or '').lower()
            
            # Profit target progress (relative to planned move)
            progress = 0.0
            try:
                if direction == 'long':
                    denominator = profit_target - entry_price
                    if denominator != 0:
                        progress = (current_price - entry_price) / denominator * 100
                elif direction == 'short':
                    denominator = entry_price - profit_target
                    if denominator != 0:
                        progress = (entry_price - current_price) / denominator * 100
            except Exception:
                progress = 0.0

            progress = max(0.0, min(progress, 100.0))
            remaining_pct = round(max(0.0, 100.0 - progress), 2)
            
            # Time in trade
            entry_time_str = position.get('entry_time', '')
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                    time_in_trade = (datetime.now() - entry_time).total_seconds() / 60  # minutes
                except Exception:
                    time_in_trade = 0
            else:
                time_in_trade = 0
            
            enhanced_context[symbol] = {
                'unrealized_pnl': unrealized_pnl,
                'profit_target_progress': round(progress, 2),
                'remaining_to_target_pct': remaining_pct,
                'time_in_trade_minutes': round(time_in_trade, 2),
                'distance_to_stop_pct': round(abs(stop_loss - current_price) / current_price * 100, 2) if current_price else 0,
                'risk_reward_ratio': round(abs(profit_target - entry_price) / abs(stop_loss - entry_price), 2) if stop_loss != entry_price else 0,
                'direction': direction,
                'trend_alignment': position.get('trend_alignment', position.get('trend_context', {}).get('alignment', 'unknown')),
                'trend_context': position.get('trend_context', {}),
                'confidence': position.get('confidence')
            }

            trailing_state = position.get('trailing') or {}
            if trailing_state:
                enhanced_context[symbol]['trailing'] = {
                    'last_update_cycle': trailing_state.get('last_update_cycle'),
                    'last_reason': trailing_state.get('last_reason'),
                    'last_stop': trailing_state.get('last_stop'),
                    'progress_percent': trailing_state.get('progress_percent'),
                    'time_in_trade_min': trailing_state.get('time_in_trade_min'),
                    'last_volume_ratio': trailing_state.get('last_volume_ratio')
                }
        
        return enhanced_context
    
    def get_market_regime_context(self) -> Dict[str, Any]:
        """Market regime detection based on objective price/EMA relationships"""
        try:
            from alpha_arena_deepseek import RealMarketData
        except ImportError:
            return {"current_regime": "unknown", "regime_strength": 0, "coin_regimes": {}}
        
        market_data = RealMarketData()
        overall_bull = 0
        overall_bear = 0
        coin_regimes = {}
        
        for coin in market_data.available_coins:
            try:
                indicators_htf = market_data.get_technical_indicators(coin, HTF_INTERVAL)
                if not isinstance(indicators_htf, dict) or 'error' in indicators_htf:
                    coin_regimes[coin] = {"regime": "unknown", "score": 0, "price_vs_ema20": "unknown"}
                    continue
                
                price = indicators_htf.get('current_price')
                ema20 = indicators_htf.get('ema_20')
                ema50 = indicators_htf.get('ema_50')
                
                score = 0
                regime = "neutral"
                price_vs_ema20 = "unknown"
                
                if isinstance(price, (int, float)) and isinstance(ema20, (int, float)):
                    price_vs_ema20 = "above" if price > ema20 else "below"
                    if price > ema20:
                        score += 1
                    elif price < ema20:
                        score -= 1
                
                if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)):
                    if ema20 > ema50:
                        score += 1
                    elif ema20 < ema50:
                        score -= 1
                
                if score >= 1:
                    regime = "bullish"
                    overall_bull += 1
                elif score <= -1:
                    regime = "bearish"
                    overall_bear += 1
                
                coin_regimes[coin] = {
                    "regime": regime,
                    "score": score,
                    "price_vs_ema20": price_vs_ema20
                }
            except Exception as e:
                print(f"⚠️ Market regime calculation error for {coin}: {e}")
                coin_regimes[coin] = {"regime": "error", "score": 0, "price_vs_ema20": "unknown"}
        
        coin_count = len(market_data.available_coins) if market_data.available_coins else 0
        neutral_count = max(0, coin_count - (overall_bull + overall_bear))
        total_bias = overall_bull + overall_bear
        strength = 0
        if coin_count == 0 or total_bias == 0:
            current_regime = "neutral"
        else:
            dominant = max(overall_bull, overall_bear)
            strength = round(dominant / coin_count, 2)
            neutral_majority = neutral_count >= (coin_count // 2 + coin_count % 2)
            if strength < Config.GLOBAL_NEUTRAL_STRENGTH_THRESHOLD or neutral_majority:
                current_regime = "neutral"
            elif overall_bull > overall_bear:
                current_regime = "bullish"
            elif overall_bear > overall_bull:
                current_regime = "bearish"
            else:
                current_regime = "neutral"
        
        return {
            "current_regime": current_regime,
            "regime_strength": strength,
            "bullish_count": overall_bull,
            "bearish_count": overall_bear,
            "neutral_count": neutral_count,
            "total_coins": coin_count,
            "coin_regimes": coin_regimes
        }
    
    def get_performance_insights(self, trade_history: List, portfolio_state: Dict) -> Dict[str, Any]:
        """Performance-based insights for better decision making"""
        if not trade_history:
            return {"insights": ["No trading history available"]}
        
        # Calculate win rate by coin
        coin_performance = {}
        for trade in trade_history:
            symbol = trade.get('symbol')
            pnl = trade.get('pnl', 0)
            
            if symbol not in coin_performance:
                coin_performance[symbol] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
            
            coin_performance[symbol]['trades'] += 1
            coin_performance[symbol]['total_pnl'] += pnl
            if pnl > 0:
                coin_performance[symbol]['wins'] += 1
        
        # Generate insights
        insights = []
        
        # Best performing coins
        profitable_coins = [coin for coin, stats in coin_performance.items() if stats['total_pnl'] > 0]
        if profitable_coins:
            insights.append(f"Historically profitable coins: {', '.join(profitable_coins)}")
        
        # Worst performing coins
        unprofitable_coins = [coin for coin, stats in coin_performance.items() if stats['total_pnl'] < -1.0]
        if unprofitable_coins:
            insights.append(f"Historically challenging coins: {', '.join(unprofitable_coins)}")
        
        # Current portfolio performance
        total_return = portfolio_state.get('total_return', 0)
        if total_return < -5:
            insights.append("Portfolio experiencing significant drawdown - consider defensive positioning")
        elif total_return > 5:
            insights.append("Portfolio performing well - current strategy effective")
        
        return {"insights": insights, "coin_performance": coin_performance}
    
    def get_directional_feedback(self, trade_history: List, lookback: int = 20) -> Dict[str, Any]:
        """Provide neutral feedback for long vs short performance"""
        feedback = {
            "long": {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0},
            "short": {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
        }
        
        if not trade_history:
            return feedback
        
        recent_trades = trade_history[-lookback:] if len(trade_history) > lookback else trade_history
        for trade in recent_trades:
            direction = trade.get('direction', '').lower()
            pnl = trade.get('pnl', 0.0)
            if direction not in feedback:
                continue
            feedback[direction]["trades"] += 1
            feedback[direction]["total_pnl"] += pnl
            if pnl > 0:
                feedback[direction]["wins"] += 1
            elif pnl < 0:
                feedback[direction]["losses"] += 1
        
        for direction, direction_stats in feedback.items():
            trades = direction_stats["trades"]
            if trades > 0:
                # Calculate win rate based on profit/loss amounts (not trade counts)
                # Win Rate = Total Profit / (|Total Profit| + |Total Loss|) * 100
                direction_trades = [t for t in recent_trades if t.get('direction', '').lower() == direction]
                direction_profit = sum(t.get('pnl', 0) for t in direction_trades if t.get('pnl', 0) > 0)
                direction_loss = abs(sum(t.get('pnl', 0) for t in direction_trades if t.get('pnl', 0) < 0))
                
                if direction_profit + direction_loss > 0:
                    direction_stats["win_rate"] = round((direction_profit / (direction_profit + direction_loss)) * 100, 1)
                else:
                    direction_stats["win_rate"] = 0.0
                direction_stats["avg_pnl"] = round(direction_stats["total_pnl"] / trades, 2)
            else:
                direction_stats["win_rate"] = 0.0
                direction_stats["avg_pnl"] = 0.0
        return feedback
    
    def get_risk_context(self, portfolio_state: Dict) -> Dict[str, Any]:
        """Enhanced risk management context"""
        positions = portfolio_state.get('positions', {})
        current_balance = portfolio_state.get('current_balance', 0)
        total_value = portfolio_state.get('total_value', 0)
        
        # Calculate risk metrics
        total_risk = sum(pos.get('risk_usd', 0) for pos in positions.values())
        
        # Position concentration
        position_count = len(positions)
        max_recommended_positions = 3
        
        # Correlation analysis (simplified)
        short_positions = [sym for sym, pos in positions.items() if pos.get('direction') == 'short']
        long_positions = [sym for sym, pos in positions.items() if pos.get('direction') == 'long']
        
        if position_count <= 0:
            diversification_score = 100.0
        else:
            raw_score = (1 - (position_count / max_recommended_positions)) * 100
            diversification_score = max(0.0, round(raw_score, 2))
        
        return {
            "total_risk_usd": round(total_risk, 2),
            "position_count": position_count,
            "max_recommended_positions": max_recommended_positions,
            "short_positions": short_positions,
            "long_positions": long_positions,
            "diversification_score": diversification_score
        }
    
    def generate_enhanced_context(self) -> Dict[str, Any]:
        """Main enhanced context generation function"""
        try:
            # Load data
            portfolio_state = self.safe_file_read(self.portfolio_state_file, {})
            trade_history = self.safe_file_read(self.trade_history_file, [])
            market_regime = self.get_market_regime_context()
            
            enhanced_context = {
                "timestamp": datetime.now().isoformat(),
                "position_context": self.get_enhanced_position_context(portfolio_state),
                "market_regime": market_regime,
                "performance_insights": self.get_performance_insights(trade_history, portfolio_state),
                "directional_feedback": self.get_directional_feedback(trade_history),
                "risk_context": self.get_risk_context(portfolio_state),
                "suggestions": self.generate_suggestions(portfolio_state, market_regime)
            }
            
            return enhanced_context
            
        except Exception as e:
            print(f"❌ Enhanced context generation error: {e}")
            return {"error": f"Context generation failed: {str(e)}"}
    
    def generate_suggestions(self, portfolio_state: Dict, market_regime: Dict) -> List[str]:
        """Non-manipulative suggestions for AI"""
        suggestions = []
        positions = portfolio_state.get('positions', {})
        
        # Market regime suggestions
        current_regime = market_regime.get('current_regime', 'unknown') if isinstance(market_regime, dict) else 'unknown'
        
        if current_regime == "bearish" and len(positions) >= 3:
            suggestions.append("[INFO] Bearish regime detected with ≥3 open positions")
        elif current_regime == "bullish" and len(positions) == 0:
            suggestions.append("[INFO] Bullish regime detected with zero current exposure")
        
        # Risk management suggestions
        
        return suggestions
    
    def print_enhanced_context(self, context: Dict):
        """Prints enhanced context in formatted way"""
        if "error" in context:
            print(f"❌ Enhanced context error: {context['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"🎯 ENHANCED CONTEXT FOR AI DECISION MAKING")
        print(f"{'='*60}")
        
        # Position Context
        position_context = context.get('position_context', {})
        if position_context:
            print(f"\n📊 POSITION CONTEXT:")
            for symbol, data in position_context.items():
                pnl = data.get('unrealized_pnl', 0)
                remaining_pct = data.get('remaining_to_target_pct', data.get('profit_target_progress', 0))
                time_in_trade = data.get('time_in_trade_minutes', 0)
                print(f"   {symbol}: ${pnl:.2f} PnL, {remaining_pct}% to target, {time_in_trade}min in trade")
        
        # Market Regime
        market_regime = context.get('market_regime', {})
        print(f"\n🌐 MARKET REGIME:")
        print(f"   Current: {market_regime.get('current_regime', 'unknown')}")
        print(f"   Strength: {market_regime.get('regime_strength', 0)}")
        
        # Performance Insights
        performance = context.get('performance_insights', {})
        insights = performance.get('insights', [])
        if insights:
            print(f"\n💡 PERFORMANCE INSIGHTS:")
            for insight in insights:
                print(f"   • {insight}")
        
        # Risk Context
        risk_context = context.get('risk_context', {})
        print(f"\n🛡️ RISK CONTEXT:")
        print(f"   Total Risk: ${risk_context.get('total_risk_usd', 0):.2f}")
        print(f"   Positions: {risk_context.get('position_count', 0)}")
        
        # Suggestions
        suggestions = context.get('suggestions', [])
        if suggestions:
            print(f"\n🎯 SUGGESTIONS (Non-binding):")
            for suggestion in suggestions:
                print(f"   • {suggestion}")
        
        print(f"\n{'='*60}")

# Main function for testing
def main():
    """Test enhanced context provider"""
    provider = EnhancedContextProvider()
    print("🔍 Generating enhanced context for AI decision making...")
    context = provider.generate_enhanced_context()
    provider.print_enhanced_context(context)

if __name__ == "__main__":
    main()
