"""Extended tests for src/core/performance_monitor.py."""

import json
import math
import os

import pytest

from config.config import Config
from src.core.constants import (
    ALIGNMENT_STRENGTH_ALL,
    ALIGNMENT_STRENGTH_MANY,
    MAX_POSITIONS_DIVERSITY,
    MIN_DIVERSITY_POSITIONS,
    REVERSAL_PCT_HIGH,
    REVERSAL_PCT_MODERATE,
    SORTINO_MIN_RETURNS,
)
from src.core.performance_monitor import PerformanceMonitor


@pytest.fixture
def monitor(tmp_path):
    """PerformanceMonitor with temp file paths."""
    m = PerformanceMonitor()
    m.cycle_history_file = str(tmp_path / "cycle_history.json")
    m.trade_history_file = str(tmp_path / "trade_history.json")
    m.portfolio_state_file = str(tmp_path / "portfolio_state.json")
    m.performance_file = str(tmp_path / "performance_report.json")
    return m


@pytest.fixture
def write_test_json(tmp_path):
    """Helper to write test JSON files into tmp_path."""

    def _write(filename, data):
        path = str(tmp_path / filename)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    return _write


# ── test_generate_adaptive_suggestions ───────────────────────────────────────


class TestGenerateAdaptiveSuggestions:
    """Tests for _generate_adaptive_suggestions with various performance patterns."""

    def test_high_profitability_low_profit_factor(self, monitor):
        """High PI but low PF triggers small-wins-big-losses suggestion."""
        report = {
            "trade_performance": {"profitability_index": 60.0, "profit_factor": 1.0},
            "portfolio_performance": {
                "total_return": 2.0,
                "sharpe_ratio": 0.5,
                "open_positions": 2,
            },
            "trading_activity": {"decision_rate": 50.0},
            "coin_performance": {},
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("Profitability Index" in s and "profit factor" in s for s in suggestions)

    def test_low_profitability_high_profit_factor(self, monitor):
        """Low PI but high PF triggers outsized-winners suggestion."""
        report = {
            "trade_performance": {"profitability_index": 30.0, "profit_factor": 3.0},
            "portfolio_performance": {
                "total_return": 2.0,
                "sharpe_ratio": 0.5,
                "open_positions": 2,
            },
            "trading_activity": {"decision_rate": 50.0},
            "coin_performance": {},
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("Profitability Index" in s and "outsized" in s.lower() for s in suggestions)

    def test_high_decision_rate_negative_return(self, monitor):
        """High decision rate with negative return triggers overtrading suggestion."""
        report = {
            "trade_performance": {"profitability_index": 45.0, "profit_factor": 1.3},
            "portfolio_performance": {
                "total_return": -1.0,
                "sharpe_ratio": 0.5,
                "open_positions": 2,
            },
            "trading_activity": {"decision_rate": 70.0},
            "coin_performance": {},
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("Decision rate" in s and "negative" in s.lower() for s in suggestions)

    def test_low_decision_rate_positive_return(self, monitor):
        """Low decision rate with positive return triggers selective-participation suggestion."""
        report = {
            "trade_performance": {"profitability_index": 45.0, "profit_factor": 1.3},
            "portfolio_performance": {
                "total_return": 8.0,
                "sharpe_ratio": 0.5,
                "open_positions": 2,
            },
            "trading_activity": {"decision_rate": 20.0},
            "coin_performance": {},
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("Decision rate" in s and "selective" in s.lower() for s in suggestions)

    def test_low_sharpe_ratio(self, monitor):
        """Sharpe below 0 triggers risk-adjusted baseline suggestion."""
        report = {
            "trade_performance": {"profitability_index": 45.0, "profit_factor": 1.3},
            "portfolio_performance": {
                "total_return": 2.0,
                "sharpe_ratio": -0.5,
                "open_positions": 2,
            },
            "trading_activity": {"decision_rate": 50.0},
            "coin_performance": {},
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("Sharpe ratio" in s for s in suggestions)

    def test_high_sharpe_ratio(self, monitor):
        """Sharpe above 1.0 triggers strong risk-adjusted suggestion."""
        report = {
            "trade_performance": {"profitability_index": 45.0, "profit_factor": 1.3},
            "portfolio_performance": {
                "total_return": 2.0,
                "sharpe_ratio": 1.5,
                "open_positions": 2,
            },
            "trading_activity": {"decision_rate": 50.0},
            "coin_performance": {},
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("Sharpe ratio" in s and "strong" in s.lower() for s in suggestions)

    def test_many_positions_negative_return(self, monitor):
        """>= MAX_POSITIONS_DIVERSITY with negative return triggers exposure suggestion."""
        report = {
            "trade_performance": {"profitability_index": 45.0, "profit_factor": 1.3},
            "portfolio_performance": {
                "total_return": -1.0,
                "sharpe_ratio": 0.5,
                "open_positions": MAX_POSITIONS_DIVERSITY + 1,
            },
            "trading_activity": {"decision_rate": 50.0},
            "coin_performance": {},
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("positions" in s.lower() and "negative" in s.lower() for s in suggestions)

    def test_few_positions_positive_return(self, monitor):
        "<= MIN_DIVERSITY_POSITIONS with positive return triggers lean positioning suggestion."
        report = {
            "trade_performance": {"profitability_index": 45.0, "profit_factor": 1.3},
            "portfolio_performance": {
                "total_return": 8.0,
                "sharpe_ratio": 0.5,
                "open_positions": MIN_DIVERSITY_POSITIONS,
            },
            "trading_activity": {"decision_rate": 50.0},
            "coin_performance": {},
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("positions" in s.lower() and "lean" in s.lower() for s in suggestions)

    def test_best_and_worst_coin_suggestions(self, monitor):
        """Strong and weak coin performers generate suggestions."""
        report = {
            "trade_performance": {"profitability_index": 45.0, "profit_factor": 1.3},
            "portfolio_performance": {
                "total_return": 2.0,
                "sharpe_ratio": 0.5,
                "open_positions": 2,
            },
            "trading_activity": {"decision_rate": 50.0},
            "coin_performance": {
                "SOL": {"total_pnl": 20.0},
                "DOGE": {"total_pnl": -15.0},
            },
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("SOL" in s and "Strong performer" in s for s in suggestions)
        assert any("DOGE" in s and "Weak performer" in s for s in suggestions)

    def test_profit_factor_critical(self, monitor):
        """Profit factor below 0.8 triggers unfavorable reward-to-risk suggestion."""
        report = {
            "trade_performance": {"profitability_index": 45.0, "profit_factor": 0.5},
            "portfolio_performance": {
                "total_return": 2.0,
                "sharpe_ratio": 0.5,
                "open_positions": 2,
            },
            "trading_activity": {"decision_rate": 50.0},
            "coin_performance": {},
        }
        suggestions = monitor._generate_adaptive_suggestions(report)
        assert any("Profit factor" in s and "unfavorable" in s.lower() for s in suggestions)

    def test_empty_report_triggers_profit_factor_critical(self, monitor):
        """An empty report yields default profit_factor=0 which triggers PF<0.8 suggestion."""
        suggestions = monitor._generate_adaptive_suggestions({})
        assert any("Profit factor" in s and "unfavorable" in s.lower() for s in suggestions)


# ── test_generate_reversal_recommendations ───────────────────────────────────


class TestGenerateReversalRecommendations:
    """Tests for _generate_reversal_recommendations with different risk levels."""

    def test_no_reversal_signals(self, monitor):
        """No strong or moderate coins triggers intact-trends recommendation."""
        summary = {
            "high_loss_risk_coins": [],
            "medium_loss_risk_coins": [],
            "low_loss_risk_coins": [],
            "loss_risk_percentage": 0,
        }
        recs = monitor._generate_reversal_recommendations(summary, {})
        assert any("No significant reversal" in r for r in recs)

    def test_multiple_strong_signals(self, monitor):
        """>= ALIGNMENT_STRENGTH_ALL strong coins triggers multiple-strong recommendation."""
        strong_coins = [f"COIN{i}" for i in range(ALIGNMENT_STRENGTH_ALL + 1)]
        summary = {
            "high_loss_risk_coins": strong_coins,
            "medium_loss_risk_coins": [],
            "low_loss_risk_coins": [],
            "loss_risk_percentage": 60.0,
        }
        recs = monitor._generate_reversal_recommendations(summary, {})
        assert any("Multiple strong reversal" in r for r in recs)
        assert any("Reversal signal percentage above" in r for r in recs)
        assert any("Reversal signals flagged" in r for r in recs)

    def test_several_strong_signals(self, monitor):
        """ALIGNMENT_STRENGTH_MANY strong coins triggers several-strong recommendation."""
        strong_coins = [f"COIN{i}" for i in range(ALIGNMENT_STRENGTH_MANY)]
        summary = {
            "high_loss_risk_coins": strong_coins,
            "medium_loss_risk_coins": [],
            "low_loss_risk_coins": [],
            "loss_risk_percentage": 40.0,
        }
        recs = monitor._generate_reversal_recommendations(summary, {})
        assert any("Several strong reversal" in r for r in recs)
        assert any("moderate reversal frequency" in r for r in recs)

    def test_moderate_reversal_percentage(self, monitor):
        """Reversal % between REVERSAL_PCT_MODERATE and REVERSAL_PCT_HIGH."""
        summary = {
            "high_loss_risk_coins": [],
            "medium_loss_risk_coins": ["ETH"],
            "low_loss_risk_coins": [],
            "loss_risk_percentage": 35.0,
        }
        recs = monitor._generate_reversal_recommendations(summary, {})
        assert any("moderate reversal frequency" in r for r in recs)

    def test_strong_coins_listed(self, monitor):
        """Strong coins appear in the joined string recommendation."""
        summary = {
            "high_loss_risk_coins": ["XRP", "SOL"],
            "medium_loss_risk_coins": [],
            "low_loss_risk_coins": [],
            "loss_risk_percentage": 60.0,
        }
        recs = monitor._generate_reversal_recommendations(summary, {})
        assert any("XRP" in r and "SOL" in r and "Strong reversal" in r for r in recs)


# ── test_analyze_performance ─────────────────────────────────────────────────


class TestAnalyzePerformance:
    """Full analyze_performance with mock data written via write_test_json."""

    def test_full_analysis(self, monitor, write_test_json):
        """Complete analysis with cycles, trades, and portfolio returns a valid report."""
        cycles = [
            {
                "decisions": {
                    "XRP": {"signal": "buy_to_enter"},
                    "DOGE": {"signal": "hold"},
                },
            },
            {
                "decisions": {
                    "SOL": {"signal": "sell_to_enter"},
                    "TRX": {"signal": "close_position"},
                },
            },
        ]
        trades = [
            {"symbol": "XRP", "pnl": 10.0},
            {"symbol": "DOGE", "pnl": -3.0},
            {"symbol": "SOL", "pnl": 5.0},
        ]
        portfolio = {
            "initial_balance": 200.0,
            "current_balance": 212.0,
            "total_value": 220.0,
            "total_return": 10.0,
            "sharpe_ratio": 1.2,
            "total_value_history": [200.0, 205.0, 210.0, 208.0, 220.0],
            "positions": {"XRP": {}},
        }

        write_test_json("cycle_history.json", cycles)
        write_test_json("trade_history.json", trades)
        write_test_json("portfolio_state.json", portfolio)

        report = monitor.analyze_performance(last_n_cycles=10)

        # Top-level keys
        assert "trading_activity" in report
        assert "trade_performance" in report
        assert "portfolio_performance" in report
        assert "coin_performance" in report
        assert "recommendations" in report

        # Trading activity counts
        activity = report["trading_activity"]
        assert activity["total_decisions"] == 4
        assert activity["entry_signals"] == 2
        assert activity["hold_signals"] == 1
        assert activity["close_signals"] == 1
        assert activity["decision_rate"] == pytest.approx(50.0)

        # Trade performance
        tp = report["trade_performance"]
        assert tp["total_trades"] == 3
        assert tp["winning_trades"] == 2
        assert tp["losing_trades"] == 1
        assert tp["total_pnl"] == pytest.approx(12.0)
        assert tp["average_pnl"] == pytest.approx(4.0)
        assert tp["profit_factor"] == pytest.approx(5.0)
        assert tp["largest_win"] == 10.0
        assert tp["largest_loss"] == -3.0

        # Portfolio performance
        pp = report["portfolio_performance"]
        assert pp["initial_balance"] == 200.0
        assert pp["current_balance"] == 212.0
        assert pp["total_return"] == pytest.approx(10.0)
        assert pp["sharpe_ratio"] == pytest.approx(1.2)
        assert pp["sortino_ratio"] == pytest.approx(
            monitor._calculate_sortino_ratio(portfolio["total_value_history"]),
        )

        # Coin performance
        cp = report["coin_performance"]
        assert "XRP" in cp
        assert cp["XRP"]["trades"] == 1
        assert cp["XRP"]["total_pnl"] == pytest.approx(10.0)
        assert cp["DOGE"]["total_pnl"] == pytest.approx(-3.0)

    def test_profitability_index_calculation(self, monitor, write_test_json):
        """Profitability index = profit / (profit + loss) * 100."""
        cycles = [{"decisions": {"XRP": {"signal": "buy_to_enter"}}}]
        trades = [
            {"symbol": "XRP", "pnl": 30.0},
            {"symbol": "DOGE", "pnl": -10.0},
        ]
        write_test_json("cycle_history.json", cycles)
        write_test_json("trade_history.json", trades)
        write_test_json("portfolio_state.json", {"positions": {}})

        report = monitor.analyze_performance()
        assert report["trade_performance"]["profitability_index"] == pytest.approx(75.0)

    def test_all_break_even_trades(self, monitor, write_test_json):
        """All trades break even gives 0 profitability index."""
        cycles = [{"decisions": {"XRP": {"signal": "hold"}}}]
        trades = [
            {"symbol": "XRP", "pnl": 0.0},
            {"symbol": "DOGE", "pnl": 0.0},
        ]
        write_test_json("cycle_history.json", cycles)
        write_test_json("trade_history.json", trades)
        write_test_json("portfolio_state.json", {"positions": {}})

        report = monitor.analyze_performance()
        assert report["trade_performance"]["profitability_index"] == 0.0
        # total_profit==0 and total_loss==0 -> code returns inf
        assert report["trade_performance"]["profit_factor"] == float("inf")


# ── test_analyze_performance_no_data ─────────────────────────────────────────


class TestAnalyzePerformanceNoData:
    """analyze_performance returns info message when data is missing."""

    def test_no_cycles_file(self, monitor):
        """Missing cycle history returns info message."""
        # No files written — defaults are empty
        result = monitor.analyze_performance()
        assert "info" in result
        assert "No valid trading data" in result["info"]

    def test_empty_cycles_list(self, monitor, write_test_json):
        """Empty cycles list returns info message."""
        write_test_json("cycle_history.json", [])
        result = monitor.analyze_performance()
        assert "info" in result
        assert "No valid trading data" in result["info"]

    def test_cycles_not_a_list(self, monitor, write_test_json):
        """Non-list cycles returns info message."""
        write_test_json("cycle_history.json", "invalid")
        result = monitor.analyze_performance()
        assert "info" in result

    def test_cycles_without_decisions_key(self, monitor, write_test_json):
        """Cycles without 'decisions' key returns info message."""
        write_test_json("cycle_history.json", [{"no_decisions": True}])
        result = monitor.analyze_performance()
        assert "info" in result
        assert "No valid cycle data" in result["info"]

    def test_no_trades_returns_zero_trade_performance(self, monitor, write_test_json):
        """Valid cycles but no trades still returns a full report with zeroed trade stats."""
        cycles = [{"decisions": {"XRP": {"signal": "hold"}}}]
        write_test_json("cycle_history.json", cycles)
        write_test_json("trade_history.json", [])
        write_test_json("portfolio_state.json", {"positions": {}})

        report = monitor.analyze_performance()
        assert report["trade_performance"]["total_trades"] == 0
        assert report["trade_performance"]["total_pnl"] == 0.0
        assert report["trade_performance"]["profitability_index"] == 0.0


# ── test_sortino_single_downside ─────────────────────────────────────────────


class TestSortinoSingleDownside:
    """Sortino ratio with exactly one downside return (stdev fallback to 0.0)."""

    def test_single_downside_return_returns_inf(self, monitor):
        """Exactly one negative return -> stdev returns 0.0 -> ratio is inf."""
        # Values: 100, 90, 95, 97, 99
        # Returns: -0.1, 0.0556, 0.0211, 0.0206
        # Only one downside return: -0.1
        values = [100.0, 90.0, 95.0, 97.0, 99.0]
        result = monitor._calculate_sortino_ratio(values)
        assert result == float("inf")

    def test_single_downside_at_start(self, monitor):
        """One drop at the start, then monotonically increasing."""
        values = [100.0, 80.0, 90.0, 100.0, 110.0]
        result = monitor._calculate_sortino_ratio(values)
        assert result == float("inf")

    def test_single_downside_at_end(self, monitor):
        """Monotonically increasing then one drop at the end."""
        values = [100.0, 110.0, 120.0, 130.0, 125.0]
        result = monitor._calculate_sortino_ratio(values)
        assert result == float("inf")

    def test_two_downside_returns_finite(self, monitor):
        """Two downside returns -> stdev is finite -> ratio is finite."""
        # Values: 100, 90, 95, 85, 90
        # Returns: -0.1, 0.0556, -0.1053, 0.0588
        # Two downside returns: -0.1 and -0.1053
        values = [100.0, 90.0, 95.0, 85.0, 90.0]
        result = monitor._calculate_sortino_ratio(values)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_all_downside_returns_finite(self, monitor):
        """All returns negative -> stdev is finite -> ratio is finite."""
        values = [100.0, 95.0, 90.0, 85.0, 80.0]
        result = monitor._calculate_sortino_ratio(values)
        assert isinstance(result, float)
        assert math.isfinite(result)
        assert result < 0
