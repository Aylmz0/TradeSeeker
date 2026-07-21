"""Tests for src/core/performance_monitor.py."""

import math

import pytest

from src.core.constants import MIN_HISTORY_FOR_ANALYSIS, SORTINO_MIN_RETURNS
from src.core.performance_monitor import PerformanceMonitor


@pytest.fixture
def monitor():
    return PerformanceMonitor()


class TestCalculateMaxDrawdown:
    """Tests for _calculate_max_drawdown."""

    def test_calculate_max_drawdown(self, monitor):
        # Peak 100 -> drop to 80 -> recover to 90 -> drop to 70
        # Max drawdown: (100-70)/100*100 = 30%
        values = [100.0, 90.0, 100.0, 80.0, 90.0, 70.0]
        result = monitor._calculate_max_drawdown(values)
        assert result == pytest.approx(-30.0)

    def test_calculate_max_drawdown_monotonic_increasing(self, monitor):
        values = [100.0, 110.0, 120.0, 130.0]
        result = monitor._calculate_max_drawdown(values)
        assert result == pytest.approx(0.0)

    def test_calculate_max_drawdown_empty(self, monitor):
        assert monitor._calculate_max_drawdown([]) == 0.0

    def test_calculate_max_drawdown_insufficient_data(self, monitor):
        # Less than MIN_HISTORY_FOR_ANALYSIS
        assert monitor._calculate_max_drawdown([100.0]) == 0.0

    def test_calculate_max_drawdown_single_drop(self, monitor):
        values = [200.0, 100.0]
        result = monitor._calculate_max_drawdown(values)
        assert result == pytest.approx(-50.0)


class TestCalculateSortinoRatio:
    """Tests for _calculate_sortino_ratio."""

    def test_calculate_sortino_ratio(self, monitor):
        # Mix of positive and negative returns
        # Values: 100, 105, 102, 108, 103
        # Returns: 0.05, -0.0286, 0.0588, -0.0463
        values = [100.0, 105.0, 102.0, 108.0, 103.0]
        result = monitor._calculate_sortino_ratio(values)
        # Should be a finite number (not inf or 0)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_calculate_sortino_ratio_no_downside(self, monitor):
        # Monotonically increasing -> no negative returns -> inf
        values = [100.0, 110.0, 120.0, 130.0, 140.0]
        result = monitor._calculate_sortino_ratio(values)
        assert result == float("inf")

    def test_calculate_sortino_ratio_empty(self, monitor):
        assert monitor._calculate_sortino_ratio([]) == 0.0

    def test_calculate_sortino_ratio_insufficient_data(self, monitor):
        # Less than SORTINO_MIN_RETURNS
        assert monitor._calculate_sortino_ratio([100.0, 110.0]) == 0.0


class TestGenerateRecommendations:
    """Tests for _generate_recommendations."""

    def test_generate_recommendations(self, monitor, tmp_path):
        # Set up portfolio file with low balance
        monitor.portfolio_state_file = str(tmp_path / "portfolio.json")

        from src.utils import safe_file_write

        safe_file_write(
            monitor.portfolio_state_file,
            {
                "current_balance": 50.0,
                "initial_balance": 200.0,
                "total_return": -2.0,
            },
        )

        coin_perf = {"XRP": {"total_pnl": -5.0}}

        recs = monitor._generate_recommendations(
            profitability_index=45.0,
            profit_factor=0.9,
            coin_performance=coin_perf,
            open_positions=4,
        )

        assert isinstance(recs, list)
        assert len(recs) > 0
        assert all(isinstance(r, str) for r in recs)

    def test_generate_recommendations_no_issues(self, monitor, tmp_path):
        monitor.portfolio_state_file = str(tmp_path / "portfolio.json")

        from src.utils import safe_file_write

        safe_file_write(
            monitor.portfolio_state_file,
            {
                "current_balance": 200.0,
                "initial_balance": 200.0,
                "total_return": 10.0,
            },
        )

        recs = monitor._generate_recommendations(
            profitability_index=60.0,
            profit_factor=2.0,
            coin_performance={},
            open_positions=1,
        )

        assert isinstance(recs, list)
        assert len(recs) == 1
        assert "stable" in recs[0].lower() or "no notable" in recs[0].lower()
