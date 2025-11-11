"""
Diagnostic script to probe counter-trend scoring and short-scenario handling.

Run with:
    python short_scenario_tests.py

It constructs synthetic indicator snapshots and feeds them directly into
PortfolioManager.validate_counter_trade / _is_counter_trend_trade to ensure
the new weighted scoring accepts valid counter-trend shorts while rejecting
illiquid setups, and that trend-following shorts are classified correctly.
"""

from pprint import pprint

try:
    from alpha_arena_deepseek import PortfolioManager
except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard for missing deps
    raise SystemExit(
        "alpha_arena_deepseek dependencies are missing. "
        "Install project requirements (e.g. pandas) before running this diagnostic."
    ) from exc


def build_pm_stub() -> PortfolioManager:
    """
    Create a PortfolioManager instance without running its heavy __init__ logic.
    We only need access to validation helpers that do not depend on internal state.
    """
    pm = PortfolioManager.__new__(PortfolioManager)  # type: ignore
    return pm


def main() -> None:
    pm = build_pm_stub()

    scenarios = [
        {
            "name": "counter_short_high_conf",
            "signal": "sell_to_enter",
            "indicators_htf": {
                "current_price": 15.5,
                "ema_20": 15.2,
            },  # Bullish HTF → short is counter-trend
            "indicators_3m": {
                "current_price": 15.0,
                "ema_20": 15.05,
                "volume": 9_000_000,
                "avg_volume": 5_000_000,
                "rsi_14": 74.0,
                "macd": -0.018,
                "macd_signal": -0.010,
            },
        },
        {
            "name": "counter_short_low_volume_blocked",
            "signal": "sell_to_enter",
            "indicators_htf": {
                "current_price": 12.8,
                "ema_20": 12.5,
            },  # Still counter-trend short
            "indicators_3m": {
                "current_price": 12.3,
                "ema_20": 12.4,
                "volume": 150_000,
                "avg_volume": 1_000_000,
                "rsi_14": 78.0,
                "macd": -0.011,
                "macd_signal": -0.005,
            },
        },
        {
            "name": "trend_following_short_classification",
            "signal": "sell_to_enter",
            "indicators_htf": {
                "current_price": 9.4,
                "ema_20": 9.9,
            },  # Bearish HTF → short is trend-following
            "indicators_3m": {
                "current_price": 9.2,
                "ema_20": 9.3,
                "volume": 2_000_000,
                "avg_volume": 1_000_000,
                "rsi_14": 55.0,
                "macd": -0.004,
                "macd_signal": -0.006,
            },
        },
    ]

    for scenario in scenarios:
        print("=" * 80)
        print(f"Scenario: {scenario['name']}")

        is_counter = pm._is_counter_trend_trade(
            coin="TEST",
            signal=scenario["signal"],
            indicators_3m=scenario["indicators_3m"],
            indicators_htf=scenario["indicators_htf"],
        )
        print(f"  Counter-trend detected: {is_counter}")

        result = pm.validate_counter_trade(
            coin="TEST",
            signal=scenario["signal"],
            indicators_3m=scenario["indicators_3m"],
            indicators_htf=scenario["indicators_htf"],
        )
        print("  Validation result:")
        pprint(result, indent=4, sort_dicts=False)


if __name__ == "__main__":
    main()

