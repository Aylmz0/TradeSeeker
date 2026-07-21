"""Unit tests for src/ai/deepseek_api.py — fallback/simulation paths only."""

import json

import pytest

from src.ai.deepseek_api import DeepSeekAPI


@pytest.fixture
def api():
    """Create a DeepSeekAPI instance with no router (all API keys unset)."""
    return DeepSeekAPI()


def test_simulation_response(api):
    """Simulation mode returns valid JSON with DECISIONS key."""
    # Remove router to force simulation path
    if hasattr(api, "router"):
        delattr(api, "router")
    result = api.get_ai_decision("{}")
    assert isinstance(result, str)
    data = json.loads(result)
    assert "CHAIN_OF_THOUGHTS" in data
    assert "DECISIONS" in data
    assert isinstance(data["DECISIONS"], dict)


def test_safe_hold_decisions(api):
    """Safe hold returns hold signal for all six coins."""
    result = api.get_safe_hold_decisions()
    data = json.loads(result)
    assert "DECISIONS" in data
    decisions = data["DECISIONS"]
    assert len(decisions) == 6
    for coin in ["XRP", "DOGE", "ASTER", "TRX", "ETH", "SOL"]:
        assert coin in decisions
        assert decisions[coin]["signal"] == "hold"


def test_error_response_connection(api):
    """Connection-type error returns cached decisions (falls back to safe hold)."""
    result = api._get_error_response("ConnectionError: refused")
    data = json.loads(result)
    assert "DECISIONS" in data
    for coin in ["XRP", "DOGE", "ASTER", "TRX", "ETH", "SOL"]:
        assert data["DECISIONS"][coin]["signal"] == "hold"


def test_error_response_generic(api):
    """Generic error returns safe hold decisions."""
    result = api._get_error_response("ValueError: bad input")
    data = json.loads(result)
    assert "DECISIONS" in data
    for coin in ["XRP", "DOGE", "ASTER", "TRX", "ETH", "SOL"]:
        assert data["DECISIONS"][coin]["signal"] == "hold"


def test_get_cached_decisions_empty(api, tmp_path, monkeypatch):
    """Empty cache returns safe hold decisions."""
    monkeypatch.setattr(
        "src.ai.deepseek_api.safe_file_read",
        lambda path, default_data=None: default_data,
    )
    result = api.get_cached_decisions()
    data = json.loads(result)
    assert "DECISIONS" in data
    for coin in ["XRP", "DOGE", "ASTER", "TRX", "ETH", "SOL"]:
        assert data["DECISIONS"][coin]["signal"] == "hold"


def test_build_system_prompt(api):
    """System prompt returns valid JSON with required keys."""
    result = api._build_system_prompt()
    data = json.loads(result)
    assert "agent_profile" in data
    assert "constraints" in data
    assert "strategy" in data
    assert "response_schema" in data
    assert data["agent_profile"]["role"] == "Elite Hybrid Intelligence Orchestrator (LLM + XGBoost)"


def test_ratelimit_error(api, monkeypatch):
    """RateLimit error falls back to cached decisions."""
    import litellm

    # Setup: mock router exists and raises RateLimitError
    class FakeRouter:
        model_list = []

        def completion(self, **kwargs):
            raise litellm.RateLimitError(
                message="Rate limit exceeded",
                model="openrouter/test",
                llm_provider="openrouter",
            )

    api.router = FakeRouter()
    api.primary_model = "openrouter/test"

    # Mock get_cached_decisions to return a known response
    monkeypatch.setattr(
        api,
        "get_cached_decisions",
        lambda: json.dumps(
            {"CHAIN_OF_THOUGHTS": "cached", "DECISIONS": {"SOL": {"signal": "buy_to_enter"}}}
        ),
    )

    result = api.get_ai_decision("{}")
    data = json.loads(result)
    assert "DECISIONS" in data
    assert data["DECISIONS"]["SOL"]["signal"] == "buy_to_enter"


def test_cached_decisions_with_ghost(api, monkeypatch):
    """Ghost entry blocking neutralizes signals for coins with open positions."""
    cached_cycles = [
        {
            "decisions": {
                "SOL": {"signal": "buy_to_enter", "confidence": 0.8},
                "XRP": {"signal": "sell_to_enter", "confidence": 0.7},
                "DOGE": {"signal": "hold"},
            }
        }
    ]
    portfolio_state = {"positions": {"SOL": {"side": "long"}}}

    def fake_read(path, default_data=None):
        if "cycle_history" in path:
            return cached_cycles
        if "portfolio_state" in path:
            return portfolio_state
        return default_data

    monkeypatch.setattr("src.ai.deepseek_api.safe_file_read", fake_read)

    result = api.get_cached_decisions()
    data = json.loads(result)

    # SOL is already open → ghost entry blocked → neutralized to hold
    assert data["DECISIONS"]["SOL"]["signal"] == "hold"
    assert "Cache fallback" in data["DECISIONS"]["SOL"]["justification"]

    # XRP is not open → signal preserved
    assert data["DECISIONS"]["XRP"]["signal"] == "sell_to_enter"

    # DOGE was already hold → unchanged
    assert data["DECISIONS"]["DOGE"]["signal"] == "hold"
