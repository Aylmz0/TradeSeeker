"""AI decision and execution report schemas."""

from pydantic import BaseModel, Field


class MLConsensus(BaseModel):
    """ML model consensus data injected into AI decisions."""

    prediction: str | None = None
    confidence: float | None = None
    win_rate: float | None = None
    sample_count: int | None = None


class AIDecision(BaseModel):
    """A single coin's AI decision from the LLM response."""

    signal: str = "hold"  # buy_to_enter | sell_to_enter | close_position | hold
    confidence: float = Field(default=0.0, ge=0, le=1)
    leverage: int = Field(default=10, ge=1, le=20)
    quantity_usd: float = Field(default=0.0, ge=0)
    profit_target: float | None = None
    stop_loss: float | None = None
    invalidation_condition: str | None = None
    risk_usd: float = Field(default=0.0, ge=0)
    ml_consensus: MLConsensus | None = None
    reasoning: str | None = None
    runtime_decision: str | None = None


class ExecutedCoin(BaseModel):
    """A coin that was executed in a trade."""

    coin: str
    signal: str
    confidence: float = 0.0
    margin_usd: float = 0.0
    leverage: int = 10
    order_id: int | None = None


class BlockedCoin(BaseModel):
    """A coin that was blocked from trading."""

    coin: str
    reason: str
    signal: str | None = None


class SkippedCoin(BaseModel):
    """A coin that was skipped."""

    coin: str
    reason: str


class ExecutionReport(BaseModel):
    """Summary of trade execution for a cycle."""

    executed: list[ExecutedCoin] = Field(default_factory=list)
    blocked: list[BlockedCoin] = Field(default_factory=list)
    skipped: list[SkippedCoin] = Field(default_factory=list)
    holds: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    debug_logs: list = Field(default_factory=list)
    timestamp: str = ""
