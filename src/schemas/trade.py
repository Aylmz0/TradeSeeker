"""Trade history and cycle history schemas."""

from pydantic import BaseModel, ConfigDict, Field


class TradeHistoryEntry(BaseModel):
    """A completed trade record."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    notional_usd: float = 0.0
    pnl: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    leverage: int = Field(default=10, ge=1)
    close_reason: str = ""


class DirectionalBias(BaseModel):
    """Directional bias tracking for long/short."""

    model_config = ConfigDict(frozen=True)

    net_pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    caution_active: bool = False
    caution_win_progress: int = 0
    loss_streak_loss_usd: float = 0.0


class CycleHistoryEntry(BaseModel):
    """A single cycle record in history."""

    model_config = ConfigDict(frozen=True)

    cycle: int
    timestamp: str = ""
    user_prompt_summary: str = ""
    chain_of_thoughts: str = ""
    decisions: dict = Field(default_factory=dict)
    status: str = "idle"  # ai_decision | tp_sl_only | manual_override | error | idle
    cooldown_status: dict = Field(default_factory=dict)
    metadata: dict | None = None
