"""Position and ExitPlan schemas for portfolio state validation."""

from pydantic import BaseModel, ConfigDict, Field


class ExitPlan(BaseModel):
    """Exit strategy for a position."""

    model_config = ConfigDict(frozen=True)

    stop_loss: float | None = None
    profit_target: float | None = None
    invalidation_condition: str | None = None


class TrailingMeta(BaseModel):
    """Trailing stop metadata."""

    model_config = ConfigDict(frozen=True)

    active: bool = False
    stop_price: float | None = None
    highest_pnl: float | None = None
    lowest_pnl: float | None = None
    last_update_cycle: int | None = None
    last_reason: str | None = None
    last_stop: float | None = None
    progress_percent: float | None = None
    time_in_trade_min: float | None = None
    last_volume_ratio: float | None = None


class TrendContext(BaseModel):
    """Trend context at entry."""

    model_config = ConfigDict(frozen=True)

    trend_at_entry: str | None = None
    alignment: str | None = None
    cycle: int | None = None


class Position(BaseModel):
    """A single open position in the portfolio."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    direction: str  # "long" | "short"
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_time: str = ""
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    notional_usd: float = 0.0
    margin_usd: float = 0.0
    leverage: int = Field(default=10, ge=1)
    liquidation_price: float = 0.0
    confidence: float = Field(default=0.0, ge=0, le=1)
    exit_plan: ExitPlan = ExitPlan()
    risk_usd: float = 0.0
    loss_cycle_count: int = Field(default=0, ge=0)
    entry_volume: float | None = None
    entry_avg_volume: float | None = None
    entry_volume_ratio: float | None = None
    entry_atr_14: float | None = None
    trend_alignment: str = ""
    trend_context: TrendContext = TrendContext()
    trailing: TrailingMeta = TrailingMeta()
    sl_oid: int = -1
    tp_oid: int = -1
    entry_oid: int = -1
    wait_for_fill: bool = False
    peak_pnl: float = 0.0
    peak_pnl_cycle: int | None = None
    erosion_from_peak: float = 0.0
    erosion_pct: float = 0.0
    erosion_status: str = "NONE"
    profit_cycle_count: int = 0
    partial_exit_flags: dict[str, bool] = Field(default_factory=dict)
