"""Base class for all backtestable strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    direction: str = "long"
    size: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_periods: int = 0

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None

    def close(self, exit_date: pd.Timestamp, exit_price: float) -> None:
        self.exit_date = exit_date
        self.exit_price = exit_price
        if self.direction == "long":
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        self.pnl = self.pnl_pct * self.size


@dataclass
class StrategyConfig:
    name: str = "Unnamed Strategy"
    initial_capital: float = 10000.0
    position_size_pct: float = 1.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    params: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """All strategies must inherit from this class and implement generate_signals()."""

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate a Series of Signal values aligned with the DataFrame index.

        The DataFrame will have columns: timestamp, open, high, low, close, volume
        plus any indicators computed by prepare_indicators().
        """

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators needed by the strategy. Override in subclass."""
        return df

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        """Return (min, max, step) for each tunable parameter.

        Override in subclasses. The evolver uses these ranges to generate
        mutations and constrain the search space.
        """
        return {}

    @property
    def name(self) -> str:
        return self.config.name
