"""Rank and filter backtest results by drawdown, win rate, and other metrics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .backtester import BacktestResult


class RankCriteria(Enum):
    WIN_RATE = "win_rate"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    PROFIT_FACTOR = "profit_factor"
    TOTAL_RETURN = "total_return"
    COMPOSITE = "composite"


@dataclass
class FilterConfig:
    min_win_rate: float = 50.0
    max_drawdown: float = 25.0
    min_trades: int = 5
    min_profit_factor: float = 1.0
    min_sharpe: float | None = None
    rank_by: RankCriteria = RankCriteria.COMPOSITE


class StrategyRanker:
    """Filters and ranks backtest results to find strategies with low drawdown and high win rates."""

    def __init__(self, config: FilterConfig | None = None):
        self.config = config or FilterConfig()

    def filter_results(self, results: list[BacktestResult]) -> list[BacktestResult]:
        """Apply filters to keep only strategies meeting drawdown/win-rate thresholds."""
        filtered = []
        for r in results:
            if r.total_trades < self.config.min_trades:
                continue
            if r.win_rate < self.config.min_win_rate:
                continue
            if r.max_drawdown_pct > self.config.max_drawdown:
                continue
            if r.profit_factor < self.config.min_profit_factor:
                continue
            if self.config.min_sharpe is not None and r.sharpe_ratio < self.config.min_sharpe:
                continue
            filtered.append(r)
        return filtered

    def rank_results(self, results: list[BacktestResult]) -> list[BacktestResult]:
        """Sort results by the configured ranking criteria (best first)."""
        if not results:
            return results

        criteria = self.config.rank_by

        if criteria == RankCriteria.WIN_RATE:
            return sorted(results, key=lambda r: r.win_rate, reverse=True)
        elif criteria == RankCriteria.MAX_DRAWDOWN:
            return sorted(results, key=lambda r: r.max_drawdown_pct)
        elif criteria == RankCriteria.SHARPE_RATIO:
            return sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)
        elif criteria == RankCriteria.PROFIT_FACTOR:
            return sorted(results, key=lambda r: r.profit_factor, reverse=True)
        elif criteria == RankCriteria.TOTAL_RETURN:
            return sorted(results, key=lambda r: r.total_return_pct, reverse=True)
        else:
            return sorted(results, key=lambda r: self._composite_score(r), reverse=True)

    def filter_and_rank(self, results: list[BacktestResult]) -> list[BacktestResult]:
        """Apply filters first, then rank the survivors."""
        filtered = self.filter_results(results)
        return self.rank_results(filtered)

    def get_all_ranked(self, results: list[BacktestResult]) -> list[BacktestResult]:
        """Rank all results without filtering (for full overview)."""
        return self.rank_results(results)

    @staticmethod
    def _composite_score(r: BacktestResult) -> float:
        """Weighted composite score favoring low drawdown and high win rate.

        Weights:
        - Win rate: 30%
        - Inverse drawdown: 25%
        - Sharpe ratio: 20%
        - Profit factor: 15%
        - Calmar ratio: 10%
        """
        win_score = min(r.win_rate / 100, 1.0)

        dd_score = max(0, 1 - r.max_drawdown_pct / 50)

        sharpe_score = max(0, min(r.sharpe_ratio / 3, 1.0)) if r.sharpe_ratio > 0 else 0

        pf_score = min(r.profit_factor / 3, 1.0) if r.profit_factor < float("inf") else 1.0

        calmar_score = max(0, min(r.calmar_ratio / 3, 1.0))

        return (
            0.30 * win_score
            + 0.25 * dd_score
            + 0.20 * sharpe_score
            + 0.15 * pf_score
            + 0.10 * calmar_score
        )
