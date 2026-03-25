"""Core backtesting engine that runs strategies against historical data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategies.base import BaseStrategy, Signal, Trade


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    timeframe: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    initial_capital: float = 10000.0
    final_capital: float = 10000.0

    # Computed metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_holding_periods: float = 0.0
    max_consecutive_losses: int = 0
    expectancy: float = 0.0


class Backtester:
    """Event-driven backtester that simulates strategy execution on historical data."""

    def __init__(self, initial_capital: float = 10000.0, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(self, strategy: BaseStrategy, df: pd.DataFrame, symbol: str = "UNKNOWN") -> BacktestResult:
        """Run a single strategy against a price DataFrame."""
        if df.empty or len(df) < 10:
            return BacktestResult(
                strategy_name=strategy.name, symbol=symbol, timeframe="unknown"
            )

        df = strategy.prepare_indicators(df.copy())
        signals = strategy.generate_signals(df)

        capital = self.initial_capital
        position_open = False
        current_trade: Trade | None = None
        trades: list[Trade] = []
        equity: list[float] = []

        for i in range(len(df)):
            row = df.iloc[i]
            signal = signals.iloc[i]
            price = row["close"]

            if position_open and current_trade:
                current_trade.holding_periods += 1

                if strategy.config.stop_loss_pct is not None:
                    stop_price = current_trade.entry_price * (1 - strategy.config.stop_loss_pct)
                    if row["low"] <= stop_price:
                        adj_price = stop_price * (1 - self.slippage_pct)
                        commission = adj_price * self.commission_pct
                        current_trade.close(row["timestamp"], adj_price - commission)
                        capital += current_trade.pnl
                        trades.append(current_trade)
                        position_open = False
                        current_trade = None
                        equity.append(capital)
                        continue

                if strategy.config.take_profit_pct is not None:
                    tp_price = current_trade.entry_price * (1 + strategy.config.take_profit_pct)
                    if row["high"] >= tp_price:
                        adj_price = tp_price * (1 - self.slippage_pct)
                        commission = adj_price * self.commission_pct
                        current_trade.close(row["timestamp"], adj_price - commission)
                        capital += current_trade.pnl
                        trades.append(current_trade)
                        position_open = False
                        current_trade = None
                        equity.append(capital)
                        continue

            if signal == Signal.BUY and not position_open:
                adj_price = price * (1 + self.slippage_pct)
                commission = adj_price * self.commission_pct
                entry_price = adj_price + commission
                size = capital * strategy.config.position_size_pct
                current_trade = Trade(
                    entry_date=row["timestamp"],
                    entry_price=entry_price,
                    size=size,
                )
                position_open = True

            elif signal == Signal.SELL and position_open and current_trade:
                adj_price = price * (1 - self.slippage_pct)
                commission = adj_price * self.commission_pct
                current_trade.close(row["timestamp"], adj_price - commission)
                capital += current_trade.pnl
                trades.append(current_trade)
                position_open = False
                current_trade = None

            if position_open and current_trade:
                unrealized = ((price - current_trade.entry_price) / current_trade.entry_price) * current_trade.size
                equity.append(capital + unrealized)
            else:
                equity.append(capital)

        if position_open and current_trade:
            last_row = df.iloc[-1]
            current_trade.close(last_row["timestamp"], last_row["close"])
            capital += current_trade.pnl
            trades.append(current_trade)
            equity[-1] = capital

        equity_series = pd.Series(equity, index=df["timestamp"])
        result = self._compute_metrics(strategy.name, symbol, trades, equity_series, self.initial_capital, capital)
        return result

    def run_multiple(
        self, strategies: list[BaseStrategy], data: dict[str, pd.DataFrame]
    ) -> list[BacktestResult]:
        """Run multiple strategies against multiple symbols."""
        results: list[BacktestResult] = []
        for strategy in strategies:
            for symbol, df in data.items():
                result = self.run(strategy, df, symbol)
                results.append(result)
        return results

    def _compute_metrics(
        self,
        strategy_name: str,
        symbol: str,
        trades: list[Trade],
        equity_curve: pd.Series,
        initial_capital: float,
        final_capital: float,
    ) -> BacktestResult:
        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe="1d",
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            final_capital=final_capital,
        )

        if not trades:
            return result

        result.total_trades = len(trades)
        closed = [t for t in trades if t.is_closed]
        if not closed:
            return result

        winners = [t for t in closed if t.pnl_pct > 0]
        losers = [t for t in closed if t.pnl_pct <= 0]

        result.winning_trades = len(winners)
        result.losing_trades = len(losers)
        result.win_rate = len(winners) / len(closed) * 100 if closed else 0

        result.total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100

        if len(equity_curve) > 1:
            trading_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if trading_days > 0:
                years = trading_days / 365.25
                if years > 0:
                    result.annualized_return_pct = (
                        ((final_capital / initial_capital) ** (1 / years) - 1) * 100
                    )

        if winners:
            result.avg_win_pct = np.mean([t.pnl_pct for t in winners]) * 100
        if losers:
            result.avg_loss_pct = np.mean([t.pnl_pct for t in losers]) * 100

        total_wins = sum(t.pnl_pct for t in winners) if winners else 0
        total_losses = abs(sum(t.pnl_pct for t in losers)) if losers else 0
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        result.max_drawdown_pct = abs(drawdown.min()) if len(drawdown) > 0 else 0

        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

                downside = returns[returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    result.sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(252)

        if result.max_drawdown_pct > 0:
            result.calmar_ratio = result.annualized_return_pct / result.max_drawdown_pct

        if closed:
            result.avg_holding_periods = np.mean([t.holding_periods for t in closed])

        max_consec = 0
        current_consec = 0
        for t in closed:
            if t.pnl_pct <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        result.max_consecutive_losses = max_consec

        if closed:
            result.expectancy = np.mean([t.pnl_pct for t in closed]) * 100

        return result
