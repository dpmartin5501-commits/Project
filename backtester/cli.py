"""Command-line interface for the Backtester crypto strategy search engine."""

from __future__ import annotations

import click

from .engine import BacktesterEngine, EngineConfig


@click.command()
@click.option("--exchange", "-e", default="binanceus", help="Exchange to fetch data from (ccxt-supported).")
@click.option("--symbols", "-s", default="BTC/USD,ETH/USD,SOL/USD", help="Comma-separated trading pairs.")
@click.option("--timeframe", "-t", default="1d", help="Candlestick timeframe (1m, 5m, 1h, 4h, 1d).")
@click.option("--days", "-d", default=365, help="Number of days of history to fetch.")
@click.option("--capital", "-c", default=10000.0, help="Initial capital for backtesting.")
@click.option("--commission", default=0.001, help="Commission per trade (0.001 = 0.1%).")
@click.option("--min-win-rate", default=50.0, help="Minimum win rate filter (%).")
@click.option("--max-drawdown", default=25.0, help="Maximum drawdown filter (%).")
@click.option("--min-trades", default=5, help="Minimum number of trades required.")
@click.option("--top", "-n", default=5, help="Number of top strategies to display.")
@click.option("--no-search", is_flag=True, help="Skip internet search, use only built-in strategies.")
@click.option("--quiet", "-q", is_flag=True, help="Reduce output verbosity.")
def main(
    exchange: str,
    symbols: str,
    timeframe: str,
    days: int,
    capital: float,
    commission: float,
    min_win_rate: float,
    max_drawdown: float,
    min_trades: int,
    top: int,
    no_search: bool,
    quiet: bool,
) -> None:
    """BACKTESTER - Search the internet for crypto trading strategies with low drawdown and high win rates.

    Searches for strategies online, backtests them against historical crypto data,
    and ranks results by win rate, drawdown, Sharpe ratio, and more.
    """
    symbol_list = [s.strip() for s in symbols.split(",")]

    config = EngineConfig(
        exchange=exchange,
        symbols=symbol_list,
        timeframe=timeframe,
        since_days=days,
        initial_capital=capital,
        commission_pct=commission,
        min_win_rate=min_win_rate,
        max_drawdown=max_drawdown,
        min_trades=min_trades,
        search_internet=not no_search,
        top_n=top,
        verbose=not quiet,
    )

    engine = BacktesterEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
