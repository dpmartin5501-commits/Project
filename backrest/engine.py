"""Main Backrest engine that orchestrates search, backtest, and ranking."""

from __future__ import annotations

from dataclasses import dataclass

from .backtester import BacktestResult, Backtester
from .data_fetcher import DataFetcher, FetchConfig
from .ranker import FilterConfig, StrategyRanker
from .reporter import Reporter
from .strategies.implementations import get_all_strategies
from .strategy_search import StrategySearcher


@dataclass
class EngineConfig:
    exchange: str = "binance"
    symbols: list[str] | None = None
    timeframe: str = "1d"
    since_days: int = 365
    initial_capital: float = 10000.0
    commission_pct: float = 0.001
    min_win_rate: float = 50.0
    max_drawdown: float = 25.0
    min_trades: int = 5
    search_internet: bool = True
    top_n: int = 5
    verbose: bool = True

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


class BackrestEngine:
    """End-to-end engine: search strategies online, backtest them, rank by drawdown/win rate."""

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.reporter = Reporter()

    def run(self) -> list[BacktestResult]:
        """Execute the full pipeline: search -> fetch data -> backtest -> rank -> report."""
        console = self.reporter.console

        console.rule("[bold cyan]BACKREST - Crypto Strategy Search Engine[/bold cyan]")
        console.print()
        console.print(f"[dim]Exchange: {self.config.exchange}[/dim]")
        console.print(f"[dim]Symbols: {', '.join(self.config.symbols)}[/dim]")
        console.print(f"[dim]Timeframe: {self.config.timeframe}[/dim]")
        console.print(f"[dim]History: {self.config.since_days} days[/dim]")
        console.print(f"[dim]Capital: ${self.config.initial_capital:,.0f}[/dim]")
        console.print()

        # --- Phase 1: Search for strategies ---
        console.rule("[bold]Phase 1: Searching for Strategies[/bold]")
        searched_strategies = self._search_strategies()

        # --- Phase 2: Fetch market data ---
        console.rule("[bold]Phase 2: Fetching Historical Data[/bold]")
        market_data = self._fetch_data()
        if not market_data:
            console.print("[red]No market data fetched. Aborting.[/red]")
            return []

        # --- Phase 3: Backtest ---
        console.rule("[bold]Phase 3: Running Backtests[/bold]")
        all_results = self._run_backtests(market_data)
        if self.config.verbose:
            self.reporter.print_backtest_results(all_results, title="All Backtest Results")

        # --- Phase 4: Filter and Rank ---
        console.rule("[bold]Phase 4: Filtering & Ranking[/bold]")
        top_results = self._filter_and_rank(all_results)

        # --- Phase 5: Report ---
        console.rule("[bold]Phase 5: Top Strategies[/bold]")
        if top_results:
            self.reporter.print_backtest_results(top_results, title="Filtered Results (Low DD + High WR)")
            self.reporter.print_top_strategies(top_results, self.config.top_n)
        else:
            console.print("[yellow]No strategies passed the filter criteria.[/yellow]")
            console.print("[dim]Showing best available results instead...[/dim]")
            ranker = StrategyRanker()
            best = ranker.get_all_ranked(all_results)[:self.config.top_n]
            self.reporter.print_backtest_results(best, title="Best Available (unfiltered)")
            self.reporter.print_top_strategies(best, self.config.top_n)
            top_results = best

        self.reporter.print_summary(
            total_searched=len(searched_strategies),
            total_backtested=len(all_results),
            total_passed=len(top_results),
        )

        return top_results

    def _search_strategies(self):
        searcher = StrategySearcher()
        console = self.reporter.console

        if self.config.search_internet:
            console.print("[cyan]Searching the internet for crypto trading strategies...[/cyan]")
            strategies = searcher.search_strategies(max_results=20)
        else:
            console.print("[cyan]Using built-in strategy library...[/cyan]")
            strategies = searcher._get_builtin_strategies()

        self.reporter.print_search_results(strategies)
        console.print(f"\n[green]Found {len(strategies)} strategies[/green]")
        return strategies

    def _fetch_data(self) -> dict[str, __import__("pandas").DataFrame]:
        console = self.reporter.console
        fetch_config = FetchConfig(
            exchange_id=self.config.exchange,
            symbols=self.config.symbols,
            timeframe=self.config.timeframe,
            since_days=self.config.since_days,
        )
        fetcher = DataFetcher(fetch_config)
        data = fetcher.fetch_all()

        for symbol, df in data.items():
            console.print(f"  [green]{symbol}[/green]: {len(df)} candles")

        return data

    def _run_backtests(self, data) -> list[BacktestResult]:
        console = self.reporter.console
        strategies = get_all_strategies(
            initial_capital=self.config.initial_capital,
            commission_pct=self.config.commission_pct,
        )

        backtester = Backtester(
            initial_capital=self.config.initial_capital,
            commission_pct=self.config.commission_pct,
        )

        console.print(f"[cyan]Running {len(strategies)} strategies across {len(data)} symbols...[/cyan]")
        results = backtester.run_multiple(strategies, data)
        console.print(f"[green]Completed {len(results)} backtests[/green]")
        return results

    def _filter_and_rank(self, results: list[BacktestResult]) -> list[BacktestResult]:
        console = self.reporter.console
        filter_config = FilterConfig(
            min_win_rate=self.config.min_win_rate,
            max_drawdown=self.config.max_drawdown,
            min_trades=self.config.min_trades,
        )
        ranker = StrategyRanker(filter_config)
        filtered = ranker.filter_and_rank(results)
        console.print(
            f"[cyan]Filtered: {len(filtered)}/{len(results)} strategies meet criteria "
            f"(WR >= {self.config.min_win_rate}%, DD <= {self.config.max_drawdown}%)[/cyan]"
        )
        return filtered
