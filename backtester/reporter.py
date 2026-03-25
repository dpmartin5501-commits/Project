"""Rich terminal reporting for backtest results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .backtester import BacktestResult
from .evolver import EvolutionResult
from .strategy_search import StrategyResult


class Reporter:
    """Formats and displays backtest results using Rich tables."""

    def __init__(self):
        self.console = Console()

    def print_search_results(self, strategies: list[StrategyResult]) -> None:
        """Display strategies found from internet search."""
        table = Table(
            title="Crypto Trading Strategies Found",
            show_lines=True,
            title_style="bold cyan",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Strategy", style="bold white", max_width=35)
        table.add_column("Indicators", style="yellow", max_width=25)
        table.add_column("Claimed WR", justify="center", width=10)
        table.add_column("Claimed DD", justify="center", width=10)
        table.add_column("Tags", style="dim", max_width=30)
        table.add_column("Source", style="dim cyan", max_width=25)

        for i, s in enumerate(strategies, 1):
            wr = f"{s.claimed_win_rate:.0f}%" if s.claimed_win_rate else "-"
            dd = f"{s.claimed_max_drawdown:.0f}%" if s.claimed_max_drawdown else "-"
            source = s.source_url if len(s.source_url) <= 25 else s.source_url[:22] + "..."
            table.add_row(
                str(i),
                s.name,
                ", ".join(s.indicators) or "-",
                wr,
                dd,
                ", ".join(s.tags) or "-",
                source,
            )

        self.console.print()
        self.console.print(table)

    def print_backtest_results(self, results: list[BacktestResult], title: str = "Backtest Results") -> None:
        """Display backtest results in a detailed table."""
        if not results:
            self.console.print("[yellow]No backtest results to display.[/yellow]")
            return

        table = Table(title=title, show_lines=True, title_style="bold green")
        table.add_column("#", style="dim", width=3)
        table.add_column("Strategy", style="bold white", max_width=30)
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Trades", justify="center", width=7)
        table.add_column("Win Rate", justify="center", width=9)
        table.add_column("Max DD", justify="center", width=9)
        table.add_column("Return", justify="center", width=10)
        table.add_column("Sharpe", justify="center", width=8)
        table.add_column("P.Factor", justify="center", width=9)
        table.add_column("Avg Win", justify="center", width=8)
        table.add_column("Avg Loss", justify="center", width=8)
        table.add_column("Calmar", justify="center", width=8)

        for i, r in enumerate(results, 1):
            wr_color = "green" if r.win_rate >= 55 else ("yellow" if r.win_rate >= 45 else "red")
            dd_color = "green" if r.max_drawdown_pct <= 15 else ("yellow" if r.max_drawdown_pct <= 25 else "red")
            ret_color = "green" if r.total_return_pct > 0 else "red"
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < float("inf") else "inf"

            table.add_row(
                str(i),
                r.strategy_name,
                r.symbol,
                str(r.total_trades),
                f"[{wr_color}]{r.win_rate:.1f}%[/{wr_color}]",
                f"[{dd_color}]{r.max_drawdown_pct:.1f}%[/{dd_color}]",
                f"[{ret_color}]{r.total_return_pct:.1f}%[/{ret_color}]",
                f"{r.sharpe_ratio:.2f}",
                pf_str,
                f"{r.avg_win_pct:.1f}%",
                f"{r.avg_loss_pct:.1f}%",
                f"{r.calmar_ratio:.2f}",
            )

        self.console.print()
        self.console.print(table)

    def print_top_strategies(self, results: list[BacktestResult], top_n: int = 5) -> None:
        """Highlight the top N strategies with detailed panels."""
        self.console.print()
        self.console.print(
            f"[bold green]Top {min(top_n, len(results))} Strategies "
            f"(Low Drawdown + High Win Rate)[/bold green]"
        )
        self.console.print()

        for i, r in enumerate(results[:top_n], 1):
            panel_text = Text()
            panel_text.append(f"Symbol: {r.symbol}\n")
            panel_text.append(f"Total Trades: {r.total_trades}\n")
            panel_text.append(f"Win Rate: {r.win_rate:.1f}%\n", style="bold green" if r.win_rate >= 55 else "yellow")
            panel_text.append(
                f"Max Drawdown: {r.max_drawdown_pct:.1f}%\n",
                style="bold green" if r.max_drawdown_pct <= 15 else "yellow",
            )
            panel_text.append(f"Total Return: {r.total_return_pct:.1f}%\n")
            panel_text.append(f"Annualized Return: {r.annualized_return_pct:.1f}%\n")
            panel_text.append(f"Sharpe Ratio: {r.sharpe_ratio:.2f}\n")
            panel_text.append(f"Sortino Ratio: {r.sortino_ratio:.2f}\n")
            panel_text.append(f"Profit Factor: {r.profit_factor:.2f}\n")
            panel_text.append(f"Calmar Ratio: {r.calmar_ratio:.2f}\n")
            panel_text.append(f"Avg Holding Period: {r.avg_holding_periods:.1f} bars\n")
            panel_text.append(f"Max Consecutive Losses: {r.max_consecutive_losses}\n")
            panel_text.append(f"Expectancy: {r.expectancy:.2f}%\n")

            border_color = "green" if r.win_rate >= 55 and r.max_drawdown_pct <= 20 else "yellow"
            self.console.print(
                Panel(
                    panel_text,
                    title=f"#{i} {r.strategy_name}",
                    border_style=border_color,
                    width=60,
                )
            )

    def print_evolution_results(self, evo_results: list[EvolutionResult]) -> None:
        """Display evolution results showing original vs evolved parameters and metrics."""
        if not evo_results:
            self.console.print("[yellow]No evolution results to display.[/yellow]")
            return

        table = Table(
            title="Strategy Evolution Results",
            show_lines=True,
            title_style="bold magenta",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Strategy", style="bold white", max_width=25)
        table.add_column("Orig WR", justify="center", width=9)
        table.add_column("Evol WR", justify="center", width=9)
        table.add_column("Orig DD", justify="center", width=9)
        table.add_column("Evol DD", justify="center", width=9)
        table.add_column("Orig Ret", justify="center", width=9)
        table.add_column("Evol Ret", justify="center", width=9)
        table.add_column("Improved", justify="center", width=9)

        for i, r in enumerate(evo_results, 1):
            orig = r.original_result
            evol = r.evolved_result
            o_wr = f"{orig.win_rate:.1f}%" if orig else "-"
            e_wr = f"{evol.win_rate:.1f}%" if evol else "-"
            o_dd = f"{orig.max_drawdown_pct:.1f}%" if orig else "-"
            e_dd = f"{evol.max_drawdown_pct:.1f}%" if evol else "-"
            o_ret = f"{orig.total_return_pct:.1f}%" if orig else "-"
            e_ret = f"{evol.total_return_pct:.1f}%" if evol else "-"
            imp = "[green]YES[/green]" if r.improved else "[red]NO[/red]"

            table.add_row(str(i), r.strategy_name, o_wr, e_wr, o_dd, e_dd, o_ret, e_ret, imp)

        self.console.print()
        self.console.print(table)

    def print_evolution_detail(self, evo_results: list[EvolutionResult], top_n: int = 5) -> None:
        """Show detailed panels for top evolved strategies."""
        improved = [r for r in evo_results if r.improved]
        to_show = improved[:top_n] if improved else evo_results[:top_n]

        self.console.print()
        label = "Improved" if improved else "Best Evolved"
        self.console.print(f"[bold magenta]Top {min(top_n, len(to_show))} {label} Strategies[/bold magenta]")
        self.console.print()

        for i, r in enumerate(to_show, 1):
            panel_text = Text()
            panel_text.append("--- Original Parameters ---\n", style="dim")
            for k, v in r.original_params.items():
                panel_text.append(f"  {k}: {v}\n")
            panel_text.append("\n")

            panel_text.append("--- Evolved Parameters ---\n", style="bold cyan")
            for k, v in r.evolved_params.items():
                orig_v = r.original_params.get(k)
                changed = orig_v != v
                style = "bold green" if changed else ""
                marker = " *" if changed else ""
                panel_text.append(f"  {k}: {v}{marker}\n", style=style)
            panel_text.append("\n")

            if r.original_result and r.evolved_result:
                o = r.original_result
                e = r.evolved_result

                def _delta(label, orig_val, evol_val, fmt=".1f", lower_better=False):
                    delta = evol_val - orig_val
                    if lower_better:
                        better = delta < 0
                    else:
                        better = delta > 0
                    arrow = "+" if delta > 0 else ""
                    color = "green" if better else ("red" if not better and delta != 0 else "")
                    panel_text.append(f"  {label}: {orig_val:{fmt}} -> {evol_val:{fmt}}", style="")
                    panel_text.append(f" ({arrow}{delta:{fmt}})\n", style=color)

                panel_text.append("--- Performance Comparison ---\n", style="dim")
                _delta("Win Rate %", o.win_rate, e.win_rate)
                _delta("Max Drawdown %", o.max_drawdown_pct, e.max_drawdown_pct, lower_better=True)
                _delta("Return %", o.total_return_pct, e.total_return_pct)
                _delta("Sharpe", o.sharpe_ratio, e.sharpe_ratio, fmt=".2f")
                _delta("Profit Factor", o.profit_factor if o.profit_factor < 1e6 else 0,
                       e.profit_factor if e.profit_factor < 1e6 else 0, fmt=".2f")
                _delta("Trades", o.total_trades, e.total_trades, fmt="d")

            panel_text.append(f"\n  Generations: {r.generations_run}")
            if r.fitness_history:
                panel_text.append(f" | Best fitness: {r.fitness_history[-1]:.4f}")

            border = "green" if r.improved else "yellow"
            self.console.print(Panel(
                panel_text,
                title=f"#{i} {r.strategy_name}",
                border_style=border,
                width=65,
            ))

    def print_summary(self, total_searched: int, total_backtested: int, total_passed: int,
                      total_evolved: int = 0, total_improved: int = 0) -> None:
        """Print a summary of the search, backtest, and evolution pipeline."""
        self.console.print()
        summary = Table(title="Pipeline Summary", title_style="bold magenta", show_lines=True)
        summary.add_column("Metric", style="bold")
        summary.add_column("Value", justify="right")
        summary.add_row("Strategies Found (Search)", str(total_searched))
        summary.add_row("Strategies Backtested", str(total_backtested))
        summary.add_row("Strategies Passing Filters", str(total_passed))
        if total_evolved > 0:
            summary.add_row("Strategies Evolved", str(total_evolved))
            summary.add_row("Strategies Improved", str(total_improved))
        self.console.print(summary)
