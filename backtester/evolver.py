"""Genetic algorithm evolver that optimizes strategy parameters.

Takes strategies and their tunable parameter ranges, creates a population
of parameter variations, backtests each, selects the fittest (lowest drawdown,
highest win rate), breeds new generations via crossover and mutation, and
converges on optimal parameter sets.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtester import BacktestResult, Backtester
from .strategies.base import BaseStrategy
from .strategies.implementations import STRATEGY_REGISTRY


@dataclass
class EvolverConfig:
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.6
    elite_count: int = 2
    tournament_size: int = 3
    fitness_weights: dict = field(default_factory=lambda: {
        "win_rate": 0.30,
        "drawdown": 0.30,
        "sharpe": 0.20,
        "profit_factor": 0.10,
        "return": 0.10,
    })
    min_trades: int = 3
    seed: int | None = None


@dataclass
class Individual:
    """A single member of the population: a strategy class + parameter set."""
    strategy_key: str
    strategy_class: type[BaseStrategy]
    params: dict[str, float | int]
    fitness: float = 0.0
    result: BacktestResult | None = None

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"Individual({self.strategy_key}, {param_str}, fitness={self.fitness:.4f})"


@dataclass
class EvolutionResult:
    """Outcome of evolving a single strategy."""
    strategy_name: str
    original_params: dict[str, float | int]
    evolved_params: dict[str, float | int]
    original_result: BacktestResult | None
    evolved_result: BacktestResult | None
    generations_run: int
    fitness_history: list[float] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        if self.original_result is None or self.evolved_result is None:
            return False
        orig_score = _compute_fitness(self.original_result)
        evol_score = _compute_fitness(self.evolved_result)
        return evol_score > orig_score


def _compute_fitness(
    result: BacktestResult,
    weights: dict | None = None,
    min_trades: int = 3,
) -> float:
    """Score a backtest result for evolutionary selection.

    Higher is better. Penalizes strategies with too few trades.
    """
    if result.total_trades < min_trades:
        return 0.0

    w = weights or {
        "win_rate": 0.30,
        "drawdown": 0.30,
        "sharpe": 0.20,
        "profit_factor": 0.10,
        "return": 0.10,
    }

    wr_score = min(result.win_rate / 100.0, 1.0)
    dd_score = max(0.0, 1.0 - result.max_drawdown_pct / 50.0)
    sharpe_score = max(0.0, min((result.sharpe_ratio + 1.0) / 4.0, 1.0))

    pf = result.profit_factor if result.profit_factor < float("inf") else 5.0
    pf_score = min(pf / 3.0, 1.0)

    ret_score = max(0.0, min((result.total_return_pct + 50.0) / 150.0, 1.0))

    return (
        w["win_rate"] * wr_score
        + w["drawdown"] * dd_score
        + w["sharpe"] * sharpe_score
        + w["profit_factor"] * pf_score
        + w["return"] * ret_score
    )


def _snap_to_step(value: float, lo: float, hi: float, step: float) -> float | int:
    """Clamp value to [lo, hi] and snap to the nearest valid step."""
    value = max(lo, min(hi, value))
    steps_from_lo = round((value - lo) / step)
    snapped = lo + steps_from_lo * step
    snapped = max(lo, min(hi, snapped))
    if isinstance(lo, int) and isinstance(step, int):
        return int(round(snapped))
    if step == int(step) and lo == int(lo):
        return int(round(snapped))
    return round(snapped, 6)


class StrategyEvolver:
    """Genetic algorithm that evolves strategy parameters to maximize fitness."""

    def __init__(self, config: EvolverConfig | None = None):
        self.config = config or EvolverConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

    def evolve_strategy(
        self,
        strategy_key: str,
        data: dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,
        timeframe: str = "1d",
    ) -> EvolutionResult | None:
        """Evolve a single strategy's parameters against the given data."""
        if strategy_key not in STRATEGY_REGISTRY:
            return None

        strategy_class = STRATEGY_REGISTRY[strategy_key]
        ranges = strategy_class.param_ranges()
        if not ranges:
            return None

        original = strategy_class(initial_capital=initial_capital, commission_pct=commission_pct)
        original_params = dict(original.config.params)

        bt = Backtester(initial_capital=initial_capital, commission_pct=commission_pct, timeframe=timeframe)
        original_result = self._evaluate_across_symbols(bt, strategy_class, original_params, data)

        population = self._init_population(strategy_key, strategy_class, ranges)
        fitness_history: list[float] = []

        for gen in range(self.config.generations):
            for ind in population:
                if ind.result is None:
                    ind.result = self._evaluate_across_symbols(
                        bt, strategy_class, ind.params, data,
                    )
                    ind.fitness = _compute_fitness(
                        ind.result, self.config.fitness_weights, self.config.min_trades,
                    )

            population.sort(key=lambda x: x.fitness, reverse=True)
            best_fitness = population[0].fitness
            fitness_history.append(best_fitness)

            if gen == self.config.generations - 1:
                break

            next_gen: list[Individual] = []

            for ind in population[: self.config.elite_count]:
                elite = Individual(
                    strategy_key=ind.strategy_key,
                    strategy_class=ind.strategy_class,
                    params=dict(ind.params),
                    fitness=ind.fitness,
                    result=ind.result,
                )
                next_gen.append(elite)

            while len(next_gen) < self.config.population_size:
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)

                if random.random() < self.config.crossover_rate:
                    child_params = self._crossover(parent_a.params, parent_b.params)
                else:
                    child_params = dict(parent_a.params)

                if random.random() < self.config.mutation_rate:
                    child_params = self._mutate(child_params, ranges)

                self._enforce_constraints(strategy_key, child_params, ranges)

                next_gen.append(Individual(
                    strategy_key=strategy_key,
                    strategy_class=strategy_class,
                    params=child_params,
                ))

            population = next_gen

        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]

        return EvolutionResult(
            strategy_name=strategy_key,
            original_params=original_params,
            evolved_params=best.params,
            original_result=original_result,
            evolved_result=best.result,
            generations_run=self.config.generations,
            fitness_history=fitness_history,
        )

    def evolve_all(
        self,
        data: dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,
        timeframe: str = "1d",
        on_progress: callable | None = None,
    ) -> list[EvolutionResult]:
        """Evolve all registered strategies."""
        results: list[EvolutionResult] = []
        keys = [k for k, cls in STRATEGY_REGISTRY.items() if cls.param_ranges()]

        for i, key in enumerate(keys):
            if on_progress:
                on_progress(key, i + 1, len(keys))
            evo_result = self.evolve_strategy(key, data, initial_capital, commission_pct, timeframe)
            if evo_result is not None:
                results.append(evo_result)

        results.sort(
            key=lambda r: _compute_fitness(r.evolved_result) if r.evolved_result else 0.0,
            reverse=True,
        )
        return results

    def _init_population(
        self,
        strategy_key: str,
        strategy_class: type[BaseStrategy],
        ranges: dict[str, tuple],
    ) -> list[Individual]:
        """Create the initial population with random parameter sets."""
        population: list[Individual] = []

        default_instance = strategy_class()
        population.append(Individual(
            strategy_key=strategy_key,
            strategy_class=strategy_class,
            params=dict(default_instance.config.params),
        ))

        for _ in range(self.config.population_size - 1):
            params = {}
            for name, (lo, hi, step) in ranges.items():
                num_steps = int((hi - lo) / step)
                chosen_step = random.randint(0, num_steps)
                value = lo + chosen_step * step
                params[name] = _snap_to_step(value, lo, hi, step)
            self._enforce_constraints(strategy_key, params, ranges)
            population.append(Individual(
                strategy_key=strategy_key,
                strategy_class=strategy_class,
                params=params,
            ))

        return population

    def _evaluate_across_symbols(
        self,
        bt: Backtester,
        strategy_class: type[BaseStrategy],
        params: dict,
        data: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Backtest a strategy variant across all symbols and aggregate results."""
        all_trades = []
        total_return = 0.0
        total_symbols = 0
        combined_equity = []

        for symbol, df in data.items():
            try:
                strategy = strategy_class(**params)
                result = bt.run(strategy, df, symbol)
            except Exception:
                continue

            all_trades.extend(result.trades)
            total_return += result.total_return_pct
            total_symbols += 1
            if result.equity_curve is not None:
                combined_equity.append(result.equity_curve)

        if total_symbols == 0:
            return BacktestResult(strategy_name=strategy_class.__name__, symbol="ALL", timeframe="1d")

        agg = BacktestResult(
            strategy_name=strategy_class.__name__,
            symbol="ALL",
            timeframe="1d",
            trades=all_trades,
            initial_capital=bt.initial_capital,
        )

        closed = [t for t in all_trades if t.is_closed]
        agg.total_trades = len(closed)

        if closed:
            winners = [t for t in closed if t.pnl_pct > 0]
            losers = [t for t in closed if t.pnl_pct <= 0]
            agg.winning_trades = len(winners)
            agg.losing_trades = len(losers)
            agg.win_rate = len(winners) / len(closed) * 100

            if winners:
                agg.avg_win_pct = np.mean([t.pnl_pct for t in winners]) * 100
            if losers:
                agg.avg_loss_pct = np.mean([t.pnl_pct for t in losers]) * 100

            total_wins = sum(t.pnl_pct for t in winners) if winners else 0
            total_losses = abs(sum(t.pnl_pct for t in losers)) if losers else 0
            agg.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
            agg.expectancy = np.mean([t.pnl_pct for t in closed]) * 100

        agg.total_return_pct = total_return / total_symbols

        if combined_equity:
            merged = pd.concat(combined_equity)
            merged = merged.sort_index()
            peak = merged.expanding().max()
            drawdown = (merged - peak) / peak * 100
            agg.max_drawdown_pct = abs(drawdown.min()) if len(drawdown) > 0 else 0

            returns = merged.pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                agg.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
                downside = returns[returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    agg.sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(252)

        if agg.max_drawdown_pct > 0:
            agg.calmar_ratio = agg.total_return_pct / agg.max_drawdown_pct

        return agg

    def _tournament_select(self, population: list[Individual]) -> Individual:
        """Select an individual via tournament selection."""
        contestants = random.sample(
            population, min(self.config.tournament_size, len(population))
        )
        return max(contestants, key=lambda x: x.fitness)

    def _crossover(self, params_a: dict, params_b: dict) -> dict:
        """Uniform crossover: each gene randomly chosen from either parent."""
        child = {}
        for key in params_a:
            child[key] = params_a[key] if random.random() < 0.5 else params_b[key]
        return child

    def _mutate(self, params: dict, ranges: dict[str, tuple]) -> dict:
        """Mutate one or more parameters by a small random offset."""
        mutated = dict(params)
        keys_to_mutate = random.sample(
            list(ranges.keys()),
            k=random.randint(1, max(1, len(ranges) // 2)),
        )
        for key in keys_to_mutate:
            lo, hi, step = ranges[key]
            current = mutated.get(key, lo)
            spread = (hi - lo) * 0.3
            offset = random.gauss(0, spread)
            new_val = current + offset
            mutated[key] = _snap_to_step(new_val, lo, hi, step)
        return mutated

    @staticmethod
    def _enforce_constraints(strategy_key: str, params: dict, ranges: dict[str, tuple]) -> None:
        """Fix parameter combinations that are logically invalid."""
        if strategy_key == "ema_crossover":
            if params.get("fast_period", 0) >= params.get("slow_period", 999):
                lo_f, hi_f, step_f = ranges["fast_period"]
                params["fast_period"] = _snap_to_step(
                    params["slow_period"] - step_f, lo_f, hi_f, step_f,
                )
        elif strategy_key == "macd_histogram":
            if params.get("fast", 0) >= params.get("slow", 999):
                lo_f, hi_f, step_f = ranges["fast"]
                params["fast"] = _snap_to_step(
                    params["slow"] - step_f, lo_f, hi_f, step_f,
                )
        elif strategy_key == "triple_ema":
            if params.get("fast", 0) >= params.get("medium", 999):
                lo_f, hi_f, step_f = ranges["fast"]
                params["fast"] = _snap_to_step(
                    params["medium"] - step_f, lo_f, hi_f, step_f,
                )
            if params.get("medium", 0) >= params.get("slow", 999):
                lo_m, hi_m, step_m = ranges["medium"]
                params["medium"] = _snap_to_step(
                    params["slow"] - step_m, lo_m, hi_m, step_m,
                )
        elif strategy_key == "ichimoku_breakout":
            if params.get("tenkan", 0) >= params.get("kijun", 999):
                lo_t, hi_t, step_t = ranges["tenkan"]
                params["tenkan"] = _snap_to_step(
                    params["kijun"] - step_t, lo_t, hi_t, step_t,
                )
            if params.get("kijun", 0) >= params.get("senkou_b", 999):
                lo_k, hi_k, step_k = ranges["kijun"]
                params["kijun"] = _snap_to_step(
                    params["senkou_b"] - step_k, lo_k, hi_k, step_k,
                )
        elif strategy_key == "parabolic_sar":
            if params.get("step", 0) >= params.get("max_step", 999):
                lo_s, hi_s, step_s = ranges["step"]
                params["step"] = _snap_to_step(
                    params["max_step"] - step_s, lo_s, hi_s, step_s,
                )
        elif strategy_key == "rsi_mean_reversion":
            if params.get("oversold", 0) >= params.get("overbought", 999):
                lo_o, hi_o, step_o = ranges["oversold"]
                params["oversold"] = _snap_to_step(
                    params["overbought"] - step_o * 5, lo_o, hi_o, step_o,
                )
