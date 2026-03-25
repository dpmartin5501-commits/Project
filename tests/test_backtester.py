"""Tests for the backtester crypto strategy search and backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from backtester.backtester import BacktestResult, Backtester
from backtester.ranker import FilterConfig, RankCriteria, StrategyRanker
from backtester.strategies.base import BaseStrategy, Signal, StrategyConfig, Trade
from backtester.strategies.implementations import (
    ADXRSITrend,
    BollingerBandSqueeze,
    DualMomentum,
    EMACrossover,
    IchimokuBreakout,
    MACDHistogramReversal,
    ParabolicSARTrend,
    RSIMeanReversion,
    StochasticRSIBounce,
    TripleEMATrend,
    VWAPReversion,
    WilliamsRReversal,
    get_all_strategies,
)
from backtester.evolver import (
    EvolverConfig,
    EvolutionResult,
    StrategyEvolver,
    _compute_fitness,
    _snap_to_step,
)
from backtester.strategy_search import StrategyResult, StrategySearcher


@pytest.fixture
def sample_df():
    """Generate a realistic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 365
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    price = 40000.0
    prices = [price]
    for _ in range(n - 1):
        change = np.random.normal(0.001, 0.025)
        price *= 1 + change
        prices.append(price)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close": prices,
            "volume": [np.random.uniform(1e6, 1e8) for _ in range(n)],
        }
    )


class TestTrade:
    def test_trade_creation(self):
        t = Trade(
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=100.0,
            size=1000.0,
        )
        assert not t.is_closed
        assert t.pnl == 0.0

    def test_trade_close_profit(self):
        t = Trade(
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=100.0,
            size=1000.0,
        )
        t.close(pd.Timestamp("2024-01-10"), 110.0)
        assert t.is_closed
        assert t.pnl_pct == pytest.approx(0.1, rel=1e-6)
        assert t.pnl == pytest.approx(0.1 * 1000.0, rel=1e-6)

    def test_trade_close_loss(self):
        t = Trade(
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=100.0,
            size=1000.0,
        )
        t.close(pd.Timestamp("2024-01-10"), 90.0)
        assert t.is_closed
        assert t.pnl_pct == pytest.approx(-0.1, rel=1e-6)


class TestStrategies:
    def test_all_strategies_instantiate(self):
        strategies = get_all_strategies()
        assert len(strategies) == 12
        for s in strategies:
            assert isinstance(s, BaseStrategy)
            assert s.name

    def test_rsi_generates_signals(self, sample_df):
        s = RSIMeanReversion()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)
        unique_signals = set(signals.dropna().unique())
        assert unique_signals.issubset({Signal.BUY, Signal.SELL, Signal.HOLD})

    def test_ema_crossover_generates_signals(self, sample_df):
        s = EMACrossover()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_bollinger_generates_signals(self, sample_df):
        s = BollingerBandSqueeze()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_macd_generates_signals(self, sample_df):
        s = MACDHistogramReversal()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_stoch_rsi_generates_signals(self, sample_df):
        s = StochasticRSIBounce()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_triple_ema_generates_signals(self, sample_df):
        s = TripleEMATrend()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_adx_rsi_generates_signals(self, sample_df):
        s = ADXRSITrend()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_dual_momentum_generates_signals(self, sample_df):
        s = DualMomentum()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_williams_r_generates_signals(self, sample_df):
        s = WilliamsRReversal()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_parabolic_sar_generates_signals(self, sample_df):
        s = ParabolicSARTrend()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_ichimoku_generates_signals(self, sample_df):
        s = IchimokuBreakout()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)

    def test_vwap_reversion_generates_signals(self, sample_df):
        s = VWAPReversion()
        df = s.prepare_indicators(sample_df.copy())
        signals = s.generate_signals(df)
        assert len(signals) == len(df)
        unique_signals = set(signals.dropna().unique())
        assert unique_signals.issubset({Signal.BUY, Signal.SELL, Signal.HOLD})


class TestBacktester:
    def test_backtest_produces_result(self, sample_df):
        bt = Backtester()
        s = RSIMeanReversion()
        result = bt.run(s, sample_df, "BTC/USDT")
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "RSI Mean Reversion"
        assert result.symbol == "BTC/USDT"

    def test_backtest_empty_df(self):
        bt = Backtester()
        s = RSIMeanReversion()
        empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = bt.run(s, empty_df)
        assert result.total_trades == 0

    def test_backtest_metrics_reasonable(self, sample_df):
        bt = Backtester()
        s = WilliamsRReversal()
        result = bt.run(s, sample_df, "BTC/USDT")
        assert 0 <= result.win_rate <= 100
        assert result.max_drawdown_pct >= 0
        assert result.total_trades >= 0
        assert result.winning_trades + result.losing_trades == result.total_trades

    def test_backtest_capital_conservation(self, sample_df):
        bt = Backtester(initial_capital=10000)
        s = EMACrossover()
        result = bt.run(s, sample_df, "BTC/USDT")
        assert result.initial_capital == 10000
        assert result.final_capital > 0

    def test_run_multiple(self, sample_df):
        bt = Backtester()
        strategies = [RSIMeanReversion(), EMACrossover()]
        data = {"BTC/USDT": sample_df}
        results = bt.run_multiple(strategies, data)
        assert len(results) == 2

    def test_backtest_timeframe_propagation(self, sample_df):
        bt = Backtester(timeframe="4h")
        s = RSIMeanReversion()
        result = bt.run(s, sample_df, "BTC/USDT")
        assert result.timeframe == "4h"

    def test_backtest_default_timeframe(self, sample_df):
        bt = Backtester()
        s = RSIMeanReversion()
        result = bt.run(s, sample_df, "BTC/USDT")
        assert result.timeframe == "1d"

    def test_backtest_vwap_strategy(self, sample_df):
        bt = Backtester()
        s = VWAPReversion()
        result = bt.run(s, sample_df, "BTC/USDT")
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "VWAP Reversion"
        assert 0 <= result.win_rate <= 100
        assert result.max_drawdown_pct >= 0


class TestRanker:
    def _make_result(self, name, win_rate, dd, trades, pf=1.5, sharpe=0.5):
        r = BacktestResult(strategy_name=name, symbol="BTC/USDT", timeframe="1d")
        r.win_rate = win_rate
        r.max_drawdown_pct = dd
        r.total_trades = trades
        r.profit_factor = pf
        r.sharpe_ratio = sharpe
        r.total_return_pct = 10.0
        r.annualized_return_pct = 10.0
        r.calmar_ratio = 10.0 / dd if dd > 0 else 0
        return r

    def test_filter_by_win_rate(self):
        ranker = StrategyRanker(FilterConfig(min_win_rate=55, max_drawdown=100))
        results = [
            self._make_result("A", 60, 10, 10),
            self._make_result("B", 40, 10, 10),
        ]
        filtered = ranker.filter_results(results)
        assert len(filtered) == 1
        assert filtered[0].strategy_name == "A"

    def test_filter_by_drawdown(self):
        ranker = StrategyRanker(FilterConfig(min_win_rate=0, max_drawdown=20))
        results = [
            self._make_result("A", 50, 15, 10),
            self._make_result("B", 50, 25, 10),
        ]
        filtered = ranker.filter_results(results)
        assert len(filtered) == 1
        assert filtered[0].strategy_name == "A"

    def test_filter_by_min_trades(self):
        ranker = StrategyRanker(FilterConfig(min_trades=10))
        results = [
            self._make_result("A", 55, 15, 15),
            self._make_result("B", 55, 15, 3),
        ]
        filtered = ranker.filter_results(results)
        assert len(filtered) == 1

    def test_rank_by_win_rate(self):
        ranker = StrategyRanker(FilterConfig(rank_by=RankCriteria.WIN_RATE, min_win_rate=0, max_drawdown=100))
        results = [
            self._make_result("A", 50, 10, 10),
            self._make_result("B", 70, 10, 10),
        ]
        ranked = ranker.rank_results(results)
        assert ranked[0].strategy_name == "B"

    def test_rank_by_drawdown(self):
        ranker = StrategyRanker(FilterConfig(rank_by=RankCriteria.MAX_DRAWDOWN, min_win_rate=0, max_drawdown=100))
        results = [
            self._make_result("A", 50, 20, 10),
            self._make_result("B", 50, 5, 10),
        ]
        ranked = ranker.rank_results(results)
        assert ranked[0].strategy_name == "B"

    def test_composite_score(self):
        ranker = StrategyRanker()
        r1 = self._make_result("Good", 65, 10, 20, pf=2.0, sharpe=1.5)
        r2 = self._make_result("Bad", 30, 40, 20, pf=0.5, sharpe=-0.5)
        s1 = ranker._composite_score(r1)
        s2 = ranker._composite_score(r2)
        assert s1 > s2


class TestStrategySearcher:
    def test_builtin_strategies(self):
        searcher = StrategySearcher()
        builtin = searcher._get_builtin_strategies()
        assert len(builtin) == 12
        for s in builtin:
            assert isinstance(s, StrategyResult)
            assert s.name
            assert s.description
            assert s.source_url

    def test_indicator_extraction(self):
        text = "Using RSI and MACD with Bollinger Bands for entry signals"
        indicators = StrategySearcher._extract_indicators(text)
        assert "RSI" in indicators
        assert "MACD" in indicators
        assert "Bollinger Bands" in indicators

    def test_percentage_extraction(self):
        text = "This strategy has a 65% win rate with only 12% max drawdown"
        wr = StrategySearcher._extract_percentage(text, r"(\d{1,3}(?:\.\d+)?)\s*%?\s*win\s*rate")
        assert wr == 65.0

    def test_tag_extraction(self):
        text = "conservative low risk mean reversion strategy using oversold conditions"
        tags = StrategySearcher._extract_tags(text)
        assert "mean-reversion" in tags
        assert "conservative" in tags

    def test_relevance_check(self):
        assert StrategySearcher._is_strategy_relevant("Trading Strategy Guide", "entry and exit signals")
        assert not StrategySearcher._is_strategy_relevant("Cat Videos", "funny compilation")


class TestEvolver:
    def test_snap_to_step_int(self):
        assert _snap_to_step(17, 5, 30, 1) == 17
        assert _snap_to_step(3, 5, 30, 1) == 5
        assert _snap_to_step(35, 5, 30, 1) == 30

    def test_snap_to_step_float(self):
        result = _snap_to_step(1.37, 1.0, 3.5, 0.25)
        assert result == 1.25 or result == 1.5

    def test_compute_fitness_few_trades(self):
        r = BacktestResult(strategy_name="test", symbol="X", timeframe="1d")
        r.total_trades = 1
        assert _compute_fitness(r, min_trades=3) == 0.0

    def test_compute_fitness_good_result(self):
        r = BacktestResult(strategy_name="test", symbol="X", timeframe="1d")
        r.total_trades = 20
        r.win_rate = 65.0
        r.max_drawdown_pct = 10.0
        r.sharpe_ratio = 1.5
        r.profit_factor = 2.0
        r.total_return_pct = 30.0
        score = _compute_fitness(r)
        assert score > 0.5

    def test_compute_fitness_bad_result(self):
        r = BacktestResult(strategy_name="test", symbol="X", timeframe="1d")
        r.total_trades = 20
        r.win_rate = 20.0
        r.max_drawdown_pct = 45.0
        r.sharpe_ratio = -1.0
        r.profit_factor = 0.3
        r.total_return_pct = -30.0
        score = _compute_fitness(r)
        assert score < 0.3

    def test_all_strategies_have_param_ranges(self):
        from backtester.strategies.implementations import STRATEGY_REGISTRY
        for key, cls in STRATEGY_REGISTRY.items():
            ranges = cls.param_ranges()
            assert isinstance(ranges, dict), f"{key} param_ranges() must return dict"
            assert len(ranges) > 0, f"{key} should have at least one tunable param"
            for name, (lo, hi, step) in ranges.items():
                assert lo < hi, f"{key}.{name}: lo ({lo}) must be < hi ({hi})"
                assert step > 0, f"{key}.{name}: step must be positive"

    def test_evolver_creates_population(self, sample_df):
        from backtester.strategies.implementations import STRATEGY_REGISTRY
        config = EvolverConfig(population_size=5, generations=1, seed=42)
        evolver = StrategyEvolver(config)

        cls = STRATEGY_REGISTRY["rsi_mean_reversion"]
        pop = evolver._init_population("rsi_mean_reversion", cls, cls.param_ranges())
        assert len(pop) == 5
        default_params = RSIMeanReversion().config.params
        assert pop[0].params == default_params

    def test_evolver_single_strategy(self, sample_df):
        config = EvolverConfig(population_size=6, generations=3, seed=42)
        evolver = StrategyEvolver(config)

        result = evolver.evolve_strategy(
            "rsi_mean_reversion",
            {"BTC/USD": sample_df},
        )
        assert result is not None
        assert isinstance(result, EvolutionResult)
        assert result.strategy_name == "rsi_mean_reversion"
        assert result.generations_run == 3
        assert len(result.fitness_history) == 3
        assert result.evolved_params is not None
        assert result.evolved_result is not None
        assert result.evolved_result.total_trades >= 0

    def test_evolver_respects_constraints(self):
        config = EvolverConfig(population_size=10, generations=1, seed=42)
        evolver = StrategyEvolver(config)
        from backtester.strategies.implementations import STRATEGY_REGISTRY
        cls = STRATEGY_REGISTRY["ema_crossover"]
        pop = evolver._init_population("ema_crossover", cls, cls.param_ranges())
        for ind in pop:
            assert ind.params["fast_period"] < ind.params["slow_period"], (
                f"fast_period ({ind.params['fast_period']}) must be < "
                f"slow_period ({ind.params['slow_period']})"
            )

    def test_evolver_unknown_strategy(self):
        evolver = StrategyEvolver()
        result = evolver.evolve_strategy("nonexistent", {})
        assert result is None

    def test_evolver_evolve_all(self, sample_df):
        config = EvolverConfig(population_size=4, generations=2, seed=42)
        evolver = StrategyEvolver(config)

        results = evolver.evolve_all({"BTC/USD": sample_df})
        assert len(results) > 0
        for r in results:
            assert isinstance(r, EvolutionResult)
            assert r.evolved_params is not None

    def test_crossover_produces_valid_params(self):
        evolver = StrategyEvolver(EvolverConfig(seed=42))
        a = {"rsi_period": 14, "oversold": 30, "overbought": 70}
        b = {"rsi_period": 20, "oversold": 25, "overbought": 75}
        child = evolver._crossover(a, b)
        for key in a:
            assert child[key] in (a[key], b[key])

    def test_mutation_stays_in_range(self):
        evolver = StrategyEvolver(EvolverConfig(seed=42))
        ranges = RSIMeanReversion.param_ranges()
        params = {"rsi_period": 14, "oversold": 30, "overbought": 70}
        for _ in range(50):
            mutated = evolver._mutate(dict(params), ranges)
            for key, (lo, hi, _) in ranges.items():
                assert lo <= mutated[key] <= hi, (
                    f"{key}={mutated[key]} outside [{lo}, {hi}]"
                )
