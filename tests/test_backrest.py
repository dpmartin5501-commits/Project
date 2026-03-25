"""Tests for the backrest crypto strategy search and backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from backrest.backtester import BacktestResult, Backtester
from backrest.ranker import FilterConfig, RankCriteria, StrategyRanker
from backrest.strategies.base import BaseStrategy, Signal, StrategyConfig, Trade
from backrest.strategies.implementations import (
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
    WilliamsRReversal,
    get_all_strategies,
)
from backrest.strategy_search import StrategyResult, StrategySearcher


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
        assert len(strategies) == 11
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
