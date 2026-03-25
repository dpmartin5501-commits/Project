# Backtester - Crypto Trading Strategy Search & Backtesting Engine

Backtester searches the internet for cryptocurrency trading strategies focused on **low drawdown** and **high win rates**, then backtests them against real historical market data to validate their performance.

## Features

- **Internet Strategy Search** - Scrapes the web for crypto trading strategies using DuckDuckGo and curated finance sites
- **12 Built-in Strategies** - Pre-configured implementations of popular technical analysis strategies
- **Historical Data Fetching** - Downloads OHLCV candlestick data from any ccxt-supported exchange (Binance US, Kraken, etc.)
- **Full Backtesting Engine** - Event-driven backtester with commission, slippage, stop-loss, and take-profit support
- **Strategy Evolution** - Genetic algorithm that optimizes strategy parameters (population, crossover, mutation) to find the best variation of each strategy
- **Comprehensive Metrics** - Win rate, max drawdown, Sharpe ratio, Sortino ratio, Calmar ratio, profit factor, expectancy
- **Smart Ranking** - Composite scoring that prioritizes low drawdown and high win rates
- **Rich Terminal Output** - Color-coded tables and detailed strategy panels

## Built-in Strategies

| Strategy | Type | Indicators |
|----------|------|-----------|
| RSI Mean Reversion | Mean Reversion | RSI(14) |
| EMA Crossover (9/21) | Trend Following | EMA(9), EMA(21) |
| Bollinger Band Squeeze | Volatility Breakout | BB(20), ATR |
| MACD Histogram Reversal | Trend Following | MACD, SMA(50) |
| Stochastic RSI Bounce | Mean Reversion | StochRSI |
| Triple EMA Trend | Trend Following | EMA(8/21/55) |
| ADX + RSI Trend | Trend Following | ADX(14), RSI(14) |
| Dual Momentum | Momentum | ROC(12), SMA(200) |
| Williams %R Reversal | Mean Reversion | Williams %R(14) |
| Parabolic SAR Trend | Trend Following | PSAR |
| Ichimoku Cloud Breakout | Trend Following | Ichimoku |
| VWAP Reversion | Mean Reversion | VWAP, ATR |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python -m backtester
```

### CLI Options

```bash
python -m backtester \
  --exchange binanceus \
  --symbols "BTC/USD,ETH/USD,SOL/USD" \
  --timeframe 1d \
  --days 365 \
  --capital 10000 \
  --min-win-rate 50 \
  --max-drawdown 25 \
  --top 5
```

### Evolve Strategy Parameters

```bash
python -m backtester --evolve --evolve-generations 15 --evolve-population 30
```

This runs a genetic algorithm on each strategy, testing hundreds of parameter combinations to find the version with the lowest drawdown and highest win rate.

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--exchange, -e` | `binanceus` | Exchange to fetch data from |
| `--symbols, -s` | `BTC/USD,ETH/USD,SOL/USD` | Comma-separated trading pairs |
| `--timeframe, -t` | `1d` | Candlestick timeframe |
| `--days, -d` | `365` | Days of historical data |
| `--capital, -c` | `10000` | Starting capital ($) |
| `--commission` | `0.001` | Commission rate (0.1%) |
| `--min-win-rate` | `50` | Min win rate filter (%) |
| `--max-drawdown` | `25` | Max drawdown filter (%) |
| `--min-trades` | `5` | Min trades required |
| `--top, -n` | `5` | Top N strategies to show |
| `--no-search` | `false` | Skip internet search |
| `--quiet, -q` | `false` | Less verbose output |
| `--evolve` | `false` | Evolve strategy parameters via genetic algorithm |
| `--evolve-population` | `20` | Population size per generation |
| `--evolve-generations` | `10` | Number of generations |
| `--evolve-mutation-rate` | `0.3` | Mutation probability (0.0-1.0) |

### Python API

```python
from backtester.engine import BacktesterEngine, EngineConfig

config = EngineConfig(
    symbols=["BTC/USD", "ETH/USD"],
    since_days=180,
    min_win_rate=55,
    max_drawdown=20,
)
engine = BacktesterEngine(config)
top_strategies = engine.run()

for result in top_strategies:
    print(f"{result.strategy_name}: WR={result.win_rate:.1f}%, DD={result.max_drawdown_pct:.1f}%")
```

## Pipeline

1. **Search** - Queries DuckDuckGo and curated sites for crypto trading strategies
2. **Fetch** - Downloads historical OHLCV data via ccxt
3. **Backtest** - Runs all strategies against the data with realistic execution simulation
4. **Filter** - Removes strategies that don't meet win rate / drawdown thresholds
5. **Rank** - Scores remaining strategies using a composite metric
6. **Evolve** (optional) - Genetic algorithm optimizes each strategy's parameters across generations
7. **Report** - Displays results in rich terminal tables

## Metrics

- **Win Rate** - Percentage of profitable trades
- **Max Drawdown** - Largest peak-to-trough decline in equity
- **Sharpe Ratio** - Risk-adjusted return (annualized)
- **Sortino Ratio** - Downside risk-adjusted return
- **Calmar Ratio** - Annualized return / max drawdown
- **Profit Factor** - Gross wins / gross losses
- **Expectancy** - Average PnL per trade

## Project Structure

```
backtester/
  __init__.py           # Package metadata
  __main__.py           # python -m backtester entry point
  cli.py                # Click CLI interface
  engine.py             # Main orchestration engine
  data_fetcher.py       # OHLCV data download via ccxt
  strategy_search.py    # Internet strategy scraper
  backtester.py         # Backtesting engine + metrics
  evolver.py            # Genetic algorithm strategy evolution
  ranker.py             # Strategy filtering and ranking
  reporter.py           # Rich terminal output
  strategies/
    __init__.py
    base.py             # Base strategy class and trade model
    implementations.py  # 12 concrete strategy implementations
```

## Requirements

- Python 3.10+
- ccxt, pandas, numpy, ta, requests, beautifulsoup4, rich, click
