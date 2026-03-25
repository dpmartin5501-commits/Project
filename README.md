# Backrest - Crypto Trading Strategy Search & Backtesting Engine

Backrest searches the internet for cryptocurrency trading strategies focused on **low drawdown** and **high win rates**, then backtests them against real historical market data to validate their performance.

## Features

- **Internet Strategy Search** - Scrapes the web for crypto trading strategies using DuckDuckGo and curated finance sites
- **12 Built-in Strategies** - Pre-configured implementations of popular technical analysis strategies
- **Historical Data Fetching** - Downloads OHLCV candlestick data from any ccxt-supported exchange (Binance, Coinbase, etc.)
- **Full Backtesting Engine** - Event-driven backtester with commission, slippage, stop-loss, and take-profit support
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

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python -m backrest
```

### CLI Options

```bash
python -m backrest \
  --exchange binance \
  --symbols "BTC/USDT,ETH/USDT,SOL/USDT" \
  --timeframe 1d \
  --days 365 \
  --capital 10000 \
  --min-win-rate 50 \
  --max-drawdown 25 \
  --top 5
```

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--exchange, -e` | `binance` | Exchange to fetch data from |
| `--symbols, -s` | `BTC/USDT,ETH/USDT,SOL/USDT` | Comma-separated trading pairs |
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

### Python API

```python
from backrest.engine import BackrestEngine, EngineConfig

config = EngineConfig(
    symbols=["BTC/USDT", "ETH/USDT"],
    since_days=180,
    min_win_rate=55,
    max_drawdown=20,
)
engine = BackrestEngine(config)
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
6. **Report** - Displays results in rich terminal tables

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
backrest/
  __init__.py           # Package metadata
  __main__.py           # python -m backrest entry point
  cli.py                # Click CLI interface
  engine.py             # Main orchestration engine
  data_fetcher.py       # OHLCV data download via ccxt
  strategy_search.py    # Internet strategy scraper
  backtester.py         # Backtesting engine + metrics
  ranker.py             # Strategy filtering and ranking
  reporter.py           # Rich terminal output
  strategies/
    __init__.py
    base.py             # Base strategy class and trade model
    implementations.py  # 11 concrete strategy implementations
```

## Requirements

- Python 3.10+
- ccxt, pandas, numpy, ta, requests, beautifulsoup4, rich, click
