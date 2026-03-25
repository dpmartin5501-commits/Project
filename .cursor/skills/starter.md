# Starter Skill — Backtester

> Use this skill whenever you need to set up, run, or test the Backtester codebase
> for the first time or after a fresh environment reset.

---

## 1. Environment Setup

### 1.1 Install dependencies

```bash
pip install -r requirements.txt
```

No virtual-environment wrapper is required in Cloud agent VMs — install directly.

### 1.2 Verify installation

```bash
python3 -c "import backtester; print(backtester.__version__)"
```

Expected output: `1.0.0` (or the current version string).

### 1.3 Key environment notes

- **Use `python3`**, not `python`. The VM may not have a `python` symlink.
- There is **no database, Docker, web server, or `.env` file**. Configuration is passed entirely through CLI flags or `EngineConfig` dataclass fields.
- There are **no feature flags or authentication**. Nothing to log into.
- There is **no linter** configured (`pyproject.toml`, `.flake8`, etc. do not exist).

---

## 2. Running the Application

### 2.1 Recommended dev invocation

```bash
python3 -m backtester --no-search --days 30
```

| Flag | Why |
|------|-----|
| `--no-search` | Skips DuckDuckGo web scraping (flaky, slow, network-dependent). Uses the 11 built-in strategies instead. |
| `--days 30` | Keeps data-fetch time short. With only 30 days of 1d candles, most strategies won't pass the default filters (≥50% win rate, ≤25% drawdown, ≥5 trades). This is expected, not a bug. |

### 2.2 Full CLI option reference

```bash
python3 -m backtester --help
```

Commonly used flags:

| Option | Default | Notes |
|--------|---------|-------|
| `--exchange, -e` | `binanceus` | Any ccxt-supported exchange ID |
| `--symbols, -s` | `BTC/USD,ETH/USD,SOL/USD` | Comma-separated pairs |
| `--timeframe, -t` | `1d` | `1m`, `5m`, `1h`, `4h`, `1d` |
| `--days, -d` | `365` | Number of history days |
| `--capital, -c` | `10000` | Starting capital |
| `--no-search` | off | Use only built-in strategies |
| `--quiet, -q` | off | Suppress verbose output |
| `--evolve` | off | Run genetic algorithm on strategies |
| `--evolve-generations` | `10` | GA generations |
| `--evolve-population` | `20` | GA population per generation |

### 2.3 Using the Python API directly

```python
from backtester.engine import BacktesterEngine, EngineConfig

config = EngineConfig(
    search_internet=False,
    since_days=30,
    symbols=["BTC/USD"],
)
engine = BacktesterEngine(config)
results = engine.run()
```

This is useful when writing integration-level tests or one-off scripts.

---

## 3. Running Tests

### 3.1 Full test suite

```bash
python3 -m pytest tests/ -v
```

- **44 tests**, all in `tests/test_backtester.py`.
- Every test uses **synthetic data** (seeded `numpy.random` + generated OHLCV DataFrame). No network, no API keys, no exchange access required.
- Tests should complete in under 60 seconds.

### 3.2 Running a subset of tests

```bash
# Single test class
python3 -m pytest tests/test_backtester.py::TestStrategies -v

# Single test method
python3 -m pytest tests/test_backtester.py::TestEvolver::test_evolver_single_strategy -v

# Keyword filter
python3 -m pytest tests/ -v -k "ranker"
```

### 3.3 Expected warnings (safe to ignore)

Pandas / NumPy may emit `FutureWarning` about bitwise inversion on `bool` Series. These come from the upstream `ta` library, not from project code. They do not indicate a problem.

---

## 4. Codebase Areas & Testing Workflows

### 4.1 Strategies (`backtester/strategies/`)

| File | Purpose |
|------|---------|
| `base.py` | `BaseStrategy` ABC, `Signal` enum, `Trade` dataclass, `StrategyConfig` |
| `implementations.py` | 11 concrete strategies + `STRATEGY_REGISTRY` dict + `get_all_strategies()` |

**What to test after changes:**

```bash
python3 -m pytest tests/test_backtester.py::TestStrategies -v
python3 -m pytest tests/test_backtester.py::TestTrade -v
```

If you add a new strategy class, also check that `STRATEGY_REGISTRY` and `get_all_strategies()` include it, and update the `test_all_strategies_instantiate` assertion count.

### 4.2 Backtesting Engine (`backtester/backtester.py`)

Core event-loop backtester. Processes bar-by-bar signals, manages entries/exits with commission, and computes all performance metrics (Sharpe, Sortino, Calmar, profit factor, drawdown, etc.).

**What to test after changes:**

```bash
python3 -m pytest tests/test_backtester.py::TestBacktester -v
```

### 4.3 Ranking & Filtering (`backtester/ranker.py`)

`StrategyRanker` applies `FilterConfig` (min win rate, max drawdown, min trades) and sorts results by composite score or a single `RankCriteria`.

**What to test after changes:**

```bash
python3 -m pytest tests/test_backtester.py::TestRanker -v
```

### 4.4 Genetic Algorithm / Evolver (`backtester/evolver.py`)

`StrategyEvolver` runs a GA over strategy parameter ranges. Each strategy class exposes `param_ranges()` returning `{name: (lo, hi, step)}`.

**What to test after changes:**

```bash
python3 -m pytest tests/test_backtester.py::TestEvolver -v
```

Evolver tests are the slowest in the suite (~20-30s) because they actually run multi-generation evolution on synthetic data.

### 4.5 Strategy Search (`backtester/strategy_search.py`)

`StrategySearcher` scrapes DuckDuckGo + curated URLs for strategy descriptions. Locally testable helpers (indicator extraction, tag extraction, relevance check) do not hit the network.

**What to test after changes:**

```bash
python3 -m pytest tests/test_backtester.py::TestStrategySearcher -v
```

Network-dependent search (`search_strategies()`) is **not covered by tests** and should be avoided in automated testing. Use `--no-search` or `search_internet=False` in `EngineConfig`.

### 4.6 Data Fetcher (`backtester/data_fetcher.py`)

Downloads OHLCV via `ccxt`. No dedicated test coverage — it's exercised through the full CLI run. When testing locally, rely on `--days 30` to keep downloads fast.

### 4.7 Reporter (`backtester/reporter.py`)

Rich terminal output (tables, panels, summary). No dedicated test coverage. Verify visually by running the CLI.

### 4.8 CLI & Engine Orchestration (`backtester/cli.py`, `backtester/engine.py`)

`cli.py` is a thin Click wrapper around `BacktesterEngine`. `engine.py` orchestrates the full pipeline (search → fetch → backtest → filter/rank → evolve → report).

**Quick smoke test:**

```bash
python3 -m backtester --no-search --days 30 --quiet
```

---

## 5. Common Workflows

### 5.1 Verify everything works end-to-end

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 -m backtester --no-search --days 30 --quiet
```

All three commands should succeed.

### 5.2 Add a new strategy

1. Create the class in `backtester/strategies/implementations.py`, subclassing `BaseStrategy`.
2. Add it to `STRATEGY_REGISTRY`.
3. Update `test_all_strategies_instantiate` count in `tests/test_backtester.py`.
4. Add a `test_<name>_generates_signals` test.
5. Run: `python3 -m pytest tests/test_backtester.py::TestStrategies -v`

### 5.3 Modify backtesting logic

1. Edit `backtester/backtester.py`.
2. Run: `python3 -m pytest tests/test_backtester.py::TestBacktester -v`
3. Smoke-test with CLI: `python3 -m backtester --no-search --days 30`

### 5.4 Modify ranking / filtering

1. Edit `backtester/ranker.py`.
2. Run: `python3 -m pytest tests/test_backtester.py::TestRanker -v`

### 5.5 Modify the evolver

1. Edit `backtester/evolver.py`.
2. Run: `python3 -m pytest tests/test_backtester.py::TestEvolver -v`

---

## 6. Keeping This Skill Up to Date

Update this file when any of the following happens:

- **New dependencies** are added to `requirements.txt` — add install notes if they need special setup.
- **New codebase areas** appear (new modules, new test files, new CLI commands) — add a subsection under §4 with the matching test command.
- **New testing tricks** are discovered (e.g. a faster way to run the CLI, a useful pytest marker, a mock pattern) — add them to §5 (Common Workflows) or to the relevant §4 subsection.
- **Environment gotchas** are hit (e.g. a package version conflict, a missing system library) — add to §1.3.
- **CI/CD is added** — document the pipeline and how to reproduce CI checks locally.

Keep entries concise. Each new section should answer: *what changed, how to test it, and what to watch out for.*
