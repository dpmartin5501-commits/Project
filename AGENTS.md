# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Backtester is a Python CLI application for searching and backtesting cryptocurrency trading strategies. It is a single-process CLI tool with no web server, no database, and no Docker.

### Running the application

**CLI mode:**
```bash
python3 -m backtester --no-search --days 30
```

The `--no-search` flag skips internet strategy search (DuckDuckGo scraping) and uses only built-in strategies. This is the recommended flag for development/testing since the web scraping can be flaky.

**Web GUI mode:**
```bash
python3 -m backtester --gui --port 5000
```

Launches a pixel-art RPG themed web interface at `http://localhost:5000`. The GUI supports all engine features (exchange selection, filters, genetic evolution) with async job processing.

See `README.md` for full CLI options.

### Running tests

```bash
python3 -m pytest tests/ -v
```

All 44 tests use synthetic data and require no external services or API keys.

### Linting

No linter is configured in this project (no pyproject.toml, .flake8, setup.cfg, or pre-commit hooks).

### Gotchas

- Use `python3` not `python` — the environment may not have `python` symlinked.
- The pandas/numpy deprecation warnings in tests (about bitwise inversion on bool) are harmless and come from upstream library code, not the project itself.
- The CLI fetches live OHLCV data from exchanges via `ccxt`. With only 30 days of data, most strategies won't generate enough trades to pass the default filter criteria (50% win rate, 25% max drawdown, 5 minimum trades). This is expected behavior, not a bug.
