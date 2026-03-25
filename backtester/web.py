"""Flask web GUI for the Backtester crypto strategy search engine.

Pixel-art RPG themed interface for configuring, running, and viewing
backtest results through a browser.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass

from flask import Flask, jsonify, render_template_string, request

from .backtester import BacktestResult, Backtester
from .data_fetcher import DataFetcher, FetchConfig
from .evolver import EvolverConfig, EvolutionResult, StrategyEvolver, _compute_fitness
from .ranker import FilterConfig, StrategyRanker
from .strategies.implementations import STRATEGY_REGISTRY, get_all_strategies
from .strategy_search import StrategySearcher

app = Flask(__name__)

_jobs: dict[str, dict] = {}


def _result_to_dict(r: BacktestResult) -> dict:
    return {
        "strategy_name": r.strategy_name,
        "symbol": r.symbol,
        "timeframe": r.timeframe,
        "initial_capital": r.initial_capital,
        "final_capital": round(r.final_capital, 2),
        "total_return_pct": round(r.total_return_pct, 2),
        "annualized_return_pct": round(r.annualized_return_pct, 2),
        "max_drawdown_pct": round(r.max_drawdown_pct, 2),
        "win_rate": round(r.win_rate, 2),
        "total_trades": r.total_trades,
        "winning_trades": r.winning_trades,
        "losing_trades": r.losing_trades,
        "avg_win_pct": round(r.avg_win_pct, 2),
        "avg_loss_pct": round(r.avg_loss_pct, 2),
        "profit_factor": round(r.profit_factor, 2) if r.profit_factor < 1e6 else "Inf",
        "sharpe_ratio": round(r.sharpe_ratio, 2),
        "sortino_ratio": round(r.sortino_ratio, 2),
        "calmar_ratio": round(r.calmar_ratio, 2),
        "avg_holding_periods": round(r.avg_holding_periods, 1),
        "max_consecutive_losses": r.max_consecutive_losses,
        "expectancy": round(r.expectancy, 2),
    }


def _evo_result_to_dict(r: EvolutionResult) -> dict:
    d = {
        "strategy_name": r.strategy_name,
        "original_params": r.original_params,
        "evolved_params": r.evolved_params,
        "generations_run": r.generations_run,
        "fitness_history": [round(f, 4) for f in r.fitness_history],
        "improved": r.improved,
    }
    if r.original_result:
        d["original_result"] = _result_to_dict(r.original_result)
    if r.evolved_result:
        d["evolved_result"] = _result_to_dict(r.evolved_result)
    return d


def _run_backtest_job(job_id: str, config: dict):
    job = _jobs[job_id]
    try:
        job["status"] = "searching"
        job["log"].append("Searching for strategies...")

        searcher = StrategySearcher()
        if config.get("search_internet"):
            strategies_found = searcher.search_strategies(max_results=20)
        else:
            strategies_found = searcher._get_builtin_strategies()

        job["log"].append(f"Found {len(strategies_found)} strategies")
        job["strategies_found"] = len(strategies_found)

        job["status"] = "fetching"
        job["log"].append("Fetching historical data...")

        fetch_config = FetchConfig(
            exchange_id=config["exchange"],
            symbols=config["symbols"],
            timeframe=config["timeframe"],
            since_days=config["days"],
        )
        fetcher = DataFetcher(fetch_config)
        data = fetcher.fetch_all()

        for symbol, df in data.items():
            job["log"].append(f"  {symbol}: {len(df)} candles")

        if not data:
            job["status"] = "error"
            job["error"] = "No market data fetched"
            return

        job["status"] = "backtesting"
        job["log"].append("Running backtests...")

        bt = Backtester(
            initial_capital=config["capital"],
            commission_pct=config["commission"],
        )
        all_strategies = get_all_strategies(
            initial_capital=config["capital"],
            commission_pct=config["commission"],
        )
        results = bt.run_multiple(all_strategies, data)
        job["log"].append(f"Completed {len(results)} backtests")

        job["status"] = "ranking"
        job["log"].append("Filtering and ranking...")

        filter_config = FilterConfig(
            min_win_rate=config["min_win_rate"],
            max_drawdown=config["max_drawdown"],
            min_trades=config["min_trades"],
        )
        ranker = StrategyRanker(filter_config)
        filtered = ranker.filter_and_rank(results)

        if filtered:
            top = filtered[: config["top_n"]]
        else:
            top = ranker.get_all_ranked(results)[: config["top_n"]]

        job["all_results"] = [_result_to_dict(r) for r in results]
        job["top_results"] = [_result_to_dict(r) for r in top]
        job["filtered_count"] = len(filtered)

        evo_results_data = []
        if config.get("evolve"):
            job["status"] = "evolving"
            job["log"].append("Evolving strategy parameters...")

            evo_config = EvolverConfig(
                population_size=config.get("evolve_population", 20),
                generations=config.get("evolve_generations", 10),
                mutation_rate=config.get("evolve_mutation_rate", 0.3),
            )
            evolver = StrategyEvolver(evo_config)

            def on_progress(name, current, total):
                job["log"].append(f"  Evolving {name} ({current}/{total})")

            evo_results = evolver.evolve_all(
                data,
                initial_capital=config["capital"],
                commission_pct=config["commission"],
                on_progress=on_progress,
            )
            evo_results_data = [_evo_result_to_dict(r) for r in evo_results]
            improved = sum(1 for r in evo_results if r.improved)
            job["log"].append(f"Evolution complete: {improved}/{len(evo_results)} improved")

        job["evolution_results"] = evo_results_data
        job["status"] = "complete"
        job["log"].append("Pipeline complete!")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["log"].append(f"Error: {e}")


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/strategies")
def api_strategies():
    return jsonify(list(STRATEGY_REGISTRY.keys()))


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.json or {}
    symbols_raw = data.get("symbols", "BTC/USD,ETH/USD,SOL/USD")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

    config = {
        "exchange": data.get("exchange", "binanceus"),
        "symbols": symbols,
        "timeframe": data.get("timeframe", "1d"),
        "days": int(data.get("days", 365)),
        "capital": float(data.get("capital", 10000)),
        "commission": float(data.get("commission", 0.001)),
        "min_win_rate": float(data.get("min_win_rate", 50)),
        "max_drawdown": float(data.get("max_drawdown", 25)),
        "min_trades": int(data.get("min_trades", 5)),
        "top_n": int(data.get("top_n", 5)),
        "search_internet": data.get("search_internet", False),
        "evolve": data.get("evolve", False),
        "evolve_population": int(data.get("evolve_population", 20)),
        "evolve_generations": int(data.get("evolve_generations", 10)),
        "evolve_mutation_rate": float(data.get("evolve_mutation_rate", 0.3)),
    }

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "queued",
        "log": [],
        "all_results": [],
        "top_results": [],
        "evolution_results": [],
        "filtered_count": 0,
        "strategies_found": 0,
        "error": None,
        "config": config,
    }

    thread = threading.Thread(target=_run_backtest_job, args=(job_id, config), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "log": job["log"],
        "error": job["error"],
        "strategies_found": job.get("strategies_found", 0),
    })


@app.route("/api/results/<job_id>")
def api_results(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "all_results": job["all_results"],
        "top_results": job["top_results"],
        "evolution_results": job["evolution_results"],
        "filtered_count": job["filtered_count"],
        "config": job.get("config", {}),
    })


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtester Quest - Crypto Strategy Arena</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

:root {
  --bg-dark: #1a1a2e;
  --bg-medium: #16213e;
  --bg-card: #0f3460;
  --bg-input: #1a1a3e;
  --accent-gold: #e6c75d;
  --accent-green: #4ade80;
  --accent-red: #ef4444;
  --accent-blue: #60a5fa;
  --accent-purple: #a78bfa;
  --accent-cyan: #22d3ee;
  --accent-orange: #fb923c;
  --text-primary: #e2e8f0;
  --text-dim: #94a3b8;
  --border-color: #334155;
  --pixel-border: #e6c75d;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Press Start 2P', monospace;
  background: var(--bg-dark);
  color: var(--text-primary);
  min-height: 100vh;
  font-size: 10px;
  line-height: 1.8;
  image-rendering: pixelated;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-gold); }
::-webkit-scrollbar-corner { background: var(--bg-dark); }

canvas#starfield {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  z-index: 0;
  pointer-events: none;
}

.header, .container { position: relative; z-index: 1; }

.pixel-border {
  border: 3px solid var(--pixel-border);
  box-shadow:
    inset -3px -3px 0 0 #b8941f,
    inset 3px 3px 0 0 #f5e08a,
    6px 6px 0 0 rgba(0,0,0,0.3);
}

.pixel-border-blue {
  border: 3px solid var(--accent-blue);
  box-shadow:
    inset -3px -3px 0 0 #2563eb,
    inset 3px 3px 0 0 #93c5fd,
    6px 6px 0 0 rgba(0,0,0,0.3);
}

.pixel-border-green {
  border: 3px solid var(--accent-green);
  box-shadow:
    inset -3px -3px 0 0 #16a34a,
    inset 3px 3px 0 0 #86efac,
    6px 6px 0 0 rgba(0,0,0,0.3);
}

.pixel-border-red {
  border: 3px solid var(--accent-red);
  box-shadow:
    inset -3px -3px 0 0 #dc2626,
    inset 3px 3px 0 0 #fca5a5,
    6px 6px 0 0 rgba(0,0,0,0.3);
}

.pixel-border-purple {
  border: 3px solid var(--accent-purple);
  box-shadow:
    inset -3px -3px 0 0 #7c3aed,
    inset 3px 3px 0 0 #c4b5fd,
    6px 6px 0 0 rgba(0,0,0,0.3);
}

.header {
  background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
  padding: 20px;
  text-align: center;
  border-bottom: 4px solid var(--pixel-border);
}

.title {
  font-size: 22px;
  color: var(--accent-gold);
  text-shadow: 3px 3px 0 #000, -1px -1px 0 #b8941f;
  margin-bottom: 6px;
  letter-spacing: 2px;
}

.subtitle {
  font-size: 8px;
  color: var(--text-dim);
  letter-spacing: 1px;
}

.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.main-grid {
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 20px;
  align-items: start;
}

.sidebar {
  position: sticky;
  top: 12px;
  max-height: calc(100vh - 24px);
  overflow-y: auto;
}

.panel {
  background: var(--bg-medium);
  padding: 16px;
  margin-bottom: 16px;
}

.panel-title {
  font-size: 11px;
  color: var(--accent-gold);
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.panel-title .icon {
  font-size: 16px;
}

.form-group {
  margin-bottom: 12px;
}

.form-group label {
  display: block;
  font-size: 8px;
  color: var(--accent-cyan);
  margin-bottom: 4px;
  text-transform: uppercase;
}

.form-group input, .form-group select {
  width: 100%;
  padding: 8px 10px;
  background: var(--bg-input);
  color: var(--text-primary);
  border: 2px solid var(--border-color);
  font-family: 'Press Start 2P', monospace;
  font-size: 9px;
  outline: none;
  transition: border-color 0.2s;
}

.form-group input:focus, .form-group select:focus {
  border-color: var(--accent-gold);
  box-shadow: 0 0 8px rgba(230, 199, 93, 0.2);
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.checkbox-group {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  cursor: pointer;
}

.checkbox-group input[type="checkbox"] {
  width: 18px;
  height: 18px;
  accent-color: var(--accent-gold);
  cursor: pointer;
}

.checkbox-group label {
  font-size: 8px;
  color: var(--text-primary);
  cursor: pointer;
}

.btn {
  display: inline-block;
  padding: 12px 24px;
  font-family: 'Press Start 2P', monospace;
  font-size: 10px;
  cursor: pointer;
  text-transform: uppercase;
  transition: all 0.1s;
  border: none;
  width: 100%;
}

.btn:active {
  transform: translate(2px, 2px);
  box-shadow: none !important;
}

.btn-quest {
  background: var(--accent-gold);
  color: #1a1a2e;
  border: 3px solid #f5e08a;
  box-shadow:
    inset -3px -3px 0 0 #b8941f,
    inset 3px 3px 0 0 #f5e08a,
    4px 4px 0 0 rgba(0,0,0,0.4);
}

.btn-quest:hover {
  background: #f5e08a;
}

.btn-quest:disabled {
  background: #666;
  color: #999;
  border-color: #888;
  cursor: not-allowed;
  box-shadow: none;
}

/* Phase progress stepper */
.phase-stepper {
  display: flex;
  gap: 2px;
  margin-top: 12px;
  margin-bottom: 4px;
}

.phase-step {
  flex: 1;
  text-align: center;
  padding: 6px 2px;
  font-size: 6px;
  background: var(--bg-dark);
  color: var(--text-dim);
  border: 1px solid var(--border-color);
  transition: all 0.3s;
  position: relative;
  overflow: hidden;
}

.phase-step.active {
  border-color: var(--accent-gold);
  color: var(--accent-gold);
  background: rgba(230, 199, 93, 0.1);
}

.phase-step.active::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent-gold);
  animation: phaseGlow 1s ease-in-out infinite;
}

@keyframes phaseGlow {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}

.phase-step.done {
  border-color: var(--accent-green);
  color: var(--accent-green);
  background: rgba(74, 222, 128, 0.08);
}

.phase-step.error {
  border-color: var(--accent-red);
  color: var(--accent-red);
}

.status-bar {
  background: #0d1117;
  padding: 10px 14px;
  font-size: 8px;
  color: var(--accent-green);
  min-height: 36px;
  display: flex;
  align-items: center;
  gap: 10px;
  border: 2px solid var(--border-color);
}

.status-bar .status-text {
  flex: 1;
}

.spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid var(--accent-gold);
  border-top-color: transparent;
  animation: spin 0.6s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.quest-log {
  background: #0d1117;
  border: 2px solid var(--border-color);
  padding: 10px;
  max-height: 160px;
  overflow-y: auto;
  font-size: 7px;
  color: var(--text-dim);
  line-height: 2;
}

.quest-log .log-entry {
  padding: 2px 0;
  border-bottom: 1px solid #1e293b;
}

.quest-log .log-entry:last-child {
  border-bottom: none;
  color: var(--accent-green);
}

.results-section {
  display: none;
}

.results-section.active {
  display: block;
}

/* Victory banner */
.victory-banner {
  background: linear-gradient(135deg, rgba(230,199,93,0.15) 0%, rgba(74,222,128,0.1) 100%);
  border: 2px solid var(--accent-gold);
  padding: 14px 18px;
  margin-bottom: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  animation: bannerSlide 0.4s ease-out;
}

@keyframes bannerSlide {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

.victory-banner .banner-title {
  font-size: 10px;
  color: var(--accent-gold);
}

.victory-stats {
  display: flex;
  gap: 16px;
}

.victory-stat {
  text-align: center;
}

.victory-stat .val {
  font-size: 12px;
  color: var(--accent-green);
}

.victory-stat .lbl {
  font-size: 6px;
  color: var(--text-dim);
  margin-top: 2px;
}

.tab-bar {
  display: flex;
  gap: 4px;
  margin-bottom: 0;
}

.tab {
  padding: 10px 16px;
  font-family: 'Press Start 2P', monospace;
  font-size: 8px;
  cursor: pointer;
  background: var(--bg-dark);
  color: var(--text-dim);
  border: 2px solid var(--border-color);
  border-bottom: none;
  transition: all 0.1s;
}

.tab.active {
  background: var(--bg-medium);
  color: var(--accent-gold);
  border-color: var(--pixel-border);
}

.tab:hover:not(.active) {
  color: var(--text-primary);
}

.tab-content {
  border: 2px solid var(--border-color);
  background: var(--bg-medium);
  padding: 16px;
  min-height: 300px;
}

.tab.active + .tab-content {
  border-color: var(--pixel-border);
}

.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 8px;
}

.results-table th {
  background: var(--bg-dark);
  color: var(--accent-cyan);
  padding: 10px 8px;
  text-align: left;
  border-bottom: 2px solid var(--pixel-border);
  position: sticky;
  top: 0;
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
  transition: color 0.2s;
}

.results-table th:hover {
  color: var(--accent-gold);
}

.results-table th .sort-arrow {
  font-size: 6px;
  margin-left: 3px;
  opacity: 0.4;
}

.results-table th.sorted .sort-arrow {
  opacity: 1;
  color: var(--accent-gold);
}

.results-table td {
  padding: 8px;
  border-bottom: 1px solid var(--border-color);
  white-space: nowrap;
}

.results-table tr:hover td {
  background: rgba(230, 199, 93, 0.05);
}

.results-table .positive { color: var(--accent-green); }
.results-table .negative { color: var(--accent-red); }
.results-table .neutral { color: var(--text-dim); }

.top-card {
  background: var(--bg-dark);
  padding: 16px;
  margin-bottom: 12px;
  transition: transform 0.15s;
}

.top-card:hover {
  transform: translateX(4px);
}

.top-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.top-card-rank {
  font-size: 14px;
  color: var(--accent-gold);
}

.top-card-name {
  font-size: 10px;
  color: var(--text-primary);
}

.top-card-symbol {
  font-size: 8px;
  color: var(--accent-cyan);
}

/* Strategy type badge */
.type-badge {
  display: inline-block;
  padding: 2px 6px;
  font-size: 6px;
  font-family: 'Press Start 2P', monospace;
  margin-left: 6px;
  vertical-align: middle;
}

.type-badge.trend { background: rgba(96, 165, 250, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
.type-badge.reversion { background: rgba(167, 139, 250, 0.2); color: var(--accent-purple); border: 1px solid var(--accent-purple); }
.type-badge.momentum { background: rgba(251, 146, 60, 0.2); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
.type-badge.volatility { background: rgba(34, 211, 238, 0.2); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }

/* Win/Loss ratio bar */
.wl-bar-container {
  display: flex;
  height: 6px;
  width: 100%;
  margin-top: 8px;
  overflow: hidden;
  background: #333;
}

.wl-bar-win {
  background: var(--accent-green);
  height: 100%;
  transition: width 0.3s;
}

.wl-bar-loss {
  background: var(--accent-red);
  height: 100%;
  transition: width 0.3s;
}

.wl-label {
  display: flex;
  justify-content: space-between;
  font-size: 6px;
  margin-top: 2px;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 8px;
}

.stat-item {
  background: var(--bg-medium);
  padding: 8px;
  border: 1px solid var(--border-color);
}

.stat-label {
  font-size: 6px;
  color: var(--text-dim);
  text-transform: uppercase;
  margin-bottom: 4px;
  cursor: help;
}

.stat-value {
  font-size: 10px;
}

.evo-card {
  background: var(--bg-dark);
  padding: 14px;
  margin-bottom: 10px;
}

.evo-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.evo-badge {
  padding: 3px 8px;
  font-size: 7px;
  font-family: 'Press Start 2P', monospace;
}

.evo-badge.improved {
  background: var(--accent-green);
  color: #000;
}

.evo-badge.not-improved {
  background: var(--accent-red);
  color: #fff;
}

.evo-params {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 10px;
}

.evo-param-section h4 {
  font-size: 7px;
  color: var(--accent-purple);
  margin-bottom: 6px;
}

.evo-param-row {
  font-size: 7px;
  padding: 3px 0;
  color: var(--text-dim);
}

.evo-param-row.changed {
  color: var(--accent-green);
}

.evo-comparison {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 6px;
}

.evo-metric {
  background: var(--bg-medium);
  padding: 6px;
  border: 1px solid var(--border-color);
}

.evo-metric-label {
  font-size: 6px;
  color: var(--text-dim);
}

.evo-metric-values {
  font-size: 7px;
  display: flex;
  gap: 4px;
  align-items: center;
}

.evo-arrow {
  color: var(--accent-gold);
  font-size: 8px;
}

.hp-bar-container {
  width: 100%;
  height: 8px;
  background: #333;
  margin-top: 4px;
  position: relative;
  overflow: hidden;
}

.hp-bar {
  height: 100%;
  background: var(--accent-green);
  transition: width 0.5s ease-out;
}

.hp-bar.medium { background: var(--accent-gold); }
.hp-bar.low { background: var(--accent-red); }

.strategy-list {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.strategy-item {
  background: var(--bg-dark);
  border: 2px solid var(--border-color);
  padding: 10px;
  display: flex;
  align-items: center;
  gap: 12px;
  transition: border-color 0.2s, transform 0.15s;
}

.strategy-item:hover {
  border-color: var(--accent-gold);
  transform: translateY(-2px);
}

.strategy-item .sprite {
  font-size: 20px;
  width: 32px;
  text-align: center;
}

.strategy-item .info {
  flex: 1;
}

.strategy-item .name {
  font-size: 8px;
  color: var(--accent-gold);
  margin-bottom: 2px;
}

.strategy-item .desc {
  font-size: 7px;
  color: var(--text-dim);
}

.empty-state {
  text-align: center;
  padding: 40px 20px;
  color: var(--text-dim);
}

.empty-state .icon {
  font-size: 40px;
  margin-bottom: 16px;
}

.empty-state .text {
  font-size: 10px;
  margin-bottom: 8px;
}

.empty-state .subtext {
  font-size: 7px;
}

.table-scroll {
  overflow-x: auto;
  max-height: 500px;
  overflow-y: auto;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.blink { animation: blink 1s step-end infinite; }

@media (max-width: 900px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
  .sidebar {
    position: static;
    max-height: none;
  }
  .strategy-list {
    grid-template-columns: 1fr;
  }
}

.collapsible-header {
  cursor: pointer;
  user-select: none;
  display: flex;
  align-items: center;
  gap: 8px;
}

.collapsible-header .arrow {
  transition: transform 0.2s;
  display: inline-block;
}

.collapsible-header.collapsed .arrow {
  transform: rotate(-90deg);
}

.collapsible-body {
  overflow: hidden;
  transition: max-height 0.3s;
}

.collapsible-body.collapsed {
  max-height: 0 !important;
  padding: 0;
}

/* Tooltip */
[data-tip] {
  position: relative;
}
[data-tip]:hover::after {
  content: attr(data-tip);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: #000;
  color: var(--text-primary);
  padding: 4px 8px;
  font-size: 6px;
  white-space: nowrap;
  z-index: 10;
  border: 1px solid var(--accent-gold);
  pointer-events: none;
}
</style>
</head>
<body>

<canvas id="starfield"></canvas>

<div class="header">
  <div class="title">&#x2694;&#xFE0F; BACKTESTER QUEST &#x1F6E1;&#xFE0F;</div>
  <div class="subtitle">CRYPTO STRATEGY ARENA &bull; FIND THE LEGENDARY LOW-DRAWDOWN BUILD</div>
</div>

<div class="container">
  <div class="main-grid">

    <!-- LEFT SIDEBAR: QUEST CONFIG (sticky) -->
    <div class="sidebar">
      <div class="panel pixel-border">
        <div class="panel-title"><span class="icon">&#x1F4DC;</span> QUEST PARAMETERS</div>

        <div class="form-group">
          <label>Exchange</label>
          <select id="exchange">
            <option value="binanceus">Binance US</option>
            <option value="kraken">Kraken</option>
            <option value="coinbasepro">Coinbase Pro</option>
            <option value="bitfinex">Bitfinex</option>
            <option value="bybit">Bybit</option>
          </select>
        </div>

        <div class="form-group">
          <label>Trading Pairs (comma-separated)</label>
          <input type="text" id="symbols" value="BTC/USD,ETH/USD,SOL/USD" />
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Timeframe</label>
            <select id="timeframe">
              <option value="1h">1 Hour</option>
              <option value="4h">4 Hours</option>
              <option value="1d" selected>1 Day</option>
              <option value="1w">1 Week</option>
            </select>
          </div>
          <div class="form-group">
            <label>History (Days)</label>
            <input type="number" id="days" value="90" min="7" max="1000" />
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Starting Gold &#x1FA99;</label>
            <input type="number" id="capital" value="10000" min="100" />
          </div>
          <div class="form-group">
            <label>Commission</label>
            <input type="number" id="commission" value="0.001" step="0.0001" min="0" />
          </div>
        </div>
      </div>

      <div class="panel pixel-border-blue">
        <div class="panel-title"><span class="icon">&#x1F3AF;</span> FILTER THRESHOLDS</div>

        <div class="form-row">
          <div class="form-group">
            <label>Min Win Rate %</label>
            <input type="number" id="min_win_rate" value="50" min="0" max="100" />
          </div>
          <div class="form-group">
            <label>Max Drawdown %</label>
            <input type="number" id="max_drawdown" value="25" min="1" max="100" />
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>Min Trades</label>
            <input type="number" id="min_trades" value="5" min="1" />
          </div>
          <div class="form-group">
            <label>Show Top N</label>
            <input type="number" id="top_n" value="5" min="1" max="50" />
          </div>
        </div>
      </div>

      <div class="panel pixel-border-purple">
        <div class="collapsible-header" onclick="toggleCollapse('evo-section', this)">
          <span class="icon">&#x1F9EC;</span>
          <span class="panel-title" style="margin-bottom:0">EVOLUTION LAB</span>
          <span class="arrow">&#x25BC;</span>
        </div>
        <div id="evo-section" class="collapsible-body collapsed" style="max-height: 0;">
          <div style="margin-top: 12px;">
            <div class="checkbox-group">
              <input type="checkbox" id="evolve" />
              <label for="evolve">Enable Genetic Evolution</label>
            </div>

            <div id="evo-params" style="display:none;">
              <div class="form-row">
                <div class="form-group">
                  <label>Population</label>
                  <input type="number" id="evolve_population" value="20" min="4" />
                </div>
                <div class="form-group">
                  <label>Generations</label>
                  <input type="number" id="evolve_generations" value="10" min="1" />
                </div>
              </div>
              <div class="form-group">
                <label>Mutation Rate</label>
                <input type="number" id="evolve_mutation_rate" value="0.3" step="0.05" min="0" max="1" />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="checkbox-group" style="margin-bottom: 16px;">
        <input type="checkbox" id="search_internet" />
        <label for="search_internet">&#x1F310; Search Internet for Strategies</label>
      </div>

      <button class="btn btn-quest" id="runBtn" onclick="startQuest()">
        &#x2694;&#xFE0F; BEGIN QUEST &#x2694;&#xFE0F;
      </button>

      <div id="phaseStepper" style="display:none;" class="phase-stepper">
        <div class="phase-step" data-phase="searching">Scout</div>
        <div class="phase-step" data-phase="fetching">Fetch</div>
        <div class="phase-step" data-phase="backtesting">Battle</div>
        <div class="phase-step" data-phase="ranking">Rank</div>
        <div class="phase-step" data-phase="complete">Done</div>
      </div>

      <div id="statusBar" class="status-bar" style="margin-top: 4px; display: none;">
        <div class="spinner" id="statusSpinner"></div>
        <div class="status-text" id="statusText">Preparing quest...</div>
      </div>

      <div id="questLog" class="quest-log" style="margin-top: 4px; display: none;">
      </div>
    </div>

    <!-- RIGHT MAIN AREA -->
    <div class="main-area">

      <!-- EMPTY STATE -->
      <div id="emptyState" class="empty-state">
        <div class="icon">&#x1F3F0;</div>
        <div class="text">No quest started yet</div>
        <div class="subtext">Configure your parameters and<br/>click BEGIN QUEST to search<br/>for legendary strategies</div>
        <div style="margin-top: 24px;">
          <div class="strategy-list" id="strategyPreview"></div>
        </div>
      </div>

      <!-- RESULTS AREA -->
      <div id="resultsArea" class="results-section">
        <div id="victoryBanner"></div>
        <div class="tab-bar" id="tabBar">
          <div class="tab active" onclick="switchTab('top')" data-tab="top">&#x1F3C6; Top Builds</div>
          <div class="tab" onclick="switchTab('all')" data-tab="all">&#x1F4CA; All Results</div>
          <div class="tab" onclick="switchTab('evo')" data-tab="evo" id="evoTab" style="display:none;">&#x1F9EC; Evolution</div>
        </div>
        <div class="tab-content" id="tabContent">
          <div id="topContent"></div>
          <div id="allContent" style="display:none;"></div>
          <div id="evoContent" style="display:none;"></div>
        </div>
      </div>

    </div>
  </div>
</div>

<script>
// Starfield background
(function() {
  const c = document.getElementById('starfield');
  const ctx = c.getContext('2d');
  let stars = [];
  function resize() {
    c.width = window.innerWidth;
    c.height = window.innerHeight;
    stars = [];
    for (let i = 0; i < 120; i++) {
      stars.push({
        x: Math.random() * c.width,
        y: Math.random() * c.height,
        r: Math.random() * 1.5 + 0.3,
        speed: Math.random() * 0.15 + 0.02,
        phase: Math.random() * Math.PI * 2,
      });
    }
  }
  resize();
  window.addEventListener('resize', resize);
  function draw() {
    ctx.clearRect(0, 0, c.width, c.height);
    const t = Date.now() * 0.001;
    stars.forEach(s => {
      const alpha = 0.3 + 0.4 * Math.sin(t * s.speed * 3 + s.phase);
      ctx.fillStyle = `rgba(230, 199, 93, ${alpha})`;
      ctx.fillRect(Math.round(s.x), Math.round(s.y), Math.ceil(s.r), Math.ceil(s.r));
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

const STRATEGY_SPRITES = {
  'RSI Mean Reversion': '\u{1F9D9}',
  'EMA Crossover': '\u{1F9DD}',
  'Bollinger Band Squeeze': '\u{1F9DE}',
  'MACD Histogram Reversal': '\u{1F977}',
  'Stochastic RSI': '\u{1F9DA}',
  'Triple EMA': '\u{1F934}',
  'ADX Trend': '\u{1FA96}',
  'Dual Momentum': '\u{1F3C7}',
  'Williams %R': '\u{1F47E}',
  'Parabolic SAR': '\u{1F680}',
  'Ichimoku Cloud': '\u{1F30A}',
};

const STRATEGY_TYPES = {
  'RSI Mean Reversion': 'reversion',
  'EMA Crossover': 'trend',
  'Bollinger Band Squeeze': 'volatility',
  'MACD Histogram Reversal': 'trend',
  'Stochastic RSI Oversold Bounce': 'reversion',
  'Triple EMA Trend Filter': 'trend',
  'ADX Trend Strength + RSI': 'trend',
  'Dual Momentum': 'momentum',
  'Williams %R Extreme Reversal': 'reversion',
  'Parabolic SAR Trend Rider': 'trend',
  'Ichimoku Cloud Breakout': 'trend',
};

const TYPE_LABELS = {
  'trend': 'TREND',
  'reversion': 'REVERT',
  'momentum': 'MOMENTUM',
  'volatility': 'VOLATILITY',
};

function getSprite(name) {
  for (const [key, val] of Object.entries(STRATEGY_SPRITES)) {
    if (name.toLowerCase().includes(key.toLowerCase().split(' ')[0].toLowerCase())) {
      return val;
    }
  }
  return '\u2694\uFE0F';
}

function getTypeBadge(name) {
  for (const [key, type] of Object.entries(STRATEGY_TYPES)) {
    if (name.toLowerCase().includes(key.toLowerCase().split(' ')[0].toLowerCase())) {
      return `<span class="type-badge ${type}">${TYPE_LABELS[type]}</span>`;
    }
  }
  return '';
}

const TOOLTIPS = {
  'Win Rate': 'Percentage of trades that were profitable',
  'Max Drawdown': 'Largest peak-to-trough equity decline',
  'Total Return': 'Net percentage gain/loss over the period',
  'Sharpe Ratio': 'Risk-adjusted return (higher = better)',
  'Profit Factor': 'Gross wins / gross losses (>1 = profitable)',
  'Total Trades': 'Number of completed round-trip trades',
  'Avg Win': 'Average profit on winning trades',
  'Avg Loss': 'Average loss on losing trades',
  'Calmar': 'Annualized return / max drawdown',
  'Expectancy': 'Average PnL per trade in percent',
  'Holding Period': 'Average bars held per trade',
  'Max Consec Loss': 'Longest streak of consecutive losses',
};

function tip(label) {
  return TOOLTIPS[label] ? ` data-tip="${TOOLTIPS[label]}"` : '';
}

function toggleCollapse(id, header) {
  const body = document.getElementById(id);
  body.classList.toggle('collapsed');
  header.classList.toggle('collapsed');
  if (!body.classList.contains('collapsed')) {
    body.style.maxHeight = body.scrollHeight + 'px';
  } else {
    body.style.maxHeight = '0';
  }
}

document.getElementById('evolve').addEventListener('change', function() {
  document.getElementById('evo-params').style.display = this.checked ? 'block' : 'none';
  const evoSection = document.getElementById('evo-section');
  if (this.checked && evoSection.classList.contains('collapsed')) {
    evoSection.classList.remove('collapsed');
    evoSection.style.maxHeight = evoSection.scrollHeight + 'px';
    evoSection.previousElementSibling.classList.remove('collapsed');
  }
});

let currentJobId = null;
let pollInterval = null;

const PHASE_ORDER = ['searching', 'fetching', 'backtesting', 'ranking', 'complete'];

function updatePhaseStepper(status) {
  const idx = PHASE_ORDER.indexOf(status);
  document.querySelectorAll('.phase-step').forEach(el => {
    const phase = el.dataset.phase;
    const pi = PHASE_ORDER.indexOf(phase);
    el.classList.remove('active', 'done', 'error');
    if (status === 'error') {
      if (pi <= idx || idx === -1) el.classList.add('error');
    } else if (pi < idx) {
      el.classList.add('done');
    } else if (pi === idx) {
      el.classList.add('active');
    }
  });
}

function startQuest() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerHTML = '&#x23F3; QUEST IN PROGRESS...';

  document.getElementById('phaseStepper').style.display = 'flex';
  document.getElementById('statusBar').style.display = 'flex';
  document.getElementById('questLog').style.display = 'block';
  document.getElementById('questLog').innerHTML = '';
  document.getElementById('statusText').textContent = 'Preparing quest...';
  document.getElementById('statusSpinner').style.display = 'inline-block';
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('resultsArea').classList.remove('active');
  document.getElementById('victoryBanner').innerHTML = '';
  updatePhaseStepper('queued');

  const config = {
    exchange: document.getElementById('exchange').value,
    symbols: document.getElementById('symbols').value,
    timeframe: document.getElementById('timeframe').value,
    days: document.getElementById('days').value,
    capital: document.getElementById('capital').value,
    commission: document.getElementById('commission').value,
    min_win_rate: document.getElementById('min_win_rate').value,
    max_drawdown: document.getElementById('max_drawdown').value,
    min_trades: document.getElementById('min_trades').value,
    top_n: document.getElementById('top_n').value,
    search_internet: document.getElementById('search_internet').checked,
    evolve: document.getElementById('evolve').checked,
    evolve_population: document.getElementById('evolve_population').value,
    evolve_generations: document.getElementById('evolve_generations').value,
    evolve_mutation_rate: document.getElementById('evolve_mutation_rate').value,
  };

  fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  .then(r => r.json())
  .then(data => {
    currentJobId = data.job_id;
    pollInterval = setInterval(pollStatus, 800);
  })
  .catch(err => {
    btn.disabled = false;
    btn.innerHTML = '&#x2694;&#xFE0F; BEGIN QUEST &#x2694;&#xFE0F;';
    alert('Failed to start quest: ' + err);
  });
}

const STATUS_LABELS = {
  'queued': '&#x23F3; Queuing quest...',
  'searching': '&#x1F50D; Scouting for strategies...',
  'fetching': '&#x1F4E6; Gathering market scrolls...',
  'backtesting': '&#x2694;&#xFE0F; Battling in the arena...',
  'ranking': '&#x1F3C6; Judging the champions...',
  'evolving': '&#x1F9EC; Evolving in the lab...',
  'complete': '&#x2705; Quest complete!',
  'error': '&#x274C; Quest failed!',
};

function pollStatus() {
  if (!currentJobId) return;

  fetch(`/api/status/${currentJobId}`)
  .then(r => r.json())
  .then(data => {
    const statusText = document.getElementById('statusText');
    statusText.innerHTML = STATUS_LABELS[data.status] || data.status;
    updatePhaseStepper(data.status);

    const log = document.getElementById('questLog');
    log.innerHTML = data.log.map(l => `<div class="log-entry">&gt; ${l}</div>`).join('');
    log.scrollTop = log.scrollHeight;

    if (data.status === 'complete') {
      clearInterval(pollInterval);
      document.getElementById('statusSpinner').style.display = 'none';
      document.getElementById('runBtn').disabled = false;
      document.getElementById('runBtn').innerHTML = '&#x2694;&#xFE0F; BEGIN QUEST &#x2694;&#xFE0F;';
      loadResults();
    } else if (data.status === 'error') {
      clearInterval(pollInterval);
      document.getElementById('statusSpinner').style.display = 'none';
      document.getElementById('runBtn').disabled = false;
      document.getElementById('runBtn').innerHTML = '&#x2694;&#xFE0F; BEGIN QUEST &#x2694;&#xFE0F;';
      statusText.innerHTML = `&#x274C; ${data.error || 'Unknown error'}`;
    }
  });
}

function loadResults() {
  fetch(`/api/results/${currentJobId}`)
  .then(r => r.json())
  .then(data => {
    renderVictoryBanner(data);
    renderTopResults(data.top_results);
    renderAllResults(data.all_results, data.filtered_count, data.config);
    if (data.evolution_results && data.evolution_results.length > 0) {
      document.getElementById('evoTab').style.display = 'inline-block';
      renderEvoResults(data.evolution_results);
    }
    document.getElementById('resultsArea').classList.add('active');
    switchTab('top');
  });
}

function renderVictoryBanner(data) {
  const total = data.all_results.length;
  const passed = data.filtered_count;
  const best = data.top_results[0];
  const banner = document.getElementById('victoryBanner');
  if (!best) { banner.innerHTML = ''; return; }
  banner.innerHTML = `
    <div class="victory-banner pixel-border">
      <div class="banner-title">&#x1F3C6; QUEST COMPLETE</div>
      <div class="victory-stats">
        <div class="victory-stat">
          <div class="val">${total}</div>
          <div class="lbl">TESTED</div>
        </div>
        <div class="victory-stat">
          <div class="val">${passed}</div>
          <div class="lbl">PASSED</div>
        </div>
        <div class="victory-stat">
          <div class="val">${best.win_rate}%</div>
          <div class="lbl">BEST WR</div>
        </div>
        <div class="victory-stat">
          <div class="val">${best.max_drawdown_pct}%</div>
          <div class="lbl">BEST DD</div>
        </div>
      </div>
    </div>`;
}

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${tab}"]`).classList.add('active');

  document.getElementById('topContent').style.display = tab === 'top' ? 'block' : 'none';
  document.getElementById('allContent').style.display = tab === 'all' ? 'block' : 'none';
  document.getElementById('evoContent').style.display = tab === 'evo' ? 'block' : 'none';
}

function colorClass(val, thresholdGood, thresholdBad, higherBetter=true) {
  if (higherBetter) {
    if (val >= thresholdGood) return 'positive';
    if (val <= thresholdBad) return 'negative';
  } else {
    if (val <= thresholdGood) return 'positive';
    if (val >= thresholdBad) return 'negative';
  }
  return 'neutral';
}

function hpBarClass(pct) {
  if (pct >= 60) return '';
  if (pct >= 30) return 'medium';
  return 'low';
}

function renderTopResults(results) {
  const container = document.getElementById('topContent');
  if (!results.length) {
    container.innerHTML = `<div class="empty-state"><div class="icon">&#x1F480;</div><div class="text">No champions emerged</div><div class="subtext">Try relaxing your filter thresholds</div></div>`;
    return;
  }

  let html = '';
  results.forEach((r, i) => {
    const rank = i + 1;
    const borderClass = rank === 1 ? 'pixel-border' : (rank <= 3 ? 'pixel-border-blue' : 'pixel-border-green');
    const wrClass = colorClass(r.win_rate, 55, 45);
    const ddClass = colorClass(r.max_drawdown_pct, 15, 25, false);
    const retClass = r.total_return_pct >= 0 ? 'positive' : 'negative';
    const pfDisplay = r.profit_factor === 'Inf' ? '&infin;' : r.profit_factor;
    const totalTrades = r.winning_trades + r.losing_trades;
    const winPct = totalTrades > 0 ? (r.winning_trades / totalTrades * 100) : 0;
    const lossPct = totalTrades > 0 ? (r.losing_trades / totalTrades * 100) : 0;

    html += `
      <div class="top-card ${borderClass}">
        <div class="top-card-header">
          <div>
            <span class="top-card-rank">#${rank}</span>
            <span style="font-size:18px; margin: 0 6px;">${getSprite(r.strategy_name)}</span>
            <span class="top-card-name">${r.strategy_name}</span>
            ${getTypeBadge(r.strategy_name)}
          </div>
          <span class="top-card-symbol">${r.symbol}</span>
        </div>
        ${totalTrades > 0 ? `
        <div class="wl-bar-container">
          <div class="wl-bar-win" style="width:${winPct}%"></div>
          <div class="wl-bar-loss" style="width:${lossPct}%"></div>
        </div>
        <div class="wl-label">
          <span class="positive">${r.winning_trades}W</span>
          <span class="negative">${r.losing_trades}L</span>
        </div>` : ''}
        <div class="stat-grid" style="margin-top:8px;">
          <div class="stat-item">
            <div class="stat-label"${tip('Win Rate')}>Win Rate</div>
            <div class="stat-value ${wrClass}">${r.win_rate}%</div>
            <div class="hp-bar-container"><div class="hp-bar ${hpBarClass(r.win_rate)}" style="width:${Math.min(r.win_rate, 100)}%"></div></div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Max Drawdown')}>Max Drawdown</div>
            <div class="stat-value ${ddClass}">${r.max_drawdown_pct}%</div>
            <div class="hp-bar-container"><div class="hp-bar ${hpBarClass(100 - r.max_drawdown_pct)}" style="width:${Math.min(r.max_drawdown_pct, 100)}%"></div></div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Total Return')}>Total Return</div>
            <div class="stat-value ${retClass}">${r.total_return_pct}%</div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Sharpe Ratio')}>Sharpe Ratio</div>
            <div class="stat-value ${colorClass(r.sharpe_ratio, 1, 0)}">${r.sharpe_ratio}</div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Profit Factor')}>Profit Factor</div>
            <div class="stat-value">${pfDisplay}</div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Total Trades')}>Total Trades</div>
            <div class="stat-value">${r.total_trades}</div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Avg Win')}>Avg Win</div>
            <div class="stat-value positive">${r.avg_win_pct}%</div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Avg Loss')}>Avg Loss</div>
            <div class="stat-value negative">${r.avg_loss_pct}%</div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Calmar')}>Calmar</div>
            <div class="stat-value">${r.calmar_ratio}</div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Expectancy')}>Expectancy</div>
            <div class="stat-value ${r.expectancy >= 0 ? 'positive' : 'negative'}">${r.expectancy}%</div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Holding Period')}>Holding Period</div>
            <div class="stat-value">${r.avg_holding_periods} bars</div>
          </div>
          <div class="stat-item">
            <div class="stat-label"${tip('Max Consec Loss')}>Max Consec Loss</div>
            <div class="stat-value">${r.max_consecutive_losses}</div>
          </div>
        </div>
      </div>`;
  });
  container.innerHTML = html;
}

let allResultsData = [];
let allResultsConfig = {};
let allResultsFilteredCount = 0;
let currentSort = { col: null, asc: true };

function renderAllResults(results, filteredCount, config) {
  allResultsData = results;
  allResultsConfig = config;
  allResultsFilteredCount = filteredCount;
  currentSort = { col: null, asc: true };
  renderAllTable();
}

function renderAllTable() {
  const container = document.getElementById('allContent');
  const results = allResultsData;
  const config = allResultsConfig;
  if (!results.length) {
    container.innerHTML = `<div class="empty-state"><div class="icon">&#x1F4ED;</div><div class="text">No results</div></div>`;
    return;
  }

  const sorted = [...results];
  if (currentSort.col !== null) {
    sorted.sort((a, b) => {
      let va = a[currentSort.col], vb = b[currentSort.col];
      if (va === 'Inf') va = 1e9;
      if (vb === 'Inf') vb = 1e9;
      if (typeof va === 'string') { va = va.toLowerCase(); vb = (vb||'').toLowerCase(); }
      if (va < vb) return currentSort.asc ? -1 : 1;
      if (va > vb) return currentSort.asc ? 1 : -1;
      return 0;
    });
  }

  const cols = [
    { key: null, label: '#' },
    { key: 'strategy_name', label: 'Strategy' },
    { key: 'symbol', label: 'Symbol' },
    { key: 'total_trades', label: 'Trades' },
    { key: 'win_rate', label: 'Win Rate' },
    { key: 'max_drawdown_pct', label: 'Max DD' },
    { key: 'total_return_pct', label: 'Return' },
    { key: 'sharpe_ratio', label: 'Sharpe' },
    { key: 'profit_factor', label: 'P.Factor' },
    { key: 'avg_win_pct', label: 'Avg Win' },
    { key: 'avg_loss_pct', label: 'Avg Loss' },
    { key: 'calmar_ratio', label: 'Calmar' },
    { key: 'expectancy', label: 'Expect.' },
  ];

  let html = `<div style="margin-bottom:10px;font-size:8px;color:var(--accent-cyan);">
    ${allResultsFilteredCount}/${results.length} strategies passed filters
    (WR &ge; ${config.min_win_rate || 50}%, DD &le; ${config.max_drawdown || 25}%)
  </div>`;

  html += `<div class="table-scroll"><table class="results-table">
    <thead><tr>`;

  cols.forEach(c => {
    const isSorted = currentSort.col === c.key && c.key !== null;
    const arrow = c.key ? `<span class="sort-arrow">${isSorted ? (currentSort.asc ? '\u25B2' : '\u25BC') : '\u25B2'}</span>` : '';
    const cls = isSorted ? ' class="sorted"' : '';
    const onclick = c.key ? ` onclick="sortAllResults('${c.key}')"` : '';
    html += `<th${cls}${onclick}>${c.label}${arrow}</th>`;
  });

  html += `</tr></thead><tbody>`;

  sorted.forEach((r, i) => {
    const wrClass = colorClass(r.win_rate, 55, 45);
    const ddClass = colorClass(r.max_drawdown_pct, 15, 25, false);
    const retClass = r.total_return_pct >= 0 ? 'positive' : 'negative';
    const pfDisplay = r.profit_factor === 'Inf' ? '&infin;' : r.profit_factor;

    html += `<tr>
      <td>${i+1}</td>
      <td>${getSprite(r.strategy_name)} ${r.strategy_name} ${getTypeBadge(r.strategy_name)}</td>
      <td>${r.symbol}</td>
      <td>${r.total_trades}</td>
      <td class="${wrClass}">${r.win_rate}%</td>
      <td class="${ddClass}">${r.max_drawdown_pct}%</td>
      <td class="${retClass}">${r.total_return_pct}%</td>
      <td class="${colorClass(r.sharpe_ratio, 1, 0)}">${r.sharpe_ratio}</td>
      <td>${pfDisplay}</td>
      <td class="positive">${r.avg_win_pct}%</td>
      <td class="negative">${r.avg_loss_pct}%</td>
      <td>${r.calmar_ratio}</td>
      <td class="${r.expectancy >= 0 ? 'positive' : 'negative'}">${r.expectancy}%</td>
    </tr>`;
  });

  html += '</tbody></table></div>';
  container.innerHTML = html;
}

function sortAllResults(col) {
  if (currentSort.col === col) {
    currentSort.asc = !currentSort.asc;
  } else {
    currentSort.col = col;
    currentSort.asc = false;
  }
  renderAllTable();
}

function renderEvoResults(results) {
  const container = document.getElementById('evoContent');
  if (!results.length) {
    container.innerHTML = `<div class="empty-state"><div class="icon">&#x1F9EC;</div><div class="text">No evolution data</div></div>`;
    return;
  }

  let html = '';
  results.forEach((r, i) => {
    const borderClass = r.improved ? 'pixel-border-green' : 'pixel-border-red';
    const badgeClass = r.improved ? 'improved' : 'not-improved';
    const badgeText = r.improved ? 'EVOLVED &#x2B06;' : 'NO CHANGE';

    html += `<div class="evo-card ${borderClass}">
      <div class="evo-header">
        <div>
          <span style="font-size:14px;">${getSprite(r.strategy_name)}</span>
          <span style="font-size:9px;color:var(--accent-gold);">${r.strategy_name}</span>
          <span style="font-size:7px;color:var(--text-dim);margin-left:8px;">Gen ${r.generations_run}</span>
        </div>
        <span class="evo-badge ${badgeClass}">${badgeText}</span>
      </div>

      <div class="evo-params">
        <div class="evo-param-section">
          <h4>Original Params</h4>
          ${Object.entries(r.original_params).map(([k,v]) => `<div class="evo-param-row">${k}: ${v}</div>`).join('')}
        </div>
        <div class="evo-param-section">
          <h4>Evolved Params</h4>
          ${Object.entries(r.evolved_params).map(([k,v]) => {
            const changed = r.original_params[k] !== v;
            return `<div class="evo-param-row ${changed ? 'changed' : ''}">${k}: ${v}${changed ? ' *' : ''}</div>`;
          }).join('')}
        </div>
      </div>`;

    if (r.original_result && r.evolved_result) {
      const o = r.original_result;
      const e = r.evolved_result;
      html += `<div class="evo-comparison">
        ${evoMetric('Win Rate', o.win_rate, e.win_rate, '%')}
        ${evoMetric('Max DD', o.max_drawdown_pct, e.max_drawdown_pct, '%', true)}
        ${evoMetric('Return', o.total_return_pct, e.total_return_pct, '%')}
        ${evoMetric('Sharpe', o.sharpe_ratio, e.sharpe_ratio)}
        ${evoMetric('Trades', o.total_trades, e.total_trades)}
      </div>`;
    }

    html += `</div>`;
  });
  container.innerHTML = html;
}

function evoMetric(label, orig, evolved, suffix='', lowerBetter=false) {
  const delta = evolved - orig;
  const better = lowerBetter ? delta < 0 : delta > 0;
  const cls = delta === 0 ? 'neutral' : (better ? 'positive' : 'negative');
  const arrow = delta > 0 ? '+' : '';
  const origDisplay = typeof orig === 'number' ? (Number.isInteger(orig) ? orig : orig.toFixed(2)) : orig;
  const evolvedDisplay = typeof evolved === 'number' ? (Number.isInteger(evolved) ? evolved : evolved.toFixed(2)) : evolved;
  const deltaDisplay = typeof delta === 'number' ? (Number.isInteger(delta) ? delta : delta.toFixed(2)) : delta;

  return `<div class="evo-metric">
    <div class="evo-metric-label">${label}</div>
    <div class="evo-metric-values">
      <span class="neutral">${origDisplay}${suffix}</span>
      <span class="evo-arrow">&rarr;</span>
      <span class="${cls}">${evolvedDisplay}${suffix}</span>
      <span class="${cls}" style="font-size:6px;">(${arrow}${deltaDisplay})</span>
    </div>
  </div>`;
}

const HERO_META = {
  'rsi_mean_reversion': ['\u{1F9D9}', 'RSI Mean Reversion', 'Mage of Mean Reversion', 'reversion'],
  'ema_crossover': ['\u{1F9DD}', 'EMA Crossover', 'Elven Trend Follower', 'trend'],
  'bollinger_squeeze': ['\u{1F9DE}', 'Bollinger Squeeze', 'Genie of Volatility', 'volatility'],
  'macd_histogram': ['\u{1F977}', 'MACD Histogram', 'Ninja of Momentum', 'trend'],
  'stoch_rsi_bounce': ['\u{1F9DA}', 'Stochastic RSI', 'Fairy of Oscillation', 'reversion'],
  'triple_ema': ['\u{1F934}', 'Triple EMA Trend', 'Prince of Pullbacks', 'trend'],
  'adx_rsi': ['\u{1FA96}', 'ADX + RSI Trend', 'Knight of Strength', 'trend'],
  'dual_momentum': ['\u{1F3C7}', 'Dual Momentum', 'Rider of Velocity', 'momentum'],
  'williams_r': ['\u{1F47E}', 'Williams %R', 'Ghost of Extremes', 'reversion'],
  'parabolic_sar': ['\u{1F680}', 'Parabolic SAR', 'Rocket of Trends', 'trend'],
  'ichimoku_breakout': ['\u{1F30A}', 'Ichimoku Cloud', 'Wave of Prophecy', 'trend'],
};

fetch('/api/strategies').then(r => r.json()).then(strategies => {
  const preview = document.getElementById('strategyPreview');
  let html = '<div style="font-size:9px;color:var(--accent-gold);margin-bottom:12px;grid-column:1/-1;text-align:center;">\u2694\uFE0F AVAILABLE HEROES \u2694\uFE0F</div>';
  strategies.forEach(key => {
    const info = HERO_META[key] || ['\u2694\uFE0F', key, '', 'trend'];
    html += `<div class="strategy-item">
      <div class="sprite">${info[0]}</div>
      <div class="info">
        <div class="name">${info[1]} <span class="type-badge ${info[3]}">${TYPE_LABELS[info[3]] || ''}</span></div>
        <div class="desc">${info[2]}</div>
      </div>
    </div>`;
  });
  preview.innerHTML = html;
});
</script>

</body>
</html>
"""


def run_gui(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Launch the web GUI server."""
    app.run(host=host, port=port, debug=debug)
