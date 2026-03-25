"""Fetch historical OHLCV data from crypto exchanges via ccxt."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import ccxt
import pandas as pd


@dataclass
class FetchConfig:
    exchange_id: str = "binanceus"
    symbols: list[str] | None = None
    timeframe: str = "1d"
    since_days: int = 365
    limit_per_request: int = 1000

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]


class DataFetcher:
    """Downloads OHLCV candlestick data from cryptocurrency exchanges."""

    def __init__(self, config: FetchConfig | None = None):
        self.config = config or FetchConfig()
        self._exchange = self._init_exchange()

    def _init_exchange(self) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self.config.exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{self.config.exchange_id}' not supported by ccxt")
        return exchange_class({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str | None = None, since_days: int | None = None) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol.

        Returns a DataFrame with columns: timestamp, open, high, low, close, volume
        """
        tf = timeframe or self.config.timeframe
        days = since_days or self.config.since_days
        since_ms = int((datetime.now(timezone.utc).timestamp() - days * 86400) * 1000)

        all_candles: list = []
        while True:
            candles = self._exchange.fetch_ohlcv(
                symbol, tf, since=since_ms, limit=self.config.limit_per_request
            )
            if not candles:
                break
            all_candles.extend(candles)
            since_ms = candles[-1][0] + 1
            if len(candles) < self.config.limit_per_request:
                break
            time.sleep(self._exchange.rateLimit / 1000)

        if not all_candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for all configured symbols."""
        results = {}
        for symbol in self.config.symbols:
            try:
                results[symbol] = self.fetch_ohlcv(symbol)
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol}: {e}")
        return results
