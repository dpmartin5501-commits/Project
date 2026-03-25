"""Search the internet for crypto trading strategies focused on low drawdown and high win rates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup


@dataclass
class StrategyResult:
    """A strategy discovered from the internet."""
    name: str
    description: str
    source_url: str
    indicators: list[str] = field(default_factory=list)
    timeframe: str = "1d"
    claimed_win_rate: float | None = None
    claimed_max_drawdown: float | None = None
    entry_rules: list[str] = field(default_factory=list)
    exit_rules: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


SEARCH_QUERIES = [
    "crypto trading strategy low drawdown high win rate",
    "cryptocurrency mean reversion strategy low risk",
    "bitcoin trading strategy RSI oversold high win rate",
    "crypto EMA crossover strategy low drawdown",
    "bollinger bands crypto strategy conservative",
    "MACD crypto strategy backtested high win rate",
    "crypto scalping strategy low drawdown",
    "moving average crossover crypto strategy backtested",
    "crypto momentum strategy risk management",
    "stochastic RSI crypto trading strategy",
]

STRATEGY_SITES = [
    "https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp",
    "https://www.babypips.com/learn/forex/summary-common-chart-indicators",
    "https://academy.binance.com/en/articles/a-complete-guide-to-cryptocurrency-trading-for-beginners",
]

KNOWN_INDICATORS = [
    "RSI", "MACD", "EMA", "SMA", "Bollinger Bands", "Stochastic",
    "ATR", "ADX", "CCI", "OBV", "VWAP", "Ichimoku",
    "Parabolic SAR", "Williams %R", "MFI", "ROC",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class StrategySearcher:
    """Searches the web for crypto trading strategies and parses them into actionable descriptions."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    def search_strategies(self, max_results: int = 20) -> list[StrategyResult]:
        """Run all search queries and aggregate strategy results."""
        all_results: list[StrategyResult] = []

        for query in SEARCH_QUERIES:
            try:
                results = self._search_duckduckgo(query)
                all_results.extend(results)
            except Exception:
                continue

        for url in STRATEGY_SITES:
            try:
                results = self._scrape_page(url)
                all_results.extend(results)
            except Exception:
                continue

        all_results.extend(self._get_builtin_strategies())

        seen_names = set()
        unique: list[StrategyResult] = []
        for r in all_results:
            if r.name not in seen_names:
                seen_names.add(r.name)
                unique.append(r)
                if len(unique) >= max_results:
                    break

        return unique

    def _search_duckduckgo(self, query: str) -> list[StrategyResult]:
        """Search DuckDuckGo HTML for strategy pages."""
        url = "https://html.duckduckgo.com/html/"
        try:
            resp = self.session.post(url, data={"q": query}, timeout=self.timeout)
            resp.raise_for_status()
        except Exception:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[StrategyResult] = []

        for result_div in soup.select(".result"):
            title_tag = result_div.select_one(".result__title a")
            snippet_tag = result_div.select_one(".result__snippet")
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            href = title_tag.get("href", "")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            if not self._is_strategy_relevant(title, snippet):
                continue

            indicators = self._extract_indicators(title + " " + snippet)
            win_rate = self._extract_percentage(snippet, r"(\d{1,3}(?:\.\d+)?)\s*%?\s*win\s*rate")
            drawdown = self._extract_percentage(snippet, r"(\d{1,3}(?:\.\d+)?)\s*%?\s*(?:max\s*)?drawdown")

            results.append(StrategyResult(
                name=self._clean_title(title),
                description=snippet,
                source_url=str(href),
                indicators=indicators,
                claimed_win_rate=win_rate,
                claimed_max_drawdown=drawdown,
                tags=self._extract_tags(title + " " + snippet),
            ))

        return results

    def _scrape_page(self, url: str) -> list[StrategyResult]:
        """Scrape a known strategy page for content."""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except Exception:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        indicators = self._extract_indicators(text)

        return [StrategyResult(
            name=f"Strategy from {soup.title.string.strip()[:60] if soup.title else url[:60]}",
            description=text[:500],
            source_url=url,
            indicators=indicators,
            tags=self._extract_tags(text[:1000]),
        )]

    def _get_builtin_strategies(self) -> list[StrategyResult]:
        """Return a curated set of well-known crypto strategies with low drawdown / high win rate."""
        return [
            StrategyResult(
                name="RSI Mean Reversion",
                description=(
                    "Buy when RSI(14) drops below 30 (oversold), sell when RSI crosses above 70 (overbought). "
                    "This mean-reversion strategy works well in ranging markets and historically shows "
                    "high win rates on daily timeframes for major crypto pairs."
                ),
                source_url="builtin://rsi-mean-reversion",
                indicators=["RSI"],
                timeframe="1d",
                claimed_win_rate=62.0,
                claimed_max_drawdown=15.0,
                entry_rules=["RSI(14) < 30"],
                exit_rules=["RSI(14) > 70"],
                tags=["mean-reversion", "oscillator", "conservative"],
            ),
            StrategyResult(
                name="EMA Crossover (9/21)",
                description=(
                    "Enter long when EMA(9) crosses above EMA(21), exit when EMA(9) crosses below EMA(21). "
                    "A classic trend-following strategy that captures momentum with relatively controlled drawdowns."
                ),
                source_url="builtin://ema-crossover-9-21",
                indicators=["EMA"],
                timeframe="1d",
                claimed_win_rate=55.0,
                claimed_max_drawdown=18.0,
                entry_rules=["EMA(9) crosses above EMA(21)"],
                exit_rules=["EMA(9) crosses below EMA(21)"],
                tags=["trend-following", "crossover", "momentum"],
            ),
            StrategyResult(
                name="Bollinger Band Squeeze",
                description=(
                    "Enter when price breaks out of a Bollinger Band squeeze (bandwidth at 6-month low). "
                    "Go long on upper band breakout, short on lower band breakout. "
                    "Uses ATR-based stops to limit drawdown."
                ),
                source_url="builtin://bollinger-squeeze",
                indicators=["Bollinger Bands", "ATR"],
                timeframe="1d",
                claimed_win_rate=58.0,
                claimed_max_drawdown=12.0,
                entry_rules=["BB bandwidth at 6-month low", "Price breaks above upper band"],
                exit_rules=["Price crosses middle band", "ATR trailing stop hit"],
                tags=["volatility-breakout", "conservative", "squeeze"],
            ),
            StrategyResult(
                name="MACD Histogram Reversal",
                description=(
                    "Enter long when MACD histogram turns positive after being negative for 3+ bars. "
                    "Exit when histogram turns negative. Filters trades with 50-day SMA trend."
                ),
                source_url="builtin://macd-histogram-reversal",
                indicators=["MACD", "SMA"],
                timeframe="1d",
                claimed_win_rate=54.0,
                claimed_max_drawdown=20.0,
                entry_rules=["MACD histogram turns positive after 3+ negative bars", "Price above SMA(50)"],
                exit_rules=["MACD histogram turns negative"],
                tags=["trend-following", "momentum", "filtered"],
            ),
            StrategyResult(
                name="Stochastic RSI Oversold Bounce",
                description=(
                    "Buy when Stochastic RSI K-line crosses above D-line while both are below 20. "
                    "Sell when K-line crosses below D-line above 80. "
                    "Combines momentum with mean-reversion for high win rates."
                ),
                source_url="builtin://stoch-rsi-bounce",
                indicators=["Stochastic", "RSI"],
                timeframe="1d",
                claimed_win_rate=60.0,
                claimed_max_drawdown=14.0,
                entry_rules=["StochRSI K crosses above D", "Both K and D below 20"],
                exit_rules=["StochRSI K crosses below D above 80"],
                tags=["mean-reversion", "oscillator", "momentum"],
            ),
            StrategyResult(
                name="Triple EMA Trend Filter",
                description=(
                    "Uses EMA(8), EMA(21), EMA(55) alignment for trend confirmation. "
                    "Enter long when all three are aligned bullish (8 > 21 > 55) and price pulls back to EMA(21). "
                    "Exit on EMA(8) crossing below EMA(21). Conservative trend-following with tight stops."
                ),
                source_url="builtin://triple-ema-trend",
                indicators=["EMA"],
                timeframe="1d",
                claimed_win_rate=52.0,
                claimed_max_drawdown=16.0,
                entry_rules=["EMA(8) > EMA(21) > EMA(55)", "Price touches EMA(21)"],
                exit_rules=["EMA(8) crosses below EMA(21)"],
                tags=["trend-following", "pullback", "conservative"],
            ),
            StrategyResult(
                name="ADX Trend Strength + RSI",
                description=(
                    "Enter long when ADX > 25 (strong trend), +DI > -DI (bullish), and RSI between 40-60 "
                    "(not overbought). Exit when ADX drops below 20 or RSI > 75."
                ),
                source_url="builtin://adx-rsi-trend",
                indicators=["ADX", "RSI"],
                timeframe="1d",
                claimed_win_rate=57.0,
                claimed_max_drawdown=13.0,
                entry_rules=["ADX > 25", "+DI > -DI", "RSI between 40-60"],
                exit_rules=["ADX < 20 or RSI > 75"],
                tags=["trend-following", "filtered", "conservative"],
            ),
            StrategyResult(
                name="Dual Momentum",
                description=(
                    "Combine absolute momentum (asset's own past returns) with relative momentum "
                    "(performance vs other assets). Enter when both are positive, exit when either turns negative. "
                    "Historically produces low drawdowns in crypto."
                ),
                source_url="builtin://dual-momentum",
                indicators=["ROC"],
                timeframe="1d",
                claimed_win_rate=56.0,
                claimed_max_drawdown=15.0,
                entry_rules=["12-period ROC > 0 (absolute momentum)", "Asset ROC > benchmark ROC"],
                exit_rules=["ROC < 0 or underperforming benchmark"],
                tags=["momentum", "relative-strength", "conservative"],
            ),
            StrategyResult(
                name="VWAP Reversion",
                description=(
                    "Buy when price drops more than 2% below VWAP, sell when price returns to VWAP. "
                    "Works well for intraday/short-term crypto trading with tight risk controls."
                ),
                source_url="builtin://vwap-reversion",
                indicators=["VWAP", "ATR"],
                timeframe="4h",
                claimed_win_rate=64.0,
                claimed_max_drawdown=10.0,
                entry_rules=["Price < VWAP * 0.98"],
                exit_rules=["Price >= VWAP", "Stop loss at VWAP * 0.95"],
                tags=["mean-reversion", "intraday", "conservative"],
            ),
            StrategyResult(
                name="Ichimoku Cloud Breakout",
                description=(
                    "Enter long when price breaks above the Ichimoku cloud and Tenkan-sen > Kijun-sen. "
                    "Exit when price enters the cloud. Known for strong trend identification."
                ),
                source_url="builtin://ichimoku-breakout",
                indicators=["Ichimoku"],
                timeframe="1d",
                claimed_win_rate=53.0,
                claimed_max_drawdown=19.0,
                entry_rules=["Price above Ichimoku cloud", "Tenkan-sen > Kijun-sen"],
                exit_rules=["Price enters cloud"],
                tags=["trend-following", "breakout", "comprehensive"],
            ),
            StrategyResult(
                name="Williams %R Extreme Reversal",
                description=(
                    "Buy when Williams %R drops below -80, sell when it rises above -20. "
                    "A high win-rate mean-reversion approach best used in ranging markets."
                ),
                source_url="builtin://williams-r-reversal",
                indicators=["Williams %R"],
                timeframe="1d",
                claimed_win_rate=61.0,
                claimed_max_drawdown=14.0,
                entry_rules=["Williams %R < -80"],
                exit_rules=["Williams %R > -20"],
                tags=["mean-reversion", "oscillator", "conservative"],
            ),
            StrategyResult(
                name="Parabolic SAR Trend Rider",
                description=(
                    "Enter long when Parabolic SAR flips below price, exit when SAR flips above price. "
                    "Simple trend-following with built-in trailing stop."
                ),
                source_url="builtin://parabolic-sar-trend",
                indicators=["Parabolic SAR"],
                timeframe="1d",
                claimed_win_rate=50.0,
                claimed_max_drawdown=22.0,
                entry_rules=["SAR flips below price"],
                exit_rules=["SAR flips above price"],
                tags=["trend-following", "trailing-stop"],
            ),
        ]

    @staticmethod
    def _is_strategy_relevant(title: str, snippet: str) -> bool:
        text = (title + " " + snippet).lower()
        strategy_terms = ["strategy", "trading", "indicator", "backtest", "signal", "entry", "exit"]
        return any(term in text for term in strategy_terms)

    @staticmethod
    def _extract_indicators(text: str) -> list[str]:
        found = []
        text_upper = text.upper()
        for ind in KNOWN_INDICATORS:
            if ind.upper() in text_upper:
                found.append(ind)
        return found

    @staticmethod
    def _extract_percentage(text: str, pattern: str) -> float | None:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    @staticmethod
    def _clean_title(title: str) -> str:
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) > 80:
            title = title[:77] + "..."
        return title

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        tags = []
        text_lower = text.lower()
        tag_map = {
            "mean-reversion": ["mean reversion", "oversold", "overbought", "reversal"],
            "trend-following": ["trend", "crossover", "moving average", "ema", "sma"],
            "momentum": ["momentum", "breakout", "strength"],
            "conservative": ["low risk", "conservative", "low drawdown", "safe"],
            "scalping": ["scalp", "scalping", "short-term"],
            "swing": ["swing", "multi-day"],
        }
        for tag, keywords in tag_map.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)
        return tags
