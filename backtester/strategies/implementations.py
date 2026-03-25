"""Concrete strategy implementations for backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from .base import BaseStrategy, Signal, StrategyConfig


class RSIMeanReversion(BaseStrategy):
    """Buy when RSI < oversold threshold, sell when RSI > overbought threshold."""

    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70, **kwargs):
        config = StrategyConfig(
            name="RSI Mean Reversion",
            params={"rsi_period": rsi_period, "oversold": oversold, "overbought": overbought},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "rsi_period": (5, 30, 1),
            "oversold": (15, 40, 1),
            "overbought": (60, 85, 1),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=p["rsi_period"]).rsi()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.config.params
        signals = pd.Series(Signal.HOLD, index=df.index)
        signals[df["rsi"] < p["oversold"]] = Signal.BUY
        signals[df["rsi"] > p["overbought"]] = Signal.SELL
        return signals


class EMACrossover(BaseStrategy):
    """Enter on fast EMA crossing above slow EMA, exit on cross below."""

    def __init__(self, fast_period: int = 9, slow_period: int = 21, **kwargs):
        config = StrategyConfig(
            name=f"EMA Crossover ({fast_period}/{slow_period})",
            params={"fast_period": fast_period, "slow_period": slow_period},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "fast_period": (3, 20, 1),
            "slow_period": (10, 60, 1),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=p["fast_period"]).ema_indicator()
        df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=p["slow_period"]).ema_indicator()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        cross_above = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
        cross_below = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
        signals[cross_above] = Signal.BUY
        signals[cross_below] = Signal.SELL
        return signals


class BollingerBandSqueeze(BaseStrategy):
    """Enter on breakout from Bollinger Band squeeze (tight bandwidth)."""

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, squeeze_lookback: int = 120, **kwargs):
        config = StrategyConfig(
            name="Bollinger Band Squeeze",
            params={"bb_period": bb_period, "bb_std": bb_std, "squeeze_lookback": squeeze_lookback},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "bb_period": (10, 40, 1),
            "bb_std": (1.0, 3.5, 0.25),
            "squeeze_lookback": (40, 200, 10),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        bb = ta.volatility.BollingerBands(df["close"], window=p["bb_period"], window_dev=p["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_width_min"] = df["bb_width"].rolling(window=p["squeeze_lookback"], min_periods=20).min()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        squeeze = df["bb_width"] <= df["bb_width_min"] * 1.05
        breakout_up = (df["close"] > df["bb_upper"]) & squeeze.shift(1).fillna(False)
        breakdown = df["close"] < df["bb_middle"]
        signals[breakout_up] = Signal.BUY
        signals[breakdown & ~breakout_up] = Signal.SELL
        return signals


class MACDHistogramReversal(BaseStrategy):
    """Enter when MACD histogram reverses from negative to positive with trend filter."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, trend_period: int = 50, **kwargs):
        config = StrategyConfig(
            name="MACD Histogram Reversal",
            params={"fast": fast, "slow": slow, "signal": signal, "trend_period": trend_period},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "fast": (6, 20, 1),
            "slow": (18, 40, 1),
            "signal": (5, 15, 1),
            "trend_period": (20, 100, 5),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        macd_ind = ta.trend.MACD(df["close"], window_slow=p["slow"], window_fast=p["fast"], window_sign=p["signal"])
        df["macd_hist"] = macd_ind.macd_diff()
        df["sma_trend"] = ta.trend.SMAIndicator(df["close"], window=p["trend_period"]).sma_indicator()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        hist_positive = df["macd_hist"] > 0
        hist_was_neg_3 = (
            (df["macd_hist"].shift(1) < 0)
            & (df["macd_hist"].shift(2) < 0)
            & (df["macd_hist"].shift(3) < 0)
        )
        above_trend = df["close"] > df["sma_trend"]
        signals[hist_positive & hist_was_neg_3 & above_trend] = Signal.BUY
        signals[(df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)] = Signal.SELL
        return signals


class StochasticRSIBounce(BaseStrategy):
    """Buy when Stochastic RSI K crosses above D in oversold zone."""

    def __init__(self, rsi_period: int = 14, stoch_period: int = 14, k_smooth: int = 3, d_smooth: int = 3, **kwargs):
        config = StrategyConfig(
            name="Stochastic RSI Oversold Bounce",
            params={
                "rsi_period": rsi_period,
                "stoch_period": stoch_period,
                "k_smooth": k_smooth,
                "d_smooth": d_smooth,
            },
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "rsi_period": (7, 28, 1),
            "stoch_period": (7, 28, 1),
            "k_smooth": (2, 7, 1),
            "d_smooth": (2, 7, 1),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        stoch_rsi = ta.momentum.StochRSIIndicator(
            df["close"], window=p["rsi_period"], smooth1=p["k_smooth"], smooth2=p["d_smooth"]
        )
        df["stoch_k"] = stoch_rsi.stochrsi_k() * 100
        df["stoch_d"] = stoch_rsi.stochrsi_d() * 100
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        k_cross_above_d = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
        oversold = (df["stoch_k"] < 20) & (df["stoch_d"] < 20)
        k_cross_below_d = (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
        overbought = (df["stoch_k"] > 80)
        signals[k_cross_above_d & oversold] = Signal.BUY
        signals[k_cross_below_d & overbought] = Signal.SELL
        return signals


class TripleEMATrend(BaseStrategy):
    """Trend-following with three EMAs: enter on pullback in aligned trend."""

    def __init__(self, fast: int = 8, medium: int = 21, slow: int = 55, **kwargs):
        config = StrategyConfig(
            name="Triple EMA Trend Filter",
            params={"fast": fast, "medium": medium, "slow": slow},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "fast": (3, 15, 1),
            "medium": (12, 35, 1),
            "slow": (30, 100, 5),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=p["fast"]).ema_indicator()
        df["ema_med"] = ta.trend.EMAIndicator(df["close"], window=p["medium"]).ema_indicator()
        df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=p["slow"]).ema_indicator()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        aligned_bull = (df["ema_fast"] > df["ema_med"]) & (df["ema_med"] > df["ema_slow"])
        pullback = (df["low"] <= df["ema_med"]) & (df["close"] > df["ema_med"])
        exit_signal = (df["ema_fast"] < df["ema_med"]) & (df["ema_fast"].shift(1) >= df["ema_med"].shift(1))
        signals[aligned_bull & pullback] = Signal.BUY
        signals[exit_signal] = Signal.SELL
        return signals


class ADXRSITrend(BaseStrategy):
    """Combine ADX trend strength with RSI filter."""

    def __init__(self, adx_period: int = 14, rsi_period: int = 14, **kwargs):
        config = StrategyConfig(
            name="ADX Trend Strength + RSI",
            params={"adx_period": adx_period, "rsi_period": rsi_period},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "adx_period": (7, 28, 1),
            "rsi_period": (7, 28, 1),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=p["adx_period"])
        df["adx"] = adx_ind.adx()
        df["di_plus"] = adx_ind.adx_pos()
        df["di_minus"] = adx_ind.adx_neg()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=p["rsi_period"]).rsi()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        strong_trend = df["adx"] > 25
        bullish = df["di_plus"] > df["di_minus"]
        rsi_ok = (df["rsi"] > 40) & (df["rsi"] < 60)
        exit_cond = (df["adx"] < 20) | (df["rsi"] > 75)
        signals[strong_trend & bullish & rsi_ok] = Signal.BUY
        signals[exit_cond & ~(strong_trend & bullish & rsi_ok)] = Signal.SELL
        return signals


class DualMomentum(BaseStrategy):
    """Absolute + relative momentum strategy using Rate of Change."""

    def __init__(self, roc_period: int = 12, **kwargs):
        config = StrategyConfig(
            name="Dual Momentum",
            params={"roc_period": roc_period},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "roc_period": (5, 30, 1),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        df["roc"] = ta.momentum.ROCIndicator(df["close"], window=p["roc_period"]).roc()
        df["sma_200"] = ta.trend.SMAIndicator(df["close"], window=200).sma_indicator()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        abs_momentum = df["roc"] > 0
        rel_momentum = df["close"] > df["sma_200"]
        signals[abs_momentum & rel_momentum] = Signal.BUY
        signals[~abs_momentum | ~rel_momentum] = Signal.SELL
        return signals


class WilliamsRReversal(BaseStrategy):
    """Mean-reversion using Williams %R extremes."""

    def __init__(self, period: int = 14, **kwargs):
        config = StrategyConfig(
            name="Williams %R Extreme Reversal",
            params={"period": period},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "period": (5, 30, 1),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            df["high"], df["low"], df["close"], lbp=p["period"]
        ).williams_r()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        signals[df["williams_r"] < -80] = Signal.BUY
        signals[df["williams_r"] > -20] = Signal.SELL
        return signals


class ParabolicSARTrend(BaseStrategy):
    """Trend-following using Parabolic SAR direction."""

    def __init__(self, step: float = 0.02, max_step: float = 0.2, **kwargs):
        config = StrategyConfig(
            name="Parabolic SAR Trend Rider",
            params={"step": step, "max_step": max_step},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "step": (0.005, 0.05, 0.005),
            "max_step": (0.1, 0.4, 0.025),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        df["psar"] = ta.trend.PSARIndicator(
            df["high"], df["low"], df["close"], step=p["step"], max_step=p["max_step"]
        ).psar()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        sar_below = df["psar"] < df["close"]
        sar_flip_below = sar_below & ~sar_below.shift(1).fillna(False)
        sar_flip_above = ~sar_below & sar_below.shift(1).fillna(True)
        signals[sar_flip_below] = Signal.BUY
        signals[sar_flip_above] = Signal.SELL
        return signals


class IchimokuBreakout(BaseStrategy):
    """Trend strategy using Ichimoku Cloud."""

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, **kwargs):
        config = StrategyConfig(
            name="Ichimoku Cloud Breakout",
            params={"tenkan": tenkan, "kijun": kijun, "senkou_b": senkou_b},
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "tenkan": (5, 20, 1),
            "kijun": (15, 40, 1),
            "senkou_b": (30, 80, 2),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        ichi = ta.trend.IchimokuIndicator(
            df["high"], df["low"],
            window1=p["tenkan"], window2=p["kijun"], window3=p["senkou_b"]
        )
        df["tenkan_sen"] = ichi.ichimoku_conversion_line()
        df["kijun_sen"] = ichi.ichimoku_base_line()
        df["senkou_a"] = ichi.ichimoku_a()
        df["senkou_b"] = ichi.ichimoku_b()
        df["cloud_top"] = df[["senkou_a", "senkou_b"]].max(axis=1)
        df["cloud_bottom"] = df[["senkou_a", "senkou_b"]].min(axis=1)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)
        above_cloud = df["close"] > df["cloud_top"]
        tenkan_above_kijun = df["tenkan_sen"] > df["kijun_sen"]
        in_cloud = (df["close"] <= df["cloud_top"]) & (df["close"] >= df["cloud_bottom"])
        signals[above_cloud & tenkan_above_kijun] = Signal.BUY
        signals[in_cloud | (df["close"] < df["cloud_bottom"])] = Signal.SELL
        return signals


class VWAPReversion(BaseStrategy):
    """Mean-reversion strategy: buy when price drops below VWAP by a threshold, sell at VWAP.

    Uses a rolling VWAP approximation (typical price * volume / cumulative volume)
    since true intraday VWAP resets each session. An ATR-based stop loss limits
    downside risk.
    """

    def __init__(self, vwap_period: int = 20, entry_deviation: float = 2.0,
                 atr_period: int = 14, atr_stop_mult: float = 2.0, **kwargs):
        config = StrategyConfig(
            name="VWAP Reversion",
            params={
                "vwap_period": vwap_period,
                "entry_deviation": entry_deviation,
                "atr_period": atr_period,
                "atr_stop_mult": atr_stop_mult,
            },
            **kwargs,
        )
        super().__init__(config)

    @classmethod
    def param_ranges(cls) -> dict[str, tuple]:
        return {
            "vwap_period": (10, 40, 1),
            "entry_deviation": (1.0, 4.0, 0.25),
            "atr_period": (7, 28, 1),
            "atr_stop_mult": (1.0, 4.0, 0.25),
        }

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.config.params
        df = df.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol = typical_price * df["volume"]
        period = p["vwap_period"]
        df["vwap"] = tp_vol.rolling(window=period).sum() / df["volume"].rolling(window=period).sum()
        df["vwap_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
        df["atr"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=p["atr_period"]
        ).average_true_range()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.config.params
        signals = pd.Series(Signal.HOLD, index=df.index)
        below_vwap = df["vwap_pct"] < -p["entry_deviation"]
        above_vwap = df["close"] >= df["vwap"]
        stop_hit = df["close"] < (df["vwap"] - p["atr_stop_mult"] * df["atr"])
        signals[below_vwap] = Signal.BUY
        signals[above_vwap | stop_hit] = Signal.SELL
        return signals


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "rsi_mean_reversion": RSIMeanReversion,
    "ema_crossover": EMACrossover,
    "bollinger_squeeze": BollingerBandSqueeze,
    "macd_histogram": MACDHistogramReversal,
    "stoch_rsi_bounce": StochasticRSIBounce,
    "triple_ema": TripleEMATrend,
    "adx_rsi": ADXRSITrend,
    "dual_momentum": DualMomentum,
    "williams_r": WilliamsRReversal,
    "parabolic_sar": ParabolicSARTrend,
    "ichimoku_breakout": IchimokuBreakout,
    "vwap_reversion": VWAPReversion,
}


def get_all_strategies(**kwargs) -> list[BaseStrategy]:
    """Instantiate all registered strategies with optional shared kwargs."""
    return [cls(**kwargs) for cls in STRATEGY_REGISTRY.values()]
