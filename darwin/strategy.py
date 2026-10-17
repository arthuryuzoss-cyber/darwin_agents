"""Estratégia: análise técnica (EMA + RSI) cruzada com sentimento de notícias.

score_técnico   = 0.7 * tanh(diferença % entre EMA rápida e lenta) + 0.3 * score_RSI
score_final     = (1 - w) * score_técnico + w * sentimento        (w = sentiment_weight)
COMPRA  se está fora da posição e score_final >= buy_threshold
VENDA   se está na posição e (stop-loss | take-profit | score_final <= sell_threshold)

Filtro de tendência longa (opcional, `trend_period` > 0; 0 = desligado): só COMPRA com o preço acima da
média simples dos últimos `trend_period` fechamentos e VENDE se o preço cair abaixo dela. Ideia: ficar em caixa
quando o mercado está em tendência de baixa. Não é garantia de lucro: o backtest é que diz se ajudou.

Aviso honesto: isto é um ponto de partida didático. Não há evidência de que gere lucro
depois de taxas; é justamente isso que a seleção darwiniana testa (e custa dinheiro real).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Sequence

# (mínimo, máximo) de cada parâmetro: a mutação NUNCA sai destes limites.
BOUNDS: dict[str, tuple[float, float]] = {
    "fast": (3, 30),
    "slow": (10, 100),
    "rsi_period": (5, 30),
    "rsi_oversold": (15, 45),
    "rsi_overbought": (55, 85),
    "sentiment_weight": (0.0, 0.8),
    "buy_threshold": (0.10, 0.80),
    "sell_threshold": (-0.80, 0.0),
    "stop_loss_pct": (0.5, 5.0),
    "take_profit_pct": (0.5, 10.0),
    "position_fraction": (0.30, 0.98),
}
INT_PARAMS = {"fast", "slow", "rsi_period"}
TREND_MIN, TREND_MAX = 50, 400      # trend_period: 0 (desligado) ou um valor neste intervalo


class StrategyError(Exception):
    pass


@dataclass(frozen=True)
class StrategyParams:
    fast: int = 9
    slow: int = 21
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 70.0
    sentiment_weight: float = 0.3
    buy_threshold: float = 0.30
    sell_threshold: float = -0.20
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 3.0
    position_fraction: float = 0.95
    trend_period: int = 0

    def __post_init__(self):
        for name, (lo, hi) in BOUNDS.items():
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise StrategyError(f"{name} inválido")
            if not lo <= v <= hi:
                raise StrategyError(f"{name}={v} fora dos limites [{lo}, {hi}]")
        if self.fast >= self.slow:
            raise StrategyError("fast deve ser menor que slow")
        tp = self.trend_period
        if isinstance(tp, bool) or not isinstance(tp, int) or not (tp == 0 or TREND_MIN <= tp <= TREND_MAX):
            raise StrategyError(f"trend_period deve ser 0 (desligado) ou inteiro entre {TREND_MIN} e {TREND_MAX}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyParams":
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise StrategyError(f"parâmetros desconhecidos: {sorted(unknown)}")
        return cls(**d)

    @property
    def min_candles(self) -> int:
        return max(self.slow, self.rsi_period) + 2


def ema(values: Sequence[float], n: int) -> float:
    if n < 1 or len(values) < n:
        raise StrategyError("dados insuficientes para EMA")
    k = 2.0 / (n + 1)
    e = sum(values[:n]) / n
    for v in values[n:]:
        e = v * k + e * (1 - k)
    return e


def rsi(values: Sequence[float], n: int) -> float:
    if n < 1 or len(values) < n + 1:
        raise StrategyError("dados insuficientes para RSI")
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = values[i] - values[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_g, avg_l = gains / n, losses / n
    for i in range(n + 1, len(values)):
        d = values[i] - values[i - 1]
        avg_g = (avg_g * (n - 1) + max(d, 0.0)) / n
        avg_l = (avg_l * (n - 1) + max(-d, 0.0)) / n
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def sma(values: Sequence[float], n: int) -> float:
    if n < 1 or len(values) < n:
        raise StrategyError("dados insuficientes para a média")
    return sum(values[-n:]) / n


def tech_score(closes: Sequence[float], p: StrategyParams) -> float:
    if len(closes) < p.min_candles:
        raise StrategyError("candles insuficientes")
    f, s = ema(closes, p.fast), ema(closes, p.slow)
    trend = math.tanh((f - s) / s * 100.0)
    r = rsi(closes, p.rsi_period)
    if r < p.rsi_oversold:
        rsi_s = (p.rsi_oversold - r) / p.rsi_oversold
    elif r > p.rsi_overbought:
        rsi_s = -(r - p.rsi_overbought) / (100.0 - p.rsi_overbought)
    else:
        rsi_s = 0.0
    return max(-1.0, min(1.0, 0.7 * trend + 0.3 * rsi_s))


def combined_score(closes: Sequence[float], p: StrategyParams, sentiment: float | None) -> float:
    """sentiment=None => notícias indisponíveis: usa só a análise técnica (peso 0)."""
    t = tech_score(closes, p)
    if sentiment is None:
        return t
    sent = max(-1.0, min(1.0, float(sentiment)))
    return (1 - p.sentiment_weight) * t + p.sentiment_weight * sent


@dataclass(frozen=True)
class Decision:
    action: str   # "buy" | "sell" | "hold"
    reason: str
    score: float


def decide(p: StrategyParams, closes: Sequence[float], sentiment: float | None, price: float,
           in_position: bool, entry_price: float = 0.0) -> Decision:
    try:
        score = combined_score(closes, p, sentiment)
    except StrategyError:
        return Decision("hold", "dados_insuficientes", 0.0)
    trend_ok: bool | None = None          # None = filtro desligado ou sem dados suficientes
    if p.trend_period and len(closes) >= p.trend_period:
        trend_ok = price > sma(closes, p.trend_period)
    if in_position and entry_price > 0:
        change = (price - entry_price) / entry_price * 100.0
        if change <= -p.stop_loss_pct:
            return Decision("sell", "stop_loss", score)
        if change >= p.take_profit_pct:
            return Decision("sell", "take_profit", score)
    if in_position:
        if trend_ok is False:
            return Decision("sell", "filtro_de_tendencia", score)
        if score <= p.sell_threshold:
            return Decision("sell", "sinal_de_saida", score)
        return Decision("hold", "posicionado", score)
    if p.trend_period and trend_ok is None:
        return Decision("hold", "dados_insuficientes", score)     # sem histórico para o filtro: não compra
    if score >= p.buy_threshold:
        if trend_ok is False:
            return Decision("hold", "contra_tendencia", score)
        return Decision("buy", "sinal_de_entrada", score)
    return Decision("hold", "sem_sinal", score)
