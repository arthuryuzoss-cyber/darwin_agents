"""Seleção natural orientada a PnL REAL.

- Julgamento só sobre PnL realizado e líquido de taxas, vindo de fills reais da exchange
  e gravado no livro-caixa (`trade_gain` - `trade_loss` dentro da janela do ciclo).
- PnL > 0  -> o agente sobrevive e CLONA: o filho nasce com parâmetros mutados
              (limitados por BOUNDS) e é financiado com capital do pai.
- Ciclo SEM nenhum trade (mercado parado) não tem PnL medido: o agente segue vivo e reinicia o
  ciclo (IDLE_KILLS=true no .env restaura a regra literal "PnL 0 também morre").
- PnL <= 0 -> o agente MORRE (o orquestrador liquida a posição, devolve o capital
              ao tesouro e remove o agente do pool).
- Um "ciclo" só fecha com o agente fora da posição e com `eval_min_roundtrips` trades
  completos (ou ao atingir `eval_max_ticks`, quando a posição é encerrada à força).

Limites práticos com banca pequena (calculados, não opinião):
o filho precisa ter pelo menos a ordem mínima da exchange E o pai precisa continuar acima
dela depois da divisão; senão o agente sobrevive mas NÃO clona.

Aviso estatístico: com poucos trades por ciclo, "lucro" e "prejuízo" são dominados por
sorte e taxas. A regra implementada é a pedida (estrita), não uma garantia de evolução.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .budget import RiskBudget
from .ledger import Ledger
from .strategy import BOUNDS, INT_PARAMS, TREND_MAX, TREND_MIN, StrategyParams


@dataclass(frozen=True)
class EvolutionConfig:
    eval_min_roundtrips: int = 3
    eval_max_ticks: int = 400
    max_population: int = 4
    max_generation: int = 5
    mutation_sigma: float = 0.15
    sentiment_sigma: float = 0.10
    idle_kills: bool = False     # True = ciclo sem NENHUM trade também mata (PnL 0). Padrão: só sem evidência
    death_tolerance_pct: float = 0.0   # só morre se a perda do ciclo passar de X% do capital (0 = qualquer PnL <= 0 mata)
    needs_min_trades: bool = False     # True = só julga (e mata) depois de eval_min_roundtrips trades; senão renova o ciclo

    def __post_init__(self):
        if self.eval_min_roundtrips < 1 or self.eval_max_ticks < 1:
            raise ValueError("janela de avaliação inválida")
        if self.max_population < 1 or self.max_generation < 1:
            raise ValueError("limites de população/geração inválidos")
        if not 0 <= self.death_tolerance_pct <= 20:
            raise ValueError("death_tolerance_pct deve estar entre 0 e 20")
        if not 0 < self.mutation_sigma <= 1 or not 0 < self.sentiment_sigma <= 0.5:
            raise ValueError("sigma de mutação inválido")


def _clamp(v: float, name: str) -> float:
    lo, hi = BOUNDS[name]
    return max(lo, min(hi, v))


def mutate_params(p: StrategyParams, rng: random.Random, sigma: float = 0.15,
                  sentiment_sigma: float = 0.10) -> StrategyParams:
    """Mutação limitada: parâmetros de estratégia (multiplicativa) e peso de sentimento (aditiva)."""
    d = p.to_dict()
    for name in BOUNDS:
        if name == "sentiment_weight":
            d[name] = _clamp(d[name] + rng.gauss(0, sentiment_sigma), name)
        elif name in INT_PARAMS:
            d[name] = int(round(_clamp(d[name] * (1 + rng.gauss(0, sigma)), name)))
        else:
            d[name] = round(_clamp(d[name] * (1 + rng.gauss(0, sigma)), name), 4)
    if d["trend_period"]:     # 0 (desligado) continua desligado; ligado muta dentro de [TREND_MIN, TREND_MAX]
        d["trend_period"] = int(round(max(TREND_MIN, min(TREND_MAX, d["trend_period"] * (1 + rng.gauss(0, sigma))))))
    if d["fast"] >= d["slow"]:
        d["fast"] = max(int(BOUNDS["fast"][0]), d["slow"] - 1)
    d["sentiment_weight"] = round(d["sentiment_weight"], 4)
    if d == p.to_dict():  # garante que o filho seja realmente diferente
        step = 0.05 if d["sentiment_weight"] + 0.05 <= BOUNDS["sentiment_weight"][1] else -0.05
        d["sentiment_weight"] = round(d["sentiment_weight"] + step, 4)
    return StrategyParams.from_dict(d)


@dataclass(frozen=True)
class Verdict:
    action: str            # "continue" | "idle" | "die" | "clone" | "survive"
    pnl_cents: int
    reason: str
    child_seed_cents: int = 0


class Darwinism:
    def __init__(self, ledger: Ledger, config: EvolutionConfig, risk: RiskBudget,
                 rng: random.Random | None = None):
        self.ledger = ledger
        self.cfg = config
        self.risk = risk
        self.rng = rng or random.Random()

    def totals(self, agent: str) -> tuple[int, int]:
        return self.ledger.total(agent, "trade_gain"), self.ledger.total(agent, "trade_loss")

    def window_pnl_cents(self, agent: str, gain0: int, loss0: int) -> int:
        gain, loss = self.totals(agent)
        return (gain - gain0) - (loss - loss0)

    def window_closed(self, roundtrips: int, ticks: int) -> bool:
        return roundtrips >= self.cfg.eval_min_roundtrips or ticks >= self.cfg.eval_max_ticks

    def judge(self, agent: str, gain0: int, loss0: int, roundtrips: int, ticks: int,
              in_position: bool, population: int, generation: int, min_trade_cents: int) -> Verdict:
        if in_position or not self.window_closed(roundtrips, ticks):
            return Verdict("continue", 0, "ciclo em andamento")
        if roundtrips == 0 and not self.cfg.idle_kills:
            return Verdict("idle", 0, f"{ticks} ticks sem nenhum trade: sem evidência para julgar")
        pnl = self.window_pnl_cents(agent, gain0, loss0)
        capital = self.ledger.balance(agent)
        if pnl <= 0 and self.cfg.needs_min_trades and roundtrips < self.cfg.eval_min_roundtrips:
            return Verdict("idle", pnl, f"só {roundtrips} trade(s) de {self.cfg.eval_min_roundtrips} necessários: "
                                        "sem evidência suficiente para julgar")
        if pnl <= 0:
            tol = int(max(0, capital - pnl) * self.cfg.death_tolerance_pct / 100)   # capital no início do ciclo ~ capital + perda
            if pnl < -tol or (tol == 0):
                return Verdict("die", pnl, f"PnL líquido do ciclo R$ {pnl/100:.2f} <= "
                                           + ("0" if tol == 0 else f"-{tol/100:.2f} (tolerância)"))
            return Verdict("survive", pnl, f"perda de R$ {-pnl/100:.2f} dentro da tolerância de R$ {tol/100:.2f}")
        if generation >= self.cfg.max_generation:
            return Verdict("survive", pnl, "geração máxima atingida")
        if population >= self.cfg.max_population:
            return Verdict("survive", pnl, "população máxima atingida")
        need = self.risk.viable_capital_cents(min_trade_cents)
        seed = min(max(pnl, need), capital - need)
        if not self.risk.can_fund_child(capital, seed, min_trade_cents):
            return Verdict("survive", pnl, "capital insuficiente para um filho viável (mínimo da exchange)")
        return Verdict("clone", pnl, f"lucro líquido R$ {pnl/100:.2f}", child_seed_cents=seed)

    def child_params(self, parent: StrategyParams) -> StrategyParams:
        return mutate_params(parent, self.rng, self.cfg.mutation_sigma, self.cfg.sentiment_sigma)
