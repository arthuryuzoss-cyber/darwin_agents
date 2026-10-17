"""Gestão de risco/orçamento (tudo em centavos, inteiros).

Aqui mora a regra do fusível: perdeu `max_loss_pct`% do patrimônio inicial => para.
Com banca de R$ 20,00 e 60%: piso = R$ 8,00 (patrimônio <= R$ 8,00 dispara).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _check_int(name: str, v, lo: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f"{name} deve ser int >= {lo}")
    return v


def fuse_floor_cents(initial_cents: int, max_loss_pct: int) -> int:
    """Piso do fusível, arredondado PARA CIMA (dispara um pouco antes, nunca depois)."""
    _check_int("initial_cents", initial_cents, 1)
    if isinstance(max_loss_pct, bool) or not isinstance(max_loss_pct, int) or not 1 <= max_loss_pct <= 99:
        raise ValueError("max_loss_pct deve estar entre 1 e 99")
    return -(-initial_cents * (100 - max_loss_pct) // 100)


def loss_pct(initial_cents: int, equity_cents: int) -> float:
    _check_int("initial_cents", initial_cents, 1)
    return (initial_cents - equity_cents) * 100.0 / initial_cents


@dataclass(frozen=True)
class RiskConfig:
    bankroll_cents: int
    max_loss_pct: int = 60
    position_fraction_cap: float = 0.98   # deixa folga para a taxa na compra
    min_notional_margin: float = 1.20     # ordem mínima com 20% de folga: um stop de até 5% ainda fica vendável

    def __post_init__(self):
        _check_int("bankroll_cents", self.bankroll_cents, 1)
        if not 1 <= self.max_loss_pct <= 99:
            raise ValueError("max_loss_pct deve estar entre 1 e 99")
        if not 0 < self.position_fraction_cap <= 1:
            raise ValueError("position_fraction_cap deve estar em (0, 1]")
        if self.min_notional_margin < 1:
            raise ValueError("min_notional_margin deve ser >= 1")


class RiskBudget:
    def __init__(self, config: RiskConfig):
        self.config = config

    def min_trade_cents(self, exchange_min_notional: float) -> int:
        """Menor ordem que vale tentar (mínimo da exchange + folga), em centavos."""
        if not math.isfinite(exchange_min_notional) or exchange_min_notional < 0:
            raise ValueError("mínimo da exchange inválido")
        return math.ceil(exchange_min_notional * self.config.min_notional_margin * 100)

    def viable_capital_cents(self, min_trade_cents: int) -> int:
        """Capital mínimo para um agente conseguir fazer UMA compra: a ordem mínima precisa caber
        dentro do teto de fração por posição (deixa folga para a taxa)."""
        return -(-min_trade_cents * 100 // int(self.config.position_fraction_cap * 100))

    def size_buy_cents(self, free_cents: int, min_trade_cents: int, fraction: float) -> int:
        """Quanto comprar agora. Compra a fração pedida, mas nunca abaixo da ordem mínima da exchange
        se o capital permitir (agente pequeno compra o mínimo). Retorna 0 se nem isso cabe."""
        _check_int("free_cents", free_cents, 0)
        frac = min(max(float(fraction), 0.0), self.config.position_fraction_cap)
        ceiling = math.floor(free_cents * self.config.position_fraction_cap)
        notional = min(max(math.floor(free_cents * frac), min_trade_cents), ceiling)
        return notional if notional >= min_trade_cents and notional > 0 else 0

    def can_fund_child(self, parent_capital_cents: int, child_seed_cents: int, min_trade_cents: int) -> bool:
        """Pai e filho precisam, ambos, conseguir comprar depois da divisão."""
        need = self.viable_capital_cents(min_trade_cents)
        return child_seed_cents >= need and parent_capital_cents - child_seed_cents >= need
