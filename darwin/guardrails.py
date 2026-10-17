"""Guardrails de execução: o que o código SE RECUSA a fazer, não importa a estratégia.

- Só opera o símbolo configurado (lista branca), apenas spot, apenas market buy/sell.
- Não existe no projeto nenhum código de saque, transferência, margem ou alavancagem.
- Compra limitada ao teto por ordem e a N compras por hora. Vendas nunca são limitadas
  por frequência (um stop-loss não pode ser bloqueado), mas respeitam a lista de símbolos.
- Manchetes de notícias são dados NÃO CONFIÁVEIS: sanitizadas antes de ir ao modelo.
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Callable


class GuardError(Exception):
    pass


@dataclass(frozen=True)
class OrderGuardConfig:
    allowed_symbols: frozenset
    max_buy_cents: int
    max_orders_per_hour: int = 12

    def __post_init__(self):
        if not self.allowed_symbols:
            raise ValueError("allowed_symbols vazio")
        if isinstance(self.max_buy_cents, bool) or not isinstance(self.max_buy_cents, int) or self.max_buy_cents <= 0:
            raise ValueError("max_buy_cents deve ser int > 0")
        if self.max_orders_per_hour < 1:
            raise ValueError("max_orders_per_hour deve ser >= 1")


class OrderGuard:
    def __init__(self, config: OrderGuardConfig, clock: Callable[[], float] = time.time):
        self.config = config
        self._clock = clock
        self._stamps: deque[float] = deque()

    def check(self, symbol: str, side: str, notional_cents: int, emergency: bool = False) -> None:
        cfg = self.config
        if symbol not in cfg.allowed_symbols:
            raise GuardError(f"símbolo não permitido: {symbol!r}")
        if side not in ("buy", "sell"):
            raise GuardError(f"lado inválido: {side!r}")
        if isinstance(notional_cents, bool) or not isinstance(notional_cents, int) or notional_cents <= 0:
            raise GuardError("valor da ordem inválido")
        if side == "buy" and notional_cents > cfg.max_buy_cents:
            raise GuardError("compra acima do teto por ordem")
        if side == "sell" or emergency:
            return  # vendas reduzem risco: nunca limitadas por frequência
        now = self._clock()
        while self._stamps and now - self._stamps[0] > 3600:
            self._stamps.popleft()
        if len(self._stamps) >= cfg.max_orders_per_hour:
            raise GuardError("limite de ordens por hora atingido")

    def register(self) -> None:
        self._stamps.append(self._clock())


_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_WS = re.compile(r"\s+")


def sanitize_headline(text: object, max_len: int = 200) -> str:
    """Remove controles, normaliza espaços, limita tamanho. Devolve '' se não for texto."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _CTRL.sub(" ", t)
    t = t.replace("<", " ").replace(">", " ").replace("`", "'")
    t = _WS.sub(" ", t).strip()
    return t[:max_len]
