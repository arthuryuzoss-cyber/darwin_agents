"""DISJUNTOR / FUSÍVEL de patrimônio.

- O patrimônio vem da API da exchange (verdade externa), não de contas internas.
- patrimônio <= piso (baseline * (100 - MAX_LOSS_PCT)%) => dispara IMEDIATAMENTE.
- Ao disparar: grava o estado em disco ANTES de agir (sobrevive a crash), depois chama os
  callbacks (liquidar posições, cancelar ordens). Se a liquidação falhar, o estado fica
  `liquidation_pending` e o próximo start tenta de novo antes de qualquer outra coisa.
- Falha repetida em ler o patrimônio também dispara (sem monitoramento = sem exposição).
- Só um humano reabre (token). Reabrir sem `rebase` volta a disparar se o patrimônio
  continuar abaixo do piso: não dá para "burlar" só resetando.
"""
from __future__ import annotations

import hmac
import json
import os
import threading
from enum import Enum
from typing import Callable

from .budget import fuse_floor_cents, loss_pct


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


class BreakerError(Exception):
    pass


class CircuitBreaker:
    def __init__(self, state_path: str | None = None, reset_token: str | None = None,
                 max_loss_pct: int = 60, max_consecutive_failures: int = 3):
        if not 1 <= max_loss_pct <= 99:
            raise ValueError("max_loss_pct deve estar entre 1 e 99")
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures deve ser >= 1")
        self.max_loss_pct = max_loss_pct
        self.max_consecutive_failures = max_consecutive_failures
        self._state_path = state_path
        self._reset_token = reset_token
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[str], None]] = []
        self._failures: dict[str, int] = {}
        self._state = State.CLOSED
        self.reason = ""
        self.liquidation_pending = False
        self.initial_equity_cents: int | None = None
        self._load()

    # ----------------------------------------------------------- persistência
    def _load(self) -> None:
        if not self._state_path or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.initial_equity_cents = d.get("initial_equity_cents")
            if d.get("tripped"):
                self._state = State.OPEN
                self.reason = str(d.get("reason", "disjuntor aberto"))
                self.liquidation_pending = not d.get("liquidated", False)
        except (OSError, ValueError):
            # arquivo corrompido => fail-safe: trata como aberto
            self._state = State.OPEN
            self.reason = "estado do disjuntor ilegível (fail-safe)"
            self.liquidation_pending = True

    def _save(self) -> None:
        if not self._state_path:
            return
        data = {"initial_equity_cents": self.initial_equity_cents, "tripped": self._state is State.OPEN,
                "reason": self.reason, "liquidated": not self.liquidation_pending}
        tmp = self._state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self._state_path)

    # -------------------------------------------------------------------- API
    @property
    def state(self) -> State:
        return self._state

    def is_open(self) -> bool:
        return self._state is State.OPEN

    def on_trip(self, callback: Callable[[str], None]) -> None:
        self._callbacks.append(callback)

    def set_initial_equity(self, cents: int) -> None:
        """Define a baseline UMA vez. Mudar exige reset com rebase."""
        if isinstance(cents, bool) or not isinstance(cents, int) or cents <= 0:
            raise BreakerError("patrimônio inicial inválido")
        with self._lock:
            if self.initial_equity_cents is not None and self.initial_equity_cents != cents:
                raise BreakerError("baseline já definida; use reset com rebase para alterá-la")
            self.initial_equity_cents = cents
            self._save()

    @property
    def floor_cents(self) -> int | None:
        if self.initial_equity_cents is None:
            return None
        return fuse_floor_cents(self.initial_equity_cents, self.max_loss_pct)

    def current_loss_pct(self, equity_cents: int) -> float | None:
        if self.initial_equity_cents is None:
            return None
        return loss_pct(self.initial_equity_cents, equity_cents)

    def trip(self, reason: str) -> None:
        with self._lock:
            if self._state is State.OPEN:
                return
            self._state = State.OPEN
            self.reason = reason
            self.liquidation_pending = True
            self._save()  # persiste ANTES de qualquer ação
        for cb in list(self._callbacks):
            try:
                cb(reason)
            except Exception:  # um callback quebrado não pode impedir os outros
                pass

    def check_equity(self, equity_cents: int) -> State:
        if isinstance(equity_cents, bool) or not isinstance(equity_cents, int) or equity_cents < 0:
            raise BreakerError("patrimônio inválido")
        floor = self.floor_cents
        if floor is None:
            raise BreakerError("baseline de patrimônio não definida")
        if equity_cents <= floor:
            self.trip(f"patrimônio R$ {equity_cents/100:.2f} <= piso R$ {floor/100:.2f} "
                      f"(perda de {self.current_loss_pct(equity_cents):.1f}% sobre R$ "
                      f"{self.initial_equity_cents/100:.2f})")
        return self._state

    def record_result(self, ok: bool, channel: str = "default") -> None:
        """Conta falhas consecutivas POR CANAL (patrimônio, mercado...). Um canal saudável não
        zera a contagem de outro. Sem monitoramento confiável, desarma."""
        if ok:
            self._failures[channel] = 0
            return
        n = self._failures.get(channel, 0) + 1
        self._failures[channel] = n
        if n >= self.max_consecutive_failures:
            self.trip(f"{n} falhas consecutivas ao monitorar a exchange ({channel})")

    def mark_liquidated(self) -> None:
        with self._lock:
            self.liquidation_pending = False
            self._save()

    def reset(self, token: str, rebase_equity_cents: int | None = None) -> bool:
        if not self._reset_token or not isinstance(token, str):
            return False
        if not hmac.compare_digest(token.encode(), self._reset_token.encode()):
            return False
        with self._lock:
            if self.liquidation_pending:
                return False  # primeiro termine a liquidação
            self._state = State.CLOSED
            self.reason = ""
            self._failures = {}
            if rebase_equity_cents is not None:
                if isinstance(rebase_equity_cents, bool) or not isinstance(rebase_equity_cents, int) \
                        or rebase_equity_cents <= 0:
                    return False
                self.initial_equity_cents = rebase_equity_cents
            self._save()
        return True
