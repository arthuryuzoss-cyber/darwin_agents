"""Livro-caixa append-only em SQLite.

Regras de ouro:
- Dinheiro SEMPRE em centavos (int). Nunca float.
- Receita só conta se for CONFIRMADA (webhook assinado ou confirmação manual
  com referência única). Não existe receita "simulada" que alimente o orçamento.
- Nenhum registro é apagado ou editado: só se acrescentam eventos
  (triggers do SQLite bloqueiam UPDATE e DELETE).
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable


class LedgerError(Exception):
    pass


@dataclass(frozen=True)
class Entry:
    id: int
    ts: float
    agent: str
    kind: str
    amount_cents: int  # sempre positivo; o sinal vem do kind
    ref: str
    note: str


CREDIT_KINDS = {"seed", "revenue", "transfer_in", "trade_gain"}
DEBIT_KINDS = {"spend", "transfer_out", "payout", "refund", "trade_loss"}
ALL_KINDS = CREDIT_KINDS | DEBIT_KINDS


class Ledger:
    def __init__(self, path: str = ":memory:", clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                agent TEXT NOT NULL,
                kind TEXT NOT NULL,
                amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
                ref TEXT NOT NULL DEFAULT '',
                note TEXT NOT NULL DEFAULT ''
            )"""
        )
        # Mesma referência não pode ser registrada duas vezes para o mesmo tipo
        # (impede contar o mesmo pagamento duas vezes: replay de webhook).
        self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_kind_ref ON entries(kind, ref) WHERE ref != ''"
        )
        self._db.execute(
            """CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON entries
               BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END"""
        )
        self._db.execute(
            """CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON entries
               BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END"""
        )
        self._db.commit()

    # ------------------------------------------------------------------ util
    @staticmethod
    def _validate_amount(amount_cents: int) -> None:
        if isinstance(amount_cents, bool) or not isinstance(amount_cents, int):
            raise LedgerError("amount_cents deve ser int (centavos), nunca float")
        if amount_cents <= 0:
            raise LedgerError("amount_cents deve ser positivo")

    def _insert(self, agent: str, kind: str, amount_cents: int, ref: str, note: str) -> Entry:
        self._validate_amount(amount_cents)
        if kind not in ALL_KINDS:
            raise LedgerError(f"tipo inválido: {kind!r}")
        if not agent or not isinstance(agent, str):
            raise LedgerError("agent obrigatório")
        ts = self._clock()
        try:
            cur = self._db.execute(
                "INSERT INTO entries(ts, agent, kind, amount_cents, ref, note) VALUES (?,?,?,?,?,?)",
                (ts, agent, kind, amount_cents, ref, note),
            )
            self._db.commit()
        except sqlite3.IntegrityError as e:
            self._db.rollback()
            raise LedgerError(f"referência duplicada: {ref!r}") from e
        return Entry(cur.lastrowid, ts, agent, kind, amount_cents, ref, note)

    def _balance_unlocked(self, agent: str) -> int:
        cur = self._db.execute(
            "SELECT kind, SUM(amount_cents) FROM entries WHERE agent=? GROUP BY kind", (agent,)
        )
        total = 0
        for kind, s in cur.fetchall():
            total += s if kind in CREDIT_KINDS else -s
        return total

    # ------------------------------------------------------------------- API
    def balance(self, agent: str) -> int:
        with self._lock:
            return self._balance_unlocked(agent)

    def seed(self, agent: str, amount_cents: int, note: str = "capital inicial") -> Entry:
        with self._lock:
            return self._insert(agent, "seed", amount_cents, "", note)

    def spend(self, agent: str, amount_cents: int, note: str = "") -> Entry:
        """Registra um gasto. Falha se o saldo ficaria negativo."""
        self._validate_amount(amount_cents)
        with self._lock:
            if self._balance_unlocked(agent) < amount_cents:
                raise LedgerError("saldo insuficiente")
            return self._insert(agent, "spend", amount_cents, "", note)

    def confirm_revenue(self, agent: str, amount_cents: int, ref: str, note: str = "") -> Entry:
        """Registra receita CONFIRMADA. `ref` (id do pagamento) é obrigatório e único."""
        if not isinstance(ref, str) or not ref.strip():
            raise LedgerError("ref do pagamento é obrigatório para confirmar receita")
        with self._lock:
            return self._insert(agent, "revenue", amount_cents, ref.strip(), note)

    def refund(self, agent: str, amount_cents: int, ref: str, note: str = "") -> Entry:
        """Reembolso a cliente. Pode deixar o saldo negativo (dívida real);
        o disjuntor/falência tratam isso."""
        if not isinstance(ref, str) or not ref.strip():
            raise LedgerError("ref do reembolso é obrigatório")
        with self._lock:
            return self._insert(agent, "refund", amount_cents, ref.strip(), note)

    def transfer(self, from_agent: str, to_agent: str, amount_cents: int, note: str = "") -> None:
        """Move capital entre agentes (replicação). Atômico sob o lock."""
        self._validate_amount(amount_cents)
        if from_agent == to_agent:
            raise LedgerError("origem e destino iguais")
        with self._lock:
            if self._balance_unlocked(from_agent) < amount_cents:
                raise LedgerError("saldo insuficiente para transferência")
            self._insert(from_agent, "transfer_out", amount_cents, "", f"to:{to_agent} {note}")
            self._insert(to_agent, "transfer_in", amount_cents, "", f"from:{from_agent} {note}")

    def record_trade(self, agent: str, pnl_cents: int, ref: str, note: str = "") -> Entry | None:
        """Registra o PnL REALIZADO (líquido de taxas) de um trade fechado na exchange.
        `ref` é o id da ordem de venda: o mesmo id nunca é contado duas vezes.
        PnL zero não gera lançamento (valores do livro são sempre > 0)."""
        if isinstance(pnl_cents, bool) or not isinstance(pnl_cents, int):
            raise LedgerError("pnl_cents deve ser int")
        if not isinstance(ref, str) or not ref.strip():
            raise LedgerError("ref da ordem é obrigatório")
        if pnl_cents == 0:
            return None
        kind = "trade_gain" if pnl_cents > 0 else "trade_loss"
        with self._lock:
            return self._insert(agent, kind, abs(pnl_cents), ref.strip(), note)

    def payout(self, agent: str, amount_cents: int, approver: str, note: str = "") -> Entry:
        """Registra um SAQUE para o dono. Só com aprovador humano identificado.
        O dinheiro em si é movido pelo humano no app do banco/gateway."""
        if not approver or not approver.strip():
            raise LedgerError("saque exige aprovador humano")
        self._validate_amount(amount_cents)
        with self._lock:
            if self._balance_unlocked(agent) < amount_cents:
                raise LedgerError("saldo insuficiente para saque")
            return self._insert(agent, "payout", amount_cents, "", f"approved_by:{approver.strip()} {note}")

    def total(self, agent: str, kind: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(amount_cents),0) FROM entries WHERE agent=? AND kind=?",
                (agent, kind),
            ).fetchone()
            return int(row[0])

    def total_since(self, agent: str, kind: str, since_ts: float) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(amount_cents),0) FROM entries WHERE agent=? AND kind=? AND ts>=?",
                (agent, kind, since_ts),
            ).fetchone()
            return int(row[0])

    def global_total(self, kind: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(amount_cents),0) FROM entries WHERE kind=?", (kind,)
            ).fetchone()
            return int(row[0])

    def global_balance(self) -> int:
        """Capital total sob gestão (transferências internas se anulam)."""
        g = self.global_total
        return (g("seed") + g("revenue") + g("trade_gain")
                - g("spend") - g("payout") - g("refund") - g("trade_loss"))

    def entries(self, agent: str | None = None) -> list[Entry]:
        with self._lock:
            q = "SELECT id, ts, agent, kind, amount_cents, ref, note FROM entries"
            args: tuple = ()
            if agent:
                q += " WHERE agent=?"
                args = (agent,)
            q += " ORDER BY id"
            return [Entry(*r) for r in self._db.execute(q, args).fetchall()]

    def close(self) -> None:
        self._db.close()
