"""Orquestrador: executa ordens reais (via CcxtGateway), mantém o fusível e a seleção natural.

Ordem de cada tick:
 1. fusível: lê o patrimônio NA API; se <= piso, liquida tudo e encerra;
 2. dados de mercado de TODOS os ativos (preço, candles) e sentimento (cache, geral e por ativo);
 3. cada agente decide (técnica x sentimento) e executa ordens a mercado: agente fora da
    posição varre todos os ativos e entra no de maior nota (se ≥ limiar); posicionado só
    acompanha o ativo em que está;
    o fusível é rechecado após CADA ordem;
 4. avaliação darwiniana dos agentes cujo ciclo fechou: PnL real > 0 clona, <= 0 morre;
 5. estado salvo em disco (atômico).

Erro ambíguo ao enviar ordem (timeout etc.) dispara o fusível: não se sabe se a ordem
entrou, então o sistema liquida e para, em vez de arriscar posição duplicada.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .breaker import CircuitBreaker, State
from .budget import RiskBudget
from .config import Config
from .evolution import Darwinism
from .exchange import CcxtGateway, ExchangeError, Fill, to_cents
from .guardrails import GuardError, OrderGuard
from .ledger import Ledger
from .strategy import StrategyParams, decide

TREASURY = "_treasury"
SEED_NAMES = ("alpha", "beta", "gamma", "delta")

EXIT_OK, EXIT_FUSE, EXIT_EXTINCT = 0, 3, 4


class OrchestratorError(Exception):
    pass


class AccountNotClean(OrchestratorError):
    pass


@dataclass
class Trader:
    name: str
    params: StrategyParams
    generation: int = 1
    parent: str | None = None
    qty: float = 0.0           # quantidade líquida do ativo base na posição aberta
    cost_cents: int = 0        # custo (com taxas) da posição aberta
    entry_price: float = 0.0
    window_ticks: int = 0
    window_roundtrips: int = 0
    gain0: int = 0             # totais do livro no início da janela
    loss0: int = 0
    pending_death: str = ""
    symbol: str = ""           # ativo da posição aberta ("" = fora da posição / estado antigo)

    @property
    def in_position(self) -> bool:
        return self.qty > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["params"] = self.params.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trader":
        d = dict(d)
        d["params"] = StrategyParams.from_dict(d["params"])
        return cls(**d)


@dataclass
class TickReport:
    tick: int
    equity_cents: int | None = None
    halted: bool = False
    halt_reason: str = ""
    events: list = field(default_factory=list)


class Orchestrator:
    def __init__(self, gateway: CcxtGateway, ledger: Ledger, breaker: CircuitBreaker, risk: RiskBudget,
                 guard: OrderGuard, darwin: Darwinism, sentiment: Any, cfg: Config,
                 state_path: str | None = None, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = print, clock: Callable[[], float] | None = None):
        self._clock = clock            # relógio (segundos) p/ detectar dado desatualizado; None = verificação desligada
        self.gw, self.ledger, self.breaker, self.risk = gateway, ledger, breaker, risk
        self.guard, self.darwin, self.sentiment, self.cfg = guard, darwin, sentiment, cfg
        self.state_path = state_path
        self._sleep = sleep
        self._log = log
        self.traders: dict[str, Trader] = {}
        self.graveyard: list[dict] = []
        self.tick_no = 0
        self.stop_reason = ""
        self.last_decision: dict = {}
        self.trade_log: list[dict] = []   # trades fechados (para relatórios/backtest)
        self.last_pick: dict = {}      # agente -> ativo avaliado/escolhido
        self.last_scan: dict = {}      # agente -> {ativo: nota}
        self.journal: Any = None       # diário de sinais opcional (nunca interfere na operação)
        q = getattr(gateway, "quote", "BRL")
        self.cur = "R$" if q == "BRL" else q
        self.breaker.on_trip(self._on_trip)

    # ----------------------------------------------------------------- helpers
    def log(self, msg: str, rep: TickReport | None = None) -> None:
        if rep is not None:
            rep.events.append(msg)
        self._log(msg)

    def min_trade_cents(self, symbol: str | None = None) -> int:
        """Ordem mínima (com folga). Sem símbolo: a do ativo mais barato de operar, usada para
        decidir se um agente tem capital para operar ALGO (nascer, clonar)."""
        mn = self.gw.min_notional_any() if symbol is None else self.gw.min_notional(symbol)
        return self.risk.min_trade_cents(mn)

    def _sym(self, tr: "Trader") -> str:
        return tr.symbol or self.gw.symbol

    def _negligible_cents(self) -> int:
        """Resíduo que não vale a pena (e não dá) vender: 1% da baseline, no mínimo R$ 0,10."""
        return max(10, (self.breaker.initial_equity_cents or 0) // 100)

    def fuse_feasibility(self, initial_cents: int | None = None) -> str | None:
        """Avisa se, no piso do fusível, a posição já estaria abaixo da ordem mínima da exchange
        (então a liquidação automática seria impossível por regra da própria exchange)."""
        initial = initial_cents or self.breaker.initial_equity_cents
        if not initial:
            return None
        from .budget import fuse_floor_cents
        floor = fuse_floor_cents(initial, self.cfg.max_loss_pct)
        min_cents = to_cents(self.gw.max_notional_any())   # o pior caso entre os ativos
        if floor >= min_cents * 1.5:
            return None
        need = -(-min_cents * 150 // (100 - self.cfg.max_loss_pct))
        return (f"ATENÇÃO: o piso do fusível ({self.cur} {floor/100:.2f}) é menor que 1,5x a ordem mínima da exchange "
                f"({self.cur} {min_cents/100:.2f}). Se o fusível disparar com posição aberta, a venda automática pode ser "
                f"IMPOSSÍVEL (a exchange recusa ordens abaixo do mínimo). A proteção real, nesse caso, é o stop-loss "
                f"por trade e a saída preventiva por liquidez. Para o fusível conseguir liquidar, a banca deveria ser "
                f"de ao menos {self.cur} {need/100:.2f}.")

    def alive(self) -> list[Trader]:
        return [self.traders[k] for k in sorted(self.traders)]

    # --------------------------------------------------------------- bootstrap
    def bootstrap(self, n_agents: int, params: StrategyParams | None = None) -> list[Trader]:
        if self.traders or self.ledger.entries():
            raise OrchestratorError("já inicializado; arquive a pasta data/ para recomeçar")
        if self.breaker.is_open():
            raise OrchestratorError("disjuntor aberto")
        if self.gw.foreign_base_value_cents() > 100:  # mais de 1,00 em algum ativo operado já na conta
            raise AccountNotClean("a conta já tem saldo de ativos que o robô opera; use uma subconta dedicada ao robô")
        equity = self.gw.equity_cents()
        if equity > int(self.cfg.bankroll_cents * 1.2):
            raise AccountNotClean(
                f"a conta tem {self.cur} {equity/100:.2f} em {self.gw.quote}, acima da banca de "
                f"{self.cur} {self.cfg.bankroll_cents/100:.2f} (+20%). O fusível mede o patrimônio TOTAL da conta; "
                "use uma subconta/conta dedicada com apenas a banca.")
        min_trade = self.risk.viable_capital_cents(self.min_trade_cents())
        if equity < min_trade:
            raise OrchestratorError(
                f"banca de {self.cur} {equity/100:.2f} menor que a ordem mínima da exchange "
                f"({self.cur} {min_trade/100:.2f}); nenhum agente conseguiria operar")
        n = max(1, min(n_agents, len(SEED_NAMES), equity // min_trade))
        self.breaker.set_initial_equity(equity)
        share, extra = divmod(equity, n)
        for i in range(n):
            amount = share + (extra if i == 0 else 0)
            name = SEED_NAMES[i]
            self.ledger.seed(name, amount)
            self.traders[name] = Trader(name, params or StrategyParams())
        self.save_state()
        return self.alive()

    # ---------------------------------------------------------------- fusível
    def _check_fuse(self, rep: TickReport | None = None) -> bool | None:
        """Lê o patrimônio na API e confere o fusível.
        True = disparado | False = ok | None = não consegui ler (não opere às cegas)."""
        try:
            eq = self.gw.equity_cents()
        except Exception as e:
            self.breaker.record_result(False, "patrimônio")
            self.log(f"⚠️ falha ao ler patrimônio ({type(e).__name__})", rep)
            return True if self.breaker.is_open() else None
        self.breaker.record_result(True, "patrimônio")
        if rep is not None:
            rep.equity_cents = eq
        return self.breaker.check_equity(eq) is State.OPEN

    def _on_trip(self, reason: str) -> None:
        self.log(f"🔌 DISJUNTOR DISPARADO: {reason}")
        self.stop_reason = "fuse"
        self.liquidate_all(reason)

    def liquidate_all(self, reason: str = "manual") -> bool:
        """Cancela ordens pendentes (todos os pares) e vende TODO ativo base da conta. 3 tentativas por par."""
        ok = True
        for attempt in range(3):
            try:
                n = self.gw.cancel_all_orders()
                if n:
                    self.log(f"   {n} ordem(ns) pendente(s) cancelada(s)")
                ok = True
                break
            except Exception as e:
                ok = False
                self.log(f"   falha ao cancelar ordens ({type(e).__name__}); tentativa {attempt + 1}/3")
                self._sleep(1)
        cleared: set[str] = set()
        for sym in self.gw.symbols:
            if self._liquidate_symbol(sym):
                cleared.add(sym)
        ok = ok and len(cleared) == len(self.gw.symbols)
        for tr in self.traders.values():
            if self._sym(tr) in cleared:
                tr.qty, tr.cost_cents, tr.entry_price, tr.symbol = 0.0, 0, 0.0, ""
        if ok:
            self.breaker.mark_liquidated()
            self.log("   ✅ liquidação concluída; execução encerrada para proteger o capital restante")
        else:
            self.log("   ❌ LIQUIDAÇÃO INCOMPLETA: rode `python main.py liquidate` ou venda manualmente na Binance")
        self.save_state()
        return ok

    def _liquidate_symbol(self, sym: str) -> bool:
        """Vende todo o ativo base de UM par. True = nada vendável restou (vendido ou poeira)."""
        base = self.gw.for_symbol(sym).base
        for attempt in range(3):
            try:
                free = self.gw.base_free(sym)
            except Exception as e:
                self.log(f"   falha ao ler saldo de {base} ({type(e).__name__}); tentativa {attempt + 1}/3")
                self._sleep(1)
                continue
            if free <= 0:
                return True
            try:
                price = self.gw.last_price(sym)
                self.guard.check(sym, "sell", max(1, to_cents(free * price)), emergency=True)
                fill = self.gw.market_sell(free, sym)
            except ExchangeError as e:
                if e.below_min:
                    left = to_cents(free * self.gw.last_price(sym))
                    if left <= self._negligible_cents():
                        self.log(f"   resíduo desprezível não vendável (poeira): {free:.8f} {base} ≈ {self.cur} {left/100:.2f}")
                        return True
                    self.log(f"   ❌ NÃO DÁ PARA VENDER POR API: a posição em {sym} ({free:.8f} {base} ≈ {self.cur} {left/100:.2f}) "
                             f"está abaixo da ordem mínima da exchange ({self.cur} {self.gw.min_notional(sym):.2f}). "
                             "Venda/converta manualmente no app da Binance e depois rode `python main.py liquidate` "
                             "para confirmar.")
                    return False
                self.log(f"   falha ao vender {base} ({e}); tentativa {attempt + 1}/3")
                self._sleep(1)
                continue
            except Exception as e:
                self.log(f"   falha ao vender {base} ({type(e).__name__}); tentativa {attempt + 1}/3")
                self._sleep(1)
                continue
            self._attribute_liquidation(sym, fill)
            self.log(f"   posição liquidada: {fill.base_qty:.8f} {base} a {self.cur} {fill.price:.2f}")
            return True
        return False

    def _attribute_liquidation(self, sym: str, fill: Fill) -> None:
        """Reparte a receita da liquidação do ativo entre os agentes que o seguravam, pela quantidade."""
        holders = [t for t in self.alive() if t.qty > 0 and self._sym(t) == sym]
        total_qty = sum(t.qty for t in holders)
        if not holders or total_qty <= 0:
            return
        proceeds = to_cents(fill.sell_proceeds)
        left = proceeds
        for i, t in enumerate(holders):
            share = left if i == len(holders) - 1 else int(proceeds * (t.qty / total_qty))
            left -= share
            self.ledger.record_trade(t.name, share - t.cost_cents, f"{fill.order_id}:{t.name}", "liquidação")
            t.window_roundtrips += 1

    def recover_if_needed(self) -> bool:
        """No start: se o fusível já estava aberto com liquidação pendente, termina o serviço."""
        if self.breaker.is_open() and self.breaker.liquidation_pending:
            self.log(f"🔌 disjuntor aberto desde a última execução ({self.breaker.reason}); liquidando…")
            self.stop_reason = "fuse"
            self.liquidate_all("retomada")
        return self.breaker.is_open()

    # ------------------------------------------------------------------ ordens
    def _open_position(self, tr: Trader, sym: str, size_cents: int, rep: TickReport) -> None:
        self.guard.check(sym, "buy", size_cents)
        fill = self.gw.market_buy(size_cents / 100, sym)
        self.guard.register()
        tr.symbol = sym
        tr.qty = fill.net_base_qty
        tr.cost_cents = to_cents(fill.buy_cost)
        tr.entry_price = fill.price
        base = self.gw.for_symbol(sym).base
        self.log(f"🟢 {tr.name}: COMPRA {fill.net_base_qty:.8f} {base} ({sym}) a {self.cur} {fill.price:.2f} "
                 f"(custo {self.cur} {tr.cost_cents/100:.2f}{' ~taxa estimada' if fill.fee_estimated else ''})", rep)

    def _close_position(self, tr: Trader, why: str, rep: TickReport | None, emergency: bool = False) -> int:
        """Vende a posição do agente. Devolve o PnL realizado em centavos (já gravado no livro)."""
        sym = self._sym(tr)
        free = self.gw.base_free(sym)
        qty = min(tr.qty, free)
        if qty <= 0:
            raise ExchangeError("sem saldo do ativo base para vender")
        price = self.gw.last_price(sym)
        self.guard.check(sym, "sell", max(1, to_cents(qty * price)), emergency=emergency)
        fill = self.gw.market_sell(qty, sym)
        pnl = to_cents(fill.sell_proceeds) - tr.cost_cents
        self.ledger.record_trade(tr.name, pnl, fill.order_id, why)
        self.trade_log.append({"agent": tr.name, "symbol": sym, "pnl_cents": pnl, "why": why, "tick": self.tick_no,
                               "cost_cents": tr.cost_cents, "entry": tr.entry_price, "exit": fill.price})
        tr.qty, tr.cost_cents, tr.entry_price, tr.symbol = 0.0, 0, 0.0, ""
        tr.window_roundtrips += 1
        self.log(f"🔴 {tr.name}: VENDA {sym} ({why}) a {self.cur} {fill.price:.2f} | PnL líquido {self.cur} {pnl/100:+.2f}", rep)
        return pnl

    # ------------------------------------------------------------------- mercado
    def _candle_limit(self) -> int:
        """Quantos candles ler por ativo: 100, ou mais se algum agente usa filtro de tendência longa."""
        need = [t.params.trend_period + 5 for t in self.traders.values() if t.params.trend_period]
        return min(1000, max(100, *need)) if need else 100

    def _snapshot(self, want: list[str]) -> tuple[dict, dict]:
        """{símbolo: (preço, fechamentos)} dos ativos pedidos e {símbolo: erro} dos que falharam."""
        snap: dict = {}
        errors: dict = {}
        limit = self._candle_limit()
        for sym in want:
            try:
                price = self.gw.last_price(sym)
                rows = self.gw.candles(self.cfg.timeframe, limit, sym)
                self._check_fresh(rows)
                closes = [c[4] for c in rows]
                snap[sym] = (price, closes)
            except Exception as e:
                errors[sym] = type(e).__name__
        return snap, errors

    def _check_fresh(self, rows: list) -> None:
        """Dado desatualizado (candle mais novo muito velho) não vira sinal: o ativo é tratado como sem dados."""
        if self._clock is None or not rows:
            return
        from .history import TF_MS
        tf = TF_MS.get(self.cfg.timeframe)
        if not tf:
            return
        age = self._clock() * 1000 - float(rows[-1][0])
        if age > tf * (1 + self.cfg.max_stale_candles):
            raise ExchangeError(f"dados desatualizados ({age / 60000:.0f} min desde o último candle)")

    @staticmethod
    def _sentiment_for(reading: Any, sym: str) -> float | None:
        if reading is None:
            return None
        base = sym.split("/")[0]
        try:
            vf = getattr(reading, "value_for", None)
            return vf(base) if vf else reading.value
        except Exception:
            return None

    # ------------------------------------------------------------------- ticks
    def tick(self) -> TickReport:
        rep = TickReport(self.tick_no)
        if self.breaker.is_open():
            rep.halted, rep.halt_reason = True, self.breaker.reason
            return rep
        fuse = self._check_fuse(rep)
        if fuse is None:  # patrimônio ilegível neste tick: não opera
            return rep
        if fuse:
            rep.halted, rep.halt_reason = True, self.breaker.reason
            return rep

        held = sorted({self._sym(t) for t in self.alive() if t.in_position})
        any_flat = any(not t.in_position for t in self.alive())
        want = list(self.gw.symbols) if any_flat else held
        snap, errors = self._snapshot(want)
        # Falha de mercado = não enxergo algum ativo em que tenho posição, ou não enxergo nada.
        if want and (any(h in errors for h in held) or not snap):
            self.breaker.record_result(False, "mercado")
            self.log(f"⚠️ falha ao ler mercado ({', '.join(f'{k}: {v}' for k, v in errors.items())})", rep)
            rep.halted = self.breaker.is_open()
            rep.halt_reason = self.breaker.reason
            return rep
        self.breaker.record_result(True, "mercado")
        if errors:
            self.log(f"⚠️ sem dados neste tick: {', '.join(errors)} (ignorados)", rep)
        try:
            reading = self.sentiment.get() if self.sentiment else None
        except Exception:
            reading = None
        sents = {sym: self._sentiment_for(reading, sym) for sym in snap}
        if self.journal is not None:
            try:
                self.journal.record(snap, sents)
            except Exception:
                pass

        for tr in self.alive():
            if tr.pending_death:
                self._kill(tr, tr.pending_death, rep)
                continue
            tr.window_ticks += 1
            try:
                self._step(tr, snap, sents, rep)
            except ExchangeError as e:
                if e.ambiguous:
                    self.breaker.trip(f"erro ambíguo ao enviar ordem ({e})")
                else:
                    self.log(f"⚠️ {tr.name}: ordem recusada ({e})", rep)
            except GuardError as e:
                self.log(f"🛡️ {tr.name}: bloqueado pelo guardrail ({e})", rep)
            if self.breaker.is_open():
                rep.halted, rep.halt_reason = True, self.breaker.reason
                return rep
            fuse = self._check_fuse(rep)
            if fuse is None:  # não consegue verificar: para de operar neste tick
                return rep
            if fuse:
                rep.halted, rep.halt_reason = True, self.breaker.reason
                return rep

        parts = []
        for n, d in sorted(self.last_decision.items()):
            if n in self.traders:
                parts.append(f"{n}: {self.last_pick.get(n, '')} {d.action}/{d.reason} ({d.score:+.2f})".replace("  ", " "))
        sent_vals = [v for v in sents.values() if v is not None]
        sent_txt = "indisponível" if not sent_vals else f"{sum(sent_vals) / len(sent_vals):+.2f}"
        self.log(f"· tick {self.tick_no} | patrimônio {self.cur} {(rep.equity_cents or 0) / 100:.2f}"
                 f" | sentimento {sent_txt} | " + " ; ".join(parts), rep)
        first = next((n for n in sorted(self.last_scan) if n in self.traders and not self.traders[n].in_position), None)
        if first:
            scan = " ".join(f"{k.split('/')[0]} {v:+.2f}" for k, v in self.last_scan[first].items())
            self.log(f"    varredura ({first}): {scan}", rep)
        self._evaluate(rep)
        if not self.traders and not self.stop_reason and self._revive(rep):
            pass
        elif not self.traders and not self.stop_reason:
            self.stop_reason = "extinct"
            self.log("☠️ EXTINÇÃO: nenhum agente restou. Capital restante permanece na conta.", rep)
        self.tick_no += 1
        self.save_state()
        return rep

    def _step(self, tr: Trader, snap: dict, sents: dict, rep: TickReport) -> None:
        if tr.in_position:
            sym = self._sym(tr)
            price, closes = snap[sym]
            if tr.qty * price * 100 < self.gw.min_notional(sym) * 115:
                # saída preventiva: a posição está perto de ficar abaixo da ordem mínima (inviável de vender)
                self._close_position(tr, "piso_de_liquidez", rep)
                return
            d = decide(tr.params, closes, sents.get(sym), price, True, tr.entry_price)
            self.last_decision[tr.name], self.last_pick[tr.name] = d, sym
            self.last_scan.pop(tr.name, None)
            if d.action == "sell":
                self._close_position(tr, d.reason, rep)
            return
        # fora da posição: varre todos os ativos e escolhe o de maior nota
        scan: dict[str, float] = {}
        cands: list[tuple[float, str]] = []
        best: tuple[str, Any] | None = None
        for sym, (price, closes) in snap.items():
            d = decide(tr.params, closes, sents.get(sym), price, False, 0.0)
            scan[sym] = d.score
            if best is None or d.score > best[1].score:
                best = (sym, d)
            if d.action == "buy":
                cands.append((d.score, sym))
        self.last_scan[tr.name] = scan
        if best is None:
            return
        self.last_decision[tr.name], self.last_pick[tr.name] = best[1], best[0]
        free_cents = self.ledger.balance(tr.name) - tr.cost_cents
        for _, sym in sorted(cands, reverse=True):
            size = self.risk.size_buy_cents(free_cents, self.min_trade_cents(sym), tr.params.position_fraction)
            if size > 0:
                self._open_position(tr, sym, size, rep)
                return
        if cands:
            self.log(f"… {tr.name}: sinal de compra, mas capital abaixo da ordem mínima", rep)

    # ---------------------------------------------------------------- darwinismo
    def _evaluate(self, rep: TickReport) -> None:
        for tr in self.alive():
            if tr.name not in self.traders or tr.pending_death:
                continue
            if tr.in_position and tr.window_ticks >= self.darwin.cfg.eval_max_ticks:
                try:
                    self._close_position(tr, "fim_do_ciclo", rep)
                except ExchangeError as e:
                    if e.ambiguous:
                        self.breaker.trip(f"erro ambíguo ao encerrar posição ({e})")
                        return
                    self.log(f"⚠️ {tr.name}: não consegui encerrar no fim do ciclo ({e})", rep)
                    continue
            v = self.darwin.judge(tr.name, tr.gain0, tr.loss0, tr.window_roundtrips, tr.window_ticks,
                                  tr.in_position, len(self.traders), tr.generation, self.min_trade_cents())
            if v.action == "continue":
                continue
            if v.action == "idle":
                self.log(f"💤 {tr.name}: {v.reason}; segue vivo e abre novo ciclo", rep)
                self._new_window(tr)
                continue
            if v.action == "die":
                self._kill(tr, v.reason, rep)
            elif v.action == "clone":
                self._clone(tr, v.child_seed_cents, v.pnl_cents, rep)
                self._new_window(tr)
            else:
                self.log(f"✅ {tr.name} lucrou {self.cur} {v.pnl_cents/100:+.2f}, sobrevive sem clonar ({v.reason})", rep)
                self._new_window(tr)

    def _revive(self, rep: TickReport) -> bool:
        """Último agente morreu: em vez de encerrar tudo, nasce um novo (parâmetros do morto, mutados) com o capital
        restante, se ele for viável. O fusível continua sendo a trava real de perda; isto só evita que UM ciclo ruim
        de UM agente encerre o sistema. REVIVE=false restaura a regra literal (extinção)."""
        if not self.cfg.revive or self.breaker.is_open() or not self.graveyard:
            return False
        bal = self.ledger.balance(TREASURY)
        if bal < self.risk.viable_capital_cents(self.min_trade_cents()):
            return False
        last = self.graveyard[-1]
        k = 1 + sum(1 for g in self.graveyard if g["name"].startswith("fenix"))
        name = f"fenix{k}"
        while name in self.traders or any(g["name"] == name for g in self.graveyard):
            k += 1
            name = f"fenix{k}"
        params = self.darwin.child_params(StrategyParams.from_dict(last["params"]))
        self.ledger.transfer(TREASURY, name, bal, f"renascimento de {last['name']}")
        t = Trader(name, params, last["generation"] + 1, None)
        t.gain0, t.loss0 = self.darwin.totals(name)
        self.traders[name] = t
        self.log(f"🔥 {last['name']} era o último agente: {name} renasce (gen {t.generation}) com {self.cur} "
                 f"{bal/100:.2f} e parâmetros mutados. (REVIVE=false desliga)", rep)
        return True

    def _new_window(self, tr: Trader) -> None:
        tr.gain0, tr.loss0 = self.darwin.totals(tr.name)
        tr.window_ticks = tr.window_roundtrips = 0

    def _clone(self, parent: Trader, seed_cents: int, pnl_cents: int, rep: TickReport) -> None:
        n = sum(1 for g in self.graveyard if g.get("parent") == parent.name) + \
            sum(1 for t in self.traders.values() if t.parent == parent.name) + 1
        name = f"{parent.name}.{n}"
        self.ledger.transfer(parent.name, name, seed_cents, "clonagem")
        child = Trader(name, self.darwin.child_params(parent.params), parent.generation + 1, parent.name)
        child.gain0, child.loss0 = self.darwin.totals(name)
        self.traders[name] = child
        self.log(f"🧬 {parent.name} lucrou {self.cur} {pnl_cents/100:+.2f} e gerou {name} (gen {child.generation}) "
                 f"com {self.cur} {seed_cents/100:.2f} | sentimento {parent.params.sentiment_weight:.2f}"
                 f" -> {child.params.sentiment_weight:.2f}", rep)

    def _kill(self, tr: Trader, reason: str, rep: TickReport | None) -> None:
        if tr.in_position:
            try:
                self._close_position(tr, "morte", rep)
            except ExchangeError as e:
                if e.ambiguous:
                    self.breaker.trip(f"erro ambíguo ao encerrar posição ({e})")
                    return
                if e.below_min and to_cents(tr.qty * self.gw.last_price(self._sym(tr))) <= self._negligible_cents():
                    self.ledger.record_trade(tr.name, -tr.cost_cents, f"writeoff:{tr.name}:{self.tick_no}", "poeira")
                    tr.qty, tr.cost_cents, tr.symbol = 0.0, 0, ""
                else:
                    tr.pending_death = reason
                    self.log(f"⏳ {tr.name}: morte adiada, não consegui vender ({e})", rep)
                    return
        bal = self.ledger.balance(tr.name)
        if bal > 0:
            self.ledger.transfer(tr.name, TREASURY, bal, f"morte de {tr.name}")
        self.graveyard.append({"name": tr.name, "parent": tr.parent, "generation": tr.generation,
                               "reason": reason, "tick": self.tick_no, "params": tr.params.to_dict()})
        del self.traders[tr.name]
        self.log(f"💀 {tr.name} MORREU e foi removido do pool: {reason}", rep)

    # ------------------------------------------------------------------- loop
    def run(self, max_ticks: int | None = None) -> int:
        if self.recover_if_needed():
            return EXIT_FUSE
        done = 0
        while True:
            rep = self.tick()
            if self.stop_reason == "fuse" or rep.halted:
                return EXIT_FUSE
            if self.stop_reason == "extinct":
                return EXIT_EXTINCT
            done += 1
            if max_ticks is not None and done >= max_ticks:
                return EXIT_OK
            self._sleep(self.cfg.tick_seconds)

    # ------------------------------------------------------------- persistência
    def save_state(self) -> None:
        if not self.state_path:
            return
        data = {"tick_no": self.tick_no, "graveyard": self.graveyard,
                "traders": [t.to_dict() for t in self.traders.values()]}
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_path)

    def load_state(self) -> bool:
        if not self.state_path or not os.path.exists(self.state_path):
            return False
        with open(self.state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tick_no = int(data["tick_no"])
        self.graveyard = list(data["graveyard"])
        self.traders = {d["name"]: Trader.from_dict(d) for d in data["traders"]}
        for t in self.traders.values():   # estado antigo (um ativo só): posição aberta era no primeiro par
            if t.qty > 0 and not t.symbol:
                t.symbol = self.gw.symbol
        return True

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        eq = self.gw.equity_cents()
        init = self.breaker.initial_equity_cents or 0
        unrealized = 0
        prices: dict[str, float] = {}
        for t in self.traders.values():
            if t.qty > 0:
                sym = self._sym(t)
                if sym not in prices:
                    prices[sym] = self.gw.last_price(sym)
                unrealized += to_cents(t.qty * prices[sym]) - t.cost_cents
        ledger_total = sum(self.ledger.balance(t.name) for t in self.traders.values()) \
            + self.ledger.balance(TREASURY)
        return {
            "equity_cents": eq, "initial_cents": init, "floor_cents": self.breaker.floor_cents,
            "account_pnl_cents": eq - init, "breaker": self.breaker.state.value,
            "breaker_reason": self.breaker.reason, "tick": self.tick_no,
            "treasury_cents": self.ledger.balance(TREASURY),
            "drift_cents": eq - (ledger_total + unrealized),
            "traders": [{"name": t.name, "gen": t.generation, "capital_cents": self.ledger.balance(t.name),
                         "in_position": t.in_position, "symbol": self._sym(t) if t.in_position else "",
                         "window_roundtrips": t.window_roundtrips,
                         "window_ticks": t.window_ticks, "sentiment_weight": t.params.sentiment_weight}
                        for t in self.alive()],
            "dead": len(self.graveyard),
        }
