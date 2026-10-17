"""Momentum diário em PAPEL (dinheiro de mentira): uma decisão por dia, mesma lógica do backtest `--momentum`.

Rode `python main.py signal` uma vez por dia, logo depois das 21h (00:00 UTC, quando o candle diário vira).
Decide com dados até o fechamento de ontem e "executa" no preço de abertura de hoje (igual ao backtest).
Se pular dias, o comando refaz os dias perdidos na ordem. Não usa chaves da Binance nem envia ordens."""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import replace

from .history import History
from .momentum import MomParams, _resolve_market, choose

DEFAULT_PARAMS = MomParams(trend_n=150, asset_trend=100, long=126, short=63)   # platô da sensibilidade, não o máximo


class PaperError(Exception):
    pass


def new_state(bankroll: float) -> dict:
    return {"bankroll": bankroll, "cash": bankroll, "held": None, "qty": 0.0, "last_day": None, "log": []}


def _day(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")


def equity(state: dict, hist: History, i: int, field: int = 0) -> float:
    """Patrimônio marcado no preço (0=abertura, 3=fechamento) do candle i."""
    v = state["cash"]
    if state["held"]:
        r = hist.ohlcv[state["held"]][i]
        if r is not None:
            v += state["qty"] * r[field]
    return v


def apply_day(state: dict, hist: History, symbols: list[str], p: MomParams, i: int, fee: float, slip: float,
              min_n: float = 10.0, exposure: float = 1.0) -> dict:
    held = state["held"]
    want = choose(hist, symbols, i - 1, held, p)          # só dados até o fechamento de i-1
    action = "manter"
    if want != held:
        if held is not None:
            o = hist.ohlcv[held][i]
            if o is not None:
                state["cash"] += state["qty"] * o[0] * (1 - slip) * (1 - fee)
                state["held"], state["qty"] = None, 0.0
                action = f"VENDER {held}"
        if want is not None and state["held"] is None and hist.ohlcv[want][i] is not None \
                and state["cash"] * exposure >= min_n:
            px = hist.ohlcv[want][i][0] * (1 + slip)
            spend = state["cash"] * exposure / (1 + fee)
            state["qty"], state["held"] = spend / px, want
            state["cash"] -= spend * (1 + fee)
            action = f"COMPRAR {want}" if action == "manter" else f"{action} e COMPRAR {want}"
    entry = {"day": _day(hist.timeline[i]), "action": action, "held": state["held"],
             "equity": round(equity(state, hist, i, 0), 4)}
    state["last_day"] = entry["day"]
    state["log"].append(entry)
    return entry


def step(state: dict, hist: History, symbols: list[str], p: MomParams, fee: float = 0.001, slip: float = 0.0005,
         min_n: float = 10.0, exposure: float = 1.0) -> list[dict]:
    """Processa os dias novos (inclusive os pulados). Devolve as entradas novas do diário."""
    if not 0 < exposure <= 1:
        raise PaperError("exposição deve estar entre 0 e 1")
    symbols = [s for s in symbols if s in hist.ohlcv]
    p = _resolve_market(hist, symbols, p)
    if p.market not in hist.ohlcv:
        raise PaperError(f"falta o histórico de {p.market}")
    if hist.n < p.need() + 2:
        raise PaperError("histórico curto: baixe mais candles diários")
    last = hist.n - 1
    days = [_day(t) for t in hist.timeline]
    if state["last_day"] is None:
        todo = [last]
    elif state["last_day"] in days:
        todo = list(range(days.index(state["last_day"]) + 1, hist.n))
    else:
        todo = [last]
    return [apply_day(state, hist, symbols, p, i, fee, slip, min_n, exposure) for i in todo]


def load_state(path: str, bankroll: float) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        if st.get("bankroll") != bankroll:
            raise PaperError(f"o estado em {path} foi criado com banca {st.get('bankroll')}; "
                             "use --reset para recomeçar ou ajuste BANKROLL_BRL")
        return st
    return new_state(bankroll)


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def format_step(state: dict, new: list[dict], hist: History, p: MomParams, exposure: float) -> str:
    last = hist.n - 1
    L = [f"=== MOMENTUM DIÁRIO — PAPEL (média {p.trend_n} dias, momentum {p.long}/{p.short}, exposição {exposure:.0%}) ==="]
    if not new:
        L.append("Hoje já foi processado. Volte depois das 21h (00:00 UTC) de amanhã.")
    for e in new:
        L.append(f"{e['day']}: {e['action']} | posição: {e['held'] or 'caixa'} | patrimônio {e['equity']:.2f}")
    eq = equity(state, hist, last, 3)
    L.append(f"\npatrimônio agora (preço de fechamento disponível): {eq:.2f} | banca inicial {state['bankroll']:.2f} "
             f"({(eq / state['bankroll'] - 1) * 100:+.1f}%) | dias registrados {len(state['log'])}")
    L.append("Isto é papel: nenhuma ordem foi enviada. Só avalie dinheiro real depois de meses de papel batendo o backtest.")
    return "\n".join(L)
