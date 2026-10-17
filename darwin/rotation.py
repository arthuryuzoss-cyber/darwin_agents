"""Rotação por força relativa (pesquisa em backtest; não opera dinheiro).

A cada `rebalance` candles, mede o retorno dos últimos `lookback` candles de cada ativo e mantém em carteira os
`top_n` mais fortes (peso igual). Com `abs_filter`, só mantém quem tem retorno positivo (senão fica em caixa).
Custos: taxa e derrapagem como no backtest principal, e valor mínimo por ordem. Passo de lote e impacto de
ordem não são modelados. Sem olhar o futuro: a decisão em `i` usa só fechamentos até `i`; executa no fechamento de `i`.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .backtest import (BacktestConfig, BacktestError, _aggregate, _brief, _date, _pct, _percentiles, buy_and_hold,
                       max_drawdown_pct, verdict)
from .history import History


@dataclass(frozen=True)
class RotParams:
    lookback: int = 168          # candles
    rebalance: int = 72          # candles entre decisões
    top_n: int = 2
    abs_filter: bool = True

    def __post_init__(self):
        if self.lookback < 2 or self.rebalance < 1 or not (1 <= self.top_n <= 6):
            raise ValueError("parâmetros de rotação inválidos")

    def to_dict(self) -> dict:
        return asdict(self)


def sample(rng: random.Random) -> RotParams:
    return RotParams(lookback=rng.choice([24, 48, 72, 120, 168, 336, 504, 720]),
                     rebalance=rng.choice([12, 24, 48, 72, 168]), top_n=rng.choice([1, 2, 3]),
                     abs_filter=rng.random() < 0.7)


def _px(hist: History, sym: str, i: int) -> float | None:
    """Último preço conhecido até i (None se ainda não listado)."""
    col = hist.ohlcv[sym]
    for j in range(i, max(hist.first_idx[sym] - 1, -1), -1):
        if col[j] is not None:
            return col[j][3]
    return None


def simulate(hist: History, bt: BacktestConfig, p: RotParams, start: int, end: int) -> dict:
    symbols = [s for s in bt.symbols if s in hist.ohlcv]
    end = min(end, hist.n)
    start = max(start, min(hist.first_idx[s] for s in symbols) + p.lookback)
    if end - start < 10:
        raise BacktestError("período curto demais para a rotação")
    fee, slip = bt.fee_rate, bt.slippage
    min_notional = bt.min_notional if bt.min_notional is not None else 10.0
    cash = bt.bankroll_cents / 100.0
    pos: dict[str, float] = {}
    fees = 0.0
    orders = 0
    curve, in_mkt = [], 0
    last = {s: None for s in symbols}
    for i in range(start, end):
        for s in symbols:
            v = hist.close(s, i)
            if v is not None:
                last[s] = v
        if (i - start) % p.rebalance == 0:
            rank = []
            for s in symbols:
                a, b = hist.close(s, i), hist.close(s, i - p.lookback)
                if a is not None and b is not None and b > 0:
                    rank.append((a / b - 1.0, s))
            rank.sort(reverse=True)
            target = [s for r, s in rank[:p.top_n] if (r > 0 or not p.abs_filter)]
            for s in list(pos):
                if s not in target and hist.close(s, i) is not None:
                    px = hist.close(s, i) * (1 - slip)
                    gross = pos.pop(s) * px
                    fees += gross * fee
                    cash += gross * (1 - fee)
                    orders += 1
            need = [s for s in target if s not in pos]
            if need and cash > 0:
                slot = cash / len(need)
                for s in need:
                    if slot < min_notional:
                        continue                                   # abaixo do mínimo da corretora: não compra
                    px = hist.close(s, i) * (1 + slip)
                    spend = slot / (1 + fee)
                    fees += spend * fee
                    pos[s] = spend / px
                    cash -= slot
                    orders += 1
        eq = cash + sum(q * last[s] for s, q in pos.items() if last[s] is not None)
        curve.append(eq)
        in_mkt += 1 if pos else 0
    initial = bt.bankroll_cents / 100.0
    return {"return_pct": (curve[-1] / initial - 1) * 100.0, "max_drawdown_pct": max_drawdown_pct(curve),
            "n_trades": orders, "exposure_pct": in_mkt / len(curve) * 100.0, "fees_cents": round(fees * 100),
            "win_rate_pct": None, "expectancy_cents": None, "profit_factor": None, "fuse": False,
            "expectancy_ci95_cents": None}


def _score(m: dict, min_orders: int) -> float:
    if m["n_trades"] < min_orders:
        return -1e9 + m["n_trades"]
    return m["return_pct"] - 0.5 * m["max_drawdown_pct"]


def walk_forward(hist: History, bt: BacktestConfig, folds: int = 4, trials: int = 30, top_k: int = 3,
                 seed: int = 1, min_orders: int = 6, progress=None) -> dict:
    symbols = [s for s in bt.symbols if s in hist.ohlcv]
    s0 = min(hist.first_idx[s] for s in symbols) + 720            # aquecimento para o maior lookback
    total = hist.n - s0
    if total < 400 * max(1, folds):
        raise BacktestError("histórico curto para esse número de folds (baixe mais dias)")
    base = s0 + total // 2
    size = (hist.n - base) // folds
    cuts = [(s0, base + k * size, base + (k + 1) * size if k < folds - 1 else hist.n) for k in range(folds)]
    rng = random.Random(seed)
    say = progress or (lambda m: None)
    out = []
    for fi, (a, b, c) in enumerate(cuts, 1):
        cands = [RotParams()] + [sample(rng) for _ in range(trials)]
        say(f"fold {fi}/{len(cuts)}: treinando {len(cands)} conjuntos…")
        tr = [simulate(hist, bt, p, a, b) for p in cands]
        notes = [_score(m, min_orders) for m in tr]
        order = sorted(range(len(cands)), key=lambda k: notes[k], reverse=True)
        chosen = order[:top_k]
        te = [simulate(hist, bt, cands[k], b, c) for k in chosen] + [simulate(hist, bt, RotParams(), b, c)]
        out.append({
            "train": [a, b], "test": [b, c],
            "train_period": [hist.timeline[a], hist.timeline[b - 1]],
            "test_period": [hist.timeline[b], hist.timeline[c - 1]],
            "chosen_params": cands[chosen[0]].to_dict(),
            "chosen_train": _brief(tr[chosen[0]]), "chosen_test": _brief(te[0]),
            "top": [{"params": cands[k].to_dict(), "train": _brief(tr[k]), "test": _brief(te[j])}
                    for j, k in enumerate(chosen)],
            "default_train": _brief(tr[0]), "default_test": _brief(te[-1]),
            "buy_hold_test": buy_and_hold(hist, symbols, b, c, bt),
            "n_trials": len(cands), "train_scores_pct": _percentiles([n for n in notes if n > -1e8]),
        })
    return {"folds": out, "aggregate": _aggregate(out), "min_trades": min_orders, "kind": "rotation"}


def format_rotation(wf: dict) -> str:
    L = ["=== ROTAÇÃO POR FORÇA RELATIVA (walk-forward, fora da amostra) ==="]
    for k, f in enumerate(wf["folds"], 1):
        ct, cs = f["chosen_train"], f["chosen_test"]
        L.append(f"\n-- Fold {k}: treino {_date(f['train_period'][0])}..{_date(f['train_period'][1])} | "
                 f"teste {_date(f['test_period'][0])}..{_date(f['test_period'][1])}")
        L.append(f"   escolhido: {f['chosen_params']}")
        L.append(f"   treino {_pct(ct['return_pct'])} -> TESTE {_pct(cs['return_pct'])} "
                 f"({cs['n_trades']} ordens, pior queda {cs['max_drawdown_pct']:.1f}%, taxas R$ {cs['fees_cents'] / 100:.2f})")
        L.append(f"   padrão no teste: {_pct(f['default_test']['return_pct'])} | "
                 f"comprar e segurar: {_pct(f['buy_hold_test']['equal_weight_return_pct'])}")
        L.append("   top (treino -> teste): " + ", ".join(
            f"{_pct(t['train']['return_pct'])} -> {_pct(t['test']['return_pct'])}" for t in f["top"]))
    a = wf["aggregate"]
    L += ["\n=== RESUMO FORA DA AMOSTRA ===",
          f"retorno composto nos testes: {_pct(a['oos_compounded_pct'])} | parâmetros padrão: "
          f"{_pct(a['default_compounded_pct'])} | comprar e segurar: {_pct(a['buy_hold_compounded_pct'])}",
          f"períodos de teste positivos: {a['positive_folds']}/{a['n_folds']} | ordens fora da amostra: {a['oos_trades']}",
          f"treino médio {_pct(a['avg_train_return_pct'])} vs teste médio {_pct(a['avg_test_return_pct'])}",
          "\nCONCLUSÃO: " + verdict(a),
          "Lembrete: sem sentimento, pares listados hoje (viés de sobrevivência), preço de fechamento, sem passo de lote."]
    return "\n".join(L)
