"""Diário de sinais: registra, a cada ~5 min e por ativo, preço + nota técnica + sentimento (sem ordens, sem risco).

Depois, `python main.py journal` cruza cada sinal com o retorno que veio DEPOIS (1h, 4h, 24h) e diz se a nota técnica
ou o sentimento de notícias realmente previram alguma coisa. É a única forma honesta de medir o sentimento ao vivo,
porque não existe histórico de notícias para o backtest. Falha de escrita nunca afeta o robô.
"""
from __future__ import annotations

import bisect
import csv
import math
import os
import random
import time
from typing import Callable

from .strategy import StrategyParams, tech_score

HEADER = ["ts_ms", "symbol", "price", "tech", "sentiment"]
HORIZONS_H = (1, 4, 24)
TOL = 0.10            # tolerância ao achar o preço futuro: 10% do horizonte (mínimo 10 min)


class Journal:
    def __init__(self, path: str, every_s: int = 300, clock: Callable[[], float] = time.time):
        self.path, self.every_ms, self._clock = path, every_s * 1000, clock
        self._last = 0
        self.errors = 0

    def record(self, snap: dict, sents: dict) -> int:
        """snap: {símbolo: (preço, fechamentos)}. Devolve quantas linhas gravou (0 se ainda não é hora)."""
        now = int(self._clock() * 1000)
        if now - self._last < self.every_ms:
            return 0
        rows = []
        for sym, (price, closes) in snap.items():
            try:
                t = tech_score(closes, StrategyParams())
            except Exception:
                continue
            s = sents.get(sym)
            if not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0 and math.isfinite(t)):
                continue
            rows.append([now, sym, repr(float(price)), f"{t:.5f}", "" if s is None else f"{float(s):.5f}"])
        if not rows:
            return 0
        try:
            new = not os.path.exists(self.path)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(HEADER)
                w.writerows(rows)
            self._last = now
        except OSError:
            self.errors += 1
            return 0
        return len(rows)


def load(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            try:
                out.append({"ts": int(rec["ts_ms"]), "symbol": rec["symbol"], "price": float(rec["price"]),
                            "tech": float(rec["tech"]),
                            "sent": float(rec["sentiment"]) if rec["sentiment"] != "" else None})
            except (KeyError, ValueError):
                continue
    return out


def forward_samples(rows: list[dict], horizon_h: int) -> list[dict]:
    """Cada linha ganha `ret` = retorno (%) até ~horizon_h depois, se houver preço futuro dentro da tolerância."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["symbol"], []).append(r)
    h = horizon_h * 3_600_000
    out = []
    for sym, lst in by.items():
        lst.sort(key=lambda r: r["ts"])
        ts = [r["ts"] for r in lst]
        for r in lst:
            j = bisect.bisect_left(ts, r["ts"] + h)
            for k in (j, j - 1):
                if 0 <= k < len(lst) and abs(ts[k] - (r["ts"] + h)) <= max(h * TOL, 600_000) and ts[k] > r["ts"]:
                    out.append({**r, "ret": (lst[k]["price"] / r["price"] - 1) * 100.0})
                    break
    return out


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 10:
        return None
    a, b = _rank(x), _rank(y)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return None if den == 0 else num / den


def spearman_ci(x: list[float], y: list[float], n_boot: int = 300, seed: int = 1, block: int = 24):
    """IC95% por bootstrap em BLOCOS (amostras vizinhas são quase iguais; reamostrar uma a uma enganaria)."""
    n = len(x)
    if n < 40:
        return None
    if n > 3000:                       # desbasta para o bootstrap não ficar lento (a correlação principal usa tudo)
        k = -(-n // 3000)
        x, y = x[::k], y[::k]
        n, block = len(x), max(4, block // k)
    rng = random.Random(seed)
    nb = max(1, n // block)
    vals = []
    for _ in range(n_boot):
        idx = []
        for _ in range(nb):
            s = rng.randrange(0, max(1, n - block + 1))
            idx.extend(range(s, s + block))
        r = spearman([x[i] for i in idx], [y[i] for i in idx])
        if r is not None:
            vals.append(r)
    if not vals:
        return None
    vals.sort()
    return vals[int(len(vals) * 0.025)], vals[min(len(vals) - 1, int(len(vals) * 0.975))]


def analyze(rows: list[dict]) -> dict:
    res: dict = {"rows": len(rows), "symbols": len({r["symbol"] for r in rows}), "horizons": {}}
    if rows:
        res["days"] = (max(r["ts"] for r in rows) - min(r["ts"] for r in rows)) / 86_400_000
    for h in HORIZONS_H:
        fs = sorted(forward_samples(rows, h), key=lambda r: (r["symbol"], r["ts"]))
        d: dict = {"n": len(fs)}
        if fs:
            d["mean_ret_pct"] = sum(r["ret"] for r in fs) / len(fs)
            tx, ry = [r["tech"] for r in fs], [r["ret"] for r in fs]
            d["tech_corr"], d["tech_ci"] = spearman(tx, ry), spearman_ci(tx, ry)
            sf = [r for r in fs if r["sent"] is not None]
            d["n_sent"] = len(sf)
            sx, sy = [r["sent"] for r in sf], [r["ret"] for r in sf]
            d["sent_corr"], d["sent_ci"] = spearman(sx, sy), spearman_ci(sx, sy)
            hi = [r["ret"] for r in fs if r["tech"] >= 0.3]
            d["n_signal"] = len(hi)
            d["signal_mean_ret_pct"] = sum(hi) / len(hi) if hi else None
        res["horizons"][h] = d
    return res


def _f(x, p=3):
    return "—" if x is None else f"{x:+.{p}f}"


def _ci(c):
    return "" if c is None else f" (IC95% {c[0]:+.2f} a {c[1]:+.2f})"


def format_analysis(a: dict) -> str:
    L = ["=== DIÁRIO DE SINAIS: a nota/sentimento previu o preço? ===",
         f"{a['rows']} registros, {a['symbols']} ativos, {a.get('days', 0):.1f} dia(s) de dados"]
    useful = False
    for h, d in a["horizons"].items():
        L.append(f"\n-- {h}h à frente: {d['n']} amostras" + (f" | retorno médio {_f(d.get('mean_ret_pct'), 2)}%" if d["n"] else ""))
        if not d["n"]:
            continue
        L.append(f"   nota técnica x retorno futuro (correlação de Spearman): {_f(d['tech_corr'])}{_ci(d['tech_ci'])}")
        L.append(f"   sentimento x retorno futuro: {_f(d['sent_corr'])}{_ci(d['sent_ci'])} ({d['n_sent']} amostras com sentimento)")
        if d["signal_mean_ret_pct"] is not None:
            L.append(f"   quando a nota ≥ 0,30 ({d['n_signal']} vezes): retorno médio {_f(d['signal_mean_ret_pct'], 2)}% "
                     f"contra {_f(d['mean_ret_pct'], 2)}% na média geral")
        for c in (d["tech_ci"], d["sent_ci"]):
            if c and c[0] > 0:
                useful = True
    if a["rows"] < 500 or a.get("days", 0) < 7:
        concl = "INCONCLUSIVO: poucos dados (junte pelo menos 2 a 4 semanas rodando o robô)."
    elif useful:
        concl = ("INDÍCIO: algum sinal tem correlação positiva com intervalo acima de zero. Não é prova: o período pode "
                 "ser atípico e o retorno ainda precisa superar taxas (~0,2% por ida e volta).")
    else:
        concl = "SEM VANTAGEM: nenhum sinal teve correlação positiva distinguível de zero."
    L.append("\nCONCLUSÃO: " + concl)
    L.append("Nota: amostras vizinhas são parecidas; o IC usa blocos de 24 amostras para não exagerar a confiança.")
    return "\n".join(L)
