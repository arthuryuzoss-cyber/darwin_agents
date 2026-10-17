"""Previsor de DIREÇÃO diária (sobe/desce amanhã), testado fora da amostra e com custos. Só pesquisa; não opera.

Ideia de um colega: em vez de prever preço, prever só se o próximo candle fecha acima do atual e ficar comprado
nos ativos com probabilidade acima de um limiar. Modelo: regressão logística (puro Python) sobre retornos passados,
volatilidade, RSI e distância às médias, treinada com todos os ativos juntos. Validação walk-forward (treina só no
passado), custos de taxa+derrapagem sobre a variação de posição, comparação com comprar e segurar.
Não modelado: valor mínimo por ordem, passo de lote, impacto. Execução: decide no fechamento de t, ganha o
retorno de t até t+1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .backtest import BacktestConfig, BacktestError, _date, _pct, max_drawdown_pct
from .history import History

LAGS = (1, 3, 7, 14, 30)
WARM = 60
NAMES = [f"ret{n}d" for n in LAGS] + ["vol14", "rsi14", "dist_sma20", "dist_sma50"]


def _features(closes: list[float | None], t: int) -> list[float] | None:
    """Só usa fechamentos até t. None se faltar histórico."""
    if t < WARM or any(closes[t - k] is None for k in range(0, 51)):
        return None
    c = lambda k: closes[t - k]
    f = [c(0) / c(n) - 1.0 for n in LAGS]
    rets = [c(k) / c(k + 1) - 1.0 for k in range(14)]
    m = sum(rets) / 14
    f.append(math.sqrt(sum((r - m) ** 2 for r in rets) / 13))
    gains = sum(max(0.0, c(k) - c(k + 1)) for k in range(14))
    loss = sum(max(0.0, c(k + 1) - c(k)) for k in range(14))
    f.append(0.5 if gains + loss == 0 else gains / (gains + loss) - 0.5)
    f.append(c(0) / (sum(c(k) for k in range(20)) / 20) - 1.0)
    f.append(c(0) / (sum(c(k) for k in range(50)) / 50) - 1.0)
    return f


def build_samples(hist: History, symbols) -> list[dict]:
    out = []
    for s in symbols:
        col = [None if r is None else r[3] for r in hist.ohlcv[s]]
        for t in range(WARM, hist.n - 1):
            f = _features(col, t)
            if f is None or col[t + 1] is None:
                continue
            out.append({"t": t, "sym": s, "x": f, "y": 1 if col[t + 1] > col[t] else 0,
                        "ret": col[t + 1] / col[t] - 1.0})
    return out


@dataclass
class Model:
    mean: list
    std: list
    w: list
    b: float

    def prob(self, x: list[float]) -> float:
        z = self.b + sum(w * (v - m) / s for w, v, m, s in zip(self.w, x, self.mean, self.std))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def fit(samples: list[dict], l2: float = 1.0, iters: int = 300, lr: float = 0.5) -> Model:
    n, d = len(samples), len(samples[0]["x"])
    mean = [sum(s["x"][j] for s in samples) / n for j in range(d)]
    std = [max(1e-9, math.sqrt(sum((s["x"][j] - mean[j]) ** 2 for s in samples) / n)) for j in range(d)]
    Z = [[(s["x"][j] - mean[j]) / std[j] for j in range(d)] for s in samples]
    ys = [s["y"] for s in samples]
    w, b = [0.0] * d, 0.0
    for _ in range(iters):
        gw, gb = [0.0] * d, 0.0
        for z, y in zip(Z, ys):
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, b + sum(a * c for a, c in zip(w, z))))))
            e = p - y
            gb += e
            for j in range(d):
                gw[j] += e * z[j]
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j] / n)
        b -= lr * gb / n
    return Model(mean, std, w, b)


def binom_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def simulate(test: list[dict], probs: list[float], n_days: tuple, bt: BacktestConfig, thr: float) -> dict:
    """Dia a dia: peso igual entre os ativos com prob >= thr (resto em caixa). Custo sobre a variação de pesos."""
    cost = bt.fee_rate + bt.slippage
    by_day: dict[int, list] = {}
    for s, p in zip(test, probs):
        by_day.setdefault(s["t"], []).append((s, p))
    eq, curve, prev, fees, orders, held_days = 1.0, [1.0], {}, 0.0, 0, 0
    bh_curve, bh = [1.0], 1.0
    for t in range(n_days[0], n_days[1]):
        rows = by_day.get(t, [])
        sig = [s for s, p in rows if p >= thr]
        w = {s["sym"]: 1.0 / len(sig) for s in sig}
        turn = sum(abs(w.get(k, 0.0) - prev.get(k, 0.0)) for k in set(w) | set(prev))
        orders += sum(1 for k in set(w) | set(prev) if abs(w.get(k, 0.0) - prev.get(k, 0.0)) > 1e-9)
        fees += turn * cost
        gain = sum(w[s["sym"]] * s["ret"] for s in sig)
        eq *= max(1e-9, 1.0 + gain - turn * cost)
        held_days += 1 if sig else 0
        prev = w
        curve.append(eq)
        if rows:
            bh *= 1.0 + sum(s["ret"] for s, _ in rows) / len(rows)       # comprar e segurar (rebalanceado diário, sem custo)
        bh_curve.append(bh)
    n = max(1, n_days[1] - n_days[0])
    return {"return_pct": (eq - 1) * 100, "max_drawdown_pct": max_drawdown_pct(curve), "orders": orders,
            "exposure_pct": held_days / n * 100, "fees_pct": fees * 100,
            "bh_return_pct": (bh - 1) * 100, "bh_drawdown_pct": max_drawdown_pct(bh_curve)}


def walk_forward(hist: History, bt: BacktestConfig, folds: int = 4, threshold: float = 0.52, l2: float = 1.0) -> dict:
    symbols = [s for s in bt.symbols if s in hist.ohlcv]
    if not symbols:
        raise BacktestError("nenhum símbolo com histórico")
    samples = build_samples(hist, symbols)
    t0 = min(s["t"] for s in samples) if samples else 0
    total = hist.n - 1 - t0
    if total < 120 * max(1, folds):
        raise BacktestError("histórico diário curto: baixe mais dias (ex.: python main.py download --days 1500 --timeframe 1d)")
    base = t0 + total // 2
    size = (hist.n - 1 - base) // folds
    cuts = [(base + k * size, base + (k + 1) * size if k < folds - 1 else hist.n - 1) for k in range(folds)]
    out, all_hits, all_n, all_up = [], 0, 0, 0
    compound, bh_compound = 1.0, 1.0
    coefs = None
    for a, b in cuts:
        # treino só com amostras cujo RÓTULO (dia t+1) já é conhecido antes de a: t+1 <= a
        train = [s for s in samples if s["t"] + 1 <= a]
        test = [s for s in samples if a <= s["t"] < b]
        if len(train) < 200 or not test:
            raise BacktestError("poucas amostras de treino/teste: baixe mais dias")
        m = fit(train, l2)
        coefs = dict(zip(NAMES, m.w))
        probs = [m.prob(s["x"]) for s in test]
        hits = sum(1 for s, p in zip(test, probs) if (p >= 0.5) == bool(s["y"]))
        ups = sum(s["y"] for s in test)
        sim = simulate(test, probs, (a, b), bt, threshold)
        compound *= 1 + sim["return_pct"] / 100
        bh_compound *= 1 + sim["bh_return_pct"] / 100
        all_hits, all_n, all_up = all_hits + hits, all_n + len(test), all_up + ups
        out.append({"period": [hist.timeline[a], hist.timeline[min(b, hist.n - 1)]], "n_train": len(train),
                    "n_test": len(test), "accuracy_pct": hits / len(test) * 100, "up_rate_pct": ups / len(test) * 100,
                    **sim})
    ci = binom_ci(all_hits, all_n)
    agg = {"accuracy_pct": all_hits / all_n * 100, "accuracy_ci95": [ci[0] * 100, ci[1] * 100],
           "up_rate_pct": all_up / all_n * 100, "n_test": all_n,
           "strategy_compounded_pct": (compound - 1) * 100, "bh_compounded_pct": (bh_compound - 1) * 100,
           "positive_folds": sum(1 for f in out if f["return_pct"] > 0), "n_folds": len(out),
           "threshold": threshold, "coefs": coefs, "assets": len(symbols)}
    return {"folds": out, "aggregate": agg}


def verdict(a: dict) -> str:
    lo = a["accuracy_ci95"][0]
    base = max(a["up_rate_pct"], 100 - a["up_rate_pct"])      # acertar chutando sempre o lado mais comum
    if lo <= base:
        edge = (f"a acurácia ({a['accuracy_pct']:.1f}%) não passa com folga de chutar sempre o lado mais comum "
                f"({base:.1f}%; IC95% {lo:.1f}% a {a['accuracy_ci95'][1]:.1f}%)")
    else:
        edge = f"a acurácia ({a['accuracy_pct']:.1f}%) superou o chute-base ({base:.1f}%) com IC95% acima dele"
    s, b = a["strategy_compounded_pct"], a["bh_compounded_pct"]
    if lo <= base and s <= 0:
        return f"SEM VANTAGEM: {edge}, e a estratégia deu {s:+.1f}% fora da amostra com custos."
    if s <= b:
        return (f"NÃO BATE COMPRAR-E-SEGURAR: {edge}; estratégia {s:+.1f}% contra {b:+.1f}% de uma carteira parada.")
    if lo <= base or a["positive_folds"] * 2 <= a["n_folds"]:
        return f"FRACO/COMPATÍVEL COM SORTE: {edge}; estratégia {s:+.1f}% contra {b:+.1f}%, " \
               f"{a['positive_folds']}/{a['n_folds']} períodos positivos."
    return (f"INDÍCIO (não é prova): {edge}; estratégia {s:+.1f}% contra {b:+.1f}% parada, "
            f"{a['positive_folds']}/{a['n_folds']} períodos positivos. Valide por mais tempo antes de usar dinheiro.")


def format_direction(wf: dict) -> str:
    L = ["=== PREVISOR DE DIREÇÃO DIÁRIA (sobe/desce amanhã; walk-forward fora da amostra) ==="]
    for k, f in enumerate(wf["folds"], 1):
        L.append(f"\n-- Fold {k}: teste {_date(f['period'][0])}..{_date(f['period'][1])} "
                 f"(treino {f['n_train']} amostras, teste {f['n_test']})")
        L.append(f"   acerto de direção {f['accuracy_pct']:.1f}% (dias de alta: {f['up_rate_pct']:.1f}%)")
        L.append(f"   estratégia {_pct(f['return_pct'])} (queda máx. {f['max_drawdown_pct']:.1f}%, exposição "
                 f"{f['exposure_pct']:.0f}%, custos {f['fees_pct']:.2f}% da banca) | comprar e segurar "
                 f"{_pct(f['bh_return_pct'])} (queda máx. {f['bh_drawdown_pct']:.1f}%)")
    a = wf["aggregate"]
    L += ["\n=== RESUMO FORA DA AMOSTRA ===",
          f"{a['assets']} ativos, {a['n_test']} previsões de teste, limiar de compra {a['threshold']:.2f}",
          f"acurácia {a['accuracy_pct']:.1f}% (IC95% {a['accuracy_ci95'][0]:.1f}% a {a['accuracy_ci95'][1]:.1f}%) | "
          f"dias de alta {a['up_rate_pct']:.1f}%",
          f"retorno composto: estratégia {_pct(a['strategy_compounded_pct'])} | comprar e segurar {_pct(a['bh_compounded_pct'])} | "
          f"períodos positivos {a['positive_folds']}/{a['n_folds']}",
          "pesos do último modelo (em desvios-padrão; >0 = empurra para 'sobe'): "
          + ", ".join(f"{k} {v:+.2f}" for k, v in a["coefs"].items()),
          "\nCONCLUSÃO: " + verdict(a),
          "Lembrete: pares listados hoje (viés de sobrevivência), execução no fechamento, sem valor mínimo por ordem "
          "nem passo de lote, custos sobre a variação de posição. O IC trata previsões como independentes "
          "(ativos correlacionados: a confiança real é menor)."]
    return "\n".join(L)
