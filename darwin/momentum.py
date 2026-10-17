"""Momentum diário com filtro de mercado (ideia do projeto de ações "darwin_stocks_v2", adaptada a cripto).

Regras (parâmetros FIXOS, sem otimização, para não criar sobreajuste):
- Filtro de mercado: só fica comprado se o BTC fechou acima da média de `trend_n` dias; senão, caixa.
- Seleção: entre os ativos acima da própria média de `asset_trend` dias e com retorno positivo em `long` e `short`
  dias, escolhe o maior escore = 0,6*ret_long + 0,4*ret_short - penalidade de volatilidade. 1 ativo por vez.
- Histerese: só troca de ativo se o novo escore superar o do atual em `switch_margin`.
- Decide no FECHAMENTO de t-1 e executa na ABERTURA de t (sem olhar o futuro). Custos: taxa + derrapagem por lado,
  ordem mínima da exchange. Passo de lote e impacto não são modelados.
Avaliação: blocos consecutivos fora da amostra (a estratégia não é ajustada), comparados com comprar e segurar.
"""
from __future__ import annotations

import math
import datetime as _dt
import random
import statistics
from dataclasses import asdict, dataclass, replace

from .backtest import BacktestConfig, BacktestError, _aggregate, _date, _pct, max_drawdown_pct, buy_and_hold, verdict
from .history import History


@dataclass(frozen=True)
class MomParams:
    trend_n: int = 200
    asset_trend: int = 100
    long: int = 126
    short: int = 63
    switch_margin: float = 0.10
    vol_ref: float = 0.80          # volatilidade anualizada "normal" de cripto (a de ações seria ~0,35)
    vol_pen: float = 0.10
    market: str = "BTC/BRL"

    def need(self) -> int:
        return max(self.trend_n, self.asset_trend, self.long, 21) + 1


def _closes(hist: History, sym: str, i: int, n: int) -> list[float] | None:
    """n fechamentos terminando em i (inclusive); None se faltar dado."""
    if i - n + 1 < hist.first_idx[sym]:
        return None
    col = hist.ohlcv[sym]
    out = []
    for j in range(i - n + 1, i + 1):
        r = col[j]
        if r is None:
            return None
        out.append(r[3])
    return out


def choose(hist: History, symbols: list[str], i: int, held: str | None, p: MomParams) -> str | None:
    """Decisão com dados até o fechamento de i."""
    m = _closes(hist, p.market, i, p.trend_n) if p.market in hist.ohlcv else None
    if m is None or m[-1] <= sum(m) / len(m):
        return None
    scores: dict[str, float] = {}
    for s in symbols:
        c = _closes(hist, s, i, max(p.asset_trend, p.long + 1, 21))
        if c is None:
            continue
        px = c[-1]
        sma = sum(c[-p.asset_trend:]) / p.asset_trend
        r_long, r_short = px / c[-1 - p.long] - 1, px / c[-1 - p.short] - 1
        if px <= sma or r_long <= 0 or r_short <= 0:
            continue
        rets = [c[k] / c[k - 1] - 1 for k in range(len(c) - 20, len(c))]
        mu = sum(rets) / 20
        vol = math.sqrt(sum((x - mu) ** 2 for x in rets) / 19) * math.sqrt(365)
        scores[s] = 0.6 * r_long + 0.4 * r_short - p.vol_pen * max(vol - p.vol_ref, 0.0)
    if not scores:
        return None
    best = max(scores, key=scores.get)
    if held in scores and best != held and scores[best] < scores[held] + p.switch_margin:
        return held
    return best


def simulate(hist: History, bt: BacktestConfig, p: MomParams, start: int, end: int) -> dict:
    symbols = [s for s in bt.symbols if s in hist.ohlcv]
    end = min(end, hist.n)
    start = max(start, min(hist.first_idx[s] for s in symbols) + p.need())
    if end - start < 10:
        raise BacktestError("período curto demais para o momentum")
    fee, slip = bt.fee_rate, bt.slippage
    min_n = bt.min_notional if bt.min_notional is not None else 10.0
    cash = bt.bankroll_cents / 100.0
    held, qty, fees, orders, last_px = None, 0.0, 0.0, 0, 0.0
    curve, in_mkt = [], 0
    for i in range(start, end):
        want = choose(hist, symbols, i - 1, held, p)          # só dados até o fechamento de i-1
        if want != held:
            if held is not None:
                o = hist.ohlcv[held][i]
                if o is not None:
                    gross = qty * o[0] * (1 - slip)
                    fees += gross * fee
                    cash += gross * (1 - fee)
                    orders += 1
                    held, qty = None, 0.0
            if want is not None and held is None and hist.ohlcv[want][i] is not None and cash >= min_n:
                o = hist.ohlcv[want][i][0] * (1 + slip)
                spend = cash / (1 + fee)
                fees += spend * fee
                qty, held, cash = spend / o, want, 0.0
                last_px = o
                orders += 1
        if held:
            r = hist.ohlcv[held][i]
            if r is not None:
                last_px = r[3]
            curve.append(cash + qty * last_px)
        else:
            curve.append(cash)
        in_mkt += 1 if held else 0
    initial = bt.bankroll_cents / 100.0
    return {"return_pct": (curve[-1] / initial - 1) * 100.0, "max_drawdown_pct": max_drawdown_pct(curve),
            "n_trades": orders, "exposure_pct": in_mkt / len(curve) * 100.0, "fees_cents": round(fees * 100),
            "win_rate_pct": None, "expectancy_cents": None, "profit_factor": None, "fuse": False,
            "expectancy_ci95_cents": None, "curve": curve}


def _resolve_market(hist: History, symbols: list[str], p: MomParams) -> MomParams:
    """Se o filtro de mercado (BTC/BRL) não existe no histórico, usa BTC na mesma moeda de cotação (ex.: BTC/USDT)."""
    if p.market in hist.ohlcv:
        return p
    quote = symbols[0].split("/")[-1] if symbols else "BRL"
    alt = f"BTC/{quote}"
    return replace(p, market=alt) if alt in hist.ohlcv else p


def walk_forward(hist: History, bt: BacktestConfig, test_days: int = 180, p: MomParams | None = None) -> dict:
    p = p or MomParams()
    symbols = [s for s in bt.symbols if s in hist.ohlcv]
    p = _resolve_market(hist, symbols, p)
    if p.market not in hist.ohlcv:
        raise BacktestError(f"o filtro de mercado precisa do histórico de {p.market}")
    s0 = min(hist.first_idx[s] for s in symbols) + p.need()
    s0 = max(s0, hist.first_idx[p.market] + p.need())
    if hist.n - s0 < test_days * 2:
        raise BacktestError("histórico diário curto: baixe mais dias (python main.py download --days 1500 --timeframe 1d)")
    folds, a = [], s0
    while hist.n - a >= test_days // 2:
        b = min(a + test_days, hist.n)
        m = simulate(hist, bt, p, a, b)
        bh = buy_and_hold(hist, symbols, a, b, bt)
        folds.append({"period": [hist.timeline[a], hist.timeline[b - 1]], "start": a, "end": b, "result": m, "buy_hold": bh})
        a = b
    full = simulate(hist, bt, p, s0, hist.n)
    curve = full.pop("curve")
    for f in folds:
        f["result"].pop("curve", None)
    full_bh = buy_and_hold(hist, symbols, s0, hist.n, bt)
    shaped = [{"chosen_test": f["result"], "default_test": f["result"], "buy_hold_test": f["buy_hold"],
               "chosen_train": f["result"]} for f in folds]
    agg = _aggregate(shaped)
    agg["median_fold_return_pct"] = statistics.median(f["result"]["return_pct"] for f in folds)
    agg["worst_fold_drawdown_pct"] = max(f["result"]["max_drawdown_pct"] for f in folds)
    return {"folds": folds, "aggregate": agg, "full": full, "full_buy_hold": full_bh, "params": asdict(p),
            "test_days": test_days, "s0": s0, "curve": curve}


def format_momentum(wf: dict) -> str:
    L = ["=== MOMENTUM DIÁRIO COM FILTRO DE MERCADO (parâmetros fixos, blocos fora da amostra) ===",
         f"parâmetros: {wf['params']}"]
    for k, f in enumerate(wf["folds"], 1):
        r, b = f["result"], f["buy_hold"]
        L.append(f"-- Bloco {k}: {_date(f['period'][0])}..{_date(f['period'][1])}: estratégia {_pct(r['return_pct'])} "
                 f"(queda máx. {r['max_drawdown_pct']:.1f}%, exposição {r['exposure_pct']:.0f}%, {r['n_trades']} ordens) | "
                 f"comprar e segurar {_pct(b['equal_weight_return_pct'])} | só BTC {_pct(b.get('btc_return_pct'))}")
    a, fu, fb = wf["aggregate"], wf["full"], wf["full_buy_hold"]
    L += ["\n=== RESUMO ===",
          f"período inteiro (contínuo): estratégia {_pct(fu['return_pct'])} (queda máx. {fu['max_drawdown_pct']:.1f}%) | "
          f"comprar e segurar {_pct(fb['equal_weight_return_pct'])} (queda máx. {fb['max_drawdown_pct']:.1f}%) | só BTC "
          f"{_pct(fb.get('btc_return_pct'))}",
          f"blocos: retorno composto {_pct(a['oos_compounded_pct'])} contra {_pct(a['buy_hold_compounded_pct'])} parado | "
          f"positivos {a['positive_folds']}/{a['n_folds']} | mediana {_pct(a['median_fold_return_pct'])} | "
          f"ordens {a['oos_trades']}",
          "\nCONCLUSÃO: " + verdict(a),
          "Lembrete: pares listados hoje (viés de sobrevivência), sem passo de lote, poucas dezenas de trades = pouca "
          "confiança estatística, e a queda máxima do próprio BTC já foi de ~75% em ciclos passados."]
    return "\n".join(L)



# ---------------------------------------------------------------- catálogo de longo prazo
def _ew_daily_returns(hist: History, symbols: list[str], a: int, b: int) -> list[float]:
    """Retorno diário da carteira igual (rebalanceada todo dia) dos ativos que existem em cada dia — referência."""
    out = []
    for i in range(a + 1, b):
        rs = []
        for s in symbols:
            x, y = hist.ohlcv[s][i - 1], hist.ohlcv[s][i]
            if x is not None and y is not None and x[3] > 0:
                rs.append(y[3] / x[3] - 1)
        out.append(sum(rs) / len(rs) if rs else 0.0)
    return out


def _curve_returns(curve: list[float], initial: float) -> list[float]:
    prev, out = initial, []
    for v in curve:
        out.append(v / prev - 1 if prev > 0 else 0.0)
        prev = v
    return out


def block_bootstrap(strat: list[float], bench: list[float], n_sims: int = 2000, block: int = 20, seed: int = 1) -> dict:
    """Bootstrap circular em blocos (preserva autocorrelação), pareado: os mesmos dias sorteados valem para
    estratégia e referência. Mostra o quanto o resultado depende da sorte da ordem dos dias."""
    n = min(len(strat), len(bench))
    if n < block * 3:
        raise BacktestError("poucos dias para o Monte Carlo")
    rng = random.Random(seed)
    tot_s, tot_b, dd_s, beat = [], [], [], 0
    for _ in range(n_sims):
        idx: list[int] = []
        while len(idx) < n:
            st = rng.randrange(n)
            idx.extend((st + k) % n for k in range(block))
        idx = idx[:n]
        es = eb = 1.0
        peak, mdd = 1.0, 0.0
        for j in idx:
            es *= 1 + strat[j]
            eb *= 1 + bench[j]
            peak = max(peak, es)
            mdd = max(mdd, (peak - es) / peak)
        tot_s.append((es - 1) * 100)
        tot_b.append((eb - 1) * 100)
        dd_s.append(mdd * 100)
        beat += 1 if es > eb else 0

    def q(v, f):
        v = sorted(v)
        return v[min(len(v) - 1, int(f * len(v)))]
    return {"n_sims": n_sims, "block": block, "days": n,
            "strat_p5": q(tot_s, .05), "strat_p50": q(tot_s, .5), "strat_p95": q(tot_s, .95),
            "bench_p50": q(tot_b, .5), "dd_p50": q(dd_s, .5), "dd_p95": q(dd_s, .95),
            "prob_loss_pct": sum(1 for x in tot_s if x < 0) / n_sims * 100,
            "prob_beat_pct": beat / n_sims * 100}


def catalogue(hist: History, bt: BacktestConfig, p: MomParams | None = None, wf: dict | None = None,
              n_sims: int = 2000) -> dict:
    """Catálogo ano a ano (cada ano calendário isolado, parâmetros fixos) + Monte Carlo."""
    p = p or MomParams()
    wf = wf or walk_forward(hist, bt, p=p)
    p = replace(p, market=wf["params"]["market"])
    symbols = [s for s in bt.symbols if s in hist.ohlcv]
    s0 = wf["s0"]
    years: dict[int, list[int]] = {}
    for i in range(s0, hist.n):
        y = _dt.datetime.fromtimestamp(hist.timeline[i] / 1000, _dt.timezone.utc).year
        years.setdefault(y, []).append(i)
    rows = []
    for y, ix in sorted(years.items()):
        a, b = ix[0], ix[-1] + 1
        if b - a < 30:
            continue
        m = simulate(hist, bt, p, a, b)
        bh = buy_and_hold(hist, symbols, a, b, bt)
        rows.append({"year": y, "days": b - a, "ret": m["return_pct"], "dd": m["max_drawdown_pct"],
                     "exposure": m["exposure_pct"], "orders": m["n_trades"],
                     "bh": bh["equal_weight_return_pct"], "btc": bh.get("btc_return_pct")})
    mc = None
    try:
        sr = _curve_returns(wf["curve"], bt.bankroll_cents / 100.0)
        br = _ew_daily_returns(hist, symbols, s0, hist.n)
        mc = block_bootstrap(sr, br, n_sims=n_sims)
    except BacktestError:
        pass
    return {"years": rows, "mc": mc}


def format_catalogue(cat: dict) -> str:
    L = ["", "=== CATÁLOGO ANO A ANO (cada ano isolado, parâmetros fixos, sem otimizar) ==="]
    L.append(f"{'ano':<6}{'estratégia':>12}{'queda máx.':>12}{'exposição':>11}{'ordens':>8}{'parado(igual)':>15}{'só BTC':>10}  resultado")
    win = 0
    for r in cat["years"]:
        beat = r["bh"] is not None and r["ret"] > r["bh"]
        win += 1 if beat else 0
        L.append(f"{r['year']:<6}{_pct(r['ret']):>12}{r['dd']:>11.1f}%{r['exposure']:>10.0f}%{r['orders']:>8}"
                 f"{_pct(r['bh']):>15}{_pct(r['btc']):>10}  {'ganhou do parado' if beat else 'perdeu do parado'}")
    if cat["years"]:
        pos = sum(1 for r in cat["years"] if r["ret"] > 0)
        L.append(f"anos positivos: {pos}/{len(cat['years'])} | anos em que bateu a carteira parada: {win}/{len(cat['years'])}")
    mc = cat.get("mc")
    if mc:
        L += [f"\n=== MONTE CARLO (bootstrap em blocos de {mc['block']} dias, {mc['n_sims']} sorteios, {mc['days']} dias) ===",
              f"retorno total da estratégia: pior 5% {_pct(mc['strat_p5'])} | mediana {_pct(mc['strat_p50'])} | melhor 5% {_pct(mc['strat_p95'])}",
              f"carteira igual rebalanceada todo dia (mediana): {_pct(mc['bench_p50'])}",
              f"queda máxima: mediana {mc['dd_p50']:.0f}% | caso ruim (95%) {mc['dd_p95']:.0f}%",
              f"chance de terminar no prejuízo: {mc['prob_loss_pct']:.0f}% | chance de bater a carteira rebalanceada: {mc['prob_beat_pct']:.0f}%",
              "(o Monte Carlo só reembaralha dias que já aconteceram: mostra sorte vs. robustez, não prevê o futuro)"]
    return "\n".join(L)


# ---------------------------------------------------------------- sensibilidade dos parâmetros
SENS_TREND = (100, 150, 200, 250, 300)
SENS_LONG = (63, 126, 189)


def sensitivity(hist: History, bt: BacktestConfig, base: MomParams | None = None,
                trends=SENS_TREND, longs=SENS_LONG) -> dict:
    """Roda o período inteiro com várias combinações de parâmetros, TODAS no mesmo período. Se só uma combinação
    dá lucro, o resultado era sorte de parâmetro; se quase todas dão, o efeito é robusto."""
    base = _resolve_market(hist, [s for s in bt.symbols if s in hist.ohlcv], base or MomParams())
    symbols = [s for s in bt.symbols if s in hist.ohlcv]
    if base.market not in hist.ohlcv:
        raise BacktestError(f"o filtro de mercado precisa do histórico de {base.market}")
    grid = [replace(base, trend_n=t, long=lg, short=max(10, lg // 2)) for t in trends for lg in longs]
    s0 = max(min(hist.first_idx[s] for s in symbols), hist.first_idx[base.market]) + max(g.need() for g in grid)
    if hist.n - s0 < 200:
        raise BacktestError("histórico curto para a sensibilidade: baixe mais dias")
    bh = buy_and_hold(hist, symbols, s0, hist.n, bt)
    rows = []
    for g in grid:
        m = simulate(hist, bt, g, s0, hist.n)
        rows.append({"trend_n": g.trend_n, "long": g.long, "ret": m["return_pct"], "dd": m["max_drawdown_pct"],
                     "orders": m["n_trades"], "exposure": m["exposure_pct"]})
    return {"rows": rows, "bh": bh["equal_weight_return_pct"], "btc": bh.get("btc_return_pct"),
            "bh_dd": bh["max_drawdown_pct"], "period": [hist.timeline[s0], hist.timeline[hist.n - 1]]}


def format_sensitivity(sv: dict) -> str:
    L = ["", f"=== SENSIBILIDADE DOS PARÂMETROS ({_date(sv['period'][0])}..{_date(sv['period'][1])}, tudo no mesmo período) ===",
         f"{'média(dias)':<12}{'momentum(dias)':>15}{'retorno':>12}{'queda máx.':>12}{'exposição':>11}{'ordens':>8}"]
    for r in sv["rows"]:
        L.append(f"{r['trend_n']:<12}{r['long']:>15}{_pct(r['ret']):>12}{r['dd']:>11.1f}%{r['exposure']:>10.0f}%{r['orders']:>8}")
    rows = sv["rows"]
    n = len(rows)
    pos = sum(1 for r in rows if r["ret"] > 0)
    beat = sum(1 for r in rows if sv["bh"] is not None and r["ret"] > sv["bh"])
    beat_btc = sum(1 for r in rows if sv["btc"] is not None and r["ret"] > sv["btc"])
    L += [f"comprar e segurar: {_pct(sv['bh'])} (queda máx. {sv['bh_dd']:.1f}%) | só BTC {_pct(sv['btc'])}",
          f"combinações com lucro: {pos}/{n} | acima do parado: {beat}/{n} | acima de só BTC: {beat_btc}/{n} | "
          f"pior retorno {_pct(min(r['ret'] for r in rows))} | pior queda {max(r['dd'] for r in rows):.1f}%",
          "LEITURA: " + ("robusto — quase todas as combinações lucram e superam o parado." if pos >= 0.9 * n and beat >= 0.7 * n
                         else "frágil — o resultado depende de escolher a combinação certa." if pos < 0.6 * n
                         else "misto — funciona em parte das combinações; não confie num número só.")]
    return "\n".join(L)
