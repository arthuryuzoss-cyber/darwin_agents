"""Backtest: replay do histórico pelo MESMO orquestrador do robô (varredura, ordens, taxas, fusível, darwinismo).

Garantias de método (cada uma tem teste):
- SEM olhar o futuro: a cada passo o mercado só entrega candles até o passo atual.
- Mesmo código de ordens: SimExchange + MultiGateway + Orchestrator, com taxa (0,1%), derrapagem (0,05%),
  ordem mínima e passo de lote reais do par (lidos no download).
- Validação fora da amostra (walk-forward): parâmetros são escolhidos SÓ com o período de treino e medidos num
  período seguinte que a busca nunca viu. O resultado de treino é otimista por construção; o que vale é o teste.
- Benchmarks (comprar e segurar) e intervalo de confiança do ganho por trade (bootstrap).

Limites que NENHUM backtest remove: não inclui sentimento de notícias (não há histórico), usa só pares que existem
hoje (viés de sobrevivência: moedas que sumiram não entram), executa no preço de fechamento do candle e não
modela o impacto de ordens grandes. Resultado passado não prevê o futuro.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from .breaker import CircuitBreaker
from .budget import RiskBudget, RiskConfig
from .config import Config
from .evolution import Darwinism, EvolutionConfig
from .exchange import MultiGateway
from .guardrails import OrderGuard, OrderGuardConfig
from .history import TF_MS, History, load_history
from .ledger import Ledger
from .orchestrator import Orchestrator
from .sim import SimExchange
from .strategy import BOUNDS, INT_PARAMS, TREND_MAX, TREND_MIN, StrategyParams

WARMUP = 100          # candles de aquecimento (o robô lê 100 candles por ativo)


class BacktestError(Exception):
    pass


# --------------------------------------------------------------------------- mercado de replay
class ReplayMarket:
    """Fonte de dados com a interface do ccxt, presa a um cursor: nunca devolve candles do futuro."""

    def __init__(self, hist: History, end: int, min_notional: float | None, default_min: float = 10.0):
        self.hist = hist
        self.end = end
        self.cursor = 0
        self._min = min_notional
        self._default_min = default_min

    def now_s(self) -> float:
        return self.hist.timeline[self.cursor] / 1000.0

    def load_markets(self):
        return {}

    def market(self, symbol: str) -> dict:
        if symbol not in self.hist.ohlcv:
            raise KeyError(f"BadSymbol {symbol}")
        meta = self.hist.meta.get(symbol, {})
        mn = self._min if self._min is not None else meta.get("min_cost", self._default_min)
        out = {"limits": {"cost": {"min": float(mn)}}}
        if meta.get("amount_step"):
            out["precision"] = {"amount": float(meta["amount_step"])}
        return out

    def _row(self, symbol: str) -> tuple:
        if self.cursor >= self.end:
            raise RuntimeError("cursor além do fim do período (bloqueio de look-ahead)")
        row = self.hist.ohlcv[symbol][self.cursor]
        if row is None:
            raise RuntimeError(f"sem dados de {symbol} neste instante")
        return row

    def fetch_ticker(self, symbol: str) -> dict:
        return {"symbol": symbol, "last": self._row(symbol)[3]}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int | None = None):
        self._row(symbol)                                  # valida cursor/dado
        col = self.hist.ohlcv[symbol]
        lo = max(self.hist.first_idx[symbol], self.cursor - (limit or WARMUP) + 1)
        out = []
        for i in range(lo, self.cursor + 1):              # SÓ até o cursor
            r = col[i]
            if r is None:
                continue
            out.append([self.hist.timeline[i], r[0], r[1], r[2], r[3], r[4]])
        return out


# --------------------------------------------------------------------------- configuração e resultado
@dataclass(frozen=True)
class BacktestConfig:
    symbols: tuple = ()
    bankroll_cents: int = 5000
    max_loss_pct: int = 60
    timeframe: str = "5m"
    agents: int = 1
    evolve: bool = False                 # False = mede UMA estratégia fixa; True = sistema completo com darwinismo
    eval_min_roundtrips: int = 3
    eval_max_ticks: int = 400
    max_population: int = 4
    max_generation: int = 5
    idle_kills: bool = False
    death_tolerance_pct: float = 0.0
    needs_min_trades: bool = False
    max_orders_per_hour: int = 12
    fee_rate: float = 0.001
    slippage: float = 0.0005
    min_notional: float | None = None    # None = valor real de cada par (meta.json); senão 10,0
    seed: int = 1
    params: dict | None = None           # StrategyParams.to_dict(); None = padrão


@dataclass
class Result:
    metrics: dict
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)      # [(ts_ms, equity_cents)]
    start: int = 0
    end: int = 0
    stop_reason: str = ""


def steps_per_year(tf: str) -> float:
    return 365.0 * 86_400_000 / TF_MS[tf]


def effective_start(hist: History, symbols) -> int:
    return min(hist.first_idx[s] for s in symbols) + WARMUP


# --------------------------------------------------------------------------- estatística
def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 1, alpha: float = 0.05):
    """IC (1-alpha) da MÉDIA por reamostragem. None com poucos dados (< 5)."""
    if len(values) < 5:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2)) - 1]


def max_drawdown_pct(curve: list[float]) -> float:
    peak, dd = -math.inf, 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            dd = max(dd, (peak - v) / peak * 100.0)
    return dd


def sharpe(curve: list[float], tf: str) -> float | None:
    if len(curve) < 3:
        return None
    rets = [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve)) if curve[i - 1] > 0]
    if len(rets) < 2:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    if var <= 0:
        return None
    return m / math.sqrt(var) * math.sqrt(steps_per_year(tf))


def trade_stats(trades: list[dict]) -> dict:
    pnl = [t["pnl_cents"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    by_asset: dict = {}
    by_why: dict = {}
    for t in trades:
        a = by_asset.setdefault(t["symbol"], {"n": 0, "pnl_cents": 0, "wins": 0})
        a["n"] += 1
        a["pnl_cents"] += t["pnl_cents"]
        a["wins"] += 1 if t["pnl_cents"] > 0 else 0
        by_why[t["why"]] = by_why.get(t["why"], 0) + 1
    ci = bootstrap_ci([float(p) for p in pnl])
    gl = -sum(losses)
    return {
        "n_trades": len(pnl), "n_wins": len(wins), "n_losses": len(losses),
        "win_rate_pct": (len(wins) / len(pnl) * 100.0) if pnl else 0.0,
        "avg_win_cents": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss_cents": (sum(losses) / len(losses)) if losses else 0.0,
        "profit_factor": (sum(wins) / gl) if gl > 0 else None,
        "expectancy_cents": (sum(pnl) / len(pnl)) if pnl else 0.0,
        "expectancy_ci95_cents": list(ci) if ci else None,
        "by_asset": by_asset, "by_reason": by_why,
    }


# --------------------------------------------------------------------------- execução
def run_backtest(hist: History, bt: BacktestConfig, start: int | None = None, end: int | None = None) -> Result:
    symbols = [s for s in bt.symbols if s in hist.ohlcv]
    if not symbols:
        raise BacktestError("nenhum dos símbolos pedidos tem histórico")
    end = hist.n if end is None else min(end, hist.n)
    start = effective_start(hist, symbols) if start is None else max(start, effective_start(hist, symbols))
    if end - start < 10:
        raise BacktestError("período curto demais para o backtest")
    market = ReplayMarket(hist, end, bt.min_notional)
    sim = SimExchange(market, symbols, quote_balance=bt.bankroll_cents / 100, fee_rate=bt.fee_rate,
                      slippage=bt.slippage)
    gw = MultiGateway(sim, symbols, sleep=lambda s: None)
    cfg = Config(mode="paper", symbols=tuple(gw.symbols), bankroll_cents=bt.bankroll_cents,
                 max_loss_pct=bt.max_loss_pct, timeframe=bt.timeframe, tick_seconds=5, initial_agents=bt.agents,
                 eval_min_roundtrips=bt.eval_min_roundtrips, eval_max_ticks=bt.eval_max_ticks,
                 max_population=bt.max_population, max_generation=bt.max_generation,
                 max_orders_per_hour=bt.max_orders_per_hour, idle_kills=bt.idle_kills,
                 death_tolerance_pct=bt.death_tolerance_pct, needs_min_trades=bt.needs_min_trades)
    ledger = Ledger()
    breaker = CircuitBreaker(None, None, bt.max_loss_pct)
    risk = RiskBudget(RiskConfig(bt.bankroll_cents, bt.max_loss_pct))
    guard = OrderGuard(OrderGuardConfig(frozenset(gw.symbols), bt.bankroll_cents * 2, bt.max_orders_per_hour),
                       clock=market.now_s)             # relógio SIMULADO: o limite por hora vale no tempo do histórico
    if bt.evolve:
        evo = EvolutionConfig(bt.eval_min_roundtrips, bt.eval_max_ticks, bt.max_population, bt.max_generation,
                              idle_kills=bt.idle_kills, death_tolerance_pct=bt.death_tolerance_pct,
                              needs_min_trades=bt.needs_min_trades)
    else:
        evo = EvolutionConfig(10 ** 9, 10 ** 9, 1, 1)  # janela que nunca fecha: estratégia fixa, sem seleção
    darwin = Darwinism(ledger, evo, risk, random.Random(bt.seed))
    orch = Orchestrator(gw, ledger, breaker, risk, guard, darwin, None, cfg, state_path=None,
                        sleep=lambda s: None, log=lambda m: None)
    params = StrategyParams.from_dict(bt.params) if bt.params else None
    market.cursor = start
    orch.bootstrap(max(1, bt.agents), params)

    equity: list[tuple[int, int]] = []
    seen = 0
    in_pos = 0
    last_eq = bt.bankroll_cents
    n_ticks = 0
    for i in range(start, end):
        market.cursor = i
        rep = orch.tick()
        n_ticks += 1
        if rep.equity_cents is not None:
            last_eq = rep.equity_cents
        equity.append((hist.timeline[i], last_eq))
        for t in orch.trade_log[seen:]:
            t["i"], t["ts"] = i, hist.timeline[i]
        seen = len(orch.trade_log)
        if any(t.in_position for t in orch.traders.values()):
            in_pos += 1
        if orch.stop_reason or rep.halted:
            break
    market.cursor = min(market.cursor, end - 1)
    try:                                              # patrimônio final marcado a mercado, já descontando a saída
        final = gw.equity_cents()
        held = 0.0
        for s, (tot, _) in gw.holdings().items():
            if tot > 0:
                held += tot * gw.last_price(s)
        final -= round(held * 100 * (bt.fee_rate + bt.slippage))
    except Exception:
        final = last_eq
    fees = 0.0
    for o in sim.orders.values():
        base = o["symbol"].split("/")[0]
        for f in o.get("fees") or []:
            fees += f["cost"] * (o["average"] if f["currency"] == base else 1.0)
    curve = [e for _, e in equity] + [final]
    ts = trade_stats(orch.trade_log)
    ret = (final - bt.bankroll_cents) / bt.bankroll_cents * 100.0
    metrics = {
        "initial_cents": bt.bankroll_cents, "final_cents": final, "return_pct": ret,
        "max_drawdown_pct": max_drawdown_pct(curve), "sharpe": sharpe(curve, bt.timeframe),
        "exposure_pct": in_pos / max(1, n_ticks) * 100.0, "fees_cents": round(fees * 100),
        "ticks": n_ticks, "steps": end - start,
        "stop_reason": orch.stop_reason, "agents_alive": len(orch.traders), "agents_dead": len(orch.graveyard),
        "clones": sum(1 for t in orch.traders.values() if t.parent) + sum(1 for g in orch.graveyard if g.get("parent")),
        "fuse": orch.stop_reason == "fuse",
        "period": [hist.timeline[start], hist.timeline[min(end, hist.n) - 1]],
        **ts,
    }
    return Result(metrics, list(orch.trade_log), equity, start, end, orch.stop_reason)


def buy_and_hold(hist: History, symbols, start: int, end: int, bt: BacktestConfig) -> dict:
    """Referência: dividir a banca igualmente entre os ativos que existem em `start` e segurar até `end`."""
    syms = [s for s in symbols if s in hist.ohlcv and hist.ohlcv[s][start] is not None]
    if not syms:
        return {"equal_weight_return_pct": None, "max_drawdown_pct": None, "btc_return_pct": None}
    k = ((1 - bt.fee_rate) / (1 + bt.slippage)) * ((1 - bt.slippage) * (1 - bt.fee_rate))   # compra + venda

    def px(s: str, i: int) -> float:
        for j in range(min(i, hist.n - 1), -1, -1):                   # último preço conhecido
            r = hist.ohlcv[s][j]
            if r is not None:
                return r[3]
        return hist.ohlcv[s][start][3]
    curve = []
    for i in range(start, end):
        curve.append(sum(px(s, i) / px(s, start) for s in syms) / len(syms))
    ret = (curve[-1] * k - 1.0) * 100.0
    out = {"equal_weight_return_pct": ret, "max_drawdown_pct": max_drawdown_pct(curve), "assets": len(syms),
           "btc_return_pct": None}
    for b in ("BTC/BRL", "BTC/USDT"):
        if b in syms:
            out["btc_return_pct"] = (px(b, end - 1) / px(b, start) * k - 1.0) * 100.0
            break
    return out


def stress_test(hist: History, bt: BacktestConfig, start: int | None, end: int | None) -> dict:
    """Refaz o teste com taxas 2x e derrapagem 3x: mostra se a vantagem sobrevive a custos piores."""
    r = run_backtest(hist, replace(bt, fee_rate=bt.fee_rate * 2, slippage=bt.slippage * 3), start, end)
    return r.metrics


# --------------------------------------------------------------------------- otimização (walk-forward)
TREND_DEFAULT = 200      # período do filtro de tendência usado como candidato-base quando trend=True


def default_params(trend: bool = False) -> StrategyParams:
    return StrategyParams(trend_period=TREND_DEFAULT) if trend else StrategyParams()


def sample_params(rng: random.Random, trend: bool = False) -> StrategyParams:
    """Sorteia parâmetros uniformemente dentro de BOUNDS. O peso de sentimento fica no padrão: o backtest
    não tem notícias históricas, então esse parâmetro não pode ser avaliado aqui."""
    d: dict = {}
    for name, (lo, hi) in BOUNDS.items():
        if name == "sentiment_weight":
            d[name] = StrategyParams().sentiment_weight
            continue
        v = rng.uniform(lo, hi)
        d[name] = int(round(v)) if name in INT_PARAMS else round(v, 4)
    if d["fast"] >= d["slow"]:
        d["fast"] = d["slow"] - 1
    if trend:
        d["trend_period"] = rng.randint(TREND_MIN, TREND_MAX)
    return StrategyParams.from_dict(d)


def score(m: dict, min_trades: int) -> float:
    """Nota de treino: retorno menos metade do drawdown. Poucos trades = desclassificado."""
    if m["n_trades"] < min_trades:
        return -1e9 + m["n_trades"]
    return m["return_pct"] - 0.5 * m["max_drawdown_pct"]


_G: dict = {}


def _init_worker(hist_dir: str, symbols: list, tf: str, bt: dict) -> None:
    bt = dict(bt)
    bt["symbols"] = tuple(bt["symbols"])
    _G["hist"] = load_history(hist_dir, symbols, tf)
    _G["bt"] = BacktestConfig(**bt)


def _work(spec: tuple) -> dict:
    params, a, b = spec
    res = run_backtest(_G["hist"], replace(_G["bt"], params=params, evolve=False), a, b)
    return res.metrics


def evaluate_many(hist: History, bt: BacktestConfig, specs: list[tuple], workers: int = 1,
                  hist_dir: str | None = None, progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """Roda vários (params, início, fim). workers > 1 usa processos (precisa de hist_dir para recarregar)."""
    out: list[dict] = []
    if workers > 1 and hist_dir:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(hist_dir, list(hist.symbols), hist.timeframe, asdict(bt))) as ex:
            for k, m in enumerate(ex.map(_work, specs, chunksize=1), 1):
                out.append(m)
                if progress:
                    progress(k, len(specs))
        return out
    for k, (params, a, b) in enumerate(specs, 1):
        out.append(run_backtest(hist, replace(bt, params=params, evolve=False), a, b).metrics)
        if progress:
            progress(k, len(specs))
    return out


def walk_forward(hist: History, bt: BacktestConfig, folds: int = 3, trials: int = 30, top_k: int = 3,
                 seed: int = 1, workers: int = 1, hist_dir: str | None = None, min_trades: int = 10,
                 train_frac: float = 0.6, progress: Callable[[str], None] | None = None,
                 trend: bool = False) -> dict:
    """Walk-forward com janela expansível.

    folds=1: treino = primeiros `train_frac` do período, teste = o resto.
    folds>1: a primeira metade é treino inicial; a segunda é dividida em `folds` blocos de teste. Cada bloco é
    testado com parâmetros escolhidos SÓ com tudo que veio antes dele.
    Em cada fold: `trials` conjuntos sorteados (+ os padrões) são medidos no treino; os `top_k` melhores e os
    padrões são medidos no teste. O parâmetro "escolhido" é o de melhor nota de treino (decisão tomada ANTES de
    ver o teste); os outros top_k aparecem só para mostrar quão frágil é a escolha.
    """
    symbols = [s for s in bt.symbols if s in hist.ohlcv]
    s0 = effective_start(hist, symbols)
    total = hist.n - s0
    if total < 400 * max(1, folds):
        raise BacktestError("histórico curto para esse número de folds (baixe mais dias)")
    if folds == 1:
        cuts = [(s0, s0 + int(total * train_frac), hist.n)]
    else:
        base = s0 + total // 2
        size = (hist.n - base) // folds
        cuts = [(s0, base + k * size, base + (k + 1) * size if k < folds - 1 else hist.n) for k in range(folds)]
    rng = random.Random(seed)
    say = progress or (lambda m: None)
    out_folds = []
    for fi, (tr_a, tr_b, te_b) in enumerate(cuts, 1):
        cands = [default_params(trend)] + [sample_params(rng, trend) for _ in range(trials)]
        say(f"fold {fi}/{len(cuts)}: treinando {len(cands)} conjuntos de parâmetros…")
        tr = evaluate_many(hist, bt, [(c.to_dict(), tr_a, tr_b) for c in cands], workers, hist_dir)
        notes = [score(m, min_trades) for m in tr]
        order = sorted(range(len(cands)), key=lambda k: notes[k], reverse=True)
        chosen = order[:top_k]
        say(f"fold {fi}/{len(cuts)}: testando fora da amostra…")
        te_specs = [(cands[k].to_dict(), tr_b, te_b) for k in chosen]
        te_specs.append((default_params(trend).to_dict(), tr_b, te_b))
        te = evaluate_many(hist, bt, te_specs, workers, hist_dir)
        out_folds.append({
            "train": [tr_a, tr_b], "test": [tr_b, te_b],
            "train_period": [hist.timeline[tr_a], hist.timeline[tr_b - 1]],
            "test_period": [hist.timeline[tr_b], hist.timeline[te_b - 1]],
            "chosen_params": cands[chosen[0]].to_dict(),
            "chosen_train": _brief(tr[chosen[0]]), "chosen_test": _brief(te[0]),
            "top": [{"params": cands[k].to_dict(), "train": _brief(tr[k]), "test": _brief(te[j])}
                    for j, k in enumerate(chosen)],
            "default_train": _brief(tr[0]), "default_test": _brief(te[-1]),
            "buy_hold_test": buy_and_hold(hist, symbols, tr_b, te_b, bt),
            "n_trials": len(cands),
            "train_scores_pct": _percentiles([n for n in notes if n > -1e8]),
        })
    return {"folds": out_folds, "aggregate": _aggregate(out_folds), "min_trades": min_trades, "trend": trend}


def _brief(m: dict) -> dict:
    keys = ("return_pct", "max_drawdown_pct", "n_trades", "win_rate_pct", "expectancy_cents", "profit_factor",
            "fees_cents", "fuse", "exposure_pct")
    d = {k: m.get(k) for k in keys}
    d["expectancy_ci95_cents"] = m.get("expectancy_ci95_cents")
    return d


def _percentiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    v = sorted(vals)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "max": v[-1]}


def _compound(rets: list[float]) -> float:
    x = 1.0
    for r in rets:
        x *= 1 + r / 100.0
    return (x - 1) * 100.0


def _aggregate(folds: list[dict]) -> dict:
    oos = [f["chosen_test"]["return_pct"] for f in folds]
    dflt = [f["default_test"]["return_pct"] for f in folds]
    bh = [f["buy_hold_test"]["equal_weight_return_pct"] for f in folds if f["buy_hold_test"]["equal_weight_return_pct"] is not None]
    tr = [f["chosen_train"]["return_pct"] for f in folds]
    return {
        "oos_compounded_pct": _compound(oos), "default_compounded_pct": _compound(dflt),
        "buy_hold_compounded_pct": _compound(bh) if bh else None,
        "positive_folds": sum(1 for r in oos if r > 0), "n_folds": len(oos),
        "avg_train_return_pct": sum(tr) / len(tr), "avg_test_return_pct": sum(oos) / len(oos),
        "overfit_gap_pct": sum(tr) / len(tr) - sum(oos) / len(oos),
        "oos_trades": sum(f["chosen_test"]["n_trades"] for f in folds),
        "any_fuse": any(f["chosen_test"]["fuse"] for f in folds),
    }


def verdict(agg: dict) -> str:
    """Conclusão em palavras, deliberadamente cética."""
    oos, bh = agg["oos_compounded_pct"], agg["buy_hold_compounded_pct"]
    if agg["oos_trades"] < 20:
        return ("INCONCLUSIVO: poucos trades fora da amostra para tirar conclusão (a estatística não sustenta "
                "nem vantagem nem ausência dela). Baixe mais dias de histórico.")
    if oos <= 0:
        return ("SEM VANTAGEM: com os parâmetros escolhidos no treino, o resultado fora da amostra foi "
                f"{oos:+.1f}% (negativo ou zero). Não há motivo para operar essa estratégia com dinheiro real.")
    if agg["positive_folds"] < max(1, agg["n_folds"] // 2 + (agg["n_folds"] % 2)):
        return (f"FRACO: lucro fora da amostra de {oos:+.1f}%, mas só {agg['positive_folds']} de {agg['n_folds']} "
                "períodos foram positivos. Isso é compatível com sorte.")
    if bh is not None and oos <= bh:
        return (f"POSITIVO, MAS NÃO BATE COMPRAR-E-SEGURAR: {oos:+.1f}% contra {bh:+.1f}% de uma carteira parada. "
                "Com taxas e risco de operar, segurar teria sido melhor ou igual.")
    return (f"INDÍCIO (não é prova): {oos:+.1f}% fora da amostra, {agg['positive_folds']}/{agg['n_folds']} períodos "
            "positivos e acima da carteira parada. Valide por mais tempo em paper antes de qualquer dinheiro real; "
            "o futuro pode não repetir o passado.")


# --------------------------------------------------------------------------- relatórios
def _brl(c: float | int | None) -> str:
    return "—" if c is None else f"R$ {c / 100:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:+.1f}%"


def _date(ms: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")


def format_result(res: Result, bench: dict, stress: dict | None = None, title: str = "BACKTEST") -> str:
    m = res.metrics
    L = [f"=== {title} ===", f"período: {_date(m['period'][0])} a {_date(m['period'][1])} ({m['steps']} candles)"]
    L.append(f"banca {_brl(m['initial_cents'])} -> patrimônio final {_brl(m['final_cents'])} "
             f"({_pct(m['return_pct'])}) | pior queda {m['max_drawdown_pct']:.1f}% | exposição {m['exposure_pct']:.0f}% do tempo")
    if bench.get("equal_weight_return_pct") is not None:
        L.append(f"comprar e segurar (carteira igual dos {bench['assets']} ativos): {_pct(bench['equal_weight_return_pct'])}"
                 f" (pior queda {bench['max_drawdown_pct']:.1f}%)"
                 + (f" | só BTC: {_pct(bench['btc_return_pct'])}" if bench.get("btc_return_pct") is not None else ""))
    pf = m["profit_factor"]
    L.append(f"trades: {m['n_trades']} | acerto {m['win_rate_pct']:.0f}% | ganho médio {_brl(m['avg_win_cents'])} x perda média "
             f"{_brl(m['avg_loss_cents'])} | fator de lucro {'—' if pf is None else f'{pf:.2f}'} | taxas pagas ≈ {_brl(m['fees_cents'])}")
    ci = m["expectancy_ci95_cents"]
    if ci:
        zero = ci[0] <= 0 <= ci[1]
        L.append(f"ganho médio por trade {_brl(m['expectancy_cents'])} (IC95% {_brl(ci[0])} a {_brl(ci[1])})"
                 + ("  -> o intervalo inclui ZERO: indistinguível de sorte" if zero else
                    ("  -> intervalo acima de zero" if ci[0] > 0 else "  -> intervalo abaixo de zero")))
    else:
        L.append("ganho médio por trade: poucos trades para estimar confiança (< 5)")
    if m["by_asset"]:
        L.append("por ativo: " + " | ".join(f"{s.split('/')[0]} {v['n']}t {_brl(v['pnl_cents'])}"
                                            for s, v in sorted(m["by_asset"].items(), key=lambda kv: -kv[1]["pnl_cents"])))
    if m["by_reason"]:
        L.append("saídas: " + ", ".join(f"{k} {v}" for k, v in sorted(m["by_reason"].items())))
    L.append(f"agentes vivos {m['agents_alive']} | mortos {m['agents_dead']} | clones {m['clones']}"
             + (f" | PAROU: {m['stop_reason']}" if m["stop_reason"] else ""))
    if m["fuse"]:
        L.append("🔌 O FUSÍVEL DISPAROU neste período.")
    if stress:
        L.append(f"estresse (taxas 2x, derrapagem 3x): {_pct(stress['return_pct'])} (normal: {_pct(m['return_pct'])})")
    L.append("Lembrete: sem sentimento de notícias, só pares listados hoje (viés de sobrevivência), preço de fechamento.")
    return "\n".join(L)


def format_walk_forward(wf: dict, trials_note: bool = True) -> str:
    L = ["=== WALK-FORWARD (otimização com validação fora da amostra"
         + (", COM filtro de tendência" if wf.get("trend") else "") + ") ==="]
    for k, f in enumerate(wf["folds"], 1):
        L.append(f"\n-- Fold {k}: treino {_date(f['train_period'][0])}..{_date(f['train_period'][1])} | "
                 f"teste {_date(f['test_period'][0])}..{_date(f['test_period'][1])}")
        ct, cs = f["chosen_train"], f["chosen_test"]
        L.append(f"   escolhido no treino ({f['n_trials']} candidatos): treino {_pct(ct['return_pct'])} "
                 f"({ct['n_trades']} trades) -> TESTE {_pct(cs['return_pct'])} ({cs['n_trades']} trades, "
                 f"pior queda {cs['max_drawdown_pct']:.1f}%)")
        L.append(f"   parâmetros padrão no teste: {_pct(f['default_test']['return_pct'])} | "
                 f"comprar e segurar no teste: {_pct(f['buy_hold_test']['equal_weight_return_pct'])}")
        if len(f["top"]) > 1:
            L.append("   top candidatos (treino -> teste): " + ", ".join(
                f"{_pct(t['train']['return_pct'])} -> {_pct(t['test']['return_pct'])}" for t in f["top"]))
        sp = f.get("train_scores_pct") or {}
        if sp:
            L.append(f"   nota de treino dos candidatos: mediana {sp['p50']:.1f}, p90 {sp['p90']:.1f}, melhor {sp['max']:.1f} "
                     "(o melhor de muitos sorteios é otimista por construção)")
    a = wf["aggregate"]
    L.append("\n=== RESUMO FORA DA AMOSTRA ===")
    L.append(f"retorno composto nos testes: {_pct(a['oos_compounded_pct'])} | parâmetros padrão: {_pct(a['default_compounded_pct'])}"
             f" | comprar e segurar: {_pct(a['buy_hold_compounded_pct'])}")
    L.append(f"períodos de teste positivos: {a['positive_folds']}/{a['n_folds']} | trades fora da amostra: {a['oos_trades']}")
    L.append(f"treino médio {_pct(a['avg_train_return_pct'])} vs teste médio {_pct(a['avg_test_return_pct'])} "
             f"-> diferença (sobreajuste) {a['overfit_gap_pct']:+.1f} pontos")
    if a["any_fuse"]:
        L.append("🔌 O fusível disparou em algum período de teste.")
    L.append("\nCONCLUSÃO: " + verdict(a))
    return "\n".join(L)


def save_outputs(out_dir: str, res: Result | None = None, wf: dict | None = None, hist: History | None = None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    if res is not None:
        p = os.path.join(out_dir, "trades.csv")
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts_ms", "agent", "symbol", "pnl_cents", "why", "entry", "exit"])
            for t in res.trades:
                w.writerow([t.get("ts"), t["agent"], t["symbol"], t["pnl_cents"], t["why"], t["entry"], t["exit"]])
        paths.append(p)
        p = os.path.join(out_dir, "equity.csv")
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts_ms", "equity_cents"])
            w.writerows(res.equity)
        paths.append(p)
        p = os.path.join(out_dir, "summary.json")
        with open(p, "w", encoding="utf-8") as _f:
            json.dump(res.metrics, _f, indent=1, default=str)
        paths.append(p)
    if wf is not None:
        p = os.path.join(out_dir, "walk_forward.json")
        with open(p, "w", encoding="utf-8") as _f:
            json.dump(wf, _f, indent=1, default=str)
        paths.append(p)
        best = wf["folds"][-1]["chosen_params"]
        p = os.path.join(out_dir, "best_params.json")
        with open(p, "w", encoding="utf-8") as _f:
            json.dump(best, _f, indent=1)
        paths.append(p)
    return paths
