"""Utilitários de teste: mercado roteirizado e montagem do orquestrador com exchange simulada."""
import os
import random
import tempfile

from darwin.breaker import CircuitBreaker
from darwin.budget import RiskBudget, RiskConfig
from darwin.config import Config
from darwin.evolution import Darwinism, EvolutionConfig
from darwin.exchange import CcxtGateway, MultiGateway
from darwin.guardrails import OrderGuard, OrderGuardConfig
from darwin.ledger import Ledger
from darwin.orchestrator import Orchestrator
from darwin.sim import SimExchange
from darwin.strategy import StrategyParams


class ScriptedMarket:
    """Fonte de dados controlada pelo teste (preço e candles)."""

    def __init__(self, prices, min_notional=10.0):
        self.prices = list(prices)
        self.min_notional = min_notional

    def advance(self, price):
        self.prices.append(float(price))

    def load_markets(self):
        return {}

    def market(self, symbol):
        return {"limits": {"cost": {"min": self.min_notional}}}

    def fetch_ticker(self, symbol):
        return {"symbol": symbol, "last": self.prices[-1]}

    def fetch_ohlcv(self, symbol, timeframe, limit=None):
        rows = self.prices[-(limit or len(self.prices)):]
        return [[i * 60000, p, p, p, p, 1.0] for i, p in enumerate(rows)]


class MultiMarket:
    """Vários ativos: {"BTC/BRL": ScriptedMarket, ...}. Mesma interface de dados do ccxt."""

    def __init__(self, series, min_notional=10.0):
        self.m = {sym: ScriptedMarket(p, min_notional) for sym, p in series.items()}

    def advance(self, sym, price):
        self.m[sym].advance(price)

    def load_markets(self):
        return {}

    def market(self, symbol):
        return self.m[symbol].market(symbol)

    def fetch_ticker(self, symbol):
        return self.m[symbol].fetch_ticker(symbol)

    def fetch_ohlcv(self, symbol, timeframe, limit=None):
        return self.m[symbol].fetch_ohlcv(symbol, timeframe, limit)


def flat(n=70, price=100.0):
    return [price] * n


def uptrend(start=100.0, n=25, step=0.006):
    out, p = [], start
    for _ in range(n):
        p *= 1 + step
        out.append(round(p, 4))
    return out


class NoSentiment:
    def get(self):
        return type("R", (), {"value": None})()


class FixedSentiment:
    def __init__(self, v):
        self.v = v

    def get(self):
        return type("R", (), {"value": self.v})()


def make_cfg(**kw):
    if "symbol" in kw:
        kw["symbols"] = (kw.pop("symbol"),)
    base = dict(mode="paper", symbols=("BTC/BRL",), bankroll_cents=4000, max_loss_pct=60, tick_seconds=5,
                eval_min_roundtrips=1, eval_max_ticks=400, max_population=4, max_generation=5,
                max_orders_per_hour=1000, data_dir="data")
    base.update(kw)
    return Config(**base)


class Rig:
    """Orquestrador completo sobre SimExchange; tudo em diretório temporário."""

    def __init__(self, prices=None, quote=40.0, base=0.0, cfg=None, sentiment=None, fee_mode="received",
                 min_notional=10.0, seed=1, data_dir=None, sim=None, market=None, **cfg_kw):
        self.dir = data_dir or tempfile.mkdtemp()
        self.cfg = cfg or make_cfg(data_dir=self.dir, **cfg_kw)
        self.market = market or ScriptedMarket(prices or flat(), min_notional)
        self.sim = sim or SimExchange(self.market, list(self.cfg.symbols), quote_balance=quote, base_balance=base,
                                      fee_mode=fee_mode, min_notional=min_notional)
        self.logs = []
        self.build(sentiment or NoSentiment(), seed)

    def build(self, sentiment, seed=1):
        c = self.cfg
        self.ledger = Ledger(os.path.join(self.dir, "darwin.db"))
        self.gw = MultiGateway(self.sim, c.symbols, sleep=lambda s: None)
        self.breaker = CircuitBreaker(os.path.join(self.dir, "breaker.json"), "token-do-dono-123", c.max_loss_pct)
        self.risk = RiskBudget(RiskConfig(c.bankroll_cents, c.max_loss_pct))
        self.guard = OrderGuard(OrderGuardConfig(frozenset(self.gw.symbols), c.bankroll_cents * 2, c.max_orders_per_hour))
        evo = EvolutionConfig(c.eval_min_roundtrips, c.eval_max_ticks, c.max_population, c.max_generation,
                              idle_kills=c.idle_kills)
        self.darwin = Darwinism(self.ledger, evo, self.risk, random.Random(seed))
        self.orch = Orchestrator(self.gw, self.ledger, self.breaker, self.risk, self.guard, self.darwin,
                                 sentiment, c, state_path=os.path.join(self.dir, "state.json"),
                                 sleep=lambda s: None, log=self.logs.append)
        self.orch.load_state()
        return self.orch

    def drive(self, prices):
        reports = []
        for p in prices:
            self.market.advance(p)
            reports.append(self.orch.tick())
            if self.orch.stop_reason or reports[-1].halted:
                break
        return reports

    def tune(self, name="alpha", **kw):
        t = self.orch.traders[name]
        t.params = StrategyParams(**{**t.params.to_dict(), **kw})
        return t

    def text(self):
        return "\n".join(self.logs)
