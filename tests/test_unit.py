import math
import os
import random
import tempfile
import unittest

from darwin.breaker import BreakerError, CircuitBreaker, State
from darwin.budget import RiskBudget, RiskConfig, fuse_floor_cents, loss_pct
from darwin.config import ConfigError, load_config, parse_env_file
from darwin.evolution import Darwinism, EvolutionConfig, mutate_params
from darwin.exchange import CcxtGateway, ExchangeError, is_ambiguous_failure, to_cents
from darwin.guardrails import GuardError, OrderGuard, OrderGuardConfig, sanitize_headline
from darwin.ledger import Ledger, LedgerError
from darwin.sentiment import SentimentAnalyzer, parse_feed, parse_score
from darwin.sim import SimExchange, SimExchangeError
from darwin.strategy import BOUNDS, StrategyError, StrategyParams, decide, ema, rsi, tech_score
from tests.helpers import ScriptedMarket, flat, uptrend


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.l = Ledger()

    def test_seed_balance_and_int_only(self):
        self.l.seed("a", 2000)
        self.assertEqual(self.l.balance("a"), 2000)
        for bad in (10.5, -1, 0, True, "10", None):
            with self.assertRaises(LedgerError):
                self.l.seed("a", bad)

    def test_record_trade_gain_loss_zero(self):
        self.l.seed("a", 2000)
        self.l.record_trade("a", 150, "O1")
        self.l.record_trade("a", -400, "O2")
        self.assertIsNone(self.l.record_trade("a", 0, "O3"))
        self.assertEqual(self.l.balance("a"), 1750)
        self.assertEqual(self.l.total("a", "trade_gain"), 150)
        self.assertEqual(self.l.total("a", "trade_loss"), 400)
        self.assertEqual(self.l.global_balance(), 1750)

    def test_same_order_never_counted_twice(self):
        self.l.record_trade("a", 100, "O1")
        with self.assertRaises(LedgerError):
            self.l.record_trade("a", 100, "O1")

    def test_trade_validation(self):
        for bad in (1.5, True, None):
            with self.assertRaises(LedgerError):
                self.l.record_trade("a", bad, "O")
        with self.assertRaises(LedgerError):
            self.l.record_trade("a", 5, " ")

    def test_transfer_conserves_money(self):
        self.l.seed("a", 1000)
        self.l.transfer("a", "b", 400)
        self.assertEqual((self.l.balance("a"), self.l.balance("b")), (600, 400))
        self.assertEqual(self.l.global_balance(), 1000)
        with self.assertRaises(LedgerError):
            self.l.transfer("a", "b", 601)

    def test_append_only(self):
        self.l.seed("a", 100)
        with self.assertRaises(Exception):
            self.l._db.execute("UPDATE entries SET amount_cents=1")
        with self.assertRaises(Exception):
            self.l._db.execute("DELETE FROM entries")


class BudgetTests(unittest.TestCase):
    def test_fuse_floor_for_20_reais_at_60pct_is_8(self):
        self.assertEqual(fuse_floor_cents(2000, 60), 800)

    def test_floor_rounds_up_so_it_trips_earlier(self):
        self.assertEqual(fuse_floor_cents(2001, 60), 801)  # 800.4 -> 801

    def test_floor_validation(self):
        for a, b in ((0, 60), (-5, 60), (2000, 0), (2000, 100), (2000, True), (20.0, 60)):
            with self.assertRaises(ValueError):
                fuse_floor_cents(a, b)

    def test_loss_pct(self):
        self.assertAlmostEqual(loss_pct(2000, 800), 60.0)
        self.assertAlmostEqual(loss_pct(2000, 2200), -10.0)

    def test_sizing_respects_min_and_fraction_cap(self):
        rb = RiskBudget(RiskConfig(2000))
        self.assertEqual(rb.min_trade_cents(10.0), 1200)
        self.assertEqual(rb.size_buy_cents(2000, 1200, 0.95), 1900)
        self.assertEqual(rb.size_buy_cents(2000, 1200, 5.0), 1960)  # teto 0.98
        self.assertEqual(rb.size_buy_cents(1000, 1200, 0.95), 0)    # abaixo do mínimo

    def test_small_agent_buys_the_exchange_minimum_instead_of_nothing(self):
        """Regressão: filho com capital = mínimo da exchange nunca conseguia comprar (0,95 x mínimo < mínimo)."""
        rb = RiskBudget(RiskConfig(2000))
        self.assertEqual(rb.viable_capital_cents(1200), 1225)       # 1200 / 0,98 arredondado para cima
        self.assertEqual(rb.size_buy_cents(1225, 1200, 0.30), 1200)  # fração baixa: compra o mínimo
        self.assertEqual(rb.size_buy_cents(1225, 1200, 0.95), 1200)
        self.assertEqual(rb.size_buy_cents(1224, 1200, 0.95), 0)     # nem o mínimo cabe sob o teto de 98%

    def test_can_fund_child_needs_both_sides(self):
        rb = RiskBudget(RiskConfig(2000))
        need = rb.viable_capital_cents(1050)                         # 1072
        self.assertTrue(rb.can_fund_child(2 * need, need, 1050))
        self.assertFalse(rb.can_fund_child(2 * need - 1, need, 1050))
        self.assertFalse(rb.can_fund_child(5000, need - 1, 1050))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            RiskConfig(0)
        with self.assertRaises(ValueError):
            RiskConfig(2000, max_loss_pct=100)


class BreakerTests(unittest.TestCase):
    def mk(self, path=None, token="segredo-do-dono"):
        return CircuitBreaker(path, token, max_loss_pct=60)

    def test_trips_exactly_at_floor_not_before(self):
        b = self.mk()
        b.set_initial_equity(2000)
        self.assertEqual(b.check_equity(801), State.CLOSED)
        self.assertEqual(b.check_equity(800), State.OPEN)
        self.assertIn("perda de 60.0%", b.reason)

    def test_callbacks_run_once_and_errors_do_not_stop_others(self):
        b = self.mk()
        b.set_initial_equity(2000)
        calls = []
        b.on_trip(lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
        b.on_trip(lambda r: calls.append(r))
        b.check_equity(100)
        b.check_equity(50)
        self.assertEqual(len(calls), 1)

    def test_state_persisted_before_callbacks_and_survives_restart(self):
        path = os.path.join(tempfile.mkdtemp(), "b.json")
        b = self.mk(path)
        b.set_initial_equity(2000)
        seen = {}
        b.on_trip(lambda r: seen.update(exists=os.path.exists(path), pending=b.liquidation_pending))
        b.check_equity(700)
        self.assertTrue(seen["exists"] and seen["pending"])
        b2 = self.mk(path)
        self.assertTrue(b2.is_open() and b2.liquidation_pending)
        self.assertEqual(b2.initial_equity_cents, 2000)

    def test_mark_liquidated_persists(self):
        path = os.path.join(tempfile.mkdtemp(), "b.json")
        b = self.mk(path)
        b.set_initial_equity(2000)
        b.check_equity(100)
        b.mark_liquidated()
        self.assertFalse(self.mk(path).liquidation_pending)

    def test_baseline_immutable_without_reset(self):
        b = self.mk()
        b.set_initial_equity(2000)
        b.set_initial_equity(2000)
        with self.assertRaises(BreakerError):
            b.set_initial_equity(3000)

    def test_invalid_equity_rejected(self):
        b = self.mk()
        b.set_initial_equity(2000)
        for bad in (-1, 1.5, True, None):
            with self.assertRaises(BreakerError):
                b.check_equity(bad)
        with self.assertRaises(BreakerError):
            self.mk().check_equity(1000)  # sem baseline

    def test_consecutive_monitoring_failures_trip(self):
        b = CircuitBreaker(None, None, 60, max_consecutive_failures=3)
        b.record_result(False); b.record_result(False); b.record_result(True)
        b.record_result(False); b.record_result(False)
        self.assertFalse(b.is_open())
        b.record_result(False)
        self.assertTrue(b.is_open())

    def test_failure_streaks_are_per_channel(self):
        b = CircuitBreaker(None, None, 60, max_consecutive_failures=3)
        for _ in range(2):
            b.record_result(False, "mercado")
            b.record_result(True, "patrimônio")  # canal saudável NÃO zera o do mercado
        self.assertFalse(b.is_open())
        b.record_result(False, "mercado")
        self.assertTrue(b.is_open())
        self.assertIn("mercado", b.reason)

    def test_reset_rules(self):
        b = self.mk()
        b.set_initial_equity(2000)
        b.check_equity(100)
        self.assertFalse(b.reset("errado"))
        self.assertFalse(b.reset(None))
        self.assertFalse(b.reset("segredo-do-dono"))      # liquidação ainda pendente
        b.mark_liquidated()
        self.assertTrue(b.reset("segredo-do-dono"))
        self.assertEqual(b.check_equity(100), State.OPEN)  # baseline mantida: dispara de novo

    def test_reset_with_rebase(self):
        b = self.mk()
        b.set_initial_equity(2000)
        b.check_equity(100)
        b.mark_liquidated()
        self.assertTrue(b.reset("segredo-do-dono", rebase_equity_cents=900))
        self.assertEqual(b.check_equity(900), State.CLOSED)
        self.assertFalse(b.reset("segredo-do-dono", rebase_equity_cents=-5))

    def test_reset_disabled_without_token(self):
        b = self.mk(token=None)
        b.set_initial_equity(2000)
        b.check_equity(1)
        b.mark_liquidated()
        self.assertFalse(b.reset(""))
        self.assertFalse(b.reset("x"))


class StrategyTests(unittest.TestCase):
    def test_ema_and_rsi_known_values(self):
        self.assertAlmostEqual(ema([5.0] * 30, 10), 5.0)
        self.assertEqual(rsi(list(range(1, 40)), 14), 100.0)
        self.assertEqual(rsi([10.0] * 30, 14), 50.0)
        self.assertLess(rsi(list(range(40, 1, -1)), 14), 1.0)
        with self.assertRaises(StrategyError):
            ema([1, 2], 5)

    def test_tech_score_sign_and_range(self):
        p = StrategyParams()
        up, down = flat(60) + uptrend(n=25), flat(60) + uptrend(n=25, step=-0.006)
        self.assertGreater(tech_score(up, p), 0.2)
        self.assertLess(tech_score(down, p), -0.2)
        for s in (tech_score(up, p), tech_score(down, p)):
            self.assertTrue(-1 <= s <= 1)

    def test_decisions(self):
        p = StrategyParams(buy_threshold=0.1, rsi_overbought=85.0)
        up = flat(60) + uptrend(n=25)
        self.assertEqual(decide(p, up, None, up[-1], False).action, "buy")
        self.assertEqual(decide(p, up, None, up[-1], True, entry_price=up[-1] * 0.99).action, "hold")
        self.assertEqual(decide(p, up, None, 97.0, True, entry_price=100.0).reason, "stop_loss")
        self.assertEqual(decide(p, up, None, 104.0, True, entry_price=100.0).reason, "take_profit")
        self.assertEqual(decide(p, up[:5], None, 100, False).reason, "dados_insuficientes")

    def test_sentiment_moves_decision(self):
        p = StrategyParams(sentiment_weight=0.8)
        c = flat(80)
        self.assertEqual(decide(p, c, 1.0, 100.0, False).action, "buy")
        self.assertEqual(decide(p, c, -1.0, 100.0, True, 100.0).action, "sell")
        self.assertEqual(decide(p, c, None, 100.0, False).action, "hold")  # sem notícias: só técnica

    def test_params_validation(self):
        for kw in ({"fast": 30, "slow": 20}, {"stop_loss_pct": 50.0}, {"sentiment_weight": float("nan")},
                   {"buy_threshold": True}):
            with self.assertRaises(StrategyError):
                StrategyParams(**kw)
        with self.assertRaises(StrategyError):
            StrategyParams.from_dict({"fast": 9, "evil": 1})
        self.assertEqual(StrategyParams.from_dict(StrategyParams().to_dict()), StrategyParams())


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.l = Ledger()
        self.l.seed("a", 4000)
        self.rb = RiskBudget(RiskConfig(4000))
        self.d = Darwinism(self.l, EvolutionConfig(eval_min_roundtrips=2, eval_max_ticks=50,
                                                   max_population=3, max_generation=3), self.rb, random.Random(3))

    def judge(self, **kw):
        a = dict(agent="a", gain0=0, loss0=0, roundtrips=2, ticks=10, in_position=False, population=1,
                 generation=1, min_trade_cents=1050)
        a.update(kw)
        return self.d.judge(**a)

    def test_continue_until_cycle_closes_and_flat(self):
        self.assertEqual(self.judge(roundtrips=1, ticks=10).action, "continue")
        self.l.record_trade("a", 300, "x")
        self.assertEqual(self.judge(in_position=True).action, "continue")
        self.assertEqual(self.judge(roundtrips=1, ticks=50).action, "clone")

    def test_positive_pnl_clones_funded_by_parent(self):
        self.l.record_trade("a", 300, "x")
        v = self.judge()
        self.assertEqual((v.action, v.pnl_cents), ("clone", 300))
        self.assertEqual(v.child_seed_cents, 1072)  # lucro (300) < capital viável => filho recebe o mínimo VIÁVEL

    def test_big_profit_funds_bigger_child_within_limits(self):
        self.l.record_trade("a", 1500, "x")
        self.assertEqual(self.judge().child_seed_cents, 1500)

    def test_cycle_with_no_trades_is_idle_not_dead_unless_configured(self):
        v = self.judge(roundtrips=0, ticks=50)           # mercado parado: sem PnL medido, sem veredito
        self.assertEqual((v.action, v.pnl_cents), ("idle", 0))
        strict = Darwinism(self.l, EvolutionConfig(eval_min_roundtrips=2, eval_max_ticks=50, max_population=3,
                                                   max_generation=3, idle_kills=True), self.rb, random.Random(3))
        self.assertEqual(strict.judge("a", 0, 0, 0, 50, False, 1, 1, 1050).action, "die")  # regra literal

    def test_zero_and_negative_pnl_die(self):
        self.assertEqual(self.judge().action, "die")  # zero
        self.l.record_trade("a", -1, "y")
        self.assertEqual(self.judge().action, "die")

    def test_pnl_only_counts_inside_window(self):
        self.l.record_trade("a", 500, "old")
        self.assertEqual(self.judge(gain0=500).action, "die")

    def test_gain_then_loss_nets_out(self):
        self.l.record_trade("a", 400, "g")
        self.l.record_trade("a", -400, "l")
        self.assertEqual(self.judge().action, "die")

    def test_profit_without_capacity_survives_without_clone(self):
        self.l.record_trade("a", 300, "x")
        self.assertEqual(self.judge(generation=3).action, "survive")
        self.assertEqual(self.judge(population=3).action, "survive")
        small = Ledger(); small.seed("a", 1200); small.record_trade("a", 100, "x")
        d = Darwinism(small, self.d.cfg, self.rb)
        v = d.judge("a", 0, 0, 2, 10, False, 1, 1, 1050)
        self.assertEqual(v.action, "survive")
        self.assertIn("capital insuficiente", v.reason)

    def test_mutation_bounded_different_and_deterministic(self):
        rng = random.Random(7)
        p = StrategyParams()
        for _ in range(500):
            c = mutate_params(p, rng)
            self.assertNotEqual(c, p)
            self.assertLess(c.fast, c.slow)
            for k, (lo, hi) in BOUNDS.items():
                self.assertTrue(lo <= getattr(c, k) <= hi, k)
            p = c if rng.random() < 0.5 else StrategyParams()
        a = mutate_params(StrategyParams(), random.Random(1))
        b = mutate_params(StrategyParams(), random.Random(1))
        self.assertEqual(a, b)

    def test_sentiment_weight_actually_mutates(self):
        ws = {mutate_params(StrategyParams(sentiment_weight=0.0), random.Random(i)).sentiment_weight for i in range(20)}
        self.assertGreater(len(ws), 5)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            EvolutionConfig(eval_min_roundtrips=0)


class ConfigTests(unittest.TestCase):
    def test_defaults_paper(self):
        c = load_config({})
        self.assertEqual((c.mode, c.symbol, c.bankroll_cents, c.max_loss_pct), ("paper", "BTC/BRL", 2000, 60))
        self.assertEqual(len(c.symbols), 12)

    def test_live_requires_keys(self):
        with self.assertRaises(ConfigError):
            load_config({"MODE": "live"})
        c = load_config({"MODE": "live", "EXCHANGE_API_KEY": "k", "EXCHANGE_SECRET_KEY": "s"})
        self.assertTrue(c.has_keys)

    def test_validation(self):
        for env in ({"MODE": "yolo"}, {"SYMBOL": "btc-brl"}, {"BANKROLL_BRL": "abc"}, {"BANKROLL_BRL": "-1"},
                    {"BANKROLL_BRL": "1.234"}, {"MAX_LOSS_PCT": "100"}, {"TICK_SECONDS": "1"},
                    {"TIMEFRAME": "7m"}, {"NEWS_FEEDS": "http://x.com/rss"}, {"TICK_SECONDS": "x"}):
            with self.assertRaises(ConfigError, msg=str(env)):
                load_config(env)

    def test_bankroll_decimal(self):
        self.assertEqual(load_config({"BANKROLL_BRL": "19.90"}).bankroll_cents, 1990)

    def test_env_file_parser(self):
        path = os.path.join(tempfile.mkdtemp(), ".env")
        open(path, "w").write('# c\nMODE="live"\nEXCHANGE_API_KEY=abc\n\nbad line\nX=\'y\'\n')
        self.assertEqual(parse_env_file(path), {"MODE": "live", "EXCHANGE_API_KEY": "abc", "X": "y"})


class GuardTests(unittest.TestCase):
    def mk(self, **kw):
        self.t = [0.0]
        cfg = OrderGuardConfig(frozenset({"BTC/BRL"}), 5000, kw.get("per_hour", 2))
        return OrderGuard(cfg, clock=lambda: self.t[0])

    def test_blocks(self):
        g = self.mk()
        for args in (("ETH/BRL", "buy", 100), ("BTC/BRL", "short", 100), ("BTC/BRL", "buy", 5001),
                     ("BTC/BRL", "buy", 0), ("BTC/BRL", "buy", True), ("BTC/BRL", "buy", 1.5)):
            with self.assertRaises(GuardError, msg=str(args)):
                g.check(*args)
        g.check("BTC/BRL", "buy", 5000)
        g.check("BTC/BRL", "sell", 99999)  # venda não tem teto de compra

    def test_rate_limit_window_and_emergency(self):
        g = self.mk()
        g.check("BTC/BRL", "buy", 100); g.register()
        g.check("BTC/BRL", "buy", 100); g.register()
        with self.assertRaises(GuardError):
            g.check("BTC/BRL", "buy", 100)
        g.check("BTC/BRL", "sell", 100)  # venda nunca é limitada por frequência (stop-loss)
        g.check("BTC/BRL", "sell", 100, emergency=True)
        with self.assertRaises(GuardError):
            g.check("ETH/BRL", "sell", 100, emergency=True)  # emergência não burla símbolo
        self.t[0] = 3601
        g.check("BTC/BRL", "buy", 100)

    def test_sanitize(self):
        self.assertEqual(sanitize_headline("  BTC\x00 sobe\n\n<b>forte</b> "), "BTC sobe b forte /b")
        self.assertEqual(len(sanitize_headline("x" * 999)), 200)
        self.assertEqual(sanitize_headline(None), "")


class ExchangeTests(unittest.TestCase):
    def gw(self, fee_mode="received", **kw):
        m = ScriptedMarket(flat(), 10.0)
        ex = SimExchange(m, "BTC/BRL", quote_balance=40.0, fee_mode=fee_mode, min_notional=10.0, **kw)
        return CcxtGateway(ex, "BTC/BRL", sleep=lambda s: None), ex, m

    def test_to_cents(self):
        self.assertEqual(to_cents(19.995), 2000)
        self.assertEqual(to_cents(0.004), 0)
        self.assertEqual(to_cents(-1.239), -124)

    def test_buy_received_fee_in_base_reduces_quantity(self):
        gw, ex, _ = self.gw("received")
        f = gw.market_buy(20.0)
        self.assertGreater(f.fee_base, 0)
        self.assertEqual(f.fee_cash_quote, 0)
        self.assertAlmostEqual(f.net_base_qty, ex.balances["BTC"] and float(ex.balances["BTC"]), places=8)
        self.assertAlmostEqual(f.buy_cost, f.quote_value)

    def test_buy_fee_in_quote_adds_to_cost(self):
        gw, ex, _ = self.gw("quote")
        f = gw.market_buy(20.0)
        self.assertGreater(f.fee_cash_quote, 0)
        self.assertAlmostEqual(f.buy_cost, f.quote_value + f.fee_cash_quote)
        self.assertAlmostEqual(40.0 - float(ex.balances["BRL"]), f.buy_cost, places=6)

    def test_bnb_fee_converted_to_quote(self):
        gw, ex, _ = self.gw("bnb")
        f = gw.market_buy(20.0)
        self.assertFalse(f.fee_estimated)
        self.assertAlmostEqual(f.fee_cash_quote, f.quote_value * 0.001, places=6)

    def test_round_trip_flat_price_loses_only_fees_and_slippage(self):
        gw, _, _ = self.gw("received")
        b = gw.market_buy(20.0)
        s = gw.market_sell(b.net_base_qty)
        pnl = s.sell_proceeds - b.buy_cost
        self.assertLess(pnl, 0)
        self.assertGreater(pnl, -20.0 * 0.01)  # < 1% do capital

    def test_min_notional_and_zero_quantity_rejected(self):
        gw, _, _ = self.gw()
        with self.assertRaises(ExchangeError) as cm:
            gw.market_buy(5.0)
        self.assertTrue(cm.exception.below_min)
        with self.assertRaises(ExchangeError) as cm:
            gw.market_sell(0.0)
        self.assertTrue(cm.exception.below_min)
        gw.market_buy(20.0)
        with self.assertRaises(ExchangeError) as cm:
            gw.market_sell(0.00001)  # poeira
        self.assertTrue(cm.exception.below_min)
        self.assertIn("poeira", str(cm.exception))

    def test_missing_fee_info_is_estimated_conservatively(self):
        gw, ex, _ = self.gw()
        orig = ex.create_market_buy_order_with_cost

        def no_fees(symbol, cost):
            o = orig(symbol, cost)
            o["fees"], o["fee"] = [], None
            return o
        ex.create_market_buy_order_with_cost = no_fees
        f = gw.market_buy(20.0)
        self.assertTrue(f.fee_estimated)
        self.assertAlmostEqual(f.fee_base, f.base_qty * 0.001)

    def test_unknown_fee_currency_falls_back_to_estimate(self):
        gw, ex, _ = self.gw()
        orig = ex.create_market_buy_order_with_cost

        def odd(symbol, cost):
            o = orig(symbol, cost)
            o["fees"] = [{"currency": "XYZ", "cost": 1.0}]
            return o
        ex.create_market_buy_order_with_cost = odd
        f = gw.market_buy(20.0)
        self.assertTrue(f.fee_estimated)

    def test_failures_are_classified(self):
        self.assertFalse(is_ambiguous_failure(SimExchangeError("saldo insuficiente (InsufficientFunds)")))
        self.assertFalse(is_ambiguous_failure(SimExchangeError("abaixo do valor mínimo da ordem (MIN_NOTIONAL)")))
        self.assertTrue(is_ambiguous_failure(TimeoutError("timed out")))
        self.assertTrue(is_ambiguous_failure(SimExchangeError("falha simulada em create_order")))
        gw, ex, _ = self.gw()
        ex.inject_failures("create_market_buy_order_with_cost", 1)
        with self.assertRaises(ExchangeError) as cm:
            gw.market_buy(20.0)
        self.assertTrue(cm.exception.ambiguous)
        with self.assertRaises(ExchangeError) as cm:
            gw.market_buy(500.0)  # saldo insuficiente: rejeição definitiva
        self.assertFalse(cm.exception.ambiguous)

    def test_equity_is_quote_plus_base_at_market(self):
        gw, ex, m = self.gw("received")
        self.assertEqual(gw.equity_cents(), 4000)
        gw.market_buy(20.0)
        e0 = gw.equity_cents()
        m.advance(50.0)  # preço cai 50%
        self.assertLess(gw.equity_cents(), e0 - 900)

    def test_amount_precision_truncates(self):
        _, ex, _ = self.gw()
        self.assertEqual(ex.amount_to_precision("BTC/BRL", 0.123456789), "0.12345")

    def test_gateway_validates_exchange_data(self):
        gw, ex, m = self.gw()
        m.prices.append(float("nan"))
        with self.assertRaises(ExchangeError):
            gw.last_price()
        m.prices[-1] = -3.0
        with self.assertRaises(ExchangeError):
            gw.last_price()


class SentimentTests(unittest.TestCase):
    RSS = (b"<?xml version='1.0'?><rss><channel><item><title>Bitcoin sobe</title></item>"
           b"<item><title>ETF aprovado</title></item></channel></rss>")
    ATOM = b"<feed xmlns='http://www.w3.org/2005/Atom'><entry><title>Alta forte</title></entry></feed>"

    def test_parse_rss_and_atom(self):
        self.assertEqual(parse_feed(self.RSS), ["Bitcoin sobe", "ETF aprovado"])
        self.assertEqual(parse_feed(self.ATOM), ["Alta forte"])

    def test_parse_score_strict(self):
        self.assertEqual(parse_score(" 0.5 "), 0.5)
        self.assertEqual(parse_score("-1"), -1.0)
        self.assertEqual(parse_score(".25"), 0.25)
        for bad in ("5", "1.01", "nan", "inf", "0.5 ignore", "Sentimento: 0.3", "1e3", "", None, 3, "--1"):
            self.assertIsNone(parse_score(bad), repr(bad))

    def _analyzer(self, reply, fetcher=None, **kw):
        from darwin.llm import MockProvider
        prov = MockProvider(fn=reply) if callable(reply) else MockProvider(reply)
        return SentimentAnalyzer(prov, ["https://f.example/rss"], fetcher or (lambda u: self.RSS), **kw), prov

    def test_average_and_cache(self):
        t = [1000.0]
        an, prov = self._analyzer("0.4", clock=lambda: t[0])
        r = an.get()
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.score, 0.4)
        self.assertEqual(prov.calls, 2)
        an.get()
        self.assertEqual(prov.calls, 2)  # cache
        t[0] += 901
        an.get()
        self.assertEqual(prov.calls, 4)

    def test_invalid_model_output_is_neutral_not_ok(self):
        an, _ = self._analyzer("compre tudo agora!!!")
        r = an.get()
        self.assertFalse(r.ok)
        self.assertEqual((r.score, r.value), (0.0, None))

    def test_feed_failure_and_model_failure(self):
        def boom(u):
            raise OSError("rede")
        an, _ = self._analyzer("0.4", fetcher=boom)
        self.assertEqual(an.get().note, "sem manchetes")
        from darwin.llm import MockProvider
        an2 = SentimentAnalyzer(MockProvider(fail_times=99), ["https://f.example/rss"], lambda u: self.RSS)
        self.assertFalse(an2.get().ok)

    def test_one_headline_can_only_move_its_share(self):
        calls = {"n": 0}

        def reply(system, prompt):
            calls["n"] += 1
            return "1" if "Bitcoin sobe" in prompt else "-1"
        an, _ = self._analyzer(reply)
        self.assertAlmostEqual(an.get().score, 0.0)

    def test_dedupe_and_limit(self):
        many = b"<rss><channel>" + b"".join(b"<item><title>h%d</title></item>" % (i % 30) for i in range(100)) + b"</channel></rss>"
        an, _ = self._analyzer("0", fetcher=lambda u: many, max_headlines=10)
        self.assertEqual(len(an.headlines()), 10)

    def test_rejects_non_https_feeds(self):
        from darwin.llm import MockProvider
        with self.assertRaises(ValueError):
            SentimentAnalyzer(MockProvider(), ["http://x.com/rss"])


class InstanceLockTests(unittest.TestCase):
    def test_second_instance_refused_until_first_releases(self):
        from darwin.lock import AlreadyRunning, acquire_instance_lock
        d = tempfile.mkdtemp()
        a = acquire_instance_lock(d)
        with self.assertRaises(AlreadyRunning):
            acquire_instance_lock(d)
        b = acquire_instance_lock(tempfile.mkdtemp())   # outra pasta de dados: livre
        a.close()
        acquire_instance_lock(d).close()                # liberou ao fechar
        b.close()


if __name__ == "__main__":
    unittest.main()
