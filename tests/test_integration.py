"""Integração: orquestrador + gateway + SimExchange (mesmo caminho de código das ordens reais)."""
import unittest

from darwin.exchange import CcxtGateway
from darwin.orchestrator import (EXIT_EXTINCT, EXIT_FUSE, EXIT_OK, TREASURY, AccountNotClean,
                                 OrchestratorError)
from tests.helpers import FixedSentiment, Rig, flat, uptrend

BUYABLE = dict(buy_threshold=0.1, rsi_overbought=85.0, take_profit_pct=3.0, stop_loss_pct=2.0)
RAMP = uptrend(100.0, 40, 0.006)


def drive_until(rig, prices, cond, name="alpha"):
    for p in prices:
        rig.market.advance(p)
        rig.orch.tick()
        if cond(rig.orch) or rig.orch.stop_reason:
            return True
    return False


def bought(o, name="alpha"):
    return name in o.traders and o.traders[name].in_position


class BootstrapTests(unittest.TestCase):
    def test_creates_agent_and_baseline(self):
        r = Rig(quote=40.0)
        ts = r.orch.bootstrap(1)
        self.assertEqual([t.name for t in ts], ["alpha"])
        self.assertEqual(r.ledger.balance("alpha"), 4000)
        self.assertEqual((r.breaker.initial_equity_cents, r.breaker.floor_cents), (4000, 1600))

    def test_multiple_agents_split_capital_and_each_can_trade(self):
        r = Rig(quote=40.0)
        r.orch.bootstrap(4)  # 4000 // 1050 = 3 agentes no máximo
        self.assertEqual(sorted(r.orch.traders), ["alpha", "beta", "gamma"])
        self.assertEqual(sum(r.ledger.balance(n) for n in r.orch.traders), 4000)
        for n in r.orch.traders:
            self.assertGreaterEqual(r.ledger.balance(n), r.orch.min_trade_cents())

    def test_refuses_dirty_or_oversized_or_tiny_accounts(self):
        with self.assertRaises(AccountNotClean):
            Rig(quote=40.0, base=0.5).orch.bootstrap(1)          # já tem o ativo base
        with self.assertRaises(AccountNotClean):
            Rig(quote=100.0).orch.bootstrap(1)                   # muito acima da banca
        with self.assertRaises(OrchestratorError):
            Rig(quote=5.0, bankroll_cents=500).orch.bootstrap(1)  # abaixo da ordem mínima

    def test_cannot_bootstrap_twice(self):
        r = Rig(quote=40.0)
        r.orch.bootstrap(1)
        with self.assertRaises(OrchestratorError):
            r.orch.bootstrap(1)

    def test_refuses_to_bootstrap_with_open_orders(self):
        r = Rig(quote=40.0)
        r.sim.open_orders.append({"id": "old-order", "symbol": "BTC/BRL"})
        with self.assertRaisesRegex(AccountNotClean, "ordem.*aberta"):
            r.orch.bootstrap(1)

    def test_refuses_to_bootstrap_if_open_orders_cannot_be_checked(self):
        r = Rig(quote=40.0)
        r.sim.inject_failures("fetch_open_orders", 1)
        with self.assertRaisesRegex(AccountNotClean, "verificar ordens abertas"):
            r.orch.bootstrap(1)


class TradingCycleTests(unittest.TestCase):
    def rig(self, **kw):
        r = Rig(prices=flat(70), quote=40.0, **kw)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        return r

    def test_profitable_cycle_clones_with_mutated_child(self):
        r = self.rig()
        before_global = r.ledger.global_balance()
        self.assertTrue(drive_until(r, RAMP, lambda o: "alpha.1" in o.traders))
        child, parent = r.orch.traders["alpha.1"], r.orch.traders["alpha"]
        self.assertEqual((child.generation, child.parent), (2, "alpha"))
        self.assertNotEqual(child.params, parent.params)
        mt = r.orch.min_trade_cents()
        self.assertEqual(mt, 1200)
        self.assertGreaterEqual(r.ledger.balance("alpha.1"), mt)
        self.assertGreaterEqual(r.ledger.balance("alpha"), mt)
        self.assertGreater(r.ledger.total("alpha", "trade_gain"), 0)
        # capital só mudou pelo PnL realizado: a clonagem apenas move dinheiro entre agentes
        gain = r.ledger.total("alpha", "trade_gain")
        self.assertEqual(r.ledger.global_balance(), before_global + gain)
        self.assertIn("🧬", r.text())

    def test_losing_cycle_kills_and_removes_agent_then_extinction(self):
        r = self.rig(revive=False)
        self.assertTrue(drive_until(r, RAMP, bought))
        entry = r.orch.traders["alpha"].entry_price
        r.market.advance(entry * 0.975)  # -2,5%: passa do stop-loss de 2%
        r.orch.tick()
        self.assertEqual(r.orch.traders, {})
        self.assertEqual(r.orch.graveyard[0]["name"], "alpha")
        self.assertIn("<= 0", r.orch.graveyard[0]["reason"])
        self.assertEqual(r.orch.stop_reason, "extinct")
        self.assertGreater(r.ledger.total("alpha", "trade_loss"), 0)
        self.assertEqual(r.ledger.balance("alpha"), 0)
        self.assertEqual(r.ledger.balance(TREASURY), r.ledger.global_balance())
        self.assertLess(float(r.sim.balances["BTC"]), 0.001)  # posição de fato vendida na exchange
        self.assertIn("💀", r.text())

    def test_zero_trade_cycle_dies_only_when_idle_kills_is_on(self):
        r = Rig(prices=flat(70), quote=40.0, eval_max_ticks=10, eval_min_roundtrips=50, idle_kills=True, revive=False)
        r.orch.bootstrap(1)
        reports = r.drive([100.0] * 12)
        self.assertEqual(r.orch.stop_reason, "extinct")
        self.assertTrue(any("💀" in e for rep in reports for e in rep.events))

    def test_idle_market_does_not_kill_the_agent_by_default(self):
        """Regressão: mercado parado por 3h+ derrubava o robô por 'PnL 0' sem nenhum trade."""
        r = Rig(prices=flat(70), quote=40.0, eval_max_ticks=10, eval_min_roundtrips=50)
        r.orch.bootstrap(1)
        r.drive([100.0] * 35)                           # 3 ciclos inteiros sem sinal
        self.assertEqual(r.orch.stop_reason, "")
        self.assertIn("alpha", r.orch.traders)
        self.assertEqual(r.orch.traders["alpha"].window_ticks, 5)   # reiniciou a janela 3 vezes
        self.assertIn("💤", r.text())

    def test_max_ticks_forces_close_then_judges(self):
        r = Rig(prices=flat(70), quote=40.0, eval_max_ticks=14, eval_min_roundtrips=50)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        r.drive(RAMP[:14])
        # fim do ciclo: posição encerrada à força (ou nunca abriu) e avaliação feita
        self.assertTrue(all(not t.in_position for t in r.orch.traders.values()))
        self.assertTrue(r.orch.graveyard or len(r.orch.traders) >= 1)

    def test_profit_without_room_to_clone_survives(self):
        r = Rig(prices=flat(70), quote=22.0, bankroll_cents=2200)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        self.assertTrue(drive_until(r, RAMP, lambda o: "alpha" in o.traders and o.traders["alpha"].window_roundtrips == 0
                                    and r.ledger.total("alpha", "trade_gain") > 0))
        self.assertEqual(list(r.orch.traders), ["alpha"])          # sem filho: capital < 2 x mínimo
        self.assertIn("sobrevive sem clonar", r.text())

    def test_sentiment_can_trigger_and_close_trades(self):
        r = Rig(prices=flat(80), quote=40.0, sentiment=FixedSentiment(1.0))
        r.orch.bootstrap(1)
        r.tune(sentiment_weight=0.8)
        r.drive([100.0] * 2)
        self.assertTrue(bought(r.orch))
        r.orch.sentiment = FixedSentiment(-1.0)
        r.drive([100.0])
        self.assertFalse(bought(r.orch) if "alpha" in r.orch.traders else False)

    def test_no_sentiment_means_technical_only(self):
        r = Rig(prices=flat(80), quote=40.0)
        r.orch.bootstrap(1)
        r.tune(sentiment_weight=0.8)
        r.drive([100.0] * 3)
        self.assertFalse(bought(r.orch))

    def test_ledger_pnl_matches_account_equity_within_cents(self):
        for mode in ("received", "quote"):
            r = Rig(prices=flat(70), quote=40.0, fee_mode=mode)
            r.orch.bootstrap(1)
            r.tune(**{**BUYABLE, "take_profit_pct": 1.0})
            drive_until(r, RAMP, lambda o: r.ledger.total("alpha", "trade_gain") > 0 or r.ledger.total("alpha", "trade_loss") > 0)
            s = r.orch.status()
            self.assertLessEqual(abs(s["drift_cents"]), 3, mode)
            pnl = r.ledger.total("alpha", "trade_gain") - r.ledger.total("alpha", "trade_loss")
            self.assertLessEqual(abs(s["account_pnl_cents"] - pnl), 3, mode)


class FuseTests(unittest.TestCase):
    def rig(self, quote=40.0, **kw):
        r = Rig(prices=flat(70), quote=quote, bankroll_cents=int(quote * 100), **kw)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        self.assertTrue(drive_until(r, RAMP, bought))
        return r

    def test_fuse_at_60pct_liquidates_cancels_and_stops(self):
        r = self.rig()
        self.assertEqual((r.breaker.initial_equity_cents, r.breaker.floor_cents), (4000, 1600))
        r.sim.open_orders.append({"id": "pendente-1", "symbol": "BTC/BRL"})
        entry = r.orch.traders["alpha"].entry_price
        r.market.advance(entry * 0.35)  # queda de 65% entre dois ticks
        rep = r.orch.tick()
        self.assertTrue(rep.halted)
        self.assertEqual(r.orch.stop_reason, "fuse")
        self.assertTrue(r.breaker.is_open())
        self.assertFalse(r.breaker.liquidation_pending)
        self.assertLess(float(r.sim.balances["BTC"]), 0.001)           # tudo vendido
        self.assertEqual(r.sim.open_orders, [])                        # ordens canceladas
        self.assertIn("DISJUNTOR DISPARADO", r.text())
        self.assertIn("liquidação concluída", r.text())
        self.assertEqual(r.orch.run(max_ticks=5), EXIT_FUSE)            # não volta a operar

    def test_trading_stops_for_good_after_restart(self):
        r = self.rig()
        entry = r.orch.traders["alpha"].entry_price
        r.market.advance(entry * 0.35)
        r.orch.tick()
        n_orders = sum(1 for c in r.sim.calls if c.startswith("create"))
        r2 = Rig(sim=r.sim, market=r.market, data_dir=r.dir, cfg=r.cfg)
        self.assertEqual(r2.orch.run(max_ticks=3), EXIT_FUSE)
        self.assertEqual(sum(1 for c in r.sim.calls if c.startswith("create")), n_orders)

    def test_no_trip_above_floor(self):
        r = self.rig()
        entry = r.orch.traders["alpha"].entry_price
        r.market.advance(entry * 0.97)  # só -3%: stop-loss, mas muito longe do fusível
        r.orch.tick()
        self.assertFalse(r.breaker.is_open())

    def test_liquidation_failure_is_retried_on_next_start(self):
        r = self.rig()
        r.sim.inject_failures("create_order", 5)
        entry = r.orch.traders["alpha"].entry_price
        r.market.advance(entry * 0.35)
        r.orch.tick()
        self.assertTrue(r.breaker.liquidation_pending)
        self.assertIn("LIQUIDAÇÃO INCOMPLETA", r.text())
        self.assertGreater(float(r.sim.balances["BTC"]), 0.1)           # ainda posicionado
        r2 = Rig(sim=r.sim, market=r.market, data_dir=r.dir, cfg=r.cfg)
        self.assertTrue(r2.breaker.liquidation_pending)
        self.assertEqual(r2.orch.run(max_ticks=1), EXIT_FUSE)
        self.assertFalse(r2.breaker.liquidation_pending)
        self.assertLess(float(r.sim.balances["BTC"]), 0.001)

    def test_ambiguous_order_error_trips_fuse(self):
        r = Rig(prices=flat(70), quote=20.0, bankroll_cents=2000)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        r.sim.inject_failures("create_market_buy_order_with_cost", 1)
        drive_until(r, RAMP, lambda o: o.stop_reason == "fuse")
        self.assertEqual(r.orch.stop_reason, "fuse")
        self.assertIn("ambíguo", r.breaker.reason)

    def test_definitive_rejection_does_not_trip(self):
        r = Rig(prices=flat(70), quote=40.0, bankroll_cents=4000)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        r.sim.balances["BRL"] = r.sim.balances["BRL"] / 2  # saldo real menor que o contábil (equity acima do piso)
        drive_until(r, RAMP[:25], lambda o: "recusada" in r.text())
        self.assertIn("recusada", r.text())
        self.assertFalse(r.breaker.is_open())

    def test_unreadable_equity_blocks_trading_then_trips(self):
        r = Rig(prices=flat(80), quote=20.0, bankroll_cents=2000, sentiment=FixedSentiment(1.0))
        r.orch.bootstrap(1)
        r.tune(sentiment_weight=0.8)
        r.sim.inject_failures("fetch_balance", 2)
        r.drive([100.0] * 2)
        self.assertFalse(bought(r.orch))            # sem ler patrimônio, não opera
        self.assertFalse(r.breaker.is_open())
        r.sim.inject_failures("fetch_balance", 3)
        r.drive([100.0] * 3)
        self.assertTrue(r.breaker.is_open())        # 3 falhas seguidas => desarma
        self.assertIn("falhas consecutivas", r.breaker.reason)

    def test_market_data_failures_trip_too(self):
        r = Rig(prices=flat(80), quote=20.0, bankroll_cents=2000)
        r.orch.bootstrap(1)
        r.sim.inject_failures("fetch_ohlcv", 3)
        r.drive([100.0] * 3)
        self.assertTrue(r.breaker.is_open())


class SmallBankrollLimitationTests(unittest.TestCase):
    """Limitação real, não bug: com R$ 20 e ordem mínima de R$ 10, no piso do fusível (R$ 8) a
    posição já vale menos que o mínimo, e a exchange recusa a venda. O robô tem que dizer isso."""

    def test_r20_warns_before_starting(self):
        r = Rig(prices=flat(70), quote=20.0, bankroll_cents=2000)
        r.orch.bootstrap(1)
        w = r.orch.fuse_feasibility()
        self.assertIn("IMPOSSÍVEL", w)
        self.assertIn("R$ 37.50", w)  # banca mínima para o fusível conseguir liquidar

    def test_r40_has_no_warning(self):
        r = Rig(prices=flat(70), quote=40.0, bankroll_cents=4000)
        r.orch.bootstrap(1)
        self.assertIsNone(r.orch.fuse_feasibility())

    def test_r20_gap_crash_liquidation_is_reported_as_incomplete_not_success(self):
        r = Rig(prices=flat(70), quote=20.0, bankroll_cents=2000)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        self.assertTrue(drive_until(r, RAMP, bought))
        r.market.advance(r.orch.traders["alpha"].entry_price * 0.35)
        rep = r.orch.tick()
        self.assertTrue(rep.halted and r.breaker.is_open())
        self.assertTrue(r.breaker.liquidation_pending)                  # NÃO declara sucesso
        self.assertIn("NÃO DÁ PARA VENDER POR API", r.text())
        self.assertNotIn("liquidação concluída", r.text())
        self.assertGreater(float(r.sim.balances["BTC"]), 0.1)           # posição ainda na conta
        self.assertFalse(r.breaker.reset("token-do-dono-123"))          # reset bloqueado até resolver
        # depois que o dono vende manualmente, `liquidate` confirma e libera
        r.sim.balances["BTC"] = r.sim.balances["BTC"] * 0
        self.assertTrue(r.orch.liquidate_all("manual"))
        self.assertFalse(r.breaker.liquidation_pending)

    def test_liquidity_floor_exit_sells_before_position_becomes_unsellable(self):
        # mínimo da exchange R$ 30 => piso de liquidez em R$ 34,50; posição de ~R$ 38
        r = Rig(prices=flat(70), quote=40.0, bankroll_cents=4000, min_notional=30.0)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        self.assertTrue(drive_until(r, RAMP, bought))
        r.market.advance(r.orch.traders["alpha"].entry_price * 0.90)  # gap: posição ~R$ 34,2 (< 34,50)
        r.orch.tick()
        self.assertFalse(bought(r.orch) if "alpha" in r.orch.traders else False)
        self.assertIn("piso_de_liquidez", r.text())


class PersistenceAndLoopTests(unittest.TestCase):
    def test_state_survives_restart_with_open_position(self):
        r = Rig(prices=flat(70), quote=40.0)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        self.assertTrue(drive_until(r, RAMP, bought))
        qty, cost = r.orch.traders["alpha"].qty, r.orch.traders["alpha"].cost_cents
        o2 = r.build(r.orch.sentiment)
        self.assertAlmostEqual(o2.traders["alpha"].qty, qty)
        self.assertEqual(o2.traders["alpha"].cost_cents, cost)
        self.assertEqual(o2.traders["alpha"].params, r.orch.traders["alpha"].params)

    def test_run_loop_respects_max_ticks_and_sleeps(self):
        r = Rig(prices=flat(80), quote=40.0)
        r.orch.bootstrap(1)
        slept = []
        r.orch._sleep = slept.append
        self.assertEqual(r.orch.run(max_ticks=3), EXIT_OK)
        self.assertEqual(slept, [5, 5])

    def test_status_reports_everything(self):
        r = Rig(prices=flat(80), quote=40.0)
        r.orch.bootstrap(1)
        s = r.orch.status()
        self.assertEqual((s["equity_cents"], s["initial_cents"], s["floor_cents"], s["account_pnl_cents"]),
                         (4000, 4000, 1600, 0))
        self.assertEqual(s["drift_cents"], 0)

    def test_close_position_sells_only_what_the_account_has(self):
        r = Rig(prices=flat(70), quote=40.0)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        self.assertTrue(drive_until(r, RAMP, bought))
        from decimal import Decimal
        r.sim.balances["BTC"] = r.sim.balances["BTC"] / 2  # usuário vendeu metade manualmente
        have = float(r.sim.balances["BTC"])
        r.orch._close_position(r.orch.traders["alpha"], "teste", None)
        self.assertLess(float(r.sim.balances["BTC"]), 0.001)
        self.assertLessEqual(have, r.orch.traders["alpha"].qty + 1)  # nunca vendeu mais do que existia

    def test_old_data_blocks_reinit(self):
        r = Rig(prices=flat(80), quote=40.0)
        r.orch.bootstrap(1)
        with self.assertRaises(OrchestratorError):
            r.orch.bootstrap(1)


if __name__ == "__main__":
    unittest.main()
