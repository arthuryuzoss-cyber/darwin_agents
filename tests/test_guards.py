"""Proteções: cooldown após saída e pausa após vários stop-loss (StoplossGuard)."""
import contextlib
import io
import os
import tempfile
import unittest

import main as cli
from darwin.config import ConfigError, load_config
from tests.helpers import Rig, flat
from tests.test_backtest import SYMS, write_csv, csv_path, rows_from_closes, walk


def rig(**kw):
    r = Rig(prices=flat(70), quote=40.0, tick_seconds=300, **kw)    # 1 tick = 1 candle de 5m
    r.orch.bootstrap(1)
    return r


class GuardTests(unittest.TestCase):
    def test_ticks_per_candle(self):
        self.assertEqual(rig().orch._ticks_per_candle(), 1)
        self.assertEqual(Rig(prices=flat(70), quote=40.0, tick_seconds=30).orch._ticks_per_candle(), 10)

    def test_cooldown_blocks_then_releases(self):
        r = rig(cooldown_candles=3)
        o = r.orch
        o.tick_no = 10
        o._register_exit("BTC/BRL", "sinal_de_saida", None)
        for t in (10, 11, 12):
            o.tick_no = t
            self.assertEqual(o._entry_blocked("BTC/BRL"), "cooldown")
        o.tick_no = 13
        self.assertEqual(o._entry_blocked("BTC/BRL"), "")
        self.assertEqual(o._entry_blocked("ETH/BRL"), "")            # só o ativo vendido

    def test_cooldown_off(self):
        r = rig(cooldown_candles=0)
        r.orch._register_exit("BTC/BRL", "stop_loss", None)
        self.assertEqual(r.orch._entry_blocked("BTC/BRL"), "")

    def test_stoploss_guard_pauses_after_k_stops(self):
        r = rig(stoploss_guard=3)
        o = r.orch
        for t in (1, 3):
            o.tick_no = t
            o._register_exit("BTC/BRL", "stop_loss", None)
            self.assertEqual(o._entry_blocked("ETH/BRL"), "")
        o.tick_no = 5
        o._register_exit("BTC/BRL", "stop_loss", None)
        self.assertEqual(o._entry_blocked("ETH/BRL"), "pausa_por_stop_loss")     # qualquer ativo
        o.tick_no = 5 + 12
        self.assertEqual(o._entry_blocked("ETH/BRL"), "")                        # pausa expira após 12 candles

    def test_only_stop_loss_counts_and_old_stops_expire(self):
        r = rig(stoploss_guard=2)
        o = r.orch
        for t in (1, 2, 3):
            o.tick_no = t
            o._register_exit("BTC/BRL", "take_profit", None)
        self.assertEqual(o._entry_blocked("ETH/BRL"), "")
        o.tick_no = 100
        o._register_exit("BTC/BRL", "stop_loss", None)
        o.tick_no = 200                                  # fora da janela de 24 candles
        o._register_exit("BTC/BRL", "stop_loss", None)
        self.assertEqual(o._entry_blocked("ETH/BRL"), "")

    def test_buy_is_actually_skipped_during_pause(self):
        from tests.test_integration import BUYABLE
        r = rig(stoploss_guard=1)
        r.tune(**BUYABLE)
        r.orch.tick_no = 0
        r.orch._register_exit("BTC/BRL", "stop_loss", None)        # K=1: pausa já
        r.market.advance(100.0)
        before = sum(1 for c in r.sim.calls if c.startswith("create"))
        for p in (101, 102, 103, 104, 105):
            r.market.advance(float(p))
            r.orch.tick()
        self.assertEqual(sum(1 for c in r.sim.calls if c.startswith("create")), before)

    def test_env_bounds(self):
        c = load_config({})
        self.assertEqual((c.cooldown_candles, c.stoploss_guard), (2, 3))
        c = load_config({"COOLDOWN_CANDLES": "0", "STOPLOSS_GUARD": "0"})
        self.assertEqual((c.cooldown_candles, c.stoploss_guard), (0, 0))
        for k in ("COOLDOWN_CANDLES", "STOPLOSS_GUARD"):
            for bad in ("-1", "x", "100000"):
                with self.assertRaises(ConfigError):
                    load_config({k: bad})


class CliGuardTests(unittest.TestCase):
    def test_no_guards_flag_changes_backtest_config(self):
        d = tempfile.mkdtemp()
        for k, s in enumerate(SYMS):
            write_csv(csv_path(d, s, "5m"), rows_from_closes(walk(20 + k, 2500, vol=0.004)))
        env = {"MODE": "paper", "DATA_DIR": os.path.join(d, "data"), "BANKROLL_BRL": "40", "SYMBOLS": ",".join(SYMS),
               "TIMEFRAME": "5m"}
        outs = []
        for extra in ([], ["--no-guards"]):
            o = io.StringIO()
            with contextlib.redirect_stdout(o):
                code = cli.main(["backtest", "--dir", d, "--out", os.path.join(d, "o"), "--fixed"] + extra, env=env)
            self.assertEqual(code, 0, o.getvalue())
            outs.append(o.getvalue())
        self.assertIn("Proteções ligadas", outs[0])
        self.assertIn("DESLIGADAS", outs[1])
