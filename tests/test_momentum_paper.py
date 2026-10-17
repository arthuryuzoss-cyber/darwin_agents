"""Momentum diário em papel: igual ao backtest, idempotente, refaz dias pulados, estado seguro."""
import contextlib
import io
import json
import os
import tempfile
import unittest

import main as cli
from darwin import momentum as mo
from darwin import momentum_paper as mp
from tests.test_momentum import SYMS3, cfg, gen, hist, rows

P = mo.MomParams(trend_n=100, asset_trend=50, long=60, short=30)


def prefix(h_series, n):
    return hist({s: c[:n] for s, c in h_series.items()})


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.series = gen(0.002, n=700, seed=11)
        self.h = hist(self.series)
        self.bt = cfg()

    def test_replay_equals_backtest(self):
        a = self.h.n - 300
        st = mp.new_state(100.0)
        for i in range(a, self.h.n):
            mp.apply_day(st, self.h, list(SYMS3), P, i, self.bt.fee_rate, self.bt.slippage, 10.0, 1.0)
        final = mp.equity(st, self.h, self.h.n - 1, 3)
        sim = mo.simulate(self.h, cfg(bankroll_cents=10000), P, a, self.h.n)
        self.assertAlmostEqual(final, 100.0 * (1 + sim["return_pct"] / 100), places=6)

    def test_idempotent_same_day(self):
        st = mp.new_state(100.0)
        first = mp.step(st, self.h, list(SYMS3), P)
        again = mp.step(st, self.h, list(SYMS3), P)
        self.assertEqual(len(first), 1)
        self.assertEqual(again, [])
        self.assertEqual(len(st["log"]), 1)

    def test_skipped_days_are_replayed_in_order(self):
        st = mp.new_state(100.0)
        mp.step(st, prefix(self.series, 650), list(SYMS3), P)
        new = mp.step(st, self.h, list(SYMS3), P)
        self.assertEqual(len(new), 50)
        days = [e["day"] for e in st["log"]]
        self.assertEqual(days, sorted(days))
        self.assertEqual(len(set(days)), len(days))

    def test_exposure_half_keeps_cash(self):
        st = mp.new_state(100.0)
        for i in range(self.h.n - 200, self.h.n):
            mp.apply_day(st, self.h, list(SYMS3), P, i, 0.001, 0.0005, 10.0, 0.5)
            if st["held"]:
                break
        self.assertTrue(st["held"])
        self.assertAlmostEqual(st["cash"], 50.0, delta=0.2)

    def test_bad_exposure_and_short_history(self):
        with self.assertRaises(mp.PaperError):
            mp.step(mp.new_state(10.0), self.h, list(SYMS3), P, exposure=0)
        with self.assertRaises(mp.PaperError):
            mp.step(mp.new_state(10.0), prefix(self.series, 50), list(SYMS3), P)

    def test_state_roundtrip_and_bankroll_mismatch(self):
        path = os.path.join(tempfile.mkdtemp(), "sub", "m.json")
        st = mp.new_state(40.0)
        mp.step(st, self.h, list(SYMS3), P)
        mp.save_state(path, st)
        self.assertEqual(mp.load_state(path, 40.0)["log"], st["log"])
        with self.assertRaises(mp.PaperError):
            mp.load_state(path, 50.0)

    def test_no_future_leak(self):
        st1, st2 = mp.new_state(100.0), mp.new_state(100.0)
        h2 = hist({s: c[:-1] + [c[-1] * 50] for s, c in self.series.items()})   # só o candle de hoje muda
        e1 = mp.step(st1, self.h, list(SYMS3), P)[0]
        e2 = mp.step(st2, h2, list(SYMS3), P)[0]
        self.assertEqual(e1["action"], e2["action"])   # decisão usa só até ontem


class SignalCliTests(unittest.TestCase):
    def test_cli_two_runs(self):
        from tests.test_backtest import FakeCcxt
        import time
        now = int(time.time() * 1000)
        day = 86_400_000
        t0 = (now // day) * day - 419 * day
        series = gen(0.002, n=420, seed=4)
        FakeCcxt.series = {s: [[t0 + i * day, c, c * 1.01, c * 0.99, c, 1.0] for i, c in enumerate(cl)]
                           for s, cl in series.items()}
        FakeCcxt.calls, FakeCcxt.fail, FakeCcxt.precision_mode = [], {}, 4
        d = tempfile.mkdtemp()
        env = {"MODE": "paper", "DATA_DIR": os.path.join(d, "data"), "BANKROLL_BRL": "40", "SYMBOLS": ",".join(SYMS3)}
        outs = []
        for _ in range(2):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cli.main(["signal", "--dir", os.path.join(d, "h"), "--trend-n", "100", "--long", "60"],
                                env=env, ccxt_module=FakeCcxt)
            self.assertEqual(code, 0, buf.getvalue())
            outs.append(buf.getvalue())
        self.assertIn("PAPEL", outs[0])
        self.assertIn("já foi processado", outs[1])
        st = json.load(open(os.path.join(d, "data", "momentum_paper.json")))
        self.assertEqual(len(st["log"]), 1)
