"""Rotação por força relativa: custos, sem olhar o futuro, filtro absoluto, walk-forward, CLI."""
import os
import random
import tempfile
import unittest
from dataclasses import replace

from darwin import rotation as rot
from darwin.backtest import BacktestError
from darwin.history import align
from tests import test_backtest as _tb
from tests.test_backtest import SYMS, TF, bt_cfg, make_hist, rows_from_closes


def hist_from(closes_by_sym):
    return align({s: rows_from_closes(c) for s, c in closes_by_sym.items()}, TF, {})


class RotationTests(unittest.TestCase):
    def test_params_validation(self):
        for kw in ({"lookback": 1}, {"rebalance": 0}, {"top_n": 0}, {"top_n": 7}):
            with self.assertRaises(ValueError):
                rot.RotParams(**kw)

    def test_picks_the_strongest_and_costs_matter(self):
        up = [100 * 1.002 ** i for i in range(600)]
        flat = [100.0] * 600
        h = hist_from({"BTC/BRL": up, "ETH/BRL": flat, "SOL/BRL": flat})
        p = rot.RotParams(lookback=24, rebalance=24, top_n=1)
        free = rot.simulate(h, bt_cfg(fee_rate=0.0, slippage=0.0), p, 0, 600)
        costly = rot.simulate(h, bt_cfg(fee_rate=0.01, slippage=0.01), p, 0, 600)
        self.assertGreater(free["return_pct"], 30)
        self.assertLess(costly["return_pct"], free["return_pct"])
        self.assertGreater(costly["fees_cents"], 0)

    def test_abs_filter_stays_in_cash_in_downtrend(self):
        down = [100 * 0.998 ** i for i in range(600)]
        h = hist_from({s: down for s in SYMS})
        on = rot.simulate(h, bt_cfg(), rot.RotParams(24, 24, 2, True), 0, 600)
        off = rot.simulate(h, bt_cfg(), rot.RotParams(24, 24, 2, False), 0, 600)
        self.assertEqual(on["n_trades"], 0)
        self.assertAlmostEqual(on["return_pct"], 0.0)
        self.assertLess(off["return_pct"], -20)

    def test_below_min_notional_does_not_buy(self):
        up = [100 * 1.002 ** i for i in range(300)]
        h = hist_from({s: up for s in SYMS})
        m = rot.simulate(h, bt_cfg(bankroll_cents=500), rot.RotParams(24, 24, 1), 0, 300)   # R$5 < mínimo R$10
        self.assertEqual(m["n_trades"], 0)

    def test_no_lookahead(self):
        base = {s: [100 * (1 + 0.001 * ((i * (k + 3)) % 11 - 5)) for i in range(900)] for k, s in enumerate(SYMS)}
        h1 = hist_from(base)
        poisoned = {s: c[:700] + [c[700] * 50] * 200 for s, c in base.items()}
        h2 = hist_from(poisoned)
        p = rot.RotParams(48, 24, 2)
        a = rot.simulate(h1, bt_cfg(), p, 0, 700)
        b = rot.simulate(h2, bt_cfg(), p, 0, 700)
        self.assertEqual(a, b)

    def test_short_period_and_short_history_raise(self):
        h = make_hist(300)
        with self.assertRaises(BacktestError):
            rot.simulate(h, bt_cfg(), rot.RotParams(24, 24, 1), 0, 20)
        with self.assertRaises(BacktestError):
            rot.walk_forward(h, bt_cfg(), folds=3)

    def test_walk_forward_shape_and_report(self):
        h = make_hist(4000, seed=3, vol=0.004)
        wf = rot.walk_forward(h, bt_cfg(), folds=3, trials=4, min_orders=1)
        self.assertEqual(len(wf["folds"]), 3)
        for f in wf["folds"]:
            self.assertLessEqual(f["train"][1], f["test"][0])        # teste depois do treino
        txt = rot.format_rotation(wf)
        for piece in ("ROTAÇÃO", "RESUMO FORA DA AMOSTRA", "CONCLUSÃO"):
            self.assertIn(piece, txt)
        self.assertEqual(wf, rot.walk_forward(h, bt_cfg(), folds=3, trials=4, min_orders=1))   # determinístico


class CliRotationTests(unittest.TestCase):
    run_cli = _tb.CliTests.run_cli
    hist_dir = _tb.CliTests.hist_dir

    def test_flag(self):
        d = self.hist_dir(3500)
        code, out = self.run_cli("backtest", "--dir", d, "--rotation", "--folds", "2", "--trials", "3")
        self.assertEqual(code, 0, out)
        self.assertIn("ROTAÇÃO POR FORÇA RELATIVA", out)

    def test_flag_short_history(self):
        d = self.hist_dir(900)
        code, out = self.run_cli("backtest", "--dir", d, "--rotation")
        self.assertEqual(code, 2)
        self.assertIn("ERRO", out)
