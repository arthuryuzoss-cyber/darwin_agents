"""Previsor de direção diária: sem olhar o futuro, treino antes do teste, custos, estatística, CLI."""
import contextlib
import io
import os
import random
import tempfile
import unittest

import main as cli
from darwin import direction as dr
from darwin.backtest import BacktestError
from darwin.history import TF_MS, align, csv_path, write_csv
from tests.test_backtest import SYMS, bt_cfg

DAY = TF_MS["1d"]
T0 = 1_600_000_000_000


def rows(closes):
    return [[T0 + i * DAY, c, c * 1.001, c * 0.999, c, 1.0] for i, c in enumerate(closes)]


def series(seed, n=1000, ar=0.0):
    rng = random.Random(seed)
    p, mom, out = 100.0, 0.0, []
    for _ in range(n):
        mom = ar * mom + rng.gauss(0, 0.02)
        p *= 1 + (0.3 * mom if ar else mom)
        out.append(p)
    return out


def hist(ar=0.0, n=1000):
    return align({s: rows(series(k, n, ar)) for k, s in enumerate(SYMS)}, "1d", {})


def cfg(**kw):
    return bt_cfg(timeframe="1d", **kw)


class FeatureTests(unittest.TestCase):
    def test_features_ignore_the_future(self):
        c = series(1, 200)
        t = 120
        a = dr._features(c, t)
        poisoned = c[:t + 1] + [1e9] * (len(c) - t - 1)
        self.assertEqual(a, dr._features(poisoned, t))
        self.assertIsNone(dr._features(c, 30))                       # histórico curto
        self.assertIsNone(dr._features(c[:40] + [None] + c[41:], 60))  # buraco

    def test_label_is_next_day(self):
        h = align({"A/BRL": rows([100.0] * 80 + [101.0, 99.0, 102.0] + [100.0] * 20)}, "1d", {})
        smp = {s["t"]: s for s in dr.build_samples(h, ["A/BRL"])}
        self.assertEqual(smp[79]["y"], 1)          # dia 79 -> 80 sobe
        self.assertEqual(smp[80]["y"], 0)          # 101 -> 99 desce
        self.assertEqual(smp[81]["y"], 1)          # 99 -> 102 sobe
        self.assertAlmostEqual(smp[80]["ret"], 99.0 / 101.0 - 1)

    def test_features_vector_matches_names(self):
        self.assertEqual(len(dr._features(series(1, 200), 100)), len(dr.NAMES))


class ModelTests(unittest.TestCase):
    def test_fit_learns_a_planted_relation(self):
        rng = random.Random(2)
        smp = []
        for _ in range(1500):
            x = [rng.gauss(0, 1) for _ in range(9)]
            smp.append({"x": x, "y": 1 if x[0] + rng.gauss(0, 0.5) > 0 else 0, "t": 0, "sym": "A", "ret": 0})
        m = dr.fit(smp)
        self.assertGreater(m.w[0], 0.5)
        self.assertLess(max(abs(w) for w in m.w[1:]), 0.3)
        acc = sum(1 for s in smp if (m.prob(s["x"]) >= 0.5) == bool(s["y"])) / len(smp)
        self.assertGreater(acc, 0.8)

    def test_binom_ci(self):
        lo, hi = dr.binom_ci(55, 100)
        self.assertTrue(lo < 0.55 < hi)
        self.assertLess(dr.binom_ci(5500, 10000)[1] - dr.binom_ci(5500, 10000)[0], hi - lo)
        self.assertIsNone(dr.binom_ci(0, 0))


class SimulateTests(unittest.TestCase):
    def test_costs_reduce_return_and_cash_when_no_signal(self):
        test = [{"t": t, "sym": "A", "ret": 0.01, "y": 1, "x": []} for t in range(50)]
        up = [0.9] * 50
        free = dr.simulate(test, up, (0, 50), cfg(fee_rate=0.0, slippage=0.0), 0.5)
        costly = dr.simulate(test, up, (0, 50), cfg(fee_rate=0.01, slippage=0.01), 0.5)
        self.assertGreater(free["return_pct"], 60)
        self.assertLess(costly["return_pct"], free["return_pct"])
        none = dr.simulate(test, [0.1] * 50, (0, 50), cfg(), 0.5)
        self.assertEqual((none["return_pct"], none["orders"], none["exposure_pct"]), (0.0, 0, 0.0))

    def test_flipping_signal_pays_each_time(self):
        test = [{"t": t, "sym": "A", "ret": 0.0, "y": 1, "x": []} for t in range(20)]
        probs = [0.9 if t % 2 == 0 else 0.1 for t in range(20)]
        m = dr.simulate(test, probs, (0, 20), cfg(), 0.5)
        self.assertEqual(m["orders"], 20)
        self.assertLess(m["return_pct"], 0)

    def test_buy_and_hold_benchmark_does_not_rebalance_daily(self):
        test = [
            {"t": 0, "sym": "A", "ret": 1.0, "y": 1, "x": []},
            {"t": 0, "sym": "B", "ret": 0.0, "y": 0, "x": []},
            {"t": 1, "sym": "A", "ret": -0.5, "y": 0, "x": []},
            {"t": 1, "sym": "B", "ret": 0.0, "y": 0, "x": []},
        ]
        m = dr.simulate(test, [0.1] * len(test), (0, 2), cfg(), 0.5)
        # A começa com 50%, dobra e depois perde metade; B fica estável.
        # Carteira estática termina em (1.0 + 1.0) / 2 = 1.0? O capital de A volta ao inicial.
        self.assertAlmostEqual(m["bh_return_pct"], 0.0, places=6)


class WalkForwardTests(unittest.TestCase):
    def test_noise_gives_no_edge_and_signal_is_detected(self):
        noise = dr.walk_forward(hist(0.0), cfg(), folds=3)["aggregate"]
        self.assertLess(noise["accuracy_ci95"][0], 55)
        self.assertTrue(noise["accuracy_ci95"][0] <= max(noise["up_rate_pct"], 100 - noise["up_rate_pct"]))
        sig = dr.walk_forward(hist(0.9), cfg(), folds=3)["aggregate"]
        self.assertGreater(sig["accuracy_pct"], noise["accuracy_pct"] + 3)

    def test_train_labels_never_overlap_test(self):
        h = hist(0.0)
        wf = dr.walk_forward(h, cfg(), folds=2)
        for f in wf["folds"]:
            self.assertGreater(f["n_train"], 200)
            self.assertLess(f["period"][0], f["period"][1])
        self.assertEqual(wf, dr.walk_forward(h, cfg(), folds=2))      # determinístico

    def test_changing_the_future_does_not_change_past_folds(self):
        base = {k: series(k, 1000) for k, _ in enumerate(SYMS)}
        h1 = align({s: rows(base[k]) for k, s in enumerate(SYMS)}, "1d", {})
        alt = {k: base[k][:] for k in base}
        for k in alt:
            alt[k][-100:] = [v * 3 for v in alt[k][-100:]]
        h2 = align({s: rows(alt[k]) for k, s in enumerate(SYMS)}, "1d", {})
        a, b = dr.walk_forward(h1, cfg(), folds=3), dr.walk_forward(h2, cfg(), folds=3)
        self.assertEqual(a["folds"][0], b["folds"][0])               # fold 1 termina bem antes da alteração

    def test_short_history_raises(self):
        with self.assertRaises(BacktestError):
            dr.walk_forward(hist(0.0, 150), cfg(), folds=4)

    def test_verdicts(self):
        base = {"accuracy_pct": 50.0, "accuracy_ci95": [48.0, 52.0], "up_rate_pct": 50.0, "n_test": 1000,
                "strategy_compounded_pct": -5.0, "bh_compounded_pct": 10.0, "positive_folds": 1, "n_folds": 4,
                "threshold": 0.52, "coefs": {}, "assets": 3}
        self.assertIn("SEM VANTAGEM", dr.verdict(base))
        self.assertIn("NÃO BATE", dr.verdict({**base, "strategy_compounded_pct": 5.0}))
        good = {**base, "accuracy_pct": 58.0, "accuracy_ci95": [55.0, 61.0], "strategy_compounded_pct": 30.0,
                "positive_folds": 3}
        self.assertIn("INDÍCIO", dr.verdict(good))

    def test_report_text(self):
        txt = dr.format_direction(dr.walk_forward(hist(0.0), cfg(), folds=2))
        for piece in ("PREVISOR DE DIREÇÃO", "RESUMO FORA DA AMOSTRA", "CONCLUSÃO", "comprar e segurar"):
            self.assertIn(piece, txt)


class CliDirectionTests(unittest.TestCase):
    def test_flag_and_1d_timeframe(self):
        d = tempfile.mkdtemp()
        for k, s in enumerate(SYMS):
            write_csv(csv_path(d, s, "1d"), rows(series(k, 900)))
        out = io.StringIO()
        env = {"MODE": "paper", "DATA_DIR": os.path.join(d, "data"), "BANKROLL_BRL": "40", "SYMBOLS": ",".join(SYMS)}
        with contextlib.redirect_stdout(out):
            code = cli.main(["backtest", "--dir", d, "--timeframe", "1d", "--direction", "--folds", "2"], env=env)
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("PREVISOR DE DIREÇÃO", out.getvalue())

    def test_short_history_is_clean_error(self):
        d = tempfile.mkdtemp()
        for k, s in enumerate(SYMS):
            write_csv(csv_path(d, s, "1d"), rows(series(k, 150)))
        out = io.StringIO()
        env = {"MODE": "paper", "DATA_DIR": os.path.join(d, "data"), "BANKROLL_BRL": "40", "SYMBOLS": ",".join(SYMS)}
        with contextlib.redirect_stdout(out):
            code = cli.main(["backtest", "--dir", d, "--timeframe", "1d", "--direction"], env=env)
        self.assertEqual(code, 2)
        self.assertIn("baixe mais dias", out.getvalue())
