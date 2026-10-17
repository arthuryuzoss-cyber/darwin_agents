"""Momentum diário com filtro de mercado: sem olhar o futuro, regime, histerese, custos, CLI."""
import contextlib
import io
import os
import random
import tempfile
import unittest

import main as cli
from darwin import momentum as mo
from darwin.backtest import BacktestError
from darwin.history import TF_MS, align, csv_path, write_csv
from tests.test_backtest import bt_cfg

DAY = TF_MS["1d"]
SYMS3 = ("BTC/BRL", "ETH/BRL", "SOL/BRL")


def rows(closes):
    return [[1_600_000_000_000 + i * DAY, c, c * 1.01, c * 0.99, c, 1.0] for i, c in enumerate(closes)]


def gen(drift, n=1400, seed=0, vol=0.03):
    out = {}
    for k, s in enumerate(SYMS3):
        rng = random.Random(seed + k)
        p, cl = 100.0, []
        for _ in range(n):
            p *= 1 + rng.gauss(drift, vol)
            cl.append(p)
        out[s] = cl
    return out


def hist(series):
    return align({s: rows(c) for s, c in series.items()}, "1d", {})


def cfg(**kw):
    return bt_cfg(symbols=SYMS3, timeframe="1d", **kw)


class ChooseTests(unittest.TestCase):
    def test_market_below_sma_means_cash(self):
        series = gen(0.003)
        series["BTC/BRL"] = [100.0 * (0.996 ** i) for i in range(1400)]       # BTC em queda: filtro fecha
        h = hist(series)
        self.assertIsNone(mo.choose(h, list(SYMS3), 1000, None, mo.MomParams()))

    def test_uptrend_picks_something_and_downtrend_assets_are_excluded(self):
        series = {s: [100.0 * (1.003 ** i) for i in range(1400)] for s in SYMS3}
        series["SOL/BRL"] = [100.0 * (0.998 ** i) for i in range(1400)]
        pick = mo.choose(hist(series), list(SYMS3), 1000, None, mo.MomParams())
        self.assertIn(pick, ("BTC/BRL", "ETH/BRL"))

    def test_hysteresis_keeps_the_held_asset(self):
        series = {"BTC/BRL": [100.0 * (1.003 ** i) for i in range(1400)],
                  "ETH/BRL": [100.0 * (1.0031 ** i) for i in range(1400)],
                  "SOL/BRL": [100.0 * (1.0029 ** i) for i in range(1400)]}
        h = hist(series)
        p = mo.MomParams()
        self.assertEqual(mo.choose(h, list(SYMS3), 1000, "SOL/BRL", p), "SOL/BRL")   # diferença < margem
        self.assertNotEqual(mo.choose(h, list(SYMS3), 1000, None, p), None)


class SimulateTests(unittest.TestCase):
    def test_no_lookahead_future_is_poisoned(self):
        base = gen(0.001, seed=3)
        h1 = hist(base)
        poisoned = {s: c[:1100] + [c[1100] * 40] * (len(c) - 1100) for s, c in base.items()}
        h2 = hist(poisoned)
        a = mo.simulate(h1, cfg(), mo.MomParams(), 0, 1100)
        b = mo.simulate(h2, cfg(), mo.MomParams(), 0, 1100)
        self.assertEqual(a, b)

    def test_decision_uses_previous_close_only(self):
        # mudar o fechamento do dia i não pode mudar a decisão de entrada em i (executa na abertura de i)
        base = gen(0.002, seed=5)
        i = 900
        alt = {s: c[:] for s, c in base.items()}
        for s in alt:
            alt[s][i] = alt[s][i] * 0.5
        p = mo.MomParams()
        d1 = mo.choose(hist(base), list(SYMS3), i - 1, None, p)
        d2 = mo.choose(hist(alt), list(SYMS3), i - 1, None, p)
        self.assertEqual(d1, d2)

    def test_costs_reduce_return(self):
        h = hist(gen(0.002, seed=1))
        free = mo.simulate(h, cfg(fee_rate=0.0, slippage=0.0), mo.MomParams(), 0, 1400)
        costly = mo.simulate(h, cfg(fee_rate=0.01, slippage=0.01), mo.MomParams(), 0, 1400)
        self.assertLess(costly["return_pct"], free["return_pct"])
        self.assertGreater(costly["fees_cents"], 0)

    def test_below_min_notional_never_buys(self):
        h = hist(gen(0.003, seed=1))
        m = mo.simulate(h, cfg(bankroll_cents=500), mo.MomParams(), 0, 1400)       # R$5 < R$10
        self.assertEqual(m["n_trades"], 0)

    def test_bear_market_loses_less_than_buy_and_hold(self):
        wf = mo.walk_forward(hist(gen(-0.002, seed=2)), cfg())
        self.assertGreater(wf["full"]["return_pct"], wf["full_buy_hold"]["equal_weight_return_pct"])

    def test_short_history_raises(self):
        with self.assertRaises(BacktestError):
            mo.walk_forward(hist(gen(0.0, n=500)), cfg())
        with self.assertRaises(BacktestError):
            mo.walk_forward(hist({s: c for s, c in gen(0.0).items() if s != "BTC/BRL"}), cfg())


class WalkForwardTests(unittest.TestCase):
    def test_shape_report_and_determinism(self):
        h = hist(gen(0.001, seed=4))
        wf = mo.walk_forward(h, cfg())
        self.assertGreaterEqual(len(wf["folds"]), 5)
        for a, b in zip(wf["folds"], wf["folds"][1:]):
            self.assertEqual(a["end"], b["start"])                       # blocos consecutivos, sem sobreposição
        txt = mo.format_momentum(wf)
        for piece in ("MOMENTUM DIÁRIO", "RESUMO", "CONCLUSÃO", "comprar e segurar"):
            self.assertIn(piece, txt)
        self.assertEqual(wf, mo.walk_forward(h, cfg()))


class CliTests(unittest.TestCase):
    def test_flag(self):
        d = tempfile.mkdtemp()
        for s, c in gen(0.001, n=1300, seed=6).items():
            write_csv(csv_path(d, s, "1d"), rows(c))
        env = {"MODE": "paper", "DATA_DIR": os.path.join(d, "data"), "BANKROLL_BRL": "40", "SYMBOLS": ",".join(SYMS3)}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["backtest", "--dir", d, "--timeframe", "1d", "--momentum"], env=env)
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("MOMENTUM DIÁRIO", out.getvalue())
        out = io.StringIO()
        d2 = tempfile.mkdtemp()
        for s, c in gen(0.001, n=300).items():
            write_csv(csv_path(d2, s, "1d"), rows(c))
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["backtest", "--dir", d2, "--timeframe", "1d", "--momentum"], env=env), 2)
        self.assertIn("baixe mais dias", out.getvalue())


class CatalogueTests(unittest.TestCase):
    def test_years_table_and_monte_carlo(self):
        h = hist(gen(0.001, n=1500, seed=5))
        wf = mo.walk_forward(h, cfg())
        cat = mo.catalogue(h, cfg(), wf=wf, n_sims=200)
        self.assertGreaterEqual(len(cat["years"]), 2)
        self.assertIsNotNone(cat["mc"])
        mc = cat["mc"]
        self.assertLessEqual(mc["strat_p5"], mc["strat_p50"])
        self.assertLessEqual(mc["strat_p50"], mc["strat_p95"])
        self.assertTrue(0 <= mc["prob_loss_pct"] <= 100)
        txt = mo.format_catalogue(cat)
        self.assertIn("CATÁLOGO ANO A ANO", txt)
        self.assertIn("MONTE CARLO", txt)
        self.assertEqual(mo.catalogue(h, cfg(), wf=wf, n_sims=200)["mc"], mc)  # determinístico

    def test_bootstrap_identical_series_never_beats(self):
        r = [0.001, -0.002] * 100
        mc = mo.block_bootstrap(r, r, n_sims=100)
        self.assertEqual(mc["prob_beat_pct"], 0.0)

    def test_bootstrap_short_raises(self):
        with self.assertRaises(BacktestError):
            mo.block_bootstrap([0.0] * 10, [0.0] * 10)

    def test_market_follows_quote_currency(self):
        usdt = tuple(s.replace("/BRL", "/USDT") for s in SYMS3)
        h = align({s: rows(c) for s, c in zip(usdt, gen(0.001, n=900, seed=2).values())}, "1d", {})
        wf = mo.walk_forward(h, bt_cfg(symbols=usdt, timeframe="1d"))
        self.assertEqual(wf["params"]["market"], "BTC/USDT")


class EnvFileTests(unittest.TestCase):
    def test_env_file_option_is_used(self):
        d = tempfile.mkdtemp()
        data = gen(0.001, n=900, seed=3)
        for s, c in data.items():
            write_csv(csv_path(d, s, "1d"), rows(c))
        envf = os.path.join(d, ".env.longo")
        with open(envf, "w") as f:
            f.write(f"MODE=paper\nDATA_DIR={os.path.join(d, 'data')}\nBANKROLL_BRL=40\nSYMBOLS={','.join(SYMS3)}\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["--env-file", envf, "backtest", "--dir", d, "--timeframe", "1d", "--momentum"])
        out = buf.getvalue()
        self.assertEqual(code, 0, out)
        self.assertIn(".env.longo carregado", out)
        self.assertIn("CATÁLOGO ANO A ANO", out)


class SensitivityTests(unittest.TestCase):
    def test_grid_shape_and_reading(self):
        h = hist(gen(0.001, n=1500, seed=7))
        sv = mo.sensitivity(h, cfg())
        self.assertEqual(len(sv["rows"]), 15)
        txt = mo.format_sensitivity(sv)
        self.assertIn("SENSIBILIDADE", txt)
        self.assertIn("LEITURA", txt)

    def test_short_history_raises(self):
        with self.assertRaises(BacktestError):
            mo.sensitivity(hist(gen(0.001, n=420, seed=1)), cfg())

    def test_cli_flag(self):
        d = tempfile.mkdtemp()
        for s, c in gen(0.001, n=1200, seed=3).items():
            write_csv(csv_path(d, s, "1d"), rows(c))
        env = {"MODE": "paper", "DATA_DIR": os.path.join(d, "data"), "BANKROLL_BRL": "40", "SYMBOLS": ",".join(SYMS3)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["backtest", "--dir", d, "--timeframe", "1d", "--momentum", "--sensitivity"], env=env)
        self.assertEqual(code, 0)
        self.assertIn("SENSIBILIDADE DOS PARÂMETROS", buf.getvalue())
