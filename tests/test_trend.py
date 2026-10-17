"""Filtro de tendência longa: estratégia, mutação, config, backtest, CLI."""
import random
import tempfile
import os
import unittest
from dataclasses import replace

from darwin import backtest as bk
from darwin.config import ConfigError, load_config
from darwin.evolution import mutate_params
from darwin.strategy import StrategyError, StrategyParams, decide, sma
from tests import test_backtest as _tb
from tests.test_backtest import bt_cfg, make_hist, rows_from_closes, TF, SYMS
from darwin.history import align


def uptrend(n=300, step=0.001):
    out, p = [], 100.0
    for _ in range(n):
        p *= 1 + step
        out.append(p)
    return out


class StrategyTrendTests(unittest.TestCase):
    def test_validation(self):
        for bad in (-1, 10, 49, 401, True, 100.5):
            with self.assertRaises(StrategyError):
                StrategyParams(trend_period=bad)
        for ok in (0, 50, 400):
            StrategyParams(trend_period=ok)

    def test_roundtrip_dict(self):
        p = StrategyParams(trend_period=120)
        self.assertEqual(StrategyParams.from_dict(p.to_dict()).trend_period, 120)

    def test_off_by_default_changes_nothing(self):
        closes = uptrend()
        a = decide(StrategyParams(), closes, None, closes[-1], False)
        b = decide(StrategyParams(trend_period=0), closes, None, closes[-1], False)
        self.assertEqual(a, b)

    def test_blocks_buy_against_trend_and_sells_in_position(self):
        closes = uptrend(300, 0.001) + [150.0 * 0.5 ** (i / 20) for i in range(30)]
        # força sinal de compra: preço sobe no curto prazo mas abaixo da SMA longa
        closes = [200.0] * 250 + [100.0 + i * 0.2 for i in range(50)]
        p = StrategyParams(trend_period=200, buy_threshold=0.1)
        price = closes[-1]
        self.assertLess(price, sma(closes, 200))
        d = decide(p, closes, None, price, False)
        self.assertEqual(d.action, "hold")
        self.assertEqual(d.reason, "contra_tendencia")
        d2 = decide(p, closes, None, price, True, entry_price=price)
        self.assertEqual((d2.action, d2.reason), ("sell", "filtro_de_tendencia"))

    def test_allows_buy_with_trend(self):
        closes = uptrend(300, 0.002)
        d = decide(StrategyParams(trend_period=200, buy_threshold=0.2), closes, None, closes[-1], False)
        self.assertEqual(d.action, "buy")

    def test_insufficient_data_holds(self):
        closes = uptrend(120, 0.002)
        d = decide(StrategyParams(trend_period=200), closes, None, closes[-1], False)
        self.assertEqual((d.action, d.reason), ("hold", "dados_insuficientes"))

    def test_mutation_stays_in_bounds_and_off_stays_off(self):
        rng = random.Random(3)
        p = StrategyParams(trend_period=60)
        for _ in range(300):
            p = mutate_params(p, rng, 0.9)
            self.assertTrue(50 <= p.trend_period <= 400)
        q = StrategyParams()
        for _ in range(50):
            q = mutate_params(q, rng, 0.9)
            self.assertEqual(q.trend_period, 0)


class ConfigTrendTests(unittest.TestCase):
    def test_env(self):
        base = {"MODE": "paper", "DATA_DIR": tempfile.mkdtemp()}
        self.assertEqual(load_config(base).trend_period, 0)
        self.assertEqual(load_config({**base, "TREND_PERIOD": "200"}).trend_period, 200)
        for bad in ("30", "500", "-5", "abc"):
            with self.assertRaises(ConfigError):
                load_config({**base, "TREND_PERIOD": bad})


class BacktestTrendTests(unittest.TestCase):
    def downtrend_hist(self):
        closes = [100.0 * (0.9995 ** i) * (1 + 0.01 * ((i // 7) % 2)) for i in range(1500)]
        return align({s: rows_from_closes(closes) for s in SYMS}, TF, {})

    def test_filter_reduces_exposure_in_downtrend(self):
        hist = self.downtrend_hist()
        off = bk.run_backtest(hist, bt_cfg(params=StrategyParams(buy_threshold=0.1).to_dict()))
        on = bk.run_backtest(hist, bt_cfg(params=StrategyParams(buy_threshold=0.1, trend_period=200).to_dict()))
        self.assertLessEqual(on.metrics["exposure_pct"], off.metrics["exposure_pct"])
        self.assertLessEqual(on.metrics["n_trades"], off.metrics["n_trades"])

    def test_no_lookahead_with_trend(self):
        # mudar o futuro não pode alterar o resultado até o ponto cortado
        h1 = make_hist(1200, seed=5)
        a = bk.run_backtest(h1, bt_cfg(params=bk.default_params(True).to_dict()), end=900)
        b = bk.run_backtest(h1, bt_cfg(params=bk.default_params(True).to_dict()), end=900)
        self.assertEqual(a.metrics["return_pct"], b.metrics["return_pct"])

    def test_sample_params_trend(self):
        rng = random.Random(1)
        for _ in range(100):
            self.assertTrue(50 <= bk.sample_params(rng, True).trend_period <= 400)
            self.assertEqual(bk.sample_params(rng).trend_period, 0)
        self.assertEqual(bk.default_params(True).trend_period, bk.TREND_DEFAULT)
        self.assertEqual(bk.default_params().trend_period, 0)

    def test_walk_forward_with_trend(self):
        hist = make_hist(2600, seed=2)
        wf = bk.walk_forward(hist, bt_cfg(), folds=2, trials=3, min_trades=1, trend=True)
        self.assertTrue(wf["trend"])
        self.assertTrue(all(f["chosen_params"]["trend_period"] >= 50 for f in wf["folds"]))
        self.assertIn("COM filtro de tendência", bk.format_walk_forward(wf))


class CliTrendTests(unittest.TestCase):
    run_cli = _tb.CliTests.run_cli
    hist_dir = _tb.CliTests.hist_dir

    def test_backtest_flag(self):
        d, out_dir = self.hist_dir(), os.path.join(tempfile.mkdtemp(), "out")
        code, out = self.run_cli("backtest", "--dir", d, "--out", out_dir, "--fixed", "--trend-filter")
        self.assertEqual(code, 0, out)
        self.assertIn("Filtro de tendência ligado", out)

    def test_optimize_flag(self):
        d, out_dir = self.hist_dir(3000), os.path.join(tempfile.mkdtemp(), "out")
        code, out = self.run_cli("backtest", "--dir", d, "--out", out_dir, "--optimize", "--folds", "2",
                                 "--trials", "2", "--min-trades", "1", "--trend-filter")
        self.assertEqual(code, 0, out)
        self.assertIn("COM filtro de tendência", out)

    def test_init_applies_env_trend(self):
        from tests.helpers import MultiMarket, flat
        from darwin.sim import SimExchange
        sim = SimExchange(MultiMarket({s: flat(70) for s in SYMS}), list(SYMS), quote_balance=40.0, min_notional=10.0)
        code, out = self.run_cli("init", env={"TREND_PERIOD": "150"}, exchange_factory=lambda cfg: sim)
        self.assertEqual(code, 0, out)
        self.assertIn("Filtro de tendência ligado: SMA de 150", out)
