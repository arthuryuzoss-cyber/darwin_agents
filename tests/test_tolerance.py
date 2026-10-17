"""Tolerância de perda e mínimo de trades antes de julgar um agente."""
import unittest

from darwin.budget import RiskBudget, RiskConfig
from darwin.config import ConfigError, load_config
from darwin.evolution import Darwinism, EvolutionConfig
from darwin.ledger import Ledger


def setup(**kw):
    led = Ledger()
    led.seed("a", 5000)
    cfg = EvolutionConfig(eval_min_roundtrips=3, eval_max_ticks=100, **kw)
    return led, Darwinism(led, cfg, RiskBudget(RiskConfig(5000, 60)))


def loss(led, cents):
    led.record_trade("a", -cents, "r1", "t")


def judge(d, rt=3, ticks=10):
    return d.judge("a", 0, 0, rt, ticks, False, 1, 1, 1000)


class ToleranceTests(unittest.TestCase):
    def test_default_literal_rule(self):
        led, d = setup()
        loss(led, 10)                                  # R$ 0,10
        v = judge(d)
        self.assertEqual(v.action, "die")

    def test_small_loss_within_tolerance_survives_without_cloning(self):
        led, d = setup(death_tolerance_pct=3.0)
        loss(led, 84)                                  # R$ 0,84 de R$ 50 (~1,7%)
        v = judge(d)
        self.assertEqual(v.action, "survive")
        self.assertIn("tolerância", v.reason)

    def test_big_loss_beyond_tolerance_dies(self):
        led, d = setup(death_tolerance_pct=3.0)
        loss(led, 300)                                 # R$ 3,00 de ~R$ 50 (6%)
        self.assertEqual(judge(d).action, "die")

    def test_zero_pnl_survives_with_tolerance(self):
        led, d = setup(death_tolerance_pct=2.0)
        self.assertEqual(judge(d, rt=1, ticks=100).action, "survive")

    def test_profit_still_clones(self):
        led, d = setup(death_tolerance_pct=3.0)
        led.record_trade("a", 400, "g1", "t")
        self.assertIn(judge(d).action, ("clone", "survive"))
        self.assertNotEqual(judge(d).action, "die")

    def test_needs_min_trades_renews_instead_of_killing(self):
        led, d = setup(needs_min_trades=True)
        loss(led, 84)
        v = judge(d, rt=1, ticks=100)                  # fim por tempo com 1 trade de 3
        self.assertEqual(v.action, "idle")
        self.assertEqual(judge(d, rt=3, ticks=100).action, "die")   # com 3 trades já pode ser julgado

    def test_bounds_and_env(self):
        with self.assertRaises(ValueError):
            EvolutionConfig(death_tolerance_pct=25)
        with self.assertRaises(ValueError):
            EvolutionConfig(death_tolerance_pct=-1)
        self.assertEqual(load_config({}).death_tolerance_pct, 0.0)
        c = load_config({"DEATH_TOLERANCE_PCT": "3,5", "DEATH_NEEDS_MIN_TRADES": "true"})
        self.assertEqual((c.death_tolerance_pct, c.needs_min_trades), (3.5, True))
        for bad in ("50", "-1", "abc", "nan"):
            with self.assertRaises(ConfigError):
                load_config({"DEATH_TOLERANCE_PCT": bad})
