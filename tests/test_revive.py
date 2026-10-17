"""Renascimento: o último agente que morre não encerra o sistema se ainda há capital viável."""
import unittest

from darwin.config import load_config
from darwin.orchestrator import TREASURY
from tests.helpers import Rig, flat
from tests.test_integration import BUYABLE, RAMP, bought, drive_until


def losing_rig(**kw):
    r = Rig(prices=flat(70), quote=40.0, **kw)
    r.orch.bootstrap(1)
    r.tune(**BUYABLE)
    assert drive_until(r, RAMP, bought)
    r.market.advance(r.orch.traders["alpha"].entry_price * 0.975)    # passa do stop-loss
    r.orch.tick()
    return r


class ReviveTests(unittest.TestCase):
    def test_last_agent_dies_and_a_mutated_one_is_born_with_the_remaining_capital(self):
        r = losing_rig()
        self.assertEqual(r.orch.stop_reason, "")
        self.assertEqual(list(r.orch.traders), ["fenix1"])
        f = r.orch.traders["fenix1"]
        self.assertEqual(f.generation, 2)
        self.assertNotEqual(f.params, r.orch.graveyard[0] and type(f.params).from_dict(r.orch.graveyard[0]["params"]))
        self.assertGreaterEqual(r.ledger.balance("fenix1"), r.orch.min_trade_cents())
        self.assertEqual(r.ledger.balance(TREASURY), 0)
        self.assertEqual(r.ledger.balance("fenix1"), r.ledger.global_balance())     # dinheiro conservado
        self.assertIn("🔥", r.text())
        r.orch.tick()                                                              # continua operando
        self.assertEqual(r.orch.stop_reason, "")

    def test_revive_off_keeps_literal_extinction(self):
        r = losing_rig(revive=False)
        self.assertEqual(r.orch.stop_reason, "extinct")
        self.assertEqual(r.orch.traders, {})

    def test_no_revive_when_capital_is_not_viable(self):
        r = Rig(prices=flat(70), quote=40.0)
        r.orch.bootstrap(1)
        r.ledger.record_trade("alpha", -3600, "perda-teste", "teste")           # sobra R$ 4 < mínimo
        r.orch._kill(r.orch.traders["alpha"], "teste", None)
        self.assertFalse(r.orch._revive(_Rep()))

    def test_no_revive_when_fuse_is_open(self):
        r = losing_rig(revive=False)
        r.orch.stop_reason = ""
        r.orch.breaker.trip("teste")
        self.assertFalse(r.orch._revive(_Rep()))

    def test_repeated_deaths_get_unique_names(self):
        r = losing_rig()
        r.orch._kill(r.orch.traders["fenix1"], "teste", None)
        self.assertTrue(r.orch._revive(_Rep()))
        self.assertEqual(list(r.orch.traders), ["fenix2"])
        self.assertEqual(r.orch.traders["fenix2"].generation, 3)

    def test_env_flag(self):
        self.assertTrue(load_config({}).revive)
        self.assertFalse(load_config({"REVIVE": "false"}).revive)
        self.assertFalse(load_config({"REVIVE": "não"}).revive)


class _Rep:
    events: list = []
    def __init__(self):
        self.events = []
