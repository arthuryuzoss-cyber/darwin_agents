"""v10: versão visível, dado desatualizado, aviso de fusível frouxo em live."""
import contextlib
import io
import unittest

import main as cli
from darwin import __version__
from darwin.config import ConfigError, load_config
from tests.helpers import Rig, flat

LAST_TS = 69 * 60000          # último candle do ScriptedMarket(flat(70))


def rig(delta_s):
    r = Rig(prices=flat(70), quote=40.0)
    r.orch.bootstrap(1)
    r.orch._clock = lambda: LAST_TS / 1000 + delta_s
    return r


class VersionTests(unittest.TestCase):
    def test_version_flag_and_banner(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["--version"])
        self.assertIn(f"darwin_agents v{__version__}", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["journal"], env={"MODE": "paper", "DATA_DIR": "/nonexistent-dir-x"})
        self.assertTrue(out.getvalue().startswith(f"darwin_agents v{__version__}"))

    def test_version_matches_readme(self):
        import os
        readme = open(os.path.join(os.path.dirname(cli.__file__), "README.md"), encoding="utf-8").read()
        self.assertIn(f"**Versão {__version__}**", readme.splitlines()[0])


class StaleDataTests(unittest.TestCase):
    def test_fresh_data_passes(self):
        r = rig(100)
        snap, errors = r.orch._snapshot(["BTC/BRL"])
        self.assertEqual(errors, {})
        self.assertIn("BTC/BRL", snap)

    def test_stale_data_is_rejected_per_symbol(self):
        r = rig(3600)
        snap, errors = r.orch._snapshot(["BTC/BRL"])
        self.assertEqual(snap, {})
        self.assertEqual(errors, {"BTC/BRL": "ExchangeError"})

    def test_stale_data_makes_no_trade_and_counts_as_market_failure(self):
        r = rig(3600)
        calls = len(r.sim.calls)
        rep = r.orch.tick()
        self.assertEqual(sum(1 for c in r.sim.calls[calls:] if c.startswith("create")), 0)
        self.assertTrue(any("falha ao ler mercado" in e for e in rep.events))

    def test_check_disabled_without_clock(self):
        r = Rig(prices=flat(70), quote=40.0)
        r.orch.bootstrap(1)
        _, errors = r.orch._snapshot(["BTC/BRL"])
        self.assertEqual(errors, {})

    def test_env_bounds(self):
        self.assertEqual(load_config({}).max_stale_candles, 3)
        self.assertEqual(load_config({"MAX_STALE_CANDLES": "10"}).max_stale_candles, 10)
        for bad in ("0", "500", "x"):
            with self.assertRaises(ConfigError):
                load_config({"MAX_STALE_CANDLES": bad})
