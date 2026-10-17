"""Funcional: a CLI de ponta a ponta (main.main) sobre exchange simulada."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import main as cli
from darwin.config import load_config
from darwin.exchange import ExchangeError, make_exchange
from darwin.sim import SimExchange
from tests.helpers import FixedSentiment, ScriptedMarket, flat

KEYS = {"EXCHANGE_API_KEY": "k" * 20, "EXCHANGE_SECRET_KEY": "s" * 20}


class Cli:
    """Mantém UMA exchange simulada entre invocações da CLI (como a conta real persiste)."""

    def __init__(self, sentiment=None, prices=None, quote=40.0, **env):
        self.dir = tempfile.mkdtemp()
        self.env = {"MODE": "paper", "DATA_DIR": self.dir, "BANKROLL_BRL": "40", "TICK_SECONDS": "5",
                    "DARWIN_RESET_TOKEN": "token-do-dono-123", **env}
        self.market = ScriptedMarket(prices or flat(80), 10.0)
        self.sim = SimExchange(self.market, "BTC/BRL", quote_balance=quote, min_notional=10.0)
        self.sentiment = sentiment
        self.sleep = lambda s: None

    def __call__(self, *argv, env=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(list(argv), env=env or self.env, exchange_factory=lambda cfg: self.sim,
                            sentiment=self.sentiment, sleep=self.sleep)
        return code, out.getvalue()


class WorkflowTests(unittest.TestCase):
    def test_check_is_read_only(self):
        c = Cli()
        code, out = c("check")
        self.assertEqual(code, 0)
        self.assertIn("nenhuma ordem foi enviada", out)
        self.assertIn("ordem mínima da exchange: 10.00 BRL", out)
        self.assertFalse(any(x.startswith("create") for x in c.sim.calls))

    def test_init_prints_fuse_and_warns_for_small_bankroll(self):
        code, out = Cli(quote=40.0)("init")
        self.assertEqual(code, 0)
        self.assertIn("baseline R$ 40,00", out)
        self.assertIn("dispara em R$ 16,00", out)
        self.assertNotIn("ATENÇÃO", out)
        code, out = Cli(quote=20.0, BANKROLL_BRL="20")("init")
        self.assertEqual(code, 0)
        self.assertIn("dispara em R$ 8,00", out)
        self.assertIn("ATENÇÃO", out)
        self.assertIn("IMPOSSÍVEL", out)

    def test_run_before_init_and_double_init_are_refused(self):
        c = Cli()
        self.assertEqual(c("run", "--max-ticks", "1")[0], 2)
        self.assertEqual(c("init")[0], 0)
        code, out = c("init")
        self.assertEqual(code, 2)
        self.assertIn("já inicializado", out)

    def test_dirty_account_is_refused(self):
        code, out = Cli(quote=500.0)("init")
        self.assertEqual(code, 2)
        self.assertIn("subconta", out)

    def test_run_and_status(self):
        c = Cli()
        c("init")
        code, out = c("run", "--max-ticks", "3")
        self.assertEqual(code, 0)
        code, out = c("status")
        self.assertEqual(code, 0)
        self.assertIn("Patrimônio (API): R$ 40,00", out)
        self.assertIn("piso do fusível: R$ 16,00", out)
        self.assertIn("alpha gen1", out)

    def test_full_fuse_story_through_the_cli(self):
        c = Cli(sentiment=FixedSentiment(1.0))        # flat + sentimento +1 => compra no 1º tick
        c("init")
        code, out = c("run", "--max-ticks", "2")
        self.assertEqual(code, 0)
        self.assertIn("COMPRA", out)
        self.assertGreater(float(c.sim.balances["BTC"]), 0.1)
        c.market.advance(100.0 * 0.35)                # queda de 65%
        code, out = c("run", "--max-ticks", "1")
        self.assertEqual(code, 3)
        self.assertIn("DISJUNTOR DISPARADO", out)
        self.assertIn("liquidação concluída", out)
        self.assertLess(float(c.sim.balances["BTC"]), 0.001)
        calls = len(c.sim.calls)
        code, out = c("run", "--max-ticks", "5")      # reiniciar não volta a operar
        self.assertEqual(code, 3)
        self.assertEqual(sum(1 for x in c.sim.calls[calls:] if x.startswith("create")), 0)
        self.assertIn("ABERTO", c("status")[1])
        # reset: token errado negado; certo reabre, mas a baseline mantida dispara de novo
        self.assertEqual(c("reset-breaker", "--token", "errado")[0], 2)
        self.assertEqual(c("reset-breaker", "--token", "token-do-dono-123")[0], 0)
        self.assertEqual(c("run", "--max-ticks", "1")[0], 3)
        # reset com rebase aceita a nova realidade e volta a rodar
        self.assertEqual(c("reset-breaker", "--token", "token-do-dono-123", "--rebase")[0], 0)
        self.assertEqual(c("run", "--max-ticks", "1")[0], 0)

    def test_second_run_is_refused_while_first_is_active(self):
        from darwin.lock import acquire_instance_lock
        c = Cli()
        c("init")
        held = acquire_instance_lock(c.dir)            # simula o 1º robô ainda rodando
        code, out = c("run", "--max-ticks", "1")
        self.assertEqual(code, 2)
        self.assertIn("já existe um robô rodando", out)
        held.close()
        self.assertEqual(c("run", "--max-ticks", "1")[0], 0)  # liberado: roda normalmente

    def test_report_summarizes_results(self):
        c = Cli(sentiment=FixedSentiment(1.0))
        c("init")
        code, out = c("report")
        self.assertEqual(code, 0)
        self.assertIn("Trades fechados: 0", out)
        c("run", "--max-ticks", "2")
        c.market.advance(95.0)
        c("run", "--max-ticks", "1")
        code, out = c("report")
        self.assertIn("Trades fechados: 1", out)
        self.assertIn("Resultado líquido de taxas", out)

    def test_liquidate_command_sells_everything(self):
        c = Cli(sentiment=FixedSentiment(1.0))
        c("init")
        c("run", "--max-ticks", "2")
        self.assertGreater(float(c.sim.balances["BTC"]), 0.1)
        code, out = c("liquidate")
        self.assertEqual(code, 0)
        self.assertLess(float(c.sim.balances["BTC"]), 0.001)

    def test_ctrl_c_warns_about_open_position_or_liquidates_on_request(self):
        def interrupt(_):
            raise KeyboardInterrupt
        c = Cli(sentiment=FixedSentiment(1.0))
        c.sleep = interrupt
        c("init")
        code, out = c("run")
        self.assertEqual(code, 0)
        self.assertIn("posição ABERTA", out)
        self.assertGreater(float(c.sim.balances["BTC"]), 0.1)
        code, out = c("run", "--liquidate-on-exit")
        self.assertLess(float(c.sim.balances["BTC"]), 0.001)


class LiveGatingTests(unittest.TestCase):
    def test_live_needs_both_switches(self):
        env = {"MODE": "live", **KEYS}
        c = Cli(**env)
        with self.assertRaises(SystemExit) as cm:
            c("check", env={**c.env})                  # MODE=live sem --live
        self.assertIn("--live", str(cm.exception))
        c2 = Cli()
        with self.assertRaises(SystemExit):
            c2("check", "--live")                      # --live com MODE=paper
        c3 = Cli(**env)
        self.assertEqual(c3("check", "--live", env=c3.env)[0], 0)   # os dois: libera

    def test_live_without_keys_is_a_config_error(self):
        c = Cli(MODE="live")
        code, out = c("check", "--live")
        self.assertEqual(code, 2)
        self.assertIn("EXCHANGE_API_KEY", out)

    def test_bad_config_exits_2(self):
        c = Cli(MODE="yolo")
        self.assertEqual(c("check")[0], 2)
        c = Cli(MAX_LOSS_PCT="100")
        self.assertEqual(c("check")[0], 2)


class FakeCcxtModule:
    """Contrato mínimo do ccxt: confere COMO o projeto cria a conexão com a Binance."""

    created = []

    class binance:
        def __init__(self, config):
            FakeCcxtModule.created.append(self)
            self.config = config
            self.sandbox = False

        def set_sandbox_mode(self, flag):
            self.sandbox = flag

        def fetch_ticker(self, s):
            return {"last": 100.0}

        def fetch_ohlcv(self, s, tf, limit=None):
            return [[0, 1, 1, 1, 100.0, 1]] * (limit or 5)

        def load_markets(self):
            return {}

        def market(self, s):
            return {"limits": {"cost": {"min": 10.0}}}


class CcxtContractTests(unittest.TestCase):
    def setUp(self):
        FakeCcxtModule.created.clear()

    def test_live_connection_uses_keys_spot_and_rate_limit(self):
        cfg = load_config({"MODE": "live", **KEYS})
        make_exchange(cfg, FakeCcxtModule)
        ex = FakeCcxtModule.created[0]
        self.assertEqual(ex.config["apiKey"], KEYS["EXCHANGE_API_KEY"])
        self.assertEqual(ex.config["secret"], KEYS["EXCHANGE_SECRET_KEY"])
        self.assertTrue(ex.config["enableRateLimit"])
        self.assertEqual(ex.config["options"]["defaultType"], "spot")
        self.assertFalse(ex.sandbox)

    def test_testnet_enables_sandbox(self):
        make_exchange(load_config({"MODE": "testnet", **KEYS}), FakeCcxtModule)
        self.assertTrue(FakeCcxtModule.created[0].sandbox)

    def test_paper_uses_public_data_without_keys_and_simulates_orders(self):
        d = tempfile.mkdtemp()
        ex = make_exchange(load_config({"MODE": "paper", "DATA_DIR": d, **KEYS}), FakeCcxtModule)
        self.assertIsInstance(ex, SimExchange)
        self.assertNotIn("apiKey", FakeCcxtModule.created[0].config)  # paper nunca usa as chaves

    def test_missing_ccxt_gives_clear_error(self):
        with mock.patch.dict(sys.modules, {"ccxt": None}):
            with self.assertRaises(ExchangeError) as cm:
                make_exchange(load_config({}))
        self.assertIn("pip install ccxt", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
