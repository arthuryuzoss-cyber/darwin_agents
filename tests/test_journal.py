"""Diário de sinais: gravação segura, retorno futuro sem erro de alinhamento, estatística, CLI."""
import contextlib
import io
import os
import random
import tempfile
import unittest

import main as cli
from darwin import journal as jn

H = 3_600_000


def rows_synth(n=600, predictive=True, seed=1, step_min=10):
    rng = random.Random(seed)
    out, price = [], 100.0
    sig = [rng.uniform(-1, 1) for _ in range(n)]
    for i in range(n):
        out.append({"ts": i * step_min * 60_000, "symbol": "BTC/BRL", "price": price, "tech": sig[i], "sent": sig[i]})
        drift = (0.002 * sig[i] if predictive else 0.0) + rng.gauss(0, 0.001)
        price *= 1 + drift
    return out


class JournalWriteTests(unittest.TestCase):
    def test_writes_throttled_and_header_once(self):
        d = tempfile.mkdtemp()
        t = [1000.0]
        j = jn.Journal(os.path.join(d, "j.csv"), every_s=300, clock=lambda: t[0])
        closes = [100 + i * 0.1 for i in range(100)]
        snap = {"BTC/BRL": (110.0, closes), "ETH/BRL": (50.0, closes)}
        self.assertEqual(j.record(snap, {"BTC/BRL": 0.3, "ETH/BRL": None}), 2)
        t[0] += 10
        self.assertEqual(j.record(snap, {}), 0)               # dentro do intervalo
        t[0] += 300
        self.assertEqual(j.record(snap, {}), 2)
        rows = jn.load(j.path)
        self.assertEqual(len(rows), 4)
        self.assertEqual(open(j.path).read().count("ts_ms"), 1)
        self.assertIsNone(rows[1]["sent"])

    def test_bad_data_is_skipped_and_io_error_is_swallowed(self):
        d = tempfile.mkdtemp()
        j = jn.Journal(os.path.join(d, "j.csv"), every_s=0)
        closes = [100.0] * 100
        self.assertEqual(j.record({"X": (float("nan"), closes), "Y": (-1.0, closes), "Z": (5.0, [1.0])}, {}), 0)
        blocker = os.path.join(d, "arquivo")
        open(blocker, "w").close()
        j2 = jn.Journal(os.path.join(blocker, "sub", "j.csv"), every_s=0)   # diretório impossível
        self.assertEqual(j2.record({"A": (5.0, closes)}, {}), 0)
        self.assertEqual(j2.errors, 1)


class StatsTests(unittest.TestCase):
    def test_forward_return_alignment_and_no_future_row(self):
        rows = [{"ts": i * H, "symbol": "A", "price": 100.0 + i, "tech": 0.0, "sent": None} for i in range(10)]
        fs = jn.forward_samples(rows, 4)
        self.assertEqual(len(fs), 6)                         # as últimas 4 não têm preço 4h depois
        self.assertAlmostEqual(fs[0]["ret"], 4.0)
        self.assertNotIn(9 * H, [r["ts"] for r in fs])

    def test_spearman_basic(self):
        self.assertAlmostEqual(jn.spearman(list(range(20)), list(range(20))), 1.0)
        self.assertAlmostEqual(jn.spearman(list(range(20)), list(range(20, 0, -1))), -1.0)
        self.assertIsNone(jn.spearman([1] * 20, list(range(20))))
        self.assertIsNone(jn.spearman([1, 2], [1, 2]))

    def test_detects_real_signal_and_rejects_noise(self):
        good = jn.analyze(rows_synth(1500, True))
        self.assertGreater(good["horizons"][1]["tech_corr"], 0.1)
        self.assertGreater(good["horizons"][1]["tech_ci"][0], 0)
        noise = jn.analyze(rows_synth(1500, False, seed=7))
        lo, hi = noise["horizons"][1]["tech_ci"]
        self.assertLess(lo, 0.05)
        self.assertGreater(hi, -0.05)

    def test_report_conclusions(self):
        self.assertIn("INCONCLUSIVO", jn.format_analysis(jn.analyze(rows_synth(100))))
        txt = jn.format_analysis(jn.analyze(rows_synth(1500, True)))
        self.assertIn("INDÍCIO", txt)
        self.assertIn("SEM VANTAGEM", jn.format_analysis(jn.analyze(rows_synth(1500, False, seed=7))))

    def test_load_skips_garbage_lines(self):
        p = os.path.join(tempfile.mkdtemp(), "j.csv")
        open(p, "w").write("ts_ms,symbol,price,tech,sentiment\n1,A,2,0.1,\nlixo,A,2,0.1,\n2,A,x,0.1,\n")
        self.assertEqual(len(jn.load(p)), 1)


class CliJournalTests(unittest.TestCase):
    def test_no_journal_yet(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["journal"], env={"MODE": "paper", "DATA_DIR": tempfile.mkdtemp()})
        self.assertEqual(code, 2)
        self.assertIn("Ainda não há diário", out.getvalue())

    def test_orchestrator_records_and_survives_journal_failure(self):
        from tests.helpers import MultiMarket, flat
        from darwin.sim import SimExchange
        syms = ("BTC/BRL", "ETH/BRL")
        d = tempfile.mkdtemp()
        sim = SimExchange(MultiMarket({s: flat(70) for s in syms}), list(syms), quote_balance=40.0, min_notional=10.0)
        env = {"MODE": "paper", "BANKROLL_BRL": "40", "SYMBOLS": ",".join(syms), "DATA_DIR": d}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["init"], env=env, exchange_factory=lambda c: sim, sentiment=None), 0)
            self.assertEqual(cli.main(["run", "--max-ticks", "3"], env=env, exchange_factory=lambda c: sim,
                                      sentiment=None, sleep=lambda s: None), 0)
        rows = jn.load(os.path.join(d, "journal.csv"))
        self.assertEqual({r["symbol"] for r in rows}, set(syms))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["journal"], env=env), 0)
        self.assertIn("DIÁRIO DE SINAIS", out.getvalue())
