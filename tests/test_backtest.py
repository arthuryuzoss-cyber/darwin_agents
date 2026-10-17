"""Histórico, backtest (sem olhar o futuro), walk-forward, estatística e comandos de CLI."""
import contextlib
import io
import json
import os
import random
import tempfile
import unittest
from dataclasses import replace

import main as cli
from darwin import backtest as bk
from darwin.history import (PAGE, TF_MS, HistoryError, align, csv_path, download_history, load_history,
                            merge_rows, read_csv, write_csv)
from darwin.strategy import BOUNDS, StrategyParams

TF = "5m"
STEP = TF_MS[TF]
T0 = 1_700_000_000_000
SYMS = ("BTC/BRL", "ETH/BRL", "SOL/BRL")


def rows_from_closes(closes, t0=T0, first=0):
    out = []
    for i, c in enumerate(closes):
        out.append([t0 + (i + first) * STEP, c, c * 1.0005, c * 0.9995, c, 10.0])
    return out


def walk(seed, n, drift=0.00002, vol=0.003, start=100.0):
    rng = random.Random(seed)
    p, closes = start, []
    for _ in range(n):
        p *= 1 + rng.gauss(drift, vol)
        closes.append(round(p, 6))
    return closes


def make_hist(n=1500, symbols=SYMS, seed=1, **kw):
    raw = {s: rows_from_closes(walk(seed + k, n, **kw)) for k, s in enumerate(symbols)}
    return align(raw, TF, {})


def bt_cfg(symbols=SYMS, **kw):
    base = dict(symbols=tuple(symbols), bankroll_cents=4000, timeframe=TF, min_notional=10.0)
    base.update(kw)
    return bk.BacktestConfig(**base)


def crash_hist(symbols=SYMS, flat_n=160, ramp_n=60, tail_n=60, drop=0.25):
    """Lateral, alta forte (o robô compra), depois queda brusca de 70% em tudo."""
    closes = [100.0] * flat_n
    p = 100.0
    for _ in range(ramp_n):
        p *= 1.006
        closes.append(p)
    closes += [p * drop] * tail_n
    return align({s: rows_from_closes(closes) for s in symbols}, TF, {})


BUYABLE = dict(buy_threshold=0.1, rsi_overbought=85.0, take_profit_pct=9.0, stop_loss_pct=5.0).items()


class FakeCcxt:
    """Contrato mínimo do ccxt para o download: paginação por `since`, limite por página, falhas injetáveis."""
    series: dict = {}
    calls: list = []
    fail: dict = {}
    precision_mode = 4

    class binance:
        def __init__(self, config):
            assert config.get("enableRateLimit")
            self.precisionMode = FakeCcxt.precision_mode

        def load_markets(self):
            return {s: {} for s in FakeCcxt.series}

        def market(self, s):
            return {"limits": {"cost": {"min": 10.0}}, "precision": {"amount": 0.001}}

        def fetch_ohlcv(self, s, tf, since=None, limit=None):
            FakeCcxt.calls.append((s, since, limit))
            if FakeCcxt.fail.get(s, 0) > 0:
                FakeCcxt.fail[s] -= 1
                raise TimeoutError("rede")
            rows = [r for r in FakeCcxt.series[s] if r[0] >= since]
            return rows[: limit or 500]


class HistoryTests(unittest.TestCase):
    def setUp(self):
        FakeCcxt.series, FakeCcxt.calls, FakeCcxt.fail, FakeCcxt.precision_mode = {}, [], {}, 4
        self.d = tempfile.mkdtemp()

    def test_csv_roundtrip_and_merge_dedupes_with_new_data_winning(self):
        a = rows_from_closes([1, 2, 3])
        p = os.path.join(self.d, "x.csv")
        write_csv(p, a)
        self.assertEqual([r[4] for r in read_csv(p)], [1.0, 2.0, 3.0])
        b = rows_from_closes([9, 4], first=2)                     # sobrepõe o 3º candle
        m = merge_rows(a, b)
        self.assertEqual([r[4] for r in m], [1.0, 2.0, 9.0, 4.0])
        self.assertEqual([r[0] for r in m], sorted(r[0] for r in m))

    def test_corrupt_csv_is_rejected(self):
        p = os.path.join(self.d, "bad.csv")
        open(p, "w").write("timestamp_ms,open,high,low,close,volume\n1,2,3\n")
        with self.assertRaises(HistoryError):
            read_csv(p)
        open(p, "w").write("timestamp_ms,open,high,low,close,volume\n1,5,4,6,5,1\n")      # high < low
        with self.assertRaises(HistoryError):
            read_csv(p)
        open(p, "w").write("timestamp_ms,open,high,low,close,volume\n1,5,6,4,nan,1\n")
        with self.assertRaises(HistoryError):
            read_csv(p)

    def test_download_paginates_saves_meta_and_skips_unlisted(self):
        n = 2500
        FakeCcxt.series = {"BTC/BRL": rows_from_closes(walk(1, n), t0=T0)}
        now = T0 + n * STEP
        rep = download_history(FakeCcxt, ["BTC/BRL", "ZZZ/BRL"], TF, 30, self.d, log=lambda m: None,
                               now_ms=now, sleep=lambda s: None)
        self.assertEqual(rep["BTC/BRL"], n)
        self.assertIn("não existe", rep["ZZZ/BRL"])
        self.assertGreaterEqual(len([c for c in FakeCcxt.calls if c[0] == "BTC/BRL"]), n // PAGE)
        self.assertEqual(len(read_csv(csv_path(self.d, "BTC/BRL", TF))), n)
        meta = json.load(open(os.path.join(self.d, "meta.json")))
        self.assertEqual(meta["BTC/BRL"], {"min_cost": 10.0, "amount_step": 0.001})
        self.assertFalse(os.path.exists(csv_path(self.d, "ZZZ/BRL", TF)))

    def test_download_is_resumable_and_only_fetches_what_is_missing(self):
        FakeCcxt.series = {"BTC/BRL": rows_from_closes(walk(1, 1200))}
        now = T0 + 1200 * STEP
        download_history(FakeCcxt, ["BTC/BRL"], TF, 4, self.d, log=lambda m: None, now_ms=now, sleep=lambda s: None)
        FakeCcxt.series["BTC/BRL"] = rows_from_closes(walk(1, 1500))      # chegaram 300 candles novos
        FakeCcxt.calls.clear()
        download_history(FakeCcxt, ["BTC/BRL"], TF, 4, self.d, log=lambda m: None,
                         now_ms=T0 + 1500 * STEP, sleep=lambda s: None)
        self.assertEqual(len(read_csv(csv_path(self.d, "BTC/BRL", TF))), 1500 - 48)   # 1ª janela começou no candle 48
        first_since = FakeCcxt.calls[0][1]
        self.assertGreater(first_since, T0 + 1000 * STEP)                 # não rebaixou tudo

    def test_asking_for_more_days_backfills_the_beginning(self):
        FakeCcxt.series = {"BTC/BRL": rows_from_closes(walk(1, 1500))}
        now = T0 + 1500 * STEP
        days_small = 800 * STEP / 86_400_000
        download_history(FakeCcxt, ["BTC/BRL"], TF, 1, self.d, log=lambda m: None, now_ms=now, sleep=lambda s: None)
        n1 = len(read_csv(csv_path(self.d, "BTC/BRL", TF)))
        self.assertLess(n1, 400)                                          # 1 dia = 288 candles
        download_history(FakeCcxt, ["BTC/BRL"], TF, 5, self.d, log=lambda m: None, now_ms=now, sleep=lambda s: None)
        rows = read_csv(csv_path(self.d, "BTC/BRL", TF))
        self.assertGreater(len(rows), n1 + 1000)                          # completou o começo
        self.assertEqual([r[0] for r in rows], sorted({r[0] for r in rows}))   # sem duplicatas e em ordem

    def test_download_retries_then_succeeds_and_failure_keeps_old_file(self):
        FakeCcxt.series = {"BTC/BRL": rows_from_closes(walk(1, 300))}
        now = T0 + 300 * STEP
        FakeCcxt.fail = {"BTC/BRL": 2}                                    # 2 falhas, 3ª tentativa passa
        rep = download_history(FakeCcxt, ["BTC/BRL"], TF, 5, self.d, log=lambda m: None, now_ms=now, sleep=lambda s: None)
        self.assertEqual(rep["BTC/BRL"], 300)
        FakeCcxt.series["BTC/BRL"] = rows_from_closes(walk(1, 400))
        FakeCcxt.fail = {"BTC/BRL": 99}                                   # falha permanente
        rep = download_history(FakeCcxt, ["BTC/BRL"], TF, 5, self.d, log=lambda m: None,
                               now_ms=T0 + 400 * STEP, sleep=lambda s: None)
        self.assertIn("falha de rede", rep["BTC/BRL"])
        self.assertEqual(len(read_csv(csv_path(self.d, "BTC/BRL", TF))), 300)   # arquivo antigo intacto

    def test_lot_step_ignored_when_ccxt_is_not_in_tick_size_mode(self):
        FakeCcxt.series = {"BTC/BRL": rows_from_closes(walk(1, 50))}
        FakeCcxt.precision_mode = 2                                       # DECIMAL_PLACES: "0.001" não seria um passo
        download_history(FakeCcxt, ["BTC/BRL"], TF, 1, self.d, log=lambda m: None, now_ms=T0 + 50 * STEP, sleep=lambda s: None)
        self.assertNotIn("amount_step", json.load(open(os.path.join(self.d, "meta.json")))["BTC/BRL"])

    def test_download_rejects_bad_arguments(self):
        with self.assertRaises(HistoryError):
            download_history(FakeCcxt, ["BTC/BRL"], "7m", 5, self.d)
        with self.assertRaises(HistoryError):
            download_history(FakeCcxt, ["BTC/BRL"], TF, 0, self.d)

    def test_align_fills_gaps_and_marks_unlisted_and_delisted(self):
        a = rows_from_closes([1, 2, 3, 4, 5, 6])
        b = [r for i, r in enumerate(rows_from_closes([10, 11, 12, 13, 14, 15])) if i != 2]   # buraco no meio
        late = rows_from_closes([7, 8], first=3)                                              # listado depois
        gone = rows_from_closes([5, 5], first=0)                                              # sumiu após 2 candles
        h = align({"A": a, "B": b, "L": late, "G": gone}, TF)
        self.assertEqual(h.n, 6)
        self.assertEqual(h.ohlcv["B"][2][3], 11.0)                        # repete o fechamento anterior
        self.assertEqual(h.ohlcv["B"][2][4], 0.0)                         # volume 0 no candle preenchido
        self.assertEqual(h.gaps["B"], 1)
        self.assertIsNone(h.ohlcv["L"][0])
        self.assertEqual(h.first_idx["L"], 3)
        self.assertIsNone(h.ohlcv["G"][4])                                # depois do último candle: sem dado
        with self.assertRaises(HistoryError):
            align({"X": [[T0, 1, 1, 1, 1, 1], [T0 + STEP + 7, 1, 1, 1, 1, 1]]}, TF)   # fora da grade
        with self.assertRaises(HistoryError):
            align({}, TF)

    def test_load_history_needs_files(self):
        with self.assertRaises(HistoryError):
            load_history(self.d, SYMS, TF)


class NoLookaheadTests(unittest.TestCase):
    def test_market_never_returns_the_future(self):
        h = make_hist(300)
        m = bk.ReplayMarket(h, 300, 10.0)
        for cur in (120, 200, 299):
            m.cursor = cur
            rows = m.fetch_ohlcv("BTC/BRL", TF, 100)
            self.assertEqual(rows[-1][0], h.timeline[cur])
            self.assertLessEqual(max(r[0] for r in rows), h.timeline[cur])
            self.assertEqual(m.fetch_ticker("BTC/BRL")["last"], h.close("BTC/BRL", cur))
            self.assertLessEqual(len(rows), 100)
        m.cursor = 300
        with self.assertRaises(RuntimeError):
            m.fetch_ticker("BTC/BRL")

    def test_changing_future_candles_cannot_change_the_past(self):
        """Envenena tudo depois do ponto de corte: o resultado até o corte tem de ser idêntico."""
        h1 = make_hist(900, seed=4)
        cut = 600
        h2 = make_hist(900, seed=4)
        for s in SYMS:
            for i in range(cut, 900):
                r = h2.ohlcv[s][i]
                h2.ohlcv[s][i] = tuple(x * 7.3 for x in r[:4]) + (r[4],)
        bt = bt_cfg()
        a = bk.run_backtest(h1, bt, end=cut).metrics
        b = bk.run_backtest(h2, bt, end=cut).metrics
        self.assertEqual(a, b)
        self.assertGreater(a["n_trades"] + a["ticks"], 0)

    def test_walk_forward_selection_ignores_the_test_period(self):
        """A escolha dos parâmetros não pode depender de nenhum candle do período de teste."""
        h1 = make_hist(2400, seed=5)
        h2 = make_hist(2400, seed=5)
        split = None
        for s in SYMS:
            n = h2.n
            for i in range(n // 2 + (n // 2) * 1 // 2, n):         # trecho final: usado só como teste do fold único
                r = h2.ohlcv[s][i]
                h2.ohlcv[s][i] = tuple(x * 3.1 for x in r[:4]) + (r[4],)
        bt = bt_cfg()
        w1 = bk.walk_forward(h1, bt, folds=1, trials=4, top_k=2, min_trades=1, train_frac=0.5, seed=3)
        w2 = bk.walk_forward(h2, bt, folds=1, trials=4, top_k=2, min_trades=1, train_frac=0.5, seed=3)
        self.assertEqual(w1["folds"][0]["chosen_train"], w2["folds"][0]["chosen_train"])
        self.assertEqual(w1["folds"][0]["chosen_params"], w2["folds"][0]["chosen_params"])


class EngineTests(unittest.TestCase):
    def test_deterministic(self):
        h = make_hist(1200, seed=2)
        a = bk.run_backtest(h, bt_cfg()).metrics
        b = bk.run_backtest(h, bt_cfg()).metrics
        self.assertEqual(a, b)

    def test_costs_reduce_the_result(self):
        h = make_hist(2500, seed=3, vol=0.004)
        base = bk.run_backtest(h, bt_cfg(params=dict(StrategyParams(buy_threshold=0.1).to_dict()))).metrics
        costly = bk.run_backtest(h, bt_cfg(fee_rate=0.01, slippage=0.005,
                                           params=dict(StrategyParams(buy_threshold=0.1).to_dict()))).metrics
        self.assertGreater(base["n_trades"], 5)
        self.assertLess(costly["return_pct"], base["return_pct"])
        free = bk.run_backtest(h, bt_cfg(fee_rate=0.0, slippage=0.0,
                                         params=dict(StrategyParams(buy_threshold=0.1).to_dict()))).metrics
        self.assertGreater(free["return_pct"], base["return_pct"])
        self.assertGreater(base["fees_cents"], 0)
        self.assertEqual(free["fees_cents"], 0)

    def test_fuse_trips_on_a_crash_and_the_loss_is_capped_near_the_floor(self):
        h = crash_hist()
        p = StrategyParams(**dict(BUYABLE)).to_dict()
        r = bk.run_backtest(h, bt_cfg(params=p))
        m = r.metrics
        self.assertTrue(m["fuse"])
        self.assertEqual(r.stop_reason, "fuse")
        self.assertLess(m["final_cents"], 4000 * 0.45)           # perdeu muito...
        self.assertGreater(m["final_cents"], 0)                  # ...mas liquidou e sobrou caixa
        self.assertLess(m["ticks"], m["steps"])                  # o replay PAROU quando o fusível disparou

    def test_crash_below_the_exchange_minimum_still_stops_the_replay(self):
        """Limite conhecido: com -90% a posição vale menos que a ordem mínima e a venda por API é impossível."""
        r = bk.run_backtest(crash_hist(drop=0.10), bt_cfg(params=StrategyParams(**dict(BUYABLE)).to_dict()))
        self.assertTrue(r.metrics["fuse"])
        self.assertLess(r.metrics["final_cents"], 4000 * 0.45)
        self.assertLess(r.metrics["ticks"], r.metrics["steps"])

    def test_without_trades_equity_is_flat_and_exposure_zero(self):
        h = align({s: rows_from_closes([100.0] * 600) for s in SYMS}, TF)
        m = bk.run_backtest(h, bt_cfg()).metrics
        self.assertEqual(m["n_trades"], 0)
        self.assertEqual(m["final_cents"], 4000)
        self.assertEqual(m["return_pct"], 0.0)
        self.assertEqual(m["exposure_pct"], 0.0)
        self.assertIsNone(m["expectancy_ci95_cents"])

    def test_trades_are_stamped_and_attributed_to_assets(self):
        h = make_hist(2500, seed=3, vol=0.004)
        r = bk.run_backtest(h, bt_cfg(params=StrategyParams(buy_threshold=0.1).to_dict()))
        self.assertTrue(r.trades)
        self.assertTrue(all(t["ts"] in h.timeline and t["symbol"] in SYMS for t in r.trades))
        self.assertEqual(sum(v["n"] for v in r.metrics["by_asset"].values()), r.metrics["n_trades"])
        self.assertEqual(sum(v["pnl_cents"] for v in r.metrics["by_asset"].values()),
                         sum(t["pnl_cents"] for t in r.trades))

    def test_full_system_mode_runs_darwinism(self):
        h = make_hist(2500, seed=3, vol=0.004)
        m = bk.run_backtest(h, bt_cfg(evolve=True, eval_min_roundtrips=2, bankroll_cents=12000,
                                      agents=1)).metrics
        self.assertGreaterEqual(m["agents_dead"] + m["agents_alive"], 1)

    def test_hourly_order_limit_runs_on_simulated_time_not_wall_clock(self):
        h = make_hist(4000, seed=3, vol=0.004)
        p = StrategyParams(buy_threshold=0.1).to_dict()
        m = bk.run_backtest(h, bt_cfg(max_orders_per_hour=1, params=p)).metrics
        self.assertGreater(m["n_trades"], 8)      # 1 compra por hora SIMULADA: dias de histórico dão muitas compras

    def test_fixed_strategy_never_dies_but_full_system_does(self):
        h = make_hist(4000, seed=9, drift=-0.0003, vol=0.004)
        p = StrategyParams(buy_threshold=0.1).to_dict()
        fixed = bk.run_backtest(h, bt_cfg(params=p, eval_min_roundtrips=1, evolve=False, bankroll_cents=40000)).metrics
        full = bk.run_backtest(h, bt_cfg(params=p, eval_min_roundtrips=1, evolve=True, bankroll_cents=40000)).metrics
        self.assertEqual(fixed["agents_dead"], 0)
        self.assertGreaterEqual(full["agents_dead"], 1)

    def test_invalid_inputs(self):
        h = make_hist(300)
        with self.assertRaises(bk.BacktestError):
            bk.run_backtest(h, bt_cfg(symbols=("ZZZ/BRL",)))
        with self.assertRaises(bk.BacktestError):
            bk.run_backtest(h, bt_cfg(), start=295, end=300)

    def test_unlisted_asset_at_start_is_ignored_until_it_exists(self):
        raw = {"BTC/BRL": rows_from_closes(walk(1, 700)), "ETH/BRL": rows_from_closes(walk(2, 300), first=400)}
        h = align(raw, TF)
        r = bk.run_backtest(h, bt_cfg(symbols=("BTC/BRL", "ETH/BRL"), params=StrategyParams(buy_threshold=0.1).to_dict()))
        self.assertFalse(r.metrics["fuse"])                       # falta de dado de um ativo não vira fusível
        eth_ts = h.timeline[400]
        self.assertTrue(all(t["ts"] >= eth_ts for t in r.trades if t["symbol"] == "ETH/BRL"))


class StatsTests(unittest.TestCase):
    def test_max_drawdown(self):
        self.assertAlmostEqual(bk.max_drawdown_pct([100, 120, 60, 90]), 50.0)
        self.assertEqual(bk.max_drawdown_pct([1, 2, 3]), 0.0)

    def test_sharpe(self):
        self.assertIsNone(bk.sharpe([100, 100, 100, 100], TF))
        self.assertGreater(bk.sharpe([100, 101, 102.5, 103, 105], TF), 0)
        self.assertLess(bk.sharpe([100, 99, 97.5, 97, 95], TF), 0)

    def test_bootstrap_ci(self):
        self.assertIsNone(bk.bootstrap_ci([1, 2, 3]))
        vals = [float(x) for x in range(-50, 51)]
        lo, hi = bk.bootstrap_ci(vals)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)                                  # média 0: o intervalo inclui o zero
        self.assertEqual(bk.bootstrap_ci(vals, seed=2), bk.bootstrap_ci(vals, seed=2))
        lo, hi = bk.bootstrap_ci([10.0 + (i % 3) for i in range(60)])
        self.assertGreater(lo, 0)                                  # ganho consistente: intervalo acima de zero

    def test_trade_stats(self):
        t = [{"symbol": "A", "pnl_cents": 300, "why": "take_profit"}, {"symbol": "A", "pnl_cents": -100, "why": "stop_loss"},
             {"symbol": "B", "pnl_cents": -100, "why": "stop_loss"}, {"symbol": "B", "pnl_cents": 100, "why": "take_profit"}]
        s = bk.trade_stats(t)
        self.assertEqual((s["n_trades"], s["n_wins"], s["n_losses"]), (4, 2, 2))
        self.assertAlmostEqual(s["profit_factor"], 2.0)
        self.assertEqual(s["by_asset"]["A"]["pnl_cents"], 200)
        self.assertEqual(s["by_reason"], {"take_profit": 2, "stop_loss": 2})
        self.assertIsNone(bk.trade_stats([])["profit_factor"])

    def test_buy_and_hold(self):
        up = rows_from_closes([100.0, 200.0, 200.0])
        flat_ = rows_from_closes([100.0, 100.0, 100.0])
        late = rows_from_closes([50.0, 50.0], first=1)
        h = align({"BTC/BRL": up, "ETH/BRL": flat_, "L/BRL": late}, TF)
        free = bk.BacktestConfig(fee_rate=0.0, slippage=0.0)
        r = bk.buy_and_hold(h, ["BTC/BRL", "ETH/BRL", "L/BRL"], 0, 3, free)
        self.assertEqual(r["assets"], 2)                           # o par listado depois não entra
        self.assertAlmostEqual(r["equal_weight_return_pct"], 50.0)
        self.assertAlmostEqual(r["btc_return_pct"], 100.0)
        self.assertLess(bk.buy_and_hold(h, ["BTC/BRL"], 0, 3, bk.BacktestConfig())["equal_weight_return_pct"], 100.0)


class WalkForwardTests(unittest.TestCase):
    def test_sample_params_are_valid_and_inside_bounds(self):
        rng = random.Random(1)
        for _ in range(300):
            p = bk.sample_params(rng)
            for k, (lo, hi) in BOUNDS.items():
                self.assertTrue(lo <= getattr(p, k) <= hi, k)
            self.assertLess(p.fast, p.slow)
            self.assertEqual(p.sentiment_weight, StrategyParams().sentiment_weight)

    def test_score_disqualifies_few_trades(self):
        ok = {"n_trades": 12, "return_pct": 5.0, "max_drawdown_pct": 4.0}
        few = {"n_trades": 3, "return_pct": 99.0, "max_drawdown_pct": 0.0}
        self.assertAlmostEqual(bk.score(ok, 10), 3.0)
        self.assertLess(bk.score(few, 10), -1e8)

    def test_fold_layout_is_expanding_and_never_overlaps(self):
        h = make_hist(5000, seed=6)
        wf = bk.walk_forward(h, bt_cfg(), folds=3, trials=1, top_k=1, min_trades=1, seed=1)
        prev_end = None
        for f in wf["folds"]:
            self.assertEqual(f["train"][0], wf["folds"][0]["train"][0])      # treino começa sempre no início
            self.assertEqual(f["train"][1], f["test"][0])                    # teste começa onde o treino termina
            self.assertLess(f["test"][0], f["test"][1])
            if prev_end is not None:
                self.assertEqual(f["test"][0], prev_end)                     # blocos de teste encostados, sem sobreposição
            prev_end = f["test"][1]
        self.assertEqual(prev_end, h.n)

    def test_choice_is_the_best_train_score_and_test_is_scored_afterwards(self):
        h = make_hist(2500, seed=7)
        calls = []

        def fake(hist, bt, specs, workers=1, hist_dir=None, progress=None):
            calls.append(specs)
            out = []
            for params, a, b in specs:
                train = len(calls) % 2 == 1
                fast = params["fast"]
                out.append({"n_trades": 20, "return_pct": float(fast) if train else -float(fast), "max_drawdown_pct": 0.0,
                            "win_rate_pct": 50.0, "expectancy_cents": 1.0, "profit_factor": 1.0, "fees_cents": 0,
                            "fuse": False, "exposure_pct": 10.0, "expectancy_ci95_cents": None})
            return out
        orig = bk.evaluate_many
        bk.evaluate_many = fake
        try:
            wf = bk.walk_forward(h, bt_cfg(), folds=1, trials=6, top_k=2, min_trades=1, seed=1)
        finally:
            bk.evaluate_many = orig
        f = wf["folds"][0]
        train_specs, test_specs = calls[0], calls[1]
        self.assertTrue(all(b == f["train"][1] for _, _, b in train_specs))   # treino termina antes do teste
        self.assertTrue(all(a == f["train"][1] for _, a, _ in test_specs))    # teste só começa depois
        best_fast = max(p["fast"] for p, _, _ in train_specs)
        self.assertEqual(f["chosen_params"]["fast"], best_fast)               # escolheu o MELHOR no treino
        self.assertEqual(f["chosen_test"]["return_pct"], -float(best_fast))   # e o teste foi medido à parte

    def test_verdicts(self):
        base = dict(oos_compounded_pct=10.0, default_compounded_pct=-5.0, buy_hold_compounded_pct=4.0,
                    positive_folds=3, n_folds=3, avg_train_return_pct=20.0, avg_test_return_pct=3.0,
                    overfit_gap_pct=17.0, oos_trades=60, any_fuse=False)
        self.assertIn("INDÍCIO", bk.verdict(base))
        self.assertIn("INCONCLUSIVO", bk.verdict({**base, "oos_trades": 5}))
        self.assertIn("SEM VANTAGEM", bk.verdict({**base, "oos_compounded_pct": -3.0}))
        self.assertIn("SEM VANTAGEM", bk.verdict({**base, "oos_compounded_pct": 0.0}))
        self.assertIn("NÃO BATE", bk.verdict({**base, "buy_hold_compounded_pct": 30.0}))
        self.assertIn("FRACO", bk.verdict({**base, "positive_folds": 1}))
        self.assertNotIn("garant", bk.verdict(base).lower().replace("não é prova", ""))

    def test_parallel_matches_sequential(self):
        d = tempfile.mkdtemp()
        raw = {s: rows_from_closes(walk(10 + k, 900)) for k, s in enumerate(SYMS)}
        for s, r in raw.items():
            write_csv(csv_path(d, s, TF), r)
        h = load_history(d, SYMS, TF)
        bt = bt_cfg()
        specs = [(StrategyParams(buy_threshold=0.1).to_dict(), 300, 700), (StrategyParams().to_dict(), 300, 700)]
        seq = bk.evaluate_many(h, bt, specs, workers=1)
        par = bk.evaluate_many(h, bt, specs, workers=2, hist_dir=d)
        self.assertEqual(seq, par)

    def test_too_short_history_is_refused(self):
        with self.assertRaises(bk.BacktestError):
            bk.walk_forward(make_hist(600), bt_cfg(), folds=3)


class ReportTests(unittest.TestCase):
    def test_report_texts_are_honest_and_complete(self):
        h = make_hist(2500, seed=3, vol=0.004)
        bt = bt_cfg(params=StrategyParams(buy_threshold=0.1).to_dict())
        r = bk.run_backtest(h, bt)
        b = bk.buy_and_hold(h, SYMS, r.start, r.end, bt)
        st = bk.stress_test(h, bt, None, None)
        txt = bk.format_result(r, b, st)
        for piece in ("BACKTEST", "comprar e segurar", "trades:", "estresse", "sem sentimento", "pior queda"):
            self.assertIn(piece, txt)
        wf = bk.walk_forward(make_hist(3000, seed=8), bt_cfg(), folds=1, trials=3, top_k=2, min_trades=1, seed=1)
        t2 = bk.format_walk_forward(wf)
        for piece in ("WALK-FORWARD", "TESTE", "RESUMO FORA DA AMOSTRA", "CONCLUSÃO", "sobreajuste"):
            self.assertIn(piece, t2)

    def test_outputs_are_written(self):
        h = make_hist(2500, seed=3, vol=0.004)
        r = bk.run_backtest(h, bt_cfg(params=StrategyParams(buy_threshold=0.1).to_dict()))
        wf = bk.walk_forward(make_hist(3000, seed=8), bt_cfg(), folds=1, trials=2, top_k=1, min_trades=1, seed=1)
        d = tempfile.mkdtemp()
        paths = bk.save_outputs(d, res=r, wf=wf)
        self.assertEqual({os.path.basename(p) for p in paths},
                         {"trades.csv", "equity.csv", "summary.json", "walk_forward.json", "best_params.json"})
        StrategyParams.from_dict(json.load(open(os.path.join(d, "best_params.json"))))   # carregável pelo robô


class CliTests(unittest.TestCase):
    def run_cli(self, *argv, env=None, **kw):
        out = io.StringIO()
        d = tempfile.mkdtemp()
        e = {"MODE": "paper", "DATA_DIR": os.path.join(d, "data"), "BANKROLL_BRL": "40",
             "SYMBOLS": ",".join(SYMS), "TIMEFRAME": TF, **(env or {})}
        with contextlib.redirect_stdout(out):
            code = cli.main(list(argv), env=e, **kw)
        return code, out.getvalue()

    def hist_dir(self, n=2500):
        d = tempfile.mkdtemp()
        for k, s in enumerate(SYMS):
            write_csv(csv_path(d, s, TF), rows_from_closes(walk(20 + k, n, vol=0.004)))
        return d

    def test_download_command(self):
        import time
        t0 = (int(time.time() * 1000) // STEP) * STEP - 400 * STEP          # candles recentes (o comando usa o relógio real)
        FakeCcxt.series = {s: rows_from_closes(walk(1, 400), t0=t0) for s in SYMS}
        FakeCcxt.calls, FakeCcxt.fail, FakeCcxt.precision_mode = [], {}, 4
        d = tempfile.mkdtemp()
        code, out = self.run_cli("download", "--days", "5", "--dir", d, ccxt_module=FakeCcxt)
        self.assertEqual(code, 0)
        self.assertIn("3 par(es) com dados", out)
        self.assertTrue(all(os.path.exists(csv_path(d, s, TF)) for s in SYMS))

    def test_download_with_nothing_available_fails_cleanly(self):
        FakeCcxt.series = {"OUTRO/BRL": rows_from_closes(walk(1, 10))}
        FakeCcxt.calls, FakeCcxt.fail, FakeCcxt.precision_mode = [], {}, 4
        code, out = self.run_cli("download", "--days", "5", "--dir", tempfile.mkdtemp(), ccxt_module=FakeCcxt)
        self.assertEqual(code, 2)
        self.assertEqual(self.run_cli("download", "--timeframe", "7m", "--dir", tempfile.mkdtemp(), ccxt_module=FakeCcxt)[0], 2)

    def test_backtest_command_end_to_end(self):
        d, out_dir = self.hist_dir(), os.path.join(tempfile.mkdtemp(), "out")
        code, out = self.run_cli("backtest", "--dir", d, "--out", out_dir, "--stress")
        self.assertEqual(code, 0, out)
        for piece in ("sistema completo", "comprar e segurar", "estresse", "Arquivos salvos"):
            self.assertIn(piece, out)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "trades.csv")))
        code, out = self.run_cli("backtest", "--dir", d, "--out", out_dir, "--fixed", "--days", "4")
        self.assertEqual(code, 0, out)
        self.assertIn("estratégia fixa", out)

    def test_backtest_without_history_explains_what_to_do(self):
        code, out = self.run_cli("backtest", "--dir", tempfile.mkdtemp())
        self.assertEqual(code, 2)
        self.assertIn("download", out)

    def test_optimize_command_and_params_file_roundtrip(self):
        d, out_dir = self.hist_dir(3500), os.path.join(tempfile.mkdtemp(), "out")
        code, out = self.run_cli("backtest", "--dir", d, "--out", out_dir, "--optimize", "--folds", "1",
                                 "--trials", "3", "--min-trades", "1")
        self.assertEqual(code, 0, out)
        self.assertIn("CONCLUSÃO", out)
        self.assertIn("STRATEGY_PARAMS_FILE", out)
        pf = os.path.join(out_dir, "best_params.json")
        # o robô aceita o arquivo no init e o agente nasce com esses parâmetros
        from tests.helpers import MultiMarket, flat
        from darwin.sim import SimExchange
        mm = MultiMarket({s: flat(70) for s in SYMS})
        sim = SimExchange(mm, list(SYMS), quote_balance=40.0, min_notional=10.0)
        dd = tempfile.mkdtemp()
        env = {"STRATEGY_PARAMS_FILE": pf, "DATA_DIR": dd}
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            c = cli.main(["init"], env={"MODE": "paper", "BANKROLL_BRL": "40", "SYMBOLS": ",".join(SYMS), **env},
                         exchange_factory=lambda cfg: sim)
        self.assertEqual(c, 0, o.getvalue())
        self.assertIn("Parâmetros da estratégia carregados", o.getvalue())
        st = json.load(open(os.path.join(dd, "state.json")))
        self.assertEqual(st["traders"][0]["params"], json.load(open(pf)))

    def test_invalid_params_file_is_refused(self):
        from tests.helpers import MultiMarket, flat
        from darwin.sim import SimExchange
        mm = MultiMarket({s: flat(70) for s in SYMS})
        sim = SimExchange(mm, list(SYMS), quote_balance=40.0, min_notional=10.0)
        bad = os.path.join(tempfile.mkdtemp(), "p.json")
        json.dump({"fast": 99, "slow": 5}, open(bad, "w"))
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            c = cli.main(["init"], env={"MODE": "paper", "BANKROLL_BRL": "40", "SYMBOLS": ",".join(SYMS),
                                        "STRATEGY_PARAMS_FILE": bad, "DATA_DIR": tempfile.mkdtemp()},
                         exchange_factory=lambda cfg: sim)
        self.assertEqual(c, 2)
        self.assertIn("STRATEGY_PARAMS_FILE inválido", o.getvalue())


if __name__ == "__main__":
    unittest.main()
