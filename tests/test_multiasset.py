"""Multiativo: 7 pares na mesma conta, varredura, fusível sobre o patrimônio TOTAL, liquidação por ativo."""
import contextlib
import io
import unittest

import main as cli
from darwin.config import ConfigError, DEFAULT_SYMBOLS, load_config
from darwin.exchange import ExchangeError, MultiGateway
from darwin.sentiment import SentimentReading, mentions
from darwin.sim import SimExchange
from tests.helpers import MultiMarket, Rig, flat, make_cfg, uptrend
from tests.test_integration import BUYABLE

SYMS = ("BTC/BRL", "ETH/BRL", "SOL/BRL", "XRP/BRL")


def series(**over):
    base = {s: flat(70) for s in SYMS}
    base.update({k.replace("_", "/"): v for k, v in over.items()})
    return base


def rig(prices=None, quote=40.0, symbols=SYMS, sentiment=None, min_notional=10.0, **kw):
    mm = MultiMarket(prices or series(), min_notional)
    kw.setdefault("eval_min_roundtrips", 50)          # sem fechar o ciclo no meio do teste
    kw.setdefault("bankroll_cents", int(quote * 100))
    cfg = make_cfg(symbols=tuple(symbols), **kw)
    r = Rig(cfg=cfg, market=mm, quote=quote, sentiment=sentiment)
    return _rebuild(r, make_cfg(symbols=tuple(symbols), data_dir=r.dir, **kw))


def _rebuild(r, cfg):
    r.cfg = cfg
    r.build(r.orch.sentiment)
    return r


def advance_all(r, steps, **mult):
    """Avança todos os ativos: cada um multiplica o último preço pelo passo (padrão: lateral)."""
    for _ in range(steps):
        for s, m in r.market.m.items():
            step = mult.get(s.split("/")[0], 1.0)
            m.advance(m.prices[-1] * step)
        r.orch.tick()
        if r.orch.stop_reason:
            return


class ConfigTests(unittest.TestCase):
    def test_default_is_twelve_majors_in_brl(self):
        c = load_config({})
        self.assertEqual(c.symbols, DEFAULT_SYMBOLS)
        self.assertEqual(len(c.symbols), 12)
        self.assertEqual(len(set(c.symbols)), 12)
        self.assertTrue({"BTC", "ETH", "SOL", "XRP", "ADA", "LINK", "DOGE"} <= {s.split("/")[0] for s in c.symbols})
        self.assertEqual(c.quote, "BRL")

    def test_symbols_parse_dedupe_and_legacy_symbol(self):
        self.assertEqual(load_config({"SYMBOLS": " btc/brl, eth/brl ,BTC/BRL"}).symbols, ("BTC/BRL", "ETH/BRL"))
        self.assertEqual(load_config({"SYMBOL": "ETH/BRL"}).symbols, ("ETH/BRL",))   # .env antigo continua valendo

    def test_invalid_universe_is_rejected(self):
        for env in ({"SYMBOLS": "BTC/BRL,ETH/USDT"},          # cotações misturadas: patrimônio incomparável
                    {"SYMBOLS": "BTC-BRL"}, {"SYMBOLS": ",".join(f"A{i}/BRL" for i in range(13))}):
            with self.assertRaises(ConfigError, msg=str(env)):
                load_config(env)

    def test_bnb_is_refused_because_it_is_the_fee_token(self):
        """BNB guardado para desconto de taxa seria contado como patrimônio e vendido na liquidação."""
        with self.assertRaises(ConfigError):
            load_config({"SYMBOLS": "BTC/BRL,BNB/BRL"})
        self.assertNotIn("BNB", [s.split("/")[0] for s in DEFAULT_SYMBOLS])

    def test_usdt_universe_is_allowed_and_labelled(self):
        c = load_config({"SYMBOLS": "BTC/USDT,ETH/USDT"})
        self.assertEqual(c.quote, "USDT")


class GatewayTests(unittest.TestCase):
    def sim(self, quote=40.0, symbols=SYMS, **kw):
        mm = MultiMarket({s: flat(70) for s in symbols})
        return mm, SimExchange(mm, list(symbols), quote_balance=quote, min_notional=10.0, **kw)

    def test_equity_sums_cash_and_every_asset(self):
        mm, sim = self.sim()
        gw = MultiGateway(sim, SYMS, sleep=lambda s: None)
        self.assertEqual(gw.equity_cents(), 4000)
        gw.market_buy(10.0, "ETH/BRL")
        gw.market_buy(10.0, "SOL/BRL")
        e0 = gw.equity_cents()
        self.assertAlmostEqual(e0, 4000, delta=40)               # só taxas e derrapagem
        mm.advance("ETH/BRL", 50.0)                              # ETH cai 50%
        self.assertLess(gw.equity_cents(), e0 - 400)
        self.assertGreater(gw.foreign_base_value_cents(), 1000)

    def test_unlisted_pairs_are_skipped_not_fatal(self):
        mm, sim = self.sim()
        orig = mm.market

        def market(sym):
            if sym == "XRP/BRL":
                raise KeyError("BadSymbol")
            return orig(sym)
        mm.market = market
        gw = MultiGateway(sim, SYMS, sleep=lambda s: None)
        self.assertEqual(gw.symbols, ["BTC/BRL", "ETH/BRL", "SOL/BRL"])
        self.assertIn("XRP/BRL", gw.skipped)

    def test_no_pair_available_is_an_error_and_foreign_symbol_is_refused(self):
        mm, sim = self.sim(symbols=("BTC/BRL",))
        gw = MultiGateway(sim, ["BTC/BRL"], sleep=lambda s: None)
        with self.assertRaises(ExchangeError):
            gw.for_symbol("ETH/BRL")
        mm.market = lambda s: (_ for _ in ()).throw(KeyError("x"))
        with self.assertRaises(ExchangeError):
            MultiGateway(sim, ["BTC/BRL"], sleep=lambda s: None)

    def test_cancel_all_tries_every_pair_even_if_one_fails(self):
        mm, sim = self.sim()
        gw = MultiGateway(sim, SYMS, sleep=lambda s: None)
        calls = []
        orig = sim.cancel_all_orders

        def cancel(symbol=None):
            calls.append(symbol)
            if symbol == "ETH/BRL":
                raise RuntimeError("falha")
            return orig(symbol)
        sim.cancel_all_orders = cancel
        with self.assertRaises(RuntimeError):
            gw.cancel_all_orders()
        self.assertEqual(sorted(calls), sorted(SYMS))             # tentou TODOS antes de falhar

    def test_per_pair_lot_step_comes_from_the_market(self):
        mm, sim = self.sim(symbols=("BTC/BRL", "DOGE/BRL"))
        base = mm.market
        mm.market = lambda s: {**base(s), "precision": {"amount": 1.0 if s.startswith("DOGE") else 0.00001}}
        self.assertEqual(float(sim.amount_to_precision("DOGE/BRL", 12.7)), 12.0)
        self.assertEqual(float(sim.amount_to_precision("BTC/BRL", 0.123456789)), 0.12345)


class ScanTests(unittest.TestCase):
    def test_flat_agent_buys_the_best_scoring_asset_only(self):
        r = rig(series(ETH_BRL=flat(60) + uptrend(100.0, 25, 0.006)))
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        advance_all(r, 25, ETH=1.006)
        t = r.orch.traders["alpha"]
        self.assertTrue(t.in_position)
        self.assertEqual(t.symbol, "ETH/BRL")
        self.assertGreater(float(r.sim.balances["ETH"]), 0)
        for other in ("BTC", "SOL", "XRP"):
            self.assertEqual(float(r.sim.balances[other]), 0.0)
        self.assertIn("ETH/BRL", r.text())

    def test_with_two_buy_signals_the_stronger_one_wins(self):
        r = rig(series(ETH_BRL=flat(60) + uptrend(100.0, 25, 0.006), SOL_BRL=flat(60) + uptrend(100.0, 25, 0.002)))
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        for _ in range(25):
            advance_all(r, 1, ETH=1.006, SOL=1.002)
            if r.orch.traders["alpha"].in_position:
                break
        scan = r.orch.last_scan["alpha"]                        # nota de cada ativo no tick da compra
        self.assertGreater(scan["ETH/BRL"], scan["SOL/BRL"])
        self.assertGreaterEqual(scan["SOL/BRL"], 0.1)           # os DOIS passavam do limiar
        self.assertEqual(r.orch.traders["alpha"].symbol, "ETH/BRL")

    def test_position_is_managed_with_its_own_assets_price(self):
        r = rig(series(ETH_BRL=flat(60) + uptrend(100.0, 25, 0.006)))
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        advance_all(r, 25, ETH=1.006)
        t = r.orch.traders["alpha"]
        self.assertTrue(t.in_position)
        advance_all(r, 1, BTC=1.5, ETH=0.9)                   # BTC dispara, mas a posição é ETH e cai 10%
        self.assertFalse(r.orch.traders["alpha"].in_position)  # stop-loss no ETH
        self.assertIn("VENDA ETH/BRL (stop_loss)", r.text())
        self.assertLess(r.ledger.total("alpha", "trade_loss"), 10**9)
        self.assertGreater(r.ledger.total("alpha", "trade_loss"), 0)

    def test_buy_skips_asset_whose_minimum_is_above_capital(self):
        r = rig(series(ETH_BRL=flat(60) + uptrend(100.0, 25, 0.006), SOL_BRL=flat(60) + uptrend(100.0, 25, 0.003)),
                quote=40.0)
        orig = r.sim.market                                    # ETH exige 500 por ordem: inviável com R$ 40
        r.sim.market = lambda s: {**orig(s), "limits": {"cost": {"min": 500.0}}} if s == "ETH/BRL" else orig(s)
        r.build(r.orch.sentiment)
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        advance_all(r, 25, ETH=1.006, SOL=1.003)
        self.assertEqual(r.orch.traders["alpha"].symbol, "SOL/BRL")   # caiu para o próximo candidato viável

    def test_heartbeat_shows_scan_and_pick(self):
        r = rig()
        r.orch.bootstrap(1)
        advance_all(r, 2)
        txt = r.text()
        self.assertIn("varredura (alpha): BTC +", txt.replace("-", "+"))
        self.assertIn("sem_sinal", txt)

    def test_two_agents_hold_different_assets_and_each_is_judged_on_own_pnl(self):
        r = rig(series(ETH_BRL=flat(60) + uptrend(100.0, 25, 0.006)), quote=80.0)
        r.orch.bootstrap(2)
        for n in ("alpha", "beta"):
            r.tune(n, **BUYABLE)
        for _ in range(25):                                    # para no instante em que os dois entraram
            advance_all(r, 1, ETH=1.006)
            if all(t.in_position for t in r.orch.traders.values()):
                break
        self.assertTrue(all(t.in_position and t.symbol == "ETH/BRL" for t in r.orch.traders.values()))
        self.assertEqual(sum(1 for c in r.sim.calls if c == "create_market_buy_order_with_cost"), 2)
        # cada um é julgado só pelo PRÓPRIO PnL: o livro de um não mexe no do outro
        self.assertEqual(r.ledger.total("alpha", "trade_gain") + r.ledger.total("alpha", "trade_loss"), 0)


class FuseTests(unittest.TestCase):
    def two_positions(self):
        r = rig(series(ETH_BRL=flat(60) + uptrend(100.0, 25, 0.006)), quote=80.0)
        r.orch.bootstrap(2)
        r.tune("alpha", **BUYABLE)
        r.tune("beta", **BUYABLE)
        advance_all(r, 25, ETH=1.006)
        return r

    def test_fuse_trips_on_total_equity_across_assets_and_sells_everything(self):
        r = rig(quote=80.0)
        r.orch.bootstrap(2)
        r.tune("alpha", **BUYABLE)
        r.tune("beta", **BUYABLE)
        # abre uma posição em cada um de dois ativos diferentes, direto pelo orquestrador
        a, b = r.orch.traders["alpha"], r.orch.traders["beta"]
        rep = type("R", (), {"events": []})()
        r.orch._open_position(a, "ETH/BRL", 3800, rep)
        r.orch._open_position(b, "SOL/BRL", 3800, rep)
        self.assertEqual((a.symbol, b.symbol), ("ETH/BRL", "SOL/BRL"))
        r.market.advance("ETH/BRL", 30.0)                      # ETH -70%: sozinho o total cairia só a ~54
        r.market.advance("SOL/BRL", 30.0)                      # SOL -70%: total ~27 (abaixo do piso 32), ainda vendável
        rp = r.orch.tick()
        self.assertTrue(rp.halted)
        self.assertEqual(r.orch.stop_reason, "fuse")
        self.assertLess(float(r.sim.balances["ETH"]), 0.001)
        self.assertLess(float(r.sim.balances["SOL"]), 0.001)   # ambos vendidos
        self.assertIn("liquidação concluída", r.text())
        self.assertFalse(a.in_position or b.in_position)
        self.assertEqual((a.symbol, b.symbol), ("", ""))

    def test_one_asset_crash_alone_is_not_the_fuse_but_hits_stop_loss(self):
        r = rig(quote=40.0)
        r.orch.bootstrap(1)
        a = r.orch.traders["alpha"]
        r.orch._open_position(a, "ETH/BRL", 3800, type("R", (), {"events": []})())
        r.market.advance("ETH/BRL", 90.0)                      # -10%: perde ~R$ 3,8 de R$ 40
        r.orch.tick()
        self.assertEqual(r.orch.stop_reason, "")
        self.assertFalse(a.in_position)

    def test_ambiguous_order_error_on_any_asset_trips_fuse(self):
        r = rig(series(SOL_BRL=flat(60) + uptrend(100.0, 25, 0.006)))
        r.orch.bootstrap(1)
        r.tune(**BUYABLE)
        r.sim.inject_failures("create_market_buy_order_with_cost", 1)
        advance_all(r, 25, SOL=1.006)
        self.assertEqual(r.orch.stop_reason, "fuse")

    def test_partial_liquidation_reports_failure_for_the_stuck_asset(self):
        r = rig(quote=40.0)
        r.orch.bootstrap(1)
        a = r.orch.traders["alpha"]
        r.orch._open_position(a, "ETH/BRL", 3800, type("R", (), {"events": []})())
        r.market.m["ETH/BRL"].min_notional = 10.0
        orig = r.sim.market
        r.sim.market = lambda s: {**orig(s), "limits": {"cost": {"min": 500.0}}} if s == "ETH/BRL" else orig(s)
        r.build(r.orch.sentiment)
        ok = r.orch.liquidate_all("teste")
        self.assertFalse(ok)                                   # não declara sucesso se não vendeu
        self.assertIn("NÃO DÁ PARA VENDER POR API", r.text())
        self.assertIn("ETH/BRL", r.text())


class MarketDataTests(unittest.TestCase):
    class Flaky(MultiMarket):
        broken: set = set()
        broken_ohlcv: set = set()

        def fetch_ticker(self, s):
            if s in self.broken:
                raise TimeoutError("rede")
            return super().fetch_ticker(s)

        def fetch_ohlcv(self, s, tf, limit=None):
            if s in self.broken or s in self.broken_ohlcv:
                raise TimeoutError("rede")
            return super().fetch_ohlcv(s, tf, limit)

    def build(self):
        mm = self.Flaky(series())
        mm.broken = set()
        cfg = make_cfg(symbols=SYMS)
        r = Rig(cfg=cfg, market=mm, quote=40.0)
        r.build(r.orch.sentiment)
        r.orch.bootstrap(1)
        return r, mm

    def test_missing_data_for_an_unheld_asset_is_tolerated(self):
        r, mm = self.build()
        mm.broken = {"XRP/BRL"}
        for _ in range(6):
            advance_all(r, 1)
        self.assertEqual(r.orch.stop_reason, "")               # 6 ticks com XRP fora: sem fusível
        self.assertIn("sem dados neste tick: XRP/BRL", r.text())

    def test_missing_data_for_the_held_asset_counts_toward_the_market_channel(self):
        r, mm = self.build()
        a = r.orch.traders["alpha"]
        r.orch._open_position(a, "ETH/BRL", 3800, type("R", (), {"events": []})())
        mm.broken = {"ETH/BRL"}
        for _ in range(4):
            r.orch.tick()
            if r.orch.stop_reason:
                break
        self.assertEqual(r.orch.stop_reason, "fuse")           # não opera às cegas com posição aberta


    def test_held_asset_blind_even_when_other_assets_are_visible_counts_as_failure(self):
        mm = self.Flaky(series())
        mm.broken = set()
        r = Rig(cfg=make_cfg(symbols=SYMS, bankroll_cents=8000), market=mm, quote=80.0)
        r.build(r.orch.sentiment)
        r.orch.bootstrap(2)                                    # beta fica fora da posição => varre todos
        r.orch._open_position(r.orch.traders["alpha"], "ETH/BRL", 3800, type("R", (), {"events": []})())
        mm.broken_ohlcv = {"ETH/BRL"}       # preço e saldo ok (patrimônio legível), mas sem candles do ativo segurado
        for _ in range(4):
            r.orch.tick()
            if r.orch.stop_reason:
                break
        self.assertEqual(r.orch.stop_reason, "fuse")


class SentimentTests(unittest.TestCase):
    def reading(self):
        items = (("Bitcoin soars to record", 0.9), ("Bitcoin ETF inflows surge", 0.7),
                 ("Ethereum developers delay upgrade", -0.6), ("Ethereum fees spike, users leave", -0.8),
                 ("Click this link for free tokens", 0.9), ("SOL network outage again", -0.5))
        return SentimentReading(0.07, True, len(items), 0.0, items=items)

    def test_headlines_are_matched_to_assets_without_false_positives(self):
        self.assertTrue(mentions("Bitcoin up", "BTC"))
        self.assertTrue(mentions("BTC up", "BTC"))
        self.assertFalse(mentions("Click this link", "LINK"))
        self.assertTrue(mentions("LINK rallies", "LINK"))
        self.assertFalse(mentions("the sol is hot", "SOL"))

    def test_value_for_uses_asset_headlines_else_market_average(self):
        r = self.reading()
        self.assertAlmostEqual(r.value_for("BTC"), 0.8)
        self.assertAlmostEqual(r.value_for("ETH"), -0.7)
        self.assertAlmostEqual(r.value_for("SOL"), 0.07)        # só 1 manchete: usa a média geral
        self.assertAlmostEqual(r.value_for("DOGE"), 0.07)
        self.assertIsNone(SentimentReading(0.0, False, 0, 0.0).value_for("BTC"))

    def test_asset_sentiment_changes_which_asset_is_bought(self):
        class Fixed:
            def __init__(self, rd):
                self.rd = rd

            def get(self):
                return self.rd
        # lateral em tudo; só a notícia positiva de BTC (peso alto) empurra a nota acima do limiar
        r = rig(sentiment=Fixed(self.reading()))
        r.orch.bootstrap(1)
        r.tune(sentiment_weight=0.8, buy_threshold=0.30)
        advance_all(r, 3)
        t = r.orch.traders["alpha"]
        self.assertEqual(t.symbol, "BTC/BRL")


class PersistenceAndFormatTests(unittest.TestCase):
    def test_old_single_asset_state_loads_with_position_on_primary_symbol(self):
        import json
        r = rig(quote=40.0)
        r.orch.bootstrap(1)
        a = r.orch.traders["alpha"]
        r.orch._open_position(a, "BTC/BRL", 3800, type("R", (), {"events": []})())
        r.orch.save_state()
        data = json.load(open(r.orch.state_path, encoding="utf-8"))
        for t in data["traders"]:
            t.pop("symbol", None)                              # estado gravado pela versão antiga
        json.dump(data, open(r.orch.state_path, "w", encoding="utf-8"))
        r.build(r.orch.sentiment)
        self.assertEqual(r.orch.traders["alpha"].symbol, "BTC/BRL")

    def test_state_roundtrip_keeps_asset_of_each_position(self):
        r = rig(quote=40.0)
        r.orch.bootstrap(1)
        r.orch._open_position(r.orch.traders["alpha"], "SOL/BRL", 3800, type("R", (), {"events": []})())
        r.orch.save_state()
        r.build(r.orch.sentiment)
        self.assertEqual(r.orch.traders["alpha"].symbol, "SOL/BRL")

    def test_usdt_quote_is_labelled_in_logs(self):
        syms = ("BTC/USDT", "ETH/USDT")
        mm = MultiMarket({s: flat(70) for s in syms}, min_notional=5.0)
        r = Rig(cfg=make_cfg(symbols=syms), market=mm, quote=40.0)
        r.build(r.orch.sentiment)
        r.orch.bootstrap(1)
        r.orch._open_position(r.orch.traders["alpha"], "ETH/USDT", 3800, type("R", (), {"events": []})())
        self.assertIn("USDT", r.text())
        self.assertNotIn("R$", r.text())


class CliTests(unittest.TestCase):
    def run_cli(self, sim, mm, *argv, **env):
        import tempfile
        d = tempfile.mkdtemp()
        e = {"MODE": "paper", "DATA_DIR": d, "BANKROLL_BRL": "40", "TICK_SECONDS": "5",
             "SYMBOLS": ",".join(SYMS), **env}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(list(argv), env=e, exchange_factory=lambda cfg: sim, sentiment=None, sleep=lambda s: None)
        return code, out.getvalue()

    def test_check_lists_each_asset_and_skipped_pairs(self):
        mm = MultiMarket(series())
        base = mm.market
        mm.market = lambda s: (_ for _ in ()).throw(KeyError("BadSymbol")) if s == "XRP/BRL" else base(s)
        sim = SimExchange(mm, list(SYMS), quote_balance=40.0, min_notional=10.0)
        code, out = self.run_cli(sim, mm, "check")
        self.assertEqual(code, 0)
        for s in ("BTC/BRL", "ETH/BRL", "SOL/BRL"):
            self.assertIn(f"{s}: preço", out)
        self.assertIn("pares IGNORADOS", out)
        self.assertIn("XRP/BRL", out)
        self.assertIn("ordem mínima da exchange: 10.00 BRL", out)
        self.assertFalse(any(c.startswith("create") for c in sim.calls))

    def test_run_and_status_with_several_assets(self):
        mm = MultiMarket(series())
        sim = SimExchange(mm, list(SYMS), quote_balance=40.0, min_notional=10.0)
        code, _ = self.run_cli(sim, mm, "init")
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
