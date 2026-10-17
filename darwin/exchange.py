"""Gateway para a Binance via ccxt (spot, ordens a mercado).

NÃO existe aqui nenhum método de saque, transferência, margem ou alavancagem.

Regra de conta: o par operado deve estar "limpo" (sem saldo prévio do ativo base e com
saldo em moeda de cotação próximo da banca). Assim o patrimônio lido da API
(cotação + base * preço) é exatamente o patrimônio do robô, e a liquidação vende
apenas o que o robô comprou.

Ordens NUNCA são reenviadas automaticamente: se `create_order` falhar com erro ambíguo
(timeout, rede), reenviar poderia duplicar a compra. O erro sobe com ambiguous=True.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Callable

from .config import Config
from .sim import SimExchange


class ExchangeError(Exception):
    def __init__(self, msg: str, ambiguous: bool = False, below_min: bool = False):
        super().__init__(msg)
        self.ambiguous = ambiguous
        self.below_min = below_min  # abaixo do mínimo/precisão da exchange: a ordem é impossível


_REJECTED_NAMES = {"InsufficientFunds", "InvalidOrder", "BadSymbol", "BadRequest", "PermissionDenied",
                   "AuthenticationError", "AccountSuspended", "AccountNotEnabled", "RateLimitExceeded",
                   "DDoSProtection", "InvalidNonce", "OrderNotFillable", "NotSupported", "ArgumentsRequired"}
_REJECTED_MARKERS = ("InsufficientFunds", "MIN_NOTIONAL", "insuficiente", "abaixo do valor mínimo")


def is_ambiguous_failure(e: Exception) -> bool:
    """True quando NÃO dá para saber se a ordem foi aceita (timeout/rede/erro genérico).
    Rejeições definitivas (saldo, mínimo, parâmetros, auth) não são ambíguas."""
    if type(e).__name__ in _REJECTED_NAMES:
        return False
    return not any(m in str(e) for m in _REJECTED_MARKERS)


def to_cents(x: float | Decimal) -> int:
    return int((Decimal(str(x)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def _num(x: Any, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ExchangeError(f"{what} inválido") from None
    if not math.isfinite(v):
        raise ExchangeError(f"{what} inválido")
    return v


@dataclass(frozen=True)
class Fill:
    side: str
    order_id: str
    price: float
    base_qty: float        # quantidade BRUTA executada (compra: recebida antes da taxa; venda: vendida)
    quote_value: float     # valor bruto em moeda de cotação
    fee_base: float        # taxa paga na moeda base (reduz a quantidade líquida da compra)
    fee_cash_quote: float  # taxa paga em cotação ou convertida (ex.: BNB) para cotação
    fee_estimated: bool = False

    @property
    def net_base_qty(self) -> float:
        return self.base_qty - self.fee_base

    @property
    def buy_cost(self) -> float:
        """Custo total da compra em moeda de cotação (inclui taxa paga em cotação/BNB)."""
        return self.quote_value + self.fee_cash_quote

    @property
    def sell_proceeds(self) -> float:
        """Receita líquida da venda em moeda de cotação (já sem taxas)."""
        return self.quote_value - self.fee_cash_quote - self.fee_base * self.price


class CcxtGateway:
    def __init__(self, ex: Any, symbol: str, assumed_fee_rate: float = 0.001,
                 sleep: Callable[[float], None] = time.sleep):
        self.ex = ex
        self.symbol = symbol
        self.assumed_fee_rate = assumed_fee_rate
        self._sleep = sleep
        self.ex.load_markets()
        m = self.ex.market(symbol)
        self.base: str = m["base"]
        self.quote: str = m["quote"]
        self._market = m

    # ----------------------------------------------------------------- leitura
    def min_notional(self) -> float:
        try:
            v = self._market["limits"]["cost"]["min"]
        except (KeyError, TypeError):
            v = None
        return _num(v, "mínimo da ordem") if v is not None else 10.0  # fallback conservador

    def last_price(self) -> float:
        p = _num(self.ex.fetch_ticker(self.symbol).get("last"), "preço")
        if p <= 0:
            raise ExchangeError("preço inválido")
        return p

    def candles(self, timeframe: str, limit: int = 100) -> list[list[float]]:
        rows = self.ex.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        out = []
        for r in rows or []:
            if len(r) < 5:
                raise ExchangeError("candle inválido")
            try:
                vals = [float(v) for v in r[:6]]
            except (TypeError, ValueError):
                raise ExchangeError("candle inválido") from None
            if not all(map(math.isfinite, vals)):
                raise ExchangeError("candle inválido")
            out.append(vals)
        if not out:
            raise ExchangeError("sem candles")
        return out

    def _balances(self) -> tuple[float, float, float]:
        b = self.ex.fetch_balance()
        q = b.get(self.quote) or {}
        s = b.get(self.base) or {}
        return (_num(q.get("total", 0) or 0, "saldo"), _num(s.get("total", 0) or 0, "saldo"),
                _num(s.get("free", 0) or 0, "saldo"))

    def quote_total(self) -> float:
        return self._balances()[0]

    def base_total(self) -> float:
        return self._balances()[1]

    def base_free(self) -> float:
        return self._balances()[2]

    def equity_cents(self) -> int:
        """Patrimônio total do par na exchange (cotação + base * preço), em centavos."""
        q, base_total, _ = self._balances()
        price = self.last_price() if base_total > 0 else 0.0
        return to_cents(Decimal(str(q)) + Decimal(str(base_total)) * Decimal(str(price)))

    def open_orders_count(self) -> int:
        return len(self.ex.fetch_open_orders(self.symbol) or [])

    # ------------------------------------------------------------------- ordens
    def market_buy(self, quote_amount: float) -> Fill:
        amount = _num(quote_amount, "valor")
        if amount < self.min_notional():
            raise ExchangeError("abaixo do valor mínimo da ordem", below_min=True)
        try:
            if hasattr(self.ex, "create_market_buy_order_with_cost"):
                order = self.ex.create_market_buy_order_with_cost(self.symbol, amount)
            else:  # fallback: converte para quantidade
                qty = float(self.ex.amount_to_precision(self.symbol, amount / self.last_price()))
                order = self.ex.create_order(self.symbol, "market", "buy", qty)
        except Exception as e:
            raise ExchangeError(f"falha ao enviar compra: {type(e).__name__}", ambiguous=is_ambiguous_failure(e)) from None
        return self._to_fill(order, "buy")

    def market_sell(self, base_qty: float) -> Fill:
        qty = float(self.ex.amount_to_precision(self.symbol, _num(base_qty, "quantidade")))
        if qty <= 0:
            raise ExchangeError("quantidade abaixo da precisão do mercado", below_min=True)
        if qty * self.last_price() < self.min_notional():
            raise ExchangeError("venda abaixo do valor mínimo da ordem (poeira)", below_min=True)
        try:
            order = self.ex.create_order(self.symbol, "market", "sell", qty)
        except Exception as e:
            raise ExchangeError(f"falha ao enviar venda: {type(e).__name__}", ambiguous=is_ambiguous_failure(e)) from None
        return self._to_fill(order, "sell")

    def cancel_all_orders(self) -> int:
        canceled = self.ex.cancel_all_orders(self.symbol)
        return len(canceled) if isinstance(canceled, list) else 0

    # ----------------------------------------------------------------- parsing
    def _to_fill(self, order: dict, side: str) -> Fill:
        oid = str(order.get("id") or "")
        if not oid:
            raise ExchangeError("ordem sem id", ambiguous=True)
        for attempt in range(3):
            if order.get("status") == "closed" and order.get("filled"):
                break
            try:
                order = self.ex.fetch_order(oid, self.symbol)
            except Exception:
                pass
            if order.get("status") == "closed":
                break
            self._sleep(0.5)
        if order.get("status") not in ("closed",):
            try:
                self.ex.cancel_order(oid, self.symbol)  # cancela o restante e usa o executado
                order = self.ex.fetch_order(oid, self.symbol)
            except Exception:
                pass
        filled = _num(order.get("filled") or 0, "filled")
        cost = _num(order.get("cost") or 0, "cost")
        if filled <= 0 or cost <= 0:
            raise ExchangeError("ordem não executada", ambiguous=True)
        price = _num(order.get("average") or (cost / filled), "preço médio")

        fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
        fee_base = 0.0
        fee_cash = 0.0
        estimated = False
        for f in fees:
            cur, amt = f.get("currency"), _num(f.get("cost") or 0, "taxa")
            if cur == self.base:
                fee_base += amt
            elif cur == self.quote:
                fee_cash += amt
            elif cur:
                try:
                    p = _num(self.ex.fetch_ticker(f"{cur}/{self.quote}").get("last"), "preço da taxa")
                    fee_cash += amt * p
                except Exception:
                    fee_cash += cost * self.assumed_fee_rate
                    estimated = True
        if not fees:  # sem informação de taxa: estima de forma conservadora
            estimated = True
            if side == "buy":
                fee_base = filled * self.assumed_fee_rate
            else:
                fee_cash = cost * self.assumed_fee_rate
        return Fill(side, oid, price, filled, cost, fee_base, fee_cash, estimated)


class MultiGateway:
    """Vários pares spot (mesma moeda de cotação) numa conta só.

    Os métodos por ativo aceitam `symbol` (padrão: o primeiro par), de modo que o código de
    um ativo só continua funcionando. O patrimônio é o TOTAL da conta: cotação + soma de
    (saldo de cada ativo base x preço). É isso que o fusível mede.
    Pares que a exchange não lista (ou estão inativos) são descartados e ficam em `skipped`.
    """

    def __init__(self, ex: Any, symbols: list[str] | tuple[str, ...], assumed_fee_rate: float = 0.001,
                 sleep: Callable[[float], None] = time.sleep):
        if not symbols:
            raise ExchangeError("nenhum símbolo configurado")
        self.ex = ex
        self.gws: dict[str, CcxtGateway] = {}
        self.skipped: dict[str, str] = {}
        for s in dict.fromkeys(symbols):
            try:
                g = CcxtGateway(ex, s, assumed_fee_rate, sleep)
                if g._market.get("active") is False:
                    raise ExchangeError("par inativo")
            except Exception as e:
                self.skipped[s] = type(e).__name__ if not isinstance(e, ExchangeError) else str(e)
                continue
            self.gws[s] = g
        if not self.gws:
            raise ExchangeError(f"nenhum dos pares existe na exchange: {', '.join(self.skipped)}")
        quotes = {g.quote for g in self.gws.values()}
        if len(quotes) != 1:
            raise ExchangeError("os pares têm moedas de cotação diferentes")
        self.symbols: list[str] = list(self.gws)
        self.quote: str = quotes.pop()

    # ------------------------------------------------------------- compat/primário
    @property
    def symbol(self) -> str:
        return self.symbols[0]

    @property
    def base(self) -> str:
        return self.gws[self.symbol].base

    def for_symbol(self, symbol: str | None = None) -> CcxtGateway:
        try:
            return self.gws[symbol or self.symbol]
        except KeyError:
            raise ExchangeError(f"símbolo fora do universo configurado: {symbol!r}") from None

    # ----------------------------------------------------------------- por ativo
    def min_notional(self, symbol: str | None = None) -> float:
        return self.for_symbol(symbol).min_notional()

    def min_notional_any(self) -> float:
        """Menor ordem mínima entre os pares (quanto um agente precisa ter para operar ALGO)."""
        return min(g.min_notional() for g in self.gws.values())

    def max_notional_any(self) -> float:
        return max(g.min_notional() for g in self.gws.values())

    def last_price(self, symbol: str | None = None) -> float:
        return self.for_symbol(symbol).last_price()

    def candles(self, timeframe: str, limit: int = 100, symbol: str | None = None) -> list[list[float]]:
        return self.for_symbol(symbol).candles(timeframe, limit)

    def base_total(self, symbol: str | None = None) -> float:
        return self.for_symbol(symbol).base_total()

    def base_free(self, symbol: str | None = None) -> float:
        return self.for_symbol(symbol).base_free()

    def market_buy(self, quote_amount: float, symbol: str | None = None) -> Fill:
        return self.for_symbol(symbol).market_buy(quote_amount)

    def market_sell(self, base_qty: float, symbol: str | None = None) -> Fill:
        return self.for_symbol(symbol).market_sell(base_qty)

    # ------------------------------------------------------------------- conta
    def quote_total(self) -> float:
        b = self.ex.fetch_balance()
        return _num((b.get(self.quote) or {}).get("total", 0) or 0, "saldo")

    def holdings(self) -> dict[str, tuple[float, float]]:
        """{símbolo: (total, livre)} do ativo base de cada par, com UMA leitura de saldo."""
        b = self.ex.fetch_balance()
        out = {}
        for s, g in self.gws.items():
            a = b.get(g.base) or {}
            out[s] = (_num(a.get("total", 0) or 0, "saldo"), _num(a.get("free", 0) or 0, "saldo"))
        return out

    def equity_cents(self) -> int:
        """Patrimônio total (cotação + todos os ativos base x preço), em centavos."""
        b = self.ex.fetch_balance()
        total = Decimal(str(_num((b.get(self.quote) or {}).get("total", 0) or 0, "saldo")))
        for s, g in self.gws.items():
            qty = _num((b.get(g.base) or {}).get("total", 0) or 0, "saldo")
            if qty > 0:
                total += Decimal(str(qty)) * Decimal(str(g.last_price()))
        return to_cents(total)

    def foreign_base_value_cents(self) -> int:
        """Valor, em centavos, dos ativos base já presentes na conta (deveria ser ~0 no início)."""
        b = self.ex.fetch_balance()
        total = Decimal(0)
        for g in self.gws.values():
            qty = _num((b.get(g.base) or {}).get("total", 0) or 0, "saldo")
            if qty > 0:
                total += Decimal(str(qty)) * Decimal(str(g.last_price()))
        return to_cents(total)

    def open_orders_count(self) -> int:
        return sum(g.open_orders_count() for g in self.gws.values())

    def cancel_all_orders(self) -> int:
        n, err = 0, None
        for g in self.gws.values():
            try:
                n += g.cancel_all_orders()
            except Exception as e:  # tenta TODOS os pares antes de reportar falha
                err = e
        if err is not None:
            raise err
        return n


def make_exchange(cfg: Config, ccxt_module: Any = None):
    """Cria o objeto ccxt (ou a simulação, em modo paper)."""
    if ccxt_module is None:
        try:
            import ccxt as ccxt_module  # type: ignore
        except ImportError:
            raise ExchangeError("biblioteca ccxt não instalada (pip install ccxt)") from None
    if cfg.mode == "paper":
        public = ccxt_module.binance({"enableRateLimit": True})
        path = f"{cfg.data_dir}/paper_account.json"
        return SimExchange(public, list(cfg.symbols), quote_balance=cfg.bankroll_cents / 100,
                           persist_path=path)
    ex = ccxt_module.binance({
        "apiKey": cfg.api_key, "secret": cfg.api_secret, "enableRateLimit": True,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    })
    if cfg.mode == "testnet":
        ex.set_sandbox_mode(True)
    return ex
