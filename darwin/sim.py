"""Exchange simulada com a MESMA interface do subconjunto do ccxt que o projeto usa.

Serve a dois propósitos:
1. Modo `paper`: dados de mercado reais (ccxt público, sem chaves) + execução simulada.
   O código de ordens exercitado é exatamente o do modo real (CcxtGateway).
2. Testes automatizados, com injeção de falhas.

Modelo de taxas (fee_mode):
- "received": taxa na moeda RECEBIDA (compra -> base, venda -> quote). É o padrão da Binance.
- "quote":    taxa sempre em moeda de cotação.
- "bnb":      taxa em BNB (terceira moeda) — exercita a conversão no gateway.
"""
from __future__ import annotations

import json
import os
import uuid
from decimal import ROUND_DOWN, Decimal
from typing import Any


class SimExchangeError(Exception):
    pass


def _d(x: Any) -> Decimal:
    return Decimal(str(x))


class SimExchange:
    def __init__(self, data_source: Any, symbol: "str | list | tuple" = "BTC/BRL", quote_balance: float = 20.0,
                 base_balance: float = 0.0, fee_rate: float = 0.001, fee_mode: str = "received",
                 slippage: float = 0.0005, min_notional: float | None = None,
                 amount_step: str = "0.00001", persist_path: str | None = None,
                 bnb_price: float = 100.0, bnb_balance: float = 1.0):
        if fee_mode not in ("received", "quote", "bnb"):
            raise ValueError("fee_mode inválido")
        self.data = data_source
        self.symbols = [symbol] if isinstance(symbol, str) else list(symbol)
        if not self.symbols:
            raise ValueError("sem símbolos")
        self.symbol = self.symbols[0]                      # primário (compatibilidade)
        self.base, self.quote = self.symbol.split("/")
        if any(x.split("/")[1] != self.quote for x in self.symbols):
            raise ValueError("todos os pares devem ter a mesma moeda de cotação")
        self.fee_rate = _d(fee_rate)
        self.fee_mode = fee_mode
        self.slippage = _d(slippage)
        self.step = _d(amount_step)
        self._step_cfg_explicit = amount_step != "0.00001"
        self.bnb_price = _d(bnb_price)
        self._min_notional_cfg = min_notional
        self.persist_path = persist_path
        self.balances = {self.quote: _d(quote_balance), "BNB": _d(bnb_balance)}
        for x in self.symbols:
            self.balances.setdefault(x.split("/")[0], Decimal(0))
        self.balances[self.base] = _d(base_balance)
        self.orders: dict[str, dict] = {}
        self.open_orders: list[dict] = []
        self._failures: dict[str, int] = {}
        self.calls: list[str] = []
        if persist_path and os.path.exists(persist_path):
            with open(persist_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k, v in saved.items():
                self.balances[k] = _d(v)

    # ------------------------------------------------------------ utilitários
    def inject_failures(self, method: str, times: int) -> None:
        self._failures[method] = times

    def _enter(self, name: str) -> None:
        self.calls.append(name)
        if self._failures.get(name, 0) > 0:
            self._failures[name] -= 1
            raise SimExchangeError(f"falha simulada em {name}")

    def _save(self) -> None:
        if self.persist_path:
            tmp = self.persist_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({k: str(v) for k, v in self.balances.items()}, f)
            os.replace(tmp, self.persist_path)

    def _check_symbol(self, symbol: str) -> None:
        if symbol not in self.symbols:
            raise SimExchangeError(f"símbolo não suportado na simulação: {symbol}")

    def _base(self, symbol: str) -> str:
        return symbol.split("/")[0]

    def _step_for(self, symbol: str) -> Decimal:
        """Passo de quantidade do par: usa o do mercado real (ccxt) quando disponível."""
        if self._step_cfg_explicit:
            return self.step
        try:
            st = self.data.market(symbol)["precision"]["amount"]
            if isinstance(st, (int, float)) and 0 < st <= 1000:
                return _d(st)
        except Exception:
            pass
        return self.step

    # ------------------------------------------------------------ API tipo ccxt
    def load_markets(self):
        self._enter("load_markets")
        if hasattr(self.data, "load_markets"):
            self.data.load_markets()
        out = {}
        for x in self.symbols:
            try:
                out[x] = self.market(x)
            except Exception:
                pass   # par que a exchange real não lista: o MultiGateway o descartará
        return out

    def market(self, symbol: str) -> dict:
        self._check_symbol(symbol)
        min_cost = self._min_notional_cfg
        real: dict = {}
        if hasattr(self.data, "market"):
            real = self.data.market(symbol) or {}   # par inexistente na exchange real => erro (BadSymbol)
        if min_cost is None:
            try:
                min_cost = real["limits"]["cost"]["min"]
            except (KeyError, TypeError):
                min_cost = None
        if min_cost is None:
            min_cost = 10.0
        return {"symbol": symbol, "base": self._base(symbol), "quote": self.quote, "active": real.get("active", True),
                "precision": {"amount": float(self._step_for(symbol))},
                "limits": {"cost": {"min": float(min_cost)}, "amount": {"min": float(self._step_for(symbol))}}}

    def fetch_ticker(self, symbol: str) -> dict:
        self._enter("fetch_ticker")
        if symbol == f"BNB/{self.quote}":
            return {"symbol": symbol, "last": float(self.bnb_price)}
        self._check_symbol(symbol)
        return self.data.fetch_ticker(symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", since=None, limit: int | None = None):
        self._enter("fetch_ohlcv")
        self._check_symbol(symbol)
        return self.data.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_balance(self) -> dict:
        self._enter("fetch_balance")
        out: dict = {"free": {}, "used": {}, "total": {}}
        for cur, amt in self.balances.items():
            out[cur] = {"free": float(amt), "used": 0.0, "total": float(amt)}
            out["free"][cur] = float(amt)
            out["total"][cur] = float(amt)
            out["used"][cur] = 0.0
        return out

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        self._check_symbol(symbol)
        st = self._step_for(symbol)
        q = (_d(amount) / st).to_integral_value(rounding=ROUND_DOWN) * st
        return format(q, "f")

    def _last(self, symbol: str | None = None) -> Decimal:
        t = self.data.fetch_ticker(symbol or self.symbol)
        return _d(t["last"])

    def _order(self, side: str, filled: Decimal, cost: Decimal, price: Decimal, fees: list,
               symbol: str | None = None) -> dict:
        oid = f"sim-{uuid.uuid4().hex[:12]}"
        order = {"id": oid, "symbol": symbol or self.symbol, "type": "market", "side": side, "status": "closed",
                 "filled": float(filled), "amount": float(filled), "cost": float(cost),
                 "average": float(price), "price": float(price), "fees": fees,
                 "fee": fees[0] if fees else None}
        self.orders[oid] = order
        return order

    def create_market_buy_order_with_cost(self, symbol: str, cost: float) -> dict:
        self._enter("create_market_buy_order_with_cost")
        self._check_symbol(symbol)
        cost_d = _d(cost)
        if cost_d <= 0:
            raise SimExchangeError("custo inválido")
        if cost_d < _d(self.market(symbol)["limits"]["cost"]["min"]):
            raise SimExchangeError("abaixo do valor mínimo da ordem (MIN_NOTIONAL)")
        base = self._base(symbol)
        step = self._step_for(symbol)
        price = self._last(symbol) * (1 + self.slippage)
        filled = (cost_d / price / step).to_integral_value(rounding=ROUND_DOWN) * step
        spent = (filled * price).quantize(Decimal("0.00000001"))
        if filled <= 0:
            raise SimExchangeError("quantidade zero")
        fee_value = spent * self.fee_rate
        fees: list[dict] = []
        if self.fee_mode == "received":
            fee_base = filled * self.fee_rate
            if self.balances[self.quote] < spent:
                raise SimExchangeError("saldo insuficiente (InsufficientFunds)")
            self.balances[self.quote] -= spent
            self.balances[base] += filled - fee_base
            fees = [{"currency": base, "cost": float(fee_base)}]
        elif self.fee_mode == "quote":
            if self.balances[self.quote] < spent + fee_value:
                raise SimExchangeError("saldo insuficiente (InsufficientFunds)")
            self.balances[self.quote] -= spent + fee_value
            self.balances[base] += filled
            fees = [{"currency": self.quote, "cost": float(fee_value)}]
        else:
            fee_bnb = fee_value / self.bnb_price
            if self.balances[self.quote] < spent or self.balances["BNB"] < fee_bnb:
                raise SimExchangeError("saldo insuficiente (InsufficientFunds)")
            self.balances[self.quote] -= spent
            self.balances[base] += filled
            self.balances["BNB"] -= fee_bnb
            fees = [{"currency": "BNB", "cost": float(fee_bnb)}]
        self._save()
        return self._order("buy", filled, spent, price, fees, symbol)

    def create_order(self, symbol: str, type: str, side: str, amount: float, price=None, params=None) -> dict:
        self._enter("create_order")
        self._check_symbol(symbol)
        if type != "market":
            raise SimExchangeError("apenas ordens a mercado na simulação")
        amt = _d(amount)
        if side == "buy":
            return self.create_market_buy_order_with_cost(symbol, float(amt * self._last(symbol)))
        if side != "sell":
            raise SimExchangeError("lado inválido")
        base = self._base(symbol)
        if amt <= 0 or amt > self.balances[base]:
            raise SimExchangeError("saldo insuficiente (InsufficientFunds)")
        price_d = self._last(symbol) * (1 - self.slippage)
        proceeds = (amt * price_d).quantize(Decimal("0.00000001"))
        if proceeds < _d(self.market(symbol)["limits"]["cost"]["min"]):
            raise SimExchangeError("abaixo do valor mínimo da ordem (MIN_NOTIONAL)")
        fee_value = proceeds * self.fee_rate
        self.balances[base] -= amt
        if self.fee_mode == "bnb":
            fee_bnb = fee_value / self.bnb_price
            self.balances[self.quote] += proceeds
            self.balances["BNB"] -= fee_bnb
            fees = [{"currency": "BNB", "cost": float(fee_bnb)}]
        else:
            self.balances[self.quote] += proceeds - fee_value
            fees = [{"currency": self.quote, "cost": float(fee_value)}]
        self._save()
        return self._order("sell", amt, proceeds, price_d, fees, symbol)

    def fetch_order(self, order_id: str, symbol: str | None = None) -> dict:
        self._enter("fetch_order")
        if order_id not in self.orders:
            raise SimExchangeError("ordem inexistente")
        return self.orders[order_id]

    def cancel_order(self, order_id: str, symbol: str | None = None):
        self._enter("cancel_order")
        return {"id": order_id, "status": "canceled"}

    def fetch_open_orders(self, symbol: str | None = None) -> list:
        self._enter("fetch_open_orders")
        if symbol is None:
            return list(self.open_orders)
        return [o for o in self.open_orders if o.get("symbol") == symbol]

    def cancel_all_orders(self, symbol: str | None = None) -> list:
        self._enter("cancel_all_orders")
        if symbol is None:
            canceled, self.open_orders = list(self.open_orders), []
            return canceled
        canceled = [o for o in self.open_orders if o.get("symbol") == symbol]
        self.open_orders = [o for o in self.open_orders if o.get("symbol") != symbol]
        return canceled
