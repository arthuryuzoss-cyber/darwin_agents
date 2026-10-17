"""Configuração via .env / variáveis de ambiente, com validação estrita.

Nomes das chaves (como pedido): EXCHANGE_API_KEY e EXCHANGE_SECRET_KEY.
As chaves nunca aparecem em repr/log. Em modo paper elas nem são necessárias.
"""
from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Mapping

MODES = ("paper", "testnet", "live")
TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h")
DEFAULT_FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
)
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,10}/[A-Z0-9]{2,10}$")
# 12 grandes por liquidez/capitalização, em BRL (a banca é em reais; veja o README sobre USDT).
# Pares que a Binance não listar em BRL são ignorados (e avisados) pelo `check`.
DEFAULT_SYMBOLS = ("BTC/BRL", "ETH/BRL", "SOL/BRL", "XRP/BRL", "ADA/BRL", "DOGE/BRL",
                   "LINK/BRL", "LTC/BRL", "AVAX/BRL", "DOT/BRL", "TRX/BRL", "SHIB/BRL")
FEE_TOKENS = ("BNB",)   # moeda de taxa: quem a guarda para desconto não pode tê-la vendida pelo robô
MAX_SYMBOLS = 12


class ConfigError(Exception):
    pass


def _float(env: Mapping[str, str], key: str, default: float, lo: float, hi: float) -> float:
    raw = env.get(key, "").strip().replace(",", ".")
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        raise ConfigError(f"{key} deve ser um número") from None
    if not (lo <= v <= hi):          # também rejeita NaN
        raise ConfigError(f"{key} deve estar entre {lo} e {hi}")
    return v


def _int(env: Mapping[str, str], key: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ConfigError(f"{key} deve ser um inteiro") from None
    if not lo <= v <= hi:
        raise ConfigError(f"{key} deve estar entre {lo} e {hi}")
    return v


@dataclass(frozen=True)
class Config:
    mode: str = "paper"
    symbols: tuple = DEFAULT_SYMBOLS
    bankroll_cents: int = 2000
    max_loss_pct: int = 60
    timeframe: str = "5m"
    tick_seconds: int = 30
    initial_agents: int = 1
    eval_min_roundtrips: int = 3
    eval_max_ticks: int = 400
    max_population: int = 4
    max_generation: int = 5
    max_orders_per_hour: int = 12
    idle_kills: bool = False
    death_tolerance_pct: float = 0.0
    needs_min_trades: bool = False
    max_stale_candles: int = 3          # candle mais novo mais velho que (1+N) períodos = dado desatualizado
    cooldown_candles: int = 2           # após vender um ativo, não recompra por N candles (0 = desligado)
    stoploss_guard: int = 3             # K stop-loss em 24 candles pausa TODAS as novas compras por 12 candles (0 = desligado)
    revive: bool = True               # extinção: renasce 1 agente (mutado) com o capital restante, se viável
    params_file: str | None = None
    trend_period: int = 0
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://127.0.0.1:11434"
    news_feeds: tuple = DEFAULT_FEEDS
    data_dir: str = "data"
    reset_token: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    api_secret: str | None = field(default=None, repr=False)

    @property
    def symbol(self) -> str:
        """Primeiro ativo da lista (compatibilidade e mensagens)."""
        return self.symbols[0]

    @property
    def quote(self) -> str:
        return self.symbols[0].split("/")[1]

    @property
    def has_keys(self) -> bool:
        return bool(self.api_key and self.api_secret)


def parse_env_file(path: str) -> dict[str, str]:
    """Parser mínimo de .env (usado se python-dotenv não estiver instalado)."""
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            out[k.strip()] = v
    return out


def env_file_permission_warning(path: str) -> str | None:
    """Avisa se o .env pode ser lido por outros usuários (POSIX)."""
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return None
    if os.name == "posix" and mode & (stat.S_IRGRP | stat.S_IROTH):
        return f"{path} é legível por outros usuários; rode: chmod 600 {path}"
    return None


def load_env(env_file: str | None = ".env") -> dict[str, str]:
    """Combina .env (se existir) com o ambiente do processo (o ambiente tem prioridade)."""
    merged: dict[str, str] = {}
    if env_file and os.path.exists(env_file):
        try:
            from dotenv import dotenv_values  # type: ignore

            merged.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
        except ImportError:
            merged.update(parse_env_file(env_file))
    merged.update({k: v for k, v in os.environ.items()})
    return merged


def load_config(env: Mapping[str, str]) -> Config:
    mode = env.get("MODE", "paper").strip().lower() or "paper"
    if mode not in MODES:
        raise ConfigError(f"MODE deve ser um de {MODES}")

    raw_syms = env.get("SYMBOLS", "").strip() or env.get("SYMBOL", "").strip()
    symbols = tuple(dict.fromkeys(x.strip().upper() for x in raw_syms.split(",") if x.strip())) or DEFAULT_SYMBOLS
    if len(symbols) > MAX_SYMBOLS:
        raise ConfigError(f"SYMBOLS aceita no máximo {MAX_SYMBOLS} pares")
    for sym in symbols:
        if not _SYMBOL_RE.match(sym):
            raise ConfigError(f"símbolo inválido {sym!r} (ex.: BTC/BRL)")
    for sym in symbols:
        if sym.split("/")[0] in FEE_TOKENS:
            raise ConfigError(f"{sym}: BNB é a moeda de desconto de taxas da Binance e costuma ficar na conta; "
                              "o robô a trataria como patrimônio dele e a venderia na liquidação. Não use BNB em SYMBOLS.")
    if len({sym.split("/")[1] for sym in symbols}) != 1:
        raise ConfigError("todos os pares de SYMBOLS devem ter a MESMA moeda de cotação (ex.: todos /BRL "
                          "ou todos /USDT), senão o patrimônio total não é comparável")

    bank_raw = (env.get("BANKROLL", "").strip() or env.get("BANKROLL_BRL", "").strip() or "20")
    try:
        bankroll = Decimal(bank_raw)
    except InvalidOperation:
        raise ConfigError("BANKROLL_BRL inválido") from None
    if bankroll <= 0 or bankroll > Decimal("1000000") or bankroll != bankroll.quantize(Decimal("0.01")):
        raise ConfigError("BANKROLL_BRL deve ser positivo, com no máximo 2 casas decimais")

    timeframe = env.get("TIMEFRAME", "5m").strip()
    if timeframe not in TIMEFRAMES:
        raise ConfigError(f"TIMEFRAME deve ser um de {TIMEFRAMES}")

    feeds_raw = env.get("NEWS_FEEDS", "").strip()
    feeds = tuple(u.strip() for u in feeds_raw.split(",") if u.strip()) or DEFAULT_FEEDS
    for u in feeds:
        if not u.startswith("https://"):
            raise ConfigError("NEWS_FEEDS aceita apenas URLs https://")

    trend_period = _int(env, "TREND_PERIOD", 0, 0, 400)
    if trend_period and trend_period < 50:
        raise ConfigError("TREND_PERIOD deve ser 0 (desligado) ou entre 50 e 400")

    key = env.get("EXCHANGE_API_KEY", "").strip() or None
    secret = env.get("EXCHANGE_SECRET_KEY", "").strip() or None
    if mode in ("testnet", "live") and not (key and secret):
        raise ConfigError("EXCHANGE_API_KEY e EXCHANGE_SECRET_KEY são obrigatórias em testnet/live")

    return Config(
        mode=mode,
        symbols=symbols,
        bankroll_cents=int(bankroll * 100),
        max_loss_pct=_int(env, "MAX_LOSS_PCT", 60, 1, 95),
        timeframe=timeframe,
        tick_seconds=_int(env, "TICK_SECONDS", 30, 5, 3600),
        initial_agents=_int(env, "INITIAL_AGENTS", 1, 1, 4),
        eval_min_roundtrips=_int(env, "EVAL_MIN_ROUNDTRIPS", 3, 1, 50),
        eval_max_ticks=_int(env, "EVAL_MAX_TICKS", 400, 10, 100000),
        max_population=_int(env, "MAX_POPULATION", 4, 1, 10),
        max_generation=_int(env, "MAX_GENERATION", 5, 1, 20),
        max_orders_per_hour=_int(env, "MAX_ORDERS_PER_HOUR", 12, 1, 120),
        params_file=env.get("STRATEGY_PARAMS_FILE", "").strip() or None,
        trend_period=trend_period,
        idle_kills=env.get("IDLE_KILLS", "").strip().lower() in ("1", "true", "yes", "sim"),
        death_tolerance_pct=_float(env, "DEATH_TOLERANCE_PCT", 0.0, 0.0, 20.0),
        needs_min_trades=env.get("DEATH_NEEDS_MIN_TRADES", "").strip().lower() in ("1", "true", "yes", "sim"),
        max_stale_candles=_int(env, "MAX_STALE_CANDLES", 3, 1, 100),
        cooldown_candles=_int(env, "COOLDOWN_CANDLES", 2, 0, 1000),
        stoploss_guard=_int(env, "STOPLOSS_GUARD", 3, 0, 100),
        revive=env.get("REVIVE", "true").strip().lower() not in ("0", "false", "no", "nao", "não"),
        ollama_model=env.get("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2",
        ollama_host=env.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip(),
        news_feeds=feeds,
        data_dir=env.get("DATA_DIR", "data").strip() or "data",
        reset_token=env.get("DARWIN_RESET_TOKEN", "").strip() or None,
        api_key=key,
        api_secret=secret,
    )
