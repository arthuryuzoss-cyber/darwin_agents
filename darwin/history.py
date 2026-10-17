"""Histórico de candles: download público (ccxt, sem chaves), cache em CSV e alinhamento no tempo.

Formato do CSV (uma pasta, um arquivo por par e timeframe): `BTC_BRL_5m.csv` com
`timestamp_ms,open,high,low,close,volume`. O download é retomável: roda de novo e só baixa o que falta.
Também guarda `meta.json` com a ordem mínima e o passo de lote REAIS de cada par.

Alinhamento: todos os pares ficam numa grade de tempo única. Candle faltando no meio = repete o
fechamento anterior (volume 0). Antes do primeiro candle do par (ainda não listado) = sem dado (None).
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
         "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
PAGE = 1000


class HistoryError(Exception):
    pass


def csv_path(out_dir: str, symbol: str, timeframe: str) -> str:
    return os.path.join(out_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")


def meta_path(out_dir: str) -> str:
    return os.path.join(out_dir, "meta.json")


def read_csv(path: str) -> list[list[float]]:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for n, rec in enumerate(csv.reader(f), 1):
            if not rec or rec[0] == "timestamp_ms":
                continue
            try:
                r = [float(x) for x in rec[:6]]
            except ValueError:
                raise HistoryError(f"{path}:{n}: linha inválida") from None
            if len(r) < 6 or not all(math.isfinite(x) for x in r) or r[4] <= 0 or r[2] < r[3] - 1e-12:
                raise HistoryError(f"{path}:{n}: candle inconsistente")
            rows.append(r)
    return rows


def write_csv(path: str, rows: list[list[float]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ms", "open", "high", "low", "close", "volume"])
        for r in rows:
            w.writerow([int(r[0])] + [repr(float(x)) for x in r[1:6]])
    os.replace(tmp, path)


def merge_rows(old: list[list[float]], new: list[list[float]]) -> list[list[float]]:
    """União por timestamp (o dado novo vence), ordenada e sem duplicatas."""
    by_ts = {int(r[0]): list(r[:6]) for r in old}
    for r in new:
        by_ts[int(r[0])] = list(r[:6])
    return [by_ts[k] for k in sorted(by_ts)]


def _market_meta(ex: Any, symbol: str) -> dict:
    m = ex.market(symbol)
    out: dict = {}
    try:
        v = m["limits"]["cost"]["min"]
        if v is not None and float(v) > 0:
            out["min_cost"] = float(v)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        step = m["precision"]["amount"]
        # só confia no passo se o ccxt estiver em modo TICK_SIZE (4); em outros modos o número é "casas decimais"
        if getattr(ex, "precisionMode", 4) == 4 and step is not None and float(step) > 0:
            out["amount_step"] = float(step)
    except (KeyError, TypeError, ValueError):
        pass
    return out


def _fetch_range(ex: Any, sym: str, timeframe: str, since: int, until: int, step_ms: int, max_retries: int,
                 sleep: Callable[[float], None]):
    """Pagina fetch_ohlcv de `since` até `until`. Retorna (linhas, None) ou (None, nome_do_erro)."""
    out: list[list[float]] = []
    err = None
    while since < until:
        page = None
        for attempt in range(max_retries):
            try:
                page = ex.fetch_ohlcv(sym, timeframe, since, PAGE)
                break
            except Exception as e:               # rede/limite: tenta de novo
                err = type(e).__name__
                sleep(1.5 * (attempt + 1))
        if page is None:
            return None, err
        if not page:
            break
        out.extend([list(map(float, r[:6])) for r in page if len(r) >= 6])
        last = int(page[-1][0])
        if last + step_ms <= since:              # sem progresso: evita laço infinito
            break
        since = last + step_ms
        if len(page) < PAGE and since >= until - step_ms:
            break
    return out, None


def download_history(ccxt_module: Any, symbols: list[str] | tuple[str, ...], timeframe: str, days: int,
                     out_dir: str, log: Callable[[str], None] = print, now_ms: int | None = None,
                     sleep: Callable[[float], None] = time.sleep, max_retries: int = 3) -> dict:
    """Baixa `days` dias de candles de cada par (API pública da Binance). Retorna {símbolo: n_candles|erro}."""
    if timeframe not in TF_MS:
        raise HistoryError(f"timeframe inválido: {timeframe}")
    if not 1 <= days <= 3650:
        raise HistoryError("days deve estar entre 1 e 3650")
    os.makedirs(out_dir, exist_ok=True)
    ex = ccxt_module.binance({"enableRateLimit": True})
    markets = ex.load_markets()
    step_ms = TF_MS[timeframe]
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    start = now - days * 86_400_000
    meta = {}
    if os.path.exists(meta_path(out_dir)):
        try:
            meta = json.load(open(meta_path(out_dir), encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    report: dict = {}
    for sym in symbols:
        if sym not in markets:
            report[sym] = "não existe na Binance"
            log(f"  {sym}: não existe na Binance (ignorado)")
            continue
        meta[sym] = _market_meta(ex, sym)
        path = csv_path(out_dir, sym, timeframe)
        have = read_csv(path) if os.path.exists(path) else []
        passes: list[tuple[int, int]] = []
        if have:
            first_cached, last_cached = int(have[0][0]), int(have[-1][0])
            if start < first_cached - step_ms:       # pediu mais dias do que o cache tem: completa o começo
                passes.append((start, first_cached))
            passes.append((last_cached + step_ms, now))
        else:
            passes.append((start, now))
        got: list[list[float]] = []
        err = None
        for since0, until in passes:
            rows, err = _fetch_range(ex, sym, timeframe, since0, until, step_ms, max_retries, sleep)
            if rows is None:
                report[sym] = f"falha de rede ({err})"
                break
            got.extend(rows)
        if sym in report:                       # falhou no meio: não grava dado parcial novo
            log(f"  {sym}: {report[sym]}")
            continue
        merged = merge_rows(have, got)
        if not merged:
            report[sym] = "sem candles no período"
            log(f"  {sym}: sem candles no período")
            continue
        write_csv(path, merged)
        report[sym] = len(merged)
        log(f"  {sym}: {len(merged)} candles ({len(got)} novos)")
    tmp = meta_path(out_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, meta_path(out_dir))
    return report


@dataclass
class History:
    timeframe: str
    timeline: list[int]
    ohlcv: dict                       # símbolo -> lista alinhada de (o,h,l,c,v) | None (ainda não listado)
    first_idx: dict                   # símbolo -> primeiro índice com dado
    meta: dict = field(default_factory=dict)
    gaps: dict = field(default_factory=dict)    # símbolo -> candles preenchidos (faltantes)

    @property
    def symbols(self) -> list[str]:
        return list(self.ohlcv)

    @property
    def n(self) -> int:
        return len(self.timeline)

    def close(self, sym: str, i: int) -> float | None:
        r = self.ohlcv[sym][i]
        return None if r is None else r[3]


def align(raw: dict, timeframe: str, meta: dict | None = None) -> History:
    """Alinha séries {símbolo: linhas} numa grade regular de tempo."""
    if timeframe not in TF_MS:
        raise HistoryError(f"timeframe inválido: {timeframe}")
    raw = {s: r for s, r in raw.items() if r}
    if not raw:
        raise HistoryError("nenhum dado de histórico")
    step = TF_MS[timeframe]
    t0 = min(int(r[0][0]) for r in raw.values())
    t1 = max(int(r[-1][0]) for r in raw.values())
    n = (t1 - t0) // step + 1
    if n > 3_000_000:
        raise HistoryError("histórico grande demais")
    timeline = [t0 + i * step for i in range(n)]
    ohlcv: dict = {}
    first: dict = {}
    gaps: dict = {}
    for sym, rows in raw.items():
        by_idx = {}
        for r in rows:
            ts = int(r[0])
            if (ts - t0) % step:
                raise HistoryError(f"{sym}: timestamp fora da grade de {timeframe}")
            by_idx[(ts - t0) // step] = (r[1], r[2], r[3], r[4], r[5])
        fi = min(by_idx)
        li = max(by_idx)
        col: list = [None] * n
        prev = None
        g = 0
        for i in range(fi, n):
            cur = by_idx.get(i)
            if cur is None:
                if i > li:
                    cur = None                      # par deixou de existir: sem dado depois do último candle
                else:
                    c = prev[3]
                    cur = (c, c, c, c, 0.0)         # buraco no meio: repete o fechamento
                    g += 1
            col[i] = cur
            prev = cur if cur is not None else prev
        ohlcv[sym] = col
        first[sym] = fi
        gaps[sym] = g
    return History(timeframe, timeline, ohlcv, first, dict(meta or {}), gaps)


def load_history(in_dir: str, symbols: list[str] | tuple[str, ...], timeframe: str) -> History:
    raw = {}
    for sym in symbols:
        p = csv_path(in_dir, sym, timeframe)
        if os.path.exists(p):
            raw[sym] = read_csv(p)
    if not raw:
        raise HistoryError(f"nenhum CSV em {in_dir} para {timeframe}; rode `python main.py download` antes")
    meta = {}
    if os.path.exists(meta_path(in_dir)):
        try:
            meta = json.load(open(meta_path(in_dir), encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    return align(raw, timeframe, meta)
