#!/usr/bin/env python3
"""Agentes darwinianos de trade (spot, Binance via ccxt) com fusível de patrimônio.

  python main.py check                 # só leitura: conexão, mínimo da exchange, saldo (não opera)
  python main.py init                  # cria o(s) agente(s) com a banca lida na API
  python main.py run                   # roda o loop (modo definido em MODE no .env)
  python main.py run --live            # DINHEIRO REAL: exige MODE=live no .env E --live aqui
  python main.py status
  python main.py report                # resumo: trades, ganhos/perdas líquidos, mortes, clones
  python main.py liquidate [--live]    # emergência: cancela ordens e vende o ativo base
  python main.py reset-breaker --token SEU_TOKEN [--rebase]
  python main.py download [--days 90] [--timeframe 5m]      # baixa histórico (público, sem chaves)
  python main.py backtest [--days N] [--stress]             # replay do sistema completo no histórico
  python main.py backtest --optimize [--folds 3 --trials 30 --workers 2]   # walk-forward fora da amostra

Códigos de saída: 0 ok | 2 erro de uso/config | 3 fusível disparado | 4 extinção
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
import sys
import time
from typing import Any, Callable, Mapping

from darwin.breaker import CircuitBreaker
from darwin.budget import RiskBudget, RiskConfig
from darwin.config import Config, ConfigError, env_file_permission_warning, load_config, load_env
from darwin.evolution import Darwinism, EvolutionConfig
from darwin.strategy import StrategyError, StrategyParams
from darwin.exchange import ExchangeError, MultiGateway, make_exchange
from darwin.guardrails import OrderGuard, OrderGuardConfig
from darwin.ledger import Ledger
from darwin.llm import OllamaProvider
from darwin.lock import AlreadyRunning, acquire_instance_lock
from darwin.orchestrator import EXIT_OK, AccountNotClean, Orchestrator, OrchestratorError
from darwin.sentiment import SentimentAnalyzer


_CUR = ["R$"]   # rótulo da moeda de cotação (R$ para BRL; senão o código, ex.: USDT)


def brl(cents: int | None) -> str:
    return "—" if cents is None else f"{_CUR[0]} {cents / 100:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build(cfg: Config, live: bool, exchange_factory: Callable | None, sentiment: Any, sleep: Callable,
          clock: Callable[[], float] | None = None) -> Orchestrator:
    if cfg.mode == "live" and not live:
        raise SystemExit("MODE=live no .env exige também a flag --live (dois interruptores de segurança).")
    if live and cfg.mode != "live":
        raise SystemExit("--live só vale quando MODE=live no .env.")
    os.makedirs(cfg.data_dir, exist_ok=True)
    ledger = Ledger(os.path.join(cfg.data_dir, "darwin.db"))
    ex = (exchange_factory or make_exchange)(cfg)
    gw = MultiGateway(ex, cfg.symbols)
    breaker = CircuitBreaker(os.path.join(cfg.data_dir, "breaker.json"), cfg.reset_token, cfg.max_loss_pct)
    risk = RiskBudget(RiskConfig(cfg.bankroll_cents, cfg.max_loss_pct))
    guard = OrderGuard(OrderGuardConfig(frozenset(gw.symbols), cfg.bankroll_cents * 2, cfg.max_orders_per_hour))
    evo = EvolutionConfig(cfg.eval_min_roundtrips, cfg.eval_max_ticks, cfg.max_population, cfg.max_generation,
                          idle_kills=cfg.idle_kills, death_tolerance_pct=cfg.death_tolerance_pct,
                          needs_min_trades=cfg.needs_min_trades)
    darwin = Darwinism(ledger, evo, risk)
    if sentiment is None:
        sentiment = SentimentAnalyzer(OllamaProvider(cfg.ollama_model, cfg.ollama_host), cfg.news_feeds)
    _CUR[0] = "R$" if gw.quote == "BRL" else gw.quote
    orch = Orchestrator(gw, ledger, breaker, risk, guard, darwin, sentiment, cfg,
                        state_path=os.path.join(cfg.data_dir, "state.json"), sleep=sleep, clock=clock)
    orch.load_state()
    from darwin.journal import Journal
    orch.journal = Journal(os.path.join(cfg.data_dir, "journal.csv"))
    return orch


def cmd_check(orch: Orchestrator, cfg: Config, args) -> int:
    gw = orch.gw
    if cfg.mode == "live" and cfg.max_loss_pct > 40:
        print(f"⚠️ fusível de {cfg.max_loss_pct}% é frouxo para dinheiro real; considere MAX_LOSS_PCT=30 ou menos.")
    print(f"banca configurada (BANKROLL_BRL): {brl(cfg.bankroll_cents)} | fusível: {cfg.max_loss_pct}%")
    print(f"modo: {cfg.mode} | chaves: {'configuradas' if cfg.has_keys else 'não necessárias'} "
          f"| ativos operados ({len(gw.symbols)}): {', '.join(gw.symbols)}")
    if gw.skipped:
        print(f"⚠️ pares IGNORADOS (não existem ou estão inativos na exchange): "
              + ", ".join(f"{k} ({v})" for k, v in gw.skipped.items()))
    bank = cfg.bankroll_cents / 100
    for sym in gw.symbols:
        g = gw.for_symbol(sym)
        price = g.last_price()
        line = f"  {sym}: preço {brl(round(price * 100))} | ordem mínima da exchange: {g.min_notional():.2f} {g.quote}"
        step = (g.ex.market(sym).get("precision") or {}).get("amount") if hasattr(g.ex, "market") else None
        if isinstance(step, (int, float)) and step > 0:
            line += f" | passo de lote {step} {g.base} ≈ {step * price / bank * 100:.1f}% da banca"
        print(line)
    print(f"saldo {gw.quote}: {gw.quote_total():.2f} | patrimônio: {brl(gw.equity_cents())} "
          f"| ordens abertas: {gw.open_orders_count()}")
    warn = orch.fuse_feasibility(gw.equity_cents())
    if warn:
        print(f"⚠️ {warn}")
    print("nenhuma ordem foi enviada.")
    return 0


def load_params_file(path: str) -> StrategyParams:
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            return StrategyParams.from_dict(json.load(f))
    except (OSError, ValueError, StrategyError, TypeError) as e:
        raise ConfigError(f"STRATEGY_PARAMS_FILE inválido ({path}): {e}") from None


def cmd_init(orch: Orchestrator, cfg: Config, args) -> int:
    try:
        params = load_params_file(cfg.params_file) if cfg.params_file else None
        if cfg.trend_period and (params is None or not params.trend_period):
            params = replace(params or StrategyParams(), trend_period=cfg.trend_period)
        ts = orch.bootstrap(cfg.initial_agents, params)
    except ConfigError as e:
        print(f"ERRO: {e}")
        return 2
    except (AccountNotClean, OrchestratorError) as e:
        print(f"ERRO: {e}")
        return 2
    print(f"Fusível: baseline {brl(orch.breaker.initial_equity_cents)} | dispara em {brl(orch.breaker.floor_cents)} "
          f"(perda de {cfg.max_loss_pct}%).")
    warn = orch.fuse_feasibility()
    if warn:
        print(f"⚠️ {warn}")
    if cfg.trend_period:
        print(f"Filtro de tendência ligado: SMA de {cfg.trend_period} candles (só compra acima dela).")
    if cfg.params_file:
        print(f"Parâmetros da estratégia carregados de {cfg.params_file} (os filhos mutam a partir deles).")
    for t in ts:
        print(f"- {t.name}: capital {brl(orch.ledger.balance(t.name))}")
    return 0


def cmd_run(orch: Orchestrator, cfg: Config, args) -> int:
    if not orch.traders and not orch.breaker.is_open():
        print("Nada inicializado: rode `python main.py init` primeiro.")
        return 2
    try:
        lock = acquire_instance_lock(cfg.data_dir)
    except AlreadyRunning:
        print("ERRO: já existe um robô rodando com esta pasta de dados (talvez em outra janela do PowerShell). "
              "Feche-o com Ctrl+C. Se não achar a janela: `Get-Process python | Stop-Process` (fecha TODOS os Python).")
        return 2
    print(f"Rodando em modo {cfg.mode.upper()} com {len(orch.gw.symbols)} ativo(s): "
          f"{', '.join(orch.gw.symbols)} — Ctrl+C para parar.")
    try:
        code = orch.run(max_ticks=args.max_ticks)
    except KeyboardInterrupt:
        open_pos = [t.name for t in orch.alive() if t.in_position]
        orch.save_state()
        if args.liquidate_on_exit:
            orch.liquidate_all("saída pelo usuário")
        elif open_pos:
            print(f"\n⚠️ Parado com posição ABERTA ({', '.join(open_pos)}). Sem o robô rodando NÃO há stop-loss. "
                  "Use `python main.py liquidate` para vender.")
        return EXIT_OK
    finally:
        lock.close()
    return code


def cmd_status(orch: Orchestrator, cfg: Config, args) -> int:
    s = orch.status()
    print(f"Patrimônio (API): {brl(s['equity_cents'])} | baseline: {brl(s['initial_cents'])} | "
          f"PnL da conta: {brl(s['account_pnl_cents'])} | piso do fusível: {brl(s['floor_cents'])}")
    state = "ABERTO (sistema parado)" if s["breaker"] == "open" else "fechado (operando)"
    print(f"Disjuntor: {state} {s['breaker_reason']} | tick {s['tick']} | mortos: {s['dead']} "
          f"| tesouro: {brl(s['treasury_cents'])} | descompasso contábil: {brl(s['drift_cents'])}")
    for t in s["traders"]:
        print(f"- {t['name']} gen{t['gen']} capital {brl(t['capital_cents'])} "
              f"{('[POSICIONADO em ' + t['symbol'] + ']') if t['in_position'] else ''} trades {t['window_roundtrips']} "
              f"ticks {t['window_ticks']} peso_sentimento {t['sentiment_weight']:.2f}")
    return 0


def cmd_report(orch: Orchestrator, cfg: Config, args) -> int:
    ents = [e for e in orch.ledger.entries() if e.kind in ("trade_gain", "trade_loss")]
    wins = [e.amount_cents for e in ents if e.kind == "trade_gain"]
    losses = [e.amount_cents for e in ents if e.kind == "trade_loss"]
    n = len(ents)
    net = sum(wins) - sum(losses)
    print(f"Trades fechados: {n} | ganhos: {len(wins)} ({brl(sum(wins))}) | perdas: {len(losses)} ({brl(sum(losses))})")
    print(f"Resultado líquido de taxas: {brl(net)} | acerto: {(len(wins) / n * 100) if n else 0:.0f}%")
    if wins and losses:
        print(f"Ganho médio {brl(sum(wins) // len(wins))} x perda média {brl(sum(losses) // len(losses))}")
    print(f"Agentes vivos: {len(orch.traders)} | mortos: {len(orch.graveyard)} | clones nascidos: "
          f"{sum(1 for t in orch.traders.values() if t.parent) + sum(1 for g in orch.graveyard if g.get('parent'))}")
    for g in orch.graveyard[-5:]:
        print(f"  💀 {g['name']} (gen {g['generation']}): {g['reason']}")
    s = orch.status()
    print(f"Patrimônio (API): {brl(s['equity_cents'])} vs baseline {brl(s['initial_cents'])} "
          f"| descompasso contábil: {brl(s['drift_cents'])}")
    return 0


def cmd_liquidate(orch: Orchestrator, cfg: Config, args) -> int:
    return 0 if orch.liquidate_all("manual") else 2


def cmd_reset(orch: Orchestrator, cfg: Config, args) -> int:
    rebase = orch.gw.equity_cents() if args.rebase else None
    if orch.breaker.reset(args.token, rebase):
        print("Disjuntor reaberto." + (f" Nova baseline: {brl(rebase)}." if rebase else
                                       " Baseline mantida: se o patrimônio seguir abaixo do piso, ele dispara de novo."))
        return 0
    print("Reset negado (token inválido, não configurado, ou liquidação pendente).")
    return 2


# --------------------------------------------------------------------------- histórico e backtest
def _import_ccxt():
    try:
        import ccxt  # type: ignore
        return ccxt
    except ImportError:
        raise ExchangeError("biblioteca ccxt não instalada (pip install ccxt)") from None


def cmd_download(cfg: Config, args, ccxt_module=None) -> int:
    from darwin.history import HistoryError, download_history
    tf = args.timeframe or cfg.timeframe
    print(f"Baixando {args.days} dia(s) de candles {tf} de {len(cfg.symbols)} par(es) para {args.dir}/ "
          "(API pública, sem chaves; pode levar alguns minutos)…")
    try:
        rep = download_history(ccxt_module or _import_ccxt(), cfg.symbols, tf, args.days, args.dir)
    except (HistoryError, ExchangeError) as e:
        print(f"ERRO: {e}")
        return 2
    ok = [s for s, v in rep.items() if isinstance(v, int)]
    bad = {s: v for s, v in rep.items() if not isinstance(v, int)}
    print(f"Pronto: {len(ok)} par(es) com dados." + (f" Sem dados: {bad}" if bad else ""))
    return 0 if ok else 2


def cmd_journal(cfg: Config, args) -> int:
    from darwin import journal as jn
    path = os.path.join(cfg.data_dir, "journal.csv")
    if not os.path.exists(path):
        print(f"Ainda não há diário em {path}. Ele é gravado sozinho enquanto o robô roda (`python main.py run`).")
        return 2
    print(jn.format_analysis(jn.analyze(jn.load(path))))
    return 0


def cmd_backtest(cfg: Config, args) -> int:
    from darwin import backtest as bk
    from darwin.history import HistoryError, TF_MS, load_history
    tf = args.timeframe or cfg.timeframe
    try:
        hist = load_history(args.dir, cfg.symbols, tf)
    except HistoryError as e:
        print(f"ERRO: {e}")
        return 2
    syms = [s for s in cfg.symbols if s in hist.ohlcv]
    miss = [s for s in cfg.symbols if s not in hist.ohlcv]
    print(f"Histórico: {len(syms)} par(es), {hist.n} candles {tf}"
          + (f" | sem dados: {', '.join(miss)}" if miss else "")
          + (f" | candles repetidos para tapar buracos: {sum(hist.gaps.values())}" if any(hist.gaps.values()) else ""))
    bt = bk.BacktestConfig(symbols=tuple(syms), bankroll_cents=cfg.bankroll_cents, max_loss_pct=cfg.max_loss_pct,
                           timeframe=tf, agents=args.agents or cfg.initial_agents, evolve=not args.fixed,
                           eval_min_roundtrips=cfg.eval_min_roundtrips, eval_max_ticks=cfg.eval_max_ticks,
                           max_population=cfg.max_population, max_generation=cfg.max_generation,
                           idle_kills=cfg.idle_kills, max_orders_per_hour=cfg.max_orders_per_hour,
                           death_tolerance_pct=cfg.death_tolerance_pct, needs_min_trades=cfg.needs_min_trades)
    start = None
    if args.days:
        start = max(0, hist.n - int(args.days * 86_400_000 / TF_MS[tf]))
    if args.trend_filter and not args.optimize:
        bt = replace(bt, params=bk.default_params(True).to_dict())
        print(f"Filtro de tendência ligado (SMA de {bk.TREND_DEFAULT} candles).")
    if getattr(args, "direction", False):
        from darwin import direction as dr
        try:
            wf = dr.walk_forward(hist, bt, folds=args.folds, threshold=args.threshold)
        except bk.BacktestError as e:
            print(f"ERRO: {e}")
            return 2
        print(dr.format_direction(wf))
        return 0
    if getattr(args, "rotation", False):
        from darwin import rotation as rot
        try:
            wf = rot.walk_forward(hist, bt, folds=args.folds, trials=args.trials, seed=args.seed, progress=print)
        except bk.BacktestError as e:
            print(f"ERRO: {e}")
            return 2
        print(rot.format_rotation(wf))
        return 0
    try:
        if args.optimize:
            wf = bk.walk_forward(hist, bt, folds=args.folds, trials=args.trials, seed=args.seed,
                                 workers=args.workers, hist_dir=args.dir, min_trades=args.min_trades,
                                 progress=print, trend=args.trend_filter)
            print(bk.format_walk_forward(wf))
            paths = bk.save_outputs(args.out, wf=wf)
            print(f"\nArquivos salvos: {', '.join(paths)}")
            print("Para o robô usar o último conjunto escolhido, ponha no .env:  "
                  f"STRATEGY_PARAMS_FILE={os.path.join(args.out, 'best_params.json')}  (e rode `init` num data/ novo). "
                  "Só faça isso se a CONCLUSÃO acima não for 'SEM VANTAGEM' ou 'INCONCLUSIVO'.")
            return 0
        res = bk.run_backtest(hist, bt, start)
        bench = bk.buy_and_hold(hist, syms, res.start, res.end, bt)
        stress = bk.stress_test(hist, bt, start, None) if args.stress else None
        title = "BACKTEST — sistema completo (com darwinismo)" if bt.evolve else "BACKTEST — estratégia fixa"
        print(bk.format_result(res, bench, stress, title))
        paths = bk.save_outputs(args.out, res=res)
        print(f"\nArquivos salvos: {', '.join(paths)}")
        return 0
    except bk.BacktestError as e:
        print(f"ERRO: {e}")
        return 2


COMMANDS = {"check": cmd_check, "init": cmd_init, "run": cmd_run, "status": cmd_status,
            "liquidate": cmd_liquidate, "report": cmd_report, "reset-breaker": cmd_reset}


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None,
         exchange_factory: Callable | None = None, sentiment: Any = None,
         sleep: Callable[[float], None] = time.sleep, ccxt_module: Any = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--live", action="store_true", help="confirma dinheiro real (exige MODE=live no .env)")
    from darwin import __version__
    p.add_argument("--version", action="version", version=f"darwin_agents v{__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "init", "status", "report", "liquidate"):
        sub.add_parser(name, parents=[common])
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--max-ticks", type=int, default=None)
    r.add_argument("--liquidate-on-exit", action="store_true")
    d = sub.add_parser("download")
    d.add_argument("--days", type=int, default=90)
    d.add_argument("--timeframe", default=None)
    d.add_argument("--dir", default="data_hist")
    sub.add_parser("journal")
    b = sub.add_parser("backtest")
    b.add_argument("--days", type=int, default=None, help="usa só os últimos N dias do histórico baixado")
    b.add_argument("--timeframe", default=None)
    b.add_argument("--dir", default="data_hist")
    b.add_argument("--out", default="backtest_out")
    b.add_argument("--agents", type=int, default=None)
    b.add_argument("--fixed", action="store_true", help="estratégia fixa, sem darwinismo")
    b.add_argument("--stress", action="store_true", help="refaz com taxas 2x e derrapagem 3x")
    b.add_argument("--optimize", action="store_true", help="walk-forward: busca de parâmetros com teste fora da amostra")
    b.add_argument("--trend-filter", action="store_true", help="liga o filtro de tendência longa (SMA 200; na otimização sorteia 50-400)")
    b.add_argument("--direction", action="store_true", help="previsor de direção diária (use --timeframe 1d)")
    b.add_argument("--threshold", type=float, default=0.52, help="prob. mínima de 'sobe' para ficar comprado (--direction)")
    b.add_argument("--rotation", action="store_true", help="testa a rotação por força relativa (walk-forward)")
    b.add_argument("--folds", type=int, default=3)
    b.add_argument("--trials", type=int, default=30)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--min-trades", type=int, default=10)
    b.add_argument("--seed", type=int, default=1)
    rs = sub.add_parser("reset-breaker", parents=[common])
    rs.add_argument("--token", required=True)
    rs.add_argument("--rebase", action="store_true")
    args = p.parse_args(argv)
    try:  # saída em tempo real mesmo com pipe (| Tee-Object): sem isso o texto fica preso no buffer
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    print(f"darwin_agents v{__version__}")
    if env is None:
        if not os.path.exists(".env"):
            print(f"⚠️ Arquivo .env NÃO encontrado em {os.getcwd()}. Usando valores padrão (banca R$ 20, modo paper). "
                  "No Windows confira se ele não ficou como '.env.txt' (ative 'Extensões de nomes de arquivos').")
        else:
            warn = env_file_permission_warning(".env")
            if warn:
                print(f"⚠️ {warn}")
            print(f"(.env carregado de {os.path.abspath('.env')})")
        env = load_env(".env")
    try:
        cfg = load_config(env)
    except ConfigError as e:
        print(f"ERRO de configuração: {e}")
        return 2
    if args.cmd == "download":
        return cmd_download(cfg, args, ccxt_module)
    if args.cmd == "backtest":
        return cmd_backtest(cfg, args)
    if args.cmd == "journal":
        return cmd_journal(cfg, args)
    try:
        orch = build(cfg, args.live, exchange_factory, sentiment, sleep,
                     clock=time.time if exchange_factory is None else None)
    except ExchangeError as e:
        print(f"ERRO: {e}")
        return 2
    return COMMANDS[args.cmd](orch, cfg, args)


if __name__ == "__main__":
    sys.exit(main())
