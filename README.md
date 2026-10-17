**Versão 10** (igual ao número do zip `darwin_agents_v10.zip`). Para conferir se a colagem funcionou, rode `python main.py --version`: tem que mostrar `darwin_agents v10`. Toda execução também imprime essa versão na primeira linha.

# Agentes darwinianos de trade (Binance spot + Ollama local)

Frota de agentes de trade com seleção natural por **PnL real líquido de taxas**, análise técnica
(EMA + RSI) cruzada com sentimento de notícias (RSS → Ollama local, grátis) e um **fusível de
patrimônio** lido na API da Binance. Opera **vários ativos ao mesmo tempo** (padrão: 12 pares em BRL: BTC, ETH, SOL,
XRP, ADA, DOGE, LINK, LTC, AVAX, DOT, TRX e SHIB; BNB é proibido de propósito: é a moeda de desconto de taxas) numa conta só.

## Leia antes de colocar dinheiro

1. **Não há garantia de lucro. O resultado mais provável é perder dinheiro.** Taxas (~0,1% por lado) e
   derrapagem consomem o ganho de uma estratégia simples; com poucos trades por ciclo, "lucro" e
   "prejuízo" são em grande parte sorte, então a seleção darwiniana praticamente sorteia quem vive.
   Com a regra estrita pedida (PnL ≤ 0 mata), o cenário comum é **extinção**.
2. **Banca pequena tem limites matemáticos.** A Binance recusa ordens abaixo de um valor mínimo (o robô
   lê o valor real do seu par; costuma ser na faixa de R$ 5–10). Consequências:
   - Com R$ 20, só cabe **1 agente**. Clonar exige que pai e filho fiquem com pelo menos a ordem mínima
     (+20% de folga); na prática o clone só acontece com banca maior.
   - **Com R$ 20 e piso de R$ 8, a liquidação do fusível pode ser impossível**: no piso, a posição já vale
     menos que a ordem mínima e a exchange recusa a venda. O robô avisa no `init`, nunca declara
     "liquidado" se não vendeu, e tem saída preventiva por liquidez e stop-loss por trade. Para o fusível
     conseguir liquidar de verdade, a banca precisa ser ≥ R$ 37,50 (com mínimo de R$ 10).
   - O arredondamento de lote (`check` mostra o passo) deixa "poeira"; em BTC/BRL pode ser grande para R$ 20.
3. O fusível é checado a cada tick (padrão 30 s), não a cada milissegundo: um **gap** de preço entre dois
   ticks pode passar do piso. E **sem o script rodando, não há stop-loss** (as ordens são a mercado).
4. Este código **nunca foi executado contra a Binance real** (o ambiente de desenvolvimento não tinha
   acesso à rede). A integração `ccxt` foi testada contra um simulador com a mesma interface. Faça
   `check` e o modo `paper` primeiro; depois, se quiser, `testnet`.
5. Eu não sou consultor financeiro ou tributário. Ganhos com cripto podem gerar obrigações fiscais.

## Multiativo: como funciona

- `SYMBOLS` no `.env` define o universo (até 12 pares, **todos com a mesma moeda de cotação**). Pares que a
  Binance não lista são descartados e avisados em `check`/`run`.
- A cada tick o robô lê preço e candles de todos os ativos. Agente **fora da posição** calcula a nota
  (técnica x sentimento) de cada ativo com os próprios parâmetros e entra no de **maior nota ≥ limiar**
  (se o capital alcança a ordem mínima daquele ativo; senão, tenta o próximo). Agente **posicionado**
  acompanha só o ativo em que está (stop-loss, take-profit, sinal de saída, com o preço DELE).
  Como os parâmetros mutam, a seleção natural também seleciona quem escolhe melhor o ativo.
- **Fusível**: patrimônio = saldo em cotação + soma (saldo de cada ativo x preço), lido na API. Dispara em
  ≤ baseline x 40%; cancela ordens de TODOS os pares e vende TODOS os ativos. Se algum ativo não puder ser
  vendido por API (abaixo da ordem mínima), o robô diz qual e não declara "liquidado".
- **Sentimento por ativo**: cada manchete é pontuada uma vez; a nota do ativo é a média das manchetes que o
  citam (nome ou TICKER em maiúsculas, mínimo 2); com menos que isso vale a média geral do mercado.
- Falta de dados de um ativo SEM posição é tolerada (ele é ignorado no tick). Falta de dados do ativo em que
  há posição conta para o fusível (3 falhas seguidas = dispara): o robô não opera às cegas.
- **Por que BRL e não USDT**: sua banca está em reais; operar /USDT exigiria antes converter BRL→USDT (mais
  uma taxa e uma ordem fora do robô) e a ordem mínima em USDT (costuma ser ~5 USDT) equivale a mais reais do que
  a mínima dos pares /BRL. Rode `check`: ele mostra a mínima REAL de cada par. Se mesmo assim quiser USDT:
  `SYMBOLS=BTC/USDT,ETH/USDT,...`, tenha USDT na conta e `BANKROLL_BRL` passa a ser em USDT.
- **Ciclo sem nenhum trade** não mata o agente (não há PnL a julgar): ele reinicia o ciclo (`IDLE_KILLS=true`
  restaura a regra literal "PnL 0 morre", que num mercado parado leva à extinção em ~3h30).
- Limites que continuam valendo: os ativos são correlacionados (caem juntos), altcoins têm gaps maiores
  entre ticks, e com banca pequena só cabem 1–2 posições simultâneas (ordem mínima por ativo).

## Backtest (testar no passado antes de arriscar)

Roda o **mesmo código do robô** (varredura dos ativos, ordens, taxas de 0,1%, derrapagem de 0,05%, ordem mínima e
passo de lote reais de cada par, fusível, darwinismo) sobre histórico baixado da Binance (API pública, sem chaves).

```bash
python main.py download --days 180                 # baixa candles dos pares de SYMBOLS para data_hist/ (retomável)
python main.py backtest --stress                   # sistema completo (com darwinismo) no histórico todo
python main.py backtest --fixed --days 60          # estratégia fixa, só os últimos 60 dias
python main.py backtest --optimize --folds 3 --trials 30 --workers 2
```

- `--optimize` faz **walk-forward**: sorteia `--trials` conjuntos de parâmetros, mede no período de TREINO, escolhe o
  melhor e mede num período seguinte que a busca **nunca viu**. A primeira metade do histórico é treino inicial e a
  segunda é dividida em `--folds` blocos de teste (janela expansível). O que vale é o resultado de **teste**, não o de
  treino (o melhor de muitos sorteios é otimista por construção). Mostra também: parâmetros padrão, comprar e segurar
  (carteira igual), quedas máximas, quantos períodos foram positivos, sobreajuste (treino − teste) e uma
  **CONCLUSÃO** cética (`SEM VANTAGEM`, `INCONCLUSIVO`, `FRACO`, `NÃO BATE COMPRAR-E-SEGURAR` ou `INDÍCIO`).
- O relatório traz o ganho médio por trade com **intervalo de confiança de 95%** (bootstrap): se o intervalo inclui
  zero, o resultado é indistinguível de sorte. `--stress` refaz o teste com taxas 2x e derrapagem 3x.
- Arquivos em `backtest_out/`: `trades.csv`, `equity.csv`, `summary.json`, `walk_forward.json`, `best_params.json`.
  Para o robô nascer com os parâmetros escolhidos: `STRATEGY_PARAMS_FILE=backtest_out/best_params.json` no `.env`
  (vale no `init`; os filhos mutam a partir deles). **Não faça isso se a conclusão for SEM VANTAGEM ou INCONCLUSIVO.**
- Tempo: o replay roda ~1.000–2.000 candles/s com 12 pares: 180 dias de 5m (~52 mil candles) levam ~30–60 s por execução
  e o `--optimize` repete isso ~`folds x (trials + 1)` vezes (3 folds x 30 trials ≈ 10–20 min com 1 processo).
  Use `--workers` (núcleos − 1; cada processo usa ~150 MB de RAM). Para um primeiro teste: `--days 60 --trials 10`.
- **Filtro de tendência** (`--trend-filter`, ou `TREND_PERIOD=200` no `.env`): só compra com o preço acima da média
  dos últimos N candles e vende se cair abaixo. Serve para ficar em caixa em quedas longas. No `--optimize` o N é
  sorteado entre 50 e 400 e o "padrão" passa a ser N=200. Reduz queda máxima, mas não garante lucro: avalie pela
  CONCLUSÃO do walk-forward. Ele lê até 405 candles por ativo (mais lento).
- **Rotação por força relativa** (`python main.py backtest --rotation --timeframe 1h --folds 4 --trials 30`): pesquisa
  separada do robô. Mantém os N ativos que mais subiram nos últimos candles (ou fica em caixa), com taxas,
  derrapagem e valor mínimo por ordem, e valida fora da amostra. Só mede; não opera.
- Garantias testadas: o mercado de replay **nunca entrega candle do futuro** (testes envenenam o futuro e exigem
  resultado idêntico); a escolha de parâmetros não depende do período de teste; o limite de ordens por hora usa o
  relógio simulado; o fusível dispara no replay como no real.
- **Limites que nenhum backtest remove**: não inclui sentimento de notícias (não existe histórico); só pares
  listados hoje (viés de sobrevivência: moedas que sumiram não entram); executa no preço de fechamento do candle;
  não modela impacto de ordens grandes; e o passado não prevê o futuro. Um backtest bom não é prova de lucro; um
  backtest ruim é um bom motivo para não operar.

## Renascimento (menos rígido)
Se o último agente morre (ciclo com PnL ≤ 0), o robô não encerra mais: nasce `fenix1` (parâmetros do morto, mutados)
com o capital restante, se ele for maior que a ordem mínima. O fusível de 60% continua sendo a trava de perda.
`REVIVE=false` volta à regra literal (extinção, código de saída 4).

## Tolerância na seleção
`DEATH_TOLERANCE_PCT=3` deixa o agente sobreviver a perdas de até 3% do capital dele no ciclo (sem clonar), e
`DEATH_NEEDS_MIN_TRADES=true` só o julga depois de `EVAL_MIN_ROUNDTRIPS` trades. Isso diminui a "morte por azar"
com poucos trades, mas também diminui a pressão de seleção; o fusível de 60% não muda.

## Previsor de direção diária (pesquisa)
`python main.py download --days 1500 --timeframe 1d` e depois `python main.py backtest --direction --timeframe 1d --folds 4`.
Treina uma regressão logística (sobe/desce amanhã) só com o passado, fica comprado nos ativos com probabilidade
acima de `--threshold` (0,52), desconta taxas e compara acerto e retorno com chutar o lado mais comum e com comprar
e segurar. Só mede; não opera.

## Diário de sinais (aprender com dados ao vivo, sem risco)
Enquanto o robô roda (em paper ou não), ele grava a cada ~5 min, por ativo, o preço, a nota técnica e o sentimento
em `data/journal.csv`. Depois de 2 a 4 semanas rode `python main.py journal`: ele cruza cada sinal com o retorno de
1h/4h/24h depois e diz se a nota técnica ou o sentimento de notícias realmente previram algo (correlação com
intervalo de confiança). É a única forma de medir o sentimento, que o backtest não tem. Não faz o robô "aprender"
sozinho: mostra se vale a pena confiar nesses sinais antes de dar a eles qualquer peso real.

## 1. Instalação (VS Code)

1. Instale Python 3.10+ (python.org) e o VS Code (code.visualstudio.com) com a extensão **Python**.
2. VS Code → **File > Open Folder…** → escolha a pasta `darwin_agents`.
3. Terminal do VS Code (**Terminal > New Terminal**):
   ```bash
   python -m venv .venv
   # Windows (PowerShell):  .venv\Scripts\Activate.ps1
   # Mac/Linux:             source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Ctrl+Shift+P → Python: Select Interpreter** → escolha o da `.venv`.
5. Ollama (grátis, local): instale em https://ollama.com e rode no terminal:
   ```bash
   ollama pull llama3.2
   ```
   O Ollama deve estar aberto em segundo plano (porta 11434). Sem ele, o robô segue só com análise técnica.

## 2. Binance e `.env`

1. Use uma **conta ou subconta dedicada** com **somente a banca em BRL** (ex.: R$ 20). O fusível mede o
   patrimônio total do par; o `init` recusa conta com saldo do ativo base ou muito acima da banca.
2. Binance → **Gerenciamento de API → Criar API**. Permissões:
   - **Enable Reading**: ligado · **Enable Spot & Margin Trading**: ligado
   - **Enable Withdrawals**: **DESLIGADO** (o código nem possui função de saque)
   - **Restrict access to trusted IPs only**: seu IP.
3. Configure:
   ```bash
   cp .env.example .env       # Windows: copy .env.example .env
   chmod 600 .env             # Mac/Linux
   ```
   Edite `.env`: `EXCHANGE_API_KEY`, `EXCHANGE_SECRET_KEY`, `BANKROLL_BRL`, `DARWIN_RESET_TOKEN` (segredo
   longo). Mantenha `MODE=paper` por enquanto. O `.env` já está no `.gitignore`.

## 3. Rampa de uso (não pule etapas)

```bash
python main.py check             # só leitura: preço, ordem mínima, passo de lote, saldo. Não opera.
python main.py init              # cria o agente; mostra baseline e piso do fusível (e avisos)
python main.py run               # MODE=paper: ordens simuladas com preços reais. Deixe horas/dias.
python main.py status            # patrimônio (API), PnL, agentes, descompasso contábil
```
Quando estiver convencido (e aceitando perder a banca): no `.env` coloque `MODE=live`, **arquive/apague a
pasta `data/`** (estado do paper) e rode:
```bash
python main.py check --live
python main.py init --live
python main.py run --live        # os DOIS interruptores (MODE=live + --live) são obrigatórios
```
Ctrl+C para parar (avisa se há posição aberta; `run --liquidate-on-exit` vende ao sair).

| Comando | O que faz |
|---|---|
| `liquidate [--live]` | emergência: cancela ordens do par e vende todo o ativo base |
| `reset-breaker --token T [--rebase]` | reabre o disjuntor (só com token e sem liquidação pendente); sem `--rebase` ele dispara de novo se o patrimônio seguir abaixo do piso |
| códigos de saída | 0 ok · 2 erro de uso/config · 3 fusível disparado · 4 extinção |

## 4. Como funciona

- **Fusível** (`budget.py`, `breaker.py`): baseline = patrimônio lido na API no `init`. Se
  `patrimônio ≤ baseline × (100 − MAX_LOSS_PCT)%` (R$ 20 → R$ 8,00), dispara: grava estado em disco,
  cancela ordens, vende o ativo base (3 tentativas), encerra. Se a liquidação falhar, o próximo start
  retoma. Erro ambíguo ao enviar ordem (timeout) também dispara. 3 falhas seguidas lendo patrimônio ou
  mercado também disparam. O reset exige token.
- **Estratégia** (`strategy.py`): `score = (1−w)·técnico + w·sentimento`; compra se ≥ limiar, vende por
  stop-loss, take-profit ou sinal. Sem notícias → só técnica.
- **Sentimento** (`sentiment.py`): RSS (somente https, XML com DOCTYPE/ENTITY rejeitado) → Ollama pontua
  cada manchete → média em [−1, 1]. A resposta do modelo só vale se for **um número** válido. Quem
  controla as notícias pode enviesar a nota (risco residual); o dano fica limitado por stop-loss e fusível.
- **Darwinismo** (`evolution.py`): um ciclo fecha com o agente fora da posição e com
  `EVAL_MIN_ROUNDTRIPS` trades (ou `EVAL_MAX_TICKS`, quando a posição é encerrada à força).
  PnL > 0 → clona (filho com parâmetros e peso de sentimento mutados dentro de limites, financiado por
  capital do pai, desde que ambos fiquem acima da ordem mínima). PnL ≤ 0 (inclui zero) → o agente é
  morto, a posição vendida, o capital vai para o tesouro e ele sai do pool. Sem agentes: extinção, o
  sistema para e **não recomeça sozinho**.
- **Guardrails** (`guardrails.py`): só o símbolo configurado, só spot, só ordens a mercado; teto por compra
  e por hora (vendas nunca limitadas). Não existe código de saque, margem ou alavancagem.
- **Contabilidade**: dinheiro em centavos inteiros; livro-caixa SQLite append-only; o PnL de cada trade vem
  do fill real (taxas na moeda base, cotação ou BNB convertidas). `status` mostra o "descompasso" entre o
  livro e o patrimônio da API. Dica: desative "pagar taxas com BNB" para a taxa aparecer no saldo do par.

## 5. Testes

```bash
python -m unittest discover -s tests -t . -v
```
286 testes: unitário, integração (orquestrador + gateway + exchange simulada), segurança, funcional (CLI),
multiativo e backtest/histórico. Em teste de mutação, 15 defeitos (fusível, stop-loss, filtro XML, `--live`, token…) + 10 do
multiativo (equity somando todos os ativos, liquidação por ativo, escolha do melhor sinal…) + 12 do backtest
(olhar o futuro, escolha de parâmetros, fusível no replay, relógio simulado…) foram todos pegos pela suíte. Isso **não** substitui testar contra a Binance real.

## 6. Estrutura

```
main.py                 CLI
darwin/config.py        .env e validação        darwin/exchange.py   gateway ccxt (Binance spot)
darwin/ledger.py        livro-caixa             darwin/sim.py        exchange simulada (paper e testes)
darwin/budget.py        risco e piso do fusível darwin/strategy.py   EMA/RSI + sentimento
darwin/breaker.py       disjuntor               darwin/sentiment.py  RSS + Ollama
darwin/guardrails.py    limites de ordem        darwin/evolution.py  seleção natural e mutação
darwin/llm.py           Ollama/Mock             darwin/orchestrator.py  loop, varredura, ordens, liquidação
darwin/history.py       download/CSV/alinhamento darwin/backtest.py   replay, métricas, walk-forward
```
