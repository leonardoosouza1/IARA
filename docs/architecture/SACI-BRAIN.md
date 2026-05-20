# Saci-Brain — Cross-Asset Knowledge Augmentation Layer

Module **adicionado** ao saci-trader. **Não substitui** o SwarmBrain
(LightningWorkers + SwarmCoordinator). Atua como camada de **hints**
factual + analógica acoplada ao SwarmCoordinator. Mantém todo o pipeline
de treino, backtest, paper trade e live trade que já existe.

> Inspirado no byte-language-lab brain (HDC + KG + gravity forest + WAL).
> Arquitetura genérica; só o **encoding** muda (de bytes/PT-BR pra ticks
> multi-asset).

## Por que existir

O SwarmBrain atual é **single-asset** (EURUSD) e aprende padrões
**técnicos** por timeframe. Não tem como expressar:

- "DXY subindo +0.3% nos últimos 5min costuma derrubar EUR em 1h"
- "FOMC em 30min → flight to risk → BTC vol explode"
- "Última vez que DXY+EUR+OURO estavam nesse combo, o que aconteceu?"

O saci-brain ataca isso com 4 estruturas, todas mergeable e backtest-friendly.

## Estruturas

### 1. Cross-Asset Knowledge Graph (KG)

Triples explícitos de relação macro. Encoded por relação tipada.

```
key                                   value
─────────────────────────────────────────────────────────────────
(DXY,         correlates_neg,    EURUSD)      0.85  (rolling 30d)
(DXY,         correlates_pos,    USDJPY)      0.78
(DXY,         correlates_neg,    BTC)         0.42  (instável)
(FOMC_hike,   leads_to,          USD_strong)  causal
(USD_strong,  leads_to,          EUR_weak)    causal (1-hop)
(QE_announce, leads_to,          BTC_rally)   causal
```

Single-hop responde: "DXY +0.3% → ?". Multi-hop (2-3): "FOMC → USD → EUR".

**Como popular:**
- Histórico: pandas + rolling correlation por par → triples com pesos.
- Eventos macro: distill_kg-style do Qwen3 (ou base curated) → triples
  causais.
- Append-only WAL — recálculo periódico atualiza pesos sem reset.

### 2. Gravity Forest (assets + correlation graph)

```
trees   = ativos (EUR, JPY, GBP, BTC, ETH, GOLD, OIL, DXY, ...)
edges   = mycelium = correlação rolling. peso = |corr|, sinal = +/-.
gravity = tick em ativo A → partícula propaga via edges → ativa trees
          B, C, D proporcional ao peso.
```

Diferença vs SwarmBrain atual: este é **por timeframe** (M5, M15, ...);
forest é **por ativo**. Complementares — timeframe captura ritmo, forest
captura **propagação cross-asset**.

### 3. HDC multi-asset memory

Encoda janela de N ticks de M ativos como um único hipervetor:

```
HV_estado = bind(
  permute_DXY(  HV(DXY_window)  ),
  permute_EUR(  HV(EUR_window)  ),
  permute_BTC(  HV(BTC_window)  ),
  permute_time( HV(hour_of_day, day_of_week, news_in_30min) ),
)
```

Retrieve: tick atual encoda → top-k passados similares (hamming/cosine).
Pergunta: **"o que aconteceu nas próximas 1h/4h/1d dessas k janelas?"**
Resposta = distribuição empírica de retornos por ativo. **Analogia
histórica direta**, sem inferência neural.

### 4. WAL append-only

`data/wal/ticks_<asset>_<date>.bin`, `data/wal/trades_<date>.bin`.
Replay determinístico. Cada decisão/trade liga a um trace_id.

## Fluxo na vida real

```
tick chega                                      ┌── KG ──┐
   │                                            │ traverse│  triples ativadas
   ▼                                            │ 1-2 hop │  (DXY+ → EUR-, BTC-)
encode HV multi-asset ─────────► retrieve ─────►│ HDC top-│  k passadas similares
                                                │ k       │  + distribuição next-1h
                                                │ forest  │  trees ativadas via
                                                │ propagar│  mycelium (pesos corr)
                                                └────┬────┘
                                                     ▼
                                            hints {
                                              kg_signals: [...],
                                              analogies:  [...],
                                              cross_asset_pressure: {EUR:-0.15, BTC:-0.5}
                                            }
                                                     │
                                                     ▼
                                       ┌─────────────────────────────┐
                                       │  SwarmCoordinator existente │
                                       │  + brain_hints (concat na   │
                                       │    cross-attention)          │
                                       └─────────────┬───────────────┘
                                                     ▼
                                              Long / Short / Wait
```

Brain **não decide**. SwarmCoordinator decide com brain como contexto.

## Layout do módulo (proposto)

```
backend/src/brain_kg/                      ← novo módulo paralelo
  __init__.py
  encoder.py          tick-window → HV multi-asset (binding, permutation)
  kg.py               triples store, traverse, weight decay
  forest.py           trees=assets, mycelium=corr matrix, gravity propagate
  wal.py              append-only tick/trade log + replay
  hints.py            orchestra: query brain → dict de hints
  build_kg.py         CLI: distill correlações do histórico → triples
  test_*.py
backend/src/brain/agents/swarm_brain.py    ← MODIFICAR (não substituir):
                                              adicionar hints como input
                                              opcional no SwarmCoordinator
docs/architecture/SACI-BRAIN.md            ← este doc
```

**Garantia:** se o módulo brain_kg falhar/desabilitado, saci treina e
opera **idêntico ao atual**. `enable_brain=False` no config → bypass total.

## Plano de implementação (fases)

### Fase 0 — WAL + replay

1. `wal.py`: append-only `data/wal/ticks_{asset}_{YYYYMM}.bin`. Schema:
   `(timestamp_ns:u64, asset_id:u16, o:f32, h:f32, l:f32, c:f32, v:f32)`.
2. Migração: script único que varre `data/raw/` e gera o WAL.
3. Reader que faz iter sobre WAL com janela deslizante.

Exit: `python -m brain_kg.wal replay --asset EURUSD --from 2025-01-01` reproduz
ticks na ordem original.

### Fase 1 — KG cross-asset

1. `build_kg.py`: para cada par de ativos, calcula rolling correlation
   (30/90/180d), emite triple com peso. Decaimento exponencial em rebuilds.
2. `kg.py`: store em memória (HashMap), persist em parquet, carrega no boot.
3. Triple-extraction de eventos macro (FOMC, NFP, CPI) via DataFrame de
   calendário econômico.

Exit: `python -m brain_kg.kg query --tick "DXY +0.3% 5min"` retorna lista
de triples ativadas.

### Fase 2 — Forest + mycelium

1. `forest.py`: trees do enum de ativos do saci. Position = embedding HDC
   do ativo (ou aprendido).
2. mycelium = matriz simétrica de correlações (mesmas usadas no KG, mas
   indexada por idx de tree).
3. `propagate(asset_idx, delta)`: difusão 1-hop ponderada pela correlação.

Exit: `propagate(DXY_idx, +0.003)` retorna pressões esperadas nos outros
trees. Validado contra rolling regression do mesmo período.

### Fase 3 — HDC multi-asset memory

1. `encoder.py`: encoda janela de N=64 ticks por ativo em HV de 1024 bits;
   bind multi-asset com permutações fixas.
2. Memory bank = `Vec<(HV, future_returns_by_asset)>`. Append toda barra
   fechada.
3. Retrieve top-k via hamming; agrega future_returns em distribuição
   empírica.

Exit: `retrieve(now)` retorna 12 análogos com seus retornos de 1h/4h/1d
por ativo. Validado: distribuição cobre o retorno real ≥60% das vezes
no walk-forward.

### Fase 4 — Hints no SwarmCoordinator

1. `hints.py`: orquestra KG + forest + HDC → dict tipado.
2. SwarmCoordinator: adiciona `brain_hints: Optional[Tensor]` no forward;
   se None → comportamento atual. Se dado → concat na cross-attention.
3. Config flag `enable_brain` controla ligar/desligar a feature por
   experimento de A/B.

Exit: walk-forward A/B com mesma semente. Métrica: Sharpe out-of-sample +
hit rate por ativo. Brain ON deve subir ou empatar — nunca degradar
(senão desliga via config).

### Fase 5 — Backtest acelerado

1. Backtest replay via WAL (já pronto da Fase 0).
2. HDC cache: muitas janelas no backtest se repetem (alta similaridade) →
   cache de decisão por HV-key dá 10-100× speedup vs re-inferência.
3. Replay determinístico: trace_id por trade, P&L reproduzível bit-a-bit.

Exit: `backtest_v9.py` com `--use-brain --replay-wal` roda em <5min para
12 meses de dados (vs ~hora atual).

### Fase 6 (futuro) — Federação

Mesmo padrão WAL union + KG merge do byte-language-lab. Múltiplas
instâncias saci compartilham KG e memory bank. Notebook ocioso → harvest
backtests → manda triples/análogos novos. Out of scope inicial.

## Métricas de validação

Cada fase tem **gate** numérico antes de ser declarada pronta:

| Fase | Métrica | Threshold |
|------|---------|-----------|
| 0 | replay determinístico | bit-exato vs raw |
| 1 | triples corretos | spot-check 20 pairs vs rolling corr |
| 2 | propagation accuracy | r² ≥ 0.6 vs regression next-bar |
| 3 | analogy hit rate | distribuição cobre realidade ≥ 60% |
| 4 | brain hints helpful | Sharpe OOS ≥ baseline + 0.1 |
| 5 | backtest speedup | ≥ 10× vs re-inferência |

## Riscos

```
REGIME CHANGE     correlações quebram (BTC vs DXY pré/pós 2024).
                  mitigação: decay exponencial + janela rolling curta,
                  /purge_noise das triples obsoletas.
DATA LEAKAGE      HDC retrieve pode pegar janela do FUTURO em backtest.
                  mitigação: hard cap por timestamp em retrieve.
OVERFIT REGIME    se brain hint domina demais, perde nos regimes novos.
                  mitigação: peso do hint configurável + held-out gate.
LATÊNCIA          retrieve em N memory bank pode demorar. profile, cap N.
```

## O que NÃO muda no saci

```
✓ pipeline de dados (data loaders, indicators)
✓ LightningWorkers por timeframe
✓ SwarmLoss multi-objetivo
✓ AMP / Triton kernels / gradient clipping
✓ rollback existente
✓ docker / mlops
```

Brain é **adição cirúrgica** num único ponto: o input do SwarmCoordinator.

## Cronograma realista

- Fase 0 + 1: 1-2 dias (KG + WAL são simples).
- Fase 2: 1-2 dias (matriz de corr já existe internamente).
- Fase 3: 2-3 dias (HDC encoding + memory bank + retrieve).
- Fase 4: 2-3 dias (integração com SwarmCoordinator + A/B).
- Fase 5: 1-2 dias.
- **Total: ~2 semanas** para o módulo completo, fase por fase com gate.

## Decisão pendente

- **Linguagem do módulo:** Python (cabe no saci atual) ou Rust path (mesma
  base do byte-language-lab brain, mas saci é Python — atritar).
  → Recomendo **Python**. Saci é Python-puro; sem atrito.
- **Storage do KG/memory:** parquet (com pyarrow) vs sqlite vs próprio
  binário. → **parquet** pra columnar + queries rápidas.
