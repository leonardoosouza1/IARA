# Inteligência de mercado: visões diferentes sobre os mesmos dados

Além do modelo direcional (que diz *quando* operar), esta camada caracteriza a
**estrutura do mercado** — como os ativos se relacionam, o que os move, e quando
essas relações quebram. Três lentes, todas em retornos log de 1h, 2021-2026.
`scripts/market_intel.py` (43.057 barras 1h alinhadas BTC/ETH/SOL/BNB/EURUSD).

---

## Lente 1 — Rede de correlação: dois blocos ortogonais

| corr 1h | BTC | ETH | SOL | BNB | EURUSD | GBPUSD |
|---|---|---|---|---|---|---|
| BTC | 1.00 | 0.84 | 0.69 | 0.74 | 0.11 | 0.14 |
| ETH | 0.84 | 1.00 | 0.71 | 0.76 | 0.10 | 0.12 |
| SOL | 0.69 | 0.71 | 1.00 | 0.66 | 0.07 | 0.09 |
| BNB | 0.74 | 0.76 | 0.66 | 1.00 | 0.07 | 0.09 |
| EURUSD | 0.11 | 0.10 | 0.07 | 0.07 | 1.00 | 0.44 |
| GBPUSD | 0.14 | 0.12 | 0.09 | 0.09 | 0.44 | 1.00 |

Clustering por distância-de-correlação acha **2 blocos limpos**:
- **Bloco cripto**: BTC-ETH-SOL-BNB (0.66-0.84 entre si)
- **Bloco FX**: EURUSD-GBPUSD (0.44 entre si)

Cripto e FX são **quase ortogonais em 1h** (corr 0.07-0.14). São dois mercados que
mal se conversam nessa cadência. (Ver `plots/corr_network.png`.)

---

## Lente 2 — Fator de mercado (PCA): cripto é UMA aposta, não quatro

| | PC1 | PC2 | PC3 | PC4 | PC5 |
|---|---|---|---|---|---|
| % variância | **64.5%** | 19.8% | 7.1% | 5.6% | 3.1% |

- **PC1 explica 64.5% de TODA a variância** e carrega quase igualmente nas 4 criptos
  (BTC −0.51, ETH −0.52, SOL −0.47, BNB −0.49), EURUSD ~0. É o **"beta-cripto"**:
  um único fator move 2/3 de tudo.
- **PC2 = 19.8%** carrega só em EURUSD (−1.00). O dólar/EUR é um fator **separado**.

### Implicação dura (gestão de risco)
As 4 criptos **não são 4 apostas independentes** — são **~1 aposta** (o fator cripto)
+ um resíduo idiossincrático pequeno. Operar BTC+ETH+SOL+BNB ao mesmo tempo ≈
**4× alavancagem no MESMO fator**, não diversificação. O sizing tem que tratar a
carteira cripto como ~1 posição de risco. É também **porque o modelo single-coin
generaliza** (CROSS_ASSET, RESULTS Phase 6): ele aprende o fator dominante, que é
comum a todas as moedas → transfere pra moedas não-treinadas.

---

## Lente 3 — Regime rolante: diversificação some no stress

Correlação média rolante (30d) entre as 4 criptos ao longo do tempo:
- média **0.74**, varia de **0.53** (calmo, moedas com vida própria) a **0.90** (stress).

Picos de correlação (>0.82, +1σ) = **risk-off / pânico**: tudo despenca junto, a
correlação intra-cripto satura perto de 1, e a (falsa) diversificação **evapora
exatamente quando seria necessária**. Padrão clássico de mercado de risco.
(Ver `plots/rolling_corr.png`.)

---

## Síntese — o que a inteligência de mercado adiciona ao projeto

1. **O mercado cripto é essencialmente 1-dimensional em 1h** (PC1 64.5%). Isso unifica
   achados anteriores: co-movimento instantâneo (CROSS_ASSET), modelo single-coin
   que generaliza (RESULTS Phase 6), confirmação cross-asset redundante — tudo são
   sintomas do mesmo fator único.
2. **FX é um eixo de risco separado** (ortogonal a cripto, PC2 próprio). A tese
   "dólar forte → cripto cai" é fraca (CROSS_ASSET §5) justamente porque vivem em
   fatores diferentes.
3. **Regime importa**: correlação não é constante (0.53→0.90). A Lente 4 liga isso
   ao modelo.

---

## Lente 4 — Edge do modelo por regime: volatilidade domina, correlação não

`scripts/regime_edge.py` rotula cada um dos 789 trades 2026 com o regime de
correlação cripto vigente (rolling 30d) na entrada e o regime de volatilidade (ATR).

**Por regime de correlação** (2026 cobre só corr 0.75-0.86 — ano de alta-corr):
| regime | n | net-win | net/trade |
|---|---|---|---|
| low (desacoplado) | 262 | 82.1% | +0.424% |
| mid | 484 | 88.2% | +0.416% |
| high (risk-off) | 43 | 93.0% | +0.476% |

**Por volatilidade (ATR tercil)** — reconfirma o lever:
| regime | n | net-win | net/trade |
|---|---|---|---|
| low | 261 | 78.2% | +0.140% |
| mid | 260 | 90.8% | +0.386% |
| **high** | 268 | 90.3% | **+0.732%** |

**Interação corr × vol** (median splits): o ganho vem de **vol alta** em qualquer
regime de correlação (corr-lo/vol-hi +0.651% com n=362; corr-hi tem só 11-32 trades,
amostra fraca).

**Veredito honesto:** a **volatilidade é o regime que importa** (+0.14 → +0.73, 5×);
o **regime de correlação adiciona pouco** em 2026 (+0.42 → +0.48). Além disso 2026 só
cobriu a banda de corr alta (0.75-0.86), então não dá pra testar regime de baixa-corr
sem trades de outros anos (walk-forward multi-ano). **Resultado negativo útil**: não
vale construir um filtro de regime-de-correlação; o sinal de risco da Lente 3 serve
pra *sizing/risco*, não pra timing de entrada. Volatilidade segue sendo o lever
principal (consistente com EDGE_FINDER §1 e §5).

> Honestidade: PCA/correlação descrevem estrutura, não preveem. São lentes de
> *entendimento e risco*, não sinais de entrada. O valor é saber que a carteira é
> menos diversificada do que parece (PC1 64.5%) e que o regime de correlação muda
> (0.53→0.90) — mas o que move o edge do modelo é a **volatilidade**, não a correlação.

---

## Lente 5 — Estrutura temporal: preço é random-walk, volatilidade é previsível

`scripts/return_structure.py`, retornos log 1h, histórico completo. Esta é a lente
que **amarra tudo**. (Ver `plots/return_structure.png`.)

### A. Retorno é quase um random-walk
Autocorrelação dos retornos 1h, lags 1-6h: **~0 em tudo** (±0.03). Variance-ratio
(q=2..48h) entre **0.86 e 1.01** — reversão fraquíssima, essencialmente passeio
aleatório.

→ **O edge NÃO vem de timing linear de preço** (não existe estrutura linear pra
explorar). Vem da seleção não-linear por confiança sobre o estado dos indicadores.
Isso explica, honestamente, por que o edge direcional bruto é fino (~56%): o sinal
linear é zero; o ganho está em *quando* o modelo tem confiança, não numa
previsibilidade do preço.

### B. Volatilidade é altamente previsível — o achado acionável
Autocorrelação de **|retorno|** (proxy de volatilidade):

| asset | 1h | 6h | 24h | 72h | 168h (1 sem) |
|---|---|---|---|---|---|
| BTC | +0.29 | +0.18 | +0.18 | +0.13 | +0.15 |
| SOL | +0.31 | +0.24 | +0.22 | +0.15 | +0.13 |
| BNB | +0.34 | +0.26 | +0.23 | +0.17 | +0.16 |

Volatilidade **clusteriza e persiste por semanas** (efeito GARCH clássico). No
diário: **P(alta-vol amanhã | alta-vol hoje) = 0.64-0.69** vs base 0.33 → **2× lift**.

→ Como o edge do modelo **escala com volatilidade** (Lente 4, EDGE_FINDER §1) e a
volatilidade é **previsível**, o regime de alto-edge é **antecipável**. Diferente de
prever direção (impossível, §A), prever *atividade* do mercado é viável e robusto.
Filtro forward-looking honesto: **estar ativo após dias de alta-vol e na sessão US** —
não é overfit, é fato estilizado de mercado.

### C. Relógio de mercado
Volatilidade do BTC por hora UTC tem pico claro em **13-15h (abertura US)**, vale em
04-05h, bump secundário 00-01h (Ásia). Confirma EDGE_FINDER §2 **direto do dado bruto**
(não só dos trades) — o efeito de sessão é uma propriedade do mercado, não artefato.

### Payoff validado — gate de volatilidade CAUSAL (`scripts/vol_gate.py`)
A afirmação "vol previsível → regime de alto-edge antecipável" foi testada como filtro
**deployável**: operar só quando a vol realizada das **24h anteriores** à entrada
(conhecida *antes* da barra, sem lookahead) está acima da mediana expandente. Contrasta
com o ATR do próprio candle do EDGE_FINDER (contemporâneo). 789 trades 2026:

| gate | n (pooled) | net-win | net/trade | Δ vs base |
|---|---|---|---|---|
| baseline | 789 (100%) | 86.4% | +0.422% | — |
| **vol_fc (causal)** | 191 (24%) | 89.0% | **+0.592%** | **+0.170pp** |
| US session | 258 (33%) | 88.4% | +0.568% | +0.146pp |
| both | 68 (9%) | 89.7% | +0.687% | +0.266pp |

O gate causal **levanta net/trade nas 4 moedas** (BNB +0.15, BTC +0.07, ETH +0.39,
SOL +0.32) — o forecast de volatilidade funciona forward-looking, não é hindsight.
**Caveat honesto:** melhora a *qualidade por trade*, não o lucro total (corta ~76% dos
trades → total cai de +333 pra +113). Serve pra sizing risk-adjusted / capital
limitado, não pra maximizar retorno absoluto. Só 2026 (um regime) — validar
walk-forward multi-ano antes de confiar.

### Síntese das 5 lentes
> **Não dá pra prever a direção do preço (random-walk), mas dá pra prever QUANDO o
> mercado fica ativo (volatilidade persiste, concentra na sessão US) — e é exatamente
> aí que o edge do modelo é maior.** A inteligência de mercado acionável não é um sinal
> de direção extra; é um *forecast de volatilidade/atividade* que diz quando ligar o
> modelo e quanto arriscar. O mercado cripto é ~1 fator (PC1 64.5%), esse fator é
> imprevisível em direção mas previsível em amplitude. O gate de vol causal **prova**
> que a inteligência é deployável (+0.17pp net/trade, sem lookahead).
