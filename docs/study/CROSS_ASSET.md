# Cross-asset: a relação existe, mas NÃO é um alerta preditivo (M5)

Pergunta: um movimento no BTC **prevê** movimento nas alts (ETH/SOL/BNB), o que
permitiria um alerta cross-asset tradeable ("BTC/USD cai → alerta de baixa nas
outras")? Medido em 505k barras M5 alinhadas (2021-2026), `scripts/cross_asset.py`.

## 1. Correlação contemporânea — FORTE
|  | BTC | ETH | SOL | BNB |
|---|---|---|---|---|
| BTC | 1.00 | 0.75 | 0.62 | 0.67 |
| ETH | 0.75 | 1.00 | 0.67 | 0.73 |
| SOL | 0.62 | 0.67 | 1.00 | 0.65 |
| BNB | 0.67 | 0.73 | 0.65 | 1.00 |

Os coins **andam juntos**. A relação que o Leonardo intuiu existe.

## 2. Lead-lag — BTC NÃO lidera
`corr(BTC_ret[t], alt_ret[t+k])`:

| asset | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|---|
| ETH | +0.746 | -0.015 | -0.013 | +0.019 | -0.002 | -0.011 | -0.016 |
| SOL | +0.624 | -0.020 | -0.014 | +0.001 | -0.001 | -0.006 | -0.021 |
| BNB | +0.666 | -0.018 | -0.009 | +0.012 | +0.001 | -0.013 | -0.024 |

**Toda a correlação está em k=0 (mesma barra). Em k≥1 ≈ zero.** O acoplamento é
**instantâneo**, não preditivo. Um alerta "BTC caiu → shorta a alt" dispararia
DEPOIS da alt já ter caído no mesmo candle → **sem edge tradeable por lag**.

Confirmado em **1h e 4h** também (não só M5): k=0 fica até mais forte (ETH 0.84),
k≥1 segue ≈ zero. **BTC não lidera alts em NENHUMA cadência** testada (5m/1h/4h) —
co-movimento instantâneo em toda resolução. Conclusão robusta.

## 3. Resposta a choque — mean-reversion, não continuação
Após |ret BTC| > p99 (>0.67%), retorno médio das próximas 12 barras:

| asset | após choque ALTA | após choque BAIXA |
|---|---|---|
| BTC | -0.046% | +0.174% |
| ETH | +0.072% | +0.169% |
| SOL | +0.119% | **+0.523%** |
| BNB | +0.107% | +0.208% |

Depois de uma **queda** forte do BTC, tudo tende a **subir** (bounce), SOL forte.
A tese "BTC cai → alts caem" **não se sustenta** nesse horizonte — elas quicam.
Sinal pós-choque é **reversão**, não pânico em cadeia.

## Conclusão e uso realista
- ❌ **Alerta lead-lag** ("leader prevê followers") **não existe** no M5 — mercado
  eficiente o bastante pra acoplar instantaneamente.
- ✅ **Confirmação contemporânea**: como os coins se movem juntos (0.62-0.75), o
  consenso entre eles na MESMA barra pode aumentar a convicção do modelo (operar
  só quando BTC e a alt sinalizam a mesma direção). Não é alerta, é filtro de
  confiança.
- ✅ **Reversão pós-choque** (SOL +0.52% após queda BTC) é um sinal próprio
  potencialmente explorável — mean-reversion, separado do modelo direcional atual.

## 4. Backtest da reversão pós-choque — FRACO após custos
`scripts/shock_reversion.py` (down-shock BTC → long alt, hold 12, custo 0.14%,
non-overlap, full history):

| alt | trades | net-win | net/trade | total | sharpe | buyhold |
|---|---|---|---|---|---|---|
| ETH | 1190 | 51.2% | -0.057% | -6.7% | -0.44 | +14.9% |
| SOL | 1190 | 53.8% | +0.171% | +22.1% | +0.99 | +482.8% |
| BNB | 1190 | 51.9% | -0.053% | -6.3% | -0.42 | +137.1% |

Só **SOL** dá edge marginal (sharpe ~1); ETH/BNB break-even-a-negativo. O bounce
do raw existe mas **não bate custo de forma robusta**. Both-sides piora tudo.
Conclusão: reversão pós-choque **não é edge confiável** isolada — bem mais fraca
que o modelo direcional (que tem net-win 80%). Arquivado.

## 5. Moedas fortes (USD) vs crypto — `scripts/fx_crypto.py`
Tese do Leonardo: "dólar forte → crypto cai". USD strength = -ret(EUR/USD)
(proxy do DXY, ~58% EUR). FX da Binance: EURUSDT tem histórico completo 2021-2026
(GBP/AUD só até 2023).

| janela | USD_STR↔BTC | USD_STR↔ETH |
|---|---|---|
| 2021-2023 (EUR+GBP+AUD, 1h) | -0.181 | -0.157 |
| 2021-2026 (EUR, 1h) | -0.106 | -0.096 |
| 2021-2026 (EUR, 4h) | -0.123 | -0.105 |

- ✅ **Tese direcionalmente certa**: correlação NEGATIVA (dólar forte ↔ crypto fraco).
- ❌ Mas **fraca** (-0.10 a -0.18, mais forte na era 2021-23 USD-bull/crypto-bear) e
  **sem lead-lag** (k≥1 ≈ zero). USD se move junto com crypto, não o antecede.
- Pós-choque USD-up nem empurra crypto pra baixo de forma confiável (BTC +0.12%).

**Não é alerta tradeable** — mesmo padrão de tudo: co-movimento real porém instantâneo.

## 6. Filtro de confirmação contemporânea — SEM lift
`pnl_backtest --confirm_leader BTCUSDM5`: operar a alt só quando a direção recente
do BTC concorda com o trade. 3-coin model, 2026:

| alt | sem confirm | com confirm BTC |
|---|---|---|
| ETH | 87.8% net-win, +0.578%/tr (90) | 85.5%, +0.584%/tr (76) |
| SOL | 88.7%, +0.582%/tr (229) | 88.4%, +0.529%/tr (172) |
| BNB | 87.0%, +0.336%/tr (246) | 87.4%, +0.329%/tr (183) |

Confirmação remove ~25% dos trades mas **não melhora** net-win nem net/trade (às
vezes pior). **O modelo já captura o co-movimento** (features scale-invariant +
alts 0.84 corr BTC). Confirmação explícita = redundante.

## Conclusão geral cross-asset (3 ângulos, todos testados)
1. **Alerta lead-lag** ("leader prevê follower") → **não existe** (acoplamento
   instantâneo em 5m/1h/4h, crypto-crypto E crypto-USD).
2. **Reversão pós-choque** → **fraca** (só SOL marginal, sharpe ~1).
3. **Filtro de confirmação** → **sem lift** (modelo já sabe).

As relações cross-asset (crypto-crypto 0.62-0.84, crypto-USD -0.10 a -0.18) são
REAIS porém **já capturadas pelo modelo single-coin / precificadas pelo mercado**.
Nenhum edge adicional de sinal cross-asset explícito. A "rede que comunica" já
existe **implicitamente**: um modelo, features scale-invariant, generaliza entre
moedas (incl. não-treinadas — ver RESULTS Phase 6). Não vale super-construir infra
de alerta cross-asset. Insight valioso e honesto: poupa esforço.
