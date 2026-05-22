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

Próximo (se valer): testar se o filtro de confirmação contemporânea melhora o
net-win do modelo; e um backtest da reversão pós-choque. Ambos com walk-forward.
