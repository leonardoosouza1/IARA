# Super inteligência financeira — roadmap e brainstorm

> **Tese:** super inteligência financeira **não é mais um sinal de direção** — é a
> **fusão de muitas lentes ortogonais** num *estimador de estado de mercado*. O modelo
> direcional responde "pra que lado, dado o estado?". A inteligência responde "**em que
> estado o mercado está, e quanto isso vale**". Direção é quase imprevisível
> (random-walk, MARKET_INTELLIGENCE §5A); o que é previsível e explorável é **amplitude,
> regime e risco**.

## O vetor de estado de mercado (objetivo)
Fundir as lentes em poucos eixos ortogonais que condicionam a estratégia:

| eixo | mede | previsível? | uso |
|---|---|---|---|
| **amplitude** | regime de volatilidade | ✅ sim (persiste, §5B) | quando operar / sizing |
| **co-movimento** | regime de correlação | parcial | risco de carteira |
| **persistência** | trend vs chop (Hurst, VR) | parcial | viés trend/revert |
| **cauda** | risco de crash, tail-dependence | regime-dependente | stop / sizing defensivo |
| **liquidez** | volume / illiquidez | ✅ sazonal | custo / slippage esperado |
| **fluxo de info** | quem lidera quem (spillover) | a testar | qual ativo armar primeiro |

Saída: um *market-state* → gate (operar?), bias (lado), size (quanto), asset (qual).

---

## Espaço de análises (taxonomia + status)

### A. Estrutura de mercado
- [x] Rede de correlação (blocos) — MARKET_INTELLIGENCE §1
- [x] Fator PCA (PC1 64.5%) — §2
- [x] Correlação rolante (regime) — §3
- [ ] **Random Matrix Theory** (Marchenko-Pastur): quais correlações são sinal vs ruído
- [ ] **MST / topologia de rede**: árvore geradora mínima dos ativos
- [ ] Hierarchical Risk Parity (sizing via estrutura de correlação)

### B. Estrutura temporal
- [x] Autocorr + variance-ratio (random-walk) — §5A
- [x] Persistência de volatilidade — §5B
- [x] Relógio de mercado (hora) — §5C
- [ ] **Hurst exponent** por ativo/escala (memória longa, trend vs revert)
- [x] **Regime-switching (HMM)** — MARKET_INTELLIGENCE §7: 3 estados (calm/normal/stress) persistentes (3.6-4.8d), não teleportam; edge monotônico calm +0.29→stress +1.02%/tr
- [ ] Sazonalidade: dia-da-semana, mês, ciclo de halving

### C. Risco de cauda e extremos
- [x] **Distribuição de retornos** — MARKET_INTELLIGENCE §8: fat tails (curtose 3.6-10.8)
- [x] **Drawdown** — §8: maxDD −71 a −96%, submerso 2.3-3.8 anos
- [x] **VaR / CVaR** — §8: CVaR99 diário −11 a −19%
- [x] **Tail dependence** — §8: λ=0.61 = 12× independência; crashes co-ocorrem
- [ ] Detecção de jumps (bipower variation): vol contínua vs saltos

### D. Cross-asset e fluxo de informação
- [x] Lead-lag retorno crypto-crypto (ausente) — CROSS_ASSET
- [x] USD/FX (fraco) — CROSS_ASSET §5
- [x] **Spillover de volatilidade** — MARKET_INTELLIGENCE §6: vol spilla (cross-lag ~0.2) mas simétrico e **redundante** (cross-asset add ~0 OOS); vol também é ~1 fator
- [ ] **Granger causality network** entre ativos
- [ ] **Transfer entropy** (fluxo de info não-linear)

### E. Inteligência de edge (modelo)
- [x] Edge por regime (vol domina) — MARKET_INTELLIGENCE §4
- [x] Gate de vol causal validado walk-forward — §5 payoff
- [ ] **Calibração de confiança**: conf 0.7 → 70% win? (reliability diagram)
- [ ] **Alpha decay**: o edge está encolhendo no tempo?
- [ ] **Edge condicional**: edge | (vol-state × corr-state × trend-state)
- [ ] Edge vs horizonte de holding

### F. Microestrutura (limitado a OHLCV)
- [ ] Relação volume-preço (volume confirma movimento?)
- [ ] Amihud illiquidity proxy
- [ ] Eficiência intrabar (|close-open| / (high-low))
- [ ] Volatility signature plot (vol realizada vs frequência de amostragem)

### G. Macro/externo (limitado por dados)
- [x] Força do dólar (FX) — CROSS_ASSET §5
- [ ] BTC dominance, fluxo de stablecoin — sem dados
- [ ] Cross-market (equities/ouro) — sem dados

---

## Ordem de execução (valor × viabilidade com OHLCV)
1. **Spillover de volatilidade** (D) — extensão da intuição original do Leonardo no
   domínio certo: retorno não lidera, mas vol pode → alerta acionável real.
2. **Regime-switching HMM** (B) — unifica os regimes em estados latentes descobríveis.
3. **Risco de cauda + drawdown + tail-dependence** (C) — o eixo de risco do vetor.
4. **Calibração + alpha decay** (E) — saúde do edge no tempo.
5. **Hurst + RMT** (A/B) — rigor estrutural.

> Honestidade que guia tudo: a maioria das lentes é **descritiva/risco**, poucas são
> **preditivas**. A previsível de verdade é a amplitude (vol). A "super inteligência"
> não é um oráculo de direção — é saber **quando, quanto e com que risco** operar o
> fator-cripto que é, em direção, essencialmente imprevisível.
