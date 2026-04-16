# PHASE 1A: Sniper Mode Convergence Strategy
## Objective: Redefining the Probability Threshold for Alpha Signals

Este documento detalha o refino da lógica de inferência do `TrinityEngine` e do `SignalEngine` para atingir o nível "Sniper" (Alta Precisão).

---

### 1. Critérios de Convergência de Sinal

O sinal só será disparado se a convergência dos seguintes fatores for atingida:

*   **LNN (Liquid Brain):** Confiança baseada em `CfCCell` > 0.85 (Ajustado de 0.82).
*   **VBI (Volume Book Imbalance):** Desequilíbrio no Order Book > 0.70 (Indicando exaustão de um lado e agressão oculta do outro).
*   **CVD (Cumulative Volume Delta):** Divergência confirmada (Ex: Preço cai, mas CVD sobe agressivamente = Acumulação).
*   **Hurst Exponent:** Valor > 0.65 (Confirmando regime de tendência clara, não ruído).

---

### 2. Modificações de Código Sugeridas

*   **`backend/src/trading/signal_generator.py`:** Adicionar a função `validate_sniper_conditions` que recebe os tensores da rede e os dados sensoriais (Raio-X) e retorna um booleano de alta confiança.
*   **`backend/src/brain/engine.py`:** Alterar a função `forward_tick` para que, se os critérios Sniper forem atingidos, a ordem seja marcada como `EXC_SNIPER` no banco de dados.

---

### 3. Validação (Dojo de Testes)

*   Rodar o script `backtester.py` com o filtro Sniper ativado.
*   Comparar a taxa de acerto (Win Rate) com o modo normal. O objetivo é reduzir o número de trades, mas aumentar drasticamente a assertividade.

---
*Status: Aguardando Implementação Técnica*
