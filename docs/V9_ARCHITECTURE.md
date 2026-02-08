# I.A.R.A. v9 "Swarm Intelligence" Architecture

## Overview

A v9 representa uma mudança paradigmática: de um **modelo único multi-timeframe** para um **enxame de especialistas coordenados**.

```
                    ┌─────────────────────────────────────┐
                    │      COORDINATOR (Main Brain)       │
                    │   Liquid CfC (256 units) + Experts  │
                    └─────────────────▲───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │    CONSENSUS FUSION LAYER           │
                    │    (Cross-Attention over Workers)   │
                    └─────────────────▲───────────────────┘
                                      │
       ┌──────────┬──────────┬────────┴────────┬──────────┬──────────┐
       │          │          │                 │          │          │
   ┌───▼───┐  ┌───▼───┐  ┌───▼───┐         ┌───▼───┐  ┌───▼───┐  ┌───▼───┐
   │ W-M5  │  │ W-M15 │  │ W-M30 │         │ W-H1  │  │ W-H4  │  │  LLM  │
   │       │  │       │  │       │         │       │  │       │  │Regime │
   │ 32dim │  │ 32dim │  │ 32dim │         │ 32dim │  │ 32dim │  │(ext)  │
   └───────┘  └───────┘  └───────┘         └───────┘  └───────┘  └───────┘
   Ultra-Light Workers (~100KB each)        Optional    External
```

---

## Design Philosophy

### Por que Workers separados?

1. **Especialização**: Cada timeframe tem características únicas
   - M5: Ruído alto, reversões rápidas
   - H1: Tendências claras, momentum
   - H4: Contexto macro, suporte/resistência chave

2. **Resiliência**: Se M5 "ficar louco" com ruído, H1/H4 mantêm sanidade

3. **Eficiência**: Workers ultra-leves (~32 units) são 10x mais rápidos que modelo único grande

4. **Interpretabilidade**: Podemos visualizar "quem discorda de quem"

---

## Components

### 1. LightningWorker (Ultra-Light Liquid Model)

```python
class LightningWorker(nn.Module):
    """
    Especialista para um único timeframe.
    Usa CfC compacto com quantização opcional.
    
    Input:  [batch, seq_len, 12 features]
    Output: 
      - embedding: [batch, 32]  (representação aprendida)
      - signal: [batch, 3]      (softmax: LONG/SHORT/WAIT confidence)
    """
    
    def __init__(self, timeframe: str, hidden_dim: int = 32):
        self.timeframe = timeframe  # "M5", "M15", etc.
        self.cfc = CfCCell(input_dim=12, hidden_dim=32)
        self.head = nn.Linear(32, 3)  # LONG/SHORT/WAIT
```

**VRAM por Worker (int8):**
- Parâmetros: ~20KB
- Activations: ~50KB
- **Total: ~70KB**

### 2. SwarmCoordinator (Main Brain)

```python
class SwarmCoordinator(nn.Module):
    """
    Cérebro central que analisa consenso entre workers.
    
    Input: 
      - worker_embeddings: [batch, 5, 32]  (5 workers x 32 dims)
      - worker_signals: [batch, 5, 3]      (5 workers x 3 classes)
    
    Output:
      - final_logit: [batch, 1]  (probabilidade de compra)
    """
    
    def __init__(self, num_workers: int = 5):
        # Cross-Attention: Qual worker está mais "confiante"?
        self.cross_attn = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        
        # Fusion: Concatena tudo
        self.fusion = nn.Linear(5 * 32 + 5 * 3, 128)
        
        # CfC Principal
        self.cfc = CfCCell(input_dim=128, hidden_dim=128)
        
        # Expert Heads
        self.long_expert = nn.Linear(128, 1)
        self.short_expert = nn.Linear(128, 1)
```

### 3. Technical Analyzer (Pattern Detection)

```python
class TechnicalAnalyzer:
    """
    Detecta padrões técnicos para enriquecer features.
    
    Features geradas:
      - trend_slope: Inclinação da linha de tendência
      - dist_to_support: Distância ao suporte mais próximo
      - dist_to_resistance: Distância à resistência mais próxima
      - pattern_score: Pin bar, Engulfing, Doji detection
    """
    
    def detect_trend(self, highs, lows, closes) -> Tensor:
        # Linear regression over last N candles
        ...
    
    def find_sr_levels(self, highs, lows) -> Tuple[Tensor, Tensor]:
        # Pivot point detection
        ...
```

---

## Data Pipeline

### Input Features per Timeframe (12 dims)

| Índice | Feature | Descrição |
|--------|---------|-----------|
| 0 | `rsi_norm` | RSI normalizado [-1, 1] |
| 1 | `macd_norm` | MACD normalizado |
| 2 | `bb_pct` | Posição nas Bollinger Bands |
| 3 | `atr_norm` | Volatilidade normalizada |
| 4 | `adx_norm` | Força da tendência |
| 5 | `stoch_k` | Stochastic K |
| 6 | `dist_sma20` | Distância à SMA 20 |
| 7 | `dist_sma200` | Distância à SMA 200 |
| 8 | `trend_slope` | Inclinação da tendência |
| 9 | `sr_position` | Posição relativa S/R |
| 10 | `hour_sin` | Tempo cíclico |
| 11 | `hour_cos` | Tempo cíclico |

---

## Training Strategy

### Loss Function (From v6)

```python
class SwarmLoss(nn.Module):
    """
    Combinação de losses para treinar coordinator + workers.
    """
    def forward(self, coord_output, worker_outputs, targets):
        # 1. Coordinator Loss (Asymmetric Focal)
        main_loss = asymmetric_focal_loss(coord_output, targets)
        
        # 2. Worker Auxiliary Losses (Each worker predicts too)
        aux_loss = sum([
            0.1 * cross_entropy(w_out, targets) 
            for w_out in worker_outputs
        ])
        
        # 3. Consensus Regularization (Penalize extreme disagreement)
        # If all workers say LONG but coordinator says SHORT, penalize
        consensus_loss = 0.05 * disagreement_penalty(worker_outputs, coord_output)
        
        return main_loss + aux_loss + consensus_loss
```

### Training Loop Fixes (From v6)

1. **Label Smoothing**: `smooth=0.1`
2. **pos_weight**: Calculado dinamicamente do dataset
3. **Gradient Clip**: `max_norm=0.5`
4. **LR Schedule**: `ReduceLROnPlateau(factor=0.8, patience=15)`

---

## VRAM Budget (12GB GPU)

| Componente | fp32 | int8 |
|------------|------|------|
| 5 Workers | 10MB | 2.5MB |
| Coordinator | 20MB | 10MB |
| Activations (batch=64) | 50MB | 50MB |
| Gradients | 100MB | 50MB |
| Optimizer States | 200MB | 100MB |
| **Total** | **380MB** | **212MB** |

> ✅ **Cabe facilmente em 12GB!** Podemos dobrar capacity se necessário.

---

## Consensus Logic

```python
def make_decision(worker_signals: Tensor, coordinator_logit: Tensor) -> str:
    """
    Lógica de decisão multi-camada.
    
    worker_signals: [5, 3] - probabilidades LONG/SHORT/WAIT por worker
    coordinator_logit: [1] - logit final do coordinator
    """
    
    # 1. Count worker votes
    worker_votes = worker_signals.argmax(dim=1)  # 0=LONG, 1=SHORT, 2=WAIT
    n_long = (worker_votes == 0).sum()
    n_short = (worker_votes == 1).sum()
    n_wait = (worker_votes == 2).sum()
    
    # 2. Coordinator decision
    coord_prob = torch.sigmoid(coordinator_logit)
    
    # 3. Consensus rules
    if n_wait >= 3:
        return "WAIT"  # Muitos workers incertos
    
    if coord_prob > 0.6 and n_long >= 3:
        return "LONG"  # Coordinator + maioria workers concordam
    
    if coord_prob < 0.4 and n_short >= 3:
        return "SHORT"
    
    # 4. Coordinator overrides if confident enough
    if coord_prob > 0.7:
        return "LONG"
    if coord_prob < 0.3:
        return "SHORT"
    
    return "WAIT"  # Sem consenso claro
```

---

## File Structure

```
src/brain/
├── models/
│   ├── liquid_v9.py      # [NEW] SwarmBrain, SwarmCoordinator
│   ├── workers.py        # [NEW] LightningWorker
│   └── analyzers.py      # [NEW] TechnicalAnalyzer
├── train_v9.py           # [NEW] Training loop corrigido
└── simulate_v9.py        # [NEW] Simulation with swarm
```

---

## Migration Path

### Phase 1: Foundation
1. Criar `LightningWorker` baseado no `CfCCell` da v6
2. Criar `SwarmCoordinator` com cross-attention
3. Portar `AsymmetricFocalLoss` e técnicas da v6

### Phase 2: Training
1. Adaptar `feature_factory.py` para gerar dados por timeframe
2. Criar `train_v9.py` com training loop corrigido
3. Testar convergência

### Phase 3: Operations
1. Integrar com dashboard
2. Adicionar visualização de "votação" dos workers
3. Otimizar para latência (quantização)

---

*Architecture locked for V9 "Swarm Intelligence"*
