#!/bin/bash
set -e

# Configurações Otimizadas (SACI-V11 Master Training)
DAYS=60
TIMEFRAME="5Min"
CHECKPOINT_DIR="checkpoints_top20"
BATCH_SIZE=256 # Seguro para GPUs de 8GB-12GB
LR=0.0005      # Mais conservador para multi-ativo
EPOCHS=300
PATIENCE=100

mkdir -p $CHECKPOINT_DIR
mkdir -p data/raw
mkdir -p data/processed

echo "🚀 Iniciando Preparação para Treinamento Master (Top 20 Ativos)"

# 1. Download Crypto (Top 10)
CRYPTO_SYMBOLS=("BTC/USD" "ETH/USD" "SOL/USD" "BNB/USD" "XRP/USD" "ADA/USD" "DOGE/USD" "AVAX/USD" "TRX/USD" "LINK/USD")
for sym in "${CRYPTO_SYMBOLS[@]}"; do
    echo "⬇️ Baixando Crypto: $sym"
    # Adicionando verificação para pular se já existir e for recente (opcional)
    python backend/src/data/crypto_data_fetcher.py --symbol "$sym" --days $DAYS --timeframe $TIMEFRAME --output-dir data/processed
done

# 2. Download Forex (Top 10)
echo "⬇️ Baixando Forex (Top 10)..."
python - <<EOF
import asyncio
from backend.src.data.downloader import run_ingestion
FOREX_TOP10 = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", 
    "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"
]
asyncio.run(run_ingestion(tickers=FOREX_TOP10, period="60d", interval="5m"))
EOF

# 3. Processar Features (Feature Factory)
echo "⚙️ Gerando features e construindo cache..."
python backend/src/data/feature_factory.py

# 4. Iniciar Treinamento Master
echo "🧠 Treinando Modelo Master (Frequência: $TIMEFRAME | Ativos: 20)"
echo "Config: Batch=$BATCH_SIZE, LR=$LR, Epochs=$EPOCHS"

# PYTHONPATH=. garante que os pacotes locais sejam encontrados
PYTHONPATH=. python backend/src/train/trainer_liquid.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --patience $PATIENCE \
    --lr $LR \
    --use-cache \
    --checkpoint-dir $CHECKPOINT_DIR

echo "✅ Treinamento finalizado! Confira os resultados em $CHECKPOINT_DIR/best_model.pth"
