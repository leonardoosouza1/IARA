#!/bin/bash
set -e

echo "🚀 Iniciando Saci-Nano Pipeline..."

# 1. Data Ingestion
echo "⬇️  [1/4] Baixando dados Forex (30+ pares)..."
docker compose exec saci-brain python src/data/downloader.py

# 2. Feature Engineering
echo "⚙️  [2/4] Gerando indicadores técnicos..."
docker compose exec saci-brain python src/data/processor.py

# 3. Tokenization
echo "🔤 [3/4] Tokenizando dados (Float -> Token IDs)..."
docker compose exec saci-brain python src/data/tokenizer.py

# 4. Training
echo "🧠 [4/4] Iniciando treinamento do Saci-Nano..."
docker compose exec saci-brain python src/train/trainer.py

echo "✅ Pipeline finalizado! Checkpoints em ./checkpoints"
