#!/bin/bash
# Script completo de treinamento e validação

echo "🚀 TREINAMENTO COMPLETO DO PIPELINE NBA PREDICTOR"
echo "=================================================="
echo ""

# 1. Treinar Calibrador
echo "1️⃣ Treinando calibrador..."
python scripts/train_calibrator.py --lookback 60 --min-samples 50

if [ $? -ne 0 ]; then
    echo "❌ Erro no treinamento do calibrador"
    exit 1
fi

echo ""
echo "=================================================="
echo ""

# 2. Validar Pipeline
echo "2️⃣ Validando pipeline..."
python scripts/validate_pipeline.py

if [ $? -ne 0 ]; then
    echo "❌ Erro na validação"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ TREINAMENTO E VALIDAÇÃO CONCLUÍDOS!"
echo "=================================================="
