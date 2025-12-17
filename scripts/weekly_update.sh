#!/bin/bash
# Script de atualização semanal do NBA Predictor
# Roda toda segunda às 10h via cron

LOG_FILE="/home/denis/nba-predictor/logs/weekly_update.log"
PROJECT_DIR="/home/denis/nba-predictor"

cd "$PROJECT_DIR"
source venv/bin/activate

echo "========================================" >> "$LOG_FILE"
echo "$(date): Iniciando atualização semanal" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# 1. Limpar cache antigo
echo "$(date): Limpando cache..." >> "$LOG_FILE"
rm -rf data/cache/

# 2. Preparar novo cache
echo "$(date): Preparando novo cache (isso demora ~8h)..." >> "$LOG_FILE"
python scripts/prepare_data_cache.py >> "$LOG_FILE" 2>&1

# 3. Retreinar ensemble
echo "$(date): Retreinando ensemble..." >> "$LOG_FILE"
python -c "from ml_pipeline.ensemble_blending import train_ensemble_blending; train_ensemble_blending(optimize_hyperparams=False)" >> "$LOG_FILE" 2>&1

echo "$(date): Atualização concluída!" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
