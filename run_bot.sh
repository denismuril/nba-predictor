#!/bin/bash

# Definir diretório do projeto
PROJECT_DIR="/home/denis/nba-predictor"
LOG_FILE="$PROJECT_DIR/logs/bot_run_$(date +%Y-%m-%d).log"

# Navegar para o diretório
cd "$PROJECT_DIR" || exit

# Ativar ambiente virtual
source venv/bin/activate

echo "🚀 Iniciando NBA Tigrinho Bot: $(date)" >> "$LOG_FILE"

# Rodar Bot
python telegram_bot/nba_tigrinho_bot.py >> "$LOG_FILE" 2>&1 &

echo "✅ Bot iniciado em background. Logs em: $LOG_FILE"
