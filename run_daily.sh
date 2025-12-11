#!/bin/bash

# Definir diretório do projeto
PROJECT_DIR="/home/denis/nba-predictor"
LOG_FILE="$PROJECT_DIR/logs/daily_run_$(date +%Y-%m-%d).log"

# Navegar para o diretório
cd "$PROJECT_DIR" || exit

# Ativar ambiente virtual
source venv/bin/activate

echo "🚀 Iniciando execução diária: $(date)" >> "$LOG_FILE"

# Rodar Pipeline (CLI)
python interfaces/cli.py --ml >> "$LOG_FILE" 2>&1

echo "✅ Execução finalizada: $(date)" >> "$LOG_FILE"
