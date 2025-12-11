#!/bin/bash

# Caminho absoluto para o projeto e python
PROJECT_DIR="/home/denis/nba-predictor"
PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"
LOG_FILE="$PROJECT_DIR/logs/cron_daily.log"

# Garantir que diretório de logs existe
mkdir -p "$PROJECT_DIR/logs"

# Comando do Orchestrator (10:00 AM Todo dia)
# Executa: Fetch Data -> Drift Check -> Monitoring -> Predictions -> Email
CRON_CMD="0 10 * * * /bin/bash $PROJECT_DIR/scripts/orchestrator.sh"

# Adicionar ao crontab (Remove anteriores do projeto para limpar)
# Removemos qualquer referência ao projeto para evitar lixo antigo
(crontab -l 2>/dev/null | grep -v "$PROJECT_DIR"; echo "$CRON_CMD") | crontab -

echo "✅ Cron job configurado: Orchestrator às 10:00 AM"
echo "📋 Comando: $CRON_CMD"
echo "📝 Logs gerenciados internamente pelo Orchestrator em logs/"
