#!/bin/bash

PROJECT_DIR="/home/denis/nba-predictor"
BOT_CMD="@reboot /bin/bash $PROJECT_DIR/scripts/run_bot.sh >> $PROJECT_DIR/logs/cron_bot.log 2>&1"

# Adicionar ao crontab se não existir
(crontab -l 2>/dev/null | grep -v "run_bot.sh"; echo "$BOT_CMD") | crontab -

echo "✅ Cron job configurado: Bot iniciará automaticamente no boot (@reboot)"
echo "📋 Comando: $BOT_CMD"

# Iniciar agora manualmente para não precisar reiniciar
if ! pgrep -f "nba_tigrinho_bot.py" > /dev/null; then
    echo "🚀 Iniciando bot agora..."
    nohup bash $PROJECT_DIR/scripts/run_bot.sh > /dev/null 2>&1 &
    echo "✅ Bot iniciado em background."
else
    echo "⚠️ Bot já está rodando."
fi
