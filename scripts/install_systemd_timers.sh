#!/bin/bash
# Script para instalar os timers systemd e remover o cron

echo "🔧 Instalando Systemd Timers para NBA Predictor..."

# Copiar arquivos para systemd
sudo cp /home/denis/nba-predictor/scripts/systemd/*.service /etc/systemd/system/
sudo cp /home/denis/nba-predictor/scripts/systemd/*.timer /etc/systemd/system/

# Recarregar systemd
sudo systemctl daemon-reload

# Ativar e iniciar timers
echo "✅ Ativando timers..."
sudo systemctl enable --now nba-orchestrator.timer
sudo systemctl enable --now nba-odds-tracking.timer
sudo systemctl enable --now nba-weekly-update.timer

# Remover cron antigo
echo "🗑️ Removendo cron antigo..."
crontab -r 2>/dev/null || echo "   (cron já vazio)"

# Status
echo ""
echo "📊 Status dos timers:"
systemctl list-timers --all | grep nba

echo ""
echo "✅ Migração concluída!"
echo ""
echo "Comandos úteis:"
echo "  - Ver todos timers: systemctl list-timers --all | grep nba"
echo "  - Ver logs: journalctl -u nba-orchestrator -f"
echo "  - Testar manualmente: sudo systemctl start nba-orchestrator.service"
