#!/bin/bash
# Script para instalar os timers systemd e remover o cron

echo "🔧 Instalando Systemd Timers para NBA Predictor..."

# Copiar arquivos para systemd
sudo cp /home/denis/nba-predictor/scripts/systemd/*.service /etc/systemd/system/
sudo cp /home/denis/nba-predictor/scripts/systemd/*.timer /etc/systemd/system/

# Recarregar systemd
sudo systemctl daemon-reload

# Parar e desativar serviços antigos (Clean up legacy)
echo "🛑 Parando serviços antigos..."
sudo systemctl disable --now nba-orchestrator.timer 2>/dev/null || true
sudo systemctl disable --now nba-odds-tracking.timer 2>/dev/null || true
sudo systemctl disable --now nba-weekly-update.timer 2>/dev/null || true

# Ativar e iniciar NOVO timer God Mode
echo "✅ Ativando God Mode Timer..."
sudo systemctl enable --now nba-god-mode.timer

# Remover cron antigo
echo "🗑️ Removendo cron antigo..."
crontab -r 2>/dev/null || echo "   (cron já vazio)"

# Status
echo ""
echo "📊 Status dos timers:"
systemctl list-timers --all | grep nba

echo ""
echo "✅ Instalação e Migração concluída!"
echo ""
echo "Comandos úteis:"
echo "  - Ver timers: systemctl list-timers --all | grep nba"
echo "  - Ver logs: ./scripts/monitor_god_mode.sh"
echo "  - Testar manualmente: sudo systemctl start nba-god-mode.service"
