#!/bin/bash
# Script para instalar o serviço da API NBA Predictor
# Uso: sudo bash scripts/install_api_service.sh

set -e

SERVICE_FILE="/home/denis/nba-predictor/systemd/nba-api.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "🏀 Instalando NBA Predictor API Service..."

# Criar diretório de logs se não existir
mkdir -p /home/denis/nba-predictor/logs

# Copiar arquivo de serviço
sudo cp "$SERVICE_FILE" "$SYSTEMD_DIR/nba-api.service"

# Recarregar systemd
sudo systemctl daemon-reload

# Habilitar serviço para iniciar no boot
sudo systemctl enable nba-api.service

# Iniciar serviço
sudo systemctl start nba-api.service

# Verificar status
echo ""
echo "✅ Serviço instalado!"
echo ""
sudo systemctl status nba-api.service --no-pager

echo ""
echo "📋 Comandos úteis:"
echo "  sudo systemctl status nba-api    # Ver status"
echo "  sudo systemctl restart nba-api   # Reiniciar"
echo "  sudo systemctl stop nba-api      # Parar"
echo "  journalctl -u nba-api -f         # Ver logs em tempo real"
echo ""
echo "🌐 API disponível em: http://localhost:8000"
