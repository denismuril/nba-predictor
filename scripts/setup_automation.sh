#!/bin/bash
# Setup de Automação para NBA Predictor
# 
# Este script configura os cron jobs para:
# 1. Recalibração diária do modelo
# 2. Atualização do dashboard de monitoramento
# 3. Backup automático de modelos

echo "🔧 Configurando Automação NBA Predictor..."

# Diretório base
NBA_DIR="/home/denis/nba-predictor"
PYTHON_PATH="/usr/bin/python3"

# Verificar se diretório existe
if [ ! -d "$NBA_DIR" ]; then
    echo "❌ Diretório não encontrado: $NBA_DIR"
    echo "ℹ️  Ajuste NBA_DIR no início deste script"
    exit 1
fi

echo "✅ Diretório encontrado: $NBA_DIR"

# Criar diretório de logs se não existir
mkdir -p "$NBA_DIR/logs"
echo "✅ Diretório de logs: $NBA_DIR/logs"

# Criar arquivo temporário para crontab
CRON_FILE="/tmp/nba_cron_$$.txt"

# Exportar crontab atual
crontab -l > "$CRON_FILE" 2>/dev/null || true

# Adicionar comentário separador
echo "" >> "$CRON_FILE"
echo "# ============================================" >> "$CRON_FILE"
echo "# NBA Predictor - Automação" >> "$CRON_FILE"
echo "# Configurado em: $(date)" >> "$CRON_FILE"
echo "# ============================================" >> "$CRON_FILE"

# Job 1: Recalibração diária às 6:00 AM
echo "" >> "$CRON_FILE"
echo "# Recalibração diária do modelo (6:00 AM)" >> "$CRON_FILE"
echo "0 6 * * * cd $NBA_DIR && $PYTHON_PATH scripts/recalibrate_model.py --lookback-days 30 >> logs/recalibration.log 2>&1" >> "$CRON_FILE"

# Job 2: Atualização dashboard às 6:05 AM (após recalibração)
echo "" >> "$CRON_FILE"
echo "# Atualização dashboard (6:05 AM)" >> "$CRON_FILE"
echo "5 6 * * * cd $NBA_DIR && $PYTHON_PATH monitoring/update_dashboard.py >> logs/monitoring.log 2>&1" >> "$CRON_FILE"

# Job 3: Backup semanal de modelos (Domingo 2:00 AM)
echo "" >> "$CRON_FILE"
echo "# Backup semanal de modelos (Domingo 2:00 AM)" >> "$CRON_FILE"
echo "0 2 * * 0 cd $NBA_DIR && tar -czf backups/models_backup_\$(date +\\%Y\\%m\\%d).tar.gz models/*.joblib models/*.pkl >> logs/backup.log 2>&1" >> "$CRON_FILE"

# Job 4: Limpeza de logs antigos (Primeiro dia do mês, 3:00 AM)
echo "" >> "$CRON_FILE"
echo "# Limpeza de logs antigos (1º do mês, 3:00 AM)" >> "$CRON_FILE"
echo "0 3 1 * * find $NBA_DIR/logs -name '*.log' -mtime +30 -delete >> logs/cleanup.log 2>&1" >> "$CRON_FILE"

echo "" >> "$CRON_FILE"
echo "# ============================================" >> "$CRON_FILE"

# Aplicar novo crontab
crontab "$CRON_FILE"

# Verificar
echo ""
echo "📋 Cron jobs instalados:"
echo "─────────────────────────────────────────────"
crontab -l | grep -A 10 "NBA Predictor"
echo "─────────────────────────────────────────────"

# Limpar
rm "$CRON_FILE"

# Criar diretório de backups
mkdir -p "$NBA_DIR/backups"

echo ""
echo "✅ Automação configurada com sucesso!"
echo ""
echo "📅 Schedule:"
echo "  • 06:00 - Recalibração diária"
echo "  • 06:05 - Update dashboard"
echo "  • 02:00 Domingo - Backup modelos"
echo "  • 03:00 1º mês - Limpeza logs"
echo ""
echo "💡 Para verificar logs:"
echo "  tail -f logs/recalibration.log"
echo "  tail -f logs/monitoring.log"
echo ""
echo "💡 Para testar manualmente:"
echo "  python scripts/recalibrate_model.py"
echo ""
