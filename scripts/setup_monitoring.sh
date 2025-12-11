#!/bin/bash
"""
Setup de Monitoramento Automático

Configura cron jobs para monitoramento contínuo do modelo.

Usage:
    bash scripts/setup_monitoring.sh
"""

echo "=================================================================================="
echo "🔧 SETUP DE MONITORAMENTO AUTOMÁTICO"
echo "=================================================================================="

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Diretório do projeto
PROJECT_DIR="$HOME/nba-predictor"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"

# Verificar se está no diretório correto
if [ ! -f "$PROJECT_DIR/scripts/monitoring_system.py" ]; then
    echo -e "${RED}❌ Erro: monitoring_system.py não encontrado!${NC}"
    echo "Execute este script de dentro do diretório nba-predictor"
    exit 1
fi

# Criar diretório de logs se não existir
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✅ Diretório de logs criado: $LOG_DIR${NC}"

# Criar diretório de monitoramento
mkdir -p "$PROJECT_DIR/data/monitoring"
echo -e "${GREEN}✅ Diretório de monitoramento criado${NC}"

# Backup do crontab atual
echo ""
echo "📋 Fazendo backup do crontab atual..."
crontab -l > "$PROJECT_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || touch "$PROJECT_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
echo -e "${GREEN}✅ Backup salvo${NC}"

# Criar arquivo de cron jobs
CRON_FILE="/tmp/nba_predictor_cron.txt"

cat > "$CRON_FILE" << EOF
# NBA Predictor - Monitoramento Automático
# Gerado automaticamente em $(date)

# Atualizar métricas diárias (todo dia às 2am)
0 2 * * * cd $PROJECT_DIR && $VENV_PYTHON scripts/monitoring_system.py --update-daily --check-alerts >> $LOG_DIR/monitoring_daily.log 2>&1

# Atualizar métricas semanais (todo domingo às 3am)
0 3 * * 0 cd $PROJECT_DIR && $VENV_PYTHON scripts/monitoring_system.py --update-weekly >> $LOG_DIR/monitoring_weekly.log 2>&1

# Gerar relatório mensal (todo dia 1 às 4am)
0 4 1 * * cd $PROJECT_DIR && $VENV_PYTHON scripts/monitoring_system.py --generate-report >> $LOG_DIR/monitoring_monthly.log 2>&1

# Verificar alertas críticos (a cada 6 horas)
0 */6 * * * cd $PROJECT_DIR && $VENV_PYTHON scripts/monitoring_system.py --check-alerts >> $LOG_DIR/monitoring_alerts.log 2>&1

EOF

echo ""
echo "📋 Cron jobs a serem configurados:"
echo "=================================================================================="
cat "$CRON_FILE"
echo "=================================================================================="

# Perguntar confirmação
echo ""
echo -e "${YELLOW}⚠️  Deseja adicionar estes cron jobs? (s/n)${NC}"
read -r response

if [[ "$response" =~ ^([sS][iI][mM]|[sS])$ ]]; then
    # Adicionar ao crontab
    (crontab -l 2>/dev/null; cat "$CRON_FILE") | crontab -
    
    echo -e "${GREEN}✅ Cron jobs configurados com sucesso!${NC}"
    echo ""
    echo "📅 Monitoramento configurado:"
    echo "   • Diário: 2am (métricas + alertas)"
    echo "   • Semanal: Domingo 3am"
    echo "   • Mensal: Dia 1 às 4am"
    echo "   • Alertas: A cada 6 horas"
    echo ""
    echo "📁 Logs salvos em: $LOG_DIR/"
    echo ""
    echo "✅ Para verificar crontab: crontab -l"
    echo "✅ Para ver logs: tail -f $LOG_DIR/monitoring_daily.log"
    
else
    echo -e "${YELLOW}⚠️  Setup cancelado. Cron jobs NÃO foram adicionados.${NC}"
    echo ""
    echo "💡 Para adicionar manualmente:"
    echo "   1. Execute: crontab -e"
    echo "   2. Adicione as linhas do arquivo: $CRON_FILE"
fi

# Limpar arquivo temporário
rm -f "$CRON_FILE"

echo ""
echo "=================================================================================="
echo "🎯 TESTE RÁPIDO"
echo "=================================================================================="
echo "Executando monitoramento de teste..."

# Testar script
$VENV_PYTHON scripts/monitoring_system.py --generate-report

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Script de monitoramento funcionando corretamente!${NC}"
else
    echo -e "${RED}❌ Erro ao executar script de monitoramento${NC}"
    exit 1
fi

echo ""
echo "=================================================================================="
echo "✅ SETUP CONCLUÍDO!"
echo "=================================================================================="
echo ""
echo "📊 Próximos passos:"
echo "   1. Verificar cron jobs: crontab -l"
echo "   2. Monitorar logs: tail -f $LOG_DIR/monitoring_daily.log"
echo "   3. Verificar alertas: cat data/monitoring/alerts.json"
echo ""
