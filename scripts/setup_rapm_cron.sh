#!/bin/bash

# Script para configurar cron job de atualização do RAPM
# Execução: às 9:30 AM todos os dias

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Configurando cron job para atualização do RAPM..."

# Criar diretório de logs se não existir
mkdir -p "$PROJECT_ROOT/logs"

# Linha do cron job
CRON_JOB="30 9 * * * cd $PROJECT_ROOT && python scripts/update_rapm.py >> logs/rapm_updates.log 2>&1"

# Verificar se o cron job já existe
if crontab -l 2>/dev/null | grep -q "update_rapm.py"; then
    echo "⚠️  Cron job já existe. Removendo versão antiga..."
    crontab -l 2>/dev/null | grep -v "update_rapm.py" | crontab -
fi

# Adicionar novo cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Cron job configurado com sucesso!"
echo ""
echo "📅 Agendamento: Todos os dias às 9:30 AM"
echo "📝 Log: $PROJECT_ROOT/logs/rapm_updates.log"
echo ""
echo "Para verificar:"
echo "  crontab -l | grep update_rapm"
echo ""
echo "Para remover:"
echo "  crontab -l | grep -v update_rapm.py | crontab -"
