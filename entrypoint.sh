#!/bin/bash
# =============================================================================
# NBA Predictor - Docker Entrypoint
# =============================================================================
# Este script:
# 1. Executa o ciclo de produção no startup
# 2. Configura cron job para execução diária às 10:00 AM ET
# 3. Mantém o container rodando
# =============================================================================

set -e

echo "🏀 NBA Predictor - Iniciando Container..."
echo "=========================================="
echo "Data/Hora: $(date)"
echo "Usuário: $(whoami)"
echo "Diretório: $(pwd)"
echo ""

# Criar diretório de logs se não existir
mkdir -p /app/logs

# Verificar variáveis de ambiente obrigatórias
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ ERRO: TELEGRAM_BOT_TOKEN não definido!"
    echo "Configure via docker-compose ou -e TELEGRAM_BOT_TOKEN=xxx"
    exit 1
fi

if [ -z "$TELEGRAM_ADMIN_ID" ]; then
    echo "❌ ERRO: TELEGRAM_ADMIN_ID não definido!"
    echo "Configure via docker-compose ou -e TELEGRAM_ADMIN_ID=xxx"
    exit 1
fi

echo "✅ Variáveis de ambiente validadas"
echo ""

# =============================================================================
# EXECUÇÃO INICIAL
# =============================================================================
echo "🚀 Executando ciclo de produção inicial..."
echo "=========================================="

# Rodar o ciclo de produção
python /app/run_production_cycle.py 2>&1 | tee -a /app/logs/production_run.log

echo ""
echo "✅ Ciclo inicial concluído"
echo ""

# =============================================================================
# CONFIGURAR CRON JOB
# =============================================================================
echo "⏰ Configurando cron job para execução diária..."

# Criar script wrapper para cron
cat > /app/run_daily_cron.sh << 'EOF'
#!/bin/bash
cd /app
source .venv/bin/activate 2>/dev/null || true
python /app/run_production_cycle.py >> /app/logs/cron_production.log 2>&1
EOF

chmod +x /app/run_daily_cron.sh

# Configurar crontab
# 10:00 AM ET (15:00 UTC no horário padrão, 14:00 UTC no horário de verão)
# Para simplificar, usamos 15:00 UTC que é ~10-11 AM ET
echo "0 15 * * * /app/run_daily_cron.sh" > /var/spool/cron/crontabs/nba 2>/dev/null || \
    echo "0 15 * * * /app/run_daily_cron.sh" | crontab -

echo "✅ Cron job configurado para 15:00 UTC (~10:00 AM ET)"
echo ""

# =============================================================================
# MANTER CONTAINER RODANDO
# =============================================================================
echo "🔄 Modo de operação: $1"
echo ""

case "$1" in
    "bot")
        echo "🤖 Iniciando bot do Telegram..."
        exec python -m telegram_bot.nba_tigrinho_bot
        ;;
    "web")
        echo "🌐 Iniciando interface web..."
        exec streamlit run nba_predictor_web.py --server.address=0.0.0.0 --server.port=8501
        ;;
    "cron")
        echo "⏰ Modo cron - aguardando próximo ciclo..."
        # Iniciar cron daemon e manter container rodando
        cron
        tail -f /app/logs/production_run.log
        ;;
    *)
        echo "📋 Modo padrão - executando comando: $@"
        exec "$@"
        ;;
esac
