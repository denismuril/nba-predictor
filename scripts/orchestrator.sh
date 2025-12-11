#!/bin/bash

# Master Orchestrator Script 🎻
# Gerencia a execução diária e semanal do NBA Predictor.
# Deve ser agendado no CRON para rodar todo dia (ex: 10:00 AM).

# Configuração de Diretórios
PROJECT_DIR="/home/denis/nba-predictor"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/orchestrator_$DATE.log"

# Garantir diretório de logs
mkdir -p "$LOG_DIR"

# Função de Log
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🚀 Iniciando Orchestrator..."

# Navegar para o projeto
cd "$PROJECT_DIR" || { log "❌ Erro: Diretório do projeto não encontrado!"; exit 1; }

# Ativar Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    log "❌ Erro: Virtual Environment não encontrado!"
    exit 1
fi

# Identificar dia da semana (1=Segunda, 7=Domingo)
DOW=$(date +%u)

# --- TAREFAS SEMANAIS (Segunda-feira) ---
if [ "$DOW" -eq 1 ]; then
    log "📅 Segunda-feira detectada. Iniciando ciclo de OTIMIZAÇÃO SEMANAL."
    
    # 1. Otimização de Hiperparâmetros (Optuna)
    log "🧪 1/3 Rodando Otimização de Hiperparâmetros..."
    python -m ml_pipeline.optimize_hyperparameters >> "$LOG_FILE" 2>&1
    
    # 2. Treinamento do Modelo Final
    log "🏋️‍♂️ 2/3 Treinando Modelo com novos parâmetros..."
    python -m ml_pipeline.train_spread_real >> "$LOG_FILE" 2>&1
    
    # 3. Validação Walk-Forward (Relatório)
    log "📈 3/3 Gerando Relatório Walk-Forward..."
    python -m ml_pipeline.walk_forward >> "$LOG_FILE" 2>&1
    
    # 4. Testes Semanais (CI/CD)
    log "🧪 4/5 Executando Testes Semanais..."
    python scripts/weekly_tests.py >> "$LOG_FILE" 2>&1
    
    # 5. Auditoria de Segurança (Pentest)
    log "🛡️ 5/6 Executando Auditoria de Segurança..."
    python scripts/security_audit.py >> "$LOG_FILE" 2>&1
    
    # 6. Relatório Semanal de Monitoramento (Fase 2/3)
    log "📊 6/6 Gerando Relatório Semanal de Performance..."
    python scripts/monitoring_system.py --update-weekly --generate-report >> "$LOG_FILE" 2>&1
    
    log "✅ Ciclo Semanal concluído."
else
    log "ℹ️ Não é segunda-feira. Pulando otimização pesada."
fi

# --- TAREFAS DIÁRIAS (Todo dia) ---
log "🔄 Iniciando Pipeline Diário..."

# 0. Atualizar Dados da NBA API (Incremental)
log "📥 Buscando novos jogos (últimos 3 dias)..."
python scripts/fetch_historical_data.py --days 3 >> "$LOG_FILE" 2>&1

# 1. Verificar Model Drift (Segurança)
# Se o modelo estiver degradado, forçamos um re-treino de emergência mesmo não sendo segunda
DRIFT_STATUS=$(python -c "from ml_pipeline.drift_monitor import check_model_drift; print(check_model_drift()['status'])" 2>> "$LOG_FILE")

if [ "$DRIFT_STATUS" == "DRIFT" ]; then
    log "⚠️ ALERTA: Model Drift detectado! Iniciando re-treino de emergência..."
    python -m ml_pipeline.train_spread_real >> "$LOG_FILE" 2>&1
fi

# 2. Atualizar Monitoramento de Performance (Fase 2/3)
log "📊 Atualizando métricas de monitoramento..."
python scripts/monitoring_system.py --update-daily --check-alerts >> "$LOG_FILE" 2>&1

# Verificar se há alertas críticos
if [ -f "data/monitoring/alerts.json" ]; then
    ALERT_COUNT=$(python -c "import json; data=json.load(open('data/monitoring/alerts.json')); print(len(data))" 2>/dev/null || echo "0")
    if [ "$ALERT_COUNT" -gt 0 ]; then
        log "⚠️ ALERTA: $ALERT_COUNT alertas de performance detectados!"
        log "   Verifique: data/monitoring/alerts.json"
    fi
fi

# 3. Executar Predições Diárias (CLI)
# Isso atualiza o DB, gera CSVs e prepara dados para o Bot
log "🔮 Gerando Predições do dia..."
python interfaces/cli.py --ml >> "$LOG_FILE" 2>&1

# 4. Enviar Relatório por Email (Fase 3)
log "📧 Enviando relatório por email..."
# Carregar variáveis de ambiente de email
if [ -f ".env.email" ]; then
    source .env.email
    python scripts/email_notification.py --daily-report >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log "✅ Email enviado com sucesso"
    else
        log "⚠️ Falha ao enviar email (verifique .env.email)"
    fi
else
    log "⚠️ Arquivo .env.email não encontrado - pulando envio de email"
fi

# (Opcional) Enviar Resumo para Telegram
# Se você tiver um script específico para enviar o resumo, coloque aqui.
# Por enquanto, o Bot Telegram lê do DB/CSV quando o usuário pede /jogos.

log "✅ Orchestrator finalizado com sucesso!"
