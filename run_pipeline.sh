#!/bin/bash

# NBA Predictor V12 Alpha - Automation Script
# Author: Denis Santos / Antigravity
# Description: Executa o pipeline diário de previsão com resiliência e logs.

# Configuração
PROJECT_DIR="/home/denis/nba-predictor"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/execution_$DATE.log"
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"

# Criar diretório de logs se não existir
mkdir -p "$LOG_DIR"

echo "==================================================" | tee -a "$LOG_FILE"
echo "🚀 Iniciando NBA Predictor Pipeline: $(date)" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# 1. Verificar Conectividade
echo "📡 Verificando conexão com a internet..." | tee -a "$LOG_FILE"
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo "✅ Conexão OK." | tee -a "$LOG_FILE"
else
    echo "❌ Sem conexão com a internet. Abortando." | tee -a "$LOG_FILE"
    exit 1
fi

# 2. Navegar para o diretório
cd "$PROJECT_DIR" || { echo "❌ Diretório do projeto não encontrado!" | tee -a "$LOG_FILE"; exit 1; }

# 3. Ativar Virtual Environment
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    echo "✅ Venv ativado." | tee -a "$LOG_FILE"
else
    echo "❌ Venv não encontrado em $VENV_ACTIVATE" | tee -a "$LOG_FILE"
    exit 1
fi

# 4. Executar Pipeline
echo "🏃 Executando main.py..." | tee -a "$LOG_FILE"
# Flags: --ml para usar modelo, --save_db para persistir (implícito no código atual)
python main.py --ml >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline concluído com sucesso." | tee -a "$LOG_FILE"
else
    echo "❌ Erro na execução do pipeline (Exit Code: $EXIT_CODE)." | tee -a "$LOG_FILE"
fi

echo "==================================================" | tee -a "$LOG_FILE"
echo "🏁 Fim da execução: $(date)" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
