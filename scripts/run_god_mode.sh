#!/bin/bash
# ==============================================================================
# God Mode Runner - wrapper para Systemd/Cron
# ==============================================================================
# Este script prepara o ambiente e executa o orchestrator.py no modo Enterprise.

# Definir caminhos absolutos para robustez
PROJECT_DIR="/home/denis/nba-predictor"
VENV_ACTIVATE="${PROJECT_DIR}/venv/bin/activate"  # Ajuste se usar outro venv
ORCHESTRATOR="${PROJECT_DIR}/orchestrator.py"

# Garantir que estamos no diretório do projeto
cd "$PROJECT_DIR" || exit 1

# Exportar PYTHONPATH para garantir imports funcionem
export PYTHONPATH="$PROJECT_DIR"

# Ativar virtualenv se existir
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
fi

# Configurar variáveis de ambiente críticas
export SNIPER_ENABLED="true"
export NEWS_FILTER_ENABLED="true"
export LOG_LEVEL="INFO"

echo "================================================================="
echo "🚀 Iniciando NBA God Mode Pipeline: $(date)"
echo "📂 Dir: $PROJECT_DIR"
echo "================================================================="

# Executar orquestrador com python do ambiente atual
python3 "$ORCHESTRATOR"

EXIT_CODE=$?

echo "================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline finalizado com SUCESSO: $(date)"
else
    echo "❌ Pipeline falhou com código $EXIT_CODE: $(date)"
fi
echo "================================================================="

exit $EXIT_CODE
