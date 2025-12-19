#!/bin/bash
# ==============================================================================
# Monitor de Logs do God Mode
# ==============================================================================

PROJECT_DIR="/home/denis/nba-predictor"
LOG_DIR="${PROJECT_DIR}/logs"

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 Iniciando Monitoramento de Logs do God Mode${NC}"
echo "--------------------------------------------------------"
echo "Monitorando:"
echo "1. god_mode_service.log (Saída do Systemd)"
echo "2. god_mode_error.log (Erros do Systemd)"
echo "3. orchestrator.jsonl (Logs estruturados do Python)"
echo "--------------------------------------------------------"
echo "Pressione Ctrl+C para sair"

# Garantir que logs existem para não dar erro no tail
touch "${LOG_DIR}/god_mode_service.log"
touch "${LOG_DIR}/god_mode_error.log"
touch "${LOG_DIR}/orchestrator.jsonl"

# Tail multi-arquivo
tail -f \
    "${LOG_DIR}/god_mode_service.log" \
    "${LOG_DIR}/god_mode_error.log" \
    "${LOG_DIR}/orchestrator.jsonl"
