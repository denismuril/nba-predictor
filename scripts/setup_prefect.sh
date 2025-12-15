#!/bin/bash
# =============================================================================
# Prefect Setup Script - NBA Predictor v24.0
# =============================================================================
# Este script configura o Prefect para orquestração profissional.
# Execute com: bash scripts/setup_prefect.sh
# =============================================================================

set -e

echo "🚀 Configurando Prefect para NBA Predictor..."

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Verificar/Instalar Prefect
echo -e "\n${YELLOW}1. Verificando instalação do Prefect...${NC}"
if ! command -v prefect &> /dev/null; then
    echo "Instalando Prefect..."
    pip install prefect prefect-shell
else
    echo -e "${GREEN}✓ Prefect já instalado$(prefect version)${NC}"
fi

# 2. Configurar Prefect (local ou cloud)
echo -e "\n${YELLOW}2. Configurando Prefect API...${NC}"

# Por padrão, usar servidor local
if [ -z "$PREFECT_API_URL" ]; then
    echo "Usando servidor local (prefect server start)"
    export PREFECT_API_URL="http://127.0.0.1:4200/api"
fi

# 3. Criar Work Pool (se não existir)
echo -e "\n${YELLOW}3. Criando Work Pool...${NC}"
prefect work-pool create default-agent-pool --type process 2>/dev/null || echo "Work pool já existe"

# 4. Deploy flows
echo -e "\n${YELLOW}4. Fazendo deploy dos flows...${NC}"
cd "$(dirname "$0")/.."
prefect deploy --all

# 5. Instruções finais
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Prefect configurado com sucesso!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Próximos passos:"
echo ""
echo "1. Iniciar o servidor Prefect (em um terminal):"
echo "   ${YELLOW}prefect server start${NC}"
echo ""
echo "2. Iniciar o worker (em outro terminal):"
echo "   ${YELLOW}prefect worker start --pool default-agent-pool${NC}"
echo ""
echo "3. Acessar UI:"
echo "   ${YELLOW}http://localhost:4200${NC}"
echo ""
echo "4. Rodar flow manualmente:"
echo "   ${YELLOW}prefect deployment run 'NBA Daily Pipeline/daily-pipeline'${NC}"
echo ""
echo "Flows agendados:"
echo "  - 08:00 BRT: Health Check"
echo "  - 09:00 BRT: Settlement (liquidação)"
echo "  - 17:00 BRT: Daily Pipeline (previsões)"
echo "  - 18:00 BRT: Paper Trading"
echo ""
