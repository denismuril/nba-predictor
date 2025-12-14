#!/bin/bash
# =============================================================================
# Script de Setup: PostgreSQL e Redis para NBA Predictor v22.0
# =============================================================================
# Uso:
#   chmod +x scripts/setup_enterprise_infra.sh
#   ./scripts/setup_enterprise_infra.sh
#
# Este script:
# 1. Instala PostgreSQL (se não instalado)
# 2. Cria banco de dados e usuário
# 3. Instala Redis (se não instalado)
# 4. Configura .env com as credenciais
# =============================================================================

set -e  # Parar em caso de erro

echo "=========================================="
echo "🚀 Setup NBA Predictor Enterprise v22.0"
echo "=========================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configurações padrão
DB_NAME="nba_predictor_db"
DB_USER="nba_admin"
DB_PASS="nba_secure_pass_2024"
REDIS_PORT=6379

# =============================================================================
# FUNÇÕES
# =============================================================================

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 encontrado"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1 não encontrado"
        return 1
    fi
}

install_postgresql() {
    echo ""
    echo "📦 Instalando PostgreSQL..."
    
    if check_command psql; then
        echo "   PostgreSQL já instalado"
    else
        sudo apt update
        sudo apt install -y postgresql postgresql-contrib
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
        echo -e "${GREEN}✓${NC} PostgreSQL instalado e iniciado"
    fi
}

setup_postgresql_db() {
    echo ""
    echo "🗄️  Configurando banco de dados PostgreSQL..."
    
    # Verificar se o banco já existe
    if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
        echo -e "${YELLOW}⚠${NC} Banco '$DB_NAME' já existe"
    else
        # Criar usuário e banco
        sudo -u postgres psql <<EOF
-- Criar usuário se não existir
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$DB_USER') THEN
        CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';
    END IF;
END
\$\$;

-- Criar banco de dados
CREATE DATABASE $DB_NAME OWNER $DB_USER;

-- Conceder permissões
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;

-- Habilitar extensões úteis
\c $DB_NAME
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- Para buscas fuzzy
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- Para UUIDs

\q
EOF
        echo -e "${GREEN}✓${NC} Banco '$DB_NAME' criado com sucesso"
    fi
    
    # Testar conexão
    echo "   Testando conexão..."
    PGPASSWORD=$DB_PASS psql -h localhost -U $DB_USER -d $DB_NAME -c "SELECT 1;" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Conexão PostgreSQL OK"
    else
        echo -e "${RED}✗${NC} Erro na conexão PostgreSQL"
        echo "   Verifique pg_hba.conf para permitir conexões locais"
    fi
}

install_redis() {
    echo ""
    echo "📦 Instalando Redis..."
    
    if check_command redis-server; then
        echo "   Redis já instalado"
    else
        sudo apt update
        sudo apt install -y redis-server
        echo -e "${GREEN}✓${NC} Redis instalado"
    fi
    
    # Iniciar Redis
    sudo systemctl start redis-server
    sudo systemctl enable redis-server
    
    # Testar conexão
    echo "   Testando conexão..."
    if redis-cli ping | grep -q "PONG"; then
        echo -e "${GREEN}✓${NC} Conexão Redis OK"
    else
        echo -e "${RED}✗${NC} Erro na conexão Redis"
    fi
}

update_env_file() {
    echo ""
    echo "📝 Atualizando arquivo .env..."
    
    ENV_FILE=".env"
    
    # Fazer backup
    if [ -f "$ENV_FILE" ]; then
        cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        echo "   Backup criado: ${ENV_FILE}.backup.*"
    fi
    
    # Adicionar configurações se não existirem
    add_env_var() {
        local key=$1
        local value=$2
        if ! grep -q "^$key=" "$ENV_FILE" 2>/dev/null; then
            echo "$key=$value" >> "$ENV_FILE"
            echo "   Adicionado: $key"
        else
            echo "   Já existe: $key"
        fi
    }
    
    # Configurações de banco de dados
    add_env_var "DB_TYPE" "postgres"
    add_env_var "DB_HOST" "localhost"
    add_env_var "DB_PORT" "5432"
    add_env_var "DB_NAME" "$DB_NAME"
    add_env_var "DB_USER" "$DB_USER"
    add_env_var "DB_PASS" "$DB_PASS"
    
    # Configurações de Redis
    add_env_var "REDIS_HOST" "localhost"
    add_env_var "REDIS_PORT" "$REDIS_PORT"
    add_env_var "REDIS_PASSWORD" ""
    add_env_var "REDIS_DB" "0"
    
    echo -e "${GREEN}✓${NC} Arquivo .env atualizado"
}

show_summary() {
    echo ""
    echo "=========================================="
    echo "✅ SETUP CONCLUÍDO!"
    echo "=========================================="
    echo ""
    echo "📊 PostgreSQL:"
    echo "   Host: localhost"
    echo "   Port: 5432"
    echo "   Database: $DB_NAME"
    echo "   User: $DB_USER"
    echo "   Password: $DB_PASS"
    echo ""
    echo "🔴 Redis:"
    echo "   Host: localhost"
    echo "   Port: $REDIS_PORT"
    echo ""
    echo "📝 Próximos passos:"
    echo "   1. Revise o arquivo .env"
    echo "   2. Execute a migração:"
    echo "      python scripts/migrate_to_postgres.py"
    echo ""
    echo "   3. Inicie o sistema:"
    echo "      python orchestrator.py"
    echo ""
}

# =============================================================================
# MAIN
# =============================================================================

echo ""
echo "Este script irá:"
echo "  1. Instalar PostgreSQL e Redis (se necessário)"
echo "  2. Criar banco de dados '$DB_NAME'"
echo "  3. Criar usuário '$DB_USER'"
echo "  4. Atualizar .env com configurações"
echo ""
read -p "Continuar? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    install_postgresql
    setup_postgresql_db
    install_redis
    update_env_file
    show_summary
else
    echo "Setup cancelado."
    exit 0
fi
