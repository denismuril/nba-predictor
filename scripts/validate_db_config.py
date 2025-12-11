"""
Script de Validação de Configuração de Banco de Dados

Valida a configuração do banco de dados, testa conexão, e verifica schema.

Usage:
    python scripts/validate_db_config.py
"""
import sys
import os
from pathlib import Path
import logging

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseValidator:
    """Validador de configuração de banco de dados."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.db = None
    
    def validate_environment(self):
        """Valida variáveis de ambiente."""
        logger.info("\n1️⃣  VALIDAÇÃO DE AMBIENTE")
        logger.info("="*60)
        
        db_type = os.getenv('DB_TYPE', 'sqlite')
        logger.info(f"DB_TYPE: {db_type}")
        
        if db_type not in ['sqlite', 'postgres', 'postgresql']:
            self.errors.append(f"DB_TYPE inválido: {db_type}")
            logger.error(f"❌ DB_TYPE deve ser 'sqlite' ou 'postgres'")
            return False
        
        # Validar credenciais PostgreSQL se necessário
        if db_type in ['postgres', 'postgresql']:
            required_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASS']
            missing = [var for var in required_vars if not os.getenv(var)]
            
            if missing:
                self.errors.append(f"Variáveis PostgreSQL faltando: {missing}")
                logger.error(f"❌ Variáveis faltando: {', '.join(missing)}")
                return False
            
            logger.info(f"DB_HOST: {os.getenv('DB_HOST')}")
            logger.info(f"DB_PORT: {os.getenv('DB_PORT')}")
            logger.info(f"DB_NAME: {os.getenv('DB_NAME')}")
            logger.info(f"DB_USER: {os.getenv('DB_USER')}")
        
        logger.info("✅ Ambiente validado")
        return True
    
    def test_connection(self):
        """Testa conexão com banco de dados."""
        logger.info("\n2️⃣  TESTE DE CONEXÃO")
        logger.info("="*60)
        
        try:
            self.db = get_db_manager()
            logger.info(f"Database: {self.db.db_type.upper()}")
            logger.info(f"Tentando conexão...")
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Executar query simples
            if self.db.db_type == 'sqlite':
                cursor.execute("SELECT sqlite_version()")
                version = cursor.fetchone()[0]
                logger.info(f"✅ Conectado - SQLite versão {version}")
            else:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                logger.info(f"✅ Conectado - PostgreSQL versão:")
                logger.info(f"   {version[:80]}...")
            
            self.db.return_connection(conn)
            return True
            
        except Exception as e:
            self.errors.append(f"Erro de conexão: {e}")
            logger.error(f"❌ Falha na conexão: {e}")
            return False
    
    def validate_schema(self):
        """Valida schema do banco de dados."""
        logger.info("\n3️⃣  VALIDAÇÃO DE SCHEMA")
        logger.info("="*60)
        
        if not self.db:
            logger.error("❌ Skipping - sem conexão")
            return False
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            # Tabelas esperadas
            expected_tables = ['games', 'game_stats', 'predictions', 'bets']
            
            # Verificar existência das tabelas
            if self.db.db_type == 'sqlite':
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            else:
                cursor.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
                )
            
            existing_tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Tabelas encontradas: {existing_tables}")
            
            # Verificar tabelas esperadas
            missing_tables = [t for t in expected_tables if t not in existing_tables]
            
            if missing_tables:
                self.warnings.append(f"Tabelas faltando: {missing_tables}")
                logger.warning(f"⚠️  Tabelas faltando: {', '.join(missing_tables)}")
            else:
                logger.info("✅ Todas tabelas esperadas existem")
            
            # Verificar índices
            logger.info("\nÍndices:")
            if self.db.db_type == 'sqlite':
                cursor.execute(
                    "SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
                )
            else:
                cursor.execute(
                    "SELECT indexname, tablename FROM pg_indexes WHERE indexname LIKE 'idx_%'"
                )
            
            indexes = cursor.fetchall()
            logger.info(f"  Total de índices customizados: {len(indexes)}")
            
            if len(indexes) < 5:
                self.warnings.append(f"Poucos índices ({len(indexes)}) - considere otimizar")
                logger.warning(f"⚠️  Apenas {len(indexes)} índices - considere rodar optimize_database.py")
            
            # Verificar dados
            logger.info("\nDados:")
            for table in ['games', 'predictions', 'bets']:
                if table in existing_tables:
                    cursor.execute(self.db._prepare_query(f"SELECT COUNT(*) FROM {table}"))
                    count = cursor.fetchone()[0]
                    logger.info(f"  {table}: {count} registros")
                    
                    if table == 'games' and count == 0:
                        self.warnings.append(f"Tabela {table} está vazia")
            
            logger.info("✅ Schema validado")
            return True
            
        except Exception as e:
            self.errors.append(f"Erro validando schema: {e}")
            logger.error(f"❌ Erro validando schema: {e}")
            return False
        finally:
            self.db.return_connection(conn)
    
    def test_operations(self):
        """Testa operações básicas."""
        logger.info("\n4️⃣  TESTE DE OPERAÇÕES")
        logger.info("="*60)
        
        if not self.db:
            logger.error("❌ Skipping - sem conexão")
            return False
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            # Teste de INSERT
            test_id = "test_validation_" + str(os.getpid())
            
            cursor.execute(
                self.db._prepare_query(
                    "INSERT INTO predictions (game_id, date, home_team, away_team) VALUES (?, ?, ?, ?)"
                ),
                (test_id, '2025-01-01', 'TEST', 'TEST')
            )
            logger.info("✅ INSERT funcional")
            
            # Teste de SELECT
            cursor.execute(
                self.db._prepare_query("SELECT * FROM predictions WHERE game_id = ?"),
                (test_id,)
            )
            result = cursor.fetchone()
            logger.info("✅ SELECT funcional")
            
            # Teste de DELETE (cleanup)
            cursor.execute(
                self.db._prepare_query("DELETE FROM predictions WHERE game_id = ?"),
                (test_id,)
            )
            logger.info("✅ DELETE funcional")
            
            conn.commit()
            logger.info("✅ Operações validadas")
            return True
            
        except Exception as e:
            conn.rollback()
            self.errors.append(f"Erro testando operações: {e}")
            logger.error(f"❌ Erro testando operações: {e}")
            return False
        finally:
            self.db.return_connection(conn)
    
    def print_summary(self):
        """Imprime resumo da validação."""
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMO DA VALIDAÇÃO")
        logger.info("="*60)
        
        if self.errors:
            logger.error(f"\n❌ ERROS ({len(self.errors)}):")
            for err in self.errors:
                logger.error(f"  • {err}")
        
        if self.warnings:
            logger.warning(f"\n⚠️  AVISOS ({len(self.warnings)}):")
            for warn in self.warnings:
                logger.warning(f"  • {warn}")
        
        if not self.errors and not self.warnings:
            logger.info("\n✅ SISTEMA TOTALMENTE VALIDADO - NENHUM PROBLEMA ENCONTRADO")
        elif not self.errors:
            logger.info("\n✅ SISTEMA FUNCIONAL - Apenas avisos não-críticos")
        else:
            logger.error("\n❌ SISTEMA COM PROBLEMAS - Correções necessárias")
        
        return len(self.errors) == 0


def main():
    logger.info("="*60)
    logger.info("🔍 VALIDADOR DE CONFIGURAÇÃO DE BANCO DE DADOS")
    logger.info("="*60)
    
    validator = DatabaseValidator()
    
    # Executar validações
    env_ok = validator.validate_environment()
    
    if env_ok:
        conn_ok = validator.test_connection()
        
        if conn_ok:
            validator.validate_schema()
            validator.test_operations()
    
    # Resumo
    success = validator.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
