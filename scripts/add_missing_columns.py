"""
Script para adicionar colunas faltantes na tabela predictions.

REFATORADO: Suporta SQLite e PostgreSQL via DatabaseManager.
"""
import sys
import logging
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_column_if_not_exists(cursor, db, table, column, dtype="REAL DEFAULT 0"):
    """Adiciona coluna se não existir (compatível SQLite/PostgreSQL)."""
    try:
        # Adaptar sintaxe para PostgreSQL
        if db.db_type == 'postgres':
            # PostgreSQL requer separação de definição de tipo e default
            if 'DEFAULT' in dtype:
                col_type, default_val = dtype.split(' DEFAULT ')
                query = f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type} DEFAULT {default_val}"
            else:
                query = f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {dtype}"
        else:
            # SQLite
            query = f"ALTER TABLE {table} ADD COLUMN {column} {dtype}"
        
        cursor.execute(query)
        logger.info(f"✅ Coluna '{column}' adicionada à tabela '{table}'.")
    except Exception as e:
        if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
            logger.info(f"ℹ️ Coluna '{column}' já existe na tabela '{table}'.")
        else:
            logger.error(f"❌ Erro ao adicionar coluna '{column}': {e}")

def migrate():
    db = get_db_manager()
    logger.info(f"📂 Database: {db.db_type.upper()}")
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    try:
        columns_to_add = [
            'ast', 'opp_ast',
            'stl', 'opp_stl',
            'blk', 'opp_blk',
            'pts', 'opp_pts'
        ]
        
        for col in columns_to_add:
            add_column_if_not_exists(cursor, db, 'predictions', col)
            
        conn.commit()
        logger.info("🏁 Migração concluída com sucesso.")
        
    except Exception as e:
        logger.error(f"❌ Erro fatal durante a migração: {e}")
        conn.rollback()
    finally:
        db.return_connection(conn)

if __name__ == "__main__":
    migrate()
