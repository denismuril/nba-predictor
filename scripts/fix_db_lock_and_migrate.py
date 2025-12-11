import sqlite3
import os
import shutil
import logging
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db_path = 'data/nba_history.db'
new_db_path = 'data/nba_history_fixed.db'
backup_path = 'data/nba_history.bak'

def add_column_if_not_exists(cursor, table, column, dtype="REAL DEFAULT 0"):
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
        logger.info(f"✅ Coluna '{column}' adicionada à tabela '{table}'.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            logger.info(f"ℹ️ Coluna '{column}' já existe na tabela '{table}'.")
        else:
            logger.error(f"❌ Erro ao adicionar coluna '{column}': {e}")

def fix_and_migrate():
    if not os.path.exists(db_path):
        logger.error(f"Banco de dados não encontrado em {db_path}")
        return

    logger.info(f"🔄 Copiando {db_path} para {new_db_path} para remover locks...")
    try:
        shutil.copy2(db_path, new_db_path)
    except Exception as e:
        logger.error(f"❌ Falha ao copiar banco de dados: {e}")
        return

    try:
        logger.info(f"🛠️ Aplicando migração em {new_db_path}...")
        conn = sqlite3.connect(new_db_path)
        cursor = conn.cursor()
        
        # Forçar modo WAL off e on para limpar estado
        cursor.execute('PRAGMA journal_mode=DELETE')
        cursor.execute('PRAGMA journal_mode=WAL')
        
        columns_to_add = [
            'ast', 'opp_ast',
            'stl', 'opp_stl',
            'blk', 'opp_blk',
            'pts', 'opp_pts'
        ]
        
        for col in columns_to_add:
            add_column_if_not_exists(cursor, 'predictions', col)
            
        conn.commit()
        conn.close()
        logger.info("✅ Migração aplicada com sucesso no novo arquivo.")
        
        # Substituir o arquivo original
        logger.info(f"🔄 Substituindo banco original...")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(db_path, backup_path)
        os.rename(new_db_path, db_path)
        
        logger.info("🏁 Banco de dados corrigido e migrado com sucesso!")
        logger.info(f"🔙 Backup salvo em {backup_path}")
        
    except Exception as e:
        logger.error(f"❌ Erro durante o processo de correção/migração: {e}")
        if os.path.exists(new_db_path):
            os.remove(new_db_path)

if __name__ == "__main__":
    fix_and_migrate()
