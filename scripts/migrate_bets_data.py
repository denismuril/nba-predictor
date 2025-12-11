"""
Script de Migração: Dados de Apostas

Migra dados existentes de data/bets.db (SQLite legado) para o banco de dados
configurado via DB_TYPE (SQLite centralizado ou PostgreSQL).

Uso:
    python scripts/migrate_bets_data.py [--dry-run]
"""
import os
import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_bets_data(dry_run=False):
    """
    Migra dados de apostas do arquivo legado bets.db para o banco configurado.
    """
    # Paths
    legacy_db_path = Path(__file__).parent.parent / 'data' / 'bets.db'
    
    if not legacy_db_path.exists():
        logger.warning(f"⚠️  Arquivo legado não encontrado: {legacy_db_path}")
        logger.info("   Nenhuma migração necessária.")
        return 0
    
    logger.info(f"🔍 Conectando ao banco legado: {legacy_db_path}")
    
    # Conectar ao banco legado (SQLite)
    legacy_conn = sqlite3.connect(str(legacy_db_path))
    legacy_cursor = legacy_conn.cursor()
    
    # Verificar se tabela bets existe
    legacy_cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='bets'"
    )
    if not legacy_cursor.fetchone():
        logger.warning("⚠️  Tabela 'bets' não encontrada no banco legado")
        legacy_conn.close()
        return 0
    
    # Contar registros
    legacy_cursor.execute("SELECT COUNT(*) FROM bets")
    total_bets = legacy_cursor.fetchone()[0]
    
    if total_bets == 0:
        logger.info("✅ Banco legado está vazio. Nenhuma migração necessária.")
        legacy_conn.close()
        return 0
    
    logger.info(f"📊 Encontrados {total_bets} registros para migrar")
    
    # Buscar todos os dados
    legacy_cursor.execute("SELECT * FROM bets ORDER BY id")
    columns = [desc[0] for desc in legacy_cursor.description]
    rows = legacy_cursor.fetchall()
    
    if dry_run:
        logger.info("🔍 Modo DRY-RUN: Nenhum dado será inserido")
        logger.info(f"   Colunas: {', '.join(columns)}")
        logger.info(f"   Exemplo (primeiro registro): {rows[0] if rows else 'N/A'}")
        legacy_conn.close()
        return total_bets
    
    # Conectar ao banco de destino
    logger.info("🔄 Conectando ao banco de destino...")
    db = get_db_manager()
    db.init_db()  # Garante que tabela bets existe
    
    logger.info(f"📂 Banco de destino: {db.db_type.upper()}")
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    try:
        # Verificar registros existentes
        cursor.execute(db._prepare_query("SELECT COUNT(*) FROM bets"))
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            logger.warning(f"⚠️  Banco de destino já possui {existing_count} registros")
            response = input("   Continuar e adicionar registros? (s/N): ")
            if response.lower() != 's':
                logger.info("❌ Migração cancelada pelo usuário")
                legacy_conn.close()
                db.return_connection(conn)
                return 0
        
        # Preparar query de inserção (sem ID para autoincrement)
        cols_without_id = [c for c in columns if c != 'id']
        placeholders = ', '.join(['?' for _ in cols_without_id])
        cols_str = ', '.join(cols_without_id)
        
        insert_query = f"INSERT INTO bets ({cols_str}) VALUES ({placeholders})"
        
        logger.info(f"💾 Iniciando migração de {total_bets} registros...")
        
        migrated = 0
        errors = 0
        
        for row in rows:
            # Criar dict dos dados (sem ID)
            row_dict = dict(zip(columns, row))
            row_dict.pop('id', None)  # Remove ID (será gerado automaticamente)
            
            values = tuple(row_dict[col] for col in cols_without_id)
            
            try:
                cursor.execute(db._prepare_query(insert_query), values)
                migrated += 1
                
                if migrated % 100 == 0:
                    logger.info(f"   {migrated}/{total_bets} registros migrados...")
                    
            except Exception as e:
                errors += 1
                logger.error(f"   ❌ Erro no registro {row[0]}: {e}")
                if errors > 10:
                    logger.error("   ⚠️  Muitos erros! Abortando migração.")
                    conn.rollback()
                    raise
        
        # Commit
        conn.commit()
        logger.info(f"✅ Migração concluída!")
        logger.info(f"   - Migrados com sucesso: {migrated}")
        logger.info(f"   - Erros: {errors}")
        
        # Criar backup do arquivo legado
        if migrated > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = legacy_db_path.parent / f"bets_legacy_backup_{timestamp}.db"
            
            import shutil
            shutil.copy2(legacy_db_path, backup_path)
            logger.info(f"💾 Backup criado: {backup_path}")
            logger.info(f"   Você pode deletar {legacy_db_path} se tudo estiver OK")
        
        return migrated
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Erro durante migração: {e}")
        raise
    finally:
        legacy_conn.close()
        db.return_connection(conn)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migração de dados de apostas")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Simular migração sem inserir dados"
    )
    
    args = parser.parse_args()
    
    try:
        count = migrate_bets_data(dry_run=args.dry_run)
        
        if args.dry_run:
            print(f"\n✅ DRY-RUN: {count} registros seriam migrados")
        else:
            print(f"\n✅ Migração concluída: {count} registros migrados")
            
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Falha na migração: {e}")
        sys.exit(1)
