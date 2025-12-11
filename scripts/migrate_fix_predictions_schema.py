"""
Script para corrigir schema da tabela predictions adicionando game_id como PRIMARY KEY.

REFATORADO: Suporta SQLite e PostgreSQL via DatabaseManager.

NOTA: Este script faz uma migração complexa (recriar tabela). 
Use com cuidado e faça backup antes.
"""
import sys
import logging
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_schema():
    db = get_db_manager()
    logger.info(f"📂 Database: {db.db_type.upper()}")
    
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        logger.info("🚀 Iniciando migração do schema da tabela 'predictions'...")

        # 1. Verificar se game_id já é PK
        if db.db_type == 'sqlite':
            cursor.execute("PRAGMA table_info(predictions)")
            columns = cursor.fetchall()
            has_game_id_pk = any(col[1] == 'game_id' and col[5] == 1 for col in columns)
        else:  # postgres
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                WHERE tc.table_name = 'predictions'
                  AND tc.constraint_type = 'PRIMARY KEY'
                  AND ccu.column_name = 'game_id'
            """)
            has_game_id_pk = cursor.fetchone() is not None
        
        if has_game_id_pk:
            logger.info("✅ Tabela 'predictions' já possui game_id como PRIMARY KEY. Nenhuma ação necessária.")
            return

        logger.info("⚠️  Migrando schema...")

        if db.db_type == 'sqlite':
            # SQLite: Recriar tabela
            cursor.execute("ALTER TABLE predictions RENAME TO predictions_old")
            
            # Criar nova tabela (db_manager.init_db() já tem o schema correto)
            db.init_db()
            
            # Migrar dados
            cursor.execute("""
                INSERT INTO predictions (game_id, date, home_team, away_team, 
                                        prob_home, prob_away, prob_mc_home, prob_mc_away,
                                        odd_home, odd_away, prediction, confidence,
                                        predicted_spread, predicted_total, ci_lower, ci_upper,
                                        model_version, created_at)
                SELECT 
                    date || '_' || REPLACE(home_team, ' ', '') || '_' || REPLACE(away_team, ' ', ''),
                    date, home_team, away_team,
                    prob_home, prob_away, prob_mc_home, prob_mc_away,
                    odd_home, odd_away, prediction, confidence,
                    predicted_spread, predicted_total, ci_lower, ci_upper,
                    model_version, created_at
                FROM predictions_old
            """)
            
            cursor.execute("DROP TABLE predictions_old")
            
        else:  # postgres
            # PostgreSQL: Adicionar constraint
            cursor.execute("ALTER TABLE predictions ADD PRIMARY KEY (game_id)")
        
        conn.commit()
        logger.info("✅ Migração concluída com sucesso!")

    except Exception as e:
        logger.error(f"❌ Erro durante a migração: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        db.return_connection(conn)

if __name__ == "__main__":
    migrate_schema()
