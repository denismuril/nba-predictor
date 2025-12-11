import logging
import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from data.repositories.db_manager import get_db_manager

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_historical_stats():
    """
    Percorre todos os jogos no banco de dados e recalcula eFG% e TS%
    se estiverem zerados, usando os dados brutos (FGM, FGA, etc).
    """
    db = get_db_manager()
    conn = db.get_connection()
    cursor = conn.cursor()
    
    logger.info("🔧 Iniciando reparo de estatísticas históricas...")
    
    try:
        # 1. Buscar todos os jogos com stats
        # Precisamos de FGM, FGA, FG3M, FTA, PTS para calcular
        query = """
        SELECT 
            game_id, team_id, is_home,
            fgm, fga, fg3m, fta, pts,
            efg_pct, ts_pct
        FROM game_stats
        WHERE fga > 0  -- Só faz sentido se tiver tentativas de arremesso
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        logger.info(f"📊 Analisando {len(rows)} registros de estatísticas...")
        
        updates = []
        
        for row in rows:
            game_id, team_id, is_home, fgm, fga, fg3m, fta, pts, efg_pct, ts_pct = row
            
            needs_update = False
            new_efg = efg_pct
            new_ts = ts_pct
            
            # Corrigir eFG%
            if efg_pct == 0 or efg_pct is None:
                if fga > 0:
                    # Fórmula: (FGM + 0.5 * 3PM) / FGA
                    new_efg = (fgm + 0.5 * fg3m) / fga
                    needs_update = True
            
            # Corrigir TS%
            if ts_pct == 0 or ts_pct is None:
                if fga > 0:
                    # Fórmula: PTS / (2 * (FGA + 0.44 * FTA))
                    denom = 2 * (fga + 0.44 * fta)
                    if denom > 0:
                        new_ts = pts / denom
                        needs_update = True
            
            if needs_update:
                updates.append((new_efg, new_ts, game_id, team_id))
        
        if updates:
            logger.info(f"🔄 Atualizando {len(updates)} registros com valores corrigidos...")
            
            # Detectar placeholder correto baseando-se no tipo de banco
            # O db_manager geralmente expõe isso, mas vamos inferir
            placeholder = "%s" if db.db_type == 'postgres' else "?"
            
            update_query = f"""
            UPDATE game_stats
            SET efg_pct = {placeholder}, ts_pct = {placeholder}
            WHERE game_id = {placeholder} AND team_id = {placeholder}
            """
            
            # Batch update para performance
            cursor.executemany(update_query, updates)
            conn.commit()
            logger.info("✅ Atualização concluída com sucesso!")
        else:
            logger.info("✅ Nenhum registro precisou de correção.")
            
    except Exception as e:
        logger.error(f"❌ Erro ao corrigir estatísticas: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    fix_historical_stats()
