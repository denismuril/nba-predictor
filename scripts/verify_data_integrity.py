import sqlite3
import pandas as pd
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

db_path = 'data/nba_history.db'

def verify_data():
    try:
        conn = sqlite3.connect(db_path)
        
        # Verificar contagem total de jogos com stats preenchidos
        count_filled = pd.read_sql("SELECT COUNT(*) as total FROM predictions WHERE pts > 0 AND ast > 0", conn).iloc[0]['total']
        total_count = pd.read_sql("SELECT COUNT(*) as total FROM predictions", conn).iloc[0]['total']
        
        logger.info(f"Total de registros: {total_count}")
        logger.info(f"Registros com stats preenchidos (pts > 0 e ast > 0): {count_filled}")
        
        if count_filled == 0:
            logger.error("❌ Nenhum jogo encontrado com estatísticas preenchidas!")
            return

        # Verificar amostra de jogos VÁLIDOS (com pontos)
        query = """
        SELECT id, date, home_team, away_team, 
               fgm, fga, fg3m, ftm, fta, oreb, dreb, ast, stl, blk, pts,
               opp_fgm, opp_fga, opp_fg3m, opp_ftm, opp_fta, opp_oreb, opp_dreb, opp_ast, opp_stl, opp_blk, opp_pts
        FROM predictions 
        WHERE pts > 0
        ORDER BY date DESC 
        LIMIT 10
        """
        df = pd.read_sql(query, conn)
        
        if df.empty:
            logger.warning("⚠️ Nenhuma linha encontrada com pts > 0.")
        else:
            logger.info("🔍 Amostra dos últimos 10 jogos COM DADOS:")
            # Mostrar apenas algumas colunas para caber na tela
            cols_to_show = ['date', 'home_team', 'away_team', 'pts', 'ast', 'reb', 'stl', 'blk']
            # Adicionar reb se não existir na query (somando oreb + dreb)
            if 'reb' not in df.columns:
                df['reb'] = df['oreb'] + df['dreb']
            
            print(df[cols_to_show].to_string(index=False))
            
            # Verificar integridade
            zeros = df[['ast', 'pts']].eq(0).sum().sum()
            if zeros > 0:
                logger.warning(f"⚠️ Encontrados {zeros} valores zerados em colunas críticas na amostra de jogos finalizados!")
            else:
                logger.info("✅ Dados históricos parecem corretos e preenchidos.")

        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Erro ao verificar dados: {e}")

if __name__ == "__main__":
    verify_data()
