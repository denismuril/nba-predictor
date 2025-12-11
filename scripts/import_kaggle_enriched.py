#!/usr/bin/env python3
"""
Script para enriquecer CSV do Kaggle com datas reais da NBA API.
Motivo: O dataset do Kaggle tem stats mas não tem a data do jogo.
"""
import sys
import os
import pandas as pd
import time
from pathlib import Path
from nba_api.stats.endpoints import leaguegamelog

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from data.repositories.db_manager import get_db_manager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_season_dates(season_str):
    """Busca datas dos jogos de uma temporada com retry."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"📥 Buscando calendário {season_str} (Tentativa {attempt+1}/{max_retries})...")
            log = leaguegamelog.LeagueGameLog(season=season_str, player_or_team_abbreviation='T', timeout=30)
            df = log.get_data_frames()[0]
            return df[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'TEAM_ABBREVIATION']]
        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar {season_str}: {e}")
            time.sleep(2 * (attempt + 1)) # Backoff: 2s, 4s, 6s
    
    logger.error(f"❌ Falha definitiva ao buscar {season_str}")
    return None

def enrich_and_import():
    csv_path = 'data/csv/final_data.csv'
    
    if not os.path.exists(csv_path):
        logger.error(f"Arquivo não encontrado: {csv_path}")
        return

    logger.info("📖 Lendo CSV do Kaggle...")
    df_kaggle = pd.read_csv(csv_path)
    
    # Normalizar Game ID (remover prefixo se houver, garantir string)
    # Kaggle IDs as vezes vem como int (21200001), NBA API usa string ('0021200001')
    df_kaggle['GAME_ID'] = df_kaggle['GAME_ID'].astype(str).str.zfill(10)
    
    # Identificar temporadas no CSV
    seasons = df_kaggle['SEASON'].unique()
    logger.info(f"📅 Temporadas encontradas: {seasons}")
    
    # Dicionário para mapear GameID -> Date
    game_date_map = {}
    
    # Buscar datas para cada temporada
    for season in seasons:
        # Converter formato Kaggle (12 ou 2012) para NBA API ('2012-13')
        season_start = int(season)
        if season_start < 100:
            season_start += 2000
            
        season_str = f"{season_start}-{str(season_start + 1)[-2:]}"
        
        df_dates = fetch_season_dates(season_str)
        if df_dates is not None:
            for _, row in df_dates.iterrows():
                game_date_map[row['GAME_ID']] = row['GAME_DATE']
        
        time.sleep(1) # Rate limit friendly
        
    logger.info(f"✅ Mapeamento de datas concluído: {len(game_date_map)} jogos mapeados.")
    
    # Aplicar datas ao DataFrame
    df_kaggle['GAME_DATE'] = df_kaggle['GAME_ID'].map(game_date_map)
    
    # Filtrar jogos sem data (provavelmente pre-season ou all-star não cobertos)
    missing = df_kaggle['GAME_DATE'].isna().sum()
    if missing > 0:
        logger.warning(f"⚠️ {missing} jogos sem data encontrada (serão ignorados).")
        df_kaggle = df_kaggle.dropna(subset=['GAME_DATE'])
        
    logger.info(f"🚀 Iniciando importação de {len(df_kaggle)} registros...")
    
    db = get_db_manager()
    inserted = 0
    
    # Agrupar por GameID para ter Home e Away
    # O dataset do Kaggle tem uma linha por time por jogo
    games_grouped = df_kaggle.groupby('GAME_ID')
    
    # Buffer para bulk insert
    games_buffer = []
    BATCH_SIZE = 1000
    
    for game_id, group in games_grouped:
        if len(group) != 2:
            continue # Precisa ter 2 times
            
        try:
            row1 = group.iloc[0]
            row2 = group.iloc[1]
            
            # Simplificação: Vamos confiar na coluna HOME_TEAM se existir
            # Se row1 é Home
            if row1.get('HOME_TEAM') == 1 or row1.get('HOME_TEAM') == '1':
                home_row, away_row = row1, row2
            else:
                home_row, away_row = row2, row1
                
            game_date = home_row['GAME_DATE']
            season = home_row['SEASON']
            season_str = f"{season}-{str(int(season)+1)[-2:]}"
            
            game_data = {
                'id': game_id,
                'date': game_date,
                'season': season_str,
                'home_team': home_row['TEAM_ABBREVIATION'],
                'away_team': away_row['TEAM_ABBREVIATION'],
                'home_score': int(home_row['PTS']),
                'away_score': int(away_row['PTS']),
                'winner': 'HOME' if home_row['PTS'] > away_row['PTS'] else 'AWAY'
            }
            
            # Mapear stats
            def map_stats(row):
                return {
                    'PTS': int(row['PTS']),
                    'FGM': int(row['FGM']), 'FGA': int(row['FGA']), 'FG_PCT': float(row['FG_PCT']),
                    'FG3M': int(row['FG3M']), 'FG3A': int(row['FG3A']), 'FG3_PCT': float(row['FG3_PCT']),
                    'FTM': int(row['FTM']), 'FTA': int(row['FTA']), 'FT_PCT': float(row['FT_PCT']),
                    'OREB': int(row['OREB']), 'DREB': int(row['DREB']), 'REB': int(row['REB']),
                    'AST': int(row['AST']), 'STL': int(row['STL']), 'BLK': int(row['BLK']),
                    'TOV': int(row['TO']), 'PF': int(row['PF'])
                }
            
            home_stats = map_stats(home_row)
            away_stats = map_stats(away_row)
            
            # Adicionar ao buffer
            games_buffer.append((game_data, home_stats, away_stats))
            
            # Executar Bulk Insert se atingir tamanho do lote
            if len(games_buffer) >= BATCH_SIZE:
                db.bulk_insert_games(games_buffer)
                inserted += len(games_buffer)
                logger.info(f"   Inseridos: {inserted} jogos...")
                games_buffer = [] # Limpar buffer
                
        except Exception as e:
            logger.error(f"Erro no jogo {game_id}: {e}")
            
    # Inserir remanescentes
    if games_buffer:
        db.bulk_insert_games(games_buffer)
        inserted += len(games_buffer)
            
    logger.info("="*80)
    logger.info(f"✅ Importação Finalizada! Total de jogos: {inserted}")

if __name__ == "__main__":
    enrich_and_import()
