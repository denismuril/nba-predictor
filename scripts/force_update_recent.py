#!/usr/bin/env python3
"""
Script para FORÇAR a atualização de dados dos jogos recentes.
Ignora a verificação de 'já processado' e sobrescreve os stats no banco.
"""
import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

# Adicionar raiz do projeto ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from data.repositories.db_manager import get_db_manager
from nba_api.stats.endpoints import teamgamelog, boxscoretraditionalv3
from nba_api.stats.static import teams

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_derived_advanced_stats(trad_stats, opp_stats):
    try:
        fga = max(trad_stats.get('FGA', 0), 1)
        fgm = trad_stats.get('FGM', 0)
        fg3m = trad_stats.get('FG3M', 0)
        fta = trad_stats.get('FTA', 0)
        tov = trad_stats.get('TOV', 0)
        oreb = trad_stats.get('OREB', 0)
        opp_dreb = opp_stats.get('DREB', 0)
        
        efg_pct = (fgm + 0.5 * fg3m) / fga
        poss_est = fga + 0.44 * fta + tov
        tov_pct = tov / max(poss_est, 1)
        orb_pct = oreb / max((oreb + opp_dreb), 1)
        
        pts = trad_stats.get('PTS', 0)
        ts_pct = pts / (2 * max((fga + 0.44 * fta), 1))
        
        opp_pts = opp_stats.get('PTS', 0)
        off_rating = 100 * (pts / max(poss_est, 1))
        def_rating = 100 * (opp_pts / max(poss_est, 1))
        
        return {
            'OFF_RATING': round(off_rating, 1),
            'DEF_RATING': round(def_rating, 1),
            'EFG_PCT': round(efg_pct, 3),
            'TS_PCT': round(ts_pct, 3),
            'PACE': 0.0,
            'PIE': 0.0
        }
    except Exception as e:
        logger.warning(f"⚠️ Erro no cálculo manual de stats: {e}")
        return {}

def fetch_traditional_boxscore(game_id):
    for attempt in range(3):
        try:
            box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            frames = box.get_data_frames()
            if frames and len(frames) > 1:
                return frames[1] # Team Stats
        except Exception:
            time.sleep(1)
    return None

def force_update_recent(days=7):
    db = get_db_manager()
    nba_teams = teams.get_teams()
    cutoff_date = datetime.now() - timedelta(days=days)
    
    logger.info(f"🚀 Iniciando atualização FORÇADA para jogos desde {cutoff_date.strftime('%Y-%m-%d')}")
    
    for team in nba_teams:
        team_id = team['id']
        team_abbr = team['abbreviation']
        
        # if team_abbr != 'LAL': continue
        
        logger.info(f"Processing {team_abbr}...")
        try:
            gamelog = teamgamelog.TeamGameLog(team_id=team_id, season='2025-26')
            df_games = gamelog.get_data_frames()[0]
        except Exception as e:
            logger.error(f"Failed to fetch gamelog for {team_abbr}: {e}")
            continue
        
        if df_games.empty:
            continue
            
        for _, row in df_games.iterrows():
            try:
                dt_obj = datetime.strptime(row['GAME_DATE'], '%b %d, %Y')
                if dt_obj < cutoff_date:
                    continue
                logger.info(f"Found recent game: {dt_obj} (Cutoff: {cutoff_date})")
            except:
                continue
                
            nba_game_id = row['Game_ID']
            game_date_str = dt_obj.strftime('%Y-%m-%d')
            
            matchup = row['MATCHUP']
            if '@' in matchup:
                away_team = team_abbr
                home_team = matchup.split('@')[-1].strip()
            else:
                home_team = team_abbr
                away_team = matchup.split('vs.')[-1].strip()
            
            my_game_id = f"{game_date_str}_{home_team}_{away_team}"
            
            # Fetch Boxscore
            logger.info(f"Fetching boxscore for {my_game_id} (NBA ID: {nba_game_id})...")
            trad_df = fetch_traditional_boxscore(nba_game_id)
            if trad_df is None or trad_df.empty:
                logger.warning(f"❌ Boxscore vazio ou falhou para {my_game_id}")
                continue
            
            logger.info(f"✅ Boxscore obtido. Linhas: {len(trad_df)}")
            
            try:
                # Mapeamento V3 -> V2
                V3_TO_V2_MAP = {
                    'FIELDGOALSMADE': 'FGM',
                    'FIELDGOALSATTEMPTED': 'FGA',
                    'FIELDGOALSPERCENTAGE': 'FG_PCT',
                    'THREEPOINTERSMADE': 'FG3M',
                    'THREEPOINTERSATTEMPTED': 'FG3A',
                    'THREEPOINTERSPERCENTAGE': 'FG3_PCT',
                    'FREETHROWSMADE': 'FTM',
                    'FREETHROWSATTEMPTED': 'FTA',
                    'FREETHROWSPERCENTAGE': 'FT_PCT',
                    'REBOUNDSOFFENSIVE': 'OREB',
                    'REBOUNDSDEFENSIVE': 'DREB',
                    'REBOUNDSTOTAL': 'REB',
                    'ASSISTS': 'AST',
                    'STEALS': 'STL',
                    'BLOCKS': 'BLK',
                    'TURNOVERS': 'TOV',
                    'FOULSPERSONAL': 'PF',
                    'POINTS': 'PTS'
                }

                trad_stats_map = {}
                for _, t_row in trad_df.iterrows():
                    tid = t_row['teamId']
                    raw_dict = {k.upper(): v for k, v in t_row.to_dict().items()}
                    mapped_dict = {}
                    for k, v in raw_dict.items():
                        if k in V3_TO_V2_MAP:
                            mapped_dict[V3_TO_V2_MAP[k]] = v
                        else:
                            mapped_dict[k] = v
                    trad_stats_map[tid] = mapped_dict
                
                team_ids = list(trad_stats_map.keys())
                if len(team_ids) != 2:
                    continue
                
                home_id = None
                away_id = None
                for tid in team_ids:
                    tricode = trad_stats_map[tid].get('TEAMTRICODE')
                    # logger.info(f"Keys for {tricode}: {list(trad_stats_map[tid].keys())}")
                    if tricode == home_team:
                        home_id = tid
                        logger.info(f"Home Keys ({tricode}): {list(trad_stats_map[tid].keys())}")
                    elif tricode == away_team:
                        away_id = tid
                
                if not home_id or not away_id:
                    continue
                    
                final_home_stats = trad_stats_map[home_id]
                final_away_stats = trad_stats_map[away_id]
                
                final_home_stats.update(calculate_derived_advanced_stats(final_home_stats, final_away_stats))
                final_away_stats.update(calculate_derived_advanced_stats(final_away_stats, final_home_stats))
                
                def sanitize(val):
                    if hasattr(val, 'item'): return val.item()
                    return val

                game_data = {
                    'id': my_game_id,
                    'date': game_date_str,
                    'season': '2024-25',
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': sanitize(final_home_stats.get('PTS', 0)),
                    'away_score': sanitize(final_away_stats.get('PTS', 0)),
                    'winner': 'HOME' if sanitize(final_home_stats.get('PTS', 0)) > sanitize(final_away_stats.get('PTS', 0)) else 'AWAY'
                }
                
                final_home_stats = {k: sanitize(v) for k, v in final_home_stats.items()}
                final_away_stats = {k: sanitize(v) for k, v in final_away_stats.items()}
                
                logger.info(f"Stats Sample (Home): FGM={final_home_stats.get('FGM')}, FGA={final_home_stats.get('FGA')}, eFG={final_home_stats.get('EFG_PCT')}")
                
                db.insert_game_stats(game_data, final_home_stats, final_away_stats)
                
            except Exception as e:
                logger.error(f"Erro processando {my_game_id}: {e}")

if __name__ == "__main__":
    force_update_recent(days=7)
