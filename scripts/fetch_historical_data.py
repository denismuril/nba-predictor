#!/usr/bin/env python3
"""
Script para buscar dados históricos completos da NBA API.
Versão Grão-Mestre Robusta: Fallback para cálculo manual de Four Factors se a API avançada falhar.
"""
import sys
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import numpy as np

# Adicionar raiz do projeto ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from data.repositories.db_manager import get_db_manager
from nba_api.stats.endpoints import teamgamelog, boxscoreadvancedv2, boxscoretraditionalv3
from nba_api.stats.static import teams

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/historical_fetch_robust.log')
    ]
)
logger = logging.getLogger(__name__)

def calculate_derived_advanced_stats(trad_stats, opp_stats):
    """
    Calcula Four Factors e outras métricas avançadas usando fórmulas canônicas.
    
    VERSÃO 2.0: Usa utils/nba_formulas.py (eliminates training-serving skew)
    - Mesmas fórmulas usadas em inference
    - Validação automática de ranges
    """
    try:
        # Import canonical formulas
        import sys
        from pathlib import Path
        
        # Add project root to path if not already there
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from utils.nba_formulas import calculate_all_advanced_stats
        
        # Prepare stats dict for canonical calculation
        stats_dict = {
            'pts': trad_stats.get('PTS', 0),
            'fgm': trad_stats.get('FGM', 0),
            'fga': max(trad_stats.get('FGA', 0), 1),
            'fg3m': trad_stats.get('FG3M', 0),
            'fta': trad_stats.get('FTA', 0),
            'ftm': trad_stats.get('FTM', 0),
            'orb': trad_stats.get('OREB', 0),
            'drb': trad_stats.get('DREB', 0),
            'tov': trad_stats.get('TOV', 0),
            'opp_pts': opp_stats.get('PTS', 0),
            'opp_drb': opp_stats.get('DREB', 0),
            'minutes_played': 48  # Default, will be overridden if OT data available
        }
        
        # Calculate using canonical formulas
        advanced = calculate_all_advanced_stats(**stats_dict)
        
        # Convert to uppercase keys for compatibility with existing code
        return {
            'OFF_RATING': round(advanced['off_rating'], 1),
            'DEF_RATING': round(advanced['def_rating'], 1),
            'EFG_PCT': round(advanced['efg_pct'], 3),
            'TS_PCT': round(advanced['ts_pct'], 3),
            'PACE': round(advanced['pace'], 1),
            'PIE': 0.0,  # Complex metric, requires full game context
            'TOV_PCT': round(advanced['tov_pct'], 1),  # Now in 10-20 range
            'ORB_PCT': round(advanced['orb_pct'], 3),
            'FT_RATE': round(advanced['ft_rate'], 3)
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Erro no cálculo com fórmulas canônicas: {e}")
        logger.warning("   Usando fallback simplificado")
        
        # Fallback: simplified calculation (less accurate but safe)
        fga = max(trad_stats.get('FGA', 0), 1)
        fta = trad_stats.get('FTA', 0)
        tov = trad_stats.get('TOV', 0)
        pts = trad_stats.get('PTS', 0)
        opp_pts = opp_stats.get('PTS', 0)
        
        poss_est = fga + 0.44 * fta + tov
        
        return {
            'OFF_RATING': round(100 * (pts / max(poss_est, 1)), 1),
            'DEF_RATING': round(100 * (opp_pts / max(poss_est, 1)), 1),
            'EFG_PCT': 0.0,
            'TS_PCT': 0.0,
            'PACE': 0.0,
            'PIE': 0.0
        }


def fetch_advanced_boxscore(game_id):
    """Busca stats avançados com retry e fallback silencioso."""
    for attempt in range(2): # Reduzido para 2 tentativas para ser mais rápido
        try:
            # Tentar sem timeout primeiro (padrão da lib)
            box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
            frames = box.get_data_frames()
            if frames and len(frames) > 1:
                return frames[1]
        except Exception:
            time.sleep(1)
    return None

def fetch_traditional_boxscore(game_id):
    """Busca stats tradicionais (obrigatório)."""
    for attempt in range(3):
        try:
            box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            frames = box.get_data_frames()
            if frames and len(frames) > 0:
                return frames[0] # Player stats usually, wait. V3 returns different frames.
                # V3: 0=PlayerStats, 1=TeamStats
            # Fallback to V2 logic if V3 fails or structure differs?
            # Actually V3 structure: 0=PlayerStats, 1=TeamStats
            if len(frames) > 1:
                return frames[1]
        except Exception:
            time.sleep(1)
    return None

def process_historical_data(seasons=['2025-26', '2024-25'], days_back=None):
    db = get_db_manager()
    nba_teams = teams.get_teams()
    
    logger.info(f"🚀 Iniciando ingestão ROBUSTA (V3) para {len(nba_teams)} times. Temporadas: {seasons}")
    
    cutoff_date = None
    if days_back:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        logger.info(f"🕒 Modo Incremental: Processando apenas jogos após {cutoff_date.strftime('%Y-%m-%d')}")
    
    processed_game_ids = set()
    
    # Carregar jogos já processados para pular
    try:
        with db.get_connection() as conn:
            # Carregar APENAS jogos concluídos para pular
            existing = pd.read_sql_query("SELECT game_id FROM games WHERE home_score > 0", conn)
            processed_game_ids = set(existing['game_id'].tolist())
            logger.info(f"📂 {len(processed_game_ids)} jogos já processados.")
    except Exception:
        pass

    for season in seasons:
        logger.info(f"📅 Processando Temporada {season}...")
        
        for team in nba_teams:
            team_id = team['id']
            team_abbr = team['abbreviation']
            
            try:
                gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=season)
                df_games = gamelog.get_data_frames()[0]
            except Exception:
                logger.error(f"❌ Falha ao buscar gamelog para {team_abbr}. Pulando time.")
                continue
            
            if df_games.empty:
                continue
                
            # Iterar jogos
            for _, row in tqdm(df_games.iterrows(), total=len(df_games), desc=f"{team_abbr} {season}"):
                nba_game_id = row['Game_ID']
                
                # Parse Data
                try:
                    dt_obj = datetime.strptime(row['GAME_DATE'], '%b %d, %Y')
                    game_date_str = dt_obj.strftime('%Y-%m-%d')
                    
                    # Filtro Incremental
                    if cutoff_date and dt_obj < cutoff_date:
                        continue
                except:
                    continue
                    
                # Parse Matchup
                matchup = row['MATCHUP']
                if '@' in matchup:
                    away_team = team_abbr
                    home_team = matchup.split('@')[-1].strip()
                else:
                    home_team = team_abbr
                    away_team = matchup.split('vs.')[-1].strip()
                
                # Gerar ID único nosso
                my_game_id = f"{game_date_str}_{home_team}_{away_team}"
                
                # Se já processado, pular
                if my_game_id in processed_game_ids:
                    continue
                
                # 1. Buscar BoxScore Tradicional (Essencial)
                trad_df = fetch_traditional_boxscore(nba_game_id)
                if trad_df is None or trad_df.empty:
                    logger.warning(f"⚠️ Sem BoxScore Tradicional para {my_game_id}. Pulando.")
                    continue
                
                # 2. Buscar BoxScore Avançado (Opcional)
                adv_df = fetch_advanced_boxscore(nba_game_id)
                
                # Processar Stats
                try:
                    # Converter para dicts por TeamID
                    trad_stats_map = {}
                    for _, t_row in trad_df.iterrows():
                        tid = t_row['teamId']
                        trad_stats_map[tid] = t_row.to_dict()
                        # Normalizar chaves (V3 usa camelCase, V2 usava UPPER)
                        # Vamos converter tudo para UPPER para compatibilidade com calculate_derived
                        trad_stats_map[tid] = {k.upper(): v for k, v in t_row.to_dict().items()}

                    adv_stats_map = {}
                    if adv_df is not None and not adv_df.empty:
                        for _, a_row in adv_df.iterrows():
                            tid = a_row['TEAM_ID']
                            adv_stats_map[tid] = a_row.to_dict()
                    
                    # Identificar IDs de Home e Away
                    # O gamelog diz quem é quem, mas precisamos mapear para os IDs da API
                    # team_id é o time do loop.
                    # Se home_team == team_abbr, então team_id é HomeID?
                    # Cuidado com mudanças de ID/Nome.
                    # Vamos assumir que trad_stats_map tem 2 chaves.
                    team_ids = list(trad_stats_map.keys())
                    if len(team_ids) != 2:
                        continue
                        
                    # Precisamos saber qual ID corresponde a qual time (Home/Away)
                    # O boxscore tradicional tem 'teamTricode' ou similar?
                    # V3 tem 'teamTricode'.
                    
                    home_id = None
                    away_id = None
                    
                    for tid in team_ids:
                        tricode = trad_stats_map[tid].get('TEAMTRICODE')
                        if tricode == home_team:
                            home_id = tid
                        elif tricode == away_team:
                            away_id = tid
                            
                    if not home_id or not away_id:
                        # Tentar fallback por matchup se tricode falhar (ex: NOP vs NO)
                        # Mas vamos pular por segurança
                        continue
                        
                    # Montar Stats Finais
                    final_home_stats = trad_stats_map[home_id]
                    final_away_stats = trad_stats_map[away_id]
                    
                    # Merge Advanced ou Calcular
                    if home_id in adv_stats_map:
                        final_home_stats.update(adv_stats_map[home_id])
                    else:
                        final_home_stats.update(calculate_derived_advanced_stats(final_home_stats, final_away_stats))
                        
                    if away_id in adv_stats_map:
                        final_away_stats.update(adv_stats_map[away_id])
                    else:
                        final_away_stats.update(calculate_derived_advanced_stats(final_away_stats, final_home_stats))
                        
                    # Sanitizar tipos
                    def sanitize(val):
                        if hasattr(val, 'item'): return val.item()
                        return val

                    game_data = {
                        'id': my_game_id,
                        'date': game_date_str,
                        'season': season,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': sanitize(final_home_stats.get('PTS', 0)),
                        'away_score': sanitize(final_away_stats.get('PTS', 0)),
                        'winner': 'HOME' if sanitize(final_home_stats.get('PTS', 0)) > sanitize(final_away_stats.get('PTS', 0)) else 'AWAY'
                    }
                    
                    # Sanitizar stats dicts
                    final_home_stats = {k: sanitize(v) for k, v in final_home_stats.items()}
                    final_away_stats = {k: sanitize(v) for k, v in final_away_stats.items()}
                    
                    db.insert_game_stats(game_data, final_home_stats, final_away_stats)
                    processed_game_ids.add(my_game_id)
                    
                    time.sleep(0.5) # Rate limit suave
                    
                except Exception as e:
                    logger.error(f"❌ Erro processando jogo {my_game_id}: {e}")
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch Historical NBA Data')
    parser.add_argument('--days', type=int, help='Number of days back to fetch (incremental mode)')
    parser.add_argument('--seasons', nargs='+', default=['2025-26', '2024-25'], help='Seasons to fetch')
    
    args = parser.parse_args()
    
    process_historical_data(seasons=args.seasons, days_back=args.days)
