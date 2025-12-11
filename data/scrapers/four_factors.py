"""
Fetch box score data for calculating Four Factors (Dean Oliver).
"""
import logging
import pandas as pd
from config.constants import TEAM_ABBREV_MAP

logger = logging.getLogger(__name__)

def get_team_four_factors(season='2025-26'):
    """
    Busca box scores da NBA API e calcula os Four Factors para cada time.
    
    Four Factors (Dean Oliver):
    1. eFG% (Effective Field Goal %): (FGM + 0.5 * 3PM) / FGA
    2. TOV% (Turnover %): TOV / (FGA + 0.44 * FTA + TOV)
    3. ORB% (Offensive Rebound %): ORB / (ORB + Opp DRB)
    4. FTR (Free Throw Rate): FTA / FGA
    
    Returns:
    --------
    dict : {team_abbr: {'efg': float, 'tov_pct': float, 'orb_pct': float, 'ftr': float}}
    """
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        
        logger.info(f"🔍 Buscando Four Factors para temporada {season}...")
        
        # Buscar estatísticas gerais
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed='PerGame'
        ).get_data_frames()[0]
        
        if stats.empty:
            logger.warning("⚠️  Nenhum dado retornado pela NBA API")
            return {}
        
        four_factors = {}
        
        for _, row in stats.iterrows():
            team_abbr = row.get('TEAM_ABBREVIATION', '')
            
            # Extrair stats necessárias
            fgm = float(row.get('FGM', 0))
            fga = float(row.get('FGA', 0))
            fg3m = float(row.get('FG3M', 0))
            tov = float(row.get('TOV', 0))
            orb = float(row.get('OREB', 0))
            drb = float(row.get('DREB', 0))  # Defensive rebounds do próprio time
            fta = float(row.get('FTA', 0))
            
            # Calcular Four Factors
            # 1. eFG%
            efg = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0
            
            # 2. TOV%
            tov_pct = tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) > 0 else 0.0
            
            # 3. ORB% - Nota: Idealmente seria ORB / (ORB + Opp DRB), mas DRB do oponente não está disponível
            # Como aproximação, usamos ORB / (ORB + DRB próprio)
            orb_pct = orb / (orb + drb) if (orb + drb) > 0 else 0.0
            
            # 4. FTR
            ftr = fta / fga if fga > 0 else 0.0
            
            four_factors[team_abbr] = {
                'efg': round(efg, 4),
                'tov_pct': round(tov_pct, 4),
                'orb_pct': round(orb_pct, 4),
                'ftr': round(ftr, 4)
            }
        
        logger.info(f"✅ Four Factors calculados para {len(four_factors)} times")
        return four_factors
        
    except ImportError:
        logger.error("❌ nba_api não instalada. Instale com: pip install nba-api")
        return {}
    except Exception as e:
        logger.error(f"❌ Erro ao buscar Four Factors: {e}")
        return {}


def get_team_box_scores_history(team_abbr, last_n_games=10, season='2025-26'):
    """
    Busca histórico de box scores recentes de um time para calcular Four Factors rolantes.
    
    Parameters:
    -----------
    team_abbr : str
        Abreviação do time (ex: 'LAL')
    last_n_games : int
        Número de jogos recentes a buscar
    season : str
        Temporada (ex: '2025-26')
        
    Returns:
    --------
    pd.DataFrame com colunas: ['date', 'fgm', 'fga', 'fg3m', 'tov', 'orb', 'fta']
    """
    try:
        from nba_api.stats.endpoints import teamgamelog
        from nba_api.stats.static import teams
        
        # Encontrar ID do time
        nba_teams = teams.get_teams()
        team_id = None
        for t in nba_teams:
            if t['abbreviation'] == team_abbr:
                team_id = t['id']
                break
        
        if not team_id:
            logger.warning(f"⚠️  Time {team_abbr} não encontrado")
            return pd.DataFrame()
        
        # Buscar game log
        log = teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
        
        if log.empty:
            return pd.DataFrame()
        
        # Selecionar últimos N jogos
        log = log.head(last_n_games)
        
        # Extrair colunas relevantes
        box_scores = pd.DataFrame({
            'date': pd.to_datetime(log['GAME_DATE']),
            'fgm': log['FGM'],
            'fga': log['FGA'],
            'fg3m': log['FG3M'],
            'tov': log['TOV'],
            'orb': log['OREB'],
            'drb': log['DREB'],
            'fta': log['FTA']
        })
        
        return box_scores.sort_values('date')
        
    except Exception as e:
        logger.warning(f"⚠️  Erro ao buscar box scores de {team_abbr}: {e}")
        return pd.DataFrame()
