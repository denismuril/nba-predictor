#!/usr/bin/env python3
"""
Importação Rápida de Dados Históricos via SportsBlaze API

Popula banco normalizado com dados históricos da temporada.
Usa API já configurada: sbfxqpy6v6fjljvobf61a5o
"""
import sys
from pathlib import Path

# Adicionar raiz ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

import logging
import time
from datetime import datetime, timedelta
from sportsblaze_integration import SportsBlazeClient
from data.repositories.db_manager import get_db_manager

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_sportsblaze_game(game_data):
    """
    Converte dados do SportsBlaze para formato do db_manager.
    
    Args:
        game_data: Dict com dados do jogo do Sportsblaze
        
    Returns:
        Tuple (game_data, home_stats, away_stats)
    """
    try:
        # Extrair informações básicas
        game_date = game_data.get('date', '')
        
        # Times
        home_team = game_data.get('homeTeam', {}).get('triCode', '')
        away_team = game_data.get('awayTeam', {}).get('triCode', '')
        
        # Placares
        home_score = game_data.get('homeTeam', {}).get('score', 0)
        away_score = game_data.get('awayTeam', {}).get('score', 0)
        
        # Game ID único
        game_id = f"{game_date}_{home_team}_{away_team}".replace(" ", "")
        
        # Determinar vencedor
        winner = 'HOME' if home_score > away_score else 'AWAY'
        
        # Dados do jogo
        game_info = {
            'id': game_id,
            'date': game_date,
            'season': '2024-25',  # Ajustar se necessário
            'home_team': home_team,
            'away_team': away_team,
            'home_score': int(home_score),
            'away_score': int(away_score),
            'winner': winner
        }
        
        # Stats do time da casa
        home_stats_raw = game_data.get('homeTeam', {}).get('statistics', {})
        home_stats = {
            'PTS': int(home_score),
            'FGM': home_stats_raw.get('fieldGoalsMade', 0),
            'FGA': home_stats_raw.get('fieldGoalsAttempted', 0),
            'FG_PCT': home_stats_raw.get('fieldGoalsPercentage', 0.0),
            'FG3M': home_stats_raw.get('threePointersMade', 0),
            'FG3A': home_stats_raw.get('threePointersAttempted', 0),
            'FG3_PCT': home_stats_raw.get('threePointersPercentage', 0.0),
            'FTM': home_stats_raw.get('freeThrowsMade', 0),
            'FTA': home_stats_raw.get('freeThrowsAttempted', 0),
            'FT_PCT': home_stats_raw.get('freeThrowsPercentage', 0.0),
            'OREB': home_stats_raw.get('reboundsOffensive', 0),
            'DREB': home_stats_raw.get('reboundsDefensive', 0),
            'REB': home_stats_raw.get('reboundsTotal', 0),
            'AST': home_stats_raw.get('assists', 0),
            'STL': home_stats_raw.get('steals', 0),
            'BLK': home_stats_raw.get('blocks', 0),
            'TOV': home_stats_raw.get('turnovers', 0),
            'PF': home_stats_raw.get('foulsPersonal', 0),
            'PLUS_MINUS': home_stats_raw.get('plusMinusPoints', 0)
        }
        
        # Stats do visitante
        away_stats_raw = game_data.get('awayTeam', {}).get('statistics', {})
        away_stats = {
            'PTS': int(away_score),
            'FGM': away_stats_raw.get('fieldGoalsMade', 0),
            'FGA': away_stats_raw.get('fieldGoalsAttempted', 0),
            'FG_PCT': away_stats_raw.get('fieldGoalsPercentage', 0.0),
            'FG3M': away_stats_raw.get('threePointersMade', 0),
            'FG3A': away_stats_raw.get('threePointersAttempted', 0),
            'FG3_PCT': away_stats_raw.get('threePointersPercentage', 0.0),
            'FTM': away_stats_raw.get('freeThrowsMade', 0),
            'FTA': away_stats_raw.get('freeThrowsAttempted', 0),
            'FT_PCT': away_stats_raw.get('freeThrowsPercentage', 0.0),
            'OREB': away_stats_raw.get('reboundsOffensive', 0),
            'DREB': away_stats_raw.get('reboundsDefensive', 0),
            'REB': away_stats_raw.get('reboundsTotal', 0),
            'AST': away_stats_raw.get('assists', 0),
            'STL': away_stats_raw.get('steals', 0),
            'BLK': away_stats_raw.get('blocks', 0),
            'TOV': away_stats_raw.get('turnovers', 0),
            'PF': away_stats_raw.get('foulsPersonal', 0),
            'PLUS_MINUS': away_stats_raw.get('plusMinusPoints', 0)
        }
        
        # Calcular Four Factors manualmente se não presentes
        for stats in [home_stats, away_stats]:
            if 'EFG_PCT' not in stats or stats.get('EFG_PCT') == 0:
                fga = max(stats.get('FGA', 1), 1)
                fgm = stats.get('FGM', 0)
                fg3m = stats.get('FG3M', 0)
                stats['EFG_PCT'] = (fgm + 0.5 * fg3m) / fga
                
            if 'TS_PCT' not in stats or stats.get('TS_PCT') == 0:
                pts = stats.get('PTS', 0)
                fga = stats.get('FGA', 0)
                fta = stats.get('FTA', 0)
                stats['TS_PCT'] = pts / (2 * max((fga + 0.44 * fta), 1))
        
        return game_info, home_stats, away_stats
        
    except Exception as e:
        logger.error(f"Erro ao parsear jogo: {e}")
        return None, None, None

def import_season_data(season='2024-25', start_date=None, end_date=None):
    """
    Importa dados de uma temporada via SportsBlaze.
    
    Args:
        season: Temporada (ex: '2024-25')
        start_date: Data início (datetime ou None)
        end_date: Data fim (datetime ou None)
    """
    logger.info("="*80)
    logger.info(f"🚀 IMPORTAÇÃO RÁPIDA - SPORTSBLAZE API")
    logger.info(f"   Temporada: {season}")
    logger.info("="*80)
    
    client = SportsBlazeClient()
    db = get_db_manager()
    
    # Definir intervalo de datas
    if not start_date:
        start_date = datetime(2024, 10, 22)  # Início temporada 2024-25
    if not end_date:
        end_date = datetime.now()
    
    logger.info(f"📅 Período: {start_date.date()} até {end_date.date()}")
    
    total_days = (end_date - start_date).days + 1
    logger.info(f"📊 Total de dias: {total_days}")
    
    inserted = 0
    skipped = 0
    errors = 0
    
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        try:
            # Buscar jogos do dia
            data = client.get_nba_boxscores(current_date)
            
            if data and 'games' in data:
                games = data['games']
                
                for game in games:
                    game_info, home_stats, away_stats = parse_sportsblaze_game(game)
                    
                    if game_info:
                        try:
                            db.insert_game_stats(game_info, home_stats, away_stats)
                            inserted += 1
                            logger.debug(f"✅ {game_info['id']}")
                        except Exception as e:
                            if 'UNIQUE constraint' in str(e) or 'duplicate key' in str(e):
                                skipped += 1
                            else:
                                logger.error(f"❌ Erro DB: {e}")
                                errors += 1
                    else:
                        errors += 1
                
                if games:
                    logger.info(f"📅 {date_str}: {len(games)} jogos | Inseridos: {inserted} | Duplicados: {skipped}")
            
            # Rate limiting (1 req/segundo)
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Erro em {date_str}: {e}")
            errors += 1
        
        current_date += timedelta(days=1)
    
    logger.info("="*80)
    logger.info("✅ IMPORTAÇÃO CONCLUÍDA")
    logger.info(f"   Inseridos: {inserted}")
    logger.info(f"   Duplicados: {skipped}")
    logger.info(f"   Erros: {errors}")
    logger.info("="*80)
    
    return inserted

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Importar dados históricos via SportsBlaze')
    parser.add_argument('--season', default='2024-25', help='Temporada (default: 2024-25)')
    parser.add_argument('--start', help='Data início (YYYY-MM-DD)')
    parser.add_argument('--end', help='Data fim (YYYY-MM-DD)')
    parser.add_argument('--last-days', type=int, help='Últimos N dias')
    
    args = parser.parse_args()
    
    start_date = None
    end_date = None
    
    if args.last_days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.last_days)
    else:
        if args.start:
            start_date = datetime.strptime(args.start, '%Y-%m-%d')
        if args.end:
            end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    count = import_season_data(season=args.season, start_date=start_date, end_date=end_date)
    
    sys.exit(0 if count > 0 else 1)
