#!/usr/bin/env python3
"""
Results Scraper - Busca resultados finalizados usando ESPN API + SportsGameOdds fallback
"""
import os
import logging
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from utils.team_normalization import normalize_team

# Carregar .env
load_dotenv()

logger = logging.getLogger(__name__)

# Mapeamento ESPN → NBA abreviações
TEAM_MAPPING = {
    'WSH': 'WAS',  # Washington
    'NY': 'NYK',   # New York
    'NO': 'NOP',   # New Orleans
    'GS': 'GSW',   # Golden State
    'UTAH': 'UTA', # Utah
    'SA': 'SAS',   # San Antonio (ESPN usa SA, precisamos de SAS)
    'PHO': 'PHX',  # Phoenix
    'BRK': 'BKN',  # Brooklyn
    'CHO': 'CHA',  # Charlotte
}


def get_game_results_nba_api(days_back=7):
    """
    Fallback: Busca resultados usando a biblioteca oficial nba_api.
    """
    try:
        from nba_api.stats.endpoints import scoreboardv2
        results = []
        today = datetime.now()
        
        logger.info(f"🏀 Tentando NBA API (Official) para os últimos {days_back} dias...")
        
        for day_offset in range(days_back):
            check_date = today - timedelta(days=day_offset)
            date_str = check_date.strftime('%Y-%m-%d')
            
            try:
                board = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=10)
                games = board.game_header.get_dict()['data']
                lines = board.line_score.get_dict()['data']
                
                # Create map of game_id -> score
                scores_map = {} # game_id -> {home_id: score, away_id: score}
                
                # Helper to find team abbreviation from ID is hard without extra call, 
                # but game_header has HOME_TEAM_ID and VISITOR_TEAM_ID.
                # standard team IDs are constant.
                
                # Parse Line Scores for points
                for line in lines:
                    # GAME_ID index 2, TEAM_ID index 3, PTS index 22
                    g_id = line[2]
                    t_id = line[3]
                    pts = line[22]
                    
                    if g_id not in scores_map: scores_map[g_id] = {}
                    scores_map[g_id][t_id] = int(pts) if pts is not None else 0

                for game in games:
                    # GAME_ID=2, GAME_STATUS_ID=4 (3=Final), 
                    # HOME_TEAM_ID=6, VISITOR_TEAM_ID=7
                    # HOME_TEAM_ID is raw ID. Need to map or just trust the order?
                    # ScoreboardV2 game_header doesn't have team abbr? 
                    # Wait, index 5 is GAME_DATE_EST, 6 is HOME_TEAM_ID, 7 is VISITOR_TEAM_ID
                    
                    game_id = game[2]
                    status_id = game[4]
                    
                    if status_id != 3: # Not Final
                        continue
                        
                    home_id = game[6]
                    away_id = game[7]
                    
                    # Need scores
                    if game_id not in scores_map:
                        continue
                        
                    home_score = scores_map[game_id].get(home_id, 0)
                    away_score = scores_map[game_id].get(away_id, 0)
                    
                    if home_score == 0 and away_score == 0:
                        continue

                    # Determine abbreviations. NBA API often needs another call for this, 
                    # but we can try to rely on the static teams list or hardcoded map if needed.
                    # Or simpler: The ESPN scraper is better for abbreviations. 
                    # BUT we can fetch from static.
                    from nba_api.stats.static import teams
                    try:
                        h_team_obj = teams.find_team_name_by_id(home_id)
                        a_team_obj = teams.find_team_name_by_id(away_id)
                        home_team_abbr = h_team_obj['abbreviation']
                        away_team_abbr = a_team_obj['abbreviation']
                    except:
                        continue # Cannot identify team
                        
                    # Normalize
                    home_team = normalize_team(home_team_abbr)
                    away_team = normalize_team(away_team_abbr)
                    
                    db_date = check_date.strftime('%Y-%m-%d')
                    # Fix: Remove spaces to match DB ID convention (e.g. "DallasMavericks")
                    db_game_id = f"{db_date}_{home_team}_{away_team}".replace(" ", "")
                    
                    results.append({
                        'id': db_game_id,
                        'date': db_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'game_id': game_id
                    })
                    
            except Exception as e:
                logger.debug(f"NBA API erro data {date_str}: {e}")
                continue
                
        if results:
            logger.info(f"✅ NBA API: {len(results)} jogos encontrados")
        
        return results

    except Exception as e:
        logger.warning(f"⚠️ NBA API Module erro: {e}")
        return []

def get_game_results_sportsgameodds(days_back=7):
    # ... legacy fallback ...
    pass # (We can keep it or remove, but prioritize NBA API)
    # Keeping the original code for SportsGameOdds below this new function or replacing the block
    # Logic in get_game_results will need to be updated to call this.
    return [] # Placeholder for now, I will modify get_game_results to call nba_api first.



def get_game_results(days_back=7):
    """
    Busca resultados de jogos finalizados dos últimos N dias usando ESPN API.
    
    Args:
        days_back: Número de dias para buscar retroativamente
        
    Returns:
        Lista de dicts com game_id, home_team, away_team, home_score, away_score, id
    """
    results = []
    
    try:
        logger.info(f"🔍 Buscando resultados dos últimos {days_back} dias via ESPN API...")
        
        today = datetime.now()
        
        for day_offset in range(days_back):
            check_date = today - timedelta(days=day_offset)
            date_str = check_date.strftime('%Y%m%d')
            
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                events = data.get('events', [])
                
                if not events:
                    logger.info(f"📅 {check_date.strftime('%Y-%m-%d')}: Sem jogos")
                    continue
                
                games_found = 0
                for event in events:
                    status = event.get('status', {})
                    state = status.get('type', {}).get('state', '')
                    
                    if state != 'post':
                        continue
                    
                    competitions = event.get('competitions', [])
                    if not competitions:
                        continue
                    
                    comp = competitions[0]
                    competitors = comp.get('competitors', [])
                    
                    if len(competitors) != 2:
                        continue
                    
                    home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                    away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                    
                    if not home or not away:
                        continue
                    
                    home_team = home.get('team', {}).get('abbreviation', 'UNK')
                    away_team = away.get('team', {}).get('abbreviation', 'UNK')
                    
                    home_team = TEAM_MAPPING.get(home_team, home_team)
                    away_team = TEAM_MAPPING.get(away_team, away_team)
                    
                    home_score = int(home.get('score', 0))
                    away_score = int(away.get('score', 0))
                    
                    if home_score == 0 and away_score == 0:
                        continue
                    
                    db_date = check_date.strftime('%Y-%m-%d')
                    db_game_id = f"{db_date}_{home_team}_{away_team}".replace(" ", "")
                    
                    result = {
                        'id': db_game_id,
                        'date': db_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'game_id': event.get('id', db_game_id)
                    }
                    
                    results.append(result)
                    games_found += 1
                    logger.info(f"✅ {db_date}: {away_team} @ {home_team} = {away_score}-{home_score}")
                
                if games_found > 0:
                    logger.info(f"📅 {check_date.strftime('%Y-%m-%d')}: {games_found} jogos finalizados")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️  Erro ao buscar data {check_date.strftime('%Y-%m-%d')}: {e}")
                continue
            except Exception as e:
                logger.warning(f"⚠️  Erro ao processar data {check_date.strftime('%Y-%m-%d')}: {e}")
                continue
        
        logger.info(f"📊 Total de {len(results)} jogos finalizados (ESPN)")
        
        # Se ESPN não retornou nada, tentar SportsGameOdds
        if not results:
            logger.info("⚠️  ESPN não retornou resultados, tentando fallback NBA API...")
            results = get_game_results_nba_api(days_back=days_back)
            
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro geral no scraper de resultados ESPN: {e}")
        import traceback
        traceback.print_exc()
        
        # Tentar fallback em caso de erro total
        logger.info("📡 Tentando fallback NBA API...")
        return get_game_results_nba_api(days_back=days_back)


def update_game_results(days_back=7):
    """
    Busca resultados e atualiza no banco de dados.
    
    Args:
        days_back: Número de dias retroativos para buscar
        
    Returns:
        Número de jogos atualizados
    """
    from data.repositories.db_manager import get_db_manager
    
    results = get_game_results(days_back=days_back)
    
    if not results:
        logger.info("Nenhum resultado novo para atualizar.")
        return 0
    
    db = get_db_manager()
    updated_count = 0
    
    for game in results:
        try:
            # Atualizar banco com placar final
            db.update_game_score(
                game_id=game['id'],
                home_score=game['home_score'],
                away_score=game['away_score']
            )
            updated_count += 1
        except Exception as e:
            logger.warning(f"⚠️ Erro ao atualizar {game['id']}: {e}")
            continue
    
    logger.info(f"✅ {updated_count} jogos atualizados no banco")
    return updated_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = get_game_results(days_back=3)
    print(f"\n📋 RESULTADOS ENCONTRADOS: {len(results)}")
    for r in results:
        print(f"{r['date']}: {r['away_team']} @ {r['home_team']} = {r['away_score']}-{r['home_score']}")
