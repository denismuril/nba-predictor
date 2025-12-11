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
}


def get_game_results_sportsgameodds(days_back=7):
    """
    Fallback: Busca resultados de jogos via SportsGameOdds API.
    
    Args:
        days_back: Número de dias para buscar retroativamente
        
    Returns:
        Lista de dicts com game_id, home_team, away_team, home_score, away_score
    """
    api_key = os.getenv('SPORTSGAMEODDS_API_KEY')
    
    if not api_key:
        logger.debug("SportsGameOdds: API key não configurada")
        return []
    
    results = []
    
    try:
        logger.info(f"📡 Tentando SportsGameOdds API (Results fallback)...")
        
        url = "https://api.sportsgameodds.com/v2/events"
        headers = {'x-api-key': api_key}
        
        today = datetime.now()
        
        for day_offset in range(days_back):
            check_date = today - timedelta(days=day_offset)
            date_str = check_date.strftime('%Y-%m-%d')
            
            params = {
                'leagueID': 'NBA',
                'startDate': date_str,
                'limit': 30
            }
            
            try:
                response = requests.get(url, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                events = data.get('data', data.get('events', []))
                if isinstance(data, list):
                    events = data
                
                for event in events:
                    try:
                        # Estrutura real: teams.home.names e teams.home.score
                        teams = event.get('teams', {})
                        home_info = teams.get('home', {})
                        away_info = teams.get('away', {})
                        
                        # Extrair scores da estrutura real
                        home_score = home_info.get('score')
                        away_score = away_info.get('score')
                        
                        # Fallback para outros campos
                        if home_score is None:
                            home_score = event.get('homeScore', event.get('home_score'))
                        if away_score is None:
                            away_score = event.get('awayScore', event.get('away_score'))
                        
                        if home_score is None or away_score is None:
                            continue
                        
                        home_score = int(home_score)
                        away_score = int(away_score)
                        
                        if home_score == 0 and away_score == 0:
                            continue
                        
                        # Extrair nomes - estrutura real: teams.home.names.short
                        home_names = home_info.get('names', {})
                        away_names = away_info.get('names', {})
                        
                        home_name = home_names.get('short', home_names.get('medium', ''))
                        away_name = away_names.get('short', away_names.get('medium', ''))
                        
                        # Fallback
                        if not home_name:
                            home_name = home_info.get('teamID', '')
                        if not away_name:
                            away_name = away_info.get('teamID', '')
                        
                        if not home_name or not away_name:
                            continue
                        
                        # Normalizar times
                        home_team = normalize_team(home_name)
                        away_team = normalize_team(away_name)
                        
                        if not home_team or not away_team:
                            continue
                        
                        db_date = check_date.strftime('%Y-%m-%d')
                        db_game_id = f"{db_date}_{home_team}_{away_team}"
                        
                        result = {
                            'id': db_game_id,
                            'date': db_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score,
                            'game_id': event.get('eventID', db_game_id)
                        }
                        
                        results.append(result)
                        logger.debug(f"✅ {db_date}: {away_team} @ {home_team} = {away_score}-{home_score}")
                        
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Erro ao parsear evento: {e}")
                        continue
                        
            except requests.exceptions.RequestException as e:
                logger.debug(f"Erro SportsGameOdds para {date_str}: {e}")
                continue
        
        if results:
            logger.info(f"✅ SportsGameOdds (Results): {len(results)} jogos encontrados")
        
        return results
        
    except Exception as e:
        logger.warning(f"⚠️  SportsGameOdds Results erro: {str(e)[:100]}")
        return []


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
            logger.info("⚠️  ESPN não retornou resultados, tentando fallback...")
            results = get_game_results_sportsgameodds(days_back=days_back)
            
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro geral no scraper de resultados ESPN: {e}")
        import traceback
        traceback.print_exc()
        
        # Tentar fallback em caso de erro total
        logger.info("📡 Tentando fallback SportsGameOdds...")
        return get_game_results_sportsgameodds(days_back=days_back)


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
