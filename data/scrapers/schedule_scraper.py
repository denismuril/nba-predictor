import requests
import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.team_normalization import normalize_team

# Carregar .env
load_dotenv()

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


def obter_schedule_sportsgameodds(data_str):
    """
    Fallback: Busca schedule via SportsGameOdds API.
    
    Args:
        data_str: Data no formato YYYY-MM-DD ou YYYYMMDD
        
    Returns:
        Lista de dicts [{'home': 'LAL', 'away': 'GSW'}, ...] ou None
    """
    api_key = os.getenv('SPORTSGAMEODDS_API_KEY')
    
    if not api_key:
        logger.debug("SportsGameOdds: API key não configurada")
        return None
    
    try:
        logger.info("📡 Tentando SportsGameOdds API (Schedule fallback)...")
        
        url = "https://api.sportsgameodds.com/v2/events"
        
        params = {
            'leagueID': 'NBA',
            'oddsAvailable': 'true',
            'limit': 50
        }
        
        # Adicionar filtro de data se disponível
        if data_str:
            # Normalizar formato de data
            clean_date = data_str.replace("-", "")
            if len(clean_date) == 8:
                formatted = f"{clean_date[:4]}-{clean_date[4:6]}-{clean_date[6:8]}"
                params['startDate'] = formatted
        
        headers = {'x-api-key': api_key}
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Processar eventos - estrutura real: data[].teams.home/away.names
        games = []
        events = data.get('data', data.get('events', []))
        
        if isinstance(data, list):
            events = data
        
        for event in events:
            try:
                # Estrutura real: teams.home.names.short e teams.away.names.short
                teams = event.get('teams', {})
                home_info = teams.get('home', {})
                away_info = teams.get('away', {})
                
                # Extrair nomes (preferencialmente short/medium)
                home_names = home_info.get('names', {})
                away_names = away_info.get('names', {})
                
                home_name = home_names.get('short', home_names.get('medium', ''))
                away_name = away_names.get('short', away_names.get('medium', ''))
                
                # Fallback para estrutura alternativa
                if not home_name:
                    home_name = home_info.get('name', home_info.get('teamID', ''))
                if not away_name:
                    away_name = away_info.get('name', away_info.get('teamID', ''))
                
                if not home_name or not away_name:
                    continue
                
                # Normalizar nomes de times
                home_normalized = normalize_team(home_name)
                away_normalized = normalize_team(away_name)
                
                if home_normalized and away_normalized:
                    games.append({'home': home_normalized, 'away': away_normalized})
                else:
                    logger.debug(f"Não normalizado: {home_name} vs {away_name}")
                    
            except Exception as e:
                logger.debug(f"Erro ao parsear evento: {e}")
                continue
        
        if games:
            logger.info(f"✅ SportsGameOdds (Schedule): {len(games)} jogos encontrados")
            return games
        else:
            logger.warning("⚠️  SportsGameOdds retornou 0 jogos válidos")
            return None
            
    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code in (401, 403):
                logger.error("❌ SportsGameOdds: API key inválida!")
            else:
                logger.warning(f"⚠️  SportsGameOdds HTTP {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"⚠️  SportsGameOdds Schedule erro: {str(e)[:100]}")
        return None


def obter_schedule_api_espn(data_str):
    """Tenta ESPN API para schedule"""
    try:
        logger.info("🔍 Tentando ESPN API (Schedule)...")
        
        url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        
        params = {}
        if data_str:
            params['dates'] = data_str.replace("-", "")

        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        
        if r.status_code == 200:
            logger.info("✅ ESPN API (Schedule) funcionou!")
            data = r.json()
            events = data.get('events', [])
            
            # Parsear schedule
            games = []
            logger.info(f"DEBUG: Found {len(events)} events.")
            for event in events:
                try:
                    # Structure: event -> competitions[0] -> competitors
                    competitions = event.get('competitions', [])
                    if not competitions:
                        continue
                        
                    competitors = competitions[0].get('competitors', [])
                    if len(competitors) < 2:
                        continue
                        
                    # Extract team names
                    team1_info = competitors[0].get('team', {})
                    team2_info = competitors[1].get('team', {})
                    
                    team1 = team1_info.get('displayName', '')
                    team2 = team2_info.get('displayName', '')
                    
                    if not team1 or not team2:
                        continue

                    # Check home/away
                    if competitors[0].get('homeAway') == 'home':
                        home, away = team1, team2
                    else:
                        home, away = team2, team1
                    
                    # 🔧 CORREÇÃO: Normalizar nomes de times para IDs de 3 letras
                    home_normalized = normalize_team(home)
                    away_normalized = normalize_team(away)
                    
                    if not home_normalized or not away_normalized:
                        logger.warning(f"⚠️ Nome de time não normalizado - IGNORANDO jogo: {home} -> {home_normalized}, {away} -> {away_normalized}")
                        continue
                    
                    logger.debug(f"✅ Times normalizados: {home} -> {home_normalized}, {away} -> {away_normalized}")
                    games.append({"home": home_normalized, "away": away_normalized})
                except Exception as e:
                    logger.warning(f"DEBUG: Error parsing event: {e}")
                    pass
            
            if games:
                logger.info(f"✅ ESPN API (Schedule) encontrou {len(games)} jogos (nomes normalizados para IDs de 3 letras).")
                return games
            else:
                logger.warning("⚠️  ESPN API (Schedule) não encontrou jogos válidos após o parse.")
                return None
        else:
            logger.warning(f"⚠️  ESPN API Schedule retornou {r.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"⚠️  ESPN API Schedule erro: {str(e)[:100]}")
        return None

def obter_schedule(data_str):
    """Pipeline: ESPN API → SportsGameOdds Fallback"""
    logger.info("\n" + "="*80)
    logger.info("MÓDULO 2: SCHEDULE (Calendário de Jogos)")
    logger.info("="*80)
    
    # 1. Tentar ESPN API (fonte primária)
    schedule = obter_schedule_api_espn(data_str)
    if schedule:
        logger.info(f"✅ Schedule (ESPN): {len(schedule)} jogos")
        return schedule
    
    # 2. Fallback: SportsGameOdds API
    schedule = obter_schedule_sportsgameodds(data_str)
    if schedule:
        logger.info(f"✅ Schedule (SportsGameOdds): {len(schedule)} jogos")
        return schedule
    
    logger.error("❌ Nenhuma fonte de Schedule disponível!")
    return []
