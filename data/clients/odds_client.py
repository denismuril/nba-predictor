import requests
import logging
import os

logger = logging.getLogger(__name__)

def obter_odds_api(api_key):
    """Obtém odds da TheOddsAPI"""
    if not api_key or api_key == "SUA_CHAVE_AQUI":
        logger.warning("⚠️  ODDS_API_KEY não configurada.")
        return {}

    try:
        logger.info("🔍 Consultando TheOddsAPI...")
        sport_key = 'basketball_nba'
        regions = 'us,eu'
        markets = 'h2h'
        odds_format = 'decimal'
        date_format = 'iso'
        
        url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
        
        params = {
            'api_key': api_key,
            'regions': regions,
            'markets': markets,
            'odds_format': odds_format,
            'date_format': date_format,
        }
        
        r = requests.get(url, params=params, timeout=10)
        
        if r.status_code != 200:
            logger.error(f"❌ Erro TheOddsAPI: {r.status_code} - {r.text}")
            return {}
            
        data = r.json()
        
        # Processar dados para formato amigável
        odds_map = {}
        
        for game in data:
            home_team = game.get('home_team')
            bookmakers = game.get('bookmakers', [])
            
            if not bookmakers:
                continue
                
            # Buscar bookmakers disponíveis em ordem de prioridade
            # Betano não está disponível para NBA na TheOddsAPI
            # Prioridade: bet365 > draftkings > fanduel > williamhill > primeiro disponível
            preferred_bookmakers = ['bet365', 'draftkings', 'fanduel', 'williamhill_us', 'betmgm']
            
            selected_book = None
            for preferred in preferred_bookmakers:
                for book in bookmakers:
                    if book['key'] == preferred:
                        selected_book = book
                        break
                if selected_book:
                    break
            
            # Se nenhum preferido, pegar o primeiro
            if not selected_book and bookmakers:
                selected_book = bookmakers[0]
            
            if not selected_book:
                continue
            
            markets_data = selected_book.get('markets', [])
            for market in markets_data:
                if market['key'] == 'h2h':
                    outcomes = market.get('outcomes', [])
                    odd_home = 0
                    odd_away = 0
                    
                    for outcome in outcomes:
                        if outcome['name'] == home_team:
                            odd_home = outcome['price']
                        else:
                            odd_away = outcome['price']
                    
                    if odd_home > 0 and odd_away > 0:
                        # FIX: cli.py espera 'home' e 'away', não 'odd_home' e 'odd_away'
                        odds_map[home_team] = {
                            'home': odd_home,
                            'away': odd_away,
                            'bookmaker': selected_book['title']
                        }
        
        logger.info(f"✅ Odds obtidas para {len(odds_map)} jogos.")
        return odds_map

    except Exception as e:
        logger.error(f"❌ Erro ao consultar TheOddsAPI: {e}")
        return {}

def obter_odds():
    """Wrapper para obter odds com chave do ambiente"""
    from config.constants import ODDS_API_KEY
    return obter_odds_api(ODDS_API_KEY)
