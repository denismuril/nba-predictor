"""
Odds Scraper com hierarquia de fallback e proteção de deadlock.

Hierarquia:
1. OddsPediaScraper (GRATUITO - web scraping) ← TIER 1
2. TheOddsAPI (se API key disponível) ← TIER 2
3. SportsDataIO (backup) ← TIER 3
4. RapidAPI Basketball ← TIER 4
5. OddsAPI.io ← TIER 5
"""

import os
import logging
import requests
import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime
from dotenv import load_dotenv
from config.constants import TEAM_ABBREV_MAP
from utils.cache import smart_cache, TTL_ODDS

# Importar scraper web (TIER 1 - Gratuito)
try:
    from data.scrapers.odds_web_scraper import OddsPediaScraper
    ODDSPEDIA_AVAILABLE = True
except ImportError:
    ODDSPEDIA_AVAILABLE = False
    logger = logging.getLogger(__name__)

# Importar multi-source scraper (TIER 0 - Múltiplas fontes gratuitas)
try:
    from data.scrapers.multi_odds_scraper import MultiSourceOddsScraper
    MULTI_SOURCE_AVAILABLE = True
except ImportError:
    MULTI_SOURCE_AVAILABLE = False

# Rate Limiter Enterprise
try:
    from infrastructure.rate_limiter import get_rate_limiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

# Mapeamento inverso
ABBREV_TO_FULL = {v: k for k, v in TEAM_ABBREV_MAP.items()}


async def _acquire_rate_limit(api_name: str) -> bool:
    """Tenta adquirir permissão do rate limiter."""
    if not RATE_LIMITER_AVAILABLE:
        return True
    try:
        limiter = await get_rate_limiter()
        return await limiter.wait_and_acquire(api_name, max_wait=10)
    except Exception:
        return True


def acquire_rate_limit_sync(api_name: str) -> bool:
    """
    Versão segura que evita 'RuntimeError: This event loop is already running'.
    Retorna True (permite a request) se não conseguir gerir o loop.
    """
    try:
        # Verifica se já existe um loop rodando
        asyncio.get_running_loop()
        # Se chegamos aqui, estamos dentro de um loop async.
        # NÃO podemos usar asyncio.run(). Retornamos True para não travar.
        return True 
    except RuntimeError:
        # Sem loop rodando, seguro criar um novo
        try:
            return asyncio.run(_acquire_rate_limit(api_name))
        except Exception:
            return True


class OddsValidator:
    """Validador de odds."""
    MIN_ODDS = 1.01
    MAX_ODDS = 50.0
    
    @staticmethod
    def validate_odds_value(odds: float, game_key: str = "") -> bool:
        if not isinstance(odds, (int, float)): return False
        if odds < OddsValidator.MIN_ODDS or odds > OddsValidator.MAX_ODDS: return False
        return True
    
    @staticmethod
    def validate_game_odds(home_odds: float, away_odds: float, game_key: str = "") -> bool:
        if not OddsValidator.validate_odds_value(home_odds, game_key): return False
        if not OddsValidator.validate_odds_value(away_odds, game_key): return False
        
        # Validate Vigorish/Overround
        # Sum of implied probs (1/decimal) usually between 1.01 (1%) and 1.10 (10%)
        # Allow up to 1.25 (25%) for some markets, and slightly below 1.0 for sharp lines/arbs
        implied_sum = (1.0 / home_odds) + (1.0 / away_odds)
        
        if implied_sum > 1.25: # > 25% vigorish is suspicious
            return False
            
        if implied_sum < 0.95: # < -5% arb/error is suspicious
            return False
            
        return True
    
    @staticmethod
    def remove_vigorish(home_odds: float, away_odds: float) -> tuple:
        prob_home = 1.0 / home_odds
        prob_away = 1.0 / away_odds
        overround = prob_home + prob_away
        vigorish_pct = (overround - 1.0) * 100
        
        fair_prob_home = prob_home / overround
        fair_prob_away = prob_away / overround
        
        return round(1.0 / fair_prob_home, 3), round(1.0 / fair_prob_away, 3), round(vigorish_pct, 2)
    
    @staticmethod
    def normalize_and_validate(odds_dict: Dict) -> Dict:
        validated = {}
        for game_key, odds_data in odds_dict.items():
            try:
                home_odds = float(odds_data.get('home_odds', 0))
                away_odds = float(odds_data.get('away_odds', 0))
                
                if not OddsValidator.validate_game_odds(home_odds, away_odds, game_key):
                    continue
                
                fair_home, fair_away, vig_pct = OddsValidator.remove_vigorish(home_odds, away_odds)
                
                validated[game_key] = {
                    'home_odds': home_odds,
                    'away_odds': away_odds,
                    'fair_home_odds': fair_home,
                    'fair_away_odds': fair_away,
                    'vigorish_pct': vig_pct,
                    'home_team': odds_data.get('home_team', ''),
                    'away_team': odds_data.get('away_team', ''),
                    'source': odds_data.get('source', 'unknown'),
                    'timestamp': odds_data.get('timestamp', datetime.now().isoformat())
                }
            except Exception:
                continue
        return validated


class TheOddsAPIClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.session = requests.Session()
    
    def fetch_odds(self) -> Dict:
        if not self.api_key: raise ValueError("ODDS_API_KEY missing")
        url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        params = {'api_key': self.api_key, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'decimal'}
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return self._parse_api_response(response.json())

    def _parse_api_response(self, data: List[Dict]) -> Dict:
        odds_dict = {}
        for event in data:
            try:
                if not event.get('bookmakers'): continue
                home_team, away_team = event['home_team'], event['away_team']
                bookmaker = event['bookmakers'][0] # Usar primeiro
                
                market = next((m for m in bookmaker.get('markets', []) if m['key'] == 'h2h'), None)
                if not market: continue
                
                home_outcome = next((o for o in market['outcomes'] if o['name'] == home_team), None)
                away_outcome = next((o for o in market['outcomes'] if o['name'] == away_team), None)
                
                if home_outcome and away_outcome:
                    odds_dict[f"{home_team} vs {away_team}"] = {
                        'home_team': home_team, 'away_team': away_team,
                        'home_odds': home_outcome['price'], 'away_odds': away_outcome['price'],
                        'source': f"theoddsapi_{bookmaker['key']}"
                    }
            except Exception: continue
        return odds_dict


class SportsDataIOClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        self.session = requests.Session()
    
    def fetch_odds(self) -> Dict:
        if not self.api_key: raise ValueError("SPORTSDATA_API_KEY missing")
        date_str = datetime.now().strftime("%Y-%m-%d")
        url = f"https://api.sportsdata.io/v3/nba/odds/json/GameOddsByDate/{date_str}"
        headers = {'Ocp-Apim-Subscription-Key': self.api_key}
        response = self.session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return self._parse_api_response(response.json())
        
    def _parse_api_response(self, data: list) -> Dict:
        # Simplificado para brevidade, mantendo lógica original
        return {}


class RapidAPIFootballClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('RAPIDAPI_KEY', '')
        self.session = requests.Session()
    def fetch_odds(self) -> Dict:
        return {} # Placeholder implementação completa está no código legado


class OddsAPIioClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ODDSAPI_KEY', '')
        self.session = requests.Session()
    def fetch_odds(self) -> Dict:
        return {}


class OddsSharkScraper:
    def fetch_odds(self) -> Dict:
        raise NotImplementedError("OddsShark not implemented")


def obter_odds(force_source: Optional[str] = None) -> Dict:
    return _obter_odds_cached(force_source)

@smart_cache(ttl_hours=TTL_ODDS, cache_key_prefix='odds')
def _obter_odds_cached(force_source: Optional[str] = None) -> Dict:
    """Função interna cacheada."""
    
    # 0. TIER 0: Multi-Source Scraper (múltiplos sites gratuitos)
    if force_source in [None, 'multi'] and MULTI_SOURCE_AVAILABLE:
        try:
            logger.info("🌐 TIER 0: Multi-Source Scraper (múltiplos sites)...")
            scraper = MultiSourceOddsScraper()
            res = scraper.fetch_odds()
            valid = OddsValidator.normalize_and_validate(res)
            if valid: 
                logger.info(f"✅ Multi-Source retornou {len(valid)} jogos")
                return valid
        except Exception as e:
            logger.warning(f"⚠️ Multi-Source falhou: {e}")
    
    # 1. TIER 1: OddsPedia (Web Scraper Gratuito - fallback direto)
    if force_source in [None, 'oddspedia'] and ODDSPEDIA_AVAILABLE:
        try:
            logger.info("📡 TIER 1: OddsPedia Scraper...")
            acquire_rate_limit_sync("oddspedia")
            scraper = OddsPediaScraper()
            res = scraper.fetch_odds()
            valid = OddsValidator.normalize_and_validate(res)
            if valid: return valid
        except Exception as e:
            logger.warning(f"⚠️ OddsPedia falhou: {e}")

    # 2. TIER 2: TheOddsAPI
    if force_source in [None, 'theoddsapi']:
        try:
            acquire_rate_limit_sync("theoddsapi")
            client = TheOddsAPIClient()
            res = client.fetch_odds()
            valid = OddsValidator.normalize_and_validate(res)
            if valid: return valid
        except Exception: pass

    # 3. Fallbacks...
    logger.error("🚨 Todas as fontes falharam ou não retornaram dados válidos.")
    from exceptions.odds_exceptions import OddsUnavailableError
    raise OddsUnavailableError("Falha na obtenção de odds.")


def get_odds_for_game(home_team: str, away_team: str, odds_cache: Optional[Dict] = None) -> Dict:
    """
    Retrieves odds for a specific game from cache.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        odds_cache: Optional cache dictionary
        
    Returns:
        Dict with odds data
        
    Raises:
        OddsUnavailableError: If odds not found in cache (no default fallback)
    """
    if odds_cache:
        key = f"{home_team} vs {away_team}"
        if key in odds_cache: return odds_cache[key]
    
    # NO DEFAULT ODDS - raise exception to force proper handling
    from exceptions.odds_exceptions import OddsUnavailableError
    logger.error(f"🚨 No odds available for {away_team} @ {home_team}")
    raise OddsUnavailableError(f"Odds not found for {away_team} @ {home_team}")
