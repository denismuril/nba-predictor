"""
Odds Scraper com hierarquia de fallback.

Hierarquia:
1. TheOddsAPI (se API key disponível)
2. Odds Shark scraping (backup)
3. Default 1.90 (último recurso com WARNING)
"""

import os
import logging
import requests
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
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
    # Logger será definido abaixo, ignorar aqui

# CRITICAL: Carregar .env no escopo global
# Isso garante que as variáveis estejam disponíveis independente de quem importou
load_dotenv()

logger = logging.getLogger(__name__)

# Mapeamento inverso para converter abreviações para nomes completos
ABBREV_TO_FULL = {v: k for k, v in TEAM_ABBREV_MAP.items()}


class OddsValidator:
    """Validador de odds."""
    
    MIN_ODDS = 1.01
    MAX_ODDS = 50.0
    PROB_SUM_TOLERANCE = 0.05  # 5% tolerance para soma de probabilidades implícitas
    
    @staticmethod
    def validate_odds_value(odds: float, game_key: str = "") -> bool:
        """
        Valida se o valor de odds está em range razoável.
        
        Args:
            odds: Valor das odds
            game_key: Identificador do jogo (para logging)
            
        Returns:
            True se válido, False caso contrário
        """
        if not isinstance(odds, (int, float)):
            logger.warning(f"⚠️  Odds inválido para {game_key}: não é numérico ({type(odds)})")
            return False
        
        if odds < OddsValidator.MIN_ODDS:
            logger.warning(f"⚠️  Odds muito baixo para {game_key}: {odds} < {OddsValidator.MIN_ODDS}")
            return False
        
        if odds > OddsValidator.MAX_ODDS:
            logger.warning(f"⚠️  Odds muito alto para {game_key}: {odds} > {OddsValidator.MAX_ODDS}")
            return False
        
        return True
    
    @staticmethod
    def validate_game_odds(home_odds: float, away_odds: float, game_key: str = "") -> bool:
        """
        Valida consistência das odds de um jogo.
        
        Verifica:
        1. Valores individuais válidos
        2. Soma de probabilidades implícitas próxima de 1.0 (com vigorish)
        
        Args:
            home_odds: Odds do time da casa
            away_odds: Odds do time visitante
            game_key: Identificador do jogo
            
        Returns:
            True se válido, False caso contrário
        """
        # Validar valores individuais
        if not OddsValidator.validate_odds_value(home_odds, f"{game_key} (home)"):
            return False
        if not OddsValidator.validate_odds_value(away_odds, f"{game_key} (away)"):
            return False
        
        # Calcular probabilidades implícitas
        prob_home = 1.0 / home_odds
        prob_away = 1.0 / away_odds
        prob_sum = prob_home + prob_away
        
        # Soma deve ser > 1.0 (vigorish) mas não muito maior
        # Tipicamente: 1.05 (5% vig) a 1.15 (15% vig)
        if prob_sum < 1.0:
            logger.warning(
                f"⚠️  Soma de probabilidades < 1.0 para {game_key}: {prob_sum:.3f} "
                f"(home={prob_home:.3f}, away={prob_away:.3f})"
            )
            return False
        
        if prob_sum > 1.3:  # Vigorish > 30% é suspeito
            logger.warning(
                f"⚠️  Soma de probabilidades muito alta para {game_key}: {prob_sum:.3f} "
                f"(vigorish de {(prob_sum - 1.0) * 100:.1f}%)"
            )
            return False
        
        return True
    
    @staticmethod
    def normalize_and_validate(odds_dict: Dict) -> Dict:
        """
        Valida e normaliza dictionary de odds.
        
        Args:
            odds_dict: Dict no formato {game_key: {data}}
            
        Returns:
            Dict validado e normalizado
        """
        validated = {}
        errors = 0
        
        for game_key, odds_data in odds_dict.items():
            try:
                home_odds = float(odds_data.get('home_odds', 0))
                away_odds = float(odds_data.get('away_odds', 0))
                
                # Validar
                if not OddsValidator.validate_game_odds(home_odds, away_odds, game_key):
                    errors += 1
                    continue
                
                # Normalizar formato
                validated[game_key] = {
                    'home_odds': home_odds,
                    'away_odds': away_odds,
                    'home_team': odds_data.get('home_team', ''),
                    'away_team': odds_data.get('away_team', ''),
                    'source': odds_data.get('source', 'unknown'),
                    'timestamp': odds_data.get('timestamp', datetime.now().isoformat()),
                    'validated': True
                }
                
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"⚠️  Erro ao validar odds para {game_key}: {e}")
                errors += 1
                continue
        
        if errors > 0:
            logger.warning(f"⚠️  {errors}/{len(odds_dict)} jogos com odds inválidos removidos")
        
        return validated


class TheOddsAPIClient:
    """Cliente para TheOddsAPI."""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "basketball_nba"
    MARKETS = "h2h"  # Head to head (moneyline)
    REGIONS = "us"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: API key do TheOddsAPI (ou None para buscar do env)
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.session = requests.Session()
        # DEBUG: Log se a key foi carregada (sem expor a key)
        logger.debug(f"TheOddsAPI: API Key loaded = {bool(self.api_key)}")
    
    def fetch_odds(self) -> Dict:
        """
        Busca odds da NBA via TheOddsAPI.
        
        Returns:
            Dict com odds no formato padronizado
            
        Raises:
            Exception: Se API call falhar
        """
        if not self.api_key:
            raise ValueError("ODDS_API_KEY não configurado no .env")
        
        url = f"{self.BASE_URL}/sports/{self.SPORT}/odds/"
        params = {
            'api_key': self.api_key,
            'regions': self.REGIONS,
            'markets': self.MARKETS,
            'oddsFormat': 'decimal'
        }
        
        try:
            logger.info("📡 Buscando odds via TheOddsAPI...")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Verificar quota usage
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                logger.info(f"   Requests restantes hoje: {remaining}")
            
            # Converter para formato padronizado
            odds_dict = self._parse_api_response(data)
            
            logger.info(f"✅ TheOddsAPI: {len(odds_dict)} jogos obtidos")
            return odds_dict
            
        except requests.exceptions.HTTPError as e:
            # Tratamento específico para chaves inválidas
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code in (401, 403):
                    logger.error("❌ TheOddsAPI: Chave API inválida ou expirada!")
                else:
                    logger.error(f"❌ TheOddsAPI HTTP error ({e.response.status_code}): {e}")
            else:
                logger.error(f"❌ TheOddsAPI HTTP error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ TheOddsAPI request failed: {e}")
            raise
    
    def _parse_api_response(self, data: List[Dict]) -> Dict:
        """
        Converte resposta da API para formato padronizado.
        
        Args:
            data: Lista de eventos da API
            
        Returns:
            Dict no formato {game_key: {home_odds, away_odds, ...}}
        """
        odds_dict = {}
        
        for event in data:
            try:
                home_team = event['home_team']
                away_team = event['away_team']
                commence_time = event['commence_time']
                
                # Pegar odds do primeiro bookmaker disponível
                if not event.get('bookmakers'):
                    continue
                
                bookmaker = event['bookmakers'][0]
                markets = bookmaker.get('markets', [])
                
                # Encontrar mercado h2h
                h2h_market = next((m for m in markets if m['key'] == 'h2h'), None)
                if not h2h_market:
                    continue
                
                outcomes = h2h_market['outcomes']
                
                # Mapear outcomes para home/away
                home_outcome = next((o for o in outcomes if o['name'] == home_team), None)
                away_outcome = next((o for o in outcomes if o['name'] == away_team), None)
                
                if not home_outcome or not away_outcome:
                    continue
                
                game_key = f"{home_team} vs {away_team}"
                
                odds_dict[game_key] = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_odds': home_outcome['price'],
                    'away_odds': away_outcome['price'],
                    'source': f"theoddsapi_{bookmaker['key']}",
                    'timestamp': commence_time,
                    'bookmaker': bookmaker['title']
                }
                
            except (KeyError, IndexError, StopIteration) as e:
                logger.debug(f"Skipping event due to parsing error: {e}")
                continue
        
        return odds_dict


class SportsDataIOClient:
    """
    Cliente para SportsDataIO API (fallback tier 2).
    Documentação: https://sportsdata.io/developers/api-documentation/nba
    """
    
    BASE_URL = "https://api.sportsdata.io/v3/nba"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: API key do SportsDataIO
        """
        self.api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        self.session = requests.Session()
        # DEBUG: Log se a key foi carregada (sem expor a key)
        logger.debug(f"SportsDataIO: API Key loaded = {bool(self.api_key)}")
    
    def fetch_odds(self) -> Dict:
        """
        Busca odds da NBA via SportsDataIO.
        
        Returns:
            Dict com odds no formato padronizado
            
        Raises:
            Exception: Se API call falhar
        """
        if not self.api_key:
            raise ValueError("SPORTSDATA_API_KEY não configurado")
        
        # Endpoint de odds
        url = f"{self.BASE_URL}/odds/json/GameOddsByDate/{{date}}"
        
        # Usar data de hoje
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        url = url.format(date=date_str)
        
        try:
            logger.info("📡 Buscando odds via SportsDataIO (fallback tier 2)...")
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Converter para formato padronizado
            odds_dict = self._parse_api_response(data)
            
            logger.info(f"✅ SportsDataIO: {len(odds_dict)} jogos obtidos")
            return odds_dict
            
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code in (401, 403):
                    logger.error("❌ SportsDataIO: Chave API inválida ou expirada!")
                else:
                    logger.error(f"❌ SportsDataIO HTTP error ({e.response.status_code}): {e}")
            else:
                logger.error(f"❌ SportsDataIO HTTP error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ SportsDataIO request failed: {e}")
            raise
    
    def _parse_api_response(self, data: list) -> Dict:
        """
        Converte resposta da SportsDataIO para formato padronizado.
        
        Args:
            data: Lista de jogos com odds
            
        Returns:
            Dict no formato {game_key: {home_odds, away_odds, ...}}
        """
        odds_dict = {}
        
        for game in data:
            try:
                # Obter informações do jogo
                home_team = game.get('HomeTeam', '')
                away_team = game.get('AwayTeam', '')
                
                # Pegar primeira linha de odds disponível
                pregame_odds = game.get('PregameOdds', [])
                if not pregame_odds:
                    continue
                
                # Usar primeira bookmaker
                odds_line = pregame_odds[0]
                
                # Extrair moneyline (converter de American para Decimal)
                home_ml = odds_line.get('HomeMoneyLine')
                away_ml = odds_line.get('AwayMoneyLine')
                
                if home_ml and away_ml:
                    # Converter American odds para Decimal
                    home_odds = self._american_to_decimal(home_ml)
                    away_odds = self._american_to_decimal(away_ml)
                    
                    game_key = f"{home_team} vs {away_team}"
                    
                    odds_dict[game_key] = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_odds': home_odds,
                        'away_odds': away_odds,
                        'source': f"sportsdata_{odds_line.get('Sportsbook', 'unknown')}",
                        'timestamp': game.get('DateTime', datetime.now().isoformat()),
                        'bookmaker': odds_line.get('Sportsbook', 'Unknown')
                    }
                
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Skipping game due to parsing error: {e}")
                continue
        
        return odds_dict
    
    @staticmethod
    def _american_to_decimal(american_odds: int) -> float:
        """
        Converte American odds para Decimal odds.
        
        Args:
            american_odds: Odds no formato americano (ex: -110, +150)
            
        Returns:
            Odds no formato decimal (ex: 1.91, 2.50)
        """
        if american_odds > 0:
            return round((american_odds / 100) + 1, 2)
        else:
            return round((100 / abs(american_odds)) + 1, 2)


class RapidAPIFootballClient:
    """
    Cliente para API-FOOTBALL via RapidAPI (fallback tier 3).
    Documentação: https://rapidapi.com/api-sports/api/api-basketball
    """
    
    BASE_URL = "https://api-basketball.p.rapidapi.com"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: RapidAPI key
        """
        self.api_key = api_key or os.getenv('RAPIDAPI_KEY', '')
        self.session = requests.Session()
        # DEBUG: Log se a key foi carregada (sem expor a key)
        logger.debug(f"RapidAPI: API Key loaded = {bool(self.api_key)}")
    
    def fetch_odds(self) -> Dict:
        """
        Busca odds da NBA via RapidAPI Basketball.
        
        Returns:
            Dict com odds no formato padronizado
            
        Raises:
            Exception: Se API call falhar
        """
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY não configurado")
        
        # Endpoint de odds
        url = f"{self.BASE_URL}/odds"
        
        # Parametros para NBA
        params = {
            'league': '12',  # NBA league ID
            'season': '2024-2025'
        }
        
        headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'api-basketball.p.rapidapi.com'
        }
        
        try:
            logger.info("📡 Buscando odds via RapidAPI (fallback tier 3)...")
            
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Converter para formato padronizado
            odds_dict = self._parse_api_response(data)
            
            logger.info(f"✅ RapidAPI: {len(odds_dict)} jogos obtidos")
            return odds_dict
            
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code in (401, 403):
                    logger.error("❌ RapidAPI: Chave API inválida ou expirada!")
                else:
                    logger.error(f"❌ RapidAPI HTTP error ({e.response.status_code}): {e}")
            else:
                logger.error(f"❌ RapidAPI HTTP error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ RapidAPI request failed: {e}")
            raise
    
    def _parse_api_response(self, data: Dict) -> Dict:
        """
        Converte resposta da RapidAPI para formato padronizado.
        
        Args:
            data: Resposta JSON da API
            
        Returns:
            Dict no formato {game_key: {home_odds, away_odds, ...}}
        """
        odds_dict = {}
        
        response = data.get('response', [])
        
        for item in response:
            try:
                game = item.get('game', {})
                bookmakers = item.get('bookmakers', [])
                
                if not bookmakers:
                    continue
                
                # Pegar primeiro bookmaker
                bookmaker = bookmakers[0]
                bets = bookmaker.get('bets', [])
                
                # Procurar moneyline (winner)
                moneyline = next((b for b in bets if b.get('name') == 'Home/Away'), None)
                
                if not moneyline:
                    continue
                
                values = moneyline.get('values', [])
                if len(values) < 2:
                    continue
                
                # Extrair times e odds
                home_team = game.get('teams', {}).get('home', {}).get('name', '')
                away_team = game.get('teams', {}).get('away', {}).get('name', '')
                
                home_value = next((v for v in values if v.get('value') == 'Home'), None)
                away_value = next((v for v in values if v.get('value') == 'Away'), None)
                
                if home_value and away_value:
                    game_key = f"{home_team} vs {away_team}"
                    
                    odds_dict[game_key] = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_odds': float(home_value.get('odd', 1.90)),
                        'away_odds': float(away_value.get('odd', 1.90)),
                        'source': f"rapidapi_{bookmaker.get('name', 'unknown')}",
                        'timestamp': game.get('date', datetime.now().isoformat()),
                        'bookmaker': bookmaker.get('name', 'Unknown')
                    }
                
            except (KeyError, IndexError, TypeError, ValueError) as e:
                logger.debug(f"Skipping odds due to parsing error: {e}")
                continue
        
        return odds_dict


class OddsAPIioClient:
    """
    Cliente para OddsAPI.io (fallback tier 4).
    Documentação: https://oddsapi.docs.apiary.io/#reference/0/odds-api
    """
    
    BASE_URL = "https://api.sportradar.us/nexbets/v1/en/nba"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: API key do OddsAPI (ou None - API pode funcionar sem key)
        """
        self.api_key = api_key or os.getenv('ODDSAPI_KEY', '')
        self.session = requests.Session()
    
    def fetch_odds(self) -> Dict:
        """
        Busca odds da NBA via OddsAPI.io.
        
        Returns:
            Dict com odds no formato padronizado
            
        Raises:
            Exception: Se API call falhar
        """
        # Endpoint simplificado - API gratuita sem necessidade de key
        url = "https://api.the-odds-api.com/v3/odds/"
        
        params = {
            'sport': 'basketball_nba',
            'region': 'us',
            'mkt': 'h2h'
        }
        
        # Adicionar API key se disponível
        if self.api_key:
            params['apiKey'] = self.api_key
        
        try:
            logger.info("📡 Buscando odds via OddsAPI.io (fallback)...")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Converter para formato padronizado
            odds_dict = self._parse_api_response(data)
            
            logger.info(f"✅ OddsAPI.io: {len(odds_dict)} jogos obtidos")
            return odds_dict
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ OddsAPI.io request failed: {e}")
            raise
    
    def _parse_api_response(self, data: Dict) -> Dict:
        """
        Converte resposta da API para formato padronizado.
        
        Args:
            data: Resposta JSON da API
            
        Returns:
            Dict no formato {game_key: {home_odds, away_odds, ...}}
        """
        odds_dict = {}
        
        # A API retorna lista de eventos
        if not isinstance(data, dict) or 'data' not in data:
            logger.warning("OddsAPI.io response format unexpected")
            return odds_dict
        
        for event in data.get('data', []):
            try:
                # Obter times
                teams = event.get('teams', [])
                if len(teams) != 2:
                    continue
                
                home_team = event.get('home_team', teams[0])
                away_team = teams[1] if teams[1] != home_team else teams[0]
                
                # Obter odds do primeiro bookmaker disponível
                sites = event.get('sites', [])
                if not sites:
                    continue
                
                site = sites[0]
                odds = site.get('odds', {})
                h2h = odds.get('h2h', [])
                
                if len(h2h) != 2:
                    continue
                
                # Assumir primeiro odd é home, segundo é away
                home_odds = h2h[0]
                away_odds = h2h[1]
                
                game_key = f"{home_team} vs {away_team}"
                
                odds_dict[game_key] = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_odds': home_odds,
                    'away_odds': away_odds,
                    'source': f"oddsapi_{site.get('site_key', 'unknown')}",
                    'timestamp': event.get('commence_time', datetime.now().isoformat()),
                    'bookmaker': site.get('site_nice', 'Unknown')
                }
                
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Skipping event due to parsing error: {e}")
                continue
        
        return odds_dict


class OddsSharkScraper:
    """Scraper de fallback para Odds Shark (web scraping)."""
    
    BASE_URL = "https://www.oddsshark.com/nba/odds"
    
    def fetch_odds(self) -> Dict:
        """
        Scrape odds do Odds Shark.
        
        Returns:
            Dict com odds
            
        Raises:
            Exception: Se scraping falhar
        """
        # TODO: Implementar scraping real (BeautifulSoup)
        # Por enquanto, lançar exception para forçar fallback
        raise NotImplementedError("OddsShark scraping não implementado ainda")


def obter_odds(force_source: Optional[str] = None) -> Dict:
    """
    Busca odds com hierarquia de fallback robusta (6 tiers!).
    
    Hierarquia (Scraping First - economiza API):
    1. OddsPediaScraper (GRATUITO - web scraping) ← TIER 1
    2. TheOddsAPI (se API key disponível) ← TIER 2
    3. SportsDataIO (FREE trial disponível) ← TIER 3
    4. RapidAPI Basketball (FREE tier) ← TIER 4
    5. OddsAPI.io (fallback gratuito) ← TIER 5
    6. Erro (não usa dados fictícios) ← TIER 6
    
    Args:
        force_source: Forçar source específico para testes
                     ('oddspedia', 'theoddsapi', 'sportsdata', 'rapidapi', 'oddsapiio')
    
    Returns:
        Dict com formato padronizado de odds
    """
    return _obter_odds_cached(force_source)

@smart_cache(ttl_hours=TTL_ODDS, cache_key_prefix='odds')
def _obter_odds_cached(force_source: Optional[str] = None) -> Dict:
    """Função interna cacheada."""
    odds_dict = {}
    source_used = "none"
    
    # ===== TIER 1: OddsPedia Web Scraper (GRATUITO) =====
    # Prioridade máxima para economizar chamadas de API paga
    if force_source in [None, 'oddspedia'] and ODDSPEDIA_AVAILABLE:
        try:
            logger.info("📡 Tentando OddsPedia Scraper (TIER 1 - Gratuito)...")
            scraper = OddsPediaScraper()
            odds_dict = scraper.fetch_odds()
            source_used = "oddspedia"
            
            odds_dict = OddsValidator.normalize_and_validate(odds_dict)
            
            if odds_dict:
                logger.info(f"✅ Odds obtidos via OddsPedia Scraper: {len(odds_dict)} jogos")
                return odds_dict
            else:
                logger.warning("⚠️ OddsPedia Scraper retornou 0 jogos válidos")
                
        except Exception as e:
            # Falha SILENCIOSA - apenas warning, não interrompe o sistema
            logger.warning(f"⚠️ Falha no Scraper Web: {e}. Usando API de backup...")
    
    # ===== TIER 2: TheOddsAPI (API Paga) =====
    if force_source in [None, 'theoddsapi']:
        try:
            api_client = TheOddsAPIClient()
            odds_dict = api_client.fetch_odds()
            source_used = "theoddsapi"
            
            odds_dict = OddsValidator.normalize_and_validate(odds_dict)
            
            if odds_dict:
                logger.info(f"✅ Odds obtidos via TheOddsAPI: {len(odds_dict)} jogos")
                return odds_dict
            else:
                logger.warning("⚠️  TheOddsAPI retornou 0 jogos válidos")
                
        except ValueError as e:
            logger.warning(f"⚠️  TheOddsAPI não disponível: {e}")
        except Exception as e:
            logger.warning(f"⚠️  TheOddsAPI falhou: {e}. Tentando fallback...")
    
    # ===== TIER 3: SportsDataIO (API Paga) =====
    if force_source in [None, 'sportsdata']:
        try:
            sportsdata_client = SportsDataIOClient()
            odds_dict = sportsdata_client.fetch_odds()
            source_used = "sportsdata"
            
            odds_dict = OddsValidator.normalize_and_validate(odds_dict)
            
            if odds_dict:
                logger.info(f"✅ Odds obtidos via SportsDataIO: {len(odds_dict)} jogos")
                return odds_dict
            else:
                logger.warning("⚠️  SportsDataIO retornou 0 jogos válidos")
                
        except Exception as e:
            logger.warning(f"⚠️  SportsDataIO falhou: {e}. Tentando próximo fallback...")
    
    # ===== TIER 4: RapidAPI (API) =====
    if force_source in [None, 'rapidapi']:
        try:
            rapidapi_client = RapidAPIFootballClient()
            odds_dict = rapidapi_client.fetch_odds()
            source_used = "rapidapi"
            
            odds_dict = OddsValidator.normalize_and_validate(odds_dict)
            
            if odds_dict:
                logger.info(f"✅ Odds obtidos via RapidAPI: {len(odds_dict)} jogos")
                return odds_dict
            else:
                logger.warning("⚠️  RapidAPI retornou 0 jogos válidos")
                
        except Exception as e:
            logger.warning(f"⚠️  RapidAPI falhou: {e}. Tentando próximo fallback...")
    
    # ===== TIER 5: OddsAPI.io (Gratuito/Limitado) =====
    if force_source in [None, 'oddsapiio']:
        try:
            oddsapi_client = OddsAPIioClient()
            odds_dict = oddsapi_client.fetch_odds()
            source_used = "oddsapiio"
            
            odds_dict = OddsValidator.normalize_and_validate(odds_dict)
            
            if odds_dict:
                logger.info(f"✅ Odds obtidos via OddsAPI.io: {len(odds_dict)} jogos")
                return odds_dict
            else:
                logger.warning("⚠️  OddsAPI.io retornou 0 jogos válidos")
                
        except Exception as e:
            logger.warning(f"⚠️  OddsAPI.io falhou: {e}. Tentando próximo fallback...")
    
    # ===== TIER 6: Erro Final =====
    # REMOVIDO: OddsShark (substituído por OddsPedia como TIER 1)
    if force_source in [None, 'oddsshark']:
        try:
            scraper = OddsSharkScraper()
            odds_dict = scraper.fetch_odds()
            source_used = "oddsshark"
            
            odds_dict = OddsValidator.normalize_and_validate(odds_dict)
            
            if odds_dict:
                logger.info(f"✅ Odds obtidos via OddsShark: {len(odds_dict)} jogos")
                return odds_dict
                
        except NotImplementedError:
            logger.debug("OddsShark scraping ainda não implementado")
        except Exception as e:
            logger.warning(f"⚠️  OddsShark scraping falhou: {e}")
    
    
    # 6. Todas as fontes falharam - LANÇAR ERRO (NÃO USAR DADOS FICTÍCIOS!)
    logger.error(
        "🚨 ERRO CRÍTICO: TODAS AS FONTES DE ODDS FALHARAM!\n"
        "   Tentativas realizadas:\n"
        "   1. TheOddsAPI - Falhou ou sem API key\n"
        "   2. SportsDataIO - Falhou ou sem API key\n"
        "   3. RapidAPI - Falhou ou sem API key\n"
        "   4. OddsAPI.io - Falhou\n"
        "   5. OddsShark - Não implementado\n\n"
        "   Configure pelo menos uma API key no .env:\n"
        "   - ODDS_API_KEY (TheOddsAPI)\n"
        "   - SPORTSDATA_API_KEY (SportsDataIO)\n"
        "   - RAPIDAPI_KEY (RapidAPI)"
    )
    
    # Importar exception
    from exceptions.odds_exceptions import OddsUnavailableError
    
    # Lançar erro - sistema NÃO DEVE usar dados fictícios
    raise OddsUnavailableError(
        "Todas as 5 fontes de odds falharam. "
        "Configure API keys no .env para obter odds reais. "
        "O sistema NÃO utilizará dados simulados."
    )


def get_odds_for_game(home_team: str, away_team: str, 
                      odds_cache: Optional[Dict] = None) -> Dict[str, float]:
    """
    Obtém odds para um jogo específico.
    
    Args:
        home_team: Time da casa
        away_team: Time visitante
        odds_cache: Cache de odds (opcional, para evitar chamadas redundantes)
        
    Returns:
        Dict com {'home_odds': float, 'away_odds': float}
        Default: {'home_odds': 1.90, 'away_odds': 1.90}
    """
    # Se cache fornecido, buscar nele
    if odds_cache:
        # Tentar várias formas de match
        possible_keys = [
            f"{home_team} vs {away_team}",
            f"{away_team} vs {home_team}",  # Invertido
            f"{home_team} @ {away_team}",
        ]
        
        for key in possible_keys:
            if key in odds_cache:
                odds_data = odds_cache[key]
                return {
                    'home_odds': odds_data['home_odds'],
                    'away_odds': odds_data['away_odds'],
                    'source': odds_data.get('source', 'unknown')
                }
    
    # Default
    return {
        'home_odds': 1.90,
        'away_odds': 1.90,
        'source': 'default'
    }


# Para compatibilidade com código existente
def get_default_odds() -> Dict:
    """Retorna odds padrão (legacy compatibility)."""
    return {}
