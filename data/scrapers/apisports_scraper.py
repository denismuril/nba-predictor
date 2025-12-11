"""
API-Sports NBA Advanced Stats Scraper

Scrapes advanced statistics from API-Sports NBA:
- Fast Break Points
- Second Chance Points  
- Points in Paint

Endpoint: /games/statistics
Documentation: https://api-sports.io/documentation/nba/v2

Usage:
    from data.scrapers.apisports_scraper import get_game_advanced_stats
    
    stats = get_game_advanced_stats(game_id=12345, api_key='your_key')
"""
import requests
import logging
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)


class APISportsNBAScraper:
    """Scraper para estatísticas avançadas da API-Sports NBA."""
    
    def __init__(self, api_key: Optional[str] = None, use_rapidapi: bool = False):
        """
        Initialize scraper.
        
        Args:
            api_key: API key (env var APISPORTS_NBA_KEY or API_FOOTBALL_KEY)
            use_rapidapi: Se True, usa formato RapidAPI. Se False, usa API-Sports direto
        """
        self.api_key = api_key or os.getenv('API_FOOTBALL_KEY') or os.getenv('APISPORTS_NBA_KEY')
        self.use_rapidapi = use_rapidapi
        
        if use_rapidapi:
            self.base_url = 'https://v2.nba.api-sports.io'
        else:
            # API-Sports direto (api-sports.io)
            self.base_url = 'https://v2.nba.api-sports.io'
        
        if not self.api_key:
            logger.warning("⚠️ API-Sports key não configurada. Usando valores default.")
    
    def get_game_statistics(self, game_id: int) -> Optional[Dict]:
        """
        Obtém estatísticas avançadas de um jogo.
        
        Args:
            game_id: ID do jogo na API-Sports
        
        Returns:
            Dict com estatísticas ou None se erro
        """
        if not self.api_key:
            logger.debug("Sem API key, retornando None")
            return None
        
        try:
            url = f"{self.base_url}/games/statistics"
            
            if self.use_rapidapi:
                # Formato RapidAPI
                headers = {
                    'x-rapidapi-host': 'v2.nba.api-sports.io',
                    'x-rapidapi-key': self.api_key
                }
            else:
                # Formato API-Sports direto
                headers = {
                    'x-apisports-key': self.api_key
                }
            
            params = {'id': game_id}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results', 0) > 0:
                    return data.get('response', [])
                else:
                    logger.warning(f"Sem resultados para game_id={game_id}")
                    return None
            else:
                logger.warning(f"API retornou status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao obter stats: {e}")
            return None
    
    def extract_advanced_stats(self, game_stats: Dict) -> Dict:
        """
        Extrai estatísticas avançadas de interesse.
        
        Args:
            game_stats: Response da API com estatísticas
        
        Returns:
            Dict com stats por time:
            {
                'home': {'fast_break': int, 'second_chance': int, 'paint': int},
                'away': {'fast_break': int, 'second_chance': int, 'paint': int}
            }
        """
        result = {
            'home': {'fast_break': 0, 'second_chance': 0, 'paint': 0},
            'away': {'fast_break': 0, 'second_chance': 0, 'paint': 0}
        }
        
        if not game_stats or len(game_stats) < 2:
            return result
        
        for team_data in game_stats:
            # Determinar se é home ou away
            # API retorna 2 elementos, normalmente [0]=home, [1]=away
            idx = game_stats.index(team_data)
            key = 'home' if idx == 0 else 'away'
            
            stats = team_data.get('statistics', [{}])[0]
            
            result[key] = {
                'fast_break': stats.get('fastBreakPoints', 0),
                'second_chance': stats.get('secondChancePoints', 0),
                'paint': stats.get('pointsInPaint', 0)
            }
        
        return result


def get_advanced_stats_for_games(game_ids: list, api_key: Optional[str] = None) -> Dict:
    """
    Obtém stats avançadas para múltiplos jogos.
    
    Args:
        game_ids: Lista de IDs de jogos
        api_key: API key (opcional, usa env var se não fornecido)
    
    Returns:
        Dict {game_id: {'home': {...}, 'away': {...}}}
    """
    scraper = APISportsNBAScraper(api_key)
    
    results = {}
    
    for game_id in game_ids:
        stats = scraper.get_game_statistics(game_id)
        
        if stats:
            advanced = scraper.extract_advanced_stats(stats)
            results[game_id] = advanced
        else:
            # Default values se não conseguir dados
            results[game_id] = {
                'home': {'fast_break': 0, 'second_chance': 0, 'paint': 0},
                'away': {'fast_break': 0, 'second_chance': 0, 'paint': 0}
            }
    
    return results


if __name__ == '__main__':
    # Demo
    print("🏀 Demo: API-Sports NBA Advanced Stats\n")
    
    # Exemplo com game_id fictício
    # Nota: requer API key válida para funcionar
    
    scraper = APISportsNBAScraper()
    
    if scraper.api_key:
        print("✅ API key configurada")
        
        # Test com game ID (exemplo)
        game_id = 10403  # Game ID de exemplo da documentação
        
        stats = scraper.get_game_statistics(game_id)
        
        if stats:
            advanced = scraper.extract_advanced_stats(stats)
            
            print(f"\n📊 Game {game_id}:")
            print(f"  Home:")
            print(f"    Fast Break: {advanced['home']['fast_break']}")
            print(f"    Second Chance: {advanced['home']['second_chance']}")
            print(f"    Paint: {advanced['home']['paint']}")
            print(f"  Away:")
            print(f"    Fast Break: {advanced['away']['fast_break']}")
            print(f"    Second Chance: {advanced['away']['second_chance']}")
            print(f"    Paint: {advanced['away']['paint']}")
        else:
            print("⚠️ Sem dados disponíveis")
    else:
        print("⚠️ API key não configurada")
        print("   Set APISPORTS_NBA_KEY environment variable")
    
    print("\n✅ Demo completo!")
