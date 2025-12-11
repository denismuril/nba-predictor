"""
Game ID Mapper - Mapeia games do sistema para IDs das APIs

Conecta nossos games (por data + teams) com IDs das 3 APIs:
- API-Football (game_id)
- SportData.io (GameKey)
- SportsBlaze (game_id)

Usage:
    from data.game_id_mapper import get_game_ids
    
    ids = get_game_ids(date='2024-11-28', home='LAL', away='GSW')
    # Returns: {'api_football': 12345, 'sportdata': 'key', 'sportsblaze': 'id'}
"""
import requests
import logging
from typing import Dict, Optional
import os
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class GameIDMapper:
    """Mapeia games para IDs das APIs."""
    
    def __init__(self):
        self.api_football_key = os.getenv('API_FOOTBALL_KEY', '01eee81ebe305e3e88ced3e2de4905c1')
        self.sportdata_key = os.getenv('SPORTDATA_KEY', 'bc2194faba594d67b396b5fc52d42bd4')
        self.sportsblaze_key = os.getenv('SPORTSBLAZE_KEY', 'sbfxqpy6v6fjljvobf61a5o')
        
        # Cache
        self.cache_file = Path('data/cache/game_id_cache.json')
        self.cache_file.parent.mkdir(exist_ok=True, parents=True)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Carrega cache de disco."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Salva cache em disco."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")
    
    def get_game_ids(self, date: str, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Busca IDs do jogo nas 3 APIs.
        
        Args:
            date: Data no formato YYYY-MM-DD
            home_team: Código do time casa (ex: 'LAL')
            away_team: Código do time visitante
        
        Returns:
            {
                'api_football': int or None,
                'sportdata': str or None,
                'sportsblaze': str or None,
                'found_in': list  # quais APIs encontraram
            }
        """
        # Check cache
        cache_key = f"{date}_{home_team}_{away_team}"
        if cache_key in self.cache:
            logger.debug(f"Cache hit: {cache_key}")
            return self.cache[cache_key]
        
        result = {
            'api_football': None,
            'sportdata': None,
            'sportsblaze': None,
            'found_in': []
        }
        
        # Try API 1: API-Football
        api_football_id = self._get_api_football_id(date, home_team, away_team)
        if api_football_id:
            result['api_football'] = api_football_id
            result['found_in'].append('api_football')
        
        # Try API 2: SportData
        sportdata_key = self._get_sportdata_id(date, home_team, away_team)
        if sportdata_key:
            result['sportdata'] = sportdata_key
            result['found_in'].append('sportdata')
        
        # Try API 3: SportsBlaze
        sportsblaze_id = self._get_sportsblaze_id(date, home_team, away_team)
        if sportsblaze_id:
            result['sportsblaze'] = sportsblaze_id
            result['found_in'].append('sportsblaze')
        
        # Cache result
        if result['found_in']:
            self.cache[cache_key] = result
            self._save_cache()
            logger.info(f"✅ Game IDs encontrados: {result['found_in']}")
        else:
            logger.warning(f"⚠️ Nenhum ID encontrado para {date} {home_team} vs {away_team}")
        
        return result if result['found_in'] else None
    
    def _get_api_football_id(self, date: str, home_team: str, away_team: str) -> Optional[int]:
        """Busca ID na API-Football."""
        try:
            url = "https://v2.nba.api-sports.io/games"
            headers = {'x-apisports-key': self.api_football_key}
            params = {
                'date': date,
                'league': '12',  # NBA
                'season': date[:4]  # Year
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                games = data.get('response', [])
                
                # Match por teams
                for game in games:
                    teams = game.get('teams', {})
                    home = teams.get('home', {}).get('code', '')
                    away = teams.get('away', {}).get('code', '')
                    
                    if self._teams_match(home, home_team) and self._teams_match(away, away_team):
                        game_id = game.get('id')
                        logger.debug(f"API-Football ID: {game_id}")
                        return game_id
            
            return None
        except Exception as e:
            logger.debug(f"API-Football search failed: {e}")
            return None
    
    def _get_sportdata_id(self, date: str, home_team: str, away_team: str) -> Optional[str]:
        """Busca GameKey no SportData.io."""
        try:
            url = f"https://api.sportsdata.io/v3/nba/scores/json/GamesByDate/{date}"
            params = {'key': self.sportdata_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                games = response.json()
                
                # Match por teams
                for game in games:
                    home = game.get('HomeTeam', '')
                    away = game.get('AwayTeam', '')
                    
                    if self._teams_match(home, home_team) and self._teams_match(away, away_team):
                        game_key = game.get('GameKey') or game.get('GameID')
                        logger.debug(f"SportData GameKey: {game_key}")
                        return str(game_key) if game_key else None
            
            return None
        except Exception as e:
            logger.debug(f"SportData search failed: {e}")
            return None
    
    def _get_sportsblaze_id(self, date: str, home_team: str, away_team: str) -> Optional[str]:
        """Busca ID no SportsBlaze."""
        try:
            url = f"https://api.sportsblaze.com/nba/v1/boxscores/daily/{date}.json"
            params = {'key': self.sportsblaze_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                games = data.get('games', [])
                
                # Match por teams
                for game in games:
                    teams = game.get('teams', {})
                    home = teams.get('home', {}).get('team', {}).get('abbreviation', '')
                    away = teams.get('away', {}).get('team', {}).get('abbreviation', '')
                    
                    if self._teams_match(home, home_team) and self._teams_match(away, away_team):
                        game_id = game.get('id')
                        logger.debug(f"SportsBlaze ID: {game_id}")
                        return str(game_id) if game_id else None
            
            return None
        except Exception as e:
            logger.debug(f"SportsBlaze search failed: {e}")
            return None
    
    def _teams_match(self, api_team: str, our_team: str) -> bool:
        """Verifica se códigos de times correspondem."""
        # Normalizar
        api_team = api_team.upper().strip()
        our_team = our_team.upper().strip()
        
        # Match direto
        if api_team == our_team:
            return True
        
        # Aliases comuns
        aliases = {
            'LAL': ['LAKERS', 'LOS ANGELES LAKERS'],
            'GSW': ['WARRIORS', 'GOLDEN STATE'],
            'BOS': ['CELTICS', 'BOSTON'],
            'MIA': ['HEAT', 'MIAMI'],
            # Adicionar mais conforme necessário
        }
        
        for code, variations in aliases.items():
            if our_team == code and api_team in variations:
                return True
            if api_team == code and our_team in variations:
                return True
        
        return False


def get_game_ids(date: str, home_team: str, away_team: str) -> Optional[Dict]:
    """
    Helper function para buscar IDs.
    
    Args:
        date: YYYY-MM-DD
        home_team: Team code
        away_team: Team code
    
    Returns:
        Dict com IDs ou None
    """
    mapper = GameIDMapper()
    return mapper.get_game_ids(date, home_team, away_team)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("🏀 Demo: Game ID Mapper\n")
    
    # Test com data recente
    test_date = '2024-11-27'  # Ajustar para data real
    test_home = 'LAL'
    test_away = 'GSW'
    
    print(f"Buscando IDs para: {test_date} - {test_home} vs {test_away}\n")
    
    ids = get_game_ids(test_date, test_home, test_away)
    
    if ids:
        print("✅ IDs encontrados:")
        print(f"  API-Football: {ids.get('api_football')}")
        print(f"  SportData: {ids.get('sportdata')}")
        print(f"  SportsBlaze: {ids.get('sportsblaze')}")
        print(f"  Found in: {', '.join(ids.get('found_in', []))}")
    else:
        print("⚠️ Nenhum ID encontrado")
        print("  (Pode ser que não tenha jogo nessa data)")
    
    print("\n✅ Demo completo!")
