"""
Referee API Scraper - Fallback via RapidAPI (API-NBA)
======================================================
Usa a API-NBA do RapidAPI para buscar árbitros de jogos.
Mais rápido que Selenium e não precisa de browser.

Documentação: https://rapidapi.com/api-sports/api/api-nba

Autor: NBA Predictor System
Data: 2025-12-05
"""

import sys
import os
from pathlib import Path

# Fix para imports quando executado diretamente
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import requests
import pandas as pd
import time
from datetime import datetime
from typing import Optional, List, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)


class RefereeAPIScraper:
    """
    Scraper de árbitros usando API-NBA (RapidAPI).
    
    Hierarquia de fallback:
    1. API-NBA (RapidAPI) - endpoint /games com officials
    2. Selenium (Basketball-Reference) - se API falhar
    """
    
    BASE_URL = "https://api-nba-v1.p.rapidapi.com"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('RAPIDAPI_KEY', '')
        
        if not self.api_key:
            logger.warning("⚠️ RAPIDAPI_KEY não configurada. API fallback desativado.")
        
        self.headers = {
            "x-rapidapi-host": "api-nba-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key
        }
    
    def get_game_officials(self, game_id: int) -> Optional[List[str]]:
        """
        Busca árbitros de um jogo específico via API-NBA.
        
        Args:
            game_id: ID do jogo na API-NBA
            
        Returns:
            Lista com nomes dos árbitros ou None
        """
        if not self.api_key:
            return None
        
        try:
            url = f"{self.BASE_URL}/games"
            params = {"id": game_id}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"API-NBA retornou {response.status_code}")
                return None
            
            data = response.json()
            
            if not data.get('results', 0):
                return None
            
            games = data.get('response', [])
            if not games:
                return None
            
            game = games[0]
            officials = game.get('officials', [])
            
            if officials:
                referee_names = [ref if isinstance(ref, str) else ref.get('name', '') for ref in officials]
                referee_names = [r for r in referee_names if r]
                logger.debug(f"✅ API-NBA: {referee_names}")
                return referee_names[:3]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro na API-NBA: {e}")
            return None
    
    def get_games_by_date(self, date: datetime) -> List[Dict]:
        """
        Busca todos os jogos de uma data específica.
        
        Args:
            date: Data dos jogos
            
        Returns:
            Lista de jogos com informações
        """
        if not self.api_key:
            return []
        
        try:
            url = f"{self.BASE_URL}/games"
            params = {"date": date.strftime("%Y-%m-%d")}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"API-NBA retornou {response.status_code}")
                return []
            
            data = response.json()
            
            if not data.get('results', 0):
                return []
            
            return data.get('response', [])
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar jogos: {e}")
            return []
    
    def scrape_game_referees_by_date(self, date: datetime, home_team: str) -> Optional[List[str]]:
        """
        Busca árbitros de um jogo específico por data e time da casa.
        
        Args:
            date: Data do jogo
            home_team: Sigla do time da casa
            
        Returns:
            Lista com nomes dos árbitros ou None
        """
        games = self.get_games_by_date(date)
        
        for game in games:
            # Verificar se é o jogo certo
            home = game.get('teams', {}).get('home', {})
            home_name = home.get('code', '') or home.get('nickname', '')
            
            if home_team.upper() in home_name.upper():
                officials = game.get('officials', [])
                if officials:
                    referee_names = [ref if isinstance(ref, str) else ref.get('name', '') for ref in officials]
                    referee_names = [r for r in referee_names if r]
                    if referee_names:
                        logger.info(f"✅ API-NBA: Árbitros para {home_team}: {referee_names}")
                        return referee_names[:3]
        
        return None


def get_referees_with_api(date: datetime, home_team: str) -> Optional[List[str]]:
    """
    Função helper para buscar árbitros via API-NBA.
    
    Usage:
        refs = get_referees_with_api(datetime(2024, 12, 1), "BOS")
    """
    scraper = RefereeAPIScraper()
    return scraper.scrape_game_referees_by_date(date, home_team)


def get_referees_with_fallback(date: datetime, home_team: str) -> Optional[List[str]]:
    """
    Busca árbitros usando hierarquia de fallback:
    1. API-NBA (RapidAPI) - rápido, sem browser
    2. Selenium (BBRef) - mais lento, mas confiável
    
    Args:
        date: Data do jogo
        home_team: Sigla do time da casa
        
    Returns:
        Lista com nomes dos árbitros ou None
    """
    # Tentar API-NBA primeiro
    logger.info(f"🔍 Tentando API-NBA para {date.date()} {home_team}...")
    refs = get_referees_with_api(date, home_team)
    
    if refs:
        return refs
    
    # Fallback para Selenium
    logger.info(f"🔄 API-NBA falhou, tentando Selenium...")
    try:
        from data.scrapers.referee_selenium_scraper import get_referees_with_selenium
        refs = get_referees_with_selenium(date, home_team)
        if refs:
            return refs
    except Exception as e:
        logger.warning(f"⚠️ Selenium fallback falhou: {e}")
    
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🏀 Testando Referee API Scraper\n")
    
    scraper = RefereeAPIScraper()
    
    if scraper.api_key:
        print(f"✅ RAPIDAPI_KEY configurada (****{scraper.api_key[-4:]})")
    else:
        print("❌ RAPIDAPI_KEY não configurada")
        print("   Defina no .env: RAPIDAPI_KEY=sua_chave")
        exit(1)
    
    # Testar busca por data
    test_date = datetime(2024, 12, 1)
    print(f"\n📅 Buscando jogos em {test_date.date()}...")
    
    games = scraper.get_games_by_date(test_date)
    print(f"   Encontrados {len(games)} jogos")
    
    if games:
        for game in games[:3]:  # Mostrar 3 primeiros
            home = game.get('teams', {}).get('home', {}).get('nickname', '?')
            away = game.get('teams', {}).get('visitors', {}).get('nickname', '?')
            officials = game.get('officials', [])
            
            if officials:
                ref_names = [r if isinstance(r, str) else r.get('name', '?') for r in officials]
                print(f"   {away} @ {home}: {', '.join(ref_names)}")
            else:
                print(f"   {away} @ {home}: Sem árbitros na API")
    
    # Testar com fallback
    print(f"\n🔄 Testando fallback (API -> Selenium)...")
    test_date2 = datetime(2024, 10, 22)
    refs = get_referees_with_fallback(test_date2, "LAL")
    
    if refs:
        print(f"✅ Árbitros encontrados: {', '.join(refs)}")
    else:
        print("❌ Nenhum árbitro encontrado")
    
    print("\n✅ Teste concluído!")
