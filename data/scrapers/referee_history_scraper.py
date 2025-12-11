"""
Scraper de Árbitros Históricos via Basketball-Reference

Este módulo busca a lista de árbitros que apitaram cada jogo histórico.
Fonte: Basketball-Reference (boxscores contêm informação de árbitros)

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

import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import random

logger = logging.getLogger(__name__)

# Headers mais robustos para evitar 403
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate", 
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Rate limit: Basketball-Reference permite ~20 req/min
# Aumentando delay e adicionando jitter para parecer mais humano
RATE_LIMIT_DELAY = 4.0  # segundos entre requests
RATE_LIMIT_JITTER = 1.5  # variação aleatória



class RefereeHistoryScraper:
    """
    Scraper para obter árbitros de jogos históricos via Basketball-Reference.
    
    Estratégia:
    1. Buscar página de schedule de uma temporada
    2. Para cada jogo, acessar o boxscore e extrair árbitros
    3. Salvar em CSV para cache
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        
    def _get_boxscore_url(self, date: datetime, home_team: str) -> str:
        """
        Constrói URL do boxscore no Basketball-Reference.
        
        Formato: https://www.basketball-reference.com/boxscores/202412050BOS.html
        (YYYYMMDD0XXX onde XXX é a sigla do time da casa)
        """
        date_str = date.strftime('%Y%m%d')
        # Mapear sigla do time para código BBRef
        team_code = self._normalize_team_code(home_team)
        return f"https://www.basketball-reference.com/boxscores/{date_str}0{team_code}.html"
    
    def _normalize_team_code(self, team: str) -> str:
        """Normaliza sigla do time para formato BBRef (3 letras maiúsculas)."""
        # Mapeamento especial para times com códigos diferentes no BBRef
        bbref_map = {
            'PHX': 'PHO',  # Phoenix
            'CHA': 'CHO',  # Charlotte (historicamente CHH)
            'BKN': 'BRK',  # Brooklyn Nets
            'NOP': 'NOP',  # New Orleans Pelicans
            'GS': 'GSW',   # Golden State
            'SA': 'SAS',   # San Antonio
            'NY': 'NYK',   # New York Knicks
            'NO': 'NOP',   # New Orleans
        }
        team_upper = team.upper().strip()
        return bbref_map.get(team_upper, team_upper)
    
    def scrape_game_referees(self, date: datetime, home_team: str) -> Optional[List[str]]:
        """
        Busca os árbitros de um jogo específico.
        
        Args:
            date: Data do jogo
            home_team: Sigla do time da casa (ex: 'BOS', 'LAL')
            
        Returns:
            Lista com nomes dos árbitros ou None se não encontrado
        """
        url = self._get_boxscore_url(date, home_team)
        
        try:
            time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 404:
                logger.debug(f"Boxscore não encontrado: {url}")
                return None
            elif response.status_code != 200:
                logger.warning(f"Erro HTTP {response.status_code} para {url}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Buscar seção de "Officials"
            # BBRef coloca em um div com texto "Officials:"
            officials = []
            
            # Método 1: Procurar strong tag com "Officials"
            for strong in soup.find_all('strong'):
                if 'Officials' in strong.get_text():
                    # O texto dos árbitros geralmente está no mesmo elemento pai
                    parent = strong.parent
                    text = parent.get_text()
                    # Extrair nomes após "Officials:"
                    match = re.search(r'Officials?:\s*(.+)', text)
                    if match:
                        refs_text = match.group(1)
                        # Separar por vírgula ou ponto
                        refs = [r.strip() for r in re.split(r'[,.]', refs_text) if r.strip()]
                        if refs:
                            officials = refs[:3]  # Máximo 3 árbitros
                            break
            
            # Método 2: Procurar na div scorebox_meta
            if not officials:
                scorebox_meta = soup.find('div', class_='scorebox_meta')
                if scorebox_meta:
                    for div in scorebox_meta.find_all('div'):
                        text = div.get_text()
                        if 'Officials' in text or 'Referee' in text:
                            match = re.search(r'(?:Officials?|Referees?):\s*(.+)', text)
                            if match:
                                refs_text = match.group(1)
                                refs = [r.strip() for r in re.split(r'[,.]', refs_text) if r.strip()]
                                if refs:
                                    officials = refs[:3]
                                    break
            
            if officials:
                logger.debug(f"Árbitros encontrados para {date.date()} {home_team}: {officials}")
                return officials
            else:
                logger.debug(f"Nenhum árbitro encontrado na página: {url}")
                return None
                
        except requests.Timeout:
            logger.warning(f"Timeout ao acessar {url}")
            return None
        except Exception as e:
            logger.error(f"Erro ao fazer scraping de {url}: {e}")
            return None
    
    def scrape_season_referees(self, season: str, max_games: int = None) -> pd.DataFrame:
        """
        Busca árbitros de todos os jogos de uma temporada.
        
        Args:
            season: Temporada no formato '2024-25'
            max_games: Limite de jogos a processar (para testes)
            
        Returns:
            DataFrame com colunas: date, home_team, away_team, referees
        """
        logger.info(f"🏀 Iniciando scraping de árbitros para temporada {season}...")
        
        cache_file = self.cache_dir / f'referees_{season.replace("-", "_")}.csv'
        
        # Verificar cache
        if cache_file.exists():
            logger.info(f"📂 Carregando do cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        # Carregar schedule da temporada do banco de dados
        from data.repositories.db_manager import get_db_manager
        db = get_db_manager()
        df_games = db.get_comprehensive_history()
        
        if df_games is None or df_games.empty:
            logger.error("❌ Nenhum jogo histórico encontrado no banco de dados")
            return pd.DataFrame()
        
        # Filtrar por temporada
        df_games['date'] = pd.to_datetime(df_games['date'])
        year_start = int(season.split('-')[0])
        start_date = pd.Timestamp(f'{year_start}-10-01')
        end_date = pd.Timestamp(f'{year_start + 1}-06-30')
        
        df_season = df_games[(df_games['date'] >= start_date) & (df_games['date'] <= end_date)].copy()
        df_season = df_season.sort_values('date').reset_index(drop=True)
        
        logger.info(f"📊 {len(df_season)} jogos encontrados na temporada {season}")
        
        if max_games:
            df_season = df_season.head(max_games)
            logger.info(f"🔬 Limitado a {max_games} jogos para teste")
        
        # Scrape cada jogo
        results = []
        for idx, row in df_season.iterrows():
            game_date = row['date']
            home_team = row['home_team']
            away_team = row['away_team']
            
            referees = self.scrape_game_referees(game_date, home_team)
            
            results.append({
                'date': game_date.strftime('%Y-%m-%d'),
                'home_team': home_team,
                'away_team': away_team,
                'referee_1': referees[0] if referees and len(referees) > 0 else None,
                'referee_2': referees[1] if referees and len(referees) > 1 else None,
                'referee_3': referees[2] if referees and len(referees) > 2 else None,
                'referees_csv': ','.join(referees) if referees else None
            })
            
            # Progress log
            if (idx + 1) % 10 == 0:
                logger.info(f"   Processados {idx + 1}/{len(df_season)} jogos...")
        
        df_referees = pd.DataFrame(results)
        
        # Salvar cache
        df_referees.to_csv(cache_file, index=False)
        logger.info(f"💾 Salvo em cache: {cache_file}")
        
        success_rate = df_referees['referees_csv'].notna().sum() / len(df_referees) * 100
        logger.info(f"✅ Scraping concluído: {success_rate:.1f}% de jogos com árbitros encontrados")
        
        return df_referees


def update_games_with_referees(df_games: pd.DataFrame, df_referees: pd.DataFrame) -> pd.DataFrame:
    """
    Faz merge dos dados de árbitros com DataFrame de jogos.
    
    Args:
        df_games: DataFrame com jogos (precisa ter 'date', 'home_team')
        df_referees: DataFrame do RefereeHistoryScraper
        
    Returns:
        df_games com coluna 'referees_csv' adicionada
    """
    df_games = df_games.copy()
    df_games['date_str'] = pd.to_datetime(df_games['date']).dt.strftime('%Y-%m-%d')
    
    df_referees = df_referees.copy()
    df_referees['date_str'] = df_referees['date']
    
    # Merge
    df_merged = df_games.merge(
        df_referees[['date_str', 'home_team', 'referees_csv']],
        on=['date_str', 'home_team'],
        how='left'
    )
    
    df_merged = df_merged.drop(columns=['date_str'])
    
    matched = df_merged['referees_csv'].notna().sum()
    logger.info(f"✅ Árbitros encontrados para {matched}/{len(df_merged)} jogos")
    
    return df_merged


if __name__ == "__main__":
    # Teste do scraper
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    scraper = RefereeHistoryScraper()
    
    # Testar com um jogo específico
    test_date = datetime(2024, 12, 5)
    test_team = "BOS"
    
    print(f"\n🔍 Testando scraping para {test_date.date()} - {test_team}...")
    referees = scraper.scrape_game_referees(test_date, test_team)
    
    if referees:
        print(f"✅ Árbitros: {', '.join(referees)}")
    else:
        print("❌ Nenhum árbitro encontrado")
    
    # Testar temporada (limitado a 5 jogos)
    print("\n🏀 Testando scraping de temporada (5 jogos)...")
    df = scraper.scrape_season_referees('2024-25', max_games=5)
    print(df)
