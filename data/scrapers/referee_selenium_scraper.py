"""
Referee Selenium Scraper - Basketball-Reference
================================================
Usa Selenium com Chrome headless para buscar árbitros de boxscores.
Baseado no bbref_selenium.py existente.

Autor: NBA Predictor System
Data: 2025-12-05
"""

import sys
from pathlib import Path

# Fix para imports quando executado diretamente
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
import time
import random
import re
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class RefereeSeleniumScraper:
    """Selenium-based scraper para árbitros de jogos no Basketball-Reference."""
    
    # Mapeamento de times para códigos BBRef
    BBREF_TEAM_MAP = {
        'PHX': 'PHO', 'CHA': 'CHO', 'BKN': 'BRK',
        'NOP': 'NOP', 'GS': 'GSW', 'SA': 'SAS',
        'NY': 'NYK', 'NO': 'NOP',
    }
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._driver = None
        
    def _get_driver(self):
        """Initialize Chrome WebDriver with anti-detection settings."""
        if self._driver:
            return self._driver
            
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
        except ImportError:
            logger.error("❌ Selenium não instalado. Instale com: pip install selenium webdriver-manager")
            return None
            
        options = Options()
        
        if self.headless:
            options.add_argument('--headless=new')
        
        # Opções para funcionar em WSL/Docker/servidor headless
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--disable-extensions')
        options.add_argument('--remote-debugging-port=9222')
        options.add_argument('--single-process')
        options.add_argument('--disable-setuid-sandbox')
        
        # Anti-detection settings
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Random User-Agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        ]
        options.add_argument(f'--user-agent={random.choice(user_agents)}')
            
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from webdriver_manager.core.os_manager import ChromeType
            import shutil
            
            if shutil.which('chromium-browser') or shutil.which('chromium'):
                logger.info("🔍 Detectado Chromium...")
                service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
            else:
                service = Service(ChromeDriverManager().install())
            
            self._driver = webdriver.Chrome(service=service, options=options)
            
        except Exception as e:
            logger.warning(f"⚠️ webdriver-manager falhou: {e}")
            try:
                from selenium import webdriver
                self._driver = webdriver.Chrome(options=options)
            except Exception as e2:
                logger.error(f"❌ Erro ao iniciar Chrome: {e2}")
                return None
        
        # Stealth JavaScript injection
        try:
            self._driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                '''
            })
        except Exception:
            pass
        
        return self._driver
    
    def _normalize_team_code(self, team: str) -> str:
        """Normaliza sigla do time para formato BBRef."""
        team_upper = team.upper().strip()
        return self.BBREF_TEAM_MAP.get(team_upper, team_upper)
    
    def _get_boxscore_url(self, date: datetime, home_team: str) -> str:
        """Constrói URL do boxscore no Basketball-Reference."""
        date_str = date.strftime('%Y%m%d')
        team_code = self._normalize_team_code(home_team)
        return f"https://www.basketball-reference.com/boxscores/{date_str}0{team_code}.html"
    
    def scrape_game_referees(self, date: datetime, home_team: str) -> Optional[List[str]]:
        """
        Busca os árbitros de um jogo específico via Selenium.
        
        Args:
            date: Data do jogo
            home_team: Sigla do time da casa (ex: 'BOS', 'LAL')
            
        Returns:
            Lista com nomes dos árbitros ou None se não encontrado
        """
        url = self._get_boxscore_url(date, home_team)
        logger.info(f"🔍 [Selenium] Buscando árbitros: {url}")
        
        driver = self._get_driver()
        if not driver:
            return None
            
        try:
            # Navegar com delay humano
            driver.get(url)
            time.sleep(random.uniform(2, 4))
            
            # Aguardar página carregar
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "scorebox"))
                )
            except Exception:
                # Page may still have loaded, continue
                pass
            
            # Scroll para simular leitura humana
            driver.execute_script("window.scrollTo(0, 800)")
            time.sleep(random.uniform(0.5, 1))
            
            # Extrair HTML
            page_source = driver.page_source
            
            # Buscar "Officials:" no texto
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            
            officials = []
            
            # Método 1: Procurar strong tag com "Officials"
            for strong in soup.find_all('strong'):
                if 'Officials' in strong.get_text():
                    parent = strong.parent
                    text = parent.get_text()
                    match = re.search(r'Officials?:\s*(.+)', text)
                    if match:
                        refs_text = match.group(1)
                        # Separar apenas por vírgula (não por ponto, que é usado em iniciais como "J. Smith")
                        refs = [r.strip() for r in refs_text.split(',') if r.strip()]
                        if refs:
                            officials = refs[:3]
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
                                # Separar apenas por vírgula
                                refs = [r.strip() for r in refs_text.split(',') if r.strip()]
                                if refs:
                                    officials = refs[:3]
                                    break
            
            if officials:
                logger.info(f"✅ Árbitros encontrados: {', '.join(officials)}")
                return officials
            else:
                logger.warning(f"⚠️ Nenhum árbitro encontrado na página")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro ao fazer scraping: {e}")
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
        logger.info(f"🏀 [Selenium] Iniciando scraping de árbitros para temporada {season}...")
        
        cache_dir = Path('data/cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f'referees_{season.replace("-", "_")}_selenium.csv'
        
        # Verificar cache
        if cache_file.exists():
            logger.info(f"📂 Carregando do cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        # Carregar schedule da temporada
        from data.repositories.db_manager import get_db_manager
        db = get_db_manager()
        df_games = db.get_comprehensive_history()
        
        if df_games is None or df_games.empty:
            logger.error("❌ Nenhum jogo histórico encontrado")
            return pd.DataFrame()
        
        # Filtrar por temporada
        df_games['date'] = pd.to_datetime(df_games['date'])
        year_start = int(season.split('-')[0])
        start_date = pd.Timestamp(f'{year_start}-10-01')
        end_date = pd.Timestamp(f'{year_start + 1}-06-30')
        
        df_season = df_games[(df_games['date'] >= start_date) & (df_games['date'] <= end_date)].copy()
        df_season = df_season.sort_values('date').reset_index(drop=True)
        
        logger.info(f"📊 {len(df_season)} jogos na temporada {season}")
        
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
            
            if (idx + 1) % 5 == 0:
                logger.info(f"   Processados {idx + 1}/{len(df_season)} jogos...")
                
            # Delay maior entre jogos (5-8 segundos)
            time.sleep(random.uniform(5, 8))
        
        df_referees = pd.DataFrame(results)
        
        # Salvar cache
        df_referees.to_csv(cache_file, index=False)
        logger.info(f"💾 Salvo em cache: {cache_file}")
        
        success_rate = df_referees['referees_csv'].notna().sum() / len(df_referees) * 100
        logger.info(f"✅ Scraping concluído: {success_rate:.1f}% de jogos com árbitros")
        
        return df_referees
    
    def close(self):
        """Fecha o browser."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
            
    def __del__(self):
        self.close()


def get_referees_with_selenium(date: datetime, home_team: str) -> Optional[List[str]]:
    """
    Função helper para buscar árbitros de um jogo usando Selenium.
    
    Usage:
        refs = get_referees_with_selenium(datetime(2024, 12, 1), "BOS")
    """
    scraper = RefereeSeleniumScraper(headless=True)
    try:
        return scraper.scrape_game_referees(date, home_team)
    finally:
        scraper.close()


if __name__ == "__main__":
    # Teste
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    scraper = RefereeSeleniumScraper(headless=True)
    
    # Testar com um jogo que existe (data anterior)
    test_date = datetime(2024, 11, 1)  # Usar data passada
    test_team = "BOS"
    
    print(f"\n🔍 Testando scraping Selenium para {test_date.date()} - {test_team}...")
    referees = scraper.scrape_game_referees(test_date, test_team)
    
    if referees:
        print(f"✅ Árbitros: {', '.join(referees)}")
    else:
        print("❌ Nenhum árbitro encontrado")
    
    scraper.close()
    
    # Testar temporada (2 jogos)
    print("\n🏀 Testando scraping de temporada (2 jogos)...")
    scraper2 = RefereeSeleniumScraper(headless=True)
    df = scraper2.scrape_season_referees('2024-25', max_games=2)
    print(df)
    scraper2.close()
