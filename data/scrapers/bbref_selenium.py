"""
Basketball-Reference Selenium Scraper
======================================
Usa Selenium com browser real (Chrome headless) para bypassing anti-bot measures.

Usage:
    from data.scrapers.bbref_selenium import BBRefSeleniumScraper
    
    scraper = BBRefSeleniumScraper()
    df = scraper.get_advanced_stats()
"""
import logging
import pandas as pd
import time
import random
from typing import Optional, List
from pathlib import Path
from io import StringIO

logger = logging.getLogger(__name__)

# Proxies gratuitos (substituir por proxies pagos para produção)
FREE_PROXIES: List[str] = [
    # Formato: "ip:port" ou "http://ip:port"
    # Adicionar proxies funcional aqui se disponíveis
]


class BBRefSeleniumScraper:
    """Selenium-based scraper for Basketball-Reference."""
    
    def __init__(self, headless: bool = True, use_proxy: bool = False):
        self.headless = headless
        self.use_proxy = use_proxy
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
        
        # ============================================================
        # CRITICAL: Opções para funcionar em WSL/Docker/servidor headless
        # ============================================================
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--disable-extensions')
        options.add_argument('--remote-debugging-port=9222')
        # options.add_argument('--single-process')  # ❌ Removido: Pode causar instabilidade em versões novas
        options.add_argument('--disable-setuid-sandbox')
        options.add_argument('--disable-features=VizDisplayCompositor') # Fix for potential WSL crashes
        options.page_load_strategy = 'eager' # Don't wait for full load (images etc)
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-notifications')
        
        # Anti-detection settings
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--start-maximized')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Random User-Agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        ]
        options.add_argument(f'--user-agent={random.choice(user_agents)}')
        
        # Proxy rotation (se habilitado)
        if self.use_proxy and FREE_PROXIES:
            proxy = random.choice(FREE_PROXIES)
            options.add_argument(f'--proxy-server={proxy}')
            logger.info(f"🔄 Usando proxy: {proxy}")
            
        try:
            # Tentar usar webdriver-manager para instalação automática
            from webdriver_manager.chrome import ChromeDriverManager
            from webdriver_manager.core.os_manager import ChromeType
            
            # Verificar se é Chrome ou Chromium
            import shutil
            if shutil.which('chromium-browser') or shutil.which('chromium'):
                # Usar Chromium (comum em Ubuntu/WSL)
                logger.info("🔍 Detectado Chromium, usando driver correspondente...")
                from webdriver_manager.core.os_manager import OperationSystemManager
                service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
            else:
                service = Service(ChromeDriverManager().install())
            
            self._driver = webdriver.Chrome(service=service, options=options)
            
        except Exception as e:
            logger.warning(f"⚠️ webdriver-manager falhou: {e}")
            # Fallback para chromedriver no PATH
            try:
                self._driver = webdriver.Chrome(options=options)
            except Exception as e2:
                logger.error(f"❌ Erro ao iniciar Chrome: {e2}")
                logger.info("💡 Soluções possíveis:")
                logger.info("   1. Instalar Chrome: sudo apt install chromium-browser")
                logger.info("   2. Instalar driver: sudo apt install chromium-chromedriver")
                logger.info("   3. pip install webdriver-manager")
                return None
        
        # Stealth JavaScript injection
        try:
            self._driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
                '''
            })
        except Exception:
            pass  # CDP pode não funcionar em todos os browsers
        
        return self._driver
        
    def get_advanced_stats(self, season: str = "2026") -> Optional[pd.DataFrame]:
        """
        Busca Advanced Stats do Basketball-Reference via Selenium.
        
        Args:
            season: Ano final da temporada (ex: "2026" para 2025-26)
            
        Returns:
            DataFrame com BPM, VORP, PER, WS, etc.
        """
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
        logger.info(f"🔍 [Selenium] Buscando BBRef Advanced Stats: {url}")
        
        driver = self._get_driver()
        if not driver:
            return None
            
        try:
            # Navegar com delay humano
            driver.get(url)
            time.sleep(random.uniform(2, 4))  # Simular comportamento humano
            
            # Aguardar tabela carregar
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "advanced_stats"))
                )
            except Exception:
                # Tentar por classe genérica
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
            
            # Scroll para simular leitura humana
            driver.execute_script("window.scrollTo(0, 500)")
            time.sleep(random.uniform(0.5, 1))
            
            # Extrair HTML da tabela
            page_source = driver.page_source
            
            # Parse com pandas (usando StringIO para evitar FutureWarning)
            dfs = pd.read_html(StringIO(page_source), match="Advanced")
            
            if dfs:
                df = dfs[0]
                
                # Limpar linhas de header repetidas
                if 'Player' in df.columns:
                    df = df[df['Player'] != 'Player'].dropna(subset=['Player'])
                
                # Converter colunas numéricas
                numeric_cols = ['BPM', 'OBPM', 'DBPM', 'VORP', 'PER', 'WS', 'WS/48']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                logger.info(f"✅ [Selenium] BBRef obtido: {len(df)} jogadores")
                return df
                
        except Exception as e:
            logger.error(f"❌ [Selenium] Erro ao buscar BBRef: {e}")
            
        return None
        
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


def get_bbref_with_selenium(season: str = "2026") -> Optional[pd.DataFrame]:
    """
    Função helper para buscar BBRef usando Selenium.
    
    Usage:
        df = get_bbref_with_selenium("2026")
    """
    scraper = BBRefSeleniumScraper(headless=True)
    try:
        return scraper.get_advanced_stats(season)
    finally:
        scraper.close()


if __name__ == "__main__":
    # Teste
    logging.basicConfig(level=logging.INFO)
    df = get_bbref_with_selenium()
    if df is not None:
        print(df.head())
        print(f"\nColunas: {df.columns.tolist()}")
    else:
        print("Falhou")
