"""
Base Scraper para sites de odds.

Classe abstrata com métodos comuns para todos os scrapers de odds.
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from datetime import datetime

from config.constants import TEAM_ABBREV_MAP, TEAMS_MAP

logger = logging.getLogger(__name__)


class BaseSiteScraper(ABC):
    """
    Classe base abstrata para scrapers de sites de odds.
    
    Atributos:
        SITE_NAME: Nome identificador do site
        BASE_URL: URL base para scraping
        REQUIRES_JS: Se True, usa Playwright (JavaScript necessário)
        MIN_DELAY: Delay mínimo entre ações
        MAX_DELAY: Delay máximo entre ações
    """
    
    SITE_NAME: str = "base"
    BASE_URL: str = ""
    REQUIRES_JS: bool = True
    MIN_DELAY: float = 2.0
    MAX_DELAY: float = 5.0
    CLOUDFLARE_WAIT: float = 15.0
    
    def __init__(self, headless: bool = True):
        """
        Inicializa o scraper.
        
        Args:
            headless: Se True, executa navegador sem GUI
        """
        self.headless = headless
        self._team_lookup = self._build_team_lookup()
    
    def _build_team_lookup(self) -> Dict[str, str]:
        """
        Constrói dicionário de lookup para normalização de nomes de times.
        
        Returns:
            Dict mapeando variantes de nomes para nomes oficiais
        """
        lookup = {}
        
        # Adiciona do TEAM_ABBREV_MAP
        for full_name, abbrev in TEAM_ABBREV_MAP.items():
            lookup[full_name.lower()] = full_name
            lookup[abbrev.lower()] = full_name
        
        # Adiciona variantes comuns
        variants = {
            "lakers": "Los Angeles Lakers",
            "la lakers": "Los Angeles Lakers",
            "clippers": "Los Angeles Clippers", 
            "la clippers": "Los Angeles Clippers",
            "knicks": "New York Knicks",
            "ny knicks": "New York Knicks",
            "nets": "Brooklyn Nets",
            "sixers": "Philadelphia 76ers",
            "76ers": "Philadelphia 76ers",
            "blazers": "Portland Trail Blazers",
            "trail blazers": "Portland Trail Blazers",
            "timberwolves": "Minnesota Timberwolves",
            "t-wolves": "Minnesota Timberwolves",
            "cavs": "Cleveland Cavaliers",
            "cavaliers": "Cleveland Cavaliers",
            "mavs": "Dallas Mavericks",
            "mavericks": "Dallas Mavericks",
            "thunder": "Oklahoma City Thunder",
            "okc": "Oklahoma City Thunder",
            "okc thunder": "Oklahoma City Thunder",
            "warriors": "Golden State Warriors",
            "gsw": "Golden State Warriors",
            "celtics": "Boston Celtics",
            "heat": "Miami Heat",
            "bulls": "Chicago Bulls",
            "spurs": "San Antonio Spurs",
            "rockets": "Houston Rockets",
            "nuggets": "Denver Nuggets",
            "jazz": "Utah Jazz",
            "suns": "Phoenix Suns",
            "kings": "Sacramento Kings",
            "pelicans": "New Orleans Pelicans",
            "grizzlies": "Memphis Grizzlies",
            "hawks": "Atlanta Hawks",
            "hornets": "Charlotte Hornets",
            "magic": "Orlando Magic",
            "pacers": "Indiana Pacers",
            "pistons": "Detroit Pistons",
            "raptors": "Toronto Raptors",
            "wizards": "Washington Wizards",
            "bucks": "Milwaukee Bucks",
        }
        
        for variant, official in variants.items():
            lookup[variant.lower()] = official
            
        return lookup
    
    def _normalize_team_name(self, raw_name: str) -> Optional[str]:
        """
        Normaliza nome de time extraído do site.
        
        Args:
            raw_name: Nome cru extraído do HTML
            
        Returns:
            Nome oficial do time ou None se não encontrado
        """
        if not raw_name:
            return None
            
        cleaned = raw_name.strip().lower()
        
        # Lookup direto
        if cleaned in self._team_lookup:
            return self._team_lookup[cleaned]
        
        # Busca parcial
        for key, value in self._team_lookup.items():
            if key in cleaned or cleaned in key:
                return value
                
        logger.debug(f"[{self.SITE_NAME}] Time não reconhecido: '{raw_name}'")
        return None
    
    def _random_delay(self, min_sec: float = None, max_sec: float = None):
        """Aplica delay aleatório para simular comportamento humano."""
        min_sec = min_sec or self.MIN_DELAY
        max_sec = max_sec or self.MAX_DELAY
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)
    
    def _parse_odds_value(self, text: str) -> Optional[float]:
        """
        Converte texto de odds para float.
        
        Args:
            text: Texto contendo o valor da odd (ex: "1.85", "2,50")
            
        Returns:
            Valor float ou None se inválido
        """
        if not text:
            return None
            
        try:
            cleaned = text.strip().replace(',', '.').replace(' ', '')
            value = float(cleaned)
            
            # Validação básica
            if 1.01 <= value <= 50.0:
                return value
            return None
        except (ValueError, AttributeError):
            return None
    
    def _get_user_agent(self) -> str:
        """Retorna user-agent aleatório para anti-detecção."""
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]
        return random.choice(agents)
    
    def _create_game_key(self, home_team: str, away_team: str) -> str:
        """Cria chave única para o jogo."""
        return f"{home_team} vs {away_team}"
    
    def _create_game_data(
        self, 
        home_team: str, 
        away_team: str, 
        home_odds: float, 
        away_odds: float
    ) -> Dict:
        """Cria dicionário padronizado de dados do jogo."""
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_odds': home_odds,
            'away_odds': away_odds,
            'source': self.SITE_NAME,
            'timestamp': datetime.now().isoformat()
        }
    
    @abstractmethod
    def fetch_odds(self) -> Dict:
        """
        Método principal para buscar odds.
        
        Returns:
            Dict no formato {game_key: game_data}
        """
        pass


class PlaywrightMixin:
    """Mixin com métodos utilitários para scrapers Playwright."""
    
    async def _scroll_page_smooth(self, page, steps: int = 3):
        """Executa scroll suave para carregar lazy content."""
        for i in range(steps):
            await page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {(i+1)/steps})")
            await page.wait_for_timeout(random.randint(500, 1000))
    
    async def _wait_for_cloudflare(self, page, timeout: int = 20000):
        """Aguarda bypass do Cloudflare challenge."""
        try:
            # Aguarda elemento de challenge desaparecer
            await page.wait_for_function(
                "() => !document.body.classList.contains('challenge') && !document.querySelector('.cf-challenge')",
                timeout=timeout
            )
        except Exception:
            # Se não há challenge, continua
            pass
    
    async def _extract_nuxt_data(self, page) -> Optional[Dict]:
        """Extrai dados do objeto window.__NUXT__ (para sites Nuxt.js)."""
        try:
            nuxt_data = await page.evaluate("() => window.__NUXT__?.data?.[0]")
            return nuxt_data
        except Exception:
            return None
