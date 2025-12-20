"""
OddsScanner Scraper - Site de comparação de odds.

URL: https://oddsscanner.com/br/basquete
"""

import asyncio
import logging
from typing import Dict, Optional

from bs4 import BeautifulSoup

from data.scrapers.odds_sites.base_scraper import BaseSiteScraper, PlaywrightMixin

logger = logging.getLogger(__name__)


class OddsScannerScraper(BaseSiteScraper, PlaywrightMixin):
    """
    Scraper para OddsScanner - odds de basquete.
    
    Site internacional com versão brasileira.
    Usa cards/elementos dinâmicos para exibir odds.
    """
    
    SITE_NAME = "oddsscanner"
    BASE_URL = "https://oddsscanner.com/br/basquete/eua/nba"
    REQUIRES_JS = True
    MIN_DELAY = 2.0
    MAX_DELAY = 4.0
    
    def fetch_odds(self) -> Dict:
        """
        Busca odds do OddsScanner.
        
        Returns:
            Dict com odds no formato padronizado
        """
        try:
            return asyncio.run(self._fetch_odds_async())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._fetch_odds_async())
        except Exception as e:
            logger.error(f"[{self.SITE_NAME}] Erro ao buscar odds: {e}")
            return {}
    
    async def _fetch_odds_async(self) -> Dict:
        """Busca odds de forma assíncrona."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error("Playwright não instalado")
            return {}
        
        odds_dict = {}
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                user_agent=self._get_user_agent(),
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            try:
                logger.info(f"[{self.SITE_NAME}] Acessando {self.BASE_URL}")
                await page.goto(self.BASE_URL, wait_until='networkidle', timeout=30000)
                
                await self._wait_for_cloudflare(page)
                await page.wait_for_timeout(3000)
                await self._scroll_page_smooth(page)
                
                html = await page.content()
                odds_dict = self._extract_games(html)
                
                logger.info(f"[{self.SITE_NAME}] Extraídos {len(odds_dict)} jogos")
                
            except Exception as e:
                logger.error(f"[{self.SITE_NAME}] Erro durante scraping: {e}")
            finally:
                await browser.close()
        
        return odds_dict
    
    def _extract_games(self, html: str) -> Dict:
        """Extrai jogos e odds do HTML."""
        odds_dict = {}
        soup = BeautifulSoup(html, 'lxml')
        
        # Seletores específicos para OddsScanner
        game_selectors = [
            '[class*="MatchCard"]',
            '[class*="match-card"]',
            '.event-card',
            '[class*="Event"]',
            '[class*="game-row"]',
            'article[class*="match"]',
        ]
        
        games = []
        for selector in game_selectors:
            games = soup.select(selector)
            if games:
                logger.debug(f"[{self.SITE_NAME}] Encontrados {len(games)} jogos com: {selector}")
                break
        
        if not games:
            # Tenta extração alternativa
            return self._extract_from_text(soup)
        
        for game in games:
            try:
                result = self._extract_single_game(game)
                if result:
                    game_key, game_data = result
                    odds_dict[game_key] = game_data
            except Exception as e:
                logger.debug(f"[{self.SITE_NAME}] Erro: {e}")
                continue
        
        return odds_dict
    
    def _extract_single_game(self, game_element) -> Optional[tuple]:
        """Extrai dados de um único jogo."""
        # Seletores para times
        team_selectors = [
            '[class*="TeamName"]',
            '[class*="team-name"]',
            '[class*="participant"]',
            '.team',
        ]
        
        teams = []
        for selector in team_selectors:
            elements = game_element.select(selector)
            if len(elements) >= 2:
                teams = [e.get_text(strip=True) for e in elements[:2]]
                break
        
        if len(teams) < 2:
            return None
        
        home_team = self._normalize_team_name(teams[0])
        away_team = self._normalize_team_name(teams[1])
        
        if not home_team or not away_team:
            return None
        
        # Seletores para odds
        odds_selectors = [
            '[class*="OddsValue"]',
            '[class*="odds-value"]',
            '[class*="bestOdd"]',
            '.coefficient',
            '[class*="price"]',
        ]
        
        odds_values = []
        for selector in odds_selectors:
            elements = game_element.select(selector)
            for el in elements:
                val = self._parse_odds_value(el.get_text())
                if val:
                    odds_values.append(val)
            if len(odds_values) >= 2:
                break
        
        if len(odds_values) < 2:
            return None
        
        game_key = self._create_game_key(home_team, away_team)
        game_data = self._create_game_data(home_team, away_team, odds_values[0], odds_values[1])
        
        return game_key, game_data
    
    def _extract_from_text(self, soup: BeautifulSoup) -> Dict:
        """Extração alternativa baseada em padrões de texto."""
        # Por ora retorna vazio
        return {}
