"""
SportyTrader Scraper - Site europeu de odds.

URL: https://www.sportytrader.com/pt-br/odds/basquete/
"""

import asyncio
import logging
from typing import Dict, Optional

from bs4 import BeautifulSoup

from data.scrapers.odds_sites.base_scraper import BaseSiteScraper, PlaywrightMixin

logger = logging.getLogger(__name__)


class SportyTraderScraper(BaseSiteScraper, PlaywrightMixin):
    """
    Scraper para SportyTrader - odds de basquete NBA.
    
    Site europeu com versão em português brasileiro.
    Usa tabelas HTML tradicionais.
    """
    
    SITE_NAME = "sportytrader"
    BASE_URL = "https://www.sportytrader.com/pt-br/odds/basquete/usa/nba/"
    REQUIRES_JS = True
    MIN_DELAY = 2.0
    MAX_DELAY = 4.0
    
    def fetch_odds(self) -> Dict:
        """Busca odds do SportyTrader."""
        try:
            return asyncio.run(self._fetch_odds_async())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._fetch_odds_async())
        except Exception as e:
            logger.error(f"[{self.SITE_NAME}] Erro: {e}")
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
                logger.error(f"[{self.SITE_NAME}] Erro: {e}")
            finally:
                await browser.close()
        
        return odds_dict
    
    def _extract_games(self, html: str) -> Dict:
        """Extrai jogos e odds do HTML."""
        odds_dict = {}
        soup = BeautifulSoup(html, 'lxml')
        
        # SportyTrader usa estrutura de tabelas
        game_selectors = [
            'tr.match',
            'tr[class*="match"]',
            '.match-item',
            '[class*="event-row"]',
            'table tbody tr',
        ]
        
        games = []
        for selector in game_selectors:
            games = soup.select(selector)
            if games:
                logger.debug(f"[{self.SITE_NAME}] Encontrados {len(games)} jogos com: {selector}")
                break
        
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
        # Para tabelas, os times geralmente estão em células específicas
        cells = game_element.select('td')
        
        if len(cells) < 3:
            return None
        
        # Tenta extrair times do texto das células
        teams = []
        for cell in cells[:3]:
            text = cell.get_text(strip=True)
            normalized = self._normalize_team_name(text)
            if normalized:
                teams.append(normalized)
            if len(teams) >= 2:
                break
        
        if len(teams) < 2:
            return None
        
        home_team, away_team = teams[0], teams[1]
        
        # Extrai odds das células
        odds_values = []
        for cell in cells:
            val = self._parse_odds_value(cell.get_text())
            if val:
                odds_values.append(val)
        
        if len(odds_values) < 2:
            return None
        
        game_key = self._create_game_key(home_team, away_team)
        game_data = self._create_game_data(home_team, away_team, odds_values[0], odds_values[1])
        
        return game_key, game_data
