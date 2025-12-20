"""
OddsShark Scraper - Site americano de odds.

URL: https://www.oddsshark.com/nba/odds
"""

import asyncio
import logging
from typing import Dict, Optional

from bs4 import BeautifulSoup

from data.scrapers.odds_sites.base_scraper import BaseSiteScraper, PlaywrightMixin

logger = logging.getLogger(__name__)


class OddsSharkScraper(BaseSiteScraper, PlaywrightMixin):
    """
    Scraper para OddsShark - odds de basquete NBA.
    
    Site americano popular com odds de várias casas.
    Estrutura mais complexa com dados em tempo real.
    """
    
    SITE_NAME = "oddsshark"
    BASE_URL = "https://www.oddsshark.com/nba/odds"
    REQUIRES_JS = True
    MIN_DELAY = 3.0
    MAX_DELAY = 5.0
    
    def fetch_odds(self) -> Dict:
        """Busca odds do OddsShark."""
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
                await page.goto(self.BASE_URL, wait_until='networkidle', timeout=45000)
                
                # OddsShark pode ter mais proteção
                await self._wait_for_cloudflare(page, timeout=25000)
                await page.wait_for_timeout(5000)
                await self._scroll_page_smooth(page, steps=4)
                
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
        
        # OddsShark usa estrutura específica
        game_selectors = [
            '[class*="GameRows"]',
            '[class*="game-row"]',
            '.odds-table tbody tr',
            '[data-testid*="game"]',
            '.matchup-container',
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
        # Seletores para times no OddsShark
        team_selectors = [
            '[class*="TeamName"]',
            '[class*="team-name"]',
            '.team',
            'a[href*="/nba/"]',
        ]
        
        teams = []
        for selector in team_selectors:
            elements = game_element.select(selector)
            for el in elements:
                text = el.get_text(strip=True)
                normalized = self._normalize_team_name(text)
                if normalized and normalized not in teams:
                    teams.append(normalized)
                if len(teams) >= 2:
                    break
            if len(teams) >= 2:
                break
        
        if len(teams) < 2:
            return None
        
        home_team, away_team = teams[0], teams[1]
        
        # Seletores para odds
        odds_selectors = [
            '[class*="MoneyLine"]',
            '[class*="money-line"]',
            '[class*="odds"]',
            '.price',
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
