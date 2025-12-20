"""
OddsAgora Scraper - Site brasileiro de comparação de odds.

URL: https://www.oddsagora.com.br/basketball/usa/nba/
"""

import asyncio
import logging
from typing import Dict, Optional

from bs4 import BeautifulSoup

from data.scrapers.odds_sites.base_scraper import BaseSiteScraper, PlaywrightMixin

logger = logging.getLogger(__name__)


class OddsAgoraScraper(BaseSiteScraper, PlaywrightMixin):
    """
    Scraper para OddsAgora - odds de basquete NBA.
    
    Site brasileiro com odds de múltiplas casas de apostas.
    Usa tabelas HTML para exibir odds.
    """
    
    SITE_NAME = "oddsagora"
    BASE_URL = "https://www.oddsagora.com.br/basketball/usa/nba/"
    REQUIRES_JS = True
    MIN_DELAY = 2.0
    MAX_DELAY = 4.0
    
    def fetch_odds(self) -> Dict:
        """
        Busca odds do OddsAgora.
        
        Returns:
            Dict com odds no formato padronizado
        """
        try:
            return asyncio.run(self._fetch_odds_async())
        except RuntimeError:
            # Se já existe event loop rodando
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
            logger.error("Playwright não instalado. Execute: pip install playwright && playwright install")
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
                
                # Aguarda Cloudflare se necessário
                await self._wait_for_cloudflare(page)
                
                # Aguarda carregamento
                await page.wait_for_timeout(3000)
                
                # Scroll para carregar conteúdo lazy
                await self._scroll_page_smooth(page)
                
                # Obtém HTML
                html = await page.content()
                odds_dict = self._extract_games(html)
                
                logger.info(f"[{self.SITE_NAME}] Extraídos {len(odds_dict)} jogos")
                
            except Exception as e:
                logger.error(f"[{self.SITE_NAME}] Erro durante scraping: {e}")
            finally:
                await browser.close()
        
        return odds_dict
    
    def _extract_games(self, html: str) -> Dict:
        """
        Extrai jogos e odds do HTML.
        
        Args:
            html: Conteúdo HTML da página
            
        Returns:
            Dict com jogos extraídos
        """
        odds_dict = {}
        soup = BeautifulSoup(html, 'lxml')
        
        # Seletores para OddsAgora - estrutura comum de sites de odds
        game_selectors = [
            '.event-row',
            '.match-row', 
            '[class*="event"]',
            '[class*="match"]',
            '.game-item',
            'tr[data-event]',
        ]
        
        games = []
        for selector in game_selectors:
            games = soup.select(selector)
            if games:
                logger.debug(f"[{self.SITE_NAME}] Encontrados {len(games)} jogos com seletor: {selector}")
                break
        
        if not games:
            # Fallback: procurar por padrões de times NBA
            odds_dict = self._extract_by_team_patterns(soup)
            return odds_dict
        
        for game in games:
            try:
                result = self._extract_single_game(game)
                if result:
                    game_key, game_data = result
                    odds_dict[game_key] = game_data
            except Exception as e:
                logger.debug(f"[{self.SITE_NAME}] Erro ao extrair jogo: {e}")
                continue
        
        return odds_dict
    
    def _extract_single_game(self, game_element) -> Optional[tuple]:
        """
        Extrai dados de um único jogo.
        
        Args:
            game_element: Elemento BeautifulSoup do jogo
            
        Returns:
            Tuple (game_key, game_data) ou None
        """
        # Tenta extrair nomes dos times
        team_selectors = [
            '.team-name',
            '.participant-name',
            '[class*="team"]',
            '.home-team, .away-team',
        ]
        
        teams = []
        for selector in team_selectors:
            team_elements = game_element.select(selector)
            if len(team_elements) >= 2:
                teams = [t.get_text(strip=True) for t in team_elements[:2]]
                break
        
        if len(teams) < 2:
            return None
        
        home_team = self._normalize_team_name(teams[0])
        away_team = self._normalize_team_name(teams[1])
        
        if not home_team or not away_team:
            return None
        
        # Tenta extrair odds
        odds_selectors = [
            '.odd-value',
            '.odds-value', 
            '[class*="odd"]',
            '.price',
            '.coefficient',
        ]
        
        odds_values = []
        for selector in odds_selectors:
            odds_elements = game_element.select(selector)
            if odds_elements:
                for el in odds_elements:
                    val = self._parse_odds_value(el.get_text())
                    if val:
                        odds_values.append(val)
                if len(odds_values) >= 2:
                    break
        
        if len(odds_values) < 2:
            return None
        
        home_odds = odds_values[0]
        away_odds = odds_values[1]
        
        game_key = self._create_game_key(home_team, away_team)
        game_data = self._create_game_data(home_team, away_team, home_odds, away_odds)
        
        return game_key, game_data
    
    def _extract_by_team_patterns(self, soup: BeautifulSoup) -> Dict:
        """
        Extração alternativa usando padrões de texto.
        
        Procura por nomes de times NBA e odds próximas.
        """
        odds_dict = {}
        text = soup.get_text()
        
        # Lista de times para buscar
        nba_teams = list(self._team_lookup.values())
        nba_teams = list(set(nba_teams))  # Remove duplicatas
        
        # Busca por padrões de odds decimais
        import re
        odds_pattern = re.compile(r'\b(\d+[.,]\d{2})\b')
        
        # Por ora, retorna vazio se não encontrar estrutura clara
        # Implementação mais avançada pode usar NLP
        logger.debug(f"[{self.SITE_NAME}] Fallback por padrões não encontrou jogos")
        return odds_dict
