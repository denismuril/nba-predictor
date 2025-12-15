"""
SBR (Sportsbook Review) Scraper - FONTE PRIMÁRIA para odds de apostas NBA.

Este scraper utiliza Playwright para extrair odds em tempo real do site
Sportsbook Review. É a fonte de maior prioridade no sistema, sendo econômico
pois não consome créditos de APIs pagas.

Características:
- User-agents rotativos para anti-detecção
- Seletores flexíveis com múltiplos fallbacks
- Delays humanizados entre ações
- Suporte completo a asyncio

v24.0: Implementação inicial como parte da arquitetura de provedores.
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import List, Optional

try:
    from playwright.async_api import async_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from data.interfaces.odds_provider import OddsProvider, GameOdds
from config.constants import TEAM_ABBREV_MAP, TEAMS_MAP

logger = logging.getLogger(__name__)

# User-agents realistas para rotação
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]

# Mapeamento reverso para normalização de nomes de times
TEAM_NAME_LOOKUP = {}
for full_name, abbrev in TEAM_ABBREV_MAP.items():
    TEAM_NAME_LOOKUP[full_name.lower()] = full_name
    TEAM_NAME_LOOKUP[abbrev.lower()] = full_name
    # Adiciona variantes comuns
    if " " in full_name:
        parts = full_name.split()
        TEAM_NAME_LOOKUP[parts[-1].lower()] = full_name  # Nome da cidade/estado
        if len(parts) > 1:
            TEAM_NAME_LOOKUP[parts[0].lower()] = full_name  # Mascote


class SBRScraper(OddsProvider):
    """
    Scraper para Sportsbook Review - Fonte primária de odds (TIER 1).

    Utiliza Playwright para navegar no site e extrair odds de forma resiliente.
    Implementa técnicas de anti-detecção e seletores flexíveis para maior robustez.

    Atributos:
        BASE_URL: URL base do site SBR
        MIN_DELAY: Delay mínimo entre ações (segundos)
        MAX_DELAY: Delay máximo entre ações (segundos)
        TIMEOUT: Tempo limite para operações de rede (ms)
    """

    BASE_URL = "https://www.sportsbookreview.com/betting-odds/nba-basketball/"
    MIN_DELAY = 1.0
    MAX_DELAY = 3.0
    TIMEOUT = 30000  # 30 segundos

    def __init__(self, headless: bool = True):
        """
        Inicializa o scraper.

        Args:
            headless: Se True, executa navegador em modo headless (sem GUI)
        """
        self.headless = headless
        self._browser: Optional[Browser] = None

    @property
    def name(self) -> str:
        """Nome identificador do provedor."""
        return "sbr_scraper"

    @property
    def priority(self) -> int:
        """Prioridade do provedor (1 = mais alta)."""
        return 1

    def _get_random_user_agent(self) -> str:
        """Obtém user-agent aleatório para anti-detecção."""
        return random.choice(USER_AGENTS)

    async def _random_delay(self, min_sec: float = None, max_sec: float = None):
        """
        Aplica delay aleatório para simular comportamento humano.

        Args:
            min_sec: Delay mínimo (default: MIN_DELAY)
            max_sec: Delay máximo (default: MAX_DELAY)
        """
        min_sec = min_sec or self.MIN_DELAY
        max_sec = max_sec or self.MAX_DELAY
        await asyncio.sleep(random.uniform(min_sec, max_sec))

    def _normalize_team_name(self, raw_name: str) -> Optional[str]:
        """
        Normaliza nome de time extraído do site.

        Args:
            raw_name: Nome cru extraído do HTML

        Retorna:
            Nome oficial do time ou None se não encontrado
        """
        if not raw_name:
            return None

        clean_name = raw_name.strip().lower()

        # Busca direta no lookup
        if clean_name in TEAM_NAME_LOOKUP:
            return TEAM_NAME_LOOKUP[clean_name]

        # Busca parcial - procura se algum termo conhecido está no nome
        for key, value in TEAM_NAME_LOOKUP.items():
            if key in clean_name or clean_name in key:
                return value

        logger.warning(f"Time não reconhecido: '{raw_name}'")
        return None

    async def _scroll_page(self, page: Page):
        """
        Executa scroll suave para carregar conteúdo lazy-loaded.

        Args:
            page: Objeto Page do Playwright
        """
        await page.evaluate("""
            async () => {
                const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
                for (let i = 0; i < 3; i++) {
                    window.scrollBy(0, window.innerHeight / 2);
                    await delay(500);
                }
            }
        """)

    def _parse_odds_value(self, text: str) -> Optional[float]:
        """
        Converte texto de odds para float.

        Suporta formatos:
        - Decimal: "1.85", "2,50" (vírgula europeia)
        - Americano: "-110", "+150"

        Args:
            text: Texto contendo o valor da odd

        Retorna:
            Valor float em formato decimal ou None se inválido
        """
        if not text:
            return None

        text = text.strip()

        try:
            # Formato decimal com vírgula europeia
            if "," in text and "." not in text:
                return float(text.replace(",", "."))

            # Formato americano
            if text.startswith("-") or text.startswith("+"):
                american = int(text)
                if american < 0:
                    return round(1 + (100 / abs(american)), 3)
                else:
                    return round(1 + (american / 100), 3)

            # Formato decimal padrão
            return float(text)

        except (ValueError, ZeroDivisionError):
            return None

    async def _extract_odds_from_page(self, page: Page) -> List[GameOdds]:
        """
        Extrai odds de jogos da página carregada.

        Utiliza múltiplos seletores para maior robustez contra mudanças no HTML.

        Args:
            page: Página do Playwright com conteúdo carregado

        Retorna:
            Lista de GameOdds extraídos
        """
        games = []

        # Seletores flexíveis para encontrar linhas de jogos
        game_selectors = [
            "[data-testid='gamerow']",
            ".GameRows_gameRow__",
            "tr[class*='GameRow']",
            "[class*='eventRow']",
            "div.game-line",
            "table tbody tr",
        ]

        game_elements = []
        for selector in game_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements and len(elements) > 0:
                    game_elements = elements
                    logger.debug(f"Encontrados {len(elements)} jogos com seletor: {selector}")
                    break
            except Exception:
                continue

        if not game_elements:
            logger.warning("Nenhum elemento de jogo encontrado com seletores conhecidos")
            return games

        for element in game_elements:
            try:
                game_odds = await self._extract_single_game(element)
                if game_odds:
                    games.append(game_odds)
            except Exception as e:
                logger.debug(f"Erro ao extrair jogo: {e}")
                continue

        return games

    async def _extract_single_game(self, element) -> Optional[GameOdds]:
        """
        Extrai dados de um único jogo.

        Args:
            element: Elemento do jogo no DOM

        Retorna:
            GameOdds ou None se extração falhar
        """
        # Seletores para times
        team_selectors = [
            "[data-testid='team-name']",
            ".TeamName",
            "[class*='teamName']",
            ".team-name",
            "td:first-child",
            "span.team",
        ]

        # Seletores para odds
        odds_selectors = [
            "[data-testid='odds']",
            ".OddsCell",
            "[class*='odds']",
            "td[class*='line']",
            ".moneyline",
        ]

        # Extrai nomes dos times
        home_team = None
        away_team = None

        for selector in team_selectors:
            try:
                team_elements = await element.query_selector_all(selector)
                if len(team_elements) >= 2:
                    away_text = await team_elements[0].inner_text()
                    home_text = await team_elements[1].inner_text()
                    away_team = self._normalize_team_name(away_text)
                    home_team = self._normalize_team_name(home_text)
                    if home_team and away_team:
                        break
            except Exception:
                continue

        if not home_team or not away_team:
            return None

        # Extrai odds
        home_odds = None
        away_odds = None

        for selector in odds_selectors:
            try:
                odds_elements = await element.query_selector_all(selector)
                if len(odds_elements) >= 2:
                    away_odds_text = await odds_elements[0].inner_text()
                    home_odds_text = await odds_elements[1].inner_text()
                    away_odds = self._parse_odds_value(away_odds_text)
                    home_odds = self._parse_odds_value(home_odds_text)
                    if home_odds and away_odds:
                        break
            except Exception:
                continue

        if not home_odds or not away_odds:
            # Odds não encontradas, usa default com warning
            logger.debug(f"Odds não encontradas para {away_team} @ {home_team}")
            return None

        # Validação básica de odds
        if not (1.01 <= home_odds <= 50.0) or not (1.01 <= away_odds <= 50.0):
            logger.debug(f"Odds fora do range válido: home={home_odds}, away={away_odds}")
            return None

        game_id = f"{away_team.replace(' ', '_')}_vs_{home_team.replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d')}"

        return GameOdds(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            home_odds=home_odds,
            away_odds=away_odds,
            bookmaker="sbr_consensus",
            source=self.name,
            timestamp=datetime.now(),
        )

    async def get_odds(self, date: str) -> List[GameOdds]:
        """
        Obtém odds para jogos em uma data específica.

        Args:
            date: Data no formato "YYYY-MM-DD"

        Retorna:
            Lista de GameOdds para todos os jogos encontrados

        Raises:
            Exception: Se o scraping falhar completamente
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright não está instalado. Execute: pip install playwright && playwright install chromium"
            )

        logger.info(f"🔍 SBR Scraper: Buscando odds para {date}...")
        games = []

        try:
            async with async_playwright() as p:
                # Inicia navegador com configurações stealth
                browser = await p.chromium.launch(
                    headless=self.headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ]
                )

                context = await browser.new_context(
                    user_agent=self._get_random_user_agent(),
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                )

                # Configura headers anti-detecção
                await context.set_extra_http_headers({
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                })

                page = await context.new_page()

                # Navega para a página
                url = self.BASE_URL
                if date:
                    # SBR usa formato de data na URL
                    url = f"{self.BASE_URL}?date={date}"

                await page.goto(url, wait_until="networkidle", timeout=self.TIMEOUT)
                await self._random_delay(0.5, 1.5)

                # Scroll para carregar conteúdo dinâmico
                await self._scroll_page(page)
                await self._random_delay(0.5, 1.0)

                # Extrai odds
                games = await self._extract_odds_from_page(page)

                await browser.close()

        except Exception as e:
            logger.error(f"❌ SBR Scraper falhou: {e}")
            raise

        if games:
            logger.info(f"✅ SBR Scraper: {len(games)} jogos encontrados")
        else:
            logger.warning("⚠️ SBR Scraper: Nenhum jogo encontrado")

        return games

    async def health_check(self) -> bool:
        """
        Verifica se o scraper está funcional.

        Faz uma requisição de teste ao site SBR.

        Retorna:
            True se o site está acessível, False caso contrário
        """
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright não disponível para health check")
            return False

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=self._get_random_user_agent()
                )
                page = await context.new_page()

                response = await page.goto(
                    self.BASE_URL,
                    wait_until="domcontentloaded",
                    timeout=15000
                )

                await browser.close()

                is_healthy = response and response.status == 200
                if is_healthy:
                    logger.info("✅ SBR Scraper: Health check OK")
                else:
                    logger.warning(f"⚠️ SBR Scraper: Health check falhou (status={response.status if response else 'None'})")

                return is_healthy

        except Exception as e:
            logger.error(f"❌ SBR Scraper health check falhou: {e}")
            return False


# Alias para compatibilidade
SportsbookReviewScraper = SBRScraper
