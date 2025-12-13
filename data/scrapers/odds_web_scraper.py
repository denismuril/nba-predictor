"""
OddsPedia Web Scraper - TIER 1 (Gratuito).

Scraper robusto para obter odds de apostas da NBA via web scraping.
Usa Playwright para renderizar JavaScript e BeautifulSoup para parsing.

Features:
- Anti-detecção (fake user-agent, delays, scroll suave)
- Validação de nomes de times contra TEAM_ABBREV_MAP
- Fallback silencioso em caso de erro
"""

import logging
import random
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from bs4 import BeautifulSoup

from config.constants import TEAM_ABBREV_MAP, TEAMS_MAP

logger = logging.getLogger(__name__)


class OddsPediaScraper:
    """
    Scraper para OddsPedia - odds gratuitas via web scraping.
    
    Atributos:
        BASE_URL: URL alvo para scraping
        MIN_DELAY: Delay mínimo entre ações (segundos)
        MAX_DELAY: Delay máximo entre ações (segundos)
    """
    
    BASE_URL = "https://oddspedia.com/br/basquete/eua/nba"
    MIN_DELAY = 2.0
    MAX_DELAY = 4.0
    
    def __init__(self, headless: bool = True):
        """
        Inicializa o scraper.
        
        Args:
            headless: Se True, executa navegador em modo headless (sem GUI)
        """
        self.headless = headless
        self._browser = None
        self._context = None
        self._page = None
        
        # Cache reverso para lookup rápido
        self._team_lookup = self._build_team_lookup()
    
    def _build_team_lookup(self) -> Dict[str, str]:
        """
        Constrói dicionário de lookup para normalização de nomes.
        
        Returns:
            Dict mapeando variantes de nomes para nomes oficiais
        """
        lookup = {}
        
        # Adicionar nomes completos do TEAM_ABBREV_MAP
        for full_name in TEAM_ABBREV_MAP.keys():
            lookup[full_name.lower()] = full_name
            
            # Adicionar partes do nome (ex: "Boston Celtics" -> "celtics")
            parts = full_name.lower().split()
            for part in parts:
                if len(part) > 3 and part not in ['the', 'los', 'new', 'san']:
                    lookup[part] = full_name
        
        # Adicionar aliases do TEAMS_MAP
        for alias, full_name in TEAMS_MAP.items():
            lookup[alias.lower()] = full_name
        
        return lookup
    
    def _normalize_team_name(self, raw_name: str) -> Optional[str]:
        """
        Normaliza nome de time extraído do site.
        
        Tenta encontrar o time correspondente no TEAM_ABBREV_MAP.
        Se não encontrar match, retorna None (NÃO ADIVINHA).
        
        Args:
            raw_name: Nome cru extraído do HTML
            
        Returns:
            Nome oficial do time ou None se não encontrado
        """
        if not raw_name:
            return None
        
        cleaned = raw_name.strip().lower()
        
        # 1. Match exato
        if cleaned in self._team_lookup:
            return self._team_lookup[cleaned]
        
        # 2. Match parcial - verificar se algum termo conhecido está no nome
        for key, full_name in self._team_lookup.items():
            # Evitar matches muito curtos (ex: "76" em "76ers")
            if len(key) >= 4 and key in cleaned:
                return full_name
        
        # 3. Verificar cada palavra do nome
        words = cleaned.split()
        for word in words:
            if word in self._team_lookup:
                return self._team_lookup[word]
        
        # Não encontrou - NÃO ADIVINHAR
        logger.debug(f"Time não reconhecido (descartado): '{raw_name}'")
        return None
    
    def _get_user_agent(self) -> str:
        """
        Obtém user-agent aleatório para anti-detecção.
        
        Returns:
            String de user-agent
        """
        try:
            from fake_useragent import UserAgent
            ua = UserAgent()
            return ua.random
        except Exception:
            # Fallback para user-agent padrão
            return (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
    
    def _random_delay(self, min_sec: float = None, max_sec: float = None) -> None:
        """
        Aplica delay aleatório para simular comportamento humano.
        
        Args:
            min_sec: Delay mínimo (default: MIN_DELAY)
            max_sec: Delay máximo (default: MAX_DELAY)
        """
        min_delay = min_sec or self.MIN_DELAY
        max_delay = max_sec or self.MAX_DELAY
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    async def _scroll_page(self, page) -> None:
        """
        Executa scroll suave para carregar lazy loading.
        
        Args:
            page: Objeto page do Playwright
        """
        # Scroll para baixo em incrementos
        viewport_height = await page.evaluate("window.innerHeight")
        scroll_height = await page.evaluate("document.body.scrollHeight")
        
        current_position = 0
        scroll_step = viewport_height // 2
        
        while current_position < scroll_height:
            current_position += scroll_step
            await page.evaluate(f"window.scrollTo(0, {current_position})")
            await page.wait_for_timeout(random.randint(300, 600))
        
        # Scroll de volta para o topo
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(500)
    
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
            # Remover espaços e substituir vírgula por ponto
            cleaned = text.strip().replace(',', '.').replace(' ', '')
            value = float(cleaned)
            
            # Validar range razoável
            if 1.01 <= value <= 50.0:
                return value
            return None
        except (ValueError, TypeError):
            return None
    
    def _extract_games_from_html(self, html: str) -> Dict:
        """
        Extrai jogos e odds do HTML da página.
        
        Args:
            html: Conteúdo HTML da página
            
        Returns:
            Dict no formato padronizado {game_key: {dados}}
        """
        odds_dict = {}
        soup = BeautifulSoup(html, 'lxml')
        
        # Tentar diferentes seletores (site pode mudar)
        selectors = [
            'div[class*="match"]',
            'div[class*="event"]',
            'div[class*="game"]',
            'article[class*="match"]',
            'li[class*="match"]',
        ]
        
        games = []
        for selector in selectors:
            games = soup.select(selector)
            if games:
                logger.debug(f"Encontrados {len(games)} jogos com seletor: {selector}")
                break
        
        if not games:
            # Fallback: procurar por padrões de odds
            logger.warning("Nenhum seletor padrão funcionou, tentando parsing alternativo")
            return self._parse_alternative(soup)
        
        for game in games:
            try:
                result = self._extract_single_game(game)
                if result:
                    game_key, game_data = result
                    odds_dict[game_key] = game_data
            except Exception as e:
                logger.debug(f"Erro ao extrair jogo: {e}")
                continue
        
        return odds_dict
    
    def _extract_single_game(self, game_element) -> Optional[Tuple[str, Dict]]:
        """
        Extrai dados de um único jogo.
        
        Args:
            game_element: Elemento BeautifulSoup do jogo
            
        Returns:
            Tuple (game_key, game_data) ou None
        """
        # Procurar nomes dos times
        team_selectors = [
            'span[class*="team"]',
            'div[class*="team"]',
            'span[class*="participant"]',
            'div[class*="participant"]',
            'a[class*="team"]',
        ]
        
        teams = []
        for selector in team_selectors:
            teams = game_element.select(selector)
            if len(teams) >= 2:
                break
        
        if len(teams) < 2:
            return None
        
        # Extrair e normalizar nomes
        home_raw = teams[0].get_text(strip=True)
        away_raw = teams[1].get_text(strip=True)
        
        home_team = self._normalize_team_name(home_raw)
        away_team = self._normalize_team_name(away_raw)
        
        # CRÍTICO: Se qualquer time não for reconhecido, descartar
        if not home_team or not away_team:
            logger.debug(f"Jogo descartado - times não reconhecidos: '{home_raw}' vs '{away_raw}'")
            return None
        
        # Procurar odds
        odds_selectors = [
            'span[class*="odd"]',
            'div[class*="odd"]',
            'button[class*="odd"]',
            'span[class*="price"]',
            'div[class*="price"]',
        ]
        
        odds = []
        for selector in odds_selectors:
            odds = game_element.select(selector)
            if len(odds) >= 2:
                break
        
        if len(odds) < 2:
            return None
        
        # Parsear odds
        home_odds = self._parse_odds_value(odds[0].get_text())
        away_odds = self._parse_odds_value(odds[1].get_text())
        
        if not home_odds or not away_odds:
            return None
        
        # Construir resultado
        game_key = f"{home_team} vs {away_team}"
        game_data = {
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "source": "oddspedia_scraper",
            "timestamp": datetime.now().isoformat()
        }
        
        return game_key, game_data
    
    def _parse_alternative(self, soup: BeautifulSoup) -> Dict:
        """
        Parsing alternativo quando seletores padrão falham.
        
        Args:
            soup: Objeto BeautifulSoup da página
            
        Returns:
            Dict no formato padronizado
        """
        odds_dict = {}
        
        # Tentar encontrar tabela de odds
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:
                    # Tentar extrair time e odds de cada célula
                    try:
                        home_team = self._normalize_team_name(cells[0].get_text())
                        away_team = self._normalize_team_name(cells[1].get_text())
                        
                        if home_team and away_team:
                            home_odds = self._parse_odds_value(cells[2].get_text())
                            away_odds = self._parse_odds_value(cells[3].get_text())
                            
                            if home_odds and away_odds:
                                game_key = f"{home_team} vs {away_team}"
                                odds_dict[game_key] = {
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "home_odds": home_odds,
                                    "away_odds": away_odds,
                                    "source": "oddspedia_scraper",
                                    "timestamp": datetime.now().isoformat()
                                }
                    except Exception:
                        continue
        
        return odds_dict
    
    async def _fetch_odds_async(self) -> Dict:
        """
        Busca odds de forma assíncrona usando Playwright.
        
        Returns:
            Dict com odds no formato padronizado
        """
        from playwright.async_api import async_playwright
        
        odds_dict = {}
        
        async with async_playwright() as p:
            # Configurar navegador
            browser = await p.chromium.launch(headless=self.headless)
            
            context = await browser.new_context(
                user_agent=self._get_user_agent(),
                viewport={'width': 1920, 'height': 1080},
                locale='pt-BR',
            )
            
            page = await context.new_page()
            
            try:
                logger.info(f"📡 Acessando {self.BASE_URL}...")
                
                # Navegar para página
                await page.goto(
                    self.BASE_URL,
                    timeout=30000,
                    wait_until='networkidle'
                )
                
                # Delay para parecer humano
                await page.wait_for_timeout(random.randint(2000, 4000))
                
                # Scroll para carregar lazy loading
                await self._scroll_page(page)
                
                # Aguardar conteúdo carregar
                await page.wait_for_timeout(random.randint(1000, 2000))
                
                # Extrair HTML
                html = await page.content()
                
                # Parsear odds
                odds_dict = self._extract_games_from_html(html)
                
                logger.info(f"✅ OddsPedia: {len(odds_dict)} jogos extraídos")
                
            except Exception as e:
                logger.error(f"❌ Erro durante scraping: {e}")
                raise
            
            finally:
                await browser.close()
        
        return odds_dict
    
    def fetch_odds(self) -> Dict:
        """
        Busca odds do OddsPedia.
        
        Método principal público. Executa scraping de forma síncrona
        usando asyncio internamente.
        
        Returns:
            Dict com odds no formato padronizado:
            {
                "HomeTeam vs AwayTeam": {
                    "home_team": "HomeTeam",
                    "away_team": "AwayTeam",
                    "home_odds": 1.90,
                    "away_odds": 1.90,
                    "source": "oddspedia_scraper",
                    "timestamp": "..."
                }
            }
            
        Raises:
            Exception: Se scraping falhar completamente
        """
        import asyncio
        
        try:
            # Rodar async em ambiente sync
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # Se já tem loop rodando (ex: Jupyter), usar nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    pass
                return loop.run_until_complete(self._fetch_odds_async())
            else:
                return loop.run_until_complete(self._fetch_odds_async())
                
        except Exception as e:
            logger.error(f"❌ OddsPedia scraper falhou: {e}")
            raise


# Alias para compatibilidade
OddsPediaClient = OddsPediaScraper
