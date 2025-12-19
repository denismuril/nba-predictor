"""
Covers Scraper - Linhas Over/Under e probabilidades NBA.

Estratégia: Extrai linhas de Over/Under do Covers.com, fonte estabelecida
para odds esportivas.

v26.3: Implementação inicial.
v26.4: Corrigido seletores para estrutura real do site (article.player-prop-article).
"""

import logging
import re
from datetime import datetime
from typing import List, Optional

from data.interfaces.player_props_provider import PlayerPropsProvider, PlayerProp
from data.scrapers.player_name_normalizer import normalize_player_name

try:
    from data.scrapers.stealth_browser import (
        create_stealth_browser,
        human_delay,
        human_scroll,
        navigate_with_retry
    )
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)


class CoversScraper(PlayerPropsProvider):
    """
    Scraper para Covers.com - odds de player props.
    
    Estrutura do site (atualizada dez/2024):
    - Container: article.player-prop-article
    - Nome: a.player-link strong
    - Tipo: div.player-event
    - Linha: texto dentro de div.other-over-odds
    - Odds: div.player-bestOdds-row a span
    """
    
    BASE_URL = "https://www.covers.com/sport/basketball/nba/odds"
    PLAYER_PROPS_URL = "https://www.covers.com/sport/basketball/nba/player-props"
    
    def __init__(self, headless: bool = True):
        self.headless = headless
    
    @property
    def name(self) -> str:
        return "covers"
    
    @property
    def priority(self) -> int:
        return 4
    
    async def get_props(self, date: str) -> List[PlayerProp]:
        """
        Obtém player props do Covers.
        
        Args:
            date: Data no formato "YYYY-MM-DD"
            
        Returns:
            Lista de PlayerProp
        """
        if not STEALTH_AVAILABLE or not BS4_AVAILABLE:
            logger.warning("Dependências não disponíveis para Covers")
            return []
        
        props = []
        
        try:
            async with create_stealth_browser(headless=self.headless) as (browser, context, page):
                logger.info(f"🔍 Covers: Navegando para player props...")
                success = await navigate_with_retry(page, self.PLAYER_PROPS_URL)
                
                if not success:
                    success = await navigate_with_retry(page, self.BASE_URL)
                    if not success:
                        logger.warning("❌ Covers: Falha ao carregar página")
                        return []
                
                # Esperar carregamento dinâmico
                await human_delay(3, 5)
                
                # Scroll para carregar mais conteúdo
                await human_scroll(page, "down")
                await human_delay(1, 2)
                await human_scroll(page, "down")
                await human_delay(1, 2)
                
                html = await page.content()
                props = self._parse_html(html)
                
                if props:
                    logger.info(f"✅ Covers: {len(props)} props extraídos")
                else:
                    logger.warning("⚠️ Covers: Nenhum prop encontrado")
                
        except Exception as e:
            logger.error(f"❌ Covers scraper falhou: {e}")
        
        return props
    
    def _parse_html(self, html: str) -> List[PlayerProp]:
        """Parseia HTML para extrair props."""
        props = []
        soup = BeautifulSoup(html, 'lxml')
        
        # Seletor principal: article.player-prop-article
        articles = soup.select('article.player-prop-article')
        
        if not articles:
            # Fallback: tentar outros seletores
            articles = soup.select('[class*="player-prop"]')
        
        logger.debug(f"Encontrados {len(articles)} artigos de props")
        
        for article in articles:
            try:
                prop = self._parse_article(article)
                if prop:
                    props.append(prop)
            except Exception as e:
                logger.debug(f"Erro ao parsear artigo: {e}")
        
        return props
    
    def _parse_article(self, article) -> Optional[PlayerProp]:
        """
        Parseia um artigo de prop.
        
        Estrutura esperada:
        <article class="player-prop-article">
            <div class="player-headshot-name">
                <a class="player-link"><strong>S. Fontecchio</strong></a>
            </div>
            <div class="other-over-odds">
                8.5
                <div class="player-event">Points Scored</div>
            </div>
            <div class="player-bestOdds-row over">
                <span class="player-bestOdds-label">Over</span>
                <a><span>-108</span></a>
            </div>
        </article>
        """
        try:
            # 1. Extrair nome do jogador
            name_elem = article.select_one('a.player-link strong')
            if not name_elem:
                name_elem = article.select_one('.player-link')
            if not name_elem:
                name_elem = article.select_one('[class*="player-name"]')
            
            if not name_elem:
                return None
            
            player_name = name_elem.get_text(strip=True)
            canonical_name = normalize_player_name(player_name)
            if not canonical_name:
                return None
            
            # 2. Extrair tipo de prop
            prop_type_elem = article.select_one('div.player-event')
            if not prop_type_elem:
                prop_type_elem = article.select_one('[class*="event"]')
            
            prop_type_raw = prop_type_elem.get_text(strip=True) if prop_type_elem else "Points"
            prop_type = self._normalize_prop_type(prop_type_raw)
            
            # 3. Extrair linha (valor numérico)
            line = None
            
            # Método 1: div.other-over-odds contém o número
            odds_container = article.select_one('div.other-over-odds')
            if odds_container:
                # Pegar texto direto, excluindo children
                text = odds_container.get_text(separator=' ', strip=True)
                # Extrair número do texto
                numbers = re.findall(r'(\d+\.?\d*)', text)
                if numbers:
                    line = float(numbers[0])
            
            # Método 2: fallback para outros seletores
            if line is None:
                line_elem = article.select_one('[class*="line"], [class*="total"]')
                if line_elem:
                    text = line_elem.get_text(strip=True)
                    numbers = re.findall(r'(\d+\.?\d*)', text)
                    if numbers:
                        line = float(numbers[0])
            
            if line is None:
                return None
            
            # 4. Extrair odds (Over e Under)
            over_odds = None
            under_odds = None
            
            # Procurar linhas de odds
            over_row = article.select_one('.player-bestOdds-row.over, [class*="over"]')
            under_row = article.select_one('.player-bestOdds-row.under, [class*="under"]')
            
            if over_row:
                odds_elem = over_row.select_one('a span, span[class*="odds"]')
                if odds_elem:
                    over_odds = self._parse_odds(odds_elem.get_text(strip=True))
            
            if under_row:
                odds_elem = under_row.select_one('a span, span[class*="odds"]')
                if odds_elem:
                    under_odds = self._parse_odds(odds_elem.get_text(strip=True))
            
            # Fallback: pegar todas as odds e assumir ordem
            if over_odds is None:
                all_odds = article.select('.player-bestOdds-row a span')
                if len(all_odds) >= 1:
                    over_odds = self._parse_odds(all_odds[0].get_text(strip=True))
                if len(all_odds) >= 2:
                    under_odds = self._parse_odds(all_odds[1].get_text(strip=True))
            
            # CRÍTICO: Não aceitar sem odds reais
            if over_odds is None or under_odds is None:
                # Se só tem over, usar mesmo valor para under (comum em alguns sites)
                if over_odds and not under_odds:
                    under_odds = over_odds
                else:
                    return None
            
            if not (1.01 <= over_odds <= 50.0) or not (1.01 <= under_odds <= 50.0):
                return None
            
            return PlayerProp(
                player_name=canonical_name,
                prop_type=prop_type,
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                source=self.name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.debug(f"Erro ao parsear prop: {e}")
            return None
    
    def _normalize_prop_type(self, raw: str) -> str:
        """Normaliza tipo de prop."""
        raw_lower = raw.lower().strip()
        
        if 'point' in raw_lower or 'pts' in raw_lower or 'scor' in raw_lower:
            return 'points'
        elif 'rebound' in raw_lower or 'reb' in raw_lower:
            return 'rebounds'
        elif 'assist' in raw_lower or 'ast' in raw_lower:
            return 'assists'
        elif 'three' in raw_lower or '3p' in raw_lower or '3-point' in raw_lower:
            return 'threes'
        elif 'steal' in raw_lower or 'stl' in raw_lower:
            return 'steals'
        elif 'block' in raw_lower or 'blk' in raw_lower:
            return 'blocks'
        
        return 'points'  # Default
    
    def _parse_odds(self, text: str) -> Optional[float]:
        """Converte texto de odds para decimal."""
        try:
            text = text.strip()
            
            if not text:
                return None
            
            # Remover caracteres não numéricos exceto +/-
            cleaned = re.sub(r'[^\d\+\-\.]', '', text)
            if not cleaned:
                return None
            
            # American format
            if cleaned.startswith('-') or cleaned.startswith('+'):
                american = int(cleaned)
                if american < 0:
                    return round(1 + (100 / abs(american)), 3)
                return round(1 + (american / 100), 3)
            
            # Decimal format
            value = float(cleaned)
            return value if 1.01 <= value <= 50.0 else None
            
        except (ValueError, ZeroDivisionError):
            return None
    
    async def health_check(self) -> bool:
        if not STEALTH_AVAILABLE:
            return False
        
        try:
            async with create_stealth_browser(headless=True) as (browser, context, page):
                return await navigate_with_retry(page, self.BASE_URL, max_retries=1, timeout=10000)
        except Exception:
            return False
