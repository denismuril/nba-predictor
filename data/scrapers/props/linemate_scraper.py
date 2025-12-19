"""
Linemate Scraper - Interceptação de XHR e DOM para props NBA.

Estratégia: Extrai props do Linemate usando seletores DOM corretos.

v26.3: Implementação inicial.
v26.4: Corrigido URL e seletores (props-entry, text-style-label-semibold).
"""

import logging
import re
from datetime import datetime
from typing import List, Optional, Dict

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


class LinemateScraper(PlayerPropsProvider):
    """
    Scraper para Linemate.io - props com hit rates.
    
    Estrutura do site (atualizada dez/2024):
    - Container: .props-entry ou .trends-page-list-items > div
    - Nome: .text-style-label-semibold
    - Prop/Line: .text-style-caption-medium (ex: "Under 0.5 Triple Double")
    - Odds: .text-style-caption-semibold (ex: "-20000")
    - Team: .text-style-caption-uppercase
    """
    
    BASE_URL = "https://linemate.io/nba/props"
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._captured_data: List[Dict] = []
    
    @property
    def name(self) -> str:
        return "linemate"
    
    @property
    def priority(self) -> int:
        return 2
    
    async def get_props(self, date: str) -> List[PlayerProp]:
        """
        Obtém player props do Linemate.
        
        Args:
            date: Data no formato "YYYY-MM-DD"
            
        Returns:
            Lista de PlayerProp
        """
        if not STEALTH_AVAILABLE or not BS4_AVAILABLE:
            logger.warning("Dependências não disponíveis para Linemate")
            return []
        
        props = []
        
        try:
            async with create_stealth_browser(headless=self.headless) as (browser, context, page):
                logger.info(f"🔍 Linemate: Navegando para {self.BASE_URL}...")
                success = await navigate_with_retry(page, self.BASE_URL)
                
                if not success:
                    logger.warning("❌ Linemate: Falha ao carregar página")
                    return []
                
                # Esperar carregamento dinâmico
                await human_delay(4, 6)
                
                # Scroll para carregar mais conteúdo
                await human_scroll(page, "down")
                await human_delay(1, 2)
                
                html = await page.content()
                props = self._parse_html(html)
                
                if props:
                    logger.info(f"✅ Linemate: {len(props)} props extraídos")
                else:
                    logger.warning("⚠️ Linemate: Nenhum prop encontrado")
                
        except Exception as e:
            logger.error(f"❌ Linemate scraper falhou: {e}")
        
        return props
    
    def _parse_html(self, html: str) -> List[PlayerProp]:
        """Parseia HTML para extrair props."""
        props = []
        soup = BeautifulSoup(html, 'lxml')
        
        # Seletor principal: .props-entry
        entries = soup.select('.props-entry')
        
        if not entries:
            # Fallback: tentar container de lista
            entries = soup.select('.trends-page-list-items > div')
        
        if not entries:
            # Fallback 2: hit-rate-entry
            entries = soup.select('.hit-rate-entry')
        
        logger.debug(f"Encontrados {len(entries)} entries de props")
        
        for entry in entries:
            try:
                prop = self._parse_entry(entry)
                if prop:
                    props.append(prop)
            except Exception as e:
                logger.debug(f"Erro ao parsear entry: {e}")
        
        return props
    
    def _parse_entry(self, entry) -> Optional[PlayerProp]:
        """
        Parseia uma entrada de prop.
        
        Estrutura esperada:
        <div class="props-entry">
            <div class="hit-rate-entry-details">
                <p class="text-style-label-semibold">D. Daniels</p>
                <p class="text-style-caption-medium">Under 0.5 Triple Double</p>
                <p class="text-style-caption-semibold">-20000</p>
            </div>
        </div>
        """
        try:
            # 1. Extrair nome do jogador
            name_elem = entry.select_one('.text-style-label-semibold')
            if not name_elem:
                name_elem = entry.select_one('[class*="label-semibold"]')
            
            if not name_elem:
                return None
            
            player_name = name_elem.get_text(strip=True)
            canonical_name = normalize_player_name(player_name)
            if not canonical_name:
                return None
            
            # 2. Extrair prop type e line
            prop_elem = entry.select_one('.text-style-caption-medium')
            if not prop_elem:
                prop_elem = entry.select_one('[class*="caption-medium"]')
            
            if not prop_elem:
                return None
            
            prop_text = prop_elem.get_text(strip=True)
            # Parse "Under 0.5 Triple Double" ou "Over 25.5 Points"
            prop_type, line = self._parse_prop_text(prop_text)
            
            if prop_type is None or line is None:
                return None
            
            # 3. Extrair odds
            odds_elem = entry.select_one('.text-style-caption-semibold')
            if not odds_elem:
                odds_elem = entry.select_one('[class*="caption-semibold"]')
            
            if not odds_elem:
                return None
            
            odds_text = odds_elem.get_text(strip=True)
            odds = self._parse_odds(odds_text)
            
            if odds is None:
                return None
            
            # Linemate geralmente mostra apenas um lado (Over ou Under)
            # Usamos a mesma odd para ambos como aproximação
            over_odds = odds
            under_odds = odds
            
            if not (1.01 <= over_odds <= 50.0):
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
            logger.debug(f"Erro ao parsear entry: {e}")
            return None
    
    def _parse_prop_text(self, text: str) -> tuple:
        """
        Parseia texto do prop (ex: "Under 0.5 Triple Double").
        
        Returns:
            Tupla (prop_type, line) ou (None, None)
        """
        text = text.strip()
        
        # Pattern: "Over/Under X.X PropType"
        match = re.match(r'(Over|Under)\s+([\d.]+)\s+(.+)', text, re.IGNORECASE)
        if match:
            line = float(match.group(2))
            raw_type = match.group(3).lower()
            
            # Normalizar tipo
            if 'point' in raw_type or 'pts' in raw_type:
                return ('points', line)
            elif 'rebound' in raw_type or 'reb' in raw_type:
                return ('rebounds', line)
            elif 'assist' in raw_type:
                return ('assists', line)
            elif 'three' in raw_type or '3p' in raw_type:
                return ('threes', line)
            elif 'steal' in raw_type:
                return ('steals', line)
            elif 'block' in raw_type:
                return ('blocks', line)
            elif 'triple' in raw_type:
                return ('triple_double', line)
            elif 'double' in raw_type:
                return ('double_double', line)
            else:
                return ('other', line)
        
        return (None, None)
    
    def _parse_odds(self, text: str) -> Optional[float]:
        """Converte texto de odds para decimal."""
        try:
            text = text.strip()
            
            if not text:
                return None
            
            # Remover caracteres não numéricos exceto +/-
            cleaned = re.sub(r'[^\d\+\-]', '', text)
            if not cleaned:
                return None
            
            # American format
            american = int(cleaned)
            if american < 0:
                return round(1 + (100 / abs(american)), 3)
            return round(1 + (american / 100), 3)
            
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
