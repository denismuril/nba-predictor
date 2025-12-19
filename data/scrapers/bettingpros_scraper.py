"""
BettingPros Scraper - Consenso de mercado para props NBA.

Estratégia: Extrai o consenso de múltiplas casas de apostas.

v26.3: Implementação inicial.
v26.4: Corrigido seletores (tr.table-row, .player-info__name, .pick-button).
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


class BettingProsScraper(PlayerPropsProvider):
    """
    Scraper para BettingPros.com - consenso de mercado.
    
    Estrutura do site (atualizada dez/2024):
    - Container: tr.table-row dentro de .props-table
    - Nome: .player-info__name
    - Prop/Line: td.table-cell--prop span.typography
    - Odds: .pick-button .cost (ex: "(-130)")
    """
    
    BASE_URL = "https://www.bettingpros.com/nba/picks/prop-bets/"
    
    def __init__(self, headless: bool = True):
        self.headless = headless
    
    @property
    def name(self) -> str:
        return "bettingpros"
    
    @property
    def priority(self) -> int:
        return 3
    
    async def get_props(self, date: str) -> List[PlayerProp]:
        """
        Obtém props de todas as categorias.
        
        Args:
            date: Data no formato "YYYY-MM-DD"
            
        Returns:
            Lista de PlayerProp
        """
        if not STEALTH_AVAILABLE or not BS4_AVAILABLE:
            logger.warning("Dependências não disponíveis para BettingPros")
            return []
        
        props = []
        
        try:
            async with create_stealth_browser(headless=self.headless) as (browser, context, page):
                logger.info(f"🔍 BettingPros: Navegando para props...")
                success = await navigate_with_retry(page, self.BASE_URL)
                
                if not success:
                    logger.warning("❌ BettingPros: Falha ao carregar página")
                    return []
                
                # Esperar carregamento dinâmico (Vue.js)
                await human_delay(4, 6)
                
                # Tentar fechar modais de cookies se existirem
                try:
                    await page.click('button:has-text("Accept")', timeout=2000)
                except Exception:
                    pass
                
                # Scroll para carregar tabela
                await human_scroll(page, "down")
                await human_delay(2, 3)
                
                html = await page.content()
                props = self._parse_html(html)
                
                if props:
                    logger.info(f"✅ BettingPros: {len(props)} props extraídos")
                else:
                    logger.warning("⚠️ BettingPros: Nenhum prop encontrado")
                
        except Exception as e:
            logger.error(f"❌ BettingPros scraper falhou: {e}")
        
        return props
    
    def _parse_html(self, html: str) -> List[PlayerProp]:
        """Parseia HTML para extrair props."""
        props = []
        soup = BeautifulSoup(html, 'lxml')
        
        # Seletor principal: tr.table-row dentro da tabela de props
        rows = soup.select('.props-table tr.table-row')
        
        if not rows:
            # Fallback: qualquer table-row
            rows = soup.select('tr.table-row')
        
        if not rows:
            # Fallback 2: tbody tr
            rows = soup.select('.props-table tbody tr')
        
        logger.debug(f"Encontradas {len(rows)} rows de props")
        
        for row in rows:
            try:
                prop = self._parse_row(row)
                if prop:
                    props.append(prop)
            except Exception as e:
                logger.debug(f"Erro ao parsear row: {e}")
        
        return props
    
    def _parse_row(self, row) -> Optional[PlayerProp]:
        """
        Parseia uma linha da tabela.
        
        Estrutura esperada:
        <tr class="table-row">
            <td class="table-cell--player">
                <a class="player-info__name">Jordan Goodwin</a>
            </td>
            <td class="table-cell--prop">
                <span>17.5</span>
                <span>Pts + Ast + Reb</span>
            </td>
            <td class="table-cell--pick">
                <button class="pick-button">
                    <span class="line">U 17.5</span>
                    <span class="cost">(-130)</span>
                </button>
            </td>
        </tr>
        """
        try:
            # 1. Extrair nome do jogador
            name_elem = row.select_one('.player-info__name')
            if not name_elem:
                name_elem = row.select_one('[class*="player-name"]')
            if not name_elem:
                name_elem = row.select_one('td:first-child a')
            
            if not name_elem:
                return None
            
            player_name = name_elem.get_text(strip=True)
            canonical_name = normalize_player_name(player_name)
            if not canonical_name:
                return None
            
            # 2. Extrair prop type e line
            prop_cell = row.select_one('.table-cell--prop, td.table-cell--prop')
            if not prop_cell:
                # Fallback: segunda célula
                cells = row.select('td')
                prop_cell = cells[1] if len(cells) > 1 else None
            
            if not prop_cell:
                return None
            
            spans = prop_cell.select('span.typography, span')
            if len(spans) >= 2:
                line_text = spans[0].get_text(strip=True)
                type_text = spans[1].get_text(strip=True)
            else:
                # Tentar extrair do texto completo
                text = prop_cell.get_text(strip=True)
                numbers = re.findall(r'(\d+\.?\d*)', text)
                line_text = numbers[0] if numbers else None
                type_text = re.sub(r'[\d.]+', '', text).strip()
            
            if not line_text:
                return None
            
            try:
                line = float(line_text)
            except ValueError:
                return None
            
            prop_type = self._normalize_prop_type(type_text)
            
            # 3. Extrair odds
            pick_button = row.select_one('.pick-button')
            odds = None
            
            if pick_button:
                cost_elem = pick_button.select_one('.cost')
                if cost_elem:
                    odds_text = cost_elem.get_text(strip=True)
                    # Remove parênteses: "(-130)" -> "-130"
                    odds_text = odds_text.replace('(', '').replace(')', '')
                    odds = self._parse_odds(odds_text)
            
            if odds is None:
                # Fallback: procurar qualquer odds
                odds_elems = row.select('[class*="odds"], [class*="cost"]')
                for elem in odds_elems:
                    text = elem.get_text(strip=True).replace('(', '').replace(')', '')
                    odds = self._parse_odds(text)
                    if odds:
                        break
            
            if odds is None:
                return None
            
            # BettingPros mostra um lado, usar mesma odd para ambos
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
                timestamp=datetime.now(),
                bookmaker="consensus"
            )
            
        except Exception as e:
            logger.debug(f"Erro ao parsear row: {e}")
            return None
    
    def _normalize_prop_type(self, raw: str) -> str:
        """Normaliza tipo de prop."""
        raw_lower = raw.lower().strip()
        
        if 'point' in raw_lower or 'pts' in raw_lower:
            return 'points'
        elif 'rebound' in raw_lower or 'reb' in raw_lower:
            return 'rebounds'
        elif 'assist' in raw_lower or 'ast' in raw_lower:
            return 'assists'
        elif 'three' in raw_lower or '3p' in raw_lower:
            return 'threes'
        elif 'steal' in raw_lower or 'stl' in raw_lower:
            return 'steals'
        elif 'block' in raw_lower or 'blk' in raw_lower:
            return 'blocks'
        elif 'pts + ast + reb' in raw_lower or 'par' in raw_lower:
            return 'pts_ast_reb'
        
        return 'points'  # Default
    
    def _parse_odds(self, text: str) -> Optional[float]:
        """Converte texto de odds para decimal."""
        try:
            text = text.strip()
            
            if not text:
                return None
            
            # Remover caracteres inválidos
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
