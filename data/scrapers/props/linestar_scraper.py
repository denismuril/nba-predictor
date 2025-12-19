"""
LineStar Scraper - DraftKings NBA Props.

Source: https://www.linestarapp.com/Props/Sport/NBA/Site/DraftKings
Strategy: Stealth HTML parsing for DraftKings props data.

v26.4: Initial implementation.
"""

import logging
import re
from datetime import datetime
from typing import List, Optional

from data.interfaces.player_props_provider import PlayerPropsProvider, PlayerProp

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

try:
    from data.scrapers.player_name_normalizer import normalize_player_name
    NORMALIZER_AVAILABLE = True
except ImportError:
    NORMALIZER_AVAILABLE = False
    def normalize_player_name(name, *args, **kwargs):
        return name

logger = logging.getLogger(__name__)


class LineStarScraper(PlayerPropsProvider):
    """
    Scraper for LineStar App - DraftKings NBA Props.
    
    Extracts NBA player props from DraftKings via LineStar interface.
    """
    
    BASE_URL = "https://www.linestarapp.com/Props/Sport/NBA/Site/DraftKings"
    
    def __init__(self, headless: bool = True):
        self.headless = headless
    
    @property
    def name(self) -> str:
        return "linestar"
    
    @property
    def priority(self) -> int:
        return 9
    
    async def get_props(self, date: str) -> List[PlayerProp]:
        """
        Fetches props from LineStar.
        
        Args:
            date: Date in "YYYY-MM-DD" format
            
        Returns:
            List of PlayerProp objects
        """
        if not STEALTH_AVAILABLE or not BS4_AVAILABLE:
            logger.warning("Dependencies not available for LineStar")
            return []
        
        props = []
        
        try:
            async with create_stealth_browser(headless=self.headless) as (browser, context, page):
                logger.info(f"🔍 LineStar: Navigating to {self.BASE_URL}")
                success = await navigate_with_retry(page, self.BASE_URL)
                
                if not success:
                    logger.warning("❌ LineStar: Failed to load page")
                    return []
                
                await human_delay(3, 5)
                
                # LineStar may have JavaScript-rendered content
                await human_scroll(page, "down")
                await human_delay(2, 3)
                await human_scroll(page, "down")
                await human_delay(1, 2)
                
                html = await page.content()
                props = self._parse_html(html)
                
                if props:
                    logger.info(f"✅ LineStar: {len(props)} props found")
                else:
                    logger.warning("⚠️ LineStar: No props found")
                    
        except Exception as e:
            logger.error(f"❌ LineStar scraper failed: {e}")
        
        return props
    
    def _parse_html(self, html: str) -> List[PlayerProp]:
        """Parse HTML to extract props."""
        props = []
        soup = BeautifulSoup(html, 'lxml')
        
        # LineStar typically uses table structure
        selectors = [
            'table.props-table tbody tr',
            '.player-row',
            '.prop-row',
            '[class*="player"][class*="prop"]',
            'tr[data-player-id]'
        ]
        
        rows = []
        for selector in selectors:
            rows = soup.select(selector)
            if rows:
                break
        
        logger.debug(f"Found {len(rows)} prop rows")
        
        for row in rows:
            try:
                prop = self._parse_row(row)
                if prop:
                    props.append(prop)
            except Exception as e:
                logger.debug(f"Error parsing row: {e}")
        
        return props
    
    def _parse_row(self, row) -> Optional[PlayerProp]:
        """Parse a single prop row."""
        try:
            # Extract player name
            name_selectors = ['.player-name', 'td:first-child', '.name', '[class*="player"]']
            name_elem = None
            for sel in name_selectors:
                name_elem = row.select_one(sel)
                if name_elem:
                    break
            
            if not name_elem:
                return None
            
            player_name = name_elem.get_text(strip=True)
            
            # Skip if it looks like a header
            if not player_name or player_name.lower() in ['player', 'name', 'team']:
                return None
            
            canonical_name = normalize_player_name(player_name) if NORMALIZER_AVAILABLE else player_name
            if not canonical_name:
                return None
            
            # Extract line and prop type
            text = row.get_text()
            numbers = re.findall(r'(\d+\.?\d*)', text)
            if not numbers:
                return None
            
            line = float(numbers[0])
            
            # Determine prop type
            text_lower = text.lower()
            if 'point' in text_lower or 'pts' in text_lower:
                prop_type = 'points'
            elif 'rebound' in text_lower or 'reb' in text_lower:
                prop_type = 'rebounds'
            elif 'assist' in text_lower or 'ast' in text_lower:
                prop_type = 'assists'
            elif 'three' in text_lower or '3pt' in text_lower or '3-pt' in text_lower:
                prop_type = 'threes'
            elif 'steal' in text_lower:
                prop_type = 'steals'
            elif 'block' in text_lower:
                prop_type = 'blocks'
            else:
                prop_type = 'points'
            
            # Extract odds
            over_odds = 1.91
            under_odds = 1.91
            
            odds_elems = row.select('[class*="odds"], [class*="line"]')
            for elem in odds_elems:
                odds_text = elem.get_text(strip=True)
                odds_val = self._parse_odds(odds_text)
                if odds_val:
                    over_odds = under_odds = odds_val
                    break
            
            return PlayerProp(
                player_name=canonical_name,
                prop_type=prop_type,
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                source=self.name,
                timestamp=datetime.now(),
                bookmaker="draftkings"
            )
            
        except Exception as e:
            logger.debug(f"Error parsing row: {e}")
            return None
    
    def _parse_odds(self, text: str) -> Optional[float]:
        """Convert American odds to decimal."""
        try:
            cleaned = re.sub(r'[^\d\+\-]', '', text)
            if not cleaned:
                return None
            
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
    
    def get_supported_prop_types(self) -> List[str]:
        return ["points", "rebounds", "assists", "threes", "steals", "blocks"]
