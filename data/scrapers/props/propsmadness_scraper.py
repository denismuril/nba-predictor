"""
PropsMadness Scraper - Props aggregator.

Source: https://www.propsmadness.com/
Strategy: Stealth HTML parsing for props data.

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


class PropsMadnessScraper(PlayerPropsProvider):
    """
    Scraper for PropsMadness.com - Props aggregator.
    
    Extracts NBA player props from the public interface.
    """
    
    BASE_URL = "https://www.propsmadness.com/"
    
    def __init__(self, headless: bool = True):
        self.headless = headless
    
    @property
    def name(self) -> str:
        return "propsmadness"
    
    @property
    def priority(self) -> int:
        return 7
    
    async def get_props(self, date: str) -> List[PlayerProp]:
        """
        Fetches props from PropsMadness.
        
        Args:
            date: Date in "YYYY-MM-DD" format
            
        Returns:
            List of PlayerProp objects
        """
        if not STEALTH_AVAILABLE or not BS4_AVAILABLE:
            logger.warning("Dependencies not available for PropsMadness")
            return []
        
        props = []
        
        try:
            async with create_stealth_browser(headless=self.headless) as (browser, context, page):
                logger.info(f"🔍 PropsMadness: Navigating to {self.BASE_URL}")
                success = await navigate_with_retry(page, self.BASE_URL)
                
                if not success:
                    logger.warning("❌ PropsMadness: Failed to load page")
                    return []
                
                await human_delay(3, 5)
                await human_scroll(page, "down")
                await human_delay(2, 3)
                
                html = await page.content()
                props = self._parse_html(html)
                
                if props:
                    logger.info(f"✅ PropsMadness: {len(props)} props found")
                else:
                    logger.warning("⚠️ PropsMadness: No props found")
                    
        except Exception as e:
            logger.error(f"❌ PropsMadness scraper failed: {e}")
        
        return props
    
    def _parse_html(self, html: str) -> List[PlayerProp]:
        """Parse HTML to extract props."""
        props = []
        soup = BeautifulSoup(html, 'lxml')
        
        # Common selectors for props tables
        selectors = [
            '.prop-row',
            '.player-prop',
            'tr[data-player]',
            '.props-table tbody tr',
            '[class*="prop"]'
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
            # Try common name selectors
            name_elem = row.select_one('.player-name, .name, [class*="player"]')
            if not name_elem:
                return None
            
            player_name = name_elem.get_text(strip=True)
            canonical_name = normalize_player_name(player_name) if NORMALIZER_AVAILABLE else player_name
            if not canonical_name:
                return None
            
            # Try to extract line
            line_elem = row.select_one('.line, .prop-line, [class*="line"]')
            if not line_elem:
                return None
            
            line_text = line_elem.get_text(strip=True)
            numbers = re.findall(r'(\d+\.?\d*)', line_text)
            if not numbers:
                return None
            
            line = float(numbers[0])
            
            # Try to extract odds
            odds_elem = row.select_one('.odds, [class*="odds"]')
            over_odds = 1.91
            under_odds = 1.91
            
            if odds_elem:
                odds_text = odds_elem.get_text(strip=True)
                odds_val = self._parse_odds(odds_text)
                if odds_val:
                    over_odds = under_odds = odds_val
            
            return PlayerProp(
                player_name=canonical_name,
                prop_type="points",
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                source=self.name,
                timestamp=datetime.now(),
                bookmaker="propsmadness"
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
        return ["points", "rebounds", "assists", "threes"]
