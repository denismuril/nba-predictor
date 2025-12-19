"""
Props.com Scraper - Best NBA Player Props Today.

Source: https://props.com/best-nba-player-props-today/
Strategy: Stealth HTML parsing for daily best props.

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


class PropsComScraper(PlayerPropsProvider):
    """
    Scraper for Props.com - Best NBA Player Props Today.
    
    Extracts curated best props for the day.
    """
    
    BASE_URL = "https://props.com/best-nba-player-props-today/"
    
    def __init__(self, headless: bool = True):
        self.headless = headless
    
    @property
    def name(self) -> str:
        return "propscom"
    
    @property
    def priority(self) -> int:
        return 8
    
    async def get_props(self, date: str) -> List[PlayerProp]:
        """
        Fetches props from Props.com.
        
        Args:
            date: Date in "YYYY-MM-DD" format
            
        Returns:
            List of PlayerProp objects
        """
        if not STEALTH_AVAILABLE or not BS4_AVAILABLE:
            logger.warning("Dependencies not available for Props.com")
            return []
        
        props = []
        
        try:
            async with create_stealth_browser(headless=self.headless) as (browser, context, page):
                logger.info(f"🔍 Props.com: Navigating to {self.BASE_URL}")
                success = await navigate_with_retry(page, self.BASE_URL)
                
                if not success:
                    logger.warning("❌ Props.com: Failed to load page")
                    return []
                
                await human_delay(3, 5)
                await human_scroll(page, "down")
                await human_delay(2, 3)
                
                html = await page.content()
                props = self._parse_html(html)
                
                if props:
                    logger.info(f"✅ Props.com: {len(props)} props found")
                else:
                    logger.warning("⚠️ Props.com: No props found")
                    
        except Exception as e:
            logger.error(f"❌ Props.com scraper failed: {e}")
        
        return props
    
    def _parse_html(self, html: str) -> List[PlayerProp]:
        """Parse HTML to extract props."""
        props = []
        soup = BeautifulSoup(html, 'lxml')
        
        # Look for prop cards or table rows
        selectors = [
            '.prop-card',
            '.player-prop-card',
            'article.prop',
            '.props-list li',
            '[class*="prop-item"]'
        ]
        
        rows = []
        for selector in selectors:
            rows = soup.select(selector)
            if rows:
                break
        
        logger.debug(f"Found {len(rows)} prop items")
        
        for row in rows:
            try:
                prop = self._parse_item(row)
                if prop:
                    props.append(prop)
            except Exception as e:
                logger.debug(f"Error parsing item: {e}")
        
        return props
    
    def _parse_item(self, item) -> Optional[PlayerProp]:
        """Parse a single prop item."""
        try:
            # Extract player name
            name_selectors = ['.player-name', 'h3', 'h4', '.name', '[class*="player"]']
            name_elem = None
            for sel in name_selectors:
                name_elem = item.select_one(sel)
                if name_elem:
                    break
            
            if not name_elem:
                return None
            
            player_name = name_elem.get_text(strip=True)
            canonical_name = normalize_player_name(player_name) if NORMALIZER_AVAILABLE else player_name
            if not canonical_name:
                return None
            
            # Extract prop type and line from text
            text = item.get_text(strip=True)
            
            # Try to find line number
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
            elif 'three' in text_lower or '3pt' in text_lower:
                prop_type = 'threes'
            else:
                prop_type = 'points'
            
            # Default odds
            over_odds = 1.91
            under_odds = 1.91
            
            # Try to extract odds
            odds_match = re.search(r'([+-]\d+)', text)
            if odds_match:
                odds_val = self._parse_odds(odds_match.group(1))
                if odds_val:
                    over_odds = under_odds = odds_val
            
            return PlayerProp(
                player_name=canonical_name,
                prop_type=prop_type,
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                source=self.name,
                timestamp=datetime.now(),
                bookmaker="propscom"
            )
            
        except Exception as e:
            logger.debug(f"Error parsing item: {e}")
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
