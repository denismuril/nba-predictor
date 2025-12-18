"""
Action Network Player Props Scraper for NBA Predictor.

This module scrapes player prop bets (Points, Rebounds, Assists) from Action Network
or ScoresAndOdds using Playwright with stealth techniques to avoid detection.

Features:
- Playwright-based scraping with anti-detection (playwright-stealth)
- Network request interception for efficient data extraction
- Player name normalization using player_name_normalizer
- Structured error logging to logs/scraping_errors.log

v26.2: Initial implementation for Player Props integration.
"""

import logging
import asyncio
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass
from playwright.async_api import async_playwright, Page, Response
import json

from data.scrapers.player_name_normalizer import normalize_player_name

logger = logging.getLogger(__name__)


@dataclass
class PlayerProp:
    """
    Represents a single player prop bet.
    
    Attributes:
        player_name: Canonical player name (normalized)
        prop_type: Type of prop ('points', 'rebounds', 'assists', 'threes')
        line: Prop line value (e.g., 25.5 points)
        over_odds: Decimal odds for over
        under_odds: Decimal odds for under
        source: Source of the prop ('action_network', 'scoresandodds')
        timestamp: When the prop was scraped
        bookmaker: Bookmaker name (e.g., 'DraftKings', 'FanDuel')
    """
    player_name: str
    prop_type: str
    line: float
    over_odds: float
    under_odds: float
    source: str
    timestamp: datetime
    bookmaker: Optional[str] = None


class ActionNetworkScraper:
    """
    Scraper for Action Network player props using Playwright with stealth.
    
    Uses network interception to capture API responses for faster, more reliable extraction.
    """
    
    BASE_URL = "https://www.actionnetwork.com/nba/props"
    
    def __init__(self, headless: bool = True):
        """
        Initialize the scraper.
        
        Args:
            headless: Run browser in headless mode
        """
        self.headless = headless
        self._intercepted_data: List[Dict] = []
    
    async def _intercept_response(self, response: Response):
        """
        Intercept network responses to capture player props data.
        
        Args:
            response: Playwright Response object
        """
        try:
            # Look for API endpoints that contain player props data
            if "api" in response.url and "props" in response.url.lower():
                if response.status == 200:
                    try:
                        data = await response.json()
                        self._intercepted_data.append(data)
                        logger.debug(f"✅ Intercepted data from {response.url}")
                    except Exception as e:
                        logger.debug(f"Could not parse JSON from {response.url}: {e}")
        except Exception as e:
            logger.debug(f"Error intercepting response: {e}")
    
    async def _extract_props_from_page(self, page: Page) -> List[PlayerProp]:
        """
        Extract player props from the loaded page.
        
        Args:
            page: Playwright Page object
            
        Returns:
            List of PlayerProp objects
        """
        props = []
        
        # First, try to extract from intercepted API data
        if self._intercepted_data:
            logger.info(f"📊 Processing {len(self._intercepted_data)} intercepted API responses")
            
            for data in self._intercepted_data:
                extracted = self._parse_api_response(data)
                props.extend(extracted)
        
        # Fallback: Parse HTML if API interception didn't work
        if not props:
            logger.warning("⚠️ No data from API interception, falling back to HTML parsing")
            props = await self._parse_html(page)
        
        return props
    
    def _parse_api_response(self, data: Dict) -> List[PlayerProp]:
        """
        Parse Action Network API response to extract player props.
        
        API Structure (REAL from inspection):
        - playerProps: Array of prop objects
        - players: Dict mapping player_id to player metadata
        - games: Dict with game information
        
        Args:
            data: JSON response data from api.actionnetwork.com
            
        Returns:
            List of PlayerProp objects
        """
        props = []
        
        try:
            # Extract player mapping (id -> name)
            players_map = {}
            if "players" in data:
                for player_id, player_data in data["players"].items():
                    full_name = player_data.get("full_name")
                    if full_name:
                        players_map[int(player_id)] = full_name
            
            # Process playerProps array
            if "playerProps" not in data:
                logger.warning("No 'playerProps' key in API response")
                return props
            
            for prop_data in data["playerProps"]:
                try:
                    player_id = prop_data.get("player_id")
                    if not player_id or player_id not in players_map:
                        continue
                    
                    raw_name = players_map[player_id]
                    
                    # Normalize player name
                    canonical_name = normalize_player_name(raw_name)
                    if not canonical_name:
                        logger.warning(f"⚠️ Could not normalize player name: {raw_name}")
                        continue
                    
                    # Prop type (e.g., "Pts", "Rebs", "Ast")
                    prop_type_display = prop_data.get("custom_pick_type_display_name", "")
                    prop_type = self._normalize_prop_type(prop_type_display)
                    
                    if not prop_type:
                        continue
                    
                    # Extract lines and odds
                    lines_data = prop_data.get("lines", [])
                    if not lines_data:
                        continue
                    
                    # Group by line value to match over/under pairs
                    line_groups = {}
                    for line_obj in lines_data:
                        value = line_obj.get("value")
                        side = line_obj.get("side", "").lower()
                        odds_american = line_obj.get("odds")
                        
                        if value is None or not side or odds_american is None:
                            continue
                        
                        if value not in line_groups:
                            line_groups[value] = {}
                        
                        # Convert American odds to decimal
                        decimal_odds = self._american_to_decimal(odds_american)
                        line_groups[value][side] = decimal_odds
                    
                    # Create props for each complete over/under pair
                    for line_value, sides in line_groups.items():
                        if "over" in sides and "under" in sides:
                            prop = PlayerProp(
                                player_name=canonical_name,
                                prop_type=prop_type,
                                line=float(line_value),
                                over_odds=sides["over"],
                                under_odds=sides["under"],
                                source="action_network",
                                timestamp=datetime.now(),
                                bookmaker="action_network"
                            )
                            props.append(prop)
                            
                except Exception as e:
                    logger.debug(f"Error parsing individual prop: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error parsing API response: {e}")
        
        return props
    
    def _normalize_prop_type(self, raw_type: str) -> Optional[str]:
        """
        Normalize prop type display name to standard format.
        
        Args:
            raw_type: Raw prop type from API (e.g., "Pts", "Rebs", "Ast")
            
        Returns:
            Normalized prop type or None
        """
        mapping = {
            "pts": "points",
            "points": "points",
            "rebs": "rebounds",
            "rebounds": "rebounds",
            "ast": "assists",
            "assists": "assists",
            "stl": "steals",
            "steals": "steals",
            "blk": "blocks",
            "blocks": "blocks",
            "3pm": "threes",
            "threes": "threes",
        }
        
        normalized = raw_type.lower().strip()
        return mapping.get(normalized)
    
    def _american_to_decimal(self, american_odds: int) -> float:
        """
        Convert American odds to decimal odds.
        
        Args:
            american_odds: Odds in American format (e.g., -110, +150)
            
        Returns:
            Decimal odds (e.g., 1.91, 2.50)
        """
        if american_odds < 0:
            return round(1 + (100 / abs(american_odds)), 3)
        else:
            return round(1 + (american_odds / 100), 3)
    
    async def _parse_html(self, page: Page) -> List[PlayerProp]:
        """
        Fallback HTML parser for player props.
        
        Args:
            page: Playwright Page object
            
        Returns:
            List of PlayerProp objects
        """
        props = []
        
        try:
            # Wait for props to load
            await page.wait_for_selector(".prop-row, .player-prop, [data-testid='prop-card']", timeout=10000)
            
            # Extract props from HTML
            # NOTE: This is a placeholder - actual selectors depend on Action Network's HTML structure
            prop_elements = await page.query_selector_all(".prop-row, .player-prop")
            
            for element in prop_elements:
                try:
                    raw_name = await element.query_selector(".player-name")
                    prop_type_elem = await element.query_selector(".prop-type")
                    line_elem = await element.query_selector(".line")
                    over_elem = await element.query_selector(".over-odds")
                    under_elem = await element.query_selector(".under-odds")
                    
                    if not all([raw_name, prop_type_elem, line_elem, over_elem, under_elem]):
                        continue
                    
                    raw_name_text = await raw_name.text_content()
                    prop_type = await prop_type_elem.text_content()
                    line = await line_elem.text_content()
                    over_odds = await over_elem.text_content()
                    under_odds = await under_elem.text_content()
                    
                    # Normalize and create prop
                    canonical_name = normalize_player_name(raw_name_text.strip())
                    if not canonical_name:
                        continue
                    
                    prop = PlayerProp(
                        player_name=canonical_name,
                        prop_type=prop_type.lower().strip(),
                        line=float(line.replace("+", "").strip()),
                        over_odds=float(over_odds),
                        under_odds=float(under_odds),
                        source="action_network",
                        timestamp=datetime.now(),
                    )
                    props.append(prop)
                    
                except Exception as e:
                    logger.debug(f"Error parsing prop element: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ HTML parsing failed: {e}")
        
        return props
    
    async def fetch_props(self, date: Optional[str] = None) -> List[PlayerProp]:
        """
        Fetch player props from Action Network.
        
        Args:
            date: Optional date string (YYYY-MM-DD). If None, uses today.
            
        Returns:
            List of PlayerProp objects
        """
        self._intercepted_data = []
        
        async with async_playwright() as p:
            try:
                # Launch browser with stealth
                browser = await p.chromium.launch(headless=self.headless)
                
                # Create context with stealth settings
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                
                # TODO: Add playwright-stealth injection here
                # await context.add_init_script(path="path/to/stealth.min.js")
                
                page = await context.new_page()
                
                # Set up response interception
                page.on("response", self._intercept_response)
                
                logger.info(f"🌐 Navigating to {self.BASE_URL}")
                await page.goto(self.BASE_URL, wait_until="networkidle", timeout=30000)
                
                # Wait for content to load
                await asyncio.sleep(3)
                
                # Extract props
                props = await self._extract_props_from_page(page)
                
                await browser.close()
                
                logger.info(f"✅ Scraped {len(props)} player props from Action Network")
                return props
                
            except Exception as e:
                logger.error(f"❌ Action Network scraping failed: {e}")
                raise
    
    def fetch_props_sync(self, date: Optional[str] = None) -> List[PlayerProp]:
        """
        Synchronous wrapper for fetch_props.
        
        Args:
            date: Optional date string (YYYY-MM-DD)
            
        Returns:
            List of PlayerProp objects
        """
        return asyncio.run(self.fetch_props(date))


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scraper = ActionNetworkScraper(headless=True)
    props = scraper.fetch_props_sync()
    
    for prop in props[:5]:  # Print first 5
        print(f"{prop.player_name} - {prop.prop_type}: {prop.line} (O: {prop.over_odds}, U: {prop.under_odds})")
