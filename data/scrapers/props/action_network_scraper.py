"""
Action Network Player Props Scraper for NBA Predictor.

This module fetches player prop bets (Points, Rebounds, Assists) from Action Network
using direct HTTP requests to their API.

Features:
- Direct API access (faster and more reliable than browser scraping)
- Player name normalization using player_name_normalizer
- Structured error logging

v26.2: Initial implementation for Player Props integration.
v26.3: Simplified from Playwright to direct HTTP requests.
"""

import logging
import asyncio
import requests
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

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
    Scraper for Action Network player props using direct HTTP API calls.
    
    Uses requests library instead of browser automation for faster,
    more reliable extraction.
    
    API: https://api.actionnetwork.com/web/v2/leagues/4/projections/available
    """
    
    API_URL = "https://api.actionnetwork.com/web/v2/leagues/4/projections/available"
    
    def __init__(self, headless: bool = True):
        """
        Initialize the scraper.
        
        Args:
            headless: Ignored (kept for backwards compatibility)
        """
        pass  # No initialization needed for requests-based approach
    
    
    def _parse_api_response(self, data: Dict) -> List[PlayerProp]:
        """
        Parse Action Network API response to extract player props.
        
        API Structure from /projections/available:
        - playerProps: Array of prop objects  
        - players: Dict mapping player_id to player metadata
        - games: Dict with game information
        
        Args:
            data: JSON response from /projections/available
            
        Returns:
            List of PlayerProp objects
        """
        props = []
        
        try:
            # DEBUG: Log structure
            logger.debug(f"🔍 API Response keys: {list(data.keys())}")
            
            # Extract player mapping (id -> name)
            # Note: players can be list or dict depending on API version
            players_map = {}
            if "players" in data:
                players_data = data["players"]
                if isinstance(players_data, dict):
                    # Dict format: {player_id: player_data}
                    for player_id, player_info in players_data.items():
                        full_name = player_info.get("full_name")
                        if full_name:
                            players_map[int(player_id)] = full_name
                elif isinstance(players_data, list):
                    # List format: [{id: ..., full_name: ...}, ...]
                    for player_info in players_data:
                        player_id = player_info.get("id")
                        full_name = player_info.get("full_name") or player_info.get("name")
                        if player_id and full_name:
                            players_map[player_id] = full_name
            
            logger.debug(f"📊 Found {len(players_map)} players")
            
            # Process playerProps array (NOT markets!)
            if "playerProps" not in data:
                logger.warning("No 'playerProps' key in API response")
                logger.debug(f"Available keys: {list(data.keys())}")
                return props
            
            player_props = data["playerProps"]
            logger.debug(f"🎯 Found {len(player_props)} playerProps")
            
            # Debug: show first prop structure
            if player_props:
                first_prop = player_props[0]
                logger.debug(f"📝 First prop keys: {list(first_prop.keys())}")
                logger.debug(f"📝 First prop sample: {first_prop}")
            
            for prop_data in player_props:
                try:
                    # Get player name
                    player_id = prop_data.get("player_id")
                    if not player_id or player_id not in players_map:
                        continue
                    
                    raw_name = players_map[player_id]
                    
                    # Normalize player name
                    canonical_name = normalize_player_name(raw_name)
                    if not canonical_name:
                        logger.debug(f"⚠️ Could not normalize: {raw_name}")
                        continue
                    
                    # Get prop type
                    raw_prop_type = prop_data.get("custom_pick_type_display_name", "")
                    prop_type = self._normalize_prop_type(raw_prop_type)
                    
                    if not prop_type:
                        logger.debug(f"⚠️ Unknown prop type: {raw_prop_type}")
                        continue
                    
                    # Get lines and odds
                    lines = prop_data.get("lines", [])
                    if not lines:
                        continue
                    
                    # Group by line value to find over/under pairs
                    line_groups: Dict[float, Dict[str, Any]] = {}
                    for line_data in lines:
                        value = line_data.get("value")
                        side = line_data.get("side", "").lower()
                        odds = line_data.get("odds")
                        
                        if value is None or not side or odds is None:
                            continue
                        
                        if value not in line_groups:
                            line_groups[value] = {}
                        
                        line_groups[value][side] = odds
                    
                    # Create props for each line (accept single-sided props too)
                    for line_value, sides in line_groups.items():
                        over_odds = sides.get("over")
                        under_odds = sides.get("under")
                        
                        # Accept props with at least one side
                        # If only one side available, use same odds for both
                        if over_odds is None and under_odds is None:
                            logger.debug(f"⚠️ Skipping {canonical_name} {prop_type} L{line_value}: no odds")
                            continue
                        
                        if over_odds is None:
                            over_odds = under_odds
                        if under_odds is None:
                            under_odds = over_odds
                        
                        # Convert American odds to decimal
                        over_decimal = self._american_to_decimal(over_odds)
                        under_decimal = self._american_to_decimal(under_odds)
                        
                        # Validate odds range [1.01, 50.0]
                        if not (1.01 <= over_decimal <= 50.0) or not (1.01 <= under_decimal <= 50.0):
                            logger.debug(f"⚠️ Skipping {canonical_name} {prop_type}: odds out of range")
                            continue

                        prop = PlayerProp(
                            player_name=canonical_name,
                            prop_type=prop_type,
                            line=float(line_value),
                            over_odds=over_decimal,
                            under_odds=under_decimal,
                            source="action_network",
                            timestamp=datetime.now()
                        )
                        props.append(prop)
                    
                except Exception as e:
                    logger.debug(f"Error parsing prop: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"❌ Error parsing API response: {e}")
            import traceback
            traceback.print_exc()
        
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
    
    
    async def fetch_props(self, date: Optional[str] = None) -> List[PlayerProp]:
        """
        Fetch player props from Action Network using direct API call.
        
        SIMPLIFIED: Uses direct HTTP request to /projections/available API
        which was verified to work (Status 200).
        
        Args:
            date: Optional date string (YYYY-MM-DD). If None, uses today.
            
        Returns:
            List of PlayerProp objects
        """
        import requests
        from datetime import datetime as dt
        
        # Format date for API (YYYYMMDD)
        if date:
            try:
                date_obj = dt.strptime(date, "%Y-%m-%d")
                date_str = date_obj.strftime("%Y%m%d")
            except ValueError:
                logger.error(f"Invalid date format: {date}. Expected YYYY-MM-DD")
                return []
        else:
            date_str = dt.now().strftime("%Y%m%d")
        
        # Direct API call to endpoint that WORKS
        url = "https://api.actionnetwork.com/web/v2/leagues/4/projections/available"
        params = {
            "date": date_str,
            "isLive": "false",
            "limit": "100"
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.actionnetwork.com/"
        }
        
        logger.info(f"🌐 Fetching player props from Action Network API for {date_str}")
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"✅ API responded successfully")
            logger.debug(f"🔍 Response keys: {list(data.keys())}")
            
            # Parse using existing parser
            props = self._parse_api_response(data)
            
            logger.info(f"✅ Scraped {len(props)} player props from Action Network")
            return props
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Action Network API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error processing props: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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

