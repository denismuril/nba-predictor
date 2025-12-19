"""
Props Scrapers Package - NBA Player Props Data Sources

This package contains all player props scrapers for the NBA Predictor system.
Each scraper implements the PlayerPropsProvider interface.

Available scrapers:
- ActionNetworkScraper: API-based, most stable
- LinemateScraper: XHR interception
- BettingProsScraper: Consensus odds from multiple books
- CoversScraper: Over/Under lines
- UnabatedScraper: Sharp odds (requires auth)
- DraftEdgeScraper: Fantasy projections (requires auth)
- PropsMadnessScraper: Public props aggregator
- PropsComScraper: Best props today
- LineStarScraper: DraftKings props

v26.4: Reorganized into dedicated package.
"""

from data.scrapers.props.action_network_scraper import ActionNetworkScraper
from data.scrapers.props.linemate_scraper import LinemateScraper
from data.scrapers.props.bettingpros_scraper import BettingProsScraper
from data.scrapers.props.covers_scraper import CoversScraper
from data.scrapers.props.unabated_scraper import UnabatedScraper
from data.scrapers.props.draftedge_scraper import DraftEdgeScraper

# New scrapers (v26.4)
try:
    from data.scrapers.props.propsmadness_scraper import PropsMadnessScraper
except ImportError:
    PropsMadnessScraper = None

try:
    from data.scrapers.props.propscom_scraper import PropsComScraper
except ImportError:
    PropsComScraper = None

try:
    from data.scrapers.props.linestar_scraper import LineStarScraper
except ImportError:
    LineStarScraper = None

__all__ = [
    "ActionNetworkScraper",
    "LinemateScraper",
    "BettingProsScraper",
    "CoversScraper",
    "UnabatedScraper",
    "DraftEdgeScraper",
    "PropsMadnessScraper",
    "PropsComScraper",
    "LineStarScraper",
]
