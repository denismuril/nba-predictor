"""
⚠️ DEPRECATED - Use data/scrapers/odds_scraper.py instead
This file contained mock odds functionality which has been removed.
Use the unified odds_scraper module which supports multiple real APIs.
"""
import logging

logger = logging.getLogger(__name__)
logger.warning(
    "⚠️ DEPRECATED: odds_api_client.py is deprecated. "
    "Use data/scrapers/odds_scraper.py for real API odds."  
)
