"""
Odds Sites Scrapers Package.

Contém scrapers individuais para diferentes sites de odds.
"""

from data.scrapers.odds_sites.base_scraper import BaseSiteScraper, PlaywrightMixin

__all__ = ['BaseSiteScraper', 'PlaywrightMixin']
