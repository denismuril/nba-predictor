#!/usr/bin/env python
"""Simple test for action network scraper."""
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)

from data.scrapers.action_network_scraper import ActionNetworkScraper

async def main():
    print("Testing ActionNetworkScraper...")
    scraper = ActionNetworkScraper()
    props = await scraper.fetch_props()
    print(f"\n=== Found {len(props)} props ===")
    for p in props[:5]:
        print(f"  {p.player_name}: {p.prop_type} {p.line} (O:{p.over_odds:.2f} U:{p.under_odds:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
