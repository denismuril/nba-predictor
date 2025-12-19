#!/usr/bin/env python3
"""Quick verification test for props scraper imports."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

async def test():
    from data.odds_manager import OddsDataManager
    
    print("Testing OddsDataManager imports...")
    m = OddsDataManager()
    props = await m.fetch_player_props(datetime.now().strftime('%Y-%m-%d'))
    print(f'✅ Props fetched: {len(props)}')
    if props:
        print(f'   Example: {props[0].player_name} - {props[0].prop_type} {props[0].line}')
    return len(props)

if __name__ == "__main__":
    count = asyncio.run(test())
    sys.exit(0 if count > 0 else 1)
