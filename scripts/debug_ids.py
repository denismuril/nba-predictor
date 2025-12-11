#!/usr/bin/env python3
"""Debug script para ver IDs pendentes vs IDs do scraper"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from data.repositories.db_manager import get_db_manager
from data.scrapers.results_scraper import get_game_results

print("🔍 IDs PENDENTES NO BANCO:")
db = get_db_manager()
pending = db.get_pending_games()
pending_24_25 = pending[pending['date'].astype(str).str.startswith(('2025-11-24', '2025-11-25'))]

for _, game in pending_24_25.iterrows():
    print(f"  {game['id']}")

print("\n🔍 IDs DO SCRAPER:")
results = get_game_results(days_back=3)
for r in results:
    print(f"  {r['id']}")

print("\n🔍 COMPARACAO:")
print(f"Pendentes: {len(pending_24_25)}")
print(f"Encontrados: {len(results)}")
