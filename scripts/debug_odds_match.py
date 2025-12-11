# scripts/debug_odds_match.py
import sys
import os
sys.path.append(os.getcwd())
from market.odds_shopping import fetch_multi_bookie_odds

print("🔍 Testando normalização de Odds...")
data = fetch_multi_bookie_odds()
print("\n📋 Times Mandantes Detectados (API -> ID Interno):")
for game in data:
    print(f"   API: '{game['home_team']}' -> ID: '{game.get('home_team_id')}'")
