#!/usr/bin/env python3
"""Test PlayerImpactCalculator."""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

from ml_pipeline.player_impact import PlayerImpactCalculator

print("=" * 50)
print("Testing PlayerImpactCalculator with RAPM")
print("=" * 50)

calc = PlayerImpactCalculator()

# Test individual players
test_players = [
    'LeBron James',
    'Luka Doncic',
    'Anthony Davis',
    'Nikola Jokic',
    'Stephen Curry'
]

print("\n1. Individual Player RAPM:")
for p in test_players:
    rapm = calc.get_player_rapm(p)
    status = "OK" if rapm != 0.0 else "NOT FOUND"
    print(f"   {p}: {rapm:+.2f} [{status}]")

# Test missing impact calculation
print("\n2. Missing Impact Calculation:")
injured = ['LeBron James', 'Anthony Davis']
impact = calc.calculate_missing_impact(injured)
print(f"   Injured: {injured}")
print(f"   Total RAPM lost: {impact:+.2f}")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)
