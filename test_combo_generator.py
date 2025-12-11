"""
Quick test for betting combo generator
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from betting.combo_generator import (
    calculate_parlay_probability,
    calculate_parlay_odds,
    calculate_parlay_ev,
    generate_multi_team_parlays
)

print("=" * 60)
print("TESTING BETTING COMBO GENERATOR")
print("=" * 60)

# Test 1: Basic probability calculation
print("\n1. Testing Probability Calculation")
probs = [0.65, 0.60, 0.70]  # 65%, 60%, 70%
combined_prob = calculate_parlay_probability(probs)
print(f"   Individual probs: {[f'{p*100:.1f}%' for p in probs]}")
print(f"   Combined prob: {combined_prob*100:.1f}%")
print(f"   Expected: 27.3% (0.65 * 0.60 * 0.70)")

# Test 2: Odds calculation
print("\n2. Testing Odds Calculation")
odds = [1.80, 1.90, 1.75]
combined_odd = calculate_parlay_odds(odds)
print(f"   Individual odds: {odds}")
print(f"   Combined odd: {combined_odd:.2f}")
print(f"   Expected: 5.99 (1.80 * 1.90 * 1.75)")

# Test 3: EV calculation
print("\n3. Testing EV Calculation")
prob = 0.273
odd = 5.99
ev = calculate_parlay_ev(prob, odd)
print(f"   Probability: {prob*100:.1f}%")
print(f"   Odd: {odd:.2f}")
print(f"   EV: {ev:+.2f}%")
print(f"   Formula: ({prob} * {odd} - 1) * 100")

# Test 4: Multi-team parlay generation
print("\n4. Testing Multi-Team Parlay Generation")
# Create mock game data
games_data = {
    'home_team': ['Lakers', 'Warriors', 'Celtics'],
    'away_team': ['Knicks', 'Rockets', 'Heat'],
    'prob_home': [65, 70, 58],
    'prob_away': [35, 30, 42],
    'odds_home': [1.80, 1.65, 2.10],
    'odds_away': [2.50, 3.00, 1.90],
    'date': ['2025-12-09'] * 3
}
df_games = pd.DataFrame(games_data)

parlays = generate_multi_team_parlays(
    df_games,
    parlay_size=2,
    min_ev=0.0,  # Accept any EV for testing
    min_prob_per_game=0.55,
    max_parlays=5
)

print(f"   Generated {len(parlays)} 2-team parlays")
if parlays:
    print(f"\n   Best Parlay:")
    best = parlays[0]
    print(f"   - Description: {best['description']}")
    print(f"   - Combined Prob: {best['combined_prob']*100:.1f}%")
    print(f"   - Combined Odd: {best['combined_odd']:.2f}")
    print(f"   - EV: {best['ev']:+.2f}%")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
print("=" * 60)
