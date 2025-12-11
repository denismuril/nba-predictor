#!/usr/bin/env python
"""Verifica features suspeitas de leakage."""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')
import joblib

features = joblib.load('data/models/feature_names_v6.joblib')
print(f'Total features: {len(features)}')
print()

# Features suspeitas
print('Features com score/point (podem ser leakage):')
for f in features:
    if 'score' in f.lower() or 'point' in f.lower():
        print(f'  ⚠️  {f}')

print()
print('Features com off_rating/def_rating RAW (leakage):')
for f in features:
    if f in ['home_off_rating', 'away_off_rating', 'home_def_rating', 'away_def_rating']:
        print(f'  🚨 {f}')
