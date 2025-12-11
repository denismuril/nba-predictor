#!/usr/bin/env python
"""Script para listar features disponíveis no DataFrame."""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

from ml_pipeline.data_preparation import load_historical_data

df = load_historical_data(raw=False)
print(f"Total colunas: {len(df.columns)}")

# Rolling features
rolling_cols = [c for c in df.columns if 'rolling' in c.lower()]
print(f"\nROLLING ({len(rolling_cols)}):")
for c in sorted(rolling_cols)[:20]:
    print(f"  {c}")

# Elo features
elo_cols = [c for c in df.columns if 'elo' in c.lower()]
print(f"\nELO ({len(elo_cols)}):")
for c in elo_cols:
    print(f"  {c}")

# Features que começam com home_ ou away_ 
home_away = [c for c in df.columns if c.startswith('home_') or c.startswith('away_')]
print(f"\nHOME_/AWAY_ ({len(home_away)}):")
for c in sorted(home_away)[:30]:
    print(f"  {c}")
