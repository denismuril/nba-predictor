#!/usr/bin/env python3
"""Script de teste rápido para validar novas features V22.0"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

from ml_pipeline.feature_engineering_v2 import add_schedule_fatigue_features, add_specific_matchup_features
import pandas as pd
import numpy as np

# Criar DataFrame dummy
dates = pd.date_range('2025-11-01', periods=30, freq='D')
np.random.seed(42)
df = pd.DataFrame({
    'date': dates,
    'home_team': ['Lakers']*15 + ['Celtics']*15,
    'away_team': ['Celtics']*15 + ['Lakers']*15,
    'home_score': np.random.randint(100, 130, 30),
    'away_score': np.random.randint(100, 130, 30)
})

print("=" * 60)
print("TESTE: add_schedule_fatigue_features")
print("=" * 60)
df_fatigue = add_schedule_fatigue_features(df)
print("Colunas criadas:", [c for c in df_fatigue.columns if 'rest' in c or 'b2b' in c or '3_in_4' in c or 'games' in c])
print("\nAmostra de dados:")
print(df_fatigue[['date', 'home_team', 'home_rest_days', 'home_is_b2b', 'rest_advantage']].head(10))

print("\n" + "=" * 60)
print("TESTE: add_specific_matchup_features")
print("=" * 60)
# Adicionar colunas rolling necessárias para matchup
df_fatigue['home_rolling_10_efg_pct'] = 0.55 + np.random.randn(30) * 0.03
df_fatigue['away_rolling_10_efg_pct'] = 0.55 + np.random.randn(30) * 0.03
df_fatigue['home_rolling_10_def_rating'] = 115 + np.random.randn(30) * 5
df_fatigue['away_rolling_10_def_rating'] = 115 + np.random.randn(30) * 5

df_matchup = add_specific_matchup_features(df_fatigue)
print("Colunas criadas:", [c for c in df_matchup.columns if 'matchup' in c or 'rebound' in c])
print("\nAmostra de dados:")
print(df_matchup[['date', 'three_pt_matchup_delta', 'rebound_advantage']].head(10))

print("\n" + "=" * 60)
print("✅ TODOS OS TESTES PASSARAM!")
print("=" * 60)
