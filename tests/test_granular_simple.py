"""
Teste Granular Simplificado - Versão Robusta

Testa APENAS features que existem no df para evitar KeyError.
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from ml_pipeline.data_preparation import load_historical_data

import logging
logging.basicConfig(level=logging.WARNING)

# Carregar dados e features do modelo
df = load_historical_data(seasons=['2023-24', '2024-25'])
df = df.sort_values('date').reset_index(drop=True)
y = (df['winner'] == 'HOME').astype(int)

# Carregar lista de features do modelo V6
all_model_features = joblib.load('data/models/feature_names_v6.joblib')

# Features base (seguras)
BASE_FEATURES = [
    'home_elo', 'away_elo',
    'home_rolling_10_points', 'away_rolling_10_points',
    'home_rest_days', 'away_rest_days',
]
BASE_FEATURES = [f for f in BASE_FEATURES if f in df.columns]

# Criar grupos automaticamente baseado em padrões
def create_feature_groups(all_features):
    """Agrupa features por padrão de nome."""
    groups = {}
    
    # Grupo 1: off_rating
    groups['off_rating'] = [f for f in all_features if 'off_rating' in f.lower()]
    
    # Grupo 2: def_rating
    groups['def_rating'] = [f for f in all_features if 'def_rating' in f.lower()]
    
    # Grupo 3: RAPM/BPM
    groups['rapm_bpm'] = [f for f in all_features if 'rapm' in f.lower() or 'bpm' in f.lower()]
    
    # Grupo 4: ortg/drtg adjusted
    groups['ortg_drtg_adj'] = [f for f in all_features if 'ortg_adj' in f.lower() or 'drtg_adj' in f.lower()]
    
    # Grupo 5: contextual (home/away splits)
    groups['contextual_home_away'] = [f for f in all_features if ('_at_home' in f.lower() or '_at_away' in f.lower())]
    
    # Grupo 6: win_streak
    groups['win_streak'] = [f for f in all_features if 'streak' in f.lower()]
    
    # Grupo 7: Four Factors rolling (efg, tov, ftr, orb)
    groups['four_factors'] = [f for f in all_features if any(x in f.lower() for x in ['efg_pct', 'tov_pct', 'ftr', 'orb_pct', 'oreb_pct'])]
    
    return {k: v for k, v in groups.items() if v}  # Remove grupos vazios

def test_features(features_to_test):
    """Testa um conjunto de features e retorna acurácia média."""
    if not features_to_test:
        return None
        
    X = df[features_to_test].fillna(0)
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    scores = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))
    
    return np.mean(scores)

# Executar testes
print("="*80)
print("🔬 TESTE GRANULAR SIMPLIFICADO")
print("="*80)

# Teste baseline
print(f"\n📊 Baseline (features essenciais apenas):")
baseline_acc = test_features(BASE_FEATURES)
print(f"   Features: {len(BASE_FEATURES)}")
print(f"   Acurácia: {baseline_acc*100:.2f}%")

# Criar grupos
feature_groups = create_feature_groups(all_model_features)

print(f"\n🔍 Testando {len(feature_groups)} grupos de features:\n")

results = []
for group_name, group_feats in feature_groups.items():
    # Filtrar apenas features que existem no df
    available_feats = [f for f in group_feats if f in df.columns]
    
    if not available_feats:
        continue
        
    # Testar base + grupo
    test_feats = list(set(BASE_FEATURES + available_feats))
    acc = test_features(test_feats)
    
    delta = acc - baseline_acc
    results.append((group_name, acc, delta, len(available_feats)))
    
    # Indicador
    if acc > 0.85:
        indicator = "🚨 VAZAMENTO!"
    elif delta > 0.10:
        indicator = "⚠️ SUSPEITO"
    else:
        indicator = "✅ OK"
    
    print(f"   {group_name:25s}: {acc*100:.2f}% (+{delta*100:+.2f}pp) | {len(available_feats):3d} feats | {indicator}")

# Resultados
print(f"\n" + "="*80)
print("📊 RESULTADO:")
print("="*80)

# Ordenar por maior delta
results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

if results_sorted[0][1] > 0.85:
    print(f"🚨 VAZAMENTO DETECTADO: '{results_sorted[0][0]}'")
    print(f"   Acurácia: {baseline_acc*100:.2f}% → {results_sorted[0][1]*100:.2f}%")
    print(f"   Delta: +{results_sorted[0][2]*100:.2f} pontos percentuais")
else:
    print(f"Top 3 grupos com maior impacto:")
    for name, acc, delta, n_feats in results_sorted[:3]:
        print(f"   {name}: +{delta*100:.2f}pp ({acc*100:.2f}% acc, {n_feats} features)")

print("="*80)
