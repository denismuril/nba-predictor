"""
Teste Granular de Features - Isolamento Incremental

Este teste adiciona features UMA A UMA ao modelo minimalista para identificar
EXATAMENTE qual feature causa o vazamento de 65% → 95%.

Estratégia:
1. Começar com modelo base (65% acc)
2. Adicionar grupo de features (ex: todas as rolling_off_rating)
3. Se acurácia pula para >80%, sabemos que o vazamento está nesse grupo
4. Então testar features individuais do grupo
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from ml_pipeline.data_preparation import load_historical_data

import logging
logging.basicConfig(level=logging.WARNING)  # Reduzir verbosidade

# Carregar dados
df = load_historical_data(seasons=['2023-24', '2024-25'])
df = df.sort_values('date').reset_index(drop=True)
y = (df['winner'] == 'HOME').astype(int)

# Features base (seguras - 65% acc)
BASE_FEATURES = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_rolling_10_points', 'away_rolling_10_points',
    'home_rest_days', 'away_rest_days', 'rest_diff',
    'home_is_back_to_back', 'away_is_back_to_back',
]

# Grupos de features para testar
FEATURE_GROUPS = {
    'off_rating': [
        'home_rolling_5_off_rating', 'away_rolling_5_off_rating',
        'home_rolling_10_off_rating', 'away_rolling_10_off_rating',
        'home_rolling_30_off_rating', 'away_rolling_30_off_rating',
    ],
    'def_rating': [
        'home_rolling_5_def_rating', 'away_rolling_5_def_rating',
        'home_rolling_10_def_rating', 'away_rolling_10_def_rating',
        'home_rolling_30_def_rating', 'away_rolling_30_def_rating',
    ],
    'four_factors_rolling': [
        'home_rolling_10_efg_pct', 'away_rolling_10_efg_pct',
        'home_rolling_10_tov_pct', 'away_rolling_10_tov_pct',
        'home_rolling_10_ftr', 'away_rolling_10_ftr',
        'home_rolling_10_orb_pct', 'away_rolling_10_orb_pct',
    ],
    'rapm_bpm': [
        'home_rapm_avg', 'away_rapm_avg',
        'home_bpm_avg', 'away_bpm_avg',
    ],
    'ortg_drtg_adj': [
        'home_ortg_adj', 'away_ortg_adj',
        'home_drtg_adj', 'away_drtg_adj',
    ],
    'contextual': [
        'home_rolling_10_win_at_home', 'away_rolling_10_win_at_away',
        'home_rolling_10_pts_at_home', 'away_rolling_10_pts_at_away',
    ],
}

def test_feature_group(base_features, new_features, group_name):
    """Testa um grupo de features e retorna acurácia."""
    # Selecionar features disponíveis
    test_features = base_features + [f for f in new_features if f in df.columns]
    
    X = df[test_features].fillna(0)
    
    # Walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    
    mean_acc = np.mean(scores)
    return mean_acc, len(test_features)

# Executar testes
print("="*80)
print("🔬 TESTE GRANULAR: Identificação de Feature com Vazamento")
print("="*80)

print(f"\n📊 Baseline (apenas features essenciais):")
baseline_acc, _ = test_feature_group(BASE_FEATURES, [], "baseline")
print(f"   Acurácia: {baseline_acc*100:.2f}%")

print(f"\n🔍 Testando grupos de features:\n")

results = {}
for group_name, group_features in FEATURE_GROUPS.items():
    acc, num_feats = test_feature_group(BASE_FEATURES, group_features, group_name)
    results[group_name] = acc
    
    # Indicador visual
    if acc > 0.80:
        indicator = "🚨 VAZAMENTO DETECTADO!"
    elif acc > 0.70:
        indicator = "⚠️ SUSPEITO"
    else:
        indicator = "✅ OK"
    
    print(f"   {group_name:25s}: {acc*100:.2f}% (+{(acc-baseline_acc)*100:+.2f}pp) - {num_feats} features - {indicator}")

# Identificar culpado
print(f"\n" + "="*80)
print("📊 RESULTADO:")
print("="*80)

max_group = max(results, key=results.get)
max_acc = results[max_group]

if max_acc > 0.85:
    print(f"🚨 VAZAMENTO CRÍTICO DETECTADO no grupo: '{max_group}'")
    print(f"   Acurácia subiu de {baseline_acc*100:.2f}% → {max_acc*100:.2f}%")
    print(f"\n   Próximo passo: Testar features INDIVIDUAIS deste grupo.")
    print(f"   Executar: python tests/test_individual_features.py --group={max_group}")
else:
    print(f"⚠️ Nenhum grupo individual causou >85% accuracy")
    print(f"   Isso sugere que o vazamento é COMBINADO (múltiplos grupos)")
    print(f"\n   Próximo passo: Testar combinações de grupos.")

print("="*80)
