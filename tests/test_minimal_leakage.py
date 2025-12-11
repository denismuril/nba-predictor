"""
Teste Diagnóstico Minimalista - Data Leakage Isolation

Treina modelo usando APENAS 10 features essenciais conhecidas como seguras:
- Elo ratings (pré-jogo)
- Rest days (calendário)
- Rolling points (médias históricas com shift)
- Back-to-back status

Objetivo: Se acurácia cair para ~55%, sabemos que o vazamento está nas outras features.
Se permanecer em 95%, o vazamento está nestas features essenciais.
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import numpy as np

from ml_pipeline.data_preparation import load_historical_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*80)
print("🔬 TESTE DIAGNÓSTICO: Modelo Minimalista (10 Features)")
print("="*80)

# 1. Carregar dados
df = load_historical_data(seasons=['2023-24', '2024-25'])
df = df.sort_values('date').reset_index(drop=True)

print(f"\n📊 Dataset: {len(df)} jogos carregados")

# 2. Selecionar APENAS 10 features essenciais conhecidas como seguras
MINIMAL_SAFE_FEATURES = [
    'home_elo',
    'away_elo', 
    'elo_diff',
    'home_rolling_10_points',
    'away_rolling_10_points',
    'home_rest_days',
    'away_rest_days',
    'rest_diff',
    'home_is_back_to_back',
    'away_is_back_to_back',
]

print(f"\n✅ Features selecionadas (apenas essenciais):")
for i, feat in enumerate(MINIMAL_SAFE_FEATURES, 1):
    print(f"   {i}. {feat}")

# 3. Preparar X e y
available_features = [f for f in MINIMAL_SAFE_FEATURES if f in df.columns]
missing_features = [f for f in MINIMAL_SAFE_FEATURES if f not in df.columns]

if missing_features:
    print(f"\n⚠️ Features faltando: {missing_features}")

X = df[available_features].fillna(0)
y = (df['winner'] == 'HOME').astype(int)

print(f"\n📈 X shape: {X.shape}")
print(f"📈 y shape: {y.shape}")
print(f"📈 Features realmente usadas: {len(available_features)}")

# 4. Walk-Forward Validation (TimeSeriesSplit)
print(f"\n🔄 Iniciando Walk-Forward Validation (5 splits)...")

tscv = TimeSeriesSplit(n_splits=5)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

fold_scores = []
for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    fold_scores.append(score)
    
    print(f"   Fold {fold_idx}/5: {score*100:.2f}%")

# 5. Resultados
mean_acc = np.mean(fold_scores)
std_acc = np.std(fold_scores)

print(f"\n" + "="*80)
print(f"📊 RESULTADO DO TESTE DIAGNÓSTICO")
print(f"="*80)
print(f"Acurácia Média: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"Features usadas: {len(available_features)} (mínimo essencial)")

print(f"\n🔍 DIAGNÓSTICO:")
if mean_acc < 0.60:
    print("✅ SUCESSO: Acurácia caiu para nível realista (~55%)")
    print("   → O vazamento está nas OUTRAS features (não essenciais)")
    print("   → Próximo passo: Adicionar features uma a uma para identificar culpado")
elif mean_acc > 0.90:
    print("❌ VAZAMENTO DETECTADO: Acurácia ainda muito alta (>90%)")
    print("   → O vazamento está em uma das 10 features ESSENCIAIS!")
    print("   → Culpados prováveis: elo_diff, rolling_10_points, ou rest days")
    print("   → CRÍTICO: Mesmo features 'seguras' podem ter bugs de implementação")
else:
    print("⚠️ INCONCLUSIVO: Acurácia em zona intermediária (60-90%)")
    print("   → Pode haver vazamento menor ou modelo genuinamente bom")

print("="*80)
