"""
Script para análise de importância de features (Feature Importance).

Objetivos:
1. Carregar o modelo treinado (RandomForest/XGBoost/Ensemble)
2. Extrair feature importances (MDI)
3. Calcular Permutation Importance (mais robusto)
4. Identificar features irrelevantes (ruído)
5. Gerar relatório visual e CSV

Usage:
    python scripts/analyze_feature_importance.py
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Adicionar raiz ao path
sys.path.append(os.getcwd())

from data.repositories.db_manager import get_db_manager
from ml_pipeline.data_preparation import load_historical_data, prepare_data_for_training

# Configuração de Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_feature_importance():
    logger.info("🚀 Iniciando Análise de Importância de Features...")
    
    # 1. Carregar Modelo e Features
    model_path = Path('data/models/ensemble_model_final.joblib')
    features_path = Path('data/models/feature_names_final.joblib')
    
    if not model_path.exists() or not features_path.exists():
        logger.error("❌ Modelo ou features não encontrados. Treine o modelo primeiro.")
        return
        
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    logger.info(f"✅ Modelo carregado: {type(model).__name__}")
    logger.info(f"✅ Total de features: {len(feature_names)}")
    
    # 2. Carregar Dados Recentes para Validação
    logger.info("📊 Carregando dados históricos para análise...")
    df = load_historical_data()
    
    # --- PRÉ-PROCESSAMENTO IDÊNTICO AO TREINO (train_ensemble_v3.py) ---
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 # REMOVER BOX SCORES BRUTOS
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 # REMOVER FOUR FACTORS BRUTOS
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 # REMOVER prob_home e prob_away
                 'prob_home', 'prob_away']
    
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # One-Hot Encoding para times
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    
    # Alinhar colunas com o modelo (adicionar faltantes com 0, remover extras)
    # Isso é crucial porque o get_dummies pode gerar colunas diferentes se o histórico carregado não tiver todos os times
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
            
    X = X[feature_names]
    
    # Target
    y = (df['winner'] == 'HOME').astype(int)
    
    # -------------------------------------------------------------------
    
    # Split (usar test set para permutation importance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # 3. MDI Feature Importance (se disponível)
    mdi_importances = pd.Series(dtype=float)
    
    # O modelo é um VotingClassifier (Ensemble), precisamos acessar os estimadores internos
    if hasattr(model, 'estimators_'):
        logger.info("🔍 Analisando estimadores do Ensemble...")
        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                logger.info(f"   Extracting from {name}...")
                importances = estimator.feature_importances_
                mdi_importances = pd.Series(importances, index=feature_names)
                
                # Salvar gráfico MDI
                plt.figure(figsize=(10, 12))
                mdi_importances.sort_values().plot(kind='barh')
                plt.title(f'Feature Importance (MDI) - {name}')
                plt.tight_layout()
                plt.savefig(f'results/feature_importance_mdi_{name}.png')
                logger.info(f"   📸 Gráfico salvo: results/feature_importance_mdi_{name}.png")
    
    # 4. Permutation Importance (Model Agnostic & Mais Robusto)
    logger.info("🔄 Calculando Permutation Importance (pode demorar)...")
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    perm_sorted_idx = result.importances_mean.argsort()
    
    # Criar DataFrame de resultados
    perm_df = pd.DataFrame({
        'feature': np.array(feature_names)[perm_sorted_idx],
        'importance_mean': result.importances_mean[perm_sorted_idx],
        'importance_std': result.importances_std[perm_sorted_idx]
    }).sort_values('importance_mean', ascending=False)
    
    # Salvar CSV
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    perm_df.to_csv(results_dir / 'feature_importance_permutation.csv', index=False)
    logger.info(f"💾 CSV salvo: {results_dir / 'feature_importance_permutation.csv'}")
    
    # Salvar Gráfico Permutation
    plt.figure(figsize=(12, 14))
    plt.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[perm_sorted_idx],
    )
    plt.title("Permutation Importance (Test Set)")
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_importance_permutation.png')
    logger.info(f"📸 Gráfico salvo: {results_dir / 'feature_importance_permutation.png'}")
    
    # 5. Análise de Features Irrelevantes
    threshold = 0.001 # 0.1% de contribuição
    low_importance = perm_df[perm_df['importance_mean'] < threshold]
    
    logger.info("\n" + "="*50)
    logger.info("📉 FEATURES DE BAIXA IMPORTÂNCIA (< 0.1%)")
    logger.info("="*50)
    
    if not low_importance.empty:
        for _, row in low_importance.iterrows():
            logger.info(f"❌ {row['feature']}: {row['importance_mean']:.6f} +/- {row['importance_std']:.6f}")
        
        logger.info(f"\n💡 Sugestão: Remover {len(low_importance)} features para simplificar o modelo.")
    else:
        logger.info("✅ Todas as features parecem contribuir significativamente.")

if __name__ == "__main__":
    analyze_feature_importance()
