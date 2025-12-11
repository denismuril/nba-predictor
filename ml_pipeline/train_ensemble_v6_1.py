"""
Ensemble Model V7 (Grand Master) - Enhanced Feature Set & Validation
"""
import sys
import os
import logging
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    StackingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit

# Configuração de Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_pipeline.feature_engineering_v2 import prepare_features_v2
from data.repositories.db_manager import get_db_manager

logger = logging.getLogger(__name__)

# Configuração
ML_SEASONS = ['2023-24', '2024-25', '2025-26']

def load_data_v7():
    """Carrega dados usando o pipeline V2."""
    db = get_db_manager()
    # Carregar raw history
    df_raw = db.get_history()
    
    # Aplicar Feature Engineering V2
    df = prepare_features_v2(df_raw)
    return df

def train_ensemble_model_v7():
    logger.info("="*80)
    logger.info("🚀 TREINANDO ENSEMBLE MODEL V7 (GRAND MASTER)")
    logger.info("="*80)
    
    df = load_data_v7()
    if df is None or df.empty:
        logger.error("❌ Sem dados para treino.")
        return None, 0
        
    df = df.sort_values('date').reset_index(drop=True)
    
    # Definir Target
    df['target'] = (df['home_score'] > df['away_score']).astype(int)
    
    # Selecionar Features (Whitelist estrita para evitar Leakage)
    # Apenas features conhecidas ANTES do jogo
    allowed_prefixes = ['home_rolling_', 'away_rolling_', 'home_last_', 'away_last_']
    allowed_exact = [
        'home_rest_days', 'away_rest_days', 
        'home_is_b2b', 'away_is_b2b',
        'home_roster_impact', 'away_roster_impact'
    ]
    
    features = []
    for c in df.columns:
        if c in allowed_exact:
            features.append(c)
        elif any(c.startswith(p) for p in allowed_prefixes):
            features.append(c)
            
    # Garantir que não estamos usando stats do próprio jogo
    # Ex: home_off_rating, home_def_rating, home_efg_pct (são do jogo atual)
    
    X = df[features]
    y = df['target']
    
    logger.info(f"📊 Features selecionadas ({len(features)}): {features[:5]} ...")
    
    # Salvar nomes das features
    joblib.dump(features, 'data/models/feature_names_v7.joblib')
    
    # Walk-Forward Validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    logger.info("🔄 Iniciando Walk-Forward Validation (5 splits)...")
    
    # Modelos Base (Padrão V6.1)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
    extra = ExtraTreesClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1, verbose=-1, random_state=42)
    hist = HistGradientBoostingClassifier(max_iter=200, max_depth=10, learning_rate=0.05, random_state=42)
    
    base_estimators = [
        ('rf', rf), ('xgb', xgb), ('extra', extra), ('lgbm', lgbm), ('hist', hist)
    ]
    
    from sklearn.ensemble import VotingClassifier
    
    ensemble = VotingClassifier(
        estimators=base_estimators,
        voting='soft',
        n_jobs=-1
    )
    
    # Validação Manual
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        ensemble.fit(X_train_fold, y_train_fold)
        score = ensemble.score(X_test_fold, y_test_fold)
        scores.append(score)
        logger.info(f"   Split {fold+1}: Acc = {score:.4f}")
        
    mean_acc = np.mean(scores)
    logger.info(f"🏆 Média Walk-Forward Acc: {mean_acc*100:.2f}%")
    
    # Treino Final (Full Data)
    logger.info("🚀 Treinando modelo final com todos os dados...")
    ensemble.fit(X, y)
    
    # Salvar Modelo
    model_path = 'data/models/ensemble_v7.joblib'
    joblib.dump(ensemble, model_path)
    logger.info(f"💾 Modelo salvo em {model_path}")
    
    return ensemble, mean_acc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    train_ensemble_model_v7()
