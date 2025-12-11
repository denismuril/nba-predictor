"""
Ensemble Model V5 - Hyperparameter Optimized
Baseado no V4 (Features Otimizadas), mas usa os melhores hiperparâmetros encontrados pelo Optuna.

Melhorias:
- Carrega features otimizadas (V4)
- Carrega hiperparâmetros otimizados (Optuna)
- Treina stack final com RF e XGB tunados
"""
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Configuração de Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_pipeline.data_preparation import load_historical_data

logger = logging.getLogger(__name__)

# Configuração
ML_SEASONS = ['2023-24', '2024-25', '2025-26']
ML_SAMPLE_WEIGHT_CONFIG = {
    'enabled': True,
    'recent_30_days': 3.0,
    'recent_60_days': 2.0,
    'recent_90_days': 1.5,
    'default': 1.0
}

def load_best_params():
    """Carrega hiperparâmetros otimizados do JSON."""
    params_path = Path('data/models/best_hyperparameters.json')
    if not params_path.exists():
        logger.warning("⚠️ Arquivo de hiperparâmetros não encontrado. Usando defaults.")
        return {}, {}
        
    with open(params_path) as f:
        params = json.load(f)
        
    logger.info("✅ Hiperparâmetros carregados com sucesso.")
    return params.get('rf', {}), params.get('xgb', {})

def train_ensemble_model_v5():
    logger.info("="*80)
    logger.info("🚀 TREINANDO ENSEMBLE MODEL V5 (HYPERPARAMETER OPTIMIZED)")
    logger.info("="*80)
    
    # 1. Carregar Hiperparâmetros
    rf_params, xgb_params = load_best_params()
    
    # 2. Carregar dados
    df, sample_weights = load_historical_data(
        seasons=ML_SEASONS, 
        apply_weights=True,
        weight_config=ML_SAMPLE_WEIGHT_CONFIG
    )
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # 3. Pré-processamento Base
    base_drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X = df.drop(columns=base_drop_cols, errors='ignore')
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    
    # 4. Seleção de Features (V4)
    feature_names_path = Path('data/models/feature_names_v4.joblib')
    if not feature_names_path.exists():
        logger.error("❌ Features V4 não encontradas. Rode train_ensemble_v4.py primeiro.")
        return None, 0
        
    feature_names = joblib.load(feature_names_path)
    
    # Alinhar colunas
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    
    logger.info(f"✅ Features selecionadas: {len(X.columns)}")
    
    y = (df['winner'] == 'HOME').astype(int)
    
    # 5. Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    weights_train = sample_weights[:split_idx]
    weights_test = sample_weights[split_idx:]
    
    # 6. Treinamento com Params Otimizados
    
    # Random Forest
    if not rf_params:
        rf_params = {'n_estimators': 200, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
    else:
        # Garantir params fixos
        rf_params['random_state'] = 42
        rf_params['n_jobs'] = -1
        
    rf = RandomForestClassifier(**rf_params)
    
    # XGBoost
    if not xgb_params:
        xgb_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1}
    else:
        xgb_params['random_state'] = 42
        xgb_params['n_jobs'] = -1
        
    xgb = XGBClassifier(**xgb_params)
    
    # Outros modelos (manter defaults ou otimizar futuramente)
    extra = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    
    base_estimators = [
        ('rf', rf),
        ('xgb', xgb),
        ('extra', extra),
        ('lgbm', lgbm)
    ]
    
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    
    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )
    
    logger.info(f"🔄 Treinando Stack ({len(base_estimators)} modelos) com params otimizados...")
    ensemble.fit(X_train, y_train, sample_weight=weights_train)
    
    # 7. Avaliação
    acc = ensemble.score(X_test, y_test, sample_weight=weights_test)
    raw_acc = ensemble.score(X_test, y_test)
    
    logger.info(f"🏆 Acurácia V5 (Tuned): {raw_acc*100:.2f}% (Weighted: {acc*100:.2f}%)")
    
    # Salvar modelo V5
    joblib.dump(ensemble, 'data/models/ensemble_model_v5.joblib')
    # Definir como modelo final de produção
    joblib.dump(ensemble, 'data/models/ensemble_model_final.joblib')
    joblib.dump(feature_names, 'data/models/feature_names_final.joblib')
    
    logger.info("💾 Modelo V5 salvo e definido como FINAL para produção.")
    
    return ensemble, raw_acc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    train_ensemble_model_v5()
