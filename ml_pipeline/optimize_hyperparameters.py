"""
Script de Otimização de Hiperparâmetros com Optuna.

Objetivo: Encontrar os melhores parâmetros para Random Forest e XGBoost
usando as features otimizadas (V4).

Usage:
    python ml_pipeline/optimize_hyperparameters.py
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
import optuna
import logging
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from xgboost import XGBClassifier

# Configuração de Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_pipeline.data_preparation import load_historical_data

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuração
ML_SEASONS = ['2023-24', '2024-25', '2025-26']
N_TRIALS = 50  # Número de tentativas por modelo
TIMEOUT = 600  # Tempo máximo por otimização (segundos)

def load_optimized_data():
    """Carrega dados e aplica seleção de features do V4."""
    logger.info("📊 Carregando dados para otimização...")
    df, sample_weights = load_historical_data(
        seasons=ML_SEASONS, 
        apply_weights=True
    )
    df = df.sort_values('date').reset_index(drop=True)
    
    # Pré-processamento Base
    base_drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X = df.drop(columns=base_drop_cols, errors='ignore')
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    
    # Carregar features do V4
    feature_names_path = Path('data/models/feature_names_v4.joblib')
    if not feature_names_path.exists():
        raise FileNotFoundError("Features V4 não encontradas. Rode train_ensemble_v4.py primeiro.")
        
    feature_names = joblib.load(feature_names_path)
    
    # Alinhar colunas
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    
    y = (df['winner'] == 'HOME').astype(int)
    
    return X, y, sample_weights

def optimize_rf(X, y, weights):
    """Otimiza Random Forest."""
    logger.info("\n🌲 Otimizando Random Forest...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        clf = RandomForestClassifier(**params)
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            w_train = weights[train_index] if weights is not None else None
            
            clf.fit(X_train, y_train, sample_weight=w_train)
            scores.append(clf.score(X_test, y_test))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    
    logger.info(f"✅ Melhor RF: {study.best_value:.4f}")
    logger.info(f"   Params: {study.best_params}")
    return study.best_params

def optimize_xgb(X, y, weights):
    """Otimiza XGBoost."""
    logger.info("\n🚀 Otimizando XGBoost...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42,
            'n_jobs': -1
        }
        
        clf = XGBClassifier(**params)
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            w_train = weights[train_index] if weights is not None else None
            
            clf.fit(X_train, y_train, sample_weight=w_train)
            scores.append(clf.score(X_test, y_test))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    
    logger.info(f"✅ Melhor XGB: {study.best_value:.4f}")
    logger.info(f"   Params: {study.best_params}")
    return study.best_params

def main():
    X, y, weights = load_optimized_data()
    
    best_params = {}
    
    # 1. Otimizar RF
    best_params['rf'] = optimize_rf(X, y, weights)
    
    # 2. Otimizar XGB
    best_params['xgb'] = optimize_xgb(X, y, weights)
    
    # Salvar resultados
    output_path = Path('data/models/best_hyperparameters.json')
    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=2)
        
    logger.info(f"\n💾 Melhores parâmetros salvos em {output_path}")

if __name__ == "__main__":
    main()
