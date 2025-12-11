#!/usr/bin/env python3
"""
Otimização de Hiperparâmetros para Ensemble Model (Moneyline)

Usa Optuna para encontrar os melhores hiperparâmetros para:
- Random Forest
- XGBoost
- LightGBM
- Extra Trees

Salva os melhores parâmetros em data/models/best_ensemble_params.joblib
"""
import optuna
import joblib
import pandas as pd
import numpy as np
import logging
import os
import sys
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# Adicionar root ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.data_preparation import load_historical_data

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    HAS_BOOSTING = True
except ImportError:
    HAS_BOOSTING = False
    print("⚠️  XGBoost/LightGBM não disponíveis")

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Caminho para salvar os melhores parâmetros
PARAMS_FILE = os.path.join("data", "models", "best_ensemble_params.joblib")

def objective_rf(trial, X, y, sample_weights=None):
    """Otimiza Random Forest."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if sample_weights is not None:
            w_train = sample_weights[train_idx]
        else:
            w_train = None
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
    
    return np.mean(scores)

def objective_xgb(trial, X, y, sample_weights=None):
    """Otimiza XGBoost."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'random_state': 42,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if sample_weights is not None:
            w_train = sample_weights[train_idx]
        else:
            w_train = None
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
    
    return np.mean(scores)

def objective_lgbm(trial, X, y, sample_weights=None):
    """Otimiza LightGBM."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if sample_weights is not None:
            w_train = sample_weights[train_idx]
        else:
            w_train = None
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
    
    return np.mean(scores)

def objective_extra(trial, X, y, sample_weights=None):
    """Otimiza Extra Trees."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if sample_weights is not None:
            w_train = sample_weights[train_idx]
        else:
            w_train = None
        
        model = ExtraTreesClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
    
    return np.mean(scores)

def run_optimization(n_trials=50):
    """Executa a otimização do Optuna para todos os modelos base."""
    logger.info("🚀 Iniciando Otimização de Hiperparâmetros para Ensemble...")
    
    # 1. Carregar dados
    from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG
    
    df, sample_weights = load_historical_data(
        seasons=ML_SEASONS,
        apply_weights=True,
        weight_config=ML_SAMPLE_WEIGHT_CONFIG
    )
    
    if df is None or df.empty:
        logger.error("❌ Sem dados para otimização.")
        return
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. Preparar features
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X = df.drop(columns=drop_cols, errors='ignore')
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    y = (df['winner'] == 'HOME').astype(int)
    
    logger.info(f"📊 Dados: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # 3. Otimizar cada modelo
    best_params = {}
    
    # Random Forest
    logger.info("\n🌲 Otimizando Random Forest...")
    study_rf = optuna.create_study(direction='maximize', study_name='RandomForest')
    study_rf.optimize(lambda trial: objective_rf(trial, X, y, sample_weights), n_trials=n_trials, show_progress_bar=True)
    best_params['random_forest'] = study_rf.best_params
    logger.info(f"✅ RF - Best Accuracy: {study_rf.best_value:.4f}")
    logger.info(f"   Params: {study_rf.best_params}")
    
    # XGBoost
    if HAS_BOOSTING:
        logger.info("\n🚀 Otimizando XGBoost...")
        study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost')
        study_xgb.optimize(lambda trial: objective_xgb(trial, X, y, sample_weights), n_trials=n_trials, show_progress_bar=True)
        best_params['xgboost'] = study_xgb.best_params
        logger.info(f"✅ XGB - Best Accuracy: {study_xgb.best_value:.4f}")
        logger.info(f"   Params: {study_xgb.best_params}")
        
        # LightGBM
        logger.info("\n💡 Otimizando LightGBM...")
        study_lgbm = optuna.create_study(direction='maximize', study_name='LightGBM')
        study_lgbm.optimize(lambda trial: objective_lgbm(trial, X, y, sample_weights), n_trials=n_trials, show_progress_bar=True)
        best_params['lightgbm'] = study_lgbm.best_params
        logger.info(f"✅ LGBM - Best Accuracy: {study_lgbm.best_value:.4f}")
        logger.info(f"   Params: {study_lgbm.best_params}")
    
    # Extra Trees
    logger.info("\n🌳 Otimizando Extra Trees...")
    study_extra = optuna.create_study(direction='maximize', study_name='ExtraTrees')
    study_extra.optimize(lambda trial: objective_extra(trial, X, y, sample_weights), n_trials=n_trials, show_progress_bar=True)
    best_params['extra_trees'] = study_extra.best_params
    logger.info(f"✅ ExtraTrees - Best Accuracy: {study_extra.best_value:.4f}")
    logger.info(f"   Params: {study_extra.best_params}")
    
    # 4. Salvar
    os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
    joblib.dump(best_params, PARAMS_FILE)
    logger.info(f"\n💾 Parâmetros salvos em: {PARAMS_FILE}")
    
    return best_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50, help='Número de trials por modelo')
    args = parser.parse_args()
    
    run_optimization(n_trials=args.n_trials)
