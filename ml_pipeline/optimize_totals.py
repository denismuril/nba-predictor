#!/usr/bin/env python3
"""
Otimização de Hiperparâmetros para Totals Model (Over/Under)

Usa Optuna para encontrar os melhores hiperparâmetros para o modelo de totais.
Salva os melhores parâmetros em data/models/best_totals_params.joblib
"""
import optuna
import joblib
import pandas as pd
import numpy as np
import logging
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Adicionar root ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.data_preparation import load_historical_data

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost não disponível")

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Caminho para salvar os melhores parâmetros
PARAMS_FILE = os.path.join("data", "models", "best_totals_params.joblib")

def objective_rf(trial, X, y, sample_weights=None):
    """Otimiza Random Forest para Totals."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 6, 20),
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
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        scores.append(mae)
    
    return np.mean(scores)

def objective_xgb(trial, X, y, sample_weights=None):
    """Otimiza XGBoost para Totals."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'objective': 'reg:absoluteerror',
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
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        scores.append(mae)
    
    return np.mean(scores)

def run_optimization(n_trials=50, use_xgb=True):
    """Executa a otimização do Optuna para totals model."""
    logger.info("🚀 Iniciando Otimização de Hiperparâmetros para Totals Model...")
    
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
                 'home_score', 'away_score', 'pt_diff',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X = df.drop(columns=drop_cols, errors='ignore')
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    y = df['total_points']
    
    # Remover NaNs
    valid_idx = ~X.isna().any(axis=1) & ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    if sample_weights is not None:
        sample_weights = sample_weights[valid_idx]
    
    logger.info(f"📊 Dados: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # 3. Otimizar modelos
    best_params = {}
    
    # Random Forest
    logger.info("\n🌲 Otimizando Random Forest...")
    study_rf = optuna.create_study(direction='minimize', study_name='RandomForest_Totals')
    study_rf.optimize(lambda trial: objective_rf(trial, X, y, sample_weights), n_trials=n_trials, show_progress_bar=True)
    best_params['random_forest'] = study_rf.best_params
    logger.info(f"✅ RF - Best MAE: {study_rf.best_value:.4f}")
    logger.info(f"   Params: {study_rf.best_params}")
    
    # XGBoost
    if use_xgb and HAS_XGB:
        logger.info("\n🚀 Otimizando XGBoost...")
        study_xgb = optuna.create_study(direction='minimize', study_name='XGBoost_Totals')
        study_xgb.optimize(lambda trial: objective_xgb(trial, X, y, sample_weights), n_trials=n_trials, show_progress_bar=True)
        best_params['xgboost'] = study_xgb.best_params
        logger.info(f"✅ XGB - Best MAE: {study_xgb.best_value:.4f}")
        logger.info(f"   Params: {study_xgb.best_params}")
    
    # 4. Salvar
    os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
    joblib.dump(best_params, PARAMS_FILE)
    logger.info(f"\n💾 Parâmetros salvos em: {PARAMS_FILE}")
    
    return best_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50, help='Número de trials por modelo')
    parser.add_argument('--no-xgb', action='store_true', help='Desabilitar XGBoost')
    args = parser.parse_args()
    
    run_optimization(n_trials=args.n_trials, use_xgb=not args.no_xgb)
