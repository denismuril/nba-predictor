#!/usr/bin/env python3
"""
Otimização de Hiperparâmetros para Pipeline V3 (Totals & Moneyline)

Usa Optuna para encontrar os melhores hiperparâmetros para:
1. Totals Model V17 (XGBRegressor)
2. Ensemble Model V7 (XGBClassifier)

Salva os melhores parâmetros em data/models/best_params_v3.joblib
"""
import optuna
import joblib
import pandas as pd
import numpy as np
import logging
import os
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Adicionar root ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.data_preparation import load_historical_data
from ml_pipeline.feature_pipeline_v3 import prepare_features_v3

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost não disponível")

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Caminho para salvar os melhores parâmetros
PARAMS_FILE = os.path.join("data", "models", "best_params_v3.joblib")

def load_and_prepare_data():
    """Carrega dados e aplica pipeline V3."""
    logger.info("📊 Carregando dados históricos (RAW)...")
    df = load_historical_data(raw=True)
    
    if df is None or df.empty:
        logger.error("❌ Sem dados para otimização.")
        return None
    
    logger.info("🔧 Executando Feature Pipeline V3...")
    df = prepare_features_v3(df)
    
    # Remover colunas não numéricas e targets
    drop_cols = ['winner', 'date', 'home_team', 'away_team', 'total_points', 
                 'home_score', 'away_score', 'pt_diff', 'point_differential']
    
    # Identificar colunas target
    y_totals = df['total_points']
    y_moneyline = (df['winner'] == 'HOME').astype(int)
    
    # Selecionar apenas features calculadas (evitar leakage)
    # Whitelist de prefixos conhecidos do pipeline V3
    valid_prefixes = [
        'home_rolling_', 'away_rolling_',
        'expected_', 'pace_',
        'home_rest_', 'away_rest_',
        'home_b2b', 'away_b2b',
        'home_games_', 'away_games_',
        'day_', 'month', 'is_weekend', 'season_',
        'home_rapm_', 'away_rapm_',
        'home_bpm_', 'away_bpm_',
        'rapm_', 'bpm_', 'depth_',
        'h2h_'
    ]
    
    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in valid_prefixes)]
    
    # Remover colunas que possam ter vazado mesmo com prefixo (ex: home_rolling_score se existisse)
    # Mas no V3 não criamos rolling score do jogo atual, apenas anteriores.
    
    X = df[feature_cols].copy()
    X = X.select_dtypes(include=[np.number])
    
    logger.info(f"✅ Features selecionadas: {len(X.columns)}")
    logger.info(f"   Exemplos: {X.columns.tolist()[:5]}")
    
    # Remover NaNs (preencher com 0 ou média, aqui 0 para simplificar e consistência com validação)
    X = X.fillna(0)
    
    return X, y_totals, y_moneyline

def objective_totals_xgb(trial, X, y):
    """Otimiza XGBoost para Totals (MAE)."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
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
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        scores.append(mae)
    
    return np.mean(scores)

def objective_moneyline_xgb(trial, X, y):
    """Otimiza XGBoost para Moneyline (Accuracy)."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2), # Balanceamento leve
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)
    
    return np.mean(scores)

def run_optimization(n_trials=20, optimize_totals=True, optimize_moneyline=True):
    """Executa a otimização."""
    if not HAS_XGB:
        logger.error("❌ XGBoost necessário para esta otimização.")
        return

    # Carregar dados
    data = load_and_prepare_data()
    if data is None:
        return
    
    X, y_totals, y_moneyline = data
    logger.info(f"📊 Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
    
    best_params = {}
    
    # Carregar parâmetros existentes se houver
    if os.path.exists(PARAMS_FILE):
        try:
            best_params = joblib.load(PARAMS_FILE)
            logger.info("📂 Parâmetros anteriores carregados.")
        except:
            pass
    
    # 1. Otimizar Totals
    if optimize_totals:
        logger.info("\n🏀 Otimizando Totals Model (XGBRegressor)...")
        study_totals = optuna.create_study(direction='minimize', study_name='Totals_V3')
        study_totals.optimize(lambda trial: objective_totals_xgb(trial, X, y_totals), n_trials=n_trials, show_progress_bar=True)
        
        best_params['totals_v17'] = study_totals.best_params
        logger.info(f"✅ Totals Best MAE: {study_totals.best_value:.4f}")
        logger.info(f"   Params: {study_totals.best_params}")
    
    # 2. Otimizar Moneyline
    if optimize_moneyline:
        logger.info("\n🎯 Otimizando Moneyline Model (XGBClassifier)...")
        study_ml = optuna.create_study(direction='maximize', study_name='Moneyline_V3')
        study_ml.optimize(lambda trial: objective_moneyline_xgb(trial, X, y_moneyline), n_trials=n_trials, show_progress_bar=True)
        
        best_params['ensemble_v7'] = study_ml.best_params
        logger.info(f"✅ Moneyline Best Accuracy: {study_ml.best_value:.4f}")
        logger.info(f"   Params: {study_ml.best_params}")
    
    # Salvar
    os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
    joblib.dump(best_params, PARAMS_FILE)
    logger.info(f"\n💾 Melhores parâmetros salvos em: {PARAMS_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=20, help='Número de trials')
    parser.add_argument('--totals-only', action='store_true', help='Otimizar apenas Totals')
    parser.add_argument('--moneyline-only', action='store_true', help='Otimizar apenas Moneyline')
    
    args = parser.parse_args()
    
    run_totals = not args.moneyline_only
    run_moneyline = not args.totals_only
    
    run_optimization(n_trials=args.n_trials, optimize_totals=run_totals, optimize_moneyline=run_moneyline)
