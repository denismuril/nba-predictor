#!/usr/bin/env python3
"""
Otimização de Hiperparâmetros V6 - Pós Data Leakage Fix

Script dedicado para re-tunar os hiperparâmetros dos modelos base do Ensemble V6
após a correção do vazamento de dados. Usa busca Bayesiana (Optuna) com
métricas focadas em probabilidades calibradas (Log Loss / Brier Score).

Modelos otimizados:
1. RandomForest - Foco em min_samples_leaf e max_depth
2. XGBoost - Foco em learning_rate baixo e regularização L1/L2
3. HistGradientBoosting - Foco em l2_regularization
4. LightGBM - Regularização e learning_rate
5. ExtraTrees - Similar ao RF

Usage:
    python ml_pipeline/optimize_hyperparameters_v6.py --n_trials 50
    python ml_pipeline/optimize_hyperparameters_v6.py --n_trials 3 --timeout 60  # Smoke test
"""
import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuração de Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_pipeline.data_preparation import load_historical_data, calculate_sample_weights

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
ML_SEASONS = ['2023-24', '2024-25', '2025-26']
ML_SAMPLE_WEIGHT_CONFIG = {
    'enabled': True,
    'recent_30_days': 3.0,
    'recent_60_days': 2.0,
    'recent_90_days': 1.5,
    'default': 1.0
}

# Colunas a remover (mesmo do train_ensemble_v6.py)
BASE_DROP_COLS = [
    # Resultado direto do jogo
    'winner', 'correct', 'date', 'prediction',
    'home_score', 'away_score', 'pt_diff', 'point_differential', 'total_points',
    
    # Estatísticas do jogo atual (leakage)
    'pts', 'opp_pts',
    'fgm', 'fga', 'fg3m', 'fg3a', 'tov', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'pf', 'fta', 'ftm',
    'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_fg3a', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_reb', 
    'opp_ast', 'opp_stl', 'opp_blk', 'opp_pf', 'opp_fta', 'opp_ftm',
    
    # Four Factors do jogo atual
    'home_efg', 'home_efg_pct', 'home_tov_pct', 'home_orb_pct', 'home_ftr', 
    'home_off_rating', 'home_def_rating', 'home_pace', 'home_pie', 
    'home_pos', 'home_ast_ratio', 'home_to_ratio', 'home_ts_pct', 'home_reb_pct',
    'away_efg', 'away_efg_pct', 'away_tov_pct', 'away_orb_pct', 'away_ftr', 
    'away_off_rating', 'away_def_rating', 'away_pace', 'away_pie', 
    'away_pos', 'away_ast_ratio', 'away_to_ratio', 'away_ts_pct', 'away_reb_pct',
    
    # Stats ajustados (leakage)
    'home_ortg_adj', 'away_ortg_adj', 'home_drtg_adj', 'away_drtg_adj',
    'liga_ortg_avg', 'liga_drtg_avg',
    
    # Elo ratings (atualizados após cada jogo)
    'home_elo', 'away_elo', 'elo_diff',
    
    # Probabilidades (outputs)
    'prob_home', 'prob_away',
    
    # IDs e flags
    'home_team', 'away_team', 'win', 'opp_win', 'game_id', 'season'
]

OUTPUT_FILE = Path('data/models/best_hyperparameters_v6.json')

# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================
def load_optimization_data():
    """Carrega e prepara dados para otimização."""
    logger.info("📊 Carregando dados para otimização V6...")
    
    df = load_historical_data(seasons=ML_SEASONS)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calcular sample weights
    sample_weights = calculate_sample_weights(df, weight_config=ML_SAMPLE_WEIGHT_CONFIG)
    
    # Preparar features (exatamente como train_ensemble_v6.py)
    X = df.drop(columns=BASE_DROP_COLS, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    
    # Verificação de leakage
    if 'point_differential' in X.columns or 'pt_diff' in X.columns:
        raise ValueError("CRITICAL LEAKAGE: point_differential found in features!")
    
    y = (df['winner'] == 'HOME').astype(int)
    
    logger.info(f"✅ Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
    return X, y, sample_weights

# =============================================================================
# FUNÇÃO DE AVALIAÇÃO COMUM
# =============================================================================
def evaluate_model(model, X, y, sample_weights, n_splits=5):
    """
    Avalia modelo com TimeSeriesSplit usando Log Loss como métrica principal.
    
    Returns:
        Tuple[float, float, float]: (neg_log_loss, brier_score, accuracy)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    log_losses = []
    brier_scores = []
    accuracies = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weights[train_idx] if sample_weights is not None else None
        
        model.fit(X_train, y_train, sample_weight=w_train)
        
        # Probabilidades para Log Loss e Brier
        proba = model.predict_proba(X_val)[:, 1]
        preds = model.predict(X_val)
        
        log_losses.append(log_loss(y_val, proba))
        brier_scores.append(brier_score_loss(y_val, proba))
        accuracies.append(accuracy_score(y_val, preds))
    
    return -np.mean(log_losses), np.mean(brier_scores), np.mean(accuracies)

# =============================================================================
# OBJETIVOS DE OTIMIZAÇÃO PARA CADA MODELO
# =============================================================================

def objective_rf(trial, X, y, weights):
    """
    Random Forest - Espaço de busca agressivo.
    Foco: min_samples_leaf aumentado para reduzir ruído.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),  # AGRESSIVO
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    neg_ll, brier, acc = evaluate_model(model, X, y, weights)
    
    # Log secundárias para análise
    trial.set_user_attr('brier_score', brier)
    trial.set_user_attr('accuracy', acc)
    
    return neg_ll  # Otimiza Log Loss

def objective_xgb(trial, X, y, weights):
    """
    XGBoost - Foco em learning_rate baixo e regularização forte.
    Crucial após remoção de data leakage.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),  # BAIXO
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),  # L1 FORTE
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),  # L2 FORTE
        'gamma': trial.suggest_float('gamma', 0, 3.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    model = XGBClassifier(**params, use_label_encoder=False)
    neg_ll, brier, acc = evaluate_model(model, X, y, weights)
    
    trial.set_user_attr('brier_score', brier)
    trial.set_user_attr('accuracy', acc)
    
    return neg_ll

def objective_hist(trial, X, y, weights):
    """
    HistGradientBoosting - Foco em l2_regularization.
    Sklearn nativo, muito rápido.
    """
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.1, 5.0, log=True),  # CRUCIAL
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
        'max_bins': trial.suggest_int('max_bins', 128, 255),
        'random_state': 42
    }
    
    model = HistGradientBoostingClassifier(**params)
    neg_ll, brier, acc = evaluate_model(model, X, y, weights)
    
    trial.set_user_attr('brier_score', brier)
    trial.set_user_attr('accuracy', acc)
    
    return neg_ll

def objective_lgbm(trial, X, y, weights):
    """
    LightGBM - Regularização e learning_rate baixo.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 3.0, log=True),  # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0, log=True),  # L2
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    neg_ll, brier, acc = evaluate_model(model, X, y, weights)
    
    trial.set_user_attr('brier_score', brier)
    trial.set_user_attr('accuracy', acc)
    
    return neg_ll

def objective_extra(trial, X, y, weights):
    """
    ExtraTrees - Similar ao RF, com aleatoriedade extra.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 25),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = ExtraTreesClassifier(**params)
    neg_ll, brier, acc = evaluate_model(model, X, y, weights)
    
    trial.set_user_attr('brier_score', brier)
    trial.set_user_attr('accuracy', acc)
    
    return neg_ll

# =============================================================================
# ORQUESTRAÇÃO DA OTIMIZAÇÃO
# =============================================================================

def run_optimization(n_trials=50, timeout=None):
    """
    Executa otimização bayesiana para todos os modelos base.
    
    Args:
        n_trials: Número de trials por modelo
        timeout: Timeout em segundos por modelo (None = sem limite)
    
    Returns:
        dict: Melhores parâmetros para cada modelo
    """
    logger.info("="*80)
    logger.info("🚀 OTIMIZAÇÃO DE HIPERPARÂMETROS V6 - PÓS DATA LEAKAGE FIX")
    logger.info(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"🔢 Trials por modelo: {n_trials}")
    logger.info(f"⏱️  Timeout: {timeout}s" if timeout else "⏱️  Sem timeout")
    logger.info("="*80)
    
    # Carregar dados
    X, y, weights = load_optimization_data()
    
    best_params = {}
    metrics_summary = {}
    
    # Define modelos a otimizar com suas funções objetivo
    models = [
        ('rf', 'Random Forest', objective_rf),
        ('xgb', 'XGBoost', objective_xgb),
        ('hist', 'HistGradientBoosting', objective_hist),
        ('lgbm', 'LightGBM', objective_lgbm),
        ('extra', 'ExtraTrees', objective_extra),
    ]
    
    for model_key, model_name, objective_fn in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔧 Otimizando {model_name}...")
        logger.info(f"{'='*60}")
        
        study = optuna.create_study(
            direction='maximize',  # Maximizar neg_log_loss (menos negativo = melhor)
            study_name=f'V6_{model_key}'
        )
        
        study.optimize(
            lambda trial: objective_fn(trial, X, y, weights),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        # Extrair melhores parâmetros
        best_params[model_key] = study.best_params
        
        # Extrair métricas secundárias do melhor trial
        best_trial = study.best_trial
        metrics_summary[model_key] = {
            'neg_log_loss': study.best_value,
            'brier_score': best_trial.user_attrs.get('brier_score', 0),
            'accuracy': best_trial.user_attrs.get('accuracy', 0),
        }
        
        logger.info(f"✅ {model_name} - Best Neg Log Loss: {study.best_value:.4f}")
        logger.info(f"   Accuracy: {metrics_summary[model_key]['accuracy']*100:.2f}%")
        logger.info(f"   Brier Score: {metrics_summary[model_key]['brier_score']:.4f}")
        logger.info(f"   Best Params: {study.best_params}")
    
    # Salvar resultados
    output_data = {
        'optimized_at': datetime.now().isoformat(),
        'n_trials': n_trials,
        'metric': 'neg_log_loss',
        'cv_method': 'TimeSeriesSplit(n_splits=5)',
        'data_leakage_fix': True,
        'models': best_params,
        'metrics': metrics_summary
    }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"💾 Resultados salvos em: {OUTPUT_FILE}")
    logger.info(f"{'='*80}")
    
    # Resumo final
    logger.info("\n📊 RESUMO DA OTIMIZAÇÃO V6:")
    logger.info("-" * 50)
    for model_key, metrics in metrics_summary.items():
        logger.info(f"  {model_key.upper():6} | Acc: {metrics['accuracy']*100:5.2f}% | Log Loss: {-metrics['neg_log_loss']:.4f} | Brier: {metrics['brier_score']:.4f}")
    
    return best_params

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Otimização de Hiperparâmetros V6 - Pós Data Leakage Fix'
    )
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=50,
        help='Número de trials por modelo (default: 50)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout em segundos por modelo (default: sem limite)'
    )
    
    args = parser.parse_args()
    
    run_optimization(n_trials=args.n_trials, timeout=args.timeout)
