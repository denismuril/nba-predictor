"""
Ensemble Model V6 - Advanced Non-Linear Stacking
Baseado no V5, mas introduz:
1. HistGradientBoostingClassifier como novo modelo base.
2. XGBoost como Meta-Learner (substituindo LogisticRegression) para stacking não-linear.

Objetivo: Capturar padrões complexos na combinação dos modelos base.
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
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    StackingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Configuração de Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_pipeline.data_preparation import load_historical_data, prepare_data_for_training

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

# 🚨 SMOKE TEST: Desativado - Treinando com histórico completo
SMOKE_TEST = False


def temporal_train_calib_split(
    df: pd.DataFrame, 
    X: pd.DataFrame, 
    y: pd.Series, 
    sample_weights: np.ndarray, 
    calib_days: int = 30
) -> tuple:
    """
    Split temporal garantido para calibração.
    """
    # Garantir que date é datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    max_date = df['date'].max()
    calib_start = max_date - pd.Timedelta(days=calib_days)
    
    train_mask = df['date'] < calib_start
    calib_mask = df['date'] >= calib_start
    
    # Validar que temos dados suficientes
    train_count = train_mask.sum()
    calib_count = calib_mask.sum()
    
    if calib_count < 50:
        logger.warning(f"⚠️ Apenas {calib_count} jogos para calibração. Aumentando período...")
        # Fallback: usar 20% mais recentes
        calib_start = df['date'].quantile(0.8)
        train_mask = df['date'] < calib_start
        calib_mask = df['date'] >= calib_start
        train_count = train_mask.sum()
        calib_count = calib_mask.sum()
    
    train_min = df[train_mask]['date'].min()
    train_max = df[train_mask]['date'].max()
    calib_min = df[calib_mask]['date'].min()
    calib_max = df[calib_mask]['date'].max()
    
    logger.info(f"📅 Split Temporal para Calibração:")
    logger.info(f"   Treino: {train_min.date()} → {train_max.date()} ({train_count} jogos)")
    logger.info(f"   Calibração: {calib_min.date()} → {calib_max.date()} ({calib_count} jogos)")
    
    return (
        X[train_mask].copy(),
        X[calib_mask].copy(),
        y[train_mask].copy(),
        y[calib_mask].copy(),
        sample_weights[train_mask.values]
    )

def load_best_params():
    """Carrega hiperparâmetros otimizados do JSON."""
    params_path_v6 = Path('data/models/best_hyperparameters_v6.json')
    params_path_legacy = Path('data/models/best_hyperparameters.json')
    
    # Priorizar V6
    if params_path_v6.exists():
        logger.info("✅ Carregando hiperparâmetros V6")
        with open(params_path_v6) as f:
            data = json.load(f)
        models = data.get('models', {})
        return (
            models.get('rf', {}),
            models.get('xgb', {}),
            models.get('extra', {}),
            models.get('lgbm', {}),
            models.get('hist', {})
        )
    elif params_path_legacy.exists():
        logger.warning("⚠️ Usando hiperparâmetros legado (V5). Rode optimize_hyperparameters_v6.py!")
        with open(params_path_legacy) as f:
            params = json.load(f)
        return params.get('rf', {}), params.get('xgb', {}), {}, {}, {}
    else:
        logger.warning("⚠️ Nenhum arquivo de hiperparâmetros encontrado. Usando defaults.")
        
        conservative_rf = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 10}
        conservative_xgb = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05}
        conservative_extra = {'n_estimators': 100, 'max_depth': 5}
        conservative_lgbm = {'n_estimators': 100, 'max_depth': 5}
        conservative_hist = {'max_iter': 100, 'max_depth': 5}
        
        return (
            conservative_rf,
            conservative_xgb,
            conservative_extra,
            conservative_lgbm,
            conservative_hist
        )


def train_ensemble_model_v6():
    logger.info("="*80)
    logger.info("🚀 TREINANDO ENSEMBLE MODEL V6 (ADVANCED STACKING - ALLOWLIST SECURE)")
    if SMOKE_TEST:
        logger.warning("🔥 MODO SMOKE TEST ATIVADO")
    logger.info("="*80)

    # 1. Carregar Hiperparâmetros
    rf_params, xgb_params, extra_params, lgbm_params, hist_params = load_best_params()

    # 2. Carregar dados
    try:
        from ml_pipeline.data_cache import load_historical_data_cached
        df = load_historical_data_cached(seasons=ML_SEASONS)
        logger.info(f"✅ Dados carregados via CACHE: {len(df)} jogos")
    except ImportError:
        df = load_historical_data(seasons=ML_SEASONS, enable_player_features=True)
        logger.info(f"✅ Dados carregados (sem cache): {len(df)} jogos")

    # 2.1 Feature Engineering V2
    try:
        from ml_pipeline.feature_engineering_v2 import prepare_advanced_features_only
        df = prepare_advanced_features_only(df)
        logger.info("✅ Feature Engineering V2 Avançado aplicado!")
    except Exception as e:
        logger.warning(f"⚠️ Feature Pipeline V4 Modular falhou: {e}")

    df = df.sort_values('date').reset_index(drop=True)

    # 2.2 Sample weights
    from ml_pipeline.data_preparation import calculate_sample_weights
    sample_weights = calculate_sample_weights(df, weight_config=ML_SAMPLE_WEIGHT_CONFIG)

    if SMOKE_TEST:
        df = df.tail(500).reset_index(drop=True)
        sample_weights = sample_weights[-500:]

    # 3. SELEÇÃO DE FEATURES (Obrigatório: Whitelist com SAFE_PREFIXES)
    X, y_temp = prepare_data_for_training(df, target='winner')
    X = X.fillna(0)
    
    # 3. Pré-processamento Base - SEGURANÇA MÁXIMA (Allowlist)
    logger.info("🛡️ Aplicando Allowlist de Features (Anti-Leakage V2)...")

    # Prefixos PERMITIDOS (Dados conhecidos ANTES do jogo)
    SAFE_PREFIXES = [
        'rolling_',      # Médias móveis passadas
        'elo_',          # Elo Ratings pré-jogo
        'rest_',         # Dias de descanso
        'is_b2b',        # Flag de fadiga
        'feat_',         # Features calculadas explicitamente
        'encoded_'       # Variáveis categóricas tratadas
    ]

    # Colunas específicas permitidas
    SAFE_COLS = ['home_elo', 'away_elo', 'home_rest_days', 'away_rest_days']

    # Filtrar: Só manter se começar com prefixo seguro OU estiver na lista segura
    input_cols = [c for c in X.columns if any(c.startswith(p) for p in SAFE_PREFIXES) or c in SAFE_COLS]

    # Trava de segurança extra: Remover vazamentos óbvios se passarem pelo filtro
    final_features = [c for c in input_cols if not any(x in c for x in ['score', 'winner', 'pts', 'odds', 'correct'])]

    # Aplicar filtro
    X = X[final_features]
    logger.info(f"✅ Features seguras mantidas: {len(X.columns)}")

    # Salvar lista de features
    joblib.dump(final_features, 'data/models/feature_names_v6.joblib')

    y = (df['winner'] == 'HOME').astype(int)

    # 5. WALK-FORWARD VALIDATION
    logger.info("📊 Iniciando Walk-Forward Validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    # 6. Configuração dos Modelos
    rf = RandomForestClassifier(n_jobs=-1, random_state=42, **rf_params)
    xgb = XGBClassifier(n_jobs=-1, random_state=42, eval_metric='logloss', **xgb_params)
    extra = ExtraTreesClassifier(n_jobs=-1, random_state=42, **extra_params)
    lgbm = LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1, **lgbm_params)
    hist = HistGradientBoostingClassifier(random_state=42, **hist_params)

    base_estimators = [('rf', rf), ('xgb', xgb), ('extra', extra), ('lgbm', lgbm), ('hist', hist)]

    # 7. Meta-Learner
    meta_clf = LogisticRegression(solver='liblinear', penalty='l1', C=0.1, max_iter=1000, random_state=42)
    ensemble = StackingClassifier(estimators=base_estimators, final_estimator=meta_clf, cv=5, n_jobs=-1)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        weights_fold = sample_weights[train_idx]

        fold_ensemble = clone(ensemble)
        fold_ensemble.fit(X_fold_train, y_fold_train, sample_weight=weights_fold)
        fold_acc = fold_ensemble.score(X_fold_val, y_fold_val)
        cv_scores.append(fold_acc)
        logger.info(f"   Fold {fold}: {fold_acc*100:.2f}%")

    avg_cv_acc = np.mean(cv_scores)
    logger.info(f"📊 Acurácia Temporal Média: {avg_cv_acc*100:.2f}%")

    # 8. CALIBRAÇÃO (Isotonic)
    logger.info("🎯 Aplicando Calibração de Probabilidade...")
    X_train, X_calib, y_train, y_calib, weights_train = temporal_train_calib_split(
        df=df, X=X, y=y, sample_weights=sample_weights, calib_days=30
    )

    ensemble.fit(X_train, y_train, sample_weight=weights_train)
    
    # Fallback para versão antiga do scikit-learn se necessário
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated_ensemble = CalibratedClassifierCV(FrozenEstimator(ensemble), method='isotonic')
    except ImportError:
        calibrated_ensemble = CalibratedClassifierCV(ensemble, method='isotonic', cv='prefit')
        
    calibrated_ensemble.fit(X_calib, y_calib)
    calib_acc = calibrated_ensemble.score(X_calib, y_calib)
    
    logger.info(f"   Acurácia Final (Calibrada): {calib_acc*100:.2f}%")

    # 10. Salvar
    joblib.dump(calibrated_ensemble, 'data/models/ensemble_model_v6.joblib')
    logger.info("💾 Modelo V6 CALIBRADO salvo.")

    return calibrated_ensemble, calib_acc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_ensemble_model_v6()
