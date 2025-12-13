"""
config/model_config.py

Single Source of Truth - Configuração Centralizada de Hiperparâmetros
=====================================================================

Este arquivo contém TODOS os hiperparâmetros usados nos modelos de ML.
Qualquer script de treino, backtest ou inferência deve importar daqui.

REGRA DE OURO: O modelo treinado em produção DEVE ser idêntico ao validado no backtest.
"""

# =============================================================================
# RANDOM FOREST - Parâmetros Conservadores
# =============================================================================
# Fonte: ml_pipeline/backtesting.py (max_depth=8 = conservador)
# CUIDADO: train_model.py usava max_depth=None (divergência corrigida aqui)

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,             # Conservador para evitar overfitting
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}


# =============================================================================
# XGBOOST - Parâmetros do Ensemble V6
# =============================================================================
# Fonte: ml_pipeline/train_ensemble_v6.py (linha 335-338)

XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}


# =============================================================================
# LIGHTGBM - Parâmetros do Ensemble V6
# =============================================================================
# Fonte: ml_pipeline/train_ensemble_v6.py (linha 354-357)

LGBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}


# =============================================================================
# EXTRA TREES - Parâmetros do Ensemble V6
# =============================================================================
# Fonte: ml_pipeline/train_ensemble_v6.py (linha 346)

EXTRA_TREES_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}


# =============================================================================
# HIST GRADIENT BOOSTING - Parâmetros do Ensemble V6
# =============================================================================
# Fonte: ml_pipeline/train_ensemble_v6.py (linha 366)

HIST_GB_PARAMS = {
    'max_iter': 200,
    'max_depth': 10,
    'learning_rate': 0.05,
    'random_state': 42
}


# =============================================================================
# MÉDIAS DA LIGA NBA 2025-26 (Fallback para Imputação)
# =============================================================================
# Fonte: ml_pipeline/feature_engineering_v2.py (LEAGUE_DEFAULTS)
# Atualizado: Dezembro 2025 (NBA.com/stats e Basketball-Reference)
#
# IMPORTANTE: Usar médias em vez de zeros evita viés em modelos de árvore.
# Um time sem histórico NÃO tem "habilidade zero", tem habilidade MÉDIA.

LEAGUE_AVERAGES = {
    'efg_pct': 0.550,       # Effective Field Goal %
    'ts_pct': 0.587,        # True Shooting %
    'tov_pct': 0.132,       # Turnover %
    'oreb_pct': 0.235,      # Offensive Rebound %
    'ft_rate': 0.255,       # Free Throw Rate
    'off_rating': 117.5,    # Offensive Rating (pts/100 posses)
    'def_rating': 117.5,    # Defensive Rating (pts/100 posses)
    'pace': 100.8,          # Pace (posses/jogo)
    'pie': 0.100,           # Player Impact Estimate (normalizado)
    'pts': 117.2,           # Pontos por jogo
    'win': 0.5              # Taxa de vitória (definição)
}


# =============================================================================
# SAMPLE WEIGHT CONFIG - Ponderação Temporal
# =============================================================================
# Jogos recentes têm peso maior no treinamento

SAMPLE_WEIGHT_CONFIG = {
    'enabled': True,
    'recent_30_days': 3.0,
    'recent_60_days': 2.0,
    'recent_90_days': 1.5,
    'default': 1.0
}


# =============================================================================
# TEMPORADAS VÁLIDAS PARA ML
# =============================================================================

ML_SEASONS = ['2023-24', '2024-25', '2025-26']


# =============================================================================
# BACKTEST CONFIG - Sincronizado com Produção
# =============================================================================
# REGRA: Backtest e treinamento DEVEM usar os mesmos parâmetros
# para garantir que a performance validada = performance real

BACKTEST_CONFIG = {
    'test_size': 0.2,           # 20% dos dados para teste
    'cv_folds': 5,              # Cross-validation k-folds
    'random_state': 42,         # Seed para reprodutibilidade
    'use_sample_weights': True,  # Ponderar jogos recentes
    'test_days': 14,            # Dias para validação temporal
}

