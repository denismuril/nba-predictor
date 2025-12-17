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
    
    AUDITORIA P1-B:
    - Usa data como critério em vez de índice
    - Garante que calibração usa apenas jogos mais recentes
    - Elimina risco de contaminação temporal
    
    Args:
        df: DataFrame com coluna 'date'
        X: Features
        y: Target
        sample_weights: Pesos de amostra
        calib_days: Dias mais recentes para calibração (default: 30)
        
    Returns:
        Tuple: (X_train, X_calib, y_train, y_calib, weights_train)
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
    """Carrega hiperparâmetros otimizados do JSON (V6 prioritário, fallback V5).
    
    FALLBACK CONSERVADOR: Se nenhum arquivo existir, retorna parâmetros
    conservadores para evitar overfitting (max_depth=5, regularização alta).
    """
    params_path_v6 = Path('data/models/best_hyperparameters_v6.json')
    params_path_legacy = Path('data/models/best_hyperparameters.json')
    
    # Priorizar V6 (otimizado pós data-leakage fix)
    if params_path_v6.exists():
        logger.info("✅ Carregando hiperparâmetros V6 (pós data-leakage fix)")
        with open(params_path_v6) as f:
            data = json.load(f)
        # V6 usa estrutura: {'models': {'rf': {...}, 'xgb': {...}, ...}}
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
        # 🛡️ FALLBACK CONSERVADOR: Parâmetros anti-overfitting
        # Math-Context: Sem hiperparâmetros otimizados, usamos configs conservadoras:
        # - max_depth=5: Limita complexidade da árvore (reduz overfitting)
        # - reg_alpha/reg_lambda: L1/L2 regularização (penaliza pesos extremos)
        # - min_samples_split=10: Exige mais amostras para dividir nós
        logger.warning("⚠️ Nenhum arquivo de hiperparâmetros encontrado.")
        logger.warning("🛡️ Usando parâmetros CONSERVADORES anti-overfitting!")
        
        conservative_rf = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        }
        conservative_xgb = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05,
            'reg_alpha': 0.5,  # L1 regularization
            'reg_lambda': 1.0  # L2 regularization
        }
        conservative_extra = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 10
        }
        conservative_lgbm = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0
        }
        conservative_hist = {
            'max_iter': 100,
            'max_depth': 5,
            'learning_rate': 0.05
        }
        
        return (
            conservative_rf,
            conservative_xgb,
            conservative_extra,
            conservative_lgbm,
            conservative_hist
        )


def train_ensemble_model_v6():
    logger.info("="*80)
    logger.info("🚀 TREINANDO ENSEMBLE MODEL V6 (ADVANCED STACKING)")
    if SMOKE_TEST:
        logger.warning("🔥 MODO SMOKE TEST ATIVADO: Usando apenas os últimos 500 jogos!")
    logger.info("="*80)

    # 1. Carregar Hiperparâmetros (V6 prioritário, fallback V5)
    rf_params, xgb_params, extra_params, lgbm_params, hist_params = load_best_params()

    # 2. Carregar dados (COM CACHE para evitar recálculo de features)
    # Math-Context: RAPM isola impacto individual, BPM aproxima via box scores
    try:
        from ml_pipeline.data_cache import load_historical_data_cached
        df = load_historical_data_cached(seasons=ML_SEASONS)
        logger.info(f"✅ Dados carregados via CACHE: {len(df)} jogos")
    except ImportError:
        df = load_historical_data(seasons=ML_SEASONS, enable_player_features=True)
        logger.info(f"✅ Dados carregados (sem cache): {len(df)} jogos")

    # 2.1 APLICAR FEATURE ENGINEERING V2 AVANÇADO (Pace, Matchup, Volatility, Shooting Luck)
    try:
        from ml_pipeline.feature_engineering_v2 import prepare_advanced_features_only
        df = prepare_advanced_features_only(df)
        logger.info("✅ Feature Engineering V2 Avançado (Steps 8-11) aplicado!")
    except Exception as e:
        logger.warning(f"⚠️ Feature Pipeline V4 Modular falhou: {e}")

    df = df.sort_values('date').reset_index(drop=True)

    # 2.2 RECRIAR sample_weights APÓS feature engineering
    from ml_pipeline.data_preparation import calculate_sample_weights
    sample_weights = calculate_sample_weights(df, weight_config=ML_SAMPLE_WEIGHT_CONFIG)
    logger.info(f"✅ Sample weights recriados: {len(sample_weights)} (alinhado com df: {len(df)})")

    # 🚨 SMOKE TEST LOGIC
    if SMOKE_TEST:
        df = df.tail(500).reset_index(drop=True)
        sample_weights = sample_weights[-500:]
        logger.warning(f"🔥 Dataset reduzido para {len(df)} jogos (Smoke Test)")

    # =========================================================================
    # 3. SELEÇÃO DE FEATURES (WHITELIST - Single Source of Truth)
    # =========================================================================
    # SECURITY FIX v2.0: Toda lógica de segurança está centralizada em
    # data_preparation.prepare_data_for_training() que usa abordagem WHITELIST.
    #
    # WHITELIST = Apenas features com prefixos seguros são permitidas:
    #   - rolling_*  : Médias móveis históricas (com shift aplicado)
    #   - elo_*      : Ratings Elo (snapshot PRE-jogo)
    #   - rest_*     : Dias de descanso (calculado antes do jogo)
    #   - interaction_* : Features derivadas de rolling
    #   - referee_*  : Stats de árbitros (expanding window histórico)
    #   - h2h_*      : Head-to-head histórico
    #
    # BLACKLIST está documentada em data_preparation.py:BLACKLISTED_FEATURES
    # Inclui: home_efg, closing_odds, line_movement, smart_money_signal, etc.
    #
    # 🛡️ POR QUE WHITELIST É MAIS SEGURO QUE BLACKLIST:
    # - Blacklist: Esquecer de adicionar nova coluna = vazamento
    # - Whitelist: Nova coluna só entra se explicitamente mapeada = seguro por padrão
    # =========================================================================

    # 4. Seleção de Features via função centralizada (Single Source of Truth)
    X, y_temp = prepare_data_for_training(df, target='winner')
    X = X.fillna(0)
    
    # =========================================================================
    # 🛡️ SECURITY FIX v22.1: SEGUNDA CAMADA DE PROTEÇÃO - ALLOWLIST POR PREFIXO
    # =========================================================================
    # AUDITORIA TÉCNICA: Reforçando segurança com filtragem explícita por prefixo.
    # Mesmo que prepare_data_for_training falhe em bloquear algo, esta camada
    # garante que APENAS colunas com prefixos seguros entrem no modelo.
    #
    # PREFIXOS SEGUROS (Allowlist):
    #   - feat_    : Features de engenharia explícitas
    #   - roll_    : Médias móveis (variação curta)
    #   - rolling_ : Médias móveis (variação completa)
    #   - elo_     : Dados Elo pré-jogo
    #   - rest_    : Dias de descanso
    #   - interaction_ : Features de interação (derivadas de rolling)
    #   - referee_ : Stats de árbitros históricos
    #   - h2h_     : Head-to-head histórico
    #   - home_rolling_, away_rolling_ : Rolling features por time
    #   - home_elo, away_elo : Elo por time
    #   - home_rest_, away_rest_ : Descanso por time
    #   - home_win_streak, away_win_streak : Sequências de vitória
    #   - home_sos_, away_sos_ : Strength of Schedule
    #   - home_net_rating_trend, away_net_rating_trend : Tendências históricas
    # =========================================================================
    
    SAFE_PREFIXES = (
        'feat_',        # Features de engenharia explícitas
        'roll_',        # Médias móveis (short form)
        'rolling_',     # Médias móveis (full form)
        'elo_',         # Elo ratings base
        'rest_',        # Descanso
        'interaction_', # Features de interação
        'referee_',     # Árbitros
        'h2h_',         # Head-to-head
    )
    
    # Colunas específicas permitidas (não seguem prefixo mas são seguras)
    SAFE_EXACT_COLS = {
        'home_elo', 'away_elo', 'elo_diff',
        'home_rest_days', 'away_rest_days', 'rest_diff',
        'home_is_back_to_back', 'away_is_back_to_back',
        'home_b2b', 'away_b2b',
        'home_games_in_7d', 'away_games_in_7d',
        'home_net_rating_trend', 'away_net_rating_trend',
        'home_sos_10', 'away_sos_10',
        'home_altitude_advantage', 'away_altitude_advantage',
        'home_win_streak', 'away_win_streak',
        'home_fatigue_score', 'away_fatigue_score',
        'home_distance_km', 'away_distance_km',
        'injury_impact_home', 'injury_impact_away', 'injury_impact_net',
    }
    
    # Padrões seguros (contidos em qualquer posição)
    SAFE_CONTAINS = (
        '_rolling_',    # home_rolling_10_points, away_rolling_5_efg
        '_elo',         # home_elo, away_elo
        '_rest_',       # variações de rest
        '_b2b',         # back-to-back
        '_sos_',        # strength of schedule
        '_streak',      # win_streak
        '_trend',       # tendências
    )
    
    def is_safe_column(col: str) -> bool:
        """Verifica se coluna é segura via allowlist."""
        # Check prefixo
        if any(col.startswith(prefix) for prefix in SAFE_PREFIXES):
            return True
        # Check exata
        if col in SAFE_EXACT_COLS:
            return True
        # Check padrão contido
        if any(pattern in col for pattern in SAFE_CONTAINS):
            return True
        return False
    
    # Aplicar filtro allowlist
    original_cols = X.columns.tolist()
    safe_cols = [col for col in original_cols if is_safe_column(col)]
    dropped_cols = [col for col in original_cols if col not in safe_cols]
    
    X = X[safe_cols]
    
    if dropped_cols:
        logger.warning(f"🛡️ ALLOWLIST FILTER: {len(dropped_cols)} colunas descartadas silenciosamente:")
        for col in dropped_cols[:10]:  # Mostrar apenas primeiras 10
            logger.warning(f"   ❌ {col}")
        if len(dropped_cols) > 10:
            logger.warning(f"   ... e mais {len(dropped_cols) - 10} colunas")
    
    # 🛡️ VALIDAÇÃO DE SEGURANÇA: Confirmar que features perigosas foram removidas
    DANGEROUS_FEATURES = [
        'home_efg', 'away_efg', 'home_pace', 'away_pace',  # Four Factors RAW
        'line_movement', 'smart_money_signal', 'closing_odds',  # Closing Odds
        'home_score', 'away_score', 'winner'  # Resultados
    ]
    leaked_features = [f for f in DANGEROUS_FEATURES if f in X.columns]
    if leaked_features:
        logger.error(f"🚨 VAZAMENTO DETECTADO! Features perigosas no treino: {leaked_features}")
        raise ValueError(f"Data Leakage: {leaked_features}")
    
    # Verificação final de segurança
    logger.info(f"✅ Features preparadas via ALLOWLIST: {len(X.columns)} colunas (de {len(original_cols)} originais)")
    logger.info(f"🛡️ Validação anti-leakage: PASSOU (0 features perigosas)")

    feature_names = X.columns.tolist()
    logger.info(f"✅ Features selecionadas dinamicamente: {len(feature_names)}")

    # Salvar nova lista de features
    joblib.dump(feature_names, 'data/models/feature_names_v6.joblib')
    logger.info("💾 Lista de features V6 salva em data/models/feature_names_v6.joblib")

    y = (df['winner'] == 'HOME').astype(int)

    # =========================================================================
    # 5. WALK-FORWARD VALIDATION (TimeSeriesSplit)
    # =========================================================================
    # Math-Context: TimeSeriesSplit garante que o modelo NUNCA veja dados do
    # futuro para prever o passado. Isso elimina data leakage temporal e produz
    # uma estimativa realista da acurácia que veremos em produção.
    #
    # Diferente de train_test_split aleatório que mistura datas, TimeSeriesSplit
    # mantém a ordem cronológica:
    # - Fold 1: Treina em Jan-Mar, testa em Abr
    # - Fold 2: Treina em Jan-Abr, testa em Mai
    # - ... e assim por diante
    # =========================================================================

    logger.info("📊 Iniciando Walk-Forward Validation (5 splits temporais)...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    # 6. Configuração dos Modelos Base (V6 - Usando parâmetros otimizados)

    # Random Forest (Tuned V6)
    rf_defaults = {'n_estimators': 200, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
    if rf_params:
        rf_defaults.update(rf_params)
    rf_defaults['random_state'] = 42
    rf_defaults['n_jobs'] = -1
    rf = RandomForestClassifier(**rf_defaults)

    # XGBoost (Tuned V6)
    xgb_defaults = {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
        'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'
    }
    if xgb_params:
        xgb_defaults.update(xgb_params)
    xgb_defaults['random_state'] = 42
    xgb_defaults['n_jobs'] = -1
    xgb = XGBClassifier(**xgb_defaults)

    # ExtraTrees (Tuned V6)
    extra_defaults = {'n_estimators': 200, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
    if extra_params:
        extra_defaults.update(extra_params)
    extra_defaults['random_state'] = 42
    extra_defaults['n_jobs'] = -1
    extra = ExtraTreesClassifier(**extra_defaults)

    # LGBM (Tuned V6)
    lgbm_defaults = {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    if lgbm_params:
        lgbm_defaults.update(lgbm_params)
    lgbm_defaults['random_state'] = 42
    lgbm_defaults['n_jobs'] = -1
    lgbm_defaults['verbose'] = -1
    lgbm = LGBMClassifier(**lgbm_defaults)

    # HistGradientBoosting (Tuned V6)
    hist_defaults = {'max_iter': 200, 'max_depth': 10, 'learning_rate': 0.05, 'random_state': 42}
    if hist_params:
        hist_defaults.update(hist_params)
    hist_defaults['random_state'] = 42
    hist = HistGradientBoostingClassifier(**hist_defaults)

    base_estimators = [
        ('rf', rf),
        ('xgb', xgb),
        ('extra', extra),
        ('lgbm', lgbm),
        ('hist', hist)
    ]

    # 7. Meta-Learner CALIBRADO (LogisticRegression com L1)
    # Math-Fix: L1 (Lasso) elimina features redundantes dos modelos base
    # Math-Fix: solver='liblinear' é obrigatório para penalty='l1'
    # Math-Fix: C=0.1 maximiza regularização para lidar com colinearidade severa
    meta_clf = LogisticRegression(
        solver='liblinear',
        penalty='l1',
        C=0.1,
        max_iter=1000,
        random_state=42
    )

    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    # Walk-Forward Cross-Validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        weights_fold = sample_weights[train_idx]

        # Clonar ensemble para cada fold (evita contaminação)
        fold_ensemble = clone(ensemble)
        fold_ensemble.fit(X_fold_train, y_fold_train, sample_weight=weights_fold)
        fold_acc = fold_ensemble.score(X_fold_val, y_fold_val)
        cv_scores.append(fold_acc)
        logger.info(f"   Fold {fold}/5: {fold_acc*100:.2f}%")

    avg_cv_acc = np.mean(cv_scores)
    std_cv_acc = np.std(cv_scores)
    logger.info(f"📊 Acurácia Temporal Média (5-Fold): {avg_cv_acc*100:.2f}% (±{std_cv_acc*100:.2f}%)")
    logger.info("   ⬆️ Esta é a 'Acurácia Real' esperada em produção (mais conservadora)")

    # =========================================================================
    # 8. CALIBRAÇÃO DE PROBABILIDADE (ISOTONIC REGRESSION)
    # =========================================================================
    # Math-Context: Kelly Criterion para Gestão de Banca
    # ------------------------------------------------
    # A fórmula de Kelly para apostas é: f* = (p*b - q) / b
    # onde:
    #   f* = fração ótima do bankroll a apostar
    #   p  = probabilidade REAL de sucesso (PRECISA SER CALIBRADA!)
    #   q  = 1 - p (probabilidade de perda)
    #   b  = odds decimais - 1 (lucro líquido por unidade apostada)
    #
    # PROBLEMA: Se o modelo diz "70% de confiança" mas na realidade só acerta
    # 55% das vezes nessa faixa de probabilidade, o Kelly vai recomendar
    # apostar uma fração MUITO maior do que deveria → RISCO DE RUÍNA.
    #
    # EXEMPLO NUMÉRICO:
    # - Odds: 2.00 (b=1), Modelo diz p=0.70, Kelly recomenda: (0.7*1-0.3)/1 = 40%
    # - Se p_real=0.55, a aposta ótima seria: (0.55*1-0.45)/1 = 10%
    # - Apostar 40% quando deveria ser 10% = OVERBET de 4x!
    #
    # SOLUÇÃO: Isotonic Regression mapeia probabilidades brutas do modelo para
    # frequências observadas. Isso garante que quando o modelo diz "70%",
    # historicamente ele realmente acertou ~70% das vezes nessa faixa.
    # =========================================================================

    logger.info("🎯 Aplicando Calibração de Probabilidade (Isotonic)...")

    # AUDITORIA P1-B: Split temporal garantido (últimos 30 dias para calibração)
    # Substitui split por índice que não garantia separação temporal
    X_train, X_calib, y_train, y_calib, weights_train = temporal_train_calib_split(
        df=df,
        X=X,
        y=y,
        sample_weights=sample_weights,
        calib_days=30
    )

    # Treinar ensemble base no conjunto de treino
    logger.info(f"🔄 Treinando Stack V6 ({len(base_estimators)} modelos)...")
    ensemble.fit(X_train, y_train, sample_weight=weights_train)

    # Avaliar modelo base (pré-calibração)
    raw_acc = ensemble.score(X_calib, y_calib)
    logger.info(f"   Acurácia Base (pré-calibração): {raw_acc*100:.2f}%")

    # Calibrar probabilidades com Isotonic Regression
    # method='isotonic': Mais flexível que sigmoid, melhor para n > 1000
    # FrozenEstimator: Substitui cv='prefit' deprecado (sklearn 1.6+)
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated_ensemble = CalibratedClassifierCV(
            FrozenEstimator(ensemble),
            method='isotonic'
        )
    except ImportError:
        # Fallback para sklearn < 1.6
        calibrated_ensemble = CalibratedClassifierCV(
            ensemble,
            method='isotonic',
            cv='prefit'
        )
    calibrated_ensemble.fit(X_calib, y_calib)

    # Verificar que calibração preserva accuracy (não deve mudar muito)
    calib_acc = calibrated_ensemble.score(X_calib, y_calib)
    logger.info(f"   Acurácia Calibrada: {calib_acc*100:.2f}%")

    # 9. Sumário Final
    logger.info("=" * 60)
    logger.info("🏆 SUMÁRIO V6 (Walk-Forward + Calibração)")
    logger.info("=" * 60)
    logger.info(f"   📊 Acurácia Temporal (CV 5-Fold): {avg_cv_acc*100:.2f}% ±{std_cv_acc*100:.2f}%")
    logger.info(f"   🎯 Acurácia Final (Calibrada):   {calib_acc*100:.2f}%")
    logger.info("   ✅ Probabilidades calibradas para uso com Kelly Criterion")

    # 10. Salvar modelo CALIBRADO V6
    # O objeto calibrated_ensemble MANTÉM os métodos .predict() e .predict_proba()
    # mas agora .predict_proba() retorna probabilidades CALIBRADAS
    joblib.dump(calibrated_ensemble, 'data/models/ensemble_model_v6.joblib')
    logger.info("💾 Modelo V6 CALIBRADO salvo em data/models/ensemble_model_v6.joblib")

    # Salvar métricas de treinamento
    training_metrics = {
        'cv_accuracy_mean': float(avg_cv_acc),
        'cv_accuracy_std': float(std_cv_acc),
        'calibrated_accuracy': float(calib_acc),
        'n_features': len(feature_names),
        'n_samples_train': len(X_train),
        'n_samples_calib': len(X_calib),
        'calibration_method': 'isotonic'
    }
    with open('data/models/training_metrics_v6.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    logger.info("📊 Métricas salvas em data/models/training_metrics_v6.json")

    return calibrated_ensemble, calib_acc

if __name__ == "__main__":
 # Configuração de Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# 🚨 FLIGHT SAFETY CHECK: MODO DE TESTE RÁPIDO
# Se True, treina com apenas 500 jogos para validar o pipeline sem esperar horas
    SMOKE_TEST = False
    train_ensemble_model_v6()
