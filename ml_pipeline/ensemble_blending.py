"""
Ensemble Blending V22.0 - Com Calibração de Probabilidade e Optuna

Math-Context: Probabilidades calibradas significam que quando o modelo
diz 70% de chance, ele acerta historicamente 70% das vezes.
Isso é CRÍTICO para cálculo de Expected Value (EV+) em apostas.

V22.0 Features:
- Calibração isotônica de probabilidade
- Otimização de hiperparâmetros com Optuna (ou RandomizedSearchCV fallback)
"""
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from ml_pipeline.data_preparation import (
    load_historical_data, add_rolling_features, add_advanced_features
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Threshold para usar Optuna vs RandomizedSearchCV
OPTUNA_MIN_SAMPLES = 1000


def optimize_xgb_optuna(X, y, n_trials: int = 30) -> dict:
    """
    Otimiza hiperparâmetros do XGBoost usando Optuna.

    Args:
        X: Features de treino
        y: Target
        n_trials: Número de trials do Optuna

    Returns:
        Dict com melhores hiperparâmetros
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("⚠️ Optuna não instalado. Usando defaults.")
        return {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05}

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"   🎯 XGB Best Accuracy: {study.best_value:.4f}")
    return study.best_params


def optimize_lgb_optuna(X, y, n_trials: int = 30) -> dict:
    """
    Otimiza hiperparâmetros do LightGBM usando Optuna.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("⚠️ Optuna não instalado. Usando defaults.")
        return {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05}

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"   🎯 LGB Best Accuracy: {study.best_value:.4f}")
    return study.best_params


def optimize_randomized_search(X, y, model_name: str) -> dict:
    """
    Fallback: Otimização via RandomizedSearchCV para datasets pequenos.
    """
    logger.info(f"   📊 Usando RandomizedSearchCV para {model_name}...")

    if model_name == 'xgb':
        model = xgb.XGBClassifier(random_state=42)
        param_dist = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        }
    else:  # lgb
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        param_dist = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [20, 31, 50, 70]
        }

    search = RandomizedSearchCV(
        model, param_dist, n_iter=20, cv=3,
        scoring='accuracy', random_state=42, n_jobs=-1
    )
    search.fit(X, y)

    logger.info(f"   🎯 {model_name.upper()} Best Accuracy: {search.best_score_:.4f}")
    return search.best_params_


def train_ensemble_blending(use_calibration: bool = True,
                            calibration_method: str = 'isotonic',
                            optimize_hyperparams: bool = True,
                            optuna_trials: int = 30):
    """
    Treina ensemble com blending, calibração e otimização de hiperparâmetros.

    V22.0: Adiciona calibração e otimização dinâmica via Optuna.

    Args:
        use_calibration: Se True, aplica calibração isotônica/sigmoid
        calibration_method: 'isotonic' ou 'sigmoid'
        optimize_hyperparams: Se True, otimiza XGB/LGB via Optuna
        optuna_trials: Número de trials do Optuna

    Returns:
        Dict com métricas finais
    """
    logger.info("🧪 Ensemble Blending V22.0 (Calibração + Optuna)")
    logger.info(f"   📏 Calibração: {use_calibration} ({calibration_method})")
    logger.info(f"   🔧 Otimização: {optimize_hyperparams}")

    # 1. Carregar Dados (com CACHE para evitar recálculo)
    try:
        from ml_pipeline.data_cache import load_historical_data_cached
        df = load_historical_data_cached(
            seasons=['2023-24', '2024-25', '2025-26'],
            apply_weights=False
        )
    except ImportError:
        # Fallback para método sem cache
        df = load_historical_data(
            seasons=['2023-24', '2024-25', '2025-26'],
            apply_weights=False
        )
        df = add_rolling_features(df)
        df = add_advanced_features(df)

    # Features
    try:
        features = joblib.load('data/models/feature_names_final.joblib')
    except Exception:
        logger.warning("⚠️ Features não encontradas. Usando numéricas.")
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclusions = ['home_score', 'away_score', 'total_points',
                      'winner', 'spread', 'actual_spread']
        features = [f for f in features if f not in exclusions]

    X = df[features].fillna(0)
    y = df['winner']

    # V22.0 FIX: Limpar valores None/NaN na coluna winner
    valid_mask = y.notna()
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        logger.warning(f"⚠️ Removendo {n_invalid} jogos com winner=None")
        X = X[valid_mask]
        y = y[valid_mask]

    # Converter winner para numérico (HOME=1, AWAY=0)
    if y.dtype == 'object':
        logger.info("   Convertendo winner: HOME=1, AWAY=0")
        y = y.map({'HOME': 1, 'AWAY': 0, 1: 1, 0: 0}).fillna(0).astype(int)
    else:
        y = y.astype(int)

    # Reset index para evitar KeyError no OOF predictions
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    logger.info(f"   📊 Dataset: {len(X)} jogos, {len(features)} features")

    # =========================================================================
    # V22.0: OTIMIZAÇÃO DE HIPERPARÂMETROS
    # =========================================================================
    if optimize_hyperparams:
        logger.info("🔧 Otimizando hiperparâmetros...")

        use_optuna = len(X) >= OPTUNA_MIN_SAMPLES
        method = "Optuna" if use_optuna else "RandomizedSearchCV"
        logger.info(f"   Método: {method} (n={len(X)})")

        if use_optuna:
            xgb_params = optimize_xgb_optuna(X, y, n_trials=optuna_trials)
            lgb_params = optimize_lgb_optuna(X, y, n_trials=optuna_trials)
        else:
            xgb_params = optimize_randomized_search(X, y, 'xgb')
            lgb_params = optimize_randomized_search(X, y, 'lgb')

        # Garantir random_state
        xgb_params['random_state'] = 42
        lgb_params['random_state'] = 42
        lgb_params['verbose'] = -1
    else:
        xgb_params = {'n_estimators': 100, 'max_depth': 6,
                      'learning_rate': 0.05, 'random_state': 42}
        lgb_params = {'n_estimators': 100, 'max_depth': 5,
                      'learning_rate': 0.05, 'random_state': 42, 'verbose': -1}

    # 2. Definir Modelos Base com hiperparâmetros otimizados
    models = {
        'rf': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'xgb': xgb.XGBClassifier(**xgb_params),
        'lgb': lgb.LGBMClassifier(**lgb_params)
    }

    # 3. Gerar Previsões OOF
    logger.info("🔄 Gerando previsões Out-of-Fold...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = pd.DataFrame(index=X.index)

    for name, model in models.items():
        logger.info(f"   Treinando {name}...")
        oof_col = f'pred_{name}'
        oof_preds[oof_col] = 0.0

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            model.fit(X_train, y_train)
            oof_preds.loc[val_idx, oof_col] = model.predict_proba(X_val)[:, 1]

    # 4. Meta-Learner
    logger.info("🧠 Treinando Meta-Learner...")
    meta_X = oof_preds
    meta_y = y

    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_X, meta_y)

    # Métricas antes da calibração
    meta_preds_uncal = meta_model.predict_proba(meta_X)[:, 1]
    acc_uncal = accuracy_score(meta_y, (meta_preds_uncal >= 0.5).astype(int))
    ll_uncal = log_loss(meta_y, meta_preds_uncal)
    brier_uncal = brier_score_loss(meta_y, meta_preds_uncal)

    logger.info(f"📊 ANTES da Calibração: Acc={acc_uncal:.2%}, Brier={brier_uncal:.4f}")

    # 5. Calibração
    if use_calibration:
        logger.info(f"🎯 Calibração ({calibration_method})...")
        calibrated_model = CalibratedClassifierCV(
            estimator=meta_model, method=calibration_method, cv='prefit'
        )
        calibrated_model.fit(meta_X, meta_y)

        meta_preds_cal = calibrated_model.predict_proba(meta_X)[:, 1]
        acc_cal = accuracy_score(meta_y, (meta_preds_cal >= 0.5).astype(int))
        ll_cal = log_loss(meta_y, meta_preds_cal)
        brier_cal = brier_score_loss(meta_y, meta_preds_cal)

        logger.info(f"📊 DEPOIS: Acc={acc_cal:.2%}, Brier={brier_cal:.4f}")

        try:
            prob_true, prob_pred = calibration_curve(meta_y, meta_preds_cal, n_bins=10)
            ece = np.mean(np.abs(prob_true - prob_pred))
            logger.info(f"   ECE: {ece:.4f}")
        except Exception:
            pass

        final_meta_model = calibrated_model
    else:
        final_meta_model = meta_model
        acc_cal, ll_cal, brier_cal = acc_uncal, ll_uncal, brier_uncal

    # 6. Salvar
    logger.info("💾 Salvando modelos...")
    final_models = {}
    for name, model in models.items():
        model.fit(X, y)
        final_models[name] = model

    joblib.dump(final_models, 'data/models/blending_base_models.joblib')
    joblib.dump(final_meta_model, 'data/models/blending_meta_model.joblib')

    if use_calibration:
        joblib.dump(calibrated_model, 'data/models/blending_calibrated_model.joblib')

    # Salvar hiperparâmetros otimizados
    if optimize_hyperparams:
        joblib.dump(
            {'xgb': xgb_params, 'lgb': lgb_params},
            'data/models/optimized_hyperparams.joblib'
        )
        logger.info("   ✅ Hiperparâmetros otimizados salvos")

    logger.info("✅ Ensemble Blending V22.0 concluído!")

    return {
        'accuracy': acc_cal,
        'log_loss': ll_cal,
        'brier_score': brier_cal,
        'calibrated': use_calibration,
        'optimized': optimize_hyperparams
    }


if __name__ == "__main__":
    train_ensemble_blending(
        use_calibration=True,
        calibration_method='isotonic',
        optimize_hyperparams=True,
        optuna_trials=30
    )


