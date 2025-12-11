"""
Train Totals Model V18 (Pipeline V4)

Uses the new Feature Pipeline V4 (Pace, Matchups, Volatility) to train an improved Totals model.
"""

import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ml_pipeline.data_preparation import load_historical_data
from ml_pipeline.feature_pipeline_v4 import prepare_features_v4
from utils.logger_config import get_logger

logger = get_logger(__name__)

def train_totals_v18():
    logger.info("🚀 Iniciando treinamento do Modelo Totals V18 (Pipeline V4)...")
    
    # 1. Carregar Dados (RAW)
    logger.info("📊 Carregando dados históricos (RAW)...")
    df = load_historical_data(raw=True)
    
    # 2. Feature Engineering (V4)
    logger.info("🔧 Executando Feature Pipeline V4...")
    df_features = prepare_features_v4(df)
    
    # 3. Preparar Dataset
    # Remover jogos sem target (total_points)
    df_train = df_features.dropna(subset=['total_points']).copy()
    
    # Ordenar por data
    df_train = df_train.sort_values('date').reset_index(drop=True)
    
    # Definir Features
    # Usar todas as features numéricas geradas, exceto IDs e Targets
    # Whitelist de prefixos para garantir que só usamos features calculadas
    valid_prefixes = [
        'home_rolling_', 'away_rolling_', 
        'home_rapm_', 'away_rapm_',
        'home_bpm_', 'away_bpm_',
        'home_rest_', 'away_rest_',
        'home_b2b', 'away_b2b',
        'h2h_',
        'season_', 'is_weekend',
        # V4 New Features
        'projected_pace', 'pace_mismatch',
        'off_matchup_', 'eff_sum', 'def_sum',
        'home_scoring_std_', 'away_scoring_std_'
    ]
    
    feature_cols = [
        c for c in df_train.columns 
        if any(c.startswith(p) for p in valid_prefixes)
        and c not in ['total_points', 'home_score', 'away_score', 'winner', 'plus_minus']
    ]
    
    # Remover features com importância zero (da análise anterior), se desejado
    # Por enquanto, mantemos tudo para testar o V4
    
    X = df_train[feature_cols].fillna(0)
    y = df_train['total_points']
    
    logger.info(f"📚 Dataset de Treino: {X.shape[0]} jogos, {X.shape[1]} features")
    
    # 4. Configurar Modelo (XGBoost)
    # Usar melhores parâmetros do V3 como base
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:absoluteerror',
        'n_jobs': -1,
        'random_state': 42
    }
    
    model = XGBRegressor(**params)
    
    # 5. Validação Cruzada (TimeSeriesSplit)
    logger.info("🔄 Executando TimeSeriesSplit (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        preds = model.predict(X_v)
        mae = mean_absolute_error(y_v, preds)
        maes.append(mae)
        logger.info(f"   Fold {fold+1}: MAE = {mae:.4f}")
        
    avg_mae = np.mean(maes)
    logger.info(f"📉 MAE Médio (CV): {avg_mae:.4f}")
    
    # 6. Treinar Modelo Final (Todo o Dataset)
    logger.info("🏆 Treinando modelo final...")
    model.fit(X, y, verbose=False)
    
    # 7. Salvar Modelo e Features
    models_dir = Path('data/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, models_dir / 'totals_model_v18.joblib')
    joblib.dump(feature_cols, models_dir / 'totals_feature_names_v18.joblib')
    
    logger.info(f"✅ Modelo salvo em: {models_dir / 'totals_model_v18.joblib'}")
    
    # 8. Feature Importance (Top 10)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n🔝 Top 10 Features V4:")
    print(importance.head(10))
    
    # Verificar se as novas features estão sendo usadas
    v4_features = [c for c in feature_cols if c in [
        'projected_pace', 'pace_mismatch', 'off_matchup_home', 'off_matchup_away', 
        'eff_sum', 'def_sum', 'home_scoring_std_10', 'away_scoring_std_10'
    ]]
    
    v4_importance = importance[importance['feature'].isin(v4_features)]
    logger.info("\n🧪 Importância das Novas Features V4:")
    print(v4_importance)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_totals_v18()
