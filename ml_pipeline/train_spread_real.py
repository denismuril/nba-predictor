"""
Treina Spread Model com dados REAIS da NBA API (2022-2024).
Evita database locks consultando diretamente a NBA API.
Dataset robusto: ~2,500+ jogos para XGBoost.
"""
import joblib
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ml_pipeline.data_preparation import load_historical_data

# Aliases para compatibilidade com o otimizador
# load_historical_data já importado acima

def prepare_features_v2(df):
    """
    Prepara features para o modelo de spread.
    Assume que df já vem com rolling features do load_historical_data.
    """
    # Features já calculadas no load_historical_data (V13)
    # Precisamos apenas selecionar as colunas relevantes
    # Target: Point Differential (Home - Away)
    # Se Home ganha por 10, diff = 10. Se perde por 5, diff = -5.
    y = df['home_score'] - df['away_score']
    
    # Remover colunas não-feature
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'prob_home', 'prob_away', 'game_id', 'id',
                 # Colunas de texto (object) que XGBoost não aceita
                 'season', 'date_str',
                 # REMOVER LEAKAGE
                 'pts', 'opp_pts',
                 'home_off_rating', 'home_def_rating', 'home_efg_pct', 'home_ts_pct', 'home_pace', 'home_pie',
                 'away_off_rating', 'away_def_rating', 'away_efg_pct', 'away_ts_pct', 'away_pace', 'away_pie',
                 'ast', 'opp_ast', 'reb', 'opp_reb', 'tov', 'opp_tov', 'stl', 'opp_stl', 'blk', 'opp_blk',
                 'pf', 'opp_pf', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_fg3a', 'opp_ftm', 'opp_fta',
                 'oreb', 'dreb', 'opp_oreb', 'opp_dreb']
                 
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # One-Hot Encoding para times (importante para spread)
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    
    return X, y

def train_spread_model_real():
    """Treina modelo de spread com dados históricos e hiperparâmetros otimizados."""
    logger.info("="*80)
    logger.info("🚀 TREINANDO SPREAD MODEL (V13 Enhanced)")
    logger.info("="*80)
    
    # Carregar dados centralizados (limpos e com rolling features)
    df, weights = load_historical_data(
        seasons=['2023-24', '2024-25', '2025-26'], 
        apply_weights=True
    )
    
    if df is None or df.empty: return None, 0, 0
    
    X, y = prepare_features_v2(df)
    
    # Split temporal
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # --- Configuração do Modelo ---
    PARAMS_FILE = os.path.join("data", "models", "best_hyperparameters.joblib")
    
    if os.path.exists(PARAMS_FILE):
        logger.info(f"💎 Carregando hiperparâmetros OTIMIZADOS de {PARAMS_FILE}...")
        best_params = joblib.load(PARAMS_FILE)
        best_params['n_jobs'] = -1
        best_params['random_state'] = 42
        best_params['objective'] = 'reg:absoluteerror'
        model = XGBRegressor(**best_params)
    else:
        logger.warning("⚠️  Arquivo de hiperparâmetros não encontrado. Usando DEFAULT (Hardcoded).")
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            objective='reg:absoluteerror'
        )

    logger.info("🏋️‍♂️ Treinando modelo final (com sample weights)...")
    # Usar pesos para o treino (suporta Series ou Array)
    if hasattr(weights, 'iloc'):
        sample_weights_train = weights.iloc[:split_idx]
    else:
        sample_weights_train = weights[:split_idx]
        
    model.fit(X_train, y_train, sample_weight=sample_weights_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    logger.info(f"✅ MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    
    models_dir = Path('data/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / 'spread_model.joblib')
    joblib.dump(list(X.columns), models_dir / 'spread_feature_names.joblib')
    
    return model, mae, rmse

    return model, mae, rmse

def predict_spread(home_team, away_team, features_df):
    """
    Prediz o spread para um único jogo.
    features_df deve conter as colunas esperadas pelo modelo.
    """
    try:
        model_path = Path('data/models/spread_model.joblib')
        if not model_path.exists():
            logger.warning("⚠️ Modelo de Spread não encontrado.")
            return None
            
        model = joblib.load(model_path)
        
        # Garantir que features_df tem as colunas certas (One-Hot Encoding)
        # Isso é complexo em tempo de inferência pois precisamos das mesmas colunas de treino
        # Simplificação: Assumir que features_df já vem preparado ou usar modelo sem OHE para times
        # Melhor abordagem: O modelo usa OHE, então precisamos recriar a estrutura
        
        # Carregar nomes das features de treino
        feature_names = joblib.load('data/models/spread_feature_names.joblib')
        
        # Ajustar features_df para ter todas as colunas, preenchendo com 0
        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
                
        # Garantir ordem
        X = features_df[feature_names]
        
        pred = model.predict(X)[0]
        return pred
    except Exception as e:
        logger.error(f"Erro ao prever spread: {e}")
        return None

def predict_spreads_batch(games_df):
    """
    Prediz spreads para vários jogos.
    games_df deve ter colunas: home_team, away_team, e as rolling features.
    """
    try:
        model_path = Path('data/models/spread_model.joblib')
        if not model_path.exists():
            return pd.DataFrame()
            
        model = joblib.load(model_path)
        feature_names = joblib.load('data/models/spread_feature_names.joblib')
        
        # Preparar features (One-Hot Encoding)
        X = pd.get_dummies(games_df, columns=['home_team', 'away_team'], drop_first=False)
        
        # Alinhar colunas
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]
        
        preds = model.predict(X)
        
        results = games_df[['home_team', 'away_team']].copy()
        results['predicted_spread'] = preds
        return results
        
    except Exception as e:
        logger.error(f"Erro batch spread: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    train_spread_model_real()
