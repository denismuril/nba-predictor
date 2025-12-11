"""
Enhanced Spread Model - v16.0

Melhoria do modelo de spread usando P2.2 features e Multi-API data.

Features:
- Regression para margem de vitória
- MAE optimization
- P2.2 15/15 features
- Dados 100% reais

Usage:
    python ml_pipeline/train_spread_enhanced.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data():
    """Carrega dados de treino com P2.2 features."""
    logger.info("📂 Carregando dados de treino...")
    
    try:
        # Load from prepared dataset
        df = pd.read_csv('data/prepared_games.csv')
        logger.info(f"✅ {len(df)} games carregados")
        return df
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados: {e}")
        return None


def prepare_spread_features(df):
    """Prepara features específicas para spread prediction."""
    logger.info("🔧 Preparando features para spread...")
    
    # P2.2 differential features (Corrected names)
    spread_features = [
        # Matchup features
        'pace_differential',
        'def_matchup_net',
        'total_reb_edge',
        'three_pt_matchup_gap',
        'tov_pressure_net',
        
        # Situational
        'clutch_differential',
        'playoff_desperation_gap',
        'ts_pct_differential',
        'ast_tov_ratio_gap',
        'injury_impact_net',
        
        # Advanced
        'schedule_density_gap',
        'travel_fatigue_net',
        # 'fastbreak_diff_norm', # Disabled
        # 'paint_diff_norm', # Disabled
        # 'second_chance_diff_norm', # Disabled
        
        # Basic stats
        'pts_diff',
        'fg_pct_diff',
        'three_pct_diff'
    ]
    
    # Filter existing columns
    available_features = [f for f in spread_features if f in df.columns]
    
    logger.info(f"✅ {len(available_features)} features disponíveis")
    
    return available_features


def train_spread_model(X, y):
    """Treina modelo de spread otimizado."""
    logger.info("\n🤖 Treinando Spread Model...")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model - Gradient Boosting otimizado para MAE
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        loss='absolute_error',  # MAE optimization
        random_state=42
    )
    
    # Train
    logger.info("  Training...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logger.info(f"\n📊 Resultados:")
    logger.info(f"   MAE: {mae:.2f} pontos")
    logger.info(f"   RMSE: {rmse:.2f} pontos")
    logger.info(f"   Train samples: {len(X_train)}")
    logger.info(f"   Test samples: {len(X_test)}")
    
    # Cross-validation
    logger.info("\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(
        model, X, y, 
        cv=5, 
        scoring='neg_mean_absolute_error'
    )
    cv_mae = -cv_scores.mean()
    
    logger.info(f"   CV MAE: {cv_mae:.2f} ± {cv_scores.std():.2f}")
    
    # Feature importance
    logger.info(f"\n📈 Top 10 Features:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        logger.info(f"   {i}. {X.columns[idx]}: {importances[idx]:.4f}")
    
    return model, mae, cv_mae


def save_model(model, mae):
    """Salva modelo treinado."""
    model_path = Path('models/spread_model_v16.joblib')
    model_path.parent.mkdir(exist_ok=True, parents=True)
    
    joblib.dump({
        'model': model,
        'mae': mae,
        'version': 'v16.0',
        'features': 'P2.2 15/15 + Multi-API'
    }, model_path)
    
    logger.info(f"\n💾 Modelo salvo: {model_path}")
    logger.info(f"   MAE: {mae:.2f} pontos")


def predict_spreads_batch(df_games):
    """
    Gera previsões de spread para um lote de jogos.
    
    Args:
        df_games: DataFrame com dados dos jogos (deve ter features calculadas)
        
    Returns:
        DataFrame com colunas 'home_team', 'away_team', 'predicted_spread'
    """
    try:
        # Carregar modelo
        model_path = Path('models/spread_model_v16.joblib')
        if not model_path.exists():
            logger.warning("⚠️ Modelo Spread v16.0 não encontrado.")
            return pd.DataFrame()
            
        saved_data = joblib.load(model_path)
        model = saved_data['model']
        
        # Preparar features
        features = prepare_spread_features(df_games)
        
        if not features:
            logger.warning("⚠️ Features insuficientes para predição de spread.")
            return pd.DataFrame()
            
        # Garantir que todas as features existam (fill 0 se faltar)
        X = df_games[features].fillna(0)
        
        # Predict
        spreads = model.predict(X)
        
        # Formatar resultado
        results = df_games[['home_team', 'away_team']].copy()
        results['predicted_spread'] = spreads
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro na predição de spread: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    logger.info("🏀 Spread Model Training - v16.0\n")
    logger.info("="*60)
    
    # Load data
    df = load_training_data()
    
    if df is not None and len(df) > 100:
        # Prepare features
        features = prepare_spread_features(df)
        
        if features:
            # Calculate target (point differential)
            if 'home_score' in df.columns and 'away_score' in df.columns:
                df['spread'] = df['home_score'] - df['away_score']
                
                X = df[features]
                y = df['spread']
                
                # Train
                model, mae, cv_mae = train_spread_model(X, y)
                
                # Save
                save_model(model, mae)
                
                print(f"\n✅ Spread Model v16.0 completo!")
                print(f"   MAE: {mae:.2f} pontos")
                print(f"   CV MAE: {cv_mae:.2f} pontos")
            else:
                logger.error("❌ Colunas home_score/away_score não encontradas")
        else:
            logger.error("❌ Nenhuma feature disponível")
    else:
        logger.warning("⚠️ Dados insuficientes para treino")
        logger.info("💡 Rode data preparation primeiro")
