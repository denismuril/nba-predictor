"""
Spread Model - Prediz margem de vitória (point differential)

Target: home_score - away_score
Uso: Apostar em spreads (ex: Lakers -5.5)
"""
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from ml_pipeline.feature_engineering_v2 import prepare_features_v2, add_calendar_features, add_roster_features
from ml_pipeline.data_preparation import load_historical_data, add_rolling_features
from data.repositories.db_manager import get_db_manager
from data.scrapers.schedule_scraper import obter_schedule

logger = logging.getLogger(__name__)

def train_spread_model():
    """
    Treina modelo XGBoost para predizer margem de vitória.
    
    Returns:
        tuple: (model, mae, rmse)
    """
    
    logger.info("📊 Treinando Spread Model (Point Differential)...")
    
    # 1. Carregar dados e gerar features V2
    df = load_historical_data()
    df = prepare_features_v2(df)
    
    if df is None or df.empty:
        logger.error("❌ Sem dados para treinar spread model.")
        return None, 0, 0
    
    # 2. Preparar X e y
    # Target: point_differential (já criado em prepare_features_v2)
    target = 'point_differential'
    
    # Remover colunas não-feature
    drop_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 
                 'winner', 'pt_diff', 'point_differential', 'total_points', 
                 'prediction', 'correct', 'home_is_winner', 'away_is_winner', 
                 'game_id', 'season']
    
    # Manter apenas colunas numéricas em X
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df[target]
    
    # Salvar nomes das features para inferência
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'data/models/spread_feature_names.joblib')
    
    # 3. Time Series Split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 4. Treinar XGBoost
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        logger.warning("⚠️  XGBoost não disponível. Usando RandomForest...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    
    # 5. Avaliar
    y_pred = model.predict(X_test)
    
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    logger.info(f"✅ Spread Model treinado")
    logger.info(f"   MAE: {mae:.2f} pontos")
    logger.info(f"   RMSE: {rmse:.2f} pontos")
    
    if mae > 13.0:
        logger.warning(f"⚠️  MAE alto ({mae:.2f}). O modelo pode não ser lucrativo contra Vegas (Target < 13.0).")
    
    # Salvar
    joblib.dump(model, 'data/models/spread_model.joblib')
    logger.info("💾 Modelo salvo: data/models/spread_model.joblib")
    
    return model, mae, rmse

def predict_spreads_batch():
    """
    Gera previsões de spread para os jogos de hoje.
    Retorna DataFrame com [home_team, away_team, predicted_spread, home_score_pred, away_score_pred]
    """
    try:
        model = joblib.load('data/models/spread_model.joblib')
        feature_names = joblib.load('data/models/spread_feature_names.joblib')
    except FileNotFoundError:
        logger.warning("⚠️  Modelo de spread não encontrado. Treine primeiro.")
        return pd.DataFrame()

    # 1. Obter jogos de hoje
    today = datetime.now().strftime('%Y-%m-%d')
    schedule = obter_schedule(today)
    
    if not schedule:
        return pd.DataFrame()
        
    df_upcoming = pd.DataFrame(schedule)
    df_upcoming = df_upcoming.rename(columns={'home': 'home_team', 'away': 'away_team'})
    df_upcoming['date'] = pd.to_datetime(today)
    
    # 2. Carregar histórico para features
    db = get_db_manager()
    df_history = db.get_comprehensive_history()
    
    # Conversões e Limpeza Histórico
    if df_history is not None and not df_history.empty:
        df_history['date'] = pd.to_datetime(df_history['date'])
        numeric_cols = ['home_score', 'away_score', 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                       'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm']
        for col in numeric_cols:
            if col in df_history.columns:
                df_history[col] = pd.to_numeric(df_history[col], errors='coerce').fillna(0)
                
        # Combinar
        df_combined = pd.concat([df_history, df_upcoming], ignore_index=True)
    else:
        df_combined = df_upcoming
        
    # 3. Gerar Features V2 (Rolling + Calendar + Roster)
    # Rolling
    df_features = add_rolling_features(df_combined, windows=[5, 10])
    
    # Calendar
    df_features = add_calendar_features(df_features)
    
    # Roster Impact (Calculado agora para o jogo real)
    # Nota: add_roster_features usa cache e calcula para todos os times no DF.
    # Para eficiência, poderíamos filtrar apenas os times de hoje, mas a função espera o DF todo.
    # Vamos filtrar o DF para apenas os times relevantes antes de chamar roster features se quisermos otimizar,
    # mas add_roster_features itera sobre unique teams do DF.
    # Vamos passar apenas o slice de hoje para add_roster_features? Não, precisamos manter a estrutura.
    # A função add_roster_features calcula para todos os times únicos no DF.
    # Como df_combined tem histórico, isso vai demorar.
    # Melhor: Calcular roster impact apenas para os jogos de hoje e fazer merge.
    
    # Filtrar apenas hoje para roster impact
    df_today = df_features[df_features['date'] == pd.to_datetime(today)].copy()
    
    if df_today.empty:
        return pd.DataFrame()

    # Calcular Roster Impact manualmente para hoje
    from core.roster_manager import get_roster_impact
    logger.info("🏥 Calculando Roster Impact para jogos de hoje...")
    df_today['home_roster_impact'] = df_today['home_team'].apply(get_roster_impact)
    df_today['away_roster_impact'] = df_today['away_team'].apply(get_roster_impact)
    
    # 4. Alinhar Features
    # Garantir que todas as colunas de feature_names existam
    for col in feature_names:
        if col not in df_today.columns:
            df_today[col] = 0
            
    X = df_today[feature_names]
    
    # 5. Prever
    preds = model.predict(X)
    
    # 6. Formatar Saída
    results = df_today[['home_team', 'away_team']].copy()
    results['predicted_spread'] = preds
    
    # Estimativa de placar (usando total médio 220 ou vindo de outro modelo)
    avg_total = 220
    results['home_score_pred'] = (avg_total + preds) / 2
    results['away_score_pred'] = (avg_total - preds) / 2
    
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_spread_model()
