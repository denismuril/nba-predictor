"""
Train Totals Model (Over/Under) v18.0 - Com Volatilidade de Pace

Treina um modelo de Gradient Boosting para prever o total de pontos (Home + Away).
Otimizado para MAE (Mean Absolute Error).

NOVO v18: Inclui features de volatilidade e tendência de Pace
Math-Fix: Pace dinâmico em vez de constante fixa
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import logging
from pathlib import Path
import sys

# Adicionar root do projeto ao path para imports funcionarem
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuração de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Math-Fix: Fallback apenas - será substituído por cálculo dinâmico
LEAGUE_AVG_PACE_FALLBACK = 99.0


def calculate_dynamic_league_pace(df, window=100):
    """
    Calcula a média de Pace da liga de forma dinâmica (expanding mean).
    Evita usar dados do futuro calculando a média dos jogos anteriores.
    
    Math-Fix: Isso permite que o modelo se adapte quando a liga fica
    mais rápida (início da temporada ~101) ou mais lenta (playoffs ~96).
    """
    df_sorted = df.sort_values('date').copy()
    
    # Pace médio por jogo = (home_pace + away_pace) / 2
    if 'home_rolling_10_pace' in df_sorted.columns and 'away_rolling_10_pace' in df_sorted.columns:
        avg_pace_proxy = (df_sorted['home_rolling_10_pace'] + df_sorted['away_rolling_10_pace']) / 2
        
        # Expanding mean para simular "o que sabíamos sobre a liga até hoje"
        league_pace_dynamic = avg_pace_proxy.expanding(min_periods=50).mean()
        
        # Preencher início com fallback
        league_pace_dynamic = league_pace_dynamic.fillna(LEAGUE_AVG_PACE_FALLBACK)
        
        logger.info(f"📈 Pace Dinâmico: min={league_pace_dynamic.min():.2f}, "
                    f"max={league_pace_dynamic.max():.2f}, last={league_pace_dynamic.iloc[-1]:.2f}")
        
        return league_pace_dynamic
    
    logger.warning("⚠️ Rolling pace não disponível, usando fallback")
    return pd.Series([LEAGUE_AVG_PACE_FALLBACK] * len(df), index=df.index)


def train_totals_model():
    logger.info("🏀 Totals Model Training - v18.0 (Com Volatilidade de Pace)")
    logger.info("="*60)
    
    # 1. Carregar Dados via pipeline (usa CACHE para evitar recálculo)
    try:
        from ml_pipeline.data_cache import load_historical_data_cached
        df = load_historical_data_cached(seasons=['2023-24', '2024-25', '2025-26'])
        if df is None or df.empty:
            raise ValueError("Nenhum dado retornado")
        logger.info(f"✅ {len(df)} games carregados via CACHE")
    except ImportError:
        # Fallback sem cache
        from ml_pipeline.data_preparation import load_historical_data
        df = load_historical_data(seasons=['2023-24', '2024-25', '2025-26'])
        if df is None or df.empty:
            raise ValueError("Nenhum dado retornado")
        logger.info(f"✅ {len(df)} games carregados (sem cache)")
    except Exception as e:
        logger.warning(f"⚠️ Fallback para CSV: {e}")
        try:
            df = pd.read_csv('data/prepared_games.csv')
            logger.info(f"✅ {len(df)} games carregados (CSV fallback)")
        except FileNotFoundError:
            logger.error("❌ Nenhuma fonte de dados disponível.")
            return

    # 2. Math-Fix: Calcular Pace Dinâmico da Liga
    logger.info("📈 Calculando Pace Dinâmico da Liga...")
    df['league_avg_pace_dynamic'] = calculate_dynamic_league_pace(df)
    
    # Calcular projected_pace_vegas com pace dinâmico
    if 'home_rolling_10_pace' in df.columns and 'away_rolling_10_pace' in df.columns:
        # Math-Fix: (Home_Pace * Away_Pace) / League_Avg modela interação não-linear
        df['projected_pace_vegas'] = (
            df['home_rolling_10_pace'] * df['away_rolling_10_pace']
        ) / df['league_avg_pace_dynamic']
        logger.info("✅ projected_pace_vegas calculado (fórmula Vegas com Pace Dinâmico)")
    
    # 3. Lista de features SEGURAS para Totals (SEM DATA LEAKAGE)
    totals_features = [
        # === Pace Features (Core) ===
        'projected_pace_vegas',  # Math-Fix: Nova feature Vegas
        'expected_pace_10', 'expected_pace_5', 'pace_differential',
        'home_rolling_10_pace', 'away_rolling_10_pace',
        'home_rolling_5_pace', 'away_rolling_5_pace',
        
        # === NOVO v18: Volatilidade de Pace ===
        'home_rolling_10_pace_std', 'away_rolling_10_pace_std',
        'home_rolling_5_pace_std', 'away_rolling_5_pace_std',
        'home_pace_trend_10', 'away_pace_trend_10',
        'home_pace_trend_5', 'away_pace_trend_5',
        'home_rolling_10_points_std', 'away_rolling_10_points_std',
        'home_rolling_5_points_std', 'away_rolling_5_points_std',
        
        # === Rolling Scoring ===
        'home_rolling_10_points', 'away_rolling_10_points',
        'home_rolling_5_points', 'away_rolling_5_points',
        
        # === Rolling Efficiency ===
        'home_rolling_10_efg', 'away_rolling_10_efg',
        'home_rolling_5_efg', 'away_rolling_5_efg',
        
        # === Rolling Four Factors ===
        'home_rolling_10_tov_pct', 'away_rolling_10_tov_pct',
        'home_rolling_10_orb_pct', 'away_rolling_10_orb_pct',
        'home_rolling_10_ftr', 'away_rolling_10_ftr',
        
        # === Metadata ===
        'rest_days_home', 'rest_days_away',
        'home_rolling_10_win', 'away_rolling_10_win'
    ]
    
    # Verificar quais features existem
    available_features = [f for f in totals_features if f in df.columns]
    
    logger.info(f"🔧 Features disponíveis: {len(available_features)}/{len(totals_features)}")
    
    if not available_features:
        logger.error("❌ Nenhuma feature disponível para treino.")
        return
    
    # Log de features novas (volatilidade)
    volatility_features = [f for f in available_features if 'std' in f or 'trend' in f]
    if volatility_features:
        logger.info(f"✅ Features de volatilidade encontradas: {len(volatility_features)}")
    else:
        logger.warning("⚠️ Nenhuma feature de volatilidade encontrada (execute data_preparation com pace_volatility)")

    # 3. Preparar X e y
    if 'total_points' not in df.columns:
        df['total_points'] = df['home_score'] + df['away_score']
        
    df_clean = df.dropna(subset=available_features + ['total_points'])
    
    X = df_clean[available_features]
    y = df_clean['total_points']
    
    logger.info(f"📊 Dataset: {len(X)} jogos | {len(available_features)} features")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Treinar Modelo (Gradient Boosting)
    logger.info("🤖 Treinando Gradient Boosting Regressor...")
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        loss='absolute_error',  # Otimizar MAE diretamente
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 5. Avaliação
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info("="*60)
    logger.info("📊 RESULTADOS (TOTALS v18)")
    logger.info("="*60)
    logger.info(f"MAE: {mae:.2f} pontos")
    logger.info(f"R²:  {r2:.4f}")
    logger.info("="*60)
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("🏆 Top 10 Features:")
    for idx, row in importance.head(10).iterrows():
        logger.info(f"   {row['feature']:<30} {row['importance']:.4f}")

    # 6. Salvar Modelo
    output_dir = Path('data/models')
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / 'totals_model_v18.joblib'
    features_path = output_dir / 'totals_feature_names_v18.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(available_features, features_path)
    
    logger.info(f"\n💾 Modelo salvo em: {model_path}")
    logger.info(f"💾 Features salvas em: {features_path}")
    
    return model, mae, r2


if __name__ == "__main__":
    train_totals_model()
