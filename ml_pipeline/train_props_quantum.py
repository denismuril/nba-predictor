"""
Train Props Quantum - Arquitetura de Modelagem "Sniper"

Implementa abordagem de dois estágios com Quantile Regression:
- Modelo A (XGBoost): Prevê MINUTOS jogados
- Modelo B (LightGBM): Prevê PPM/RPM/APM com quantis

Autor: Lead Quant Researcher & AI Architect
Versão: 1.0.0 - Quantum Edition
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import mlflow
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

# Configuração
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml_pipeline" / "models"
MODELS_DIR.mkdir(exist_ok=True)

# MLflow Setup
mlflow.set_tracking_uri(f"file://{BASE_DIR}/mlruns")
mlflow.set_experiment("NBA_Quantum_Props")


# =============================================================================
# CONFIGURAÇÃO DE FEATURES
# =============================================================================

# Features para modelo de MINUTOS (Estágio 1)
MINUTES_FEATURES = [
    # Histórico de minutos
    'min_l5', 'min_l10', 'min_season_avg',
    # Fadiga e rotação
    'rest_days', 'is_b2b', 'games_in_5_days',
    'fatigue_score', 'fatigue_circadian_disruption',
    'fatigue_altitude_impact', 'fatigue_thirst_index',
    # Game Script
    'blowout_abs_spread', 'blowout_star_minutes_adj',
    # Contexto
    'is_home', 'opponent_pace', 'playoff_implications',
    # Usage atual
    'usage_pct', 'usage_teammate_out_impact'
]

# Features para modelo de TAXA POR MINUTO (Estágio 2)
RATE_FEATURES = [
    # Taxa histórica
    'ppm_l5', 'ppm_l10', 'rpm_l5', 'rpm_l10', 'apm_l5', 'apm_l10',
    # Matchup
    'dvp_foul_advantage', 'dvp_reb_advantage', 'dvp_total_pf',
    # Usage
    'usage_projected_usage_boost', 'usage_projected_assist_boost',
    'usage_projected_reb_boost',
    # Contexto
    'is_home', 'opponent_def_rating', 'pace_diff',
    # Estilo
    'three_pt_rate', 'ft_rate', 'ast_ratio'
]

# Targets
TARGETS = ['PTS', 'REB', 'AST', 'MIN']


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================

def load_training_data() -> Optional[pd.DataFrame]:
    """
    Carrega dados de treinamento para Player Props.
    
    Tenta carregar de:
    1. CSV de boxscores históricos
    2. Dados sintéticos se não houver dados reais
    
    Returns:
        DataFrame com dados de treinamento ou None
    """
    logger.info("📊 Carregando dados de treinamento...")
    
    # Tentar carregar de múltiplas fontes
    possible_files = [
        DATA_DIR / "player_boxscores_history.csv",
        DATA_DIR / "player_props" / "training_data.csv",
        DATA_DIR / "nba_player_stats.csv"
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            logger.info(f"✅ Carregando de {file_path}")
            df = pd.read_csv(file_path)
            
            # Verificar se o arquivo tem apenas colunas básicas (precisa enriquecer)
            basic_cols = ['Player', 'Team', 'Date', 'PTS', 'REB', 'AST', 'MIN', 'Location', 'Opponent']
            if all(c in df.columns for c in basic_cols) and 'min_l5' not in df.columns:
                logger.info("🔄 Arquivo básico detectado, gerando features...")
                df = enrich_basic_boxscores(df)
            
            # Padronizar nomes de colunas (lowercase exceto targets)
            df.columns = [c.lower() if c not in ['PTS', 'REB', 'AST', 'MIN'] else c for c in df.columns]
            
            return df
    
    # Se não encontrou, gerar dados sintéticos
    logger.warning("⚠️ Dados reais não encontrados. Gerando dados sintéticos...")
    return generate_synthetic_data()


def enrich_basic_boxscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece boxscores básicos com features para treinamento.
    
    Transforma um CSV simples (Player, Team, Date, PTS, REB, AST, MIN, Location, Opponent)
    em um dataset com todas as features necessárias para os modelos Quantum.
    
    Args:
        df: DataFrame com colunas básicas
        
    Returns:
        DataFrame enriquecido com features
    """
    logger.info("🛠️ Gerando features a partir de boxscores básicos...")
    
    df = df.copy()
    
    # Padronizar nomes
    col_map = {
        'Player': 'player',
        'Team': 'team',
        'Date': 'date',
        'Location': 'location',
        'Opponent': 'opponent'
    }
    df = df.rename(columns=col_map)
    
    # Converter data
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['player', 'date'])
    
    # Garantir que MIN seja numérico e positivo
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(25).clip(lower=5, upper=48)
    
    # Derivar is_home e is_b2b
    df['is_home'] = (df['location'].str.lower() == 'home').astype(int)
    
    # Calcular is_b2b baseado na diferença de dias entre jogos
    df['days_since_last'] = df.groupby('player')['date'].diff().dt.days.fillna(3)
    df['is_b2b'] = (df['days_since_last'] == 1).astype(int)
    df['rest_days'] = df['days_since_last'].clip(lower=1, upper=7)
    
    # Rolling averages por jogador (usando shift para evitar leakage)
    for stat in ['MIN', 'PTS', 'REB', 'AST']:
        lower_stat = stat.lower()
        df[f'{lower_stat}_l5'] = df.groupby('player')[stat].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df[f'{lower_stat}_l10'] = df.groupby('player')[stat].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        df[f'{lower_stat}_season_avg'] = df.groupby('player')[stat].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    
    # Calcular taxas por minuto (VETORIZADO - muito mais rápido que .apply())
    df['ppm_l5'] = np.where(df['min_l5'] > 0, df['pts_l5'] / df['min_l5'], 0.5)
    df['ppm_l10'] = np.where(df['min_l10'] > 0, df['pts_l10'] / df['min_l10'], 0.5)
    df['rpm_l5'] = np.where(df['min_l5'] > 0, df['reb_l5'] / df['min_l5'], 0.2)
    df['rpm_l10'] = np.where(df['min_l10'] > 0, df['reb_l10'] / df['min_l10'], 0.2)
    df['apm_l5'] = np.where(df['min_l5'] > 0, df['ast_l5'] / df['min_l5'], 0.15)
    df['apm_l10'] = np.where(df['min_l10'] > 0, df['ast_l10'] / df['min_l10'], 0.15)
    
    # Features de fadiga (simuladas baseadas em dados disponíveis)
    np.random.seed(42)
    n_rows = len(df)
    
    df['fatigue_score'] = np.where(
        df['is_b2b'] == 1,
        np.random.uniform(40, 80, n_rows),
        np.random.uniform(10, 40, n_rows)
    )
    df['fatigue_circadian_disruption'] = np.random.uniform(0, 3, n_rows)
    df['fatigue_altitude_impact'] = np.where(
        df['opponent'].isin(['DEN', 'UTA', 'Denver', 'Utah']), 
        0.5, 
        0.0
    )
    df['fatigue_thirst_index'] = np.random.uniform(0, 50, n_rows)
    
    # Games in 5 days (usar abordagem simples baseada em rest_days)
    # Média: se rest_days < 2, provavelmente está jogando mais frequentemente
    df['games_in_5_days'] = np.where(
        df['rest_days'] <= 1, 3,
        np.where(df['rest_days'] <= 2, 2, 1)
    )

    
    # Blowout features (simuladas)
    df['blowout_abs_spread'] = np.random.uniform(2, 15, n_rows)
    df['blowout_star_minutes_adj'] = np.random.uniform(0.85, 1.0, n_rows)
    
    # Matchup features (simuladas)
    df['dvp_foul_advantage'] = np.random.uniform(0.3, 0.7, n_rows)
    df['dvp_reb_advantage'] = np.random.uniform(0.3, 0.7, n_rows)
    df['dvp_total_pf'] = np.random.uniform(18, 25, n_rows)
    
    # Usage features (simuladas)
    df['usage_pct'] = np.random.uniform(0.15, 0.32, n_rows)
    df['usage_projected_usage_boost'] = 1.0 + np.random.uniform(-0.1, 0.2, n_rows)
    df['usage_projected_assist_boost'] = 1.0 + np.random.uniform(-0.1, 0.15, n_rows)
    df['usage_projected_reb_boost'] = 1.0 + np.random.uniform(-0.05, 0.1, n_rows)
    df['usage_teammate_out_impact'] = np.random.choice([0, 30, 60], n_rows, p=[0.7, 0.2, 0.1])
    
    # Context features (simuladas)
    df['opponent_pace'] = np.random.uniform(95, 105, n_rows)
    df['opponent_def_rating'] = np.random.uniform(105, 118, n_rows)
    df['pace_diff'] = np.random.uniform(-3, 3, n_rows)
    df['playoff_implications'] = np.random.uniform(0, 1, n_rows)
    
    # Style features (simuladas)
    df['three_pt_rate'] = np.random.uniform(0.25, 0.45, n_rows)
    df['ft_rate'] = np.random.uniform(0.15, 0.35, n_rows)
    df['ast_ratio'] = np.random.uniform(0.10, 0.30, n_rows)
    
    # Remover linhas com NaN em colunas críticas
    critical_cols = ['min_l5', 'min_l10', 'MIN', 'PTS', 'REB', 'AST']
    df = df.dropna(subset=critical_cols)
    
    logger.info(f"✅ Features geradas: {len(df.columns)} colunas, {len(df)} registros")
    
    return df


def generate_synthetic_data(n_players: int = 50, n_games: int = 30) -> pd.DataFrame:
    """
    Gera dados sintéticos para treinamento quando dados reais não estão disponíveis.
    
    ATENÇÃO: Isso é apenas para demonstração e testes.
    Em produção, use dados reais.
    """
    logger.info(f"🎲 Gerando {n_players * n_games} registros sintéticos...")
    
    np.random.seed(42)
    
    # Jogadores fictícios
    star_players = [
        ('LeBron James', 'LAL', 'F', 27.5, 7.5, 8.0, 35.5),
        ('Stephen Curry', 'GSW', 'G', 26.0, 5.5, 6.5, 33.0),
        ('Giannis Antetokounmpo', 'MIL', 'F', 30.0, 12.0, 5.5, 35.0),
        ('Nikola Jokic', 'DEN', 'C', 26.5, 12.5, 9.0, 34.0),
        ('Luka Doncic', 'DAL', 'G', 28.0, 9.0, 8.5, 36.0),
        ('Kevin Durant', 'PHX', 'F', 27.0, 7.0, 5.5, 34.5),
        ('Jayson Tatum', 'BOS', 'F', 27.0, 8.0, 4.5, 36.0),
        ('Joel Embiid', 'PHI', 'C', 28.0, 11.0, 4.0, 33.5),
        ('Anthony Davis', 'LAL', 'C', 24.0, 11.5, 3.0, 34.0),
        ('Jimmy Butler', 'MIA', 'F', 22.0, 6.0, 6.0, 33.0),
    ]
    
    role_players = [
        (f'Bench Player {i}', 'TM', 'G', 8 + np.random.normal(0, 2), 
         3 + np.random.normal(0, 1), 2 + np.random.normal(0, 1), 20 + np.random.normal(0, 3))
        for i in range(n_players - len(star_players))
    ]
    
    all_players = star_players + role_players
    
    teams = ['LAL', 'BOS', 'GSW', 'MIL', 'PHX', 'DEN', 'MIA', 'PHI', 'DAL', 'NYK']
    
    data = []
    base_date = datetime(2024, 10, 22)
    
    for player_name, team, pos, avg_pts, avg_reb, avg_ast, avg_min in all_players:
        for game_idx in range(n_games):
            game_date = base_date + pd.Timedelta(days=game_idx * 2 + np.random.randint(0, 2))
            opponent = np.random.choice([t for t in teams if t != team[:3]])
            is_home = np.random.random() > 0.5
            is_b2b = game_idx > 0 and np.random.random() < 0.15
            
            # Variação nos stats
            min_var = np.random.normal(avg_min, 4)
            min_var = max(10, min(42, min_var))
            
            ppm = avg_pts / avg_min  # Points per minute
            rpm = avg_reb / avg_min
            apm = avg_ast / avg_min
            
            # Simular fatiga
            fatigue_score = np.random.uniform(10, 70) if is_b2b else np.random.uniform(0, 40)
            
            # Ajustar por fatiga
            fatigue_mult = 1 - (fatigue_score / 200)
            
            pts = max(0, min_var * ppm * fatigue_mult + np.random.normal(0, 3))
            reb = max(0, min_var * rpm * fatigue_mult + np.random.normal(0, 2))
            ast = max(0, min_var * apm * fatigue_mult + np.random.normal(0, 1.5))
            
            data.append({
                'player': player_name,
                'team': team[:3],
                'position': pos,
                'date': game_date,
                'opponent': opponent,
                'is_home': is_home,
                'is_b2b': is_b2b,
                
                # Targets
                'PTS': round(pts, 1),
                'REB': round(reb, 1),
                'AST': round(ast, 1),
                'MIN': round(min_var, 1),
                
                # Features pré-calculadas
                'min_l5': avg_min + np.random.normal(0, 2),
                'min_l10': avg_min + np.random.normal(0, 1.5),
                'min_season_avg': avg_min,
                'rest_days': np.random.choice([1, 2, 3, 4], p=[0.15, 0.50, 0.25, 0.10]),
                'games_in_5_days': np.random.choice([1, 2, 3], p=[0.40, 0.50, 0.10]),
                'fatigue_score': fatigue_score,
                'fatigue_circadian_disruption': np.random.uniform(0, 3),
                'fatigue_altitude_impact': 0.5 if opponent == 'DEN' else 0.0,
                'fatigue_thirst_index': np.random.uniform(0, 50),
                
                # Blowout
                'blowout_abs_spread': np.random.uniform(2, 15),
                'blowout_star_minutes_adj': np.random.uniform(0.85, 1.0),
                
                # Taxa por minuto
                'ppm_l5': ppm + np.random.normal(0, 0.1),
                'ppm_l10': ppm + np.random.normal(0, 0.08),
                'rpm_l5': rpm + np.random.normal(0, 0.08),
                'rpm_l10': rpm + np.random.normal(0, 0.06),
                'apm_l5': apm + np.random.normal(0, 0.06),
                'apm_l10': apm + np.random.normal(0, 0.04),
                
                # Matchup
                'dvp_foul_advantage': np.random.uniform(0.3, 0.7),
                'dvp_reb_advantage': np.random.uniform(0.3, 0.7),
                'dvp_total_pf': np.random.uniform(18, 25),
                
                # Usage
                'usage_pct': np.random.uniform(0.15, 0.32),
                'usage_projected_usage_boost': 1.0 + np.random.uniform(-0.1, 0.2),
                'usage_projected_assist_boost': 1.0 + np.random.uniform(-0.1, 0.15),
                'usage_projected_reb_boost': 1.0 + np.random.uniform(-0.05, 0.1),
                'usage_teammate_out_impact': np.random.choice([0, 30, 60], p=[0.7, 0.2, 0.1]),
                
                # Contexto
                'opponent_pace': np.random.uniform(95, 105),
                'opponent_def_rating': np.random.uniform(105, 118),
                'pace_diff': np.random.uniform(-3, 3),
                'playoff_implications': np.random.random(),
                
                # Estilo
                'three_pt_rate': np.random.uniform(0.25, 0.45),
                'ft_rate': np.random.uniform(0.15, 0.35),
                'ast_ratio': np.random.uniform(0.10, 0.30),
            })
    
    df = pd.DataFrame(data)
    
    # Salvar para reutilização
    output_path = DATA_DIR / "player_props" / "training_data.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"💾 Dados sintéticos salvos em {output_path}")
    
    return df


def prepare_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    Prepara features para treinamento, preenchendo valores faltantes.
    
    Args:
        df: DataFrame com dados
        feature_list: Lista de features desejadas
        
    Returns:
        DataFrame com apenas as features selecionadas
    """
    # Verificar quais features existem
    existing = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    
    if missing:
        logger.warning(f"⚠️ Features faltantes: {missing[:5]}...")
    
    X = df[existing].copy()
    
    # Preencher NaN com médias
    X = X.fillna(X.median())
    
    # Features ainda faltantes: criar com valor 0
    for feat in missing:
        X[feat] = 0.0
    
    return X[feature_list]


# =============================================================================
# MODELO DE MINUTOS (ESTÁGIO 1)
# =============================================================================

def train_minutes_model(df: pd.DataFrame) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Treina modelo XGBoost para prever MINUTOS jogados.
    
    Este modelo captura:
    - Padrões de rotação do técnico
    - Impacto de fadiga
    - Risco de blowout
    
    Returns:
        Modelo treinado e métricas
    """
    logger.info("\n" + "="*60)
    logger.info("🎯 ESTÁGIO 1: Treinando Modelo de MINUTOS")
    logger.info("="*60)
    
    # Preparar dados
    available_features = [f for f in MINUTES_FEATURES if f in df.columns or f.replace('_', '') in df.columns]
    if len(available_features) < 5:
        logger.warning("⚠️ Poucas features disponíveis, usando alternativas...")
        available_features = ['min_l5', 'min_l10', 'rest_days', 'is_b2b', 'fatigue_score', 'is_home']
        available_features = [f for f in available_features if f in df.columns]
    
    X = prepare_features(df, available_features)
    y = df['MIN'].values
    
    # Split temporal (mais realista que random)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logger.info(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"📊 Features: {len(available_features)}")
    
    with mlflow.start_run(run_name="Minutes_XGBoost", nested=True):
        # Parâmetros otimizados para previsão de minutos
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        }
        
        mlflow.log_params(params)
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Avaliação
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        mlflow.log_metrics(metrics)
        
        logger.info(f"✅ MAE: {mae:.2f} min | RMSE: {rmse:.2f} | R²: {r2:.3f}")
        
        # Salvar modelo
        model_path = MODELS_DIR / "quantum_minutes_xgb.joblib"
        joblib.dump({'model': model, 'features': available_features}, model_path)
        logger.info(f"💾 Modelo salvo em {model_path}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': available_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"\n📊 Top Features:\n{importance.head(10)}")
    
    return model, metrics


# =============================================================================
# MODELO DE TAXA POR MINUTO (ESTÁGIO 2)
# =============================================================================

def train_rate_model(
    df: pd.DataFrame, 
    target: str,
    quantiles: List[float] = [0.10, 0.50, 0.90]
) -> Tuple[Dict[float, lgb.LGBMRegressor], Dict]:
    """
    Treina modelo LightGBM com Quantile Regression para taxa por minuto.
    
    Retorna modelos para múltiplos quantis (10th, 50th, 90th percentil).
    
    Args:
        df: DataFrame com dados
        target: Target (PTS, REB, AST)
        quantiles: Lista de quantis para treinar
        
    Returns:
        Dict de modelos por quantil e métricas
    """
    logger.info(f"\n{'-'*40}")
    logger.info(f"🎯 ESTÁGIO 2: Treinando Modelo de Taxa para {target}")
    logger.info(f"{'-'*40}")
    
    # Calcular taxa por minuto (target / minutos)
    df_rate = df.copy()
    rate_col = f'{target.lower()}_per_minute'
    df_rate[rate_col] = df_rate[target] / df_rate['MIN'].replace(0, 1)
    
    # Preparar features
    available_features = [f for f in RATE_FEATURES if f in df_rate.columns]
    if len(available_features) < 5:
        logger.warning("⚠️ Poucas features, usando alternativas...")
        alt_features = [f'{target.lower()}_l5', f'{target.lower()}_l10', 
                       'is_home', 'dvp_foul_advantage', 'usage_projected_usage_boost']
        # Adaptar nomes
        alt_features = ['ppm_l5', 'ppm_l10', 'is_home', 'dvp_foul_advantage', 'usage_projected_usage_boost']
        available_features = [f for f in alt_features if f in df_rate.columns]
    
    X = prepare_features(df_rate, available_features)
    y = df_rate[rate_col].values
    
    # Split temporal
    train_size = int(len(df_rate) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    models = {}
    all_metrics = {}
    
    with mlflow.start_run(run_name=f"Rate_{target}_Quantile", nested=True):
        for q in quantiles:
            logger.info(f"  📈 Treinando quantil {q*100:.0f}th...")
            
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            models[q] = model
            all_metrics[f'q{int(q*100)}_mae'] = mae
            
            mlflow.log_metric(f"mae_q{int(q*100)}", mae)
        
        # Métricas combinadas
        logger.info(f"  ✅ MAE Q10: {all_metrics.get('q10_mae', 0):.4f}")
        logger.info(f"  ✅ MAE Q50: {all_metrics.get('q50_mae', 0):.4f}")
        logger.info(f"  ✅ MAE Q90: {all_metrics.get('q90_mae', 0):.4f}")
        
        # Salvar modelos
        model_path = MODELS_DIR / f"quantum_rate_{target.lower()}_lgb.joblib"
        joblib.dump({'models': models, 'features': available_features}, model_path)
        logger.info(f"💾 Modelos salvos em {model_path}")
    
    return models, all_metrics


# =============================================================================
# PREVISÃO EM DOIS ESTÁGIOS
# =============================================================================

def predict_with_confidence(
    player_features: Dict,
    minutes_model: xgb.XGBRegressor,
    rate_models: Dict[str, Dict[float, lgb.LGBMRegressor]],
    minutes_features: List[str],
    rate_features: List[str]
) -> Dict[str, Dict]:
    """
    Faz previsão completa em dois estágios com intervalo de confiança.
    
    Estágio 1: Prever minutos
    Estágio 2: Prever taxa por minuto (com quantis)
    Final: Minutos * Taxa
    
    Args:
        player_features: Dict com features do jogador
        minutes_model: Modelo de minutos
        rate_models: Dict de modelos de taxa por target
        minutes_features: Lista de features para minutos
        rate_features: Lista de features para taxas
        
    Returns:
        Dict com previsões por target
    """
    # Preparar features
    X_minutes = pd.DataFrame([player_features])[minutes_features]
    X_rate = pd.DataFrame([player_features])[rate_features]
    
    # Estágio 1: Prever minutos
    predicted_minutes = minutes_model.predict(X_minutes)[0]
    predicted_minutes = max(5, min(42, predicted_minutes))  # Bounds
    
    predictions = {
        'minutes': {
            'value': predicted_minutes
        }
    }
    
    # Estágio 2: Prever taxa para cada target
    for target in ['PTS', 'REB', 'AST']:
        if target in rate_models:
            target_models = rate_models[target]
            
            rate_10 = target_models[0.10].predict(X_rate)[0]
            rate_50 = target_models[0.50].predict(X_rate)[0]
            rate_90 = target_models[0.90].predict(X_rate)[0]
            
            # Previsão final = Minutos * Taxa
            pred_low = predicted_minutes * rate_10
            pred_median = predicted_minutes * rate_50
            pred_high = predicted_minutes * rate_90
            
            predictions[target] = {
                'low': round(pred_low, 1),
                'median': round(pred_median, 1),
                'high': round(pred_high, 1),
                'confidence_interval': (round(pred_low, 1), round(pred_high, 1))
            }
    
    return predictions


def should_bet(
    prediction: Dict,
    line: float,
    target: str,
    odds_over: float = 1.91,
    odds_under: float = 1.91
) -> Dict:
    """
    Determina se deve apostar baseado na previsão vs linha.
    
    Regra de OURO: Se nossa previsão do percentil 10 > linha, é ALL-IN OVER.
                   Se nossa previsão do percentil 90 < linha, é ALL-IN UNDER.
    
    Args:
        prediction: Dict de previsão para o target
        line: Linha proposta pela casa de apostas
        target: Target (PTS, REB, AST)
        odds_over: Odds para OVER
        odds_under: Odds para UNDER
        
    Returns:
        Dict com recomendação de aposta
    """
    if target not in prediction:
        return {'bet': 'SKIP', 'reason': 'Target não previsto'}
    
    pred = prediction[target]
    p10, p50, p90 = pred['low'], pred['median'], pred['high']
    
    # Calcular probabilidade implícita
    prob_over = 1 / odds_over
    prob_under = 1 / odds_under
    
    # Diferença percentual
    diff_pct = ((p50 - line) / line) * 100
    
    # Regra ALL-IN
    if p10 > line:
        # Nosso PISO é maior que a linha = forte OVER
        model_prob_over = 0.85  # 85% confiança
        ev_over = (model_prob_over * odds_over) - 1
        return {
            'bet': 'OVER',
            'strength': 'ALL-IN',
            'prediction': p50,
            'line': line,
            'diff_pct': diff_pct,
            'ev': round(ev_over * 100, 2),
            'confidence': 'HIGH',
            'reason': f'P10 ({p10}) > linha ({line})'
        }
    elif p90 < line:
        # Nosso TETO é menor que a linha = forte UNDER
        model_prob_under = 0.85
        ev_under = (model_prob_under * odds_under) - 1
        return {
            'bet': 'UNDER',
            'strength': 'ALL-IN',
            'prediction': p50,
            'line': line,
            'diff_pct': diff_pct,
            'ev': round(ev_under * 100, 2),
            'confidence': 'HIGH',
            'reason': f'P90 ({p90}) < linha ({line})'
        }
    elif p50 > line * 1.1:  # 10% acima
        # Mediana significativamente maior
        model_prob = 0.60
        ev = (model_prob * odds_over) - 1
        return {
            'bet': 'OVER',
            'strength': 'MEDIUM',
            'prediction': p50,
            'line': line,
            'diff_pct': diff_pct,
            'ev': round(ev * 100, 2),
            'confidence': 'MEDIUM',
            'reason': f'Mediana 10%+ acima'
        }
    elif p50 < line * 0.9:  # 10% abaixo
        model_prob = 0.60
        ev = (model_prob * odds_under) - 1
        return {
            'bet': 'UNDER',
            'strength': 'MEDIUM',
            'prediction': p50,
            'line': line,
            'diff_pct': diff_pct,
            'ev': round(ev * 100, 2),
            'confidence': 'MEDIUM',
            'reason': f'Mediana 10%+ abaixo'
        }
    else:
        return {
            'bet': 'SKIP',
            'prediction': p50,
            'line': line,
            'diff_pct': diff_pct,
            'ev': 0,
            'confidence': 'LOW',
            'reason': 'Sem vantagem clara'
        }


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def train_all_quantum_models() -> Dict:
    """
    Treina todos os modelos Quantum Props.
    
    Returns:
        Dict com todos os modelos e métricas
    """
    logger.info("\n" + "="*80)
    logger.info("🚀 QUANTUM PROPS - PIPELINE DE TREINAMENTO")
    logger.info("="*80)
    
    # Carregar dados
    df = load_training_data()
    
    if df is None or df.empty:
        logger.error("❌ Não foi possível carregar dados de treinamento")
        return {}
    
    logger.info(f"📊 Total de registros: {len(df)}")
    logger.info(f"📊 Colunas: {len(df.columns)}")
    
    with mlflow.start_run(run_name="Quantum_Props_Full_Training"):
        results = {
            'models': {},
            'metrics': {}
        }
        
        # Estágio 1: Modelo de Minutos
        minutes_model, minutes_metrics = train_minutes_model(df)
        results['models']['minutes'] = minutes_model
        results['metrics']['minutes'] = minutes_metrics
        
        # Estágio 2: Modelos de Taxa para cada target
        for target in ['PTS', 'REB', 'AST']:
            logger.info(f"\n{'='*40}")
            logger.info(f"🎯 Treinando modelos de taxa para {target}")
            
            try:
                rate_models, rate_metrics = train_rate_model(df, target)
                results['models'][f'rate_{target}'] = rate_models
                results['metrics'][f'rate_{target}'] = rate_metrics
            except Exception as e:
                logger.error(f"❌ Erro ao treinar modelo {target}: {e}")
                continue
        
        # Resumo final
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMO DO TREINAMENTO")
        logger.info("="*80)
        
        for model_name, metrics in results['metrics'].items():
            logger.info(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        
        # Salvar resumo
        summary_path = MODELS_DIR / "quantum_training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'metrics': {k: {mk: float(mv) for mk, mv in v.items()} 
                           for k, v in results['metrics'].items()}
            }, f, indent=2)
        
        logger.info(f"\n✅ Treinamento concluído! Resumo em {summary_path}")
    
    return results


def load_quantum_models() -> Dict:
    """
    Carrega modelos Quantum treinados.
    
    Returns:
        Dict com modelos carregados
    """
    models = {}
    
    # Carregar modelo de minutos
    minutes_path = MODELS_DIR / "quantum_minutes_xgb.joblib"
    if minutes_path.exists():
        data = joblib.load(minutes_path)
        models['minutes'] = data['model']
        models['minutes_features'] = data['features']
        logger.info("✅ Modelo de minutos carregado")
    
    # Carregar modelos de taxa
    for target in ['PTS', 'REB', 'AST']:
        rate_path = MODELS_DIR / f"quantum_rate_{target.lower()}_lgb.joblib"
        if rate_path.exists():
            data = joblib.load(rate_path)
            models[f'rate_{target}'] = data['models']
            models[f'rate_{target}_features'] = data['features']
            logger.info(f"✅ Modelos de taxa {target} carregados")
    
    return models


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("🚀 Iniciando Pipeline de Treinamento Quantum Props...")
    
    results = train_all_quantum_models()
    
    if results:
        logger.info("\n" + "="*80)
        logger.info("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("="*80)
        logger.info("\nPróximos passos:")
        logger.info("1. Execute: python scripts/quantum_props_run.py")
        logger.info("2. Acesse o dashboard: streamlit run nba_predictor_web.py")
        logger.info("3. Navegue até a aba '💰 PROP SNIPER'")
    else:
        logger.error("❌ Falha no treinamento")
