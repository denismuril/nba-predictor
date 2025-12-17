import pandas as pd
import numpy as np
import logging
from data.repositories.db_manager import get_db_manager
from core.contextual_features import add_all_contextual_features
from datetime import datetime, timedelta
from utils.team_normalization import normalize_team, team_to_full_name  # NEW: Centralized normalization
from ml_pipeline.player_aggregation import (
    aggregate_player_stats_by_team, 
    merge_player_features_to_games,
    get_cached_player_stats
)

logger = logging.getLogger(__name__)

def load_multi_season_data(seasons=None):
    """
    Carrega dados de múltiplas temporadas combinados.
    
    Args:
        seasons: Lista de temporadas (ex: ['2023-24', '2024-25', '2025-26'])
                 Se None, carrega apenas a temporada atual
    
    Returns:
        DataFrame combinado com dados de todas as temporadas
    """
    if seasons is None:
        seasons = ['2025-26']  # Default: apenas temporada atual
    
    logger.info(f"📦 Carregando dados de {len(seasons)} temporada(s): {', '.join(seasons)}")
    
    db = get_db_manager()
    df = db.get_comprehensive_history()
    
    if df is None or df.empty:
        logger.warning("⚠️  Nenhum dado histórico encontrado.")
        return pd.DataFrame()
    
    # Converter data
    df['date'] = pd.to_datetime(df['date'])
    
    # Filtrar por temporadas (aproximado - baseado em ano)
    # NBA season 2025-26 vai de out/2025 a jun/2026
    # Simplificado: se season = '2025-26', pegar jogos com data >= 2025-10-01
    if seasons != ['all']:
        season_dfs = []
        for season in seasons:
            year_start = int(season.split('-')[0])
            # Temporada começa em outubro do primeiro ano
            start_date = pd.Timestamp(f'{year_start}-10-01')
            # E termina em junho do segundo ano
            end_date = pd.Timestamp(f'{year_start + 1}-06-30')
            
            season_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            logger.info(f"  Temporada {season}: {len(season_df)} jogos")
            season_dfs.append(season_df)
        
        df = pd.concat(season_dfs, ignore_index=True) if season_dfs else pd.DataFrame()
    
    # Ordenar por data
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"✅ Total carregado: {len(df)} jogos de {df['date'].min()} a {df['date'].max()}")
    return df


def calculate_sample_weights(df, weight_config=None, use_exponential=False):
    """
    Calcula pesos para cada jogo baseado na recência.
    Jogos mais recentes recebem peso maior no treinamento.
    
    Args:
        df: DataFrame com coluna 'date'
        weight_config: Dict com configuração de pesos
                      SE use_exponential=False:
                       {'recent_30_days': 3.0, 'recent_60_days': 2.0, 
                        'recent_90_days': 1.5, 'default': 1.0}
                      SE use_exponential=True:
                       {'decay_constant': 45}  # tau em dias
        use_exponential: Se True, usa exponential decay; se False, usa step-based
    
    Returns:
        numpy array com pesos para cada linha
    """
    # Data mais recente no dataset
    max_date = df['date'].max()
    
    # Calcular dias desde a data mais recente
    days_from_latest = (max_date - df['date']).dt.days
    
    if use_exponential:
        # Exponential Decay: weight = exp(-days / tau)
        # tau (decay_constant) controla a velocidade de decaimento
        # Padrão: tau = 45 dias (peso cai pela metade a cada ~30 dias)
        if weight_config is None or 'decay_constant' not in weight_config:
            tau = 45
        else:
            tau = weight_config['decay_constant']
        
        weights = np.exp(-days_from_latest / tau)
        
        # Normalizar para que o peso máximo seja 3.0 (comparável ao step-based)
        weights = weights * 3.0 / weights.max()
        
        logger.info(f"📊 Exponential Decay Weighting (tau={tau}):")
        logger.info(f"   Peso médio: {weights.mean():.2f}")
        logger.info(f"   Peso mín: {weights.min():.2f}, máx: {weights.max():.2f}")
        logger.info(f"   Jogos com peso > 2.0: {(weights > 2.0).sum()}")
        
    else:
        # Step-Based Weighting (Original)
        if weight_config is None:
            weight_config = {
                'recent_30_days': 1.5,  # Reduzido de 3.0 para 1.5 (menos variância)
                'recent_60_days': 1.2,  # Reduzido de 2.0 para 1.2
                'recent_90_days': 1.1,  # Ajuste fino
                'default': 1.0
            }
        
        # Aplicar pesos baseados na recência
        weights = np.ones(len(df)) * weight_config['default']
        
        # Últimos 30 dias
        weights[days_from_latest <= 30] = weight_config['recent_30_days']
        
        # 31-60 dias
        mask_60 = (days_from_latest > 30) & (days_from_latest <= 60)
        weights[mask_60] = weight_config['recent_60_days']
        
        # 61-90 dias
        mask_90 = (days_from_latest > 60) & (days_from_latest <= 90)
        weights[mask_90] = weight_config['recent_90_days']
        
    return weights

def calculate_four_factors(df):
    """
    Calcula os Four Factors para cada jogo (Home e Away).
    Fórmulas aproximadas usando box scores disponíveis.
    Inclui fallback para jogos sem stats detalhadas.
    """
    # Verificar se temos stats detalhadas
    detailed_cols = ['fgm', 'fga', 'fg3m', 'fta', 'tov', 'oreb', 'dreb',
                     'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_fta', 'opp_tov', 'opp_oreb', 'opp_dreb']
    has_detailed = all(col in df.columns for col in detailed_cols)
    
    # Sempre criar ratings a partir dos scores (essencial para opponent-adjusted)
    if 'home_score' in df.columns and 'away_score' in df.columns:
        # Ratings = pontos (aproximação, já que não temos posses)
        df['home_off_rating'] = df['home_score'].fillna(0).astype(float)
        df['away_off_rating'] = df['away_score'].fillna(0).astype(float)
        df['home_def_rating'] = df['away_score'].fillna(0).astype(float)
        df['away_def_rating'] = df['home_score'].fillna(0).astype(float)
        
        # Para jogos futuros (score = 0), usar média da liga
        zero_mask = (df['home_off_rating'] == 0) | (df['away_off_rating'] == 0)
        if zero_mask.any():
            df.loc[zero_mask, 'home_off_rating'] = 112.0
            df.loc[zero_mask, 'away_off_rating'] = 112.0
            df.loc[zero_mask, 'home_def_rating'] = 112.0
            df.loc[zero_mask, 'away_def_rating'] = 112.0
    
    if has_detailed:
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        df['home_efg'] = (df['fgm'] + 0.5 * df['fg3m']) / df['fga'].replace(0, 1)
        df['away_efg'] = (df['opp_fgm'] + 0.5 * df['opp_fg3m']) / df['opp_fga'].replace(0, 1)
        
        # TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        df['home_tov_pct'] = df['tov'] / (df['fga'] + 0.44 * df['fta'] + df['tov']).replace(0, 1)
        df['away_tov_pct'] = df['opp_tov'] / (df['opp_fga'] + 0.44 * df['opp_fta'] + df['opp_tov']).replace(0, 1)
        
        # ORB% = OREB / (OREB + OppDREB)
        df['home_orb_pct'] = df['oreb'] / (df['oreb'] + df['opp_dreb']).replace(0, 1)
        df['away_orb_pct'] = df['opp_oreb'] / (df['opp_oreb'] + df['dreb']).replace(0, 1)
        
        # FTR = FTA / FGA
        df['home_ftr'] = df['fta'] / df['fga'].replace(0, 1)
        df['away_ftr'] = df['opp_fta'] / df['opp_fga'].replace(0, 1)
        
        # Math-Fix: PACE = Posses estimadas por jogo
        # Fórmula: FGA - OREB + TOV + 0.44*FTA
        df['home_pace'] = (df['fga'] - df['oreb'] + df['tov'] + 0.44 * df['fta']).clip(lower=70)
        df['away_pace'] = (df['opp_fga'] - df['opp_oreb'] + df['opp_tov'] + 0.44 * df['opp_fta']).clip(lower=70)
        
    else:
        # Fallback: usar valores neutros para Four Factors
        df['home_efg'] = 0.5
        df['away_efg'] = 0.5
        df['home_tov_pct'] = 0.12  # ~12% TOV
        df['away_tov_pct'] = 0.12
        df['home_orb_pct'] = 0.25  # ~25% ORB
        df['away_orb_pct'] = 0.25
        df['home_ftr'] = 0.25  # ~25% FTR
        df['away_ftr'] = 0.25
        df['home_pace'] = 99.5  # Média da liga
        df['away_pace'] = 99.5
    
    return df

def add_rolling_features(df, windows=[5, 10]):
    """
    Adiciona features de média móvel (rolling) para cada time.
    """
    # Debug: Verificar colunas antes de criar home_df
    required_cols = ['date', 'home_team', 'home_score', 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ add_rolling_features: Colunas faltando para home_df: {missing}")
        logger.info(f"   Colunas disponíveis: {df.columns.tolist()}")
        raise KeyError(f"Missing columns: {missing}")

    # Criar DF longo para cálculos por team
    # Math-Fix: Incluir pace para rolling pace features
    pace_col = 'home_pace' if 'home_pace' in df.columns else None
    
    if pace_col and 'away_pace' in df.columns:
        home_df = df[['date', 'home_team', 'home_score', 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr', 'home_pace']].copy()
        home_df.columns = ['date', 'team', 'points', 'efg', 'tov_pct', 'orb_pct', 'ftr', 'pace']
    else:
        home_df = df[required_cols].copy()
        home_df.columns = ['date', 'team', 'points', 'efg', 'tov_pct', 'orb_pct', 'ftr']
    home_df['win'] = (df['home_score'] > df['away_score']).astype(int)
    home_df['is_home'] = 1
    
    if pace_col and 'away_pace' in df.columns:
        away_df = df[['date', 'away_team', 'away_score', 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr', 'away_pace']].copy()
        away_df.columns = ['date', 'team', 'points', 'efg', 'tov_pct', 'orb_pct', 'ftr', 'pace']
    else:
        away_df = df[['date', 'away_team', 'away_score', 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr']].copy()
        away_df.columns = ['date', 'team', 'points', 'efg', 'tov_pct', 'orb_pct', 'ftr']
    away_df['win'] = (df['away_score'] > df['home_score']).astype(int)
    away_df['is_home'] = 0
    
    long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date']).reset_index(drop=True)
    
    # Calcular rolling stats com EWMA (consistente com feature_engineering_v2)
    # AUDIT FIX #3: EWMA dá mais peso a jogos recentes (mais reativo que SMA)
    metrics = ['points', 'efg', 'tov_pct', 'orb_pct', 'ftr', 'win']
    if 'pace' in long_df.columns:
        metrics.append('pace')
    
    for window in windows:
        for metric in metrics:
            col_name = f'rolling_{window}_{metric}'
            long_df[col_name] = long_df.groupby('team')[metric].transform(
                lambda x: x.shift(1).ewm(span=window, min_periods=1).mean()
            )
            
    # Win streak - SHIFTED to avoid leakage!
    long_df['win_shifted'] = long_df.groupby('team')['win'].shift(1)
    long_df['win_streak'] = long_df.groupby('team')['win_shifted'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    ) * long_df['win_shifted'].fillna(0)  # Se win=0, streak=0
    
    # Merge de volta
    roll_cols = [c for c in long_df.columns if 'rolling' in c or 'streak' in c]
    
    # Home
    long_home = long_df[long_df['is_home'] == 1].copy()
    rename_home = {c: f'home_{c}' for c in roll_cols}
    long_home = long_home.rename(columns=rename_home)
    
    # Away
    long_away = long_df[long_df['is_home'] == 0].copy()
    rename_away = {c: f'away_{c}' for c in roll_cols}
    long_away = long_away.rename(columns=rename_away)
    
    # Merge
    df = pd.merge(df, long_home[['date', 'team'] + list(rename_home.values())], 
                  left_on=['date', 'home_team'], right_on=['date', 'team'], how='left')
    df = df.drop(columns=['team'])
    
    df = pd.merge(df, long_away[['date', 'team'] + list(rename_away.values())], 
                  left_on=['date', 'away_team'], right_on=['date', 'team'], how='left')
    df = df.drop(columns=['team'])
    
    # 🔧 DIAGNÓSTICO: Verificar qualidade do merge
    sample_rolling_col = f'home_rolling_{windows[0]}_points'
    if sample_rolling_col in df.columns:
        nan_pct = (df[sample_rolling_col].isna().sum() / len(df)) * 100 if len(df) > 0 else 0
        logger.info(f"📊 Merge de rolling features: {nan_pct:.1f}% NaN em '{sample_rolling_col}'")
        
        # Tolerância aumentada para 15% para acomodar "cold start" em datasets pequenos
        if nan_pct > 15:
            logger.warning(f"⚠️ Alta taxa de NaN ({nan_pct:.1f}%) - verifique nomes de times se > 20%")
            # Mostrar exemplo de times não encontrados
            if df[df[sample_rolling_col].isna()].shape[0] > 0:
                missing = df[df[sample_rolling_col].isna()][['date', 'home_team', 'away_team']].head(3)
                logger.warning(f"   Exemplos de jogos com NaN:\n{missing}")
        elif nan_pct > 0:
            logger.info(f"ℹ️ Taxa de NaN esperada ({nan_pct:.1f}%) devido a cold start (início do histórico)")
    
    return df

def add_advanced_features(df):
    """
    Adiciona features avançadas (Fase 1).
    """
    # Recriar long_df para cálculos complexos
    home_df = df[['date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    home_df.columns = ['date', 'team', 'opp_team', 'points', 'opp_points']
    home_df['is_home'] = 1
    
    away_df = df[['date', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
    away_df.columns = ['date', 'team', 'opp_team', 'points', 'opp_points']
    away_df['is_home'] = 0
    
    long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date']).reset_index(drop=True)
    long_df['win'] = (long_df['points'] > long_df['opp_points']).astype(int)
    
    # === 1. REST DAYS ===
    long_df['prev_game_date'] = long_df.groupby('team')['date'].shift(1)
    long_df['rest_days'] = (long_df['date'] - long_df['prev_game_date']).dt.days
    long_df['rest_days'] = long_df['rest_days'].fillna(3)
    # Cap rest_days at 7 to ignore inter-season gaps (preseason)
    long_df['rest_days'] = long_df['rest_days'].clip(upper=7)
    
    # === 2. BACK-TO-BACK ===
    long_df['is_back_to_back'] = (long_df['rest_days'] <= 1).astype(int)
    
    # === 3. NET RATING TREND ===
    long_df['net_rating'] = long_df['points'] - long_df['opp_points']
    
    long_df['net_rating_5'] = long_df.groupby('team')['net_rating'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    long_df['net_rating_10'] = long_df.groupby('team')['net_rating'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    long_df['net_rating_trend'] = long_df['net_rating_5'] - long_df['net_rating_10']
    long_df['net_rating_trend'] = long_df['net_rating_trend'].fillna(0)
    
    # === 4. STRENGTH OF SCHEDULE (SOS) ===
    long_df['cumulative_wins'] = long_df.groupby('team')['win'].cumsum()
    long_df['cumulative_games'] = long_df.groupby('team').cumcount() + 1
    long_df['team_win_pct'] = (long_df['cumulative_wins'] / long_df['cumulative_games']).fillna(0.5)
    
    win_pct_lookup = long_df.set_index(['date', 'team'])['team_win_pct'].to_dict()
    
    def get_opp_win_pct(row):
        key = (row['date'], row['opp_team'])
        return win_pct_lookup.get(key, 0.5)
    
    long_df['opp_win_pct'] = long_df.apply(get_opp_win_pct, axis=1)
    
    long_df['sos_10'] = long_df.groupby('team')['opp_win_pct'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    long_df['sos_10'] = long_df['sos_10'].fillna(0.5)
    
    # === 5. ALTITUDE ADVANTAGE ===
    # AUDIT FIX #6: Altitude contínua em pés ao invés de binário (DEN/UTA = 1)
    # Arenas NBA com altitude significativa (acima de 400m/1300ft)
    ARENA_ALTITUDE_FEET = {
        'DEN': 5280,   # Ball Arena, Denver - 1609m
        'UTA': 4226,   # Delta Center, Salt Lake City - 1288m
        'PHX': 1117,   # Footprint Center, Phoenix - 340m
        'ATL': 1050,   # State Farm Arena, Atlanta - 320m
        'SAS': 650,    # AT&T Center, San Antonio - 198m
        # Arenas próximas ao nível do mar (<100 pés): MIA, LAL, GSW, BOS, NY, etc.
    }
    
    # Normalizar altitude: 0 = nível do mar, 1 = altitude de Denver
    MAX_ALTITUDE = 5280
    
    def get_altitude_advantage(row):
        if row['is_home'] == 1:
            altitude = ARENA_ALTITUDE_FEET.get(row['team'], 0)
            return altitude / MAX_ALTITUDE
        return 0
    
    long_df['altitude_advantage'] = long_df.apply(get_altitude_advantage, axis=1)
    
    # === MERGE ===
    adv_features = ['rest_days', 'is_back_to_back', 'net_rating_trend', 'sos_10', 'altitude_advantage']
    
    long_home = long_df[long_df['is_home'] == 1][['date', 'team'] + adv_features].copy()
    long_away = long_df[long_df['is_home'] == 0][['date', 'team'] + adv_features].copy()
    
    long_home.columns = ['date', 'team'] + [f'home_{f}' for f in adv_features]
    long_away.columns = ['date', 'team'] + [f'away_{f}' for f in adv_features]
    
    # Drop existing cols to avoid duplicates
    cols_to_drop = [c for c in long_home.columns if c in df.columns and c not in ['date', 'home_team']]
    if cols_to_drop: df = df.drop(columns=cols_to_drop)
        
    df = pd.merge(df, long_home, left_on=['date', 'home_team'], right_on=['date', 'team'], how='left')
    df = df.drop(columns=['team'])
    
    cols_to_drop_away = [c for c in long_away.columns if c in df.columns and c not in ['date', 'away_team']]
    if cols_to_drop_away: df = df.drop(columns=cols_to_drop_away)

    df = pd.merge(df, long_away, left_on=['date', 'away_team'], right_on=['date', 'team'], how='left')
    df = df.drop(columns=['team'])
    
    # Rest Diff
    if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
        df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']
        df['rest_diff'] = df['rest_diff'].fillna(0)
    else:
        df['rest_diff'] = 0
    
    logger.info(f"✅ Advanced features adicionadas: {len(adv_features)} base + rest_diff")
    return df

def create_interaction_features(df):
    """
    Cria features de interação que capturam relações não-lineares (Fase 3).
    """
    interactions_created = []
    
    # 1. Produtos
    if 'home_rolling_10_efg' in df.columns and 'away_rolling_10_efg' in df.columns:
        df['interaction_efg_product'] = df['home_rolling_10_efg'] * df['away_rolling_10_efg']
        interactions_created.append('interaction_efg_product')
    
    if 'home_sos_10' in df.columns and 'away_sos_10' in df.columns:
        df['interaction_sos_product'] = df['home_sos_10'] * df['away_sos_10']
        interactions_created.append('interaction_sos_product')
    
    if 'home_rolling_10_win' in df.columns and 'away_rolling_10_win' in df.columns:
        df['interaction_win_product'] = df['home_rolling_10_win'] * df['away_rolling_10_win']
        interactions_created.append('interaction_win_product')
    
    # 2. Diferenças
    if 'home_sos_10' in df.columns and 'away_sos_10' in df.columns:
        df['interaction_sos_diff'] = df['home_sos_10'] - df['away_sos_10']
        interactions_created.append('interaction_sos_diff')
    
    if 'home_win_streak' in df.columns and 'away_win_streak' in df.columns:
        df['interaction_streak_diff'] = df['home_win_streak'] - df['away_win_streak']
        interactions_created.append('interaction_streak_diff')
    
    if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
        df['interaction_rest_diff'] = df['home_rest_days'] - df['away_rest_days']
        interactions_created.append('interaction_rest_diff')
    
    # 3. Ratios
    if 'home_rolling_10_win' in df.columns and 'away_rolling_10_win' in df.columns:
        df['interaction_win_ratio'] = df['home_rolling_10_win'] / (df['away_rolling_10_win'] + 0.01)
        interactions_created.append('interaction_win_ratio')
    
    # 4. Interactions complexas
    if 'home_win_streak' in df.columns and 'home_rolling_5_win' in df.columns:
        df['interaction_home_momentum'] = df['home_win_streak'] * df['home_rolling_5_win']
        interactions_created.append('interaction_home_momentum')
    
    if 'away_win_streak' in df.columns and 'away_rolling_5_win' in df.columns:
        df['interaction_away_momentum'] = df['away_win_streak'] * df['away_rolling_5_win']
        interactions_created.append('interaction_away_momentum')
    
    if all(f in df.columns for f in ['home_sos_10', 'home_rolling_10_win', 'home_rest_days']):
        df['interaction_home_composite'] = (
            df['home_sos_10'] * 
            df['home_rolling_10_win'] * 
            np.clip(df['home_rest_days'] / 5, 0, 2)
        )
        interactions_created.append('interaction_home_composite')
    
    logger.info(f"✅ {len(interactions_created)} feature interactions criadas")
    return df

def load_historical_data(seasons=None, apply_weights=False, weight_config=None, enable_player_features=False, raw=False):
    """
    Carrega dados históricos e prepara features para treinamento (V13 Enhanced).
    
    Args:
        seasons: Lista de temporadas para carregar
        apply_weights: Se True, retorna também sample weights
        weight_config: Config de pesos para sample weighting
        enable_player_features: Se True, adiciona features de RAPM/BPM dos jogadores
        raw: Se True, retorna apenas os dados brutos (sem feature engineering antigo)
    """
    logger.info(f"🔄 Iniciando carregamento de dados históricos (V13) [raw={raw}]...")
    
    if seasons is not None:
        df = load_multi_season_data(seasons)
    else:
        db = get_db_manager()
        df = db.get_comprehensive_history()
    
    logger.debug(f"Shape after DB load: {df.shape}")

    if df is None or df.empty:
        logger.warning("⚠️  Nenhum dado histórico encontrado.")
        return None if not apply_weights else (None, None)
    
    # 🧹 FILTRAR PRÉ-TEMPORADA E OFFSEASON
    # Manter apenas jogos de temporada regular (novembro a abril)
    logger.info("🧹 Filtrando jogos de pré-temporada e offseason...")
    
    total_before = len(df)
    
    # Converter date para datetime se necessário
    df['date'] = pd.to_datetime(df['date'])
    
    # Extrair mês
    df['month'] = df['date'].dt.month
    
    # Temporada regular NBA: Outubro a Abril (10, 11, 12, 1, 2, 3, 4)
    # Excluir: Maio-Setembro (5-9) = playoffs finais + offseason + pré-temporada
    REGULAR_SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4]
    df = df[df['month'].isin(REGULAR_SEASON_MONTHS)].copy()
    df = df.drop(columns=['month'])
    
    logger.debug(f"Shape after filtering: {df.shape}")
    
    removed = total_before - len(df)
    if removed > 0:
        logger.info(f"   ✅ Removidos {removed} jogos fora da temporada regular ({(removed/total_before)*100:.1f}%)")
        logger.info(f"   ✅ Mantidos {len(df)} jogos limpos (nov-abr)")
    else:
        logger.info(f"   ✅ Todos os {len(df)} jogos já são de temporada regular")
    
    # Conversões básicas
    df['date'] = pd.to_datetime(df['date'])
    numeric_cols = ['home_score', 'away_score', 'odds_home', 'odds_away', 
                   'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                   'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 🧹 NORMALIZAÇÃO DE TIMES (CRÍTICO)
    # Garantir que todos os times usem IDs de 3 letras consistentes
    logger.info("🧹 Normalizando nomes de times...")
    df['home_team'] = df['home_team'].apply(normalize_team)
    df['away_team'] = df['away_team'].apply(normalize_team)
    
    # Remover jogos onde a normalização falhou (None)
    invalid_teams = df[df['home_team'].isna() | df['away_team'].isna()]
    if not invalid_teams.empty:
        logger.warning(f"⚠️  Removendo {len(invalid_teams)} jogos com nomes de times inválidos")
        df = df.dropna(subset=['home_team', 'away_team']).reset_index(drop=True)
    
    # 🆔 CRIAR GAME_ID ÚNICO (AUDIT FIX: criar no início para uso em merges)
    # Formato: YYYY-MM-DD_HOME_AWAY (ex: 2025-12-10_LAL_BOS)
    # Isso evita ambiguidade em doubleheaders (mesmo time jogando 2x no dia)
    df['game_id'] = (
        df['date'].dt.strftime('%Y-%m-%d') + '_' + 
        df['home_team'].astype(str) + '_' + 
        df['away_team'].astype(str)
    )
    logger.info(f"🆔 game_id criado: {df['game_id'].nunique()} jogos únicos")
            
    if raw:
        logger.info("📦 Retornando dados brutos (raw=True)")
        return df

    # Calcular targets
    
    # 1. Calcular Four Factors (Raw)
    logger.info("📊 Calculando Four Factors (Raw)...")
    df = calculate_four_factors(df)
    
    # 1.1 🎯 AJUSTAR STATS POR FORÇA DO OPONENTE (NOVO - Crítico!)
    try:
        from ml_pipeline.opponent_adjusted_stats import calcular_stats_ajustados_oponente
        logger.info("🎯 Calculando stats ajustados por oponente (Strength of Schedule)...")
        df = calcular_stats_ajustados_oponente(df)
        logger.info("✅ Stats ajustados adicionados: home/away_ortg_adj, home/away_drtg_adj")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao calcular stats ajustados: {e}")

    # 2.1 🏀 CALCULAR ELO RATINGS (NOVO - Power Ratings Profissionais)
    try:
        from ml_pipeline.elo_system import calcular_elo_ratings_historico
        logger.info("🏀 Calculando Elo Ratings (Power Ratings)...")
        df = calcular_elo_ratings_historico(df)
        logger.info(f"✅ Elo features adicionadas: home_elo, away_elo, elo_diff")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao calcular Elo: {e}")
        logger.warning("Continuando sem Elo features...")
        # Adicionar placeholders
        df['home_elo'] = 1500
        df['away_elo'] = 1500
        df['elo_diff'] = 0
    
    # 2.2 Adicionar Contextual Features (Rest, Travel, Schedule)
    df = add_all_contextual_features(df)
        
    # 3. Adicionar Rolling Features
    # CRITICAL: Use add_rolling_features for model compatibility
    # Model V6 expects: home_rolling_5_points, home_rolling_5_efg, etc.
    logger.info("🔄 Calculando rolling features (5 e 10 jogos)...")
    df = add_rolling_features(df, windows=[5, 10])
    
    # Add Four Factors rolling for V4 advanced features (ts_pct, off_rating)
    # BUG FIXED: Now preserves existing columns and only adds new ones
    logger.info("🔄 Calculando rolling Four Factors (5, 10, 30 jogos)...")
    from ml_pipeline.feature_engineering_v2 import add_rolling_four_factors
    df = add_rolling_four_factors(df, windows=[5, 10, 30])

    # 3.0.1 v25.0: Adicionar Pace Volatility Features (Totals model)
    # Math-Context: Volatilidade de pace prediz variância em totals
    try:
        from ml_pipeline.pace_volatility import add_pace_volatility_features
        df = add_pace_volatility_features(df, windows=[5, 10])
        logger.info("   ✅ v25.0 Pace Volatility features added")
    except Exception as e:
        logger.warning(f"   ⚠️ Pace Volatility features failed: {e}")

    # 3.1 v19.0: Adicionar Context-Aware Rolling Features (Home vs Away)
    try:
        from ml_pipeline.feature_engineering_v2 import add_contextual_rolling_features
        df = add_contextual_rolling_features(df, window=10)
        logger.info("   ✅ v19.0 Context-aware rolling features added (Home/Away)")
    except Exception as e:
        logger.warning(f"   ⚠️ Context-aware rolling features failed: {e}")

    # 3.2 v20.0: Adicionar Referee Features (Árbitros)
    # Math-Context: Árbitros com home_win_pct > 0.55 beneficiam o mandante
    try:
        from ml_pipeline.feature_engineering_v2 import add_referee_features
        df = add_referee_features(df, referee_names_col=None)  # Sem coluna específica, usa média
        logger.info("   ✅ v20.0 Referee features added")
    except Exception as e:
        logger.warning(f"   ⚠️ Referee features failed: {e}")

    # 3.3 v20.0: Adicionar Smart Money Features (Movimentação de Odds)
    # Math-Context: Sharp bettors movem linhas - seguir o dinheiro inteligente
    try:
        from ml_pipeline.feature_engineering_v2 import add_smart_money_features
        # FIX: Verificar se temos dados REAIS de abertura/fechamento
        opening_col = 'opening_odds' if 'opening_odds' in df.columns else None
        closing_col = 'closing_odds' if 'closing_odds' in df.columns else None
        
        if opening_col and closing_col:
            df = add_smart_money_features(df, opening_col=opening_col, closing_col=closing_col)
            logger.info("   ✅ Smart Money features com dados REAIS de movimentação")
        else:
            # AUDIT FIX #3: Sem dados reais - NÃO criar features falsas
            # Features constantes prejudicam o modelo mais que ajudam
            # Anteriormente: df['smart_money_signal'] = 0 (criava ruído)
            # Agora: features simplesmente não são criadas
            logger.info("   ⏭️ Smart Money: dados opening/closing indisponíveis. Features EXCLUÍDAS do treino.")
    except Exception as e:
        logger.warning(f"   ⚠️ Smart Money features failed: {e}")

    # 3.4 v21.7 FASE 70%+: Fatigue Score (Travel Distance + B2B + Schedule Density)
    # Math-Context: Times em road trip longa (>3000km/semana) perdem ~3% win rate
    # ⚠️ DISABLED v27.0: calculate_schedule_fatigue causa OOM (Exit Code 137)
    # TODO: Usar add_travel_km_last_3_days de advanced_features.py que é otimizado
    try:
        # TEMPORARIAMENTE DESABILITADO - causa OOM em datasets grandes
        # from core.travel_calculator import calculate_schedule_fatigue
        # df = calculate_schedule_fatigue(df)
        # logger.info("   ✅ v21.7 Fatigue Score features added (travel + b2b + density)")
        
        # Fallback: adicionar placeholders neutros (features de travel vêm via add_domain_expert_features)
        df['home_fatigue_score'] = 0.0
        df['away_fatigue_score'] = 0.0
        df['home_distance_km'] = 0.0
        df['away_distance_km'] = 0.0
        logger.info("   ⏭️ Fatigue Score: usando fallback (v27.0 travel features via add_domain_expert_features)")
    except Exception as e:
        logger.warning(f"   ⚠️ Fatigue Score features failed: {e}")
        # Fallback: adicionar placeholders neutros
        df['home_fatigue_score'] = 0.0
        df['away_fatigue_score'] = 0.0
        df['home_distance_km'] = 0.0
        df['away_distance_km'] = 0.0

    # 3.5 v21.7 FASE 70%+: Injury Impact (Star players OUT afeta spread em ~3 pontos)
    # Math-Context: Jogadores com RAPM > 4 OUT = impacto de -10 a -15 pontos esperados
    try:
        from ml_pipeline.advanced_features import add_injury_impact
        df = add_injury_impact(df)
        logger.info("   ✅ v21.7 Injury Impact features added")
    except Exception as e:
        logger.warning(f"   ⚠️ Injury Impact features failed: {e}")
        # Fallback: adicionar placeholders neutros
        df['injury_impact_home'] = 0.0
        df['injury_impact_away'] = 0.0
        df['injury_impact_net'] = 0.0

    
    # 4. Validação de Rolling Features
    required_rolling = ['home_rolling_5_points', 'home_rolling_10_points',
                       'away_rolling_5_points', 'away_rolling_10_points']
    missing_features = [f for f in required_rolling if f not in df.columns]
    
    if missing_features:
        logger.error(f"❌ Features obrigatórias faltando: {missing_features}")
        return None if not apply_weights else (None, None)
        df['h2h_home_win_rate'] = 0.5
        df['h2h_avg_point_diff'] = 0.0
        df['h2h_games_played'] = 0
    
    # 7. Adicionar Player Impact Features (OPCIONAL)
    if enable_player_features:
        logger.info("👥 Adicionando player impact features (RAPM/BPM)...")
        try:
            # Tentar obter stats de jogadores
            df_players = get_cached_player_stats()
            
            if df_players is not None and not df_players.empty:
                # Agregar por time
                df_player_agg = aggregate_player_stats_by_team(df_players, top_n=5)
                
                # Merge com jogos
                df = merge_player_features_to_games(df, df_player_agg, fillna_strategy='median')
                
                logger.info(f"✅ Player features adicionadas com sucesso")
            else:
                logger.warning("⚠️ Stats de jogadores não disponíveis. Usando fallback (zeros).")
                # Adicionar features zeradas para manter consistência
                for prefix in ['home', 'away']:
                    for col in ['rapm_avg', 'rapm_top', 'rapm_std', 'bpm_avg', 'bpm_top', 'depth_score']:
                        df[f'{prefix}_{col}'] = 0.0
                df['rapm_diff'] = 0.0
                df['bpm_diff'] = 0.0
                df['depth_diff'] = 0.0
        except Exception as e:
            logger.error(f"❌ Erro ao adicionar player features: {e}")
            logger.warning("Continuando sem player features...")
    
    # Limpeza final
    df = df.dropna(subset=required_rolling).reset_index(drop=True)
    
    # Remover duplicados
    df['game_id'] = df['date'].astype(str) + '_' + df['home_team'] + '_' + df['away_team']
    df = df.drop_duplicates(subset=['game_id'], keep='first').reset_index(drop=True)
    df = df.drop(columns=['game_id'])
    
    logger.info(f"✅ Dados preparados (V13): {len(df)} jogos de {df['date'].min().date()} a {df['date'].max().date()}")
    logger.info(f"   Features totais: {len(df.columns)}")
    
    if apply_weights:
        weights = calculate_sample_weights(df, weight_config)
        return df, weights
    else:
        return df


def prepare_data_for_training(df, target='winner'):
    """
    Prepara X e y para treinamento usando WHITELIST de features seguras.
    
    SECURITY FIX v2.0: 
    - Removidas features de Smart Money (causam Training-Serving Skew)
    - Odds permitidas APENAS se explicitamente de abertura
    
    Features BLOQUEADAS (Data Leakage):
    - line_movement: Requer closing_odds (pós-jogo)
    - implied_prob_diff: Requer closing_odds (pós-jogo)
    - smart_money_signal: Derivado de closing_odds
    - odds_home/odds_away genéricos: Ambíguos (podem ser closing)
    """
    
    # AUDIT FIX: BLACKLIST explícita de features perigosas
    BLACKLISTED_FEATURES = {
        # Closing odds e features derivadas (futuro)
        'line_movement',
        'implied_prob_diff', 
        'smart_money_signal',
        'closing_odds',
        'closing_odds_home',
        'closing_odds_away',
        
        # 🚨 CRITICAL LEAKAGE FIX: Four Factors RAW do jogo atual
        # Estas features são calculadas do BOX SCORE após o jogo terminar
        # Evidência: Feature importance de 17% e 14% (top 2 features!)
        'home_efg',           # eFG% do jogo atual (calculado de fgm/fga)
        'away_efg',  # eFG% do jogo atual
        'home_tov_pct',       # TOV% do jogo atual
        'away_tov_pct',
        'home_orb_pct',       # ORB% do jogo atual
        'away_orb_pct',
        'home_ftr',           # FT Rate do jogo atual
        'away_ftr',
        'home_pace',          # Pace do jogo atual (calculado de posses)
        'away_pace',
        'home_ts_pct',        # True Shooting % do jogo atual
        'away_ts_pct',
        'home_off_rating',    # Offensive rating do jogo atual
        'away_off_rating',
        'home_def_rating',    # Defensive rating do jogo atual
        'away_def_rating',
        'home_pie',           # PIE do jogo atual
        'away_pie',
        
        # 🚨 VAZAMENTO CRÍTICO: ortg_adj e drtg_adj BASE (não rolling)
        # Identificado via teste granular: causam salto de 65% → 95% accuracy
        # Evidência: opponent_adjusted_stats.py linhas 142-165
        # Problema: Usam home_off_rating e away_def_rating DO JOGO ATUAL
        # Exemplo: home_ortg_adj = home_off_rating - liga_avg + away_def_rating
        #          home_off_rating = home_score (do jogo atual!)
        # Nota: rolling_ortg_adj são SEGURAS (usam shift), mas BASE não!
        'home_ortg_adj',      # Calculated from home_off_rating (current game)
        'away_ortg_adj',
        'home_drtg_adj',      # Calculated from home_def_rating (current game)
        'away_drtg_adj',
    }
    
    # WHITELIST: Prefixos SEGUROS (calculados ANTES do jogo)
    SAFE_PREFIXES = (
        'rolling_',     # Médias móveis históricas (shift aplicado)
        'rest_',        # Dias de descanso (calculado antes)
        'elo_',         # Elo ratings (snapshot pré-jogo)
        'context_',     # Features contextuais (travel, b2b)
        'interaction_', # Features de interação (derivadas de rolling)
        'referee_',     # Stats de árbitros (expanding window histórico)
        'h2h_',         # Head-to-head histórico
        # ❌ REMOVIDO: 'smart_money_', 'line_movement', 'implied_prob_diff'
    )
    
    # FIX: Padrões que podem aparecer em QUALQUER posição (não apenas no início)
    # ⚠️ LEAKAGE FIX: Removidas features RAW calculadas do score do jogo atual
    SAFE_CONTAINS_PATTERNS = (
        '_rolling_',    # home_rolling_10_points,away_rolling_5_efg, etc. (SHIFT aplicado)
        '_elo',         # home_elo, away_elo (PRÉ-JOGO)
        '_rest_',       # Variações de rest days
        '_b2b',         # Back-to-back
        '_games_in_',   # Schedule density
        '_ortg_adj',    # Offensive rating AJUSTADO (histórico)
        '_drtg_adj',    # Defensive rating AJUSTADO (histórico)
        '_rapm',        # RAPM (player impact)
        '_bpm',         # BPM (player impact)
        # ❌ REMOVIDO: '_shooting_luck' - Pode usar dados do jogo atual via rolling_efg
        '_volatility',  # Volatility features
        '_trend',       # Trend features
        # ✅ FASE 70%+: Features de fadiga e lesões (PRÉ-JOGO)
        '_fatigue',     # fatigue_score, fatigue_index
        '_injury_impact',  # injury_impact_home, injury_impact_away
        '_distance',    # travel_distance_km
        # ❌ REMOVIDO (LEAKAGE): '_off_rating', '_def_rating', '_efg_pct', '_ts_pct', '_pace', '_net_rating'
        # Estas são calculadas do score do jogo atual, não de dados históricos
    )
    
    # WHITELIST: Colunas específicas seguras (todas PRÉ-JOGO)
    # ⚠️ LEAKAGE FIX: Removidas features calculadas do score do jogo atual
    SAFE_EXACT_COLS = {
        'home_elo', 'away_elo', 'elo_diff',
        'home_rest_days', 'away_rest_days', 'rest_diff',
        'home_is_back_to_back', 'away_is_back_to_back',
        'home_b2b', 'away_b2b',
        'home_games_in_7d', 'away_games_in_7d',
        'home_net_rating_trend', 'away_net_rating_trend',  # Trend de jogos ANTERIORES
        'home_sos_10', 'away_sos_10',
        'home_altitude_advantage', 'away_altitude_advantage',
        'home_win_streak', 'away_win_streak',
        'home_ortg_adj', 'away_ortg_adj',  # Ratings AJUSTADOS (histórico)
        'home_drtg_adj', 'away_drtg_adj',
        # ✅ FASE 70%+: Fatigue e Injury features (PRÉ-JOGO)
        'home_fatigue_score', 'away_fatigue_score',
        'home_distance_km', 'away_distance_km',
        'injury_impact_home', 'injury_impact_away', 'injury_impact_net',
        'home_injury_impact', 'away_injury_impact', 'total_injury_impact',
        'fatigue_index',
        # ✅ V27.0 ENTERPRISE: Granular Rest Advantage features
        'net_rest_days', 'rest_advantage',
        'rest_disadvantage_home', 'rest_disadvantage_away',
        'home_travel_km_3d', 'away_travel_km_3d', 'travel_km_advantage',
        # ❌ REMOVIDO (LEAKAGE): home_efg, home_pace, home_ftr, home_orb_pct, home_tov_pct
        # ❌ REMOVIDO (LEAKAGE): home_off_rating, home_def_rating, home_pie
        # Estas features são calculadas do BOX SCORE do jogo atual (pós-jogo)
    }
    
    # WHITELIST: Odds APENAS se EXPLICITAMENTE de abertura
    # ❌ REMOVIDO: 'odds_home', 'odds_away' (ambíguos - podem ser closing)
    SAFE_ODDS_PATTERNS = ('opening_odds_home', 'opening_odds_away', 'opening_spread')
    
    # Aplicar whitelist com blacklist check
    safe_cols = []
    blocked_cols = []
    
    for col in df.columns:
        # AUDIT FIX: Check blacklist primeiro (prioridade)
        # P0 LEAKAGE FIX: Added smart_money, public_pct, and expanded filters
        leakage_terms = ['closing', 'line_movement', 'implied_prob', 'smart_money', 'public_pct', 'midpoint', 'price']
        if col in BLACKLISTED_FEATURES or any(bl in col.lower() for bl in leakage_terms):
            blocked_cols.append(col)
            continue
            
        # Check exact match
        if col in SAFE_EXACT_COLS:
            safe_cols.append(col)
            continue
            
        # Check prefixes
        if any(col.startswith(prefix) for prefix in SAFE_PREFIXES):
            safe_cols.append(col)
            continue
        
        # FIX: Check CONTAINS patterns (para home_rolling_*, away_rolling_*, etc.)
        if any(pattern in col for pattern in SAFE_CONTAINS_PATTERNS):
            safe_cols.append(col)
            continue
            
        # Check odds patterns (apenas opening explícito)
        if any(pattern in col.lower() for pattern in SAFE_ODDS_PATTERNS):
            safe_cols.append(col)
    
    # AUDIT FIX: Log para auditoria de features bloqueadas
    if blocked_cols:
        logger.warning(f"🚫 Features BLOQUEADAS por risco de leakage: {blocked_cols}")
    
    # X: Apenas features seguras e numéricas
    available_safe_cols = [c for c in safe_cols if c in df.columns]
    X = df[available_safe_cols].select_dtypes(include=[np.number])
    
    # 🚨 CRITICAL LEAKAGE FIX (Opção C): Force Drop de Features RAW
    # Estas features são criadas em calculate_four_factors() ANTES de prepare_data_for_training
    # ser chamada, então elas já estão no df e podem passar pela whitelist.
    # Evidência: Feature importance mostrou home_efg (14%) e away_efg (17%) como top 2!
    # Solução: Forçar drop explícito aqui para garantir que NUNCA entrem no treino.
    FORCE_DROP_RAW_FEATURES = [
        'home_efg', 'away_efg',           # eFG% do jogo atual (TOP 1 e 2 em importance!)
        'home_tov_pct', 'away_tov_pct',   # TOV% do jogo atual
        'home_orb_pct', 'away_orb_pct',   # ORB% do jogo atual  
        'home_ftr', 'away_ftr',           # FT Rate do jogo atual
        'home_pace', 'away_pace',         # Pace do jogo atual
        'home_ts_pct', 'away_ts_pct',     # True Shooting % do jogo atual
        'home_off_rating', 'away_off_rating',  # Offensive rating do jogo atual
        'home_def_rating', 'away_def_rating',  # Defensive rating do jogo atual
        'home_pie', 'away_pie',           # PIE do jogo atual
    ]
    
    force_dropped = [col for col in FORCE_DROP_RAW_FEATURES if col in X.columns]
    if force_dropped:
        logger.warning(f"🚨 FORCE DROP: Removendo {len(force_dropped)} features RAW que passaram pela whitelist: {force_dropped}")
        X = X.drop(columns=force_dropped)
    
    logger.info(f"🔒 Feature Whitelist: {len(X.columns)} features seguras")
    logger.info(f"🚫 Features bloqueadas: {len(blocked_cols)}")
    
    # Verificar se features críticas estão presentes
    critical_features = ['home_elo', 'away_elo', 'home_rolling_10_points', 'away_rolling_10_points']
    missing_critical = [f for f in critical_features if f not in X.columns]
    if missing_critical:
        logger.warning(f"⚠️ Features críticas faltando: {missing_critical}")
    
    # y: Target
    if target in df.columns:
        y = df[target]
    else:
        y = None
        
    return X, y
