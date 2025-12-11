"""
Pace Calculator - Cálculo Correto de Pace/Possessões

Este módulo implementa o cálculo profissional de Pace (possessões por 48 minutos)
usando a fórmula canônica de Dean Oliver.

Formula:
    Possessões = FGA + 0.44 * FTA - ORB + TOV
    Pace = 48 * (Possessões / Minutos)

Para jogos de 48 minutos (sem OT):
    Pace ≈ Possessões

Referência: Dean Oliver - "Basketball on Paper" (2003)
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_possessions(df: pd.DataFrame, prefix: str = '') -> pd.Series:
    """
    Calcula possessões estimadas usando box score stats.
    
    Formula (Dean Oliver):
        Poss = FGA + 0.44 * FTA - ORB + TOV
    
    Args:
        df: DataFrame com stats do jogo
        prefix: Prefixo das colunas ('home_', 'away_', '' ou 'opp_')
    
    Returns:
        Series com possessões calculadas
    """
    fga_col = f'{prefix}fga' if prefix else 'fga'
    fta_col = f'{prefix}fta' if prefix else 'fta'
    orb_col = f'{prefix}oreb' if prefix else 'oreb'
    tov_col = f'{prefix}tov' if prefix else 'tov'
    
    # Verificar se colunas existem
    required_cols = [fga_col, fta_col, orb_col, tov_col]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        logger.warning(f"⚠️ Colunas faltando para calcular possessões: {missing}")
        return pd.Series(0, index=df.index)
    
    # Formula
    possessions = (
        df[fga_col] + 
        0.44 * df[fta_col] - 
        df[orb_col] + 
        df[tov_col]
    )
    
    # Sanitize (não pode ser negativo)
    possessions = possessions.clip(lower=0)
    
    return possessions


def calculate_pace(df: pd.DataFrame, prefix: str = '', minutes: float = 48.0) -> pd.Series:
    """
    Calcula Pace (possessões por 48 minutos).
    
    Args:
        df: DataFrame com stats do jogo
        prefix: Prefixo das colunas
        minutes: Minutos do jogo (48 para regular, ajustar para OT se disponível)
    
    Returns:
        Series com Pace calculado
    """
    possessions = calculate_possessions(df, prefix)
    
    # Pace = 48 * (Poss / Minutes)
    pace = 48 * (possessions / minutes)
    
    return pace


def add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features de Pace para home e away teams.
    
    Colunas criadas:
        - home_possessions, away_possessions
        - home_pace, away_pace
    
    Args:
        df: DataFrame com box scores
    
    Returns:
        DataFrame com features de pace adicionadas
    """
    logger.info("📊 Calculando Pace features (Dean Oliver formula)...")
    
    # Calcular possessões
    df['home_possessions'] = calculate_possessions(df, prefix='')
    df['away_possessions'] = calculate_possessions(df, prefix='opp_')
    
    # Calcular Pace
    # Se tiver coluna de minutos, usar; senão assumir 48min
    if 'minutes' in df.columns:
        df['home_pace'] = 48 * (df['home_possessions'] / df['minutes'])
        df['away_pace'] = 48 * (df['away_possessions'] / df['minutes'])
    else:
        # Assumir 48min (regular game)
        df['home_pace'] = df['home_possessions']
        df['away_pace'] = df['away_possessions']
    
    # Validação
    pace_mean_home = df['home_pace'].mean()
    pace_mean_away = df['away_pace'].mean()
    
    logger.info(f"   ✅ Home Pace médio: {pace_mean_home:.1f} possessões/48min")
    logger.info(f"   ✅ Away Pace médio: {pace_mean_away:.1f} possessões/48min")
    
    # Sanity check (NBA pace típico: 95-105)
    if not (90 <= pace_mean_home <= 110) or not (90 <= pace_mean_away <= 110):
        logger.warning(
            f"⚠️ Pace fora do range esperado (90-110). "
            f"Verificar cálculo ou dados de input."
        )
    
    return df


def add_rolling_pace_features(df: pd.DataFrame, windows: list = [5, 10]) -> pd.DataFrame:
    """
    Adiciona rolling features de Pace (sem leakage).
    
    Features criadas:
        - home_rolling_5_pace, home_rolling_10_pace
        - away_rolling_5_pace, away_rolling_10_pace
    
    Args:
        df: DataFrame com pace calculado
        windows: Janelas de rolling (ex: [5, 10])
    
    Returns:
        DataFrame com rolling pace features
    """
    logger.info("🔄 Calculando Rolling Pace features...")
    
    # Verificar se pace já foi calculado
    if 'home_pace' not in df.columns or 'away_pace' not in df.columns:
        logger.warning("⚠️ Pace não calculado. Executando add_pace_features() primeiro...")
        df = add_pace_features(df)
    
    # Criar formato longo (long format)
    home_df = df[['date', 'home_team', 'home_pace']].copy()
    home_df.columns = ['date', 'team', 'pace']
    
    away_df = df[['date', 'away_team', 'away_pace']].copy()
    away_df.columns = ['date', 'team', 'pace']
    
    long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date'])
    
    # Calcular rolling (com shift para evitar leakage)
    for window in windows:
        col_name = f'rolling_{window}_pace'
        long_df[col_name] = long_df.groupby('team')['pace'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=min(3, window)).mean()
        )
        
        # Log de NaNs
        nan_count = long_df[col_name].isna().sum()
        if nan_count > 0:
            logger.debug(f"   {nan_count} NaNs em {col_name} (início de temporada)")
    
    # Merge de volta para home
    home_cols = ['date', 'team'] + [f'rolling_{w}_pace' for w in windows]
    home_merged = long_df[long_df['team'].isin(df['home_team'])][home_cols].copy()
    home_merged.columns = ['date', 'home_team'] + [f'home_{col}' for col in home_cols[2:]]
    
    df = df.merge(
        home_merged,
        on=['date', 'home_team'],
        how='left'
    )
    
    # Merge de volta para away
    away_merged = long_df[long_df['team'].isin(df['away_team'])][home_cols].copy()
    away_merged.columns = ['date', 'away_team'] + [f'away_{col}' for col in home_cols[2:]]
    
    df = df.merge(
        away_merged,
        on=['date', 'away_team'],
        how='left'
    )
    
    # Preencher NaNs (início de temporada) com 0 ou média da liga
    LEAGUE_AVG_PACE = 99.5  # NBA 2025-26 aprox
    
    for window in windows:
        for prefix in ['home', 'away']:
            col = f'{prefix}_rolling_{window}_pace'
            if col in df.columns:
                nan_count_before = df[col].isna().sum()
                df[col] = df[col].fillna(LEAGUE_AVG_PACE)
                
                if nan_count_before > 0:
                    logger.info(
                        f"   🔧 Preenchidos {nan_count_before} NaNs em '{col}' "
                        f"com league avg ({LEAGUE_AVG_PACE})"
                    )
    

if __name__ == '__main__':
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    print("🏀 Pace Calculator Demo\n")
    
    # Simular jogo
    sample_game = pd.DataFrame({
        'date': ['2025-11-20'],
        'home_team': ['LAL'],
        'away_team': ['BOS'],
        'fga': [88],
        'fta': [24],
        'oreb': [10],
        'tov': [14],
        'opp_fga': [85],
        'opp_fta': [22],
        'opp_oreb': [9],
        'opp_tov': [12]
    })
    
    # Calcular Pace
    result = add_pace_features(sample_game)
    
    print(f"Home Possessions: {result['home_possessions'].iloc[0]:.1f}")
    print(f"Away Possessions: {result['away_possessions'].iloc[0]:.1f}")
    print(f"Home Pace: {result['home_pace'].iloc[0]:.1f}")
    print(f"Away Pace: {result['away_pace'].iloc[0]:.1f}")
    
    print("\n✅ Demo completo!")
