"""
Pace Calculator - Cálculo Correto de Pace/Possessões com Volatilidade

Este módulo implementa o cálculo profissional de Pace (possessões por 48 minutos)
usando a fórmula canônica de Dean Oliver + features de volatilidade para Totals.

Referência: Dean Oliver - "Basketball on Paper" (2003)
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_possessions(df: pd.DataFrame, prefix: str = '') -> pd.Series:
    """Calcula possessões estimadas usando box score stats."""
    fga_col = f'{prefix}fga' if prefix else 'fga'
    fta_col = f'{prefix}fta' if prefix else 'fta'
    orb_col = f'{prefix}oreb' if prefix else 'oreb'
    tov_col = f'{prefix}tov' if prefix else 'tov'
    
    required_cols = [fga_col, fta_col, orb_col, tov_col]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        logger.warning(f"⚠️ Colunas faltando: {missing}")
        return pd.Series(0, index=df.index)
    
    possessions = df[fga_col] + 0.44 * df[fta_col] - df[orb_col] + df[tov_col]
    return possessions.clip(lower=0)


def add_pace_volatility_features(df: pd.DataFrame, windows: list = [5, 10]) -> pd.DataFrame:
    """
    Adiciona features de volatilidade de Pace (NOVO - Melhoria #3).
    
    Features criadas:
        - home_rolling_X_pace_std, away_rolling_X_pace_std 
        - home_pace_trend_X, away_pace_trend_X
        - home_rolling_X_points_std, away_rolling_X_points_std
    """
    logger.info(f"📊 Calculando volatilidade de Pace (windows={windows})...")
    
    if 'home_pace' not in df.columns:
        logger.error("❌ Pace não calculado!")
        return df
    
    # Criar long format
    home_df = df[['date', 'home_team', 'home_pace']].copy()
    away_df = df[['date', 'away_team', 'away_pace']].copy()
    
    calc_points = 'home_score' in df.columns
    if calc_points:
        home_df['pts'] = df['home_score']
        away_df['pts'] = df['away_score']
    
    home_df.columns = ['date', 'team', 'pace'] + (['pts'] if calc_points else [])
    away_df.columns = ['date', 'team', 'pace'] + (['pts'] if calc_points else [])
    
    long = pd.concat([home_df, away_df]).sort_values(['team', 'date']).reset_index(drop=True)
    
    for w in windows:
        # Volatilidade (std)
        long[f'rolling_{w}_pace_std'] = long.groupby('team')['pace'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=3).std()
        ).fillna(0)
        
        if calc_points:
            long[f'rolling_{w}_points_std'] = long.groupby('team')['pts'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=3).std()
            ).fillna(0)
        
        # Tendência (slope)
        long[f'pace_trend_{w}'] = long.groupby('team')['pace'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=3).apply(
                lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s) >= 3 else 0, raw=True
            )
        ).fillna(0)
    
    # Merge de volta
    vol_cols = [f'rolling_{w}_pace_std' for w in windows]
    trend_cols = [f'pace_trend_{w}' for w in windows]
    if calc_points:
        vol_cols += [f'rolling_{w}_points_std' for w in windows]
    
    all_cols = ['date', 'team'] + vol_cols + trend_cols
    
    # Home
    home_long = long[long['team'].isin(df['home_team'])][all_cols].drop_duplicates()
    home_long.columns = ['date', 'home_team'] + [f'home_{c}' for c in vol_cols + trend_cols]
    df = df.merge(home_long, on=['date', 'home_team'], how='left')
    
    # Away  
    away_long = long[long['team'].isin(df['away_team'])][all_cols].drop_duplicates()
    away_long.columns = ['date', 'away_team'] + [f'away_{c}' for c in vol_cols + trend_cols]
    df = df.merge(away_long, on=['date', 'away_team'], how='left')
    
    logger.info(f"✅ {len(vol_cols + trend_cols) * 2} features de volatilidade criadas")
    return df
