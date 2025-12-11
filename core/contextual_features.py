"""
Contextual Features - Rest, Schedule, Travel

Features que capturam contexto do jogo:
- Dias de descanso (rest days)
- Back-to-back games
- Schedule density (jogos em X dias)
- Travel distance e fatigue
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_rest_days(df):
    """
    Calcula dias de descanso entre jogos para cada time.
    
    Returns:
        DataFrame com colunas home_rest_days, away_rest_days, rest_advantage
    """
    df = df.sort_values('date').copy()
    
    # Preparar dados long format (cada linha = um time jogando)
    home_games = df[['date', 'home_team']].copy()
    home_games.columns = ['date', 'team']
    home_games['location'] = 'home'
    
    away_games = df[['date', 'away_team']].copy()
    away_games.columns = ['date', 'team']
    away_games['location'] = 'away'
    
    all_games = pd.concat([home_games, away_games]).sort_values(['team', 'date'])
    
    # Calcular dias desde último jogo
    all_games['prev_game_date'] = all_games.groupby('team')['date'].shift(1)
    all_games['rest_days'] = (all_games['date'] - all_games['prev_game_date']).dt.days
    
    # Preencher NaN (primeiro jogo da temporada) com 3 dias
    all_games['rest_days'] = all_games['rest_days'].fillna(3)
    
    # Merge de volta ao df original
    home_rest = all_games[all_games['location'] == 'home'][['date', 'team', 'rest_days']]
    home_rest.columns = ['date', 'home_team', 'home_rest_days']
    
    away_rest = all_games[all_games['location'] == 'away'][['date', 'team', 'rest_days']]
    away_rest.columns = ['date', 'away_team', 'away_rest_days']
    
    # Drop existing columns to avoid _x, _y suffixes
    cols_to_drop = [c for c in ['home_rest_days', 'away_rest_days'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # SAFETY: Drop duplicates and NaNs to prevent merge explosion
    home_rest = home_rest.dropna(subset=['date', 'home_team']).drop_duplicates(subset=['date', 'home_team'])
    away_rest = away_rest.dropna(subset=['date', 'away_team']).drop_duplicates(subset=['date', 'away_team'])
    
    df = df.merge(home_rest, on=['date', 'home_team'], how='left')
    df = df.merge(away_rest, on=['date', 'away_team'], how='left')
    
    # Rest advantage (positivo = casa mais descansada)
    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
    
    return df


def detect_back_to_back(df):
    """
    Detecta jogos back-to-back (0 dias de descanso).
    
    Returns:
        DataFrame com colunas home_b2b, away_b2b
    """
    if 'home_rest_days' not in df.columns:
        df = calculate_rest_days(df)
    
    df['home_b2b'] = (df['home_rest_days'] == 1).astype(int)  # Jogo no dia seguinte
    df['away_b2b'] = (df['away_rest_days'] == 1).astype(int)
    
    return df


def calculate_schedule_density(df, window=7):
    """
    Calcula número de jogos nos últimos N dias.
    
    Args:
        window: Janela de dias (default: 7)
    
    Returns:
        DataFrame com colunas home_games_in_Xd, away_games_in_Xd
    """
    df = df.sort_values('date').copy()
    
    # Preparar long format
    home_games = df[['date', 'home_team']].copy()
    home_games.columns = ['date', 'team']
    
    away_games = df[['date', 'away_team']].copy()
    away_games.columns = ['date', 'team']
    
    all_games = pd.concat([home_games, away_games]).sort_values(['team', 'date'])
    
    # Contar jogos nos últimos N dias (rolling)
    all_games['games_in_window'] = 0
    
    for team in all_games['team'].unique():
        team_mask = all_games['team'] == team
        team_games = all_games[team_mask].copy()
        
        # Para cada jogo, contar quantos jogos teve nos últimos N dias
        games_count = []
        for idx, row in team_games.iterrows():
            game_date = row['date']
            window_start = game_date - timedelta(days=window)
            
            # Contar jogos entre window_start e game_date (excluindo o jogo atual)
            count = ((team_games['date'] >= window_start) & 
                    (team_games['date'] < game_date)).sum()
            games_count.append(count)
        
        all_games.loc[team_mask, 'games_in_window'] = games_count
    
    # Merge de volta
    home_density = all_games[all_games['team'].isin(df['home_team'].unique())]
    home_density = home_density.groupby(['date', 'team']).first().reset_index()
    home_density.columns = ['date', 'home_team', f'home_games_in_{window}d']
    
    away_density = all_games[all_games['team'].isin(df['away_team'].unique())]
    away_density = away_density.groupby(['date', 'team']).first().reset_index()
    away_density.columns = ['date', 'away_team', f'away_games_in_{window}d']
    
    # Drop existing columns to avoid _x, _y suffixes
    cols_to_drop = [c for c in [f'home_games_in_{window}d', f'away_games_in_{window}d'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    df = df.merge(home_density, on=['date', 'home_team'], how='left')
    df = df.merge(away_density, on=['date', 'away_team'], how='left')
    
    # Fill NaN com 0
    df[f'home_games_in_{window}d'] = df[f'home_games_in_{window}d'].fillna(0)
    df[f'away_games_in_{window}d'] = df[f'away_games_in_{window}d'].fillna(0)
    
    return df


def add_all_contextual_features(df):
    """
    Adiciona todas as features contextuais de uma vez.
    
    Returns:
        DataFrame com todas as features
    """
    logger.info("📅 Adicionando features contextuais (rest, schedule, travel)...")
    
    # Rest days
    df = calculate_rest_days(df)
    logger.info(f"   ✅ Rest days calculados")
    
    # Back-to-back
    df = detect_back_to_back(df)
    b2b_count = (df['home_b2b'] + df['away_b2b']).sum()
    logger.info(f"   ✅ Back-to-back detectados: {b2b_count} times em B2B")
    
    # Schedule density
    df = calculate_schedule_density(df, window=7)
    logger.info(f"   ✅ Schedule density calculado (7 dias)")
    
    return df
