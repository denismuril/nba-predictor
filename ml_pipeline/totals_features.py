"""
Totals-Specific Features

Features engineered specifically for predicting total points (Over/Under).
Focuses on pace interactions and offensive/defensive synergies.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_totals_features(df):
    """
    Add features specifically optimized for Totals (Over/Under) prediction.
    
    Features added:
    - Pace interactions (product, harmonic mean, sum, min, max)
    - Offensive vs Defensive synergy
    - Combined offensive firepower
    - True Shooting combined metrics
    
    Args:
        df: DataFrame with game data including rolling stats
        
    Returns:
        DataFrame with additional totals-specific features
    """
    logger.info("🎯 Adding Totals-specific features...")
    
    df = df.copy()
    
    # ==================== PACE INTERACTIONS ====================
    # These are critical for totals - fast pace teams = high scoring
    
    if 'home_rolling_10_pace' in df.columns and 'away_rolling_10_pace' in df.columns:
        # Pace product: Both teams fast = very high total
        df['pace_product'] = df['home_rolling_10_pace'] * df['away_rolling_10_pace']
        
        # Harmonic mean: Better for averaging rates
        df['pace_harmonic_mean'] = 2 * (df['home_rolling_10_pace'] * df['away_rolling_10_pace']) / \
                                   (df['home_rolling_10_pace'] + df['away_rolling_10_pace'] + 1e-10)
        
        # Pace sum: Total game tempo
        df['pace_sum'] = df['home_rolling_10_pace'] + df['away_rolling_10_pace']
        
        # Pace min/max: Slowest team often dictates tempo
        df['pace_min'] = df[['home_rolling_10_pace', 'away_rolling_10_pace']].min(axis=1)
        df['pace_max'] = df[['home_rolling_10_pace', 'away_rolling_10_pace']].max(axis=1)
        
        logger.info("   ✅ Added 5 pace interaction features")
    else:
        logger.warning("   ⚠️  Pace columns not found, skipping pace interactions")
    
    # ==================== OFFENSIVE/DEFENSIVE SYNERGY ====================
    
    if all(col in df.columns for col in ['home_rolling_10_off_rating', 'away_rolling_10_off_rating',
                                           'home_rolling_10_def_rating', 'away_rolling_10_def_rating']):
        
        # Home offense vs Away defense
        df['home_off_vs_away_def'] = df['home_rolling_10_off_rating'] - df['away_rolling_10_def_rating']
        
        # Away offense vs Home defense  
        df['away_off_vs_home_def'] = df['away_rolling_10_off_rating'] - df['home_rolling_10_def_rating']
        
        # Combined offensive firepower
        df['combined_off_rating'] = df['home_rolling_10_off_rating'] + df['away_rolling_10_off_rating']
        
        # Combined defensive weakness (higher = both teams weak defense = high scoring)
        df['combined_def_rating'] = df['home_rolling_10_def_rating'] + df['away_rolling_10_def_rating']
        
        # Net offensive advantage
        df['net_off_advantage'] = (df['home_rolling_10_off_rating'] + df['away_rolling_10_off_rating']) - \
                                  (df['home_rolling_10_def_rating'] + df['away_rolling_10_def_rating'])
        
        logger.info("   ✅ Added 5 offensive/defensive synergy features")
    else:
        logger.warning("   ⚠️  Rating columns not found, skipping synergy features")
    
    # ==================== TRUE SHOOTING COMBINED ====================
    
    if 'home_rolling_10_ts_pct' in df.columns and 'away_rolling_10_ts_pct' in df.columns:
        # Combined TS%: Both teams efficient = higher scoring
        df['combined_ts_pct'] = (df['home_rolling_10_ts_pct'] + df['away_rolling_10_ts_pct']) / 2
        
        # TS differential: Large gap might indicate blowout (lower total)
        df['ts_differential'] = np.abs(df['home_rolling_10_ts_pct'] - df['away_rolling_10_ts_pct'])
        
        logger.info("   ✅ Added 2 True Shooting combined features")
    else:
        logger.warning("   ⚠️  TS% columns not found, skipping TS features")
    
    # ==================== SCORING TREND MOMENTUM ====================
    
    if all(col in df.columns for col in ['home_rolling_5_pts', 'home_rolling_10_pts',
                                           'away_rolling_5_pts', 'away_rolling_10_pts']):
        
        # Recent momentum (5-game vs 10-game)
        df['home_scoring_momentum'] = df['home_rolling_5_pts'] - df['home_rolling_10_pts']
        df['away_scoring_momentum'] = df['away_rolling_5_pts'] - df['away_rolling_10_pts']
        
        # Combined expected points (simple sum of recent averages)
        df['expected_total_simple'] = df['home_rolling_10_pts'] + df['away_rolling_10_pts']
        
        logger.info("   ✅ Added 3 scoring momentum features")
    
    # ==================== FOUR FACTORS FOR TOTALS ====================
    
    if all(col in df.columns for col in ['home_rolling_10_efg_pct', 'away_rolling_10_efg_pct']):
        # Combined shooting efficiency
        df['combined_efg_pct'] = (df['home_rolling_10_efg_pct'] + df['away_rolling_10_efg_pct']) / 2
        
        logger.info("   ✅ Added combined eFG%")
    
    # Fill any NaNs with 0 (shouldn't happen, but safety)
    new_cols = [col for col in df.columns if col not in df.columns or df[col].isna().any()]
    if new_cols:
        for col in new_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
    
    n_new_features = len([col for col in df.columns if any(x in col for x in 
                          ['pace_', 'combined_', 'synergy', 'momentum', 'expected_total'])])
    
    logger.info(f"✅ Totals features complete: ~{n_new_features} new features added")
    
    return df


def add_advanced_interactions(df):
    """
    Adiciona features avançadas de interação para modelo de Totals.
    
    FASE 1: Features de Interação (pace × ratings)
    
    Features:
    - pace × defensive_rating (sinergia ofensiva)
    - pace × offensive_rating (potencial de pontos)
    - ts_pct × pace (eficiência × volume)
    - rest_days × pace (fadiga × ritmo)
    - matchup_quality_score (score combinado)
    
    Args:
        df: DataFrame com features básicas de Totals
        
    Returns:
        DataFrame com 12 features adicionais de interação
    """
    logger.info("🔄 Adicionando features avançadas de interação...")
    
    df = df.copy()
    features_added = 0
    
    # ==================== PACE × DEFENSIVE RATING ====================
    # Time rápido contra defesa ruim = explosão de pontos
    
    if all(col in df.columns for col in ['home_rolling_10_pace', 'away_rolling_10_pace',
                                          'home_rolling_10_def_rating', 'away_rolling_10_def_rating']):
        
        # Home pace vs Away defense (120 - def_rating pois menor é melhor)
        df['home_pace_vs_away_defense'] = (
            df['home_rolling_10_pace'] * (120 - df['away_rolling_10_def_rating'])
        )
        
        # Away pace vs Home defense
        df['away_pace_vs_home_defense'] = (
            df['away_rolling_10_pace'] * (120 - df['home_rolling_10_def_rating'])
        )
        
        # Total esperado baseado em pace vs defesa
        df['total_pace_defense_interaction'] = (
            df['home_pace_vs_away_defense'] + df['away_pace_vs_home_defense']
        ) / 100  # Normalizar para escala razoável
        
        features_added += 3
        logger.info(f"   ✅ Adicionadas 3 features pace × defense")
    
    # ==================== PACE × OFFENSIVE RATING ====================
    # Mede potencial ofensivo considerando ritmo
    
    if all(col in df.columns for col in ['home_rolling_10_pace', 'away_rolling_10_pace',
                                          'home_rolling_10_off_rating', 'away_rolling_10_off_rating']):
        
        df['home_offensive_potential'] = (
            df['home_rolling_10_pace'] * df['home_rolling_10_off_rating'] / 100
        )
        df['away_offensive_potential'] = (
            df['away_rolling_10_pace'] * df['away_rolling_10_off_rating'] / 100
        )
        df['combined_offensive_potential'] = (
            df['home_offensive_potential'] + df['away_offensive_potential']
        )
        
        features_added += 3
        logger.info(f"   ✅ Adicionadas 3 features pace × offense")
    
    # ==================== TRUE SHOOTING × PACE ====================
    # Eficiência de arremesso × volume de posses
    
    if all(col in df.columns for col in ['home_rolling_10_ts_pct', 'away_rolling_10_ts_pct',
                                          'home_rolling_10_pace', 'away_rolling_10_pace']):
        
        df['home_scoring_efficiency'] = (
            df['home_rolling_10_ts_pct'] * df['home_rolling_10_pace']
        )
        df['away_scoring_efficiency'] = (
            df['away_rolling_10_ts_pct'] * df['away_rolling_10_pace']
        )
        df['total_scoring_efficiency'] = (
            df['home_scoring_efficiency'] + df['away_scoring_efficiency']
        )
        
        features_added += 3
        logger.info(f"   ✅ Adicionadas 3 features TS% × pace")
    
    # ==================== REST DAYS × PACE (FADIGA) ====================
    # Times cansados jogam mais devagar
    
    if all(col in df.columns for col in ['home_rest_days', 'away_rest_days',
                                          'home_rolling_10_pace', 'away_rolling_10_pace']):
        
        df['home_fatigue_impact'] = df['home_rest_days'] * df['home_rolling_10_pace']
        df['away_fatigue_impact'] = df['away_rest_days'] * df['away_rolling_10_pace']
        df['fatigue_differential'] = df['home_fatigue_impact'] - df['away_fatigue_impact']
        
        features_added += 3
        logger.info(f"   ✅ Adicionadas 3 features rest × pace (fadiga)")
    
    # ==================== MATCHUP QUALITY SCORE ====================
    # Score combinado que resume a qualidade do jogo
    
    required_cols = ['combined_offensive_potential', 'total_pace_defense_interaction', 
                     'total_scoring_efficiency']
    
    if all(col in df.columns for col in required_cols):
        df['matchup_quality_score'] = (
            df['combined_offensive_potential'] * 0.4 +
            df['total_pace_defense_interaction'] * 0.3 +
            df['total_scoring_efficiency'] * 0.3
        )
        features_added += 1
        logger.info(f"   ✅ Adicionado matchup_quality_score")
    
    # Total
    logger.info(f"✅ Features de interação completas: {features_added} novas features")
    
    return df
