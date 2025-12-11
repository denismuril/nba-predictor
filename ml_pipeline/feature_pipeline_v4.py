"""
Feature Pipeline V4 - Totals R&D (Pace, Matchups, Volatility)

Extends V3 with advanced features specifically designed to improve Totals prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

def prepare_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering Pipeline V4 - Totals R&D.
    
    Steps (V3 + New V4 Steps):
        1. Pace Features (raw, rolling, expected)
        2. Four Factors (eFG%, TS%, TOV%, ORB%, FTR%)
        3. Opponent Adjustments (eFG% adjusted)
        4. Contextual (rest, B2B, schedule, travel)
        5. Calendar (day, month, season stage)
        6. Player/Roster (RAPM, BPM aggregations)
        7. H2H (head-to-head history)
        --- V4 NEW STEPS ---
        8. Advanced Pace (Projected, Mismatch)
        9. Matchup Efficiency (Off vs Def)
        10. Volatility & Trends (Std Dev)
    
    Args:
        df: Input DataFrame with game data
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("=" * 70)
    logger.info("🧪 FEATURE PIPELINE V4 - TOTALS R&D")
    logger.info("=" * 70)
    
    initial_shape = df.shape
    logger.info(f"📊 Input: {initial_shape[0]} rows, {initial_shape[1]} cols")
    
    # Ensure datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Execute pipeline steps in FIXED ORDER
    steps = [
        # --- V3 BASE STEPS ---
        ("1️⃣ Pace Features", add_pace_features_v3),
        ("2️⃣ Four Factors", add_four_factors_v3),
        ("3️⃣ Opponent Adjustments", add_opponent_adjustments_v3),
        ("4️⃣ Contextual Features", add_contextual_features_v3),
        ("5️⃣ Calendar Features", add_calendar_features_v3),
        ("6️⃣ Player Features", add_player_features_v3),
        ("7️⃣ H2H Features", add_h2h_features_v3),
        
        # --- V4 NEW STEPS ---
        ("8️⃣ Advanced Pace", add_advanced_pace_features_v4),
        ("9️⃣ Matchup Efficiency", add_matchup_efficiency_features_v4),
        ("🔟 Volatility & Trends", add_volatility_features_v4),
    ]
    
    for step_name, step_func in steps:
        cols_before = df.shape[1]
        logger.info(f"\n{step_name}")
        
        try:
            df = step_func(df)
            cols_after = df.shape[1]
            new_cols = cols_after - cols_before
            logger.info(f"   ✅ Added {new_cols} columns (total: {cols_after})")
        except Exception as e:
            logger.error(f"   ❌ FAILED: {e}")
            raise  # Don't continue if a step fails
            
    final_shape = df.shape
    logger.info(f"   Added: {final_shape[1] - initial_shape[1]} features")
    logger.info("=" * 70)
    
    return df

# ============================================================================
# V3 STEPS (IMPORTED)
# ============================================================================

# Import V3 steps directly to avoid code duplication
from ml_pipeline.feature_pipeline_v3 import (
    add_pace_features_v3,
    add_four_factors_v3,
    add_opponent_adjustments_v3,
    add_contextual_features_v3,
    add_calendar_features_v3,
    add_player_features_v3,
    add_h2h_features_v3
)

# ============================================================================
# STEP 8: ADVANCED PACE FEATURES
# ============================================================================

def add_advanced_pace_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 8: Add advanced pace metrics.
    
    Generates:
    - projected_pace: Average of both teams' rolling pace
    - pace_mismatch: Absolute difference in pace
    """
    # Ensure base pace features exist (from Step 1)
    if 'home_rolling_10_pace' not in df.columns:
        logger.warning(f"⚠️ Base pace features missing. Columns available: {[c for c in df.columns if 'pace' in c]}")
        return df
        
    # Projected Pace (Simple Average)
    df['projected_pace'] = (df['home_rolling_10_pace'] + df['away_rolling_10_pace']) / 2
    
    # Pace Mismatch (Fast vs Slow)
    df['pace_mismatch'] = (df['home_rolling_10_pace'] - df['away_rolling_10_pace']).abs()
    
    return df

# ============================================================================
# STEP 9: MATCHUP EFFICIENCY
# ============================================================================

def add_matchup_efficiency_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 9: Add matchup efficiency features (Off vs Def).
    
    Generates:
    - off_matchup_home: Home Off Rtg - Away Def Rtg
    - off_matchup_away: Away Off Rtg - Home Def Rtg
    - eff_sum: Sum of Offensive Ratings (Total scoring potential)
    """
    # Ensure Four Factors exist (from Step 2)
    required = ['home_rolling_10_off_rating', 'away_rolling_10_def_rating', 
                'away_rolling_10_off_rating', 'home_rolling_10_def_rating']
                
    if not all(col in df.columns for col in required):
        logger.warning("⚠️ Four Factors missing. Skipping matchup efficiency.")
        return df
        
    # Home Offense vs Away Defense (Positive = Home Advantage)
    # Note: Def Rtg is points allowed per 100 poss. Lower is better defense.
    # So: High Off Rtg - High Def Rtg (Bad Def) = High Score?
    # Actually: Off Rtg (115) - Def Rtg (110) = +5 (Net Rating approx)
    # But for Totals, we care about the SUM of points.
    
    # Matchup Advantage (Net Rating proxy)
    df['off_matchup_home'] = df['home_rolling_10_off_rating'] - df['away_rolling_10_def_rating']
    df['off_matchup_away'] = df['away_rolling_10_off_rating'] - df['home_rolling_10_def_rating']
    
    # Total Scoring Potential (Sum of Offenses)
    # This is a naive proxy for total points, assuming average defense
    df['eff_sum'] = df['home_rolling_10_off_rating'] + df['away_rolling_10_off_rating']
    
    # Defensive Struggle (Sum of Defenses)
    # Lower = Lower scoring game
    df['def_sum'] = df['home_rolling_10_def_rating'] + df['away_rolling_10_def_rating']
    
    return df

# ============================================================================
# STEP 10: VOLATILITY & TRENDS
# ============================================================================

def add_volatility_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 10: Add volatility (standard deviation) features.
    
    Generates:
    - home_scoring_std_10: Std Dev of Home points (last 10)
    - away_scoring_std_10: Std Dev of Away points (last 10)
    """
    # We need to calculate rolling std dev. 
    # Since the dataframe is game-based, we need to reconstruct team history.
    # This is expensive, so we'll use a simplified approach or reuse existing logic if possible.
    
    # Re-using logic from feature_engineering_v2/v3 but for std()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create team-centric view
    home_games = df[['date', 'home_team', 'home_score']].rename(
        columns={'home_team': 'team', 'home_score': 'pts'}
    )
    away_games = df[['date', 'away_team', 'away_score']].rename(
        columns={'away_team': 'team', 'away_score': 'pts'}
    )
    
    team_stats = pd.concat([home_games, away_games]).sort_values('date')
    
    # Calculate Rolling Std (Shifted to avoid leakage)
    # Group by team, rolling 10, std, shift 1
    team_stats['pts_std_10'] = team_stats.groupby('team')['pts'].transform(
        lambda x: x.rolling(window=10, min_periods=5).std().shift(1)
    )
    
    # Merge back to main df
    # Rename for merge
    home_std = team_stats[['date', 'team', 'pts_std_10']].rename(
        columns={'team': 'home_team', 'pts_std_10': 'home_scoring_std_10'}
    )
    away_std = team_stats[['date', 'team', 'pts_std_10']].rename(
        columns={'team': 'away_team', 'pts_std_10': 'away_scoring_std_10'}
    )
    
    # SAFETY: Drop duplicates to prevent merge explosion
    home_std = home_std.drop_duplicates(subset=['date', 'home_team'])
    away_std = away_std.drop_duplicates(subset=['date', 'away_team'])
    
    # Merge Home
    df = pd.merge(df, home_std, on=['date', 'home_team'], how='left')
    
    # Merge Away
    df = pd.merge(df, away_std, on=['date', 'away_team'], how='left')
    
    # Fill NaNs (first games of season)
    df['home_scoring_std_10'] = df['home_scoring_std_10'].fillna(10.0) # Default std ~10 pts
    df['away_scoring_std_10'] = df['away_scoring_std_10'].fillna(10.0)
    
    return df

# ============================================================================
# STEP 11: SHOOTING LUCK (REGRESSION TO THE MEAN)
# ============================================================================

def add_shooting_luck_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 11: Add Shooting Luck features (Mean Reversion Detection).
    
    Concept: Teams shooting unsustainably hot/cold will regress to their mean.
    
    Generates:
    - home_shooting_luck_ts: Short-term TS% - Long-term TS% (Hot if >0)
    - away_shooting_luck_ts: Same for away team
    - home_shooting_luck_efg: Short-term eFG% - Long-term eFG%
    - away_shooting_luck_efg: Same for away team
    
    Betting Signal:
    - Positive luck (+5%): Team is "hot", likely to regress → Fade them
    - Negative luck (-5%): Team is "cold", likely to improve → Back them
    """
    # Check for required rolling features
    ts_cols_needed = ['home_rolling_5_ts_pct', 'home_rolling_30_ts_pct',
                      'away_rolling_5_ts_pct', 'away_rolling_30_ts_pct']
    efg_cols_needed = ['home_rolling_5_efg_pct', 'home_rolling_30_efg_pct',
                       'away_rolling_5_efg_pct', 'away_rolling_30_efg_pct']
    
    # TS% Luck (True Shooting %)
    if all(col in df.columns for col in ts_cols_needed):
        # AUDIT FIX: fillna(0) para prevenir NaN quando rolling_30 não existe (início época)
        df['home_shooting_luck_ts'] = (
            df['home_rolling_5_ts_pct'].fillna(0) - df['home_rolling_30_ts_pct'].fillna(0)
        )
        df['away_shooting_luck_ts'] = (
            df['away_rolling_5_ts_pct'].fillna(0) - df['away_rolling_30_ts_pct'].fillna(0)
        )
        # Debug: contar quantos valores foram preenchidos
        nan_count = df[['home_rolling_5_ts_pct', 'home_rolling_30_ts_pct']].isna().sum().sum()
        if nan_count > 0:
            logger.debug(f"   🔧 fillna aplicado em {nan_count} valores TS% (início época)")
        logger.info("   ✅ TS% Shooting Luck calculated")
    else:
        logger.warning("   ⚠️ TS% rolling features missing. Skipping TS Luck.")
        df['home_shooting_luck_ts'] = 0.0
        df['away_shooting_luck_ts'] = 0.0
    
    # eFG% Luck (Effective Field Goal %)
    if all(col in df.columns for col in efg_cols_needed):
        # AUDIT FIX: fillna(0) para prevenir NaN quando rolling_30 não existe (início época)
        df['home_shooting_luck_efg'] = (
            df['home_rolling_5_efg_pct'].fillna(0) - df['home_rolling_30_efg_pct'].fillna(0)
        )
        df['away_shooting_luck_efg'] = (
            df['away_rolling_5_efg_pct'].fillna(0) - df['away_rolling_30_efg_pct'].fillna(0)
        )
        # Debug: contar quantos valores foram preenchidos
        nan_count = df[['home_rolling_5_efg_pct', 'home_rolling_30_efg_pct']].isna().sum().sum()
        if nan_count > 0:
            logger.debug(f"   🔧 fillna aplicado em {nan_count} valores eFG% (início época)")
        logger.info("   ✅ eFG% Shooting Luck calculated")
    else:
        logger.warning("   ⚠️ eFG% rolling features missing. Skipping eFG Luck.")
        df['home_shooting_luck_efg'] = 0.0
        df['away_shooting_luck_efg'] = 0.0
    
    # Composite Luck Score (average of both metrics)
    # AUDIT FIX: fillna(0) extra safety para garantir nenhum NaN escapa
    df['home_shooting_luck'] = (
        (df['home_shooting_luck_ts'] + df['home_shooting_luck_efg']) / 2
    ).fillna(0)
    df['away_shooting_luck'] = (
        (df['away_shooting_luck_ts'] + df['away_shooting_luck_efg']) / 2
    ).fillna(0)
    
    return df

# ============================================================================
# MODULAR FUNCTION: ADVANCED FEATURES ONLY
# ============================================================================

def prepare_advanced_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modular Feature Engineering - ADVANCED STEPS ONLY (V4).
    
    This function assumes that BASE features (Pace, Four Factors, Rolling Stats)
    have ALREADY been calculated by the main pipeline (e.g., in predict.py).
    
    It only adds the ADVANCED V4 steps:
        - Step 8: Advanced Pace (projected_pace, pace_mismatch)
        - Step 9: Matchup Efficiency (off_matchup_*, eff_sum)
        - Step 10: Volatility (scoring_std_10)
        - Step 11: Shooting Luck (regression to mean detection)
    
    Args:
        df: DataFrame with BASE features already present
        
    Returns:
        DataFrame with ADVANCED V4 features added
        
    Raises:
        Warning: If required base columns are missing, skips that step
    """
    logger.info("⚡ Applying ADVANCED Features Only (V4 Modular)...")
    
    initial_cols = df.shape[1]
    
    # Advanced Steps (8, 9, 10, 11)
    advanced_steps = [
        ("8️⃣ Advanced Pace", add_advanced_pace_features_v4),
        ("9️⃣ Matchup Efficiency", add_matchup_efficiency_features_v4),
        ("🔟 Volatility & Trends", add_volatility_features_v4),
        ("1️⃣1️⃣ Shooting Luck (Mean Reversion)", add_shooting_luck_features_v4),
    ]
    
    for step_name, step_func in advanced_steps:
        logger.info(f"   {step_name}")
        try:
            df = step_func(df)
        except Exception as e:
            logger.error(f"      ❌ FAILED: {e}")
            # Don't raise - allow pipeline to continue with partial features
    
    final_cols = df.shape[1]
    added = final_cols - initial_cols
    
    logger.info(f"   ✅ Added {added} advanced features (total: {final_cols})")
    
    return df


if __name__ == "__main__":
    # Test pipeline
    logging.basicConfig(level=logging.INFO)
    from ml_pipeline.data_preparation import load_historical_data
    
    print("🧪 Testing Feature Pipeline V4")
    df = load_historical_data()
    df = df.tail(200)
    
    df_features = prepare_features_v4(df)
    
    print("\n✅ Pipeline V4 complete!")
    print(f"New features sample:")
    print(df_features[['projected_pace', 'off_matchup_home', 'home_scoring_std_10']].tail())
