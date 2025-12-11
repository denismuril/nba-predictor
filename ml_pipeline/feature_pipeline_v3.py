"""
Feature Pipeline V3 - Deterministic & Reproducible

Complete rewrite for perfect reproducibility between training and validation.

Architecture:
- Step-based pipeline
- Fixed execution order
- No conditionals
- Clear logging
- Easy to test

Author: Refactored 03/12/2025
"""

import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

def prepare_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering Pipeline V3 - Deterministic & Reproducible.
    
    CRITICAL: Order of steps is FIXED. Do not modify without retraining models.
    
    Steps (always in this order):
        1. Pace Features (raw, rolling, expected)
        2. Four Factors (eFG%, TS%, TOV%, ORB%, FTR%)
        3. Opponent Adjustments (eFG% adjusted)
        4. Contextual (rest, B2B, schedule, travel)
        5. Calendar (day, month, season stage)
        6. Player/Roster (RAPM, BPM aggregations)
        7. H2H (head-to-head history)
    
    Args:
        df: Input DataFrame with game data
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("=" * 70)
    logger.info("🔄 FEATURE PIPELINE V3 - DETERMINISTIC")
    logger.info("=" * 70)
    
    initial_shape = df.shape
    logger.info(f"📊 Input: {initial_shape[0]} rows, {initial_shape[1]} cols")
    
    # Ensure datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Execute pipeline steps in FIXED ORDER
    steps = [
        ("1️⃣ Pace Features", add_pace_features_v3),
        ("2️⃣ Four Factors", add_four_factors_v3),
        ("3️⃣ Opponent Adjustments", add_opponent_adjustments_v3),
        ("4️⃣ Contextual Features", add_contextual_features_v3),
        ("5️⃣ Calendar Features", add_calendar_features_v3),
        ("6️⃣ Player Features", add_player_features_v3),
        ("7️⃣ H2H Features", add_h2h_features_v3),
    ]
    
    for step_name, step_func in steps:
        cols_before = df.shape[1]
        logger.info(f"\n{step_name}")
        
        try:
            df = step_func(df)
            cols_after = df.shape[1]
            new_cols = cols_after - cols_before
            logger.info(f"   ✅ Added {new_cols} columns (total: {cols_after})")
            logger.info(f"   DEBUG: Shape after {step_name}: {df.shape}")
        except Exception as e:
            logger.error(f"   ❌ FAILED: {e}")
            raise  # Don't continue if a step fails
            
    final_shape = df.shape
    logger.info(f"   Added: {final_shape[1] - initial_shape[1]} features")
    logger.info("=" * 70)
    
    return df


# ============================================================================
# STEP 1: PACE FEATURES
# ============================================================================

def add_pace_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Add Pace features (Dean Oliver formula).
    
    Generates:
    - home_pace, away_pace (raw)
    - home_rolling_5_pace, away_rolling_5_pace
    - home_rolling_10_pace, away_rolling_10_pace
    - expected_pace_5, expected_pace_10
    - pace_differential
    """
    try:
        from ml_pipeline.pace_calculator import (
            add_pace_features,
            add_rolling_pace_features,
            add_combined_pace_features
        )
        
        # Raw pace
        df = add_pace_features(df)
        
        # Rolling pace
        df = add_rolling_pace_features(df, windows=[5, 10])
        
        # Expected/combined pace
        df = add_combined_pace_features(df)
        
    except ImportError:
        logger.warning("pace_calculator not available - skipping Pace features")
        # Add placeholder columns to maintain consistency
        for prefix in ['home', 'away']:
            for window in [5, 10]:
                df[f'{prefix}_rolling_{window}_pace'] = 100.0  # League avg
        df['expected_pace_5'] = 100.0
        df['expected_pace_10'] = 100.0
        df['pace_differential'] = 0.0
    
    return df


# ============================================================================
# STEP 2: FOUR FACTORS
# ============================================================================

def add_four_factors_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Add Four Factors with rolling averages.

    Generates rolling windows [5, 10, 30] for:
    - eFG%, TS%, TOV%, ORB%, FTR%

    v19.0: Also adds contextual Home/Away rolling features.
    """
    from ml_pipeline.feature_engineering_v2 import (
        add_rolling_four_factors,
        add_contextual_rolling_features
    )

    # Use existing implementation (it's robust)
    df = add_rolling_four_factors(df, windows=[5, 10, 30])

    # v19.0: Add context-aware rolling features (Home vs Away)
    try:
        df = add_contextual_rolling_features(df, window=10)
        logger.info("   ✅ v19.0 Context-aware rolling features added (Home/Away)")
    except Exception as e:
        logger.warning(f"   ⚠️ Context-aware rolling failed: {e}")

    return df


# ============================================================================
# STEP 3: OPPONENT ADJUSTMENTS
# ============================================================================

def add_opponent_adjustments_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Add opponent-adjusted features.
    
    Generates:
    - home_rolling_5_efg_adj, away_rolling_5_efg_adj
    - home_rolling_10_efg_adj, away_rolling_10_efg_adj
    """
    try:
        from ml_pipeline.opponent_adjustments import add_opponent_adjusted_efg
        df = add_opponent_adjusted_efg(df, windows=[5, 10])
    except ImportError:
        logger.warning("opponent_adjustments not available - skipping")
    except Exception as e:
        logger.warning(f"opponent_adjustments error: {e} - skipping")
    
    return df


# ============================================================================
# STEP 4: CONTEXTUAL FEATURES
# ============================================================================

def add_contextual_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Add contextual features (rest, B2B, schedule).
    
    Generates:
    - home_rest_days, away_rest_days
    - home_b2b, away_b2b
    - home_games_in_7d, away_games_in_7d
    - home_travel_dist, away_travel_dist (if available)
    """
    from core.contextual_features import add_all_contextual_features
    
    df = add_all_contextual_features(df)
    
    return df


# ============================================================================
# STEP 5: CALENDAR FEATURES
# ============================================================================

def add_calendar_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Add calendar features.
    
    Generates:
    - day_of_week, month, is_weekend
    - season_stage (early/mid/late)
    """
    # Day of week
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Month
    df['month'] = df['date'].dt.month
    
    # Weekend
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Season stage (simplified)
    df['days_into_season'] = (df['date'] - pd.Timestamp('2025-10-01')).dt.days
    df['season_stage'] = pd.cut(
        df['days_into_season'],
        bins=[-1, 60, 120, 365],
        labels=['early', 'mid', 'late']
    ).astype(str)
    
    # One-hot encode season stage
    stage_dummies = pd.get_dummies(df['season_stage'], prefix='season')
    df = pd.concat([df, stage_dummies], axis=1)
    df = df.drop(['season_stage', 'days_into_season'], axis=1)
    
    return df


# ============================================================================
# STEP 6: PLAYER FEATURES
# ============================================================================

def add_player_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6: Add player aggregation features.
    
    Generates:
    - home_rapm_avg, away_rapm_avg
    - home_bpm_avg, away_bpm_avg
    - rapm_diff, bpm_diff
    - depth_score, etc.
    """
    try:
        from ml_pipeline.player_aggregation import (
            get_cached_player_stats,
            aggregate_player_stats_by_team,
            merge_player_features_to_games
        )
        
        df_players = get_cached_player_stats()
        
        if df_players is not None and not df_players.empty:
            df_player_agg = aggregate_player_stats_by_team(df_players, top_n=5)
            df = merge_player_features_to_games(df, df_player_agg, fillna_strategy='median')
        else:
            # Fallback: add zeros
            _add_player_fallback_features(df)
            
    except Exception as e:
        logger.warning(f"Player features error: {e} - using fallback")
        _add_player_fallback_features(df)
    
    return df


def _add_player_fallback_features(df: pd.DataFrame) -> None:
    """Add zero-filled player features as fallback."""
    for prefix in ['home', 'away']:
        for col in ['rapm_avg', 'rapm_top', 'rapm_std', 'bpm_avg', 'bpm_top', 'depth_score']:
            df[f'{prefix}_{col}'] = 0.0
    
    df['rapm_diff'] = 0.0
    df['bpm_diff'] = 0.0
    df['depth_diff'] = 0.0


# ============================================================================
# STEP 7: H2H FEATURES
# ============================================================================

def add_h2h_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 7: Add head-to-head matchup features.
    
    Generates:
    - h2h_home_win_rate
    - h2h_avg_point_diff
    - h2h_games_played
    """
    try:
        from ml_pipeline.h2h_features import calculate_h2h_stats
        df = calculate_h2h_stats(df)
    except Exception as e:
        logger.warning(f"H2H features error: {e} - using fallback")
        df['h2h_home_win_rate'] = 0.5
        df['h2h_avg_point_diff'] = 0.0
        df['h2h_games_played'] = 0
    
    return df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Test pipeline
    logging.basicConfig(level=logging.INFO)
    
    from ml_pipeline.data_preparation import load_historical_data
    
    print("🧪 Testing Feature Pipeline V3")
    print("=" * 70)
    
    # Load sample data
    df = load_historical_data()
    df = df.tail(100)  # Test with last 100 games
    
    print(f"Loaded {len(df)} games for testing")
    
    # Run pipeline
    df_features = prepare_features_v3(df)
    
    print(f"\n✅ Pipeline complete!")
    print(f"Final shape: {df_features.shape}")
    print(f"\nSample columns:")
    print(df_features.columns.tolist()[:20])
