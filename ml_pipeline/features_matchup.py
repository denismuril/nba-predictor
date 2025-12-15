"""
Micro-Matchup Feature Engineering (Quant Edge)
===============================================
Advanced features analyzing specific matchup dynamics.

Features:
1. Defense vs Position Impact - How well does defense guard opponent's key scorer
2. Pace Clash Factor - Non-linear differential when pace styles clash
3. Style Mismatch Score - 3PT heavy vs paint-dominant matchups

Usage:
    from ml_pipeline.features_matchup import add_matchup_features
    df = add_matchup_features(df)

Author: NBA Predictor v24.0 - Quant Edge
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# LEAGUE AVERAGES (2024-25 Season Baseline)
# =============================================================================
LEAGUE_AVERAGES = {
    'pace': 100.8,
    'off_rtg': 113.5,
    'def_rtg': 113.5,
    'efg_pct': 0.535,
    'tov_pct': 0.125,
    'orb_pct': 0.265,
    'ftr': 0.280,
    'fg3_rate': 0.395,  # % of FGA from 3PT
    'paint_pts_pct': 0.43,  # % of points in paint
}

# Position mapping for defense analysis
POSITION_WEIGHTS = {
    'PG': {'perimeter': 0.9, 'paint': 0.1, 'mid': 0.0},
    'SG': {'perimeter': 0.7, 'paint': 0.1, 'mid': 0.2},
    'SF': {'perimeter': 0.5, 'paint': 0.2, 'mid': 0.3},
    'PF': {'perimeter': 0.3, 'paint': 0.5, 'mid': 0.2},
    'C': {'perimeter': 0.1, 'paint': 0.8, 'mid': 0.1},
}


# =============================================================================
# PACE CLASH ANALYSIS
# =============================================================================
def calculate_pace_clash_factor(
    home_pace: float,
    away_pace: float,
    league_avg_pace: float = LEAGUE_AVERAGES['pace']
) -> Dict[str, float]:
    """
    Calculate non-linear pace clash factor.
    
    When a fast-paced team plays a slow-paced team, the result is 
    often closer to the slower team's preferred pace (defense controls tempo).
    
    Args:
        home_pace: Home team's average pace
        away_pace: Away team's average pace
        league_avg_pace: League average for normalization
        
    Returns:
        Dict with pace-related features
    """
    # Normalize paces relative to league average
    home_pace_diff = (home_pace - league_avg_pace) / league_avg_pace if league_avg_pace > 0 else 0
    away_pace_diff = (away_pace - league_avg_pace) / league_avg_pace if league_avg_pace > 0 else 0
    
    # Linear pace differential
    pace_diff = home_pace - away_pace
    
    # Non-linear clash factor
    # When paces diverge significantly, the game tends toward the slower pace
    pace_clash_magnitude = abs(home_pace_diff - away_pace_diff)
    
    # Who controls tempo? Defense (slower) usually wins
    if home_pace > away_pace:
        # Home is faster, away controls tempo -> disadvantage for home
        tempo_control = -pace_clash_magnitude * 0.3
    else:
        # Home is slower, controls tempo -> advantage for home
        tempo_control = pace_clash_magnitude * 0.3
    
    # Squared term for high-magnitude clashes (very fast vs very slow)
    pace_clash_squared = pace_clash_magnitude ** 2 * np.sign(tempo_control)
    
    # Game pace prediction (weighted toward slower team)
    predicted_pace = (home_pace * 0.4 + away_pace * 0.6) if away_pace < home_pace else \
                     (home_pace * 0.6 + away_pace * 0.4)
    
    return {
        'pace_diff': pace_diff,
        'pace_clash_factor': tempo_control,
        'pace_clash_squared': pace_clash_squared,
        'predicted_game_pace': predicted_pace,
        'pace_volatility': pace_clash_magnitude,
    }


# =============================================================================
# DEFENSE VS POSITION ANALYSIS
# =============================================================================
def calculate_defense_vs_position(
    opp_star_position: str,
    def_perimeter_rating: float,
    def_paint_rating: float,
    def_mid_rating: float,
    league_avg_def: float = LEAGUE_AVERAGES['def_rtg']
) -> Dict[str, float]:
    """
    Calculate how well a team's defense guards an opponent's star position.
    
    Example: If opponent's star is a Center (paint scorer), and we're 
    elite at paint defense, we have an advantage.
    
    Args:
        opp_star_position: Primary position of opponent's best scorer
        def_perimeter_rating: Defensive rating vs perimeter (3PT+mid)
        def_paint_rating: Defensive rating vs paint
        def_mid_rating: Defensive rating vs mid-range
        league_avg_def: League average for normalization
        
    Returns:
        Dict with defense vs position features
    """
    if opp_star_position not in POSITION_WEIGHTS:
        opp_star_position = 'SF'  # Default to wing
    
    weights = POSITION_WEIGHTS[opp_star_position]
    
    # Weighted defensive rating against this position
    weighted_def = (
        weights['perimeter'] * def_perimeter_rating +
        weights['paint'] * def_paint_rating +
        weights['mid'] * def_mid_rating
    )
    
    # How much better/worse than league average
    def_vs_position_diff = (league_avg_def - weighted_def) / league_avg_def if league_avg_def > 0 else 0
    
    # Is this a favorable matchup? (positive = our defense is elite vs their star)
    favorable_matchup = 1 if def_vs_position_diff > 0.02 else (
        -1 if def_vs_position_diff < -0.02 else 0
    )
    
    return {
        'def_vs_position_rating': weighted_def,
        'def_vs_position_diff': def_vs_position_diff * 100,  # As percentage
        'favorable_matchup_flag': favorable_matchup,
    }


# =============================================================================
# STYLE MISMATCH ANALYSIS
# =============================================================================
def calculate_style_mismatch(
    home_fg3_rate: float,
    home_paint_pct: float,
    away_fg3_rate: float,
    away_paint_pct: float,
    away_def_3pt_pct: float,
    away_def_paint_pct_allowed: float,
    home_def_3pt_pct: float,
    home_def_paint_pct_allowed: float,
) -> Dict[str, float]:
    """
    Calculate style mismatch score.
    
    When a 3PT-heavy team plays against a paint-heavy team,
    the matchup dynamics differ from style-similar matchups.
    
    Args:
        home_fg3_rate: Home team's 3PT attempt rate
        home_paint_pct: Home team's % of points in paint
        away_*: Same for away team
        *_def_*: Defensive metrics (what they allow)
        
    Returns:
        Dict with style mismatch features
    """
    # Style classification
    # 3PT heavy: fg3_rate > 0.40, Paint heavy: paint_pct > 0.48
    
    home_style_score = home_fg3_rate - home_paint_pct  # Positive = perimeter, Negative = paint
    away_style_score = away_fg3_rate - away_paint_pct
    
    # Style clash magnitude
    style_clash = abs(home_style_score - away_style_score)
    
    # Specific mismatches
    # Home is 3PT heavy vs Away's 3PT defense
    if home_fg3_rate > LEAGUE_AVERAGES['fg3_rate']:
        home_3pt_mismatch = (away_def_3pt_pct - 0.35) / 0.35  # Positive = away is bad at 3PT D
    else:
        home_3pt_mismatch = 0
    
    # Home is paint heavy vs Away's paint defense  
    if home_paint_pct > LEAGUE_AVERAGES['paint_pts_pct']:
        home_paint_mismatch = (away_def_paint_pct_allowed - 0.43) / 0.43
    else:
        home_paint_mismatch = 0
    
    # Same for away team
    if away_fg3_rate > LEAGUE_AVERAGES['fg3_rate']:
        away_3pt_mismatch = (home_def_3pt_pct - 0.35) / 0.35
    else:
        away_3pt_mismatch = 0
        
    if away_paint_pct > LEAGUE_AVERAGES['paint_pts_pct']:
        away_paint_mismatch = (home_def_paint_pct_allowed - 0.43) / 0.43
    else:
        away_paint_mismatch = 0
    
    # Net mismatch advantage for home
    home_mismatch_advantage = (home_3pt_mismatch + home_paint_mismatch) - \
                               (away_3pt_mismatch + away_paint_mismatch)
    
    return {
        'style_clash_magnitude': style_clash,
        'home_style_score': home_style_score,
        'away_style_score': away_style_score,
        'home_mismatch_advantage': home_mismatch_advantage,
        'home_3pt_exploit': home_3pt_mismatch,
        'home_paint_exploit': home_paint_mismatch,
    }


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================
def add_matchup_features(
    df: pd.DataFrame,
    use_fallback: bool = True
) -> pd.DataFrame:
    """
    Add micro-matchup features to DataFrame.
    
    Expects columns:
    - home_pace, away_pace (or pace_rolling variants)
    - home_def_rtg, away_def_rtg
    - home_fg3a_rate, away_fg3a_rate (optional)
    
    Args:
        df: DataFrame with team stats
        use_fallback: If True, use league averages for missing data
        
    Returns:
        DataFrame with new matchup features
    """
    logger.info("🔧 Adding micro-matchup features...")
    df = df.copy()
    
    n_rows = len(df)
    
    # Initialize feature arrays
    pace_features = {
        'pace_diff': np.zeros(n_rows),
        'pace_clash_factor': np.zeros(n_rows),
        'pace_clash_squared': np.zeros(n_rows),
        'predicted_game_pace': np.full(n_rows, LEAGUE_AVERAGES['pace']),
        'pace_volatility': np.zeros(n_rows),
    }
    
    style_features = {
        'style_clash_magnitude': np.zeros(n_rows),
        'home_mismatch_advantage': np.zeros(n_rows),
    }
    
    # Get pace columns (try various naming conventions)
    home_pace_col = None
    away_pace_col = None
    
    for col in ['home_pace_avg_10', 'home_pace_rolling_10', 'home_pace_avg', 'home_pace']:
        if col in df.columns:
            home_pace_col = col
            break
    
    for col in ['away_pace_avg_10', 'away_pace_rolling_10', 'away_pace_avg', 'away_pace']:
        if col in df.columns:
            away_pace_col = col
            break
    
    # Calculate pace features if columns exist
    if home_pace_col and away_pace_col:
        for i in range(n_rows):
            home_pace = df[home_pace_col].iloc[i]
            away_pace = df[away_pace_col].iloc[i]
            
            # Handle NaN
            if pd.isna(home_pace):
                home_pace = LEAGUE_AVERAGES['pace']
            if pd.isna(away_pace):
                away_pace = LEAGUE_AVERAGES['pace']
            
            result = calculate_pace_clash_factor(home_pace, away_pace)
            for key, value in result.items():
                pace_features[key][i] = value
        
        logger.info(f"   ✅ Pace clash features calculated from {home_pace_col}, {away_pace_col}")
    else:
        logger.warning("   ⚠️ Pace columns not found, using zeros for pace features")
    
    # Add pace features to DataFrame
    for key, values in pace_features.items():
        df[f'matchup_{key}'] = values
    
    # Style mismatch (if we have the data)
    fg3_rate_home = df.get('home_fg3_rate', df.get('home_fg3a_rate', None))
    fg3_rate_away = df.get('away_fg3_rate', df.get('away_fg3a_rate', None))
    
    if fg3_rate_home is not None and fg3_rate_away is not None:
        # Simplified style clash (without full defensive data)
        df['matchup_style_clash_magnitude'] = abs(
            fg3_rate_home.fillna(LEAGUE_AVERAGES['fg3_rate']) - 
            fg3_rate_away.fillna(LEAGUE_AVERAGES['fg3_rate'])
        )
        logger.info("   ✅ Style clash features calculated")
    else:
        df['matchup_style_clash_magnitude'] = 0.0
        logger.info("   ⚠️ FG3 rate columns not found, style clash set to 0")
    
    # Interaction features
    if 'matchup_pace_diff' in df.columns:
        # Pace * Offensive Rating interaction
        if 'home_off_rtg_avg_10' in df.columns and 'away_off_rtg_avg_10' in df.columns:
            off_rtg_diff = df['home_off_rtg_avg_10'].fillna(110) - df['away_off_rtg_avg_10'].fillna(110)
            df['matchup_pace_ortg_interaction'] = df['matchup_pace_diff'] * off_rtg_diff / 100
            logger.info("   ✅ Pace-OffRtg interaction calculated")
    
    # Add default matchup advantage (can be enhanced with lineup data)
    df['matchup_home_mismatch_advantage'] = 0.0
    
    logger.info(f"   📊 Added {sum(1 for c in df.columns if c.startswith('matchup_'))} matchup features")
    
    return df


# =============================================================================
# QUICK INTEGRATION FUNCTION
# =============================================================================
def integrate_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick integration point for feature_engineering_v2.py.
    
    Call this after calculating rolling features.
    
    Args:
        df: DataFrame with rolling features
        
    Returns:
        DataFrame with matchup features added
    """
    try:
        return add_matchup_features(df)
    except Exception as e:
        logger.warning(f"⚠️ Matchup features failed: {e}, returning original DataFrame")
        return df


# =============================================================================
# CLI TEST
# =============================================================================
if __name__ == "__main__":
    print("🧪 Testing Micro-Matchup Features...")
    
    # Create dummy data
    dummy_df = pd.DataFrame({
        'home_pace_avg_10': [105, 95, 100, 110, 88],
        'away_pace_avg_10': [98, 102, 100, 92, 108],
        'home_off_rtg_avg_10': [115, 108, 112, 118, 105],
        'away_off_rtg_avg_10': [110, 115, 112, 108, 112],
        'home_fg3a_rate': [0.42, 0.35, 0.38, 0.45, 0.32],
        'away_fg3a_rate': [0.38, 0.40, 0.38, 0.35, 0.42],
    })
    
    result = add_matchup_features(dummy_df)
    
    print("\n📊 Matchup Features Added:")
    matchup_cols = [c for c in result.columns if c.startswith('matchup_')]
    print(result[matchup_cols].head())
    
    print("\n✅ Test completed successfully!")
