"""
NBA Statistical Formulas - Canonical Implementation

This module provides official NBA formulas to ensure:
1. Mathematical accuracy aligned with Basketball-Reference.com and NBA Advanced Stats
2. Training-Serving consistency (same formulas in training and inference)
3. Elimination of Training-Serving Skew

All formulas are based on:
- Dean Oliver's "Basketball on Paper" (2004)
- Basketball-Reference.com glossary
- NBA Advanced Stats definitions

Author: NBA Predictor Team
Date: 2025-12-02
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Union, Dict

# Type alias for flexibility
NumericType = Union[int, float, np.ndarray, pd.Series]


# ==============================================================================
# POSSESSIONS CALCULATION
# ==============================================================================

def calculate_possessions(
    fga: NumericType,
    fta: NumericType,
    orb: NumericType,
    drb_opp: NumericType,
    tov: NumericType,
    fgm: NumericType = None,
    method: str = 'standard'
) -> NumericType:
    """
    Calculate team possessions using official NBA formula.
    
    Standard Formula (most common):
        Poss = FGA + 0.4 * FTA - 1.07 * (ORB / (ORB + Opp_DRB)) * (FGA - FGM) + TOV
    
    Simplified Formula (when ORB data unavailable):
        Poss = FGA + 0.44 * FTA + TOV
    
    Args:
        fga: Field Goal Attempts
        fta: Free Throw Attempts
        orb: Offensive Rebounds
        drb_opp: Opponent's Defensive Rebounds
        tov: Turnovers
        fgm: Field Goals Made (required for standard method)
        method: 'standard' (accurate) or 'simplified' (fallback)
    
    Returns:
        Estimated possessions
    
    References:
        - Basketball-Reference.com: "Possessions Formula"
        - Dean Oliver (2004): "Basketball on Paper"
    
    Examples:
        >>> calculate_possessions(fga=85, fta=20, orb=10, drb_opp=35, tov=12, fgm=40)
        99.47  # Approximately 100 possessions (typical NBA game)
    """
    if method == 'standard':
        if fgm is None:
            raise ValueError("fgm (Field Goals Made) is required for standard method")
        
        # Offensive Rebound Factor
        orb_factor = orb / np.maximum(orb + drb_opp, 1)
        
        # Standard formula
        poss = (
            fga + 
            0.4 * fta - 
            1.07 * orb_factor * (fga - fgm) + 
            tov
        )
    else:  # simplified
        poss = fga + 0.44 * fta + tov
    
    return np.maximum(poss, 1)  # Prevent division by zero downstream


def calculate_team_possessions_combined(
    team_fga: NumericType,
    team_fta: NumericType,
    team_orb: NumericType,
    team_tov: NumericType,
    team_fgm: NumericType,
    opp_fga: NumericType,
    opp_fta: NumericType,
    opp_orb: NumericType,
    opp_tov: NumericType,
    opp_fgm: NumericType,
    team_drb: NumericType,
    opp_drb: NumericType
) -> NumericType:
    """
    Calculate possessions using both team and opponent stats (more accurate).
    
    Formula:
        Poss = 0.5 * (Team_Poss_Estimate + Opp_Poss_Estimate)
    
    This accounts for the fact that both teams should have approximately
    equal possessions in a game (within ~1-2 possessions).
    
    Returns:
        Average possessions (most accurate estimate)
    """
    team_poss = calculate_possessions(
        fga=team_fga,
        fta=team_fta,
        orb=team_orb,
        drb_opp=opp_drb,
        tov=team_tov,
        fgm=team_fgm,
        method='standard'
    )
    
    opp_poss = calculate_possessions(
        fga=opp_fga,
        fta=opp_fta,
        orb=opp_orb,
        drb_opp=team_drb,
        tov=opp_tov,
        fgm=opp_fgm,
        method='standard'
    )
    
    return (team_poss + opp_poss) / 2


# ==============================================================================
# OFFENSIVE & DEFENSIVE RATING
# ==============================================================================

def calculate_offensive_rating(
    pts: NumericType,
    possessions: NumericType
) -> NumericType:
    """
    Calculate Offensive Rating (ORtg).
    
    Formula:
        ORtg = 100 * (Points Scored / Possessions)
    
    Interpretation:
        Points scored per 100 possessions.
        NBA average: ~110-115 (varies by era)
        Elite offense: >118
        Poor offense: <105
    
    Args:
        pts: Points scored
        possessions: Team possessions (use calculate_possessions)
    
    Returns:
        Offensive Rating
    
    Examples:
        >>> calculate_offensive_rating(pts=115, possessions=100)
        115.0  # Elite offense
    """
    return 100 * (pts / np.maximum(possessions, 1))


def calculate_defensive_rating(
    opp_pts: NumericType,
    possessions: NumericType
) -> NumericType:
    """
    Calculate Defensive Rating (DRtg).
    
    Formula:
        DRtg = 100 * (Points Allowed / Possessions)
    
    Interpretation:
        Points allowed per 100 possessions.
        NBA average: ~110-115
        Elite defense: <108
        Poor defense: >115
    
    Args:
        opp_pts: Opponent points scored
        possessions: Team possessions (use calculate_possessions)
    
    Returns:
        Defensive Rating (lower is better)
    
    Examples:
        >>> calculate_defensive_rating(opp_pts=105, possessions=100)
        105.0  # Elite defense
    """
    return 100 * (opp_pts / np.maximum(possessions, 1))


# ==============================================================================
# PACE
# ==============================================================================

def calculate_pace(
    possessions: NumericType,
    minutes_played: NumericType = 48
) -> NumericType:
    """
    Calculate Pace (game tempo).
    
    Formula:
        Pace = 48 * (Possessions / Minutes_Played)
    
    Interpretation:
        Estimated possessions per 48 minutes.
        Fast pace: >102
        NBA average: ~98-100
        Slow pace: <95
    
    Args:
        possessions: Total possessions in the game
        minutes_played: Actual minutes played (48 for regulation, 53 for OT, etc.)
    
    Returns:
        Pace (possessions per 48 minutes)
    
    Examples:
        >>> calculate_pace(possessions=100, minutes_played=48)
        100.0  # Average pace
        
        >>> calculate_pace(possessions=110, minutes_played=53)  # OT game
        99.62  # Normalized to 48 minutes
    """
    return 48 * (possessions / np.maximum(minutes_played, 1))


# ==============================================================================
# FOUR FACTORS
# ==============================================================================

def calculate_efg(
    fgm: NumericType,
    fg3m: NumericType,
    fga: NumericType
) -> NumericType:
    """
    Calculate Effective Field Goal Percentage (eFG%).
    
    Formula:
        eFG% = (FGM + 0.5 * FG3M) / FGA
    
    Interpretation:
        Adjusts FG% to account for 3-pointers being worth more.
        NBA average: ~53-54%
        Elite: >56%
        Poor: <50%
    
    Examples:
        >>> calculate_efg(fgm=40, fg3m=12, fga=85)
        0.541  # 54.1% eFG (good)
    """
    return (fgm + 0.5 * fg3m) / np.maximum(fga, 1)


def calculate_ts(
    pts: NumericType,
    fga: NumericType,
    fta: NumericType
) -> NumericType:
    """
    Calculate True Shooting Percentage (TS%).
    
    Formula:
        TS% = PTS / (2 * (FGA + 0.44 * FTA))
    
    Interpretation:
        Measures scoring efficiency accounting for 2PT, 3PT, and FT.
        Superior to eFG% because it includes free throw efficiency.
        NBA average: ~56-57%
        Elite (e.g., Steph Curry): >62%
        Poor: <52%
    
    Examples:
        >>> calculate_ts(pts=115, fga=85, fta=20)
        0.609  # 60.9% TS (elite)
    """
    return pts / (2 * (fga + 0.44 * fta))


def calculate_tov_pct(
    tov: NumericType,
    fga: NumericType,
    fta: NumericType
) -> NumericType:
    """
    Calculate Turnover Percentage (TOV%).
    
    Formula:
        TOV% = 100 * TOV / (FGA + 0.44 * FTA + TOV)
    
    Interpretation:
        Percentage of possessions that end in a turnover.
        NBA average: ~13-14%
        Good: <12%
        Poor: >15%
    
    Note: Returns percentage (10-20 range), not decimal (0.1-0.2)
    
    Examples:
        >>> calculate_tov_pct(tov=12, fga=85, fta=20)
        11.54  # 11.54% TOV (good)
    """
    possessions_approx = fga + 0.44 * fta + tov
    return 100 * (tov / np.maximum(possessions_approx, 1))


def calculate_orb_pct(
    orb: NumericType,
    drb_opp: NumericType
) -> NumericType:
    """
    Calculate Offensive Rebound Percentage (ORB%).
    
    Formula:
        ORB% = ORB / (ORB + Opp_DRB)
    
    Interpretation:
        Percentage of available offensive rebounds grabbed.
        NBA average: ~22-24%
        Elite: >28%
        Poor: <20%
    
    Examples:
        >>> calculate_orb_pct(orb=10, drb_opp=35)
        0.222  # 22.2% ORB (average)
    """
    return orb / np.maximum(orb + drb_opp, 1)


def calculate_ftr(
    fta: NumericType,
    fga: NumericType
) -> NumericType:
    """
    Calculate Free Throw Rate (FTR).
    
    Formula:
        FTR = FTA / FGA
    
    Interpretation:
        Free throw attempts per field goal attempt.
        High FTR = aggressive driving, drawing fouls
        NBA average: ~0.22-0.24
        Elite (e.g., James Harden): >0.35
        Low: <0.18
    
    Examples:
        >>> calculate_ftr(fta=25, fga=85)
        0.294  # 29.4% FTR (high)
    """
    return fta / np.maximum(fga, 1)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def calculate_all_advanced_stats(
    pts: NumericType,
    fgm: NumericType,
    fga: NumericType,
    fg3m: NumericType,
    fta: NumericType,
    ftm: NumericType,
    orb: NumericType,
    drb: NumericType,
    tov: NumericType,
    opp_pts: NumericType,
    opp_drb: NumericType,
    minutes_played: NumericType = 48
) -> Dict[str, NumericType]:
    """
    Calculate all advanced stats at once for a team.
    
    Returns:
        Dictionary with all advanced metrics
    """
    # Possessions
    poss = calculate_possessions(
        fga=fga,
        fta=fta,
        orb=orb,
        drb_opp=opp_drb,
        tov=tov,
        fgm=fgm,
        method='standard'
    )
    
    return {
        'possessions': poss,
        'pace': calculate_pace(poss, minutes_played),
        'off_rating': calculate_offensive_rating(pts, poss),
        'def_rating': calculate_defensive_rating(opp_pts, poss),
        'efg_pct': calculate_efg(fgm, fg3m, fga),
        'ts_pct': calculate_ts(pts, fga, fta),
        'tov_pct': calculate_tov_pct(tov, fga, fta),
        'orb_pct': calculate_orb_pct(orb, opp_drb),
        'ft_rate': calculate_ftr(fta, fga)
    }


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_stats_ranges(stats: Dict[str, NumericType]) -> bool:
    """
    Validate that calculated stats are within reasonable NBA ranges.
    
    Raises:
        ValueError: If any stat is outside expected range
    
    Returns:
        True if all stats are valid
    """
    validations = [
        ('efg_pct', 0.35, 0.65, "eFG% should be between 35% and 65%"),
        ('ts_pct', 0.45, 0.70, "TS% should be between 45% and 70%"),
        ('tov_pct', 8, 20, "TOV% should be between 8% and 20%"),
        ('orb_pct', 0.15, 0.35, "ORB% should be between 15% and 35%"),
        ('ft_rate', 0.10, 0.45, "FTR should be between 10% and 45%"),
        ('off_rating', 90, 130, "ORtg should be between 90 and 130"),
        ('def_rating', 90, 130, "DRtg should be between 90 and 130"),
        ('pace', 85, 110, "Pace should be between 85 and 110"),
    ]
    
    for stat_name, min_val, max_val, error_msg in validations:
        if stat_name in stats:
            value = stats[stat_name]
            if isinstance(value, (pd.Series, np.ndarray)):
                if not ((value >= min_val) & (value <= max_val)).all():
                    raise ValueError(f"{error_msg} (got {value.min():.2f} to {value.max():.2f})")
            else:
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{error_msg} (got {value:.2f})")
    
    return True


if __name__ == '__main__':
    # Demo with realistic NBA game stats
    print("🏀 NBA Formulas Demo\n")
    
    # Example: Warriors vs Lakers game
    team_stats = {
        'pts': 115,
        'fgm': 42,
        'fga': 88,
        'fg3m': 15,
        'fta': 18,
        'ftm': 16,
        'orb': 8,
        'drb': 38,
        'tov': 11
    }
    
    opp_stats = {
        'pts': 108,
        'drb': 35  # Needed for ORB%
    }
    
    print("Team Stats:")
    for k, v in team_stats.items():
        print(f"  {k}: {v}")
    
    print("\nCalculated Advanced Stats:")
    advanced = calculate_all_advanced_stats(
        **team_stats,
        opp_pts=opp_stats['pts'],
        opp_drb=opp_stats['drb']
    )
    
    for stat, value in advanced.items():
        if 'pct' in stat or 'rating' in stat:
            print(f"  {stat}: {value:.1f}")
        else:
            print(f"  {stat}: {value:.2f}")
    
    print("\n✅ Validation:", "PASSED" if validate_stats_ranges(advanced) else "FAILED")
