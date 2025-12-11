"""
Range Validation Utilities for Feature Validation

Validates that features are within realistic NBA ranges to detect data corruption
or processing errors before making predictions.
"""
import logging
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Define realistic ranges for critical NBA features
FEATURE_RANGES = {
    # Rolling averages (per game)
    'home_rolling_10_pts': (80, 140),
    'away_rolling_10_pts': (80, 140),
    'home_rolling_5_pts': (80, 140),
    'away_rolling_5_pts': (80, 140),
    
    # Efficiency metrics (percentages as decimals)
    'home_rolling_10_efg_pct': (0.35, 0.65),
    'away_rolling_10_efg_pct': (0.35, 0.65),
    'home_rolling_10_ts_pct': (0.40, 0.70),
    'away_rolling_10_ts_pct': (0.40, 0.70),
    
    # Advanced stats
    'home_rolling_10_off_rating': (90, 130),
    'away_rolling_10_off_rating': (90, 130),
    'home_rolling_10_def_rating': (90, 130),
    'away_rolling_10_def_rating': (90, 130),
    
    # Pace
    'home_rolling_10_pace': (85, 110),
    'away_rolling_10_pace': (85, 110),
    'pace_average': (85, 110),
    'pace_differential': (-15, 15),
    
    # Four Factors
    'home_rolling_10_tov_pct': (5, 25),  # Now in percentage scale
    'away_rolling_10_tov_pct': (5, 25),
    'home_rolling_10_oreb_pct': (0.10, 0.40),
    'away_rolling_10_oreb_pct': (0.10, 0.40),
    'home_rolling_10_ft_rate': (0.10, 0.50),
    'away_rolling_10_ft_rate': (0.10, 0.50),
}


def validate_feature_ranges(
    df: pd.DataFrame,
    ranges: Dict[str, Tuple[float, float]] = None
) -> Dict[str, List[str]]:
    """
    Validate that features are within realistic ranges.
    
    Args:
        df: DataFrame with features
        ranges: Custom range dict (optional, defaults to FEATURE_RANGES)
    
    Returns:
        Dict with 'errors' and 'warnings' lists
    """
    if ranges is None:
        ranges = FEATURE_RANGES
    
    errors = []
    warnings = []
    
    for feature, (min_val, max_val) in ranges.items():
        if feature not in df.columns:
            continue  # Skip if feature not present
        
        # Check for values outside range
        out_of_range = df[
            (df[feature] < min_val) | (df[feature] > max_val)
        ]
        
        if len(out_of_range) > 0:
            n_bad = len(out_of_range)
            pct_bad = (n_bad / len(df)) * 100
            
            # Get some example bad values
            bad_values = out_of_range[feature].head(3).tolist()
            
            message = (
                f"{feature}: {n_bad} values ({pct_bad:.1f}%) outside "
                f"range [{min_val}, {max_val}]. "
                f"Examples: {[f'{v:.2f}' for v in bad_values]}"
            )
            
            # Critical if >10% of data is bad
            if pct_bad > 10:
                errors.append(message)
            else:
                warnings.append(message)
    
    return {'errors': errors, 'warnings': warnings}


def validate_and_log(df: pd.DataFrame, context: str = "Features") -> bool:
    """
    Validate features and log results.
    
    Args:
        df: DataFrame to validate
        context: Description of what's being validated
    
    Returns:
        True if validation passed (no errors), False otherwise
    """
    logger.info(f"🔍 Validating {context} ranges...")
    
    results = validate_feature_ranges(df)
    
    # Log warnings
    if results['warnings']:
        logger.warning(f"⚠️ Range validation warnings for {context}:")
        for warning in results['warnings']:
            logger.warning(f"   {warning}")
    
    # Log errors
    if results['errors']:
        logger.error(f"❌ Range validation ERRORS for {context}:")
        for error in results['errors']:
            logger.error(f"   {error}")
        return False
    
    if not results['warnings'] and not results['errors']:
        logger.info(f"✅ All {context} within valid ranges")
    
    return True


def validate_critical_features(df: pd.DataFrame) -> None:
    """
    Validate critical features exist and are not all NaN.
    
    Raises ValueError if critical features are missing or invalid.
    """
    CRITICAL_FEATURES = [
        'home_rolling_10_pts',
        'away_rolling_10_pts',
        'home_rolling_10_efg_pct',
        'away_rolling_10_efg_pct',
    ]
    
    missing = [f for f in CRITICAL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing critical features: {missing}. "
            f"Cannot make predictions without these features."
        )
    
    # Check for all-NaN features  
    all_nan = [
        f for f in CRITICAL_FEATURES 
        if df[f].isna().all()
    ]
    if all_nan:
        raise ValueError(
            f"Critical features are all NaN: {all_nan}. "
            f"This indicates a feature engineering failure."
        )
    
    logger.info(f"✅ All {len(CRITICAL_FEATURES)} critical features present and valid")
