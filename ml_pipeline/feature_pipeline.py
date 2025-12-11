"""
Wrapper para adicionar domain expert features ao pipeline.

Usage em data_preparation.py:
    from ml_pipeline.feature_pipeline import add_all_features
    
    df = add_all_features(df)
"""
import pandas as pd
import logging
from ml_pipeline.advanced_features import add_domain_expert_features

logger = logging.getLogger(__name__)


def add_all_features(df: pd.DataFrame, include_domain=True) -> pd.DataFrame:
    """
    Wrapper que adiciona TODAS as features ao DataFrame.
    
    Args:
        df: DataFrame base com stats
        include_domain: Se True, adiciona domain expert features (default: True)
    
    Returns:
        DataFrame com todas as features adicionadas
    """
    logger.info("🎯 Adicionando features completas ao pipeline...")
    
    # Domain expert features (10 funcionais)
    if include_domain:
        logger.info("  → Domain expert features (10)")
        df = add_domain_expert_features(df)
    
    logger.info(f"✅ Pipeline completo: {len(df.columns)} colunas totais")
    return df


if __name__ == '__main__':
    # Test
    import numpy as np
    
    test_df = pd.DataFrame({
        'home_fga': [85],
        'away_fga': [88],
        'home_pts': [110],
        'away_pts': [108],
        'home_orb': [10],
        'home_drb': [32],
        'away_orb': [8],
        'away_drb': [35],
        'home_fg3a': [35],
        'away_fg3a': [38],
        'home_tov': [12],
        'away_tov': [14],
        'home_ast': [25],
        'away_ast': [23],
        'home_wins': [30],
        'home_losses': [15],
        'away_wins': [20],
        'away_losses': [25]
    })
    
    result = add_all_features(test_df)
    print(f"✅ Test: {len(result.columns)} colunas criadas")
