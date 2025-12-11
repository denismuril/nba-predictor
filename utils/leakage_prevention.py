"""
PRAGMATIC Anti-Leakage Validation - Standalone Module

Uso:
    from utils.leakage_prevention import validate_rolling_features
    
    # Após criar rolling features:
    validate_rolling_features(df, team_col='team')  # Auto-detecta features
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def validate_rolling_feature(df, feature_name, team_col='team', threshold_pct=50):
    """
    Valida que rolling feature tem shift(1) aplicado corretamente.
    
    REGRA PRAGMÁTICA: ≥50% dos teams devem ter NaN no primeiro valor.
    Isso indica que shift(1) foi aplicado antes do rolling().
    
    Args:
        df: DataFrame com a feature
        feature_name: Nome da feature a validar
        team_col: Coluna de agrupamento (default: 'team')
        threshold_pct: % mínimo de teams com NaN (default: 50)
    
    Returns:
        bool: True se passou, False se falhou
    
    Raises:
        ValueError: Se leakage crítico detectado
    """
    if feature_name not in df.columns:
        logger.debug(f"Feature '{feature_name}' não encontrada, skip validation")
        return True
    
   # Pegar primeiro VALOR (primeira LINHA) por team
    # CRITICAL: Use nth(0) NOT first() - first() skips NaN!
    first_values = df.groupby(team_col)[feature_name].nth(0)
    nan_count = first_values.isna().sum()
    total_teams = len(first_values)
    
    if total_teams == 0:
        return True
    
    nan_pct = (nan_count / total_teams) * 100
    
    # Validação
    if nan_pct < threshold_pct:
        error_msg = (
            f"🚨 DATA LEAKAGE em '{feature_name}'!\n"
            f"   Apenas {nan_pct:.0f}% teams com NaN no 1º valor.\n"
            f"   Esperado ≥{threshold_pct}%.\n"
            f"   FIX: Garanta que shift(1) está ANTES do rolling()."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(f"✅ '{feature_name}': {nan_pct:.0f}% teams NaN (OK)")
    return True


def validate_rolling_features(df, team_col='team', threshold_pct=50, strict=True):
    """
    Valida TODAS as rolling features em um DataFrame.
    
    Auto-detecta colunas com 'rolling' no nome e valida cada uma.
    
    Args:
        df: DataFrame com features
        team_col: Coluna de agrupamento
        threshold_pct: % mínimo de NaN esperado
        strict: Se True, raise ValueError em leakage. Se False, apenas log warning.
    
    Returns:
        Dict com resultado: {'valid': bool, 'passed': [], 'failed': []}
    """
    # Auto-detectar rolling features
    rolling_features = [col for col in df.columns if 'rolling' in col.lower()]
    
    if not rolling_features:
        logger.info("Nenhuma rolling feature encontrada para validar")
        return {'valid': True, 'passed': [], 'failed': []}
    
    logger.info(f"🔍 Validando {len(rolling_features)} rolling features...")
    
    passed = []
    failed = []
    
    for feature in rolling_features:
        try:
            validate_rolling_feature(df, feature, team_col, threshold_pct)
            passed.append(feature)
        except ValueError as e:
            failed.append(feature)
            if strict:
                raise  # Re-raise para bloquear execução
            else:
                logger.warning(f"⚠️ Validação falhou (non-strict): {feature}")
    
    valid = len(failed) == 0
    
    if valid:
        logger.info(f"✅ Todas {len(passed)} rolling features validadas - SEM leakage!")
    else:
        logger.error(f"❌ {len(failed)}/{len(rolling_features)} features com leakage!")
    
    return {
        'valid': valid,
        'passed': passed,
        'failed': failed,
        'total': len(rolling_features)
    }


# Convenience function para uso rápido
def assert_no_leakage(df, team_col='team'):
    """
    Versão assertion-style - falha IMEDIATAMENTE se detectar leakage.
    
    Usage:
        df = add_rolling_features(df)
        assert_no_leakage(df)  # Crash if leakage detected
    """
    result = validate_rolling_features(df, team_col=team_col, strict=True)
    if not result['valid']:
        raise AssertionError(f"Data leakage detected in {len(result['failed'])} features!")
    return True


if __name__ == '__main__':
    # Test básico
    import numpy as np
    
    # Create test data with MULTIPLE teams
    test_df = pd.DataFrame({
        'team': ['LAL'] * 5 + ['GSW'] * 5,
        'points': [100, 110, 105, 115, 108, 95, 102, 98, 105, 100]
    })
    
    # IMPORTANT: Sort by team BEFORE applying rolling
    test_df = test_df.sort_values(['team']).reset_index(drop=True)
    
    # CORRECT rolling (with shift) - NO min_periods to ensure NaN
    test_df['rolling_3_points_OK'] = test_df.groupby('team')['points'].transform(
        lambda x: x.shift(1).rolling(3).mean()  # shift(1) → NaN at start
    )
    
    # Validate
    result = validate_rolling_features(test_df, team_col='team', strict=False)
    print(f"Test result: {result}")
    
    # Check that first value per team is NaN
    first_lal = test_df[test_df['team'] == 'LAL'].iloc[0]['rolling_3_points_OK']
    first_gsw = test_df[test_df['team'] == 'GSW'].iloc[0]['rolling_3_points_OK']
    
    assert pd.isna(first_lal), f"First LAL should be NaN, got {first_lal}"
    assert pd.isna(first_gsw), f"First GSW should be NaN, got {first_gsw}"
    assert result['valid'], f"Validation should pass for correct rolling! Got: {result}"
    
    print("✅ Self-test passed!")

