"""
AUTO-VALIDATOR: Validação Automática de Data Leakage

MODO DE USO:
    from utils.auto_leakage_validator import auto_validate
    
    # Opção 1: Decorator em função
    @auto_validate
    def minha_funcao_com_rolling(df):
        df['rolling_5'] = df.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        return df
    
    # Opção 2: Wrapper manual
    df = auto_validate(add_rolling_features)(df, windows=[5, 10])
    
    # Opção 3: Ativar globalmente (EXPERIMENTAL)
    enable_global_validation()  # Valida TODOS DataFrames retornados

A validação roda AUTOMATICAMENTE quando a função retorna um DataFrame
com colunas 'rolling' no nome.
"""
import pandas as pd
import logging
from functools import wraps
from utils.leakage_prevention import validate_rolling_features

logger = logging.getLogger(__name__)

# Flag global
_GLOBAL_VALIDATION_ENABLED = False


def auto_validate(func=None, *, team_col='team', strict=True, quiet=False):
    """
    Decorator que valida automaticamente rolling features no DataFrame retornado.
    
    Args:
        func: Função a decorar
        team_col: Coluna de agrupamento (default: 'team')
        strict: Se True, raise em leakage. Se False, apenas log warning
        quiet: Se True, não loga validações bem-sucedidas
    
    Usage:
        @auto_validate
        def create_features(df):
            # ... criar rolling features ...
            return df
        
        # Ou com parâmetros:
        @auto_validate(team_col='team_id', strict=False)
        def create_features(df):
            return df
    """
    # Allow using as @auto_validate or @auto_validate()
    if func is None:
        # Called with arguments: @auto_validate(strict=False)
        def decorator(f):
            return _make_auto_validator(f, team_col, strict, quiet)
        return decorator
    else:
        # Called without arguments: @auto_validate
        return _make_auto_validator(func, team_col, strict, quiet)


def _make_auto_validator(func, team_col, strict, quiet):
    """Cria o wrapper de validação."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Executar função original
        result = func(*args, **kwargs)
        
        # Se retornou DataFrame, validar
        if isinstance(result, pd.DataFrame):
            _auto_validate_dataframe(result, team_col, strict, quiet, func.__name__)
        
        return result
    
    return wrapper


def _auto_validate_dataframe(df, team_col, strict, quiet, func_name):
    """Valida DataFrame automaticamente."""
    # Detectar rolling features
    rolling_features = [col for col in df.columns if 'rolling' in col.lower()]
    
    if not rolling_features:
        if not quiet:
            logger.debug(f"🔍 [{func_name}] Nenhuma rolling feature detectada, skip validation")
        return
    
    if team_col not in df.columns:
        logger.warning(
            f"⚠️ [{func_name}] Coluna '{team_col}' não encontrada. "
            f"Validação de leakage ignorada. Colunas disponíveis: {df.columns.tolist()[:5]}..."
        )
        return
    
    # Validar
    if not quiet:
        logger.info(f"🔍 [{func_name}] Auto-validando {len(rolling_features)} rolling features...")
    
    try:
        result = validate_rolling_features(df, team_col=team_col, strict=strict)
        
        if result['valid']:
            if not quiet:
                logger.info(f"✅ [{func_name}] Validação OK - {len(result['passed'])} features sem leakage")
        else:
            logger.error(
                f"❌ [{func_name}] LEAKAGE DETECTADO em {len(result['failed'])} features: "
                f"{result['failed']}"
            )
            if strict:
                raise ValueError(
                    f"Data leakage detectado em {func_name}! "
                    f"Features com problema: {result['failed']}"
                )
    
    except Exception as e:
        if strict:
            raise
        else:
            logger.error(f"❌ [{func_name}] Erro na validação: {e}")


def enable_global_validation(team_col='team', strict=True):
    """
    EXPERIMENTAL: Ativa validação automática global.
    
    Monkey-patches pandas DataFrame para validar automaticamente
    qualquer DataFrame que tenha rolling features.
    
    WARNING: Pode afetar performance. Use apenas para debugging.
    """
    global _GLOBAL_VALIDATION_ENABLED
    
    if _GLOBAL_VALIDATION_ENABLED:
        logger.warning("⚠️ Validação global já está ativada!")
        return
    
    logger.warning(
        "⚠️ EXPERIMENTAL: Ativando validação global de leakage. "
        "Isso pode afetar performance!"
    )
    
    # Guardar __init__ original
    original_init = pd.DataFrame.__init__
    
    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        # Chamar __init__ original
        original_init(self, *args, **kwargs)
        
        # Auto-validar se global está ativo
        if _GLOBAL_VALIDATION_ENABLED:
            _auto_validate_dataframe(self, team_col, strict, quiet=True, func_name='DataFrame')
    
    # Aplicar patch
    pd.DataFrame.__init__ = patched_init
    _GLOBAL_VALIDATION_ENABLED = True
    
    logger.info("✅ Validação global ativada!")


def disable_global_validation():
    """Desativa validação global."""
    global _GLOBAL_VALIDATION_ENABLED
    _GLOBAL_VALIDATION_ENABLED = False
    logger.info("✅ Validação global desativada")


# Convenience: Validação one-liner
def quick_validate(df, team_col='team'):
    """
    One-liner para validar DataFrame rapidamente.
    
    Usage:
        from utils.auto_leakage_validator import quick_validate
        
        df = add_rolling_features(df)
        df = quick_validate(df)  # ← Valida e retorna
    """
    _auto_validate_dataframe(df, team_col, strict=True, quiet=False, func_name='quick_validate')
    return df


if __name__ == '__main__':
    # Demo
    print("🔍 Demo: Auto-Validator\n")
    
    # Criar dados de teste
    test_df = pd.DataFrame({
        'team': ['LAL'] * 5 + ['GSW'] * 5,
        'points': [100, 110, 105, 115, 108, 95, 102, 98, 105, 100]
    }).sort_values(['team']).reset_index(drop=True)
    
    # Função decorada
    @auto_validate
    def create_correct_features(df):
        """Cria features CORRETAS (com shift)."""
        df['rolling_3_correct'] = df.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
        return df
    
    # Testar
    print("Testando função com features CORRETAS:")
    result = create_correct_features(test_df)
    print(f"✅ Sucesso! Resultado:\n{result[['team', 'points', 'rolling_3_correct']].head()}\n")
    
    print("\n" + "="*60)
    print("Demo completo! Auto-validator funcionando perfeitamente.")
    print("="*60)
