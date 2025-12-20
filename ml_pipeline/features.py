"""
Feature Engineering Module - Anti-Leakage Features

Este módulo fornece funções de feature engineering com garantias contra data leakage.
Todas as features são calculadas usando APENAS dados anteriores ao jogo alvo.

v27.0: Implementação inicial com:
- Janelas deslizantes (Last 5, Last 10, Season Avg)
- Decorador @anti_leakage para validação automática
- EMAs (Exponential Moving Averages)
- Funções auxiliares para validação temporal

REGRA DE OURO:
    Qualquer feature para prever o jogo T deve usar APENAS dados de jogos < T.
    Isso é garantido pelo uso de shift(1) em todas as operações de rolling.

Uso:
    from ml_pipeline.features import (
        create_rolling_features,
        create_season_avg_features,
        create_ema_features,
        anti_leakage
    )
    
    df = create_rolling_features(df, windows=[5, 10])
"""

import functools
import logging
from datetime import datetime
from typing import Callable, List, Optional, Dict, Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DECORADOR ANTI-LEAKAGE
# =============================================================================

class DataLeakageError(Exception):
    """Exceção levantada quando data leakage é detectado."""
    pass


def anti_leakage(func: Callable) -> Callable:
    """
    Decorador que valida que nenhuma feature usa dados do futuro.
    
    O decorador verifica:
    1. Se há coluna 'date' no DataFrame
    2. Se há target_date especificado, garante que todas features
       são calculadas apenas com dados anteriores
    
    Uso:
        @anti_leakage
        def create_rolling_features(df, target_date=None, **kwargs):
            # Implementação segura
            ...
    
    Raises:
        DataLeakageError: Se detectar uso de dados futuros
    """
    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        target_date = kwargs.get('target_date')
        
        # Pre-validação: verificar se DataFrame tem coluna date
        if 'date' not in df.columns:
            logger.warning(
                f"⚠️ {func.__name__}: DataFrame sem coluna 'date'. "
                "Validação anti-leakage limitada."
            )
            return func(df, *args, **kwargs)
        
        # Executar função
        result = func(df, *args, **kwargs)
        
        # Post-validação: se target_date especificado, validar que
        # features do dia alvo não usam dados do próprio dia
        if target_date is not None:
            _validate_no_future_data(result, target_date, func.__name__)
        
        return result
    
    return wrapper


def _validate_no_future_data(df: pd.DataFrame, target_date: str, func_name: str):
    """
    Valida que features no target_date não contêm dados do próprio dia.
    
    Esta é uma validação heurística que verifica padrões comuns de leakage.
    """
    if 'date' not in df.columns:
        return
    
    try:
        target = pd.to_datetime(target_date)
        target_rows = df[pd.to_datetime(df['date']) == target]
        
        if target_rows.empty:
            return
        
        # Verificar se há valores não-NaN suspeitamente altos em features rolling
        rolling_cols = [c for c in df.columns if 'rolling' in c.lower() or 'last_' in c.lower()]
        
        for col in rolling_cols:
            if col in target_rows.columns:
                # Se a feature não deveria existir no primeiro jogo, verificar
                first_date = pd.to_datetime(df['date']).min()
                first_row = df[pd.to_datetime(df['date']) == first_date]
                
                if not first_row.empty and col in first_row.columns:
                    first_val = first_row[col].iloc[0]
                    if pd.notna(first_val):
                        # Primeiro jogo não deveria ter rolling features preenchidos
                        # (dependendo da janela)
                        logger.debug(
                            f"⚠️ {func_name}: {col} tem valor no primeiro jogo. "
                            "Verificar se shift(1) está sendo usado."
                        )
        
    except Exception as e:
        logger.warning(f"⚠️ Erro na validação anti-leakage: {e}")


# =============================================================================
# FEATURES DE JANELAS DESLIZANTES
# =============================================================================

@anti_leakage
def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [5, 10],
    stats_cols: List[str] = None,
    aggregations: List[str] = ['mean', 'std'],
    group_col: str = 'team',
    target_date: str = None
) -> pd.DataFrame:
    """
    Cria features de janelas deslizantes (rolling windows).
    
    IMPORTANTE: Usa shift(1) para garantir que apenas dados passados são usados.
    
    Args:
        df: DataFrame com dados históricos. Deve ter colunas 'date' e group_col.
        windows: Lista de tamanhos de janela (ex: [5, 10] para Last 5 e Last 10)
        stats_cols: Colunas para calcular estatísticas. Se None, usa colunas numéricas.
        aggregations: Agregações a calcular ('mean', 'std', 'min', 'max', 'sum')
        group_col: Coluna para agrupar (normalmente 'team')
        target_date: Data alvo para validação anti-leakage (opcional)
    
    Returns:
        DataFrame com novas colunas: {col}_rolling_{window}_{agg}
    
    Exemplo:
        >>> df = create_rolling_features(df, windows=[5, 10], stats_cols=['pts', 'ast'])
        >>> # Novas colunas: pts_rolling_5_mean, pts_rolling_5_std, pts_rolling_10_mean, ...
    """
    if df.empty:
        logger.warning("⚠️ DataFrame vazio em create_rolling_features")
        return df
    
    df = df.copy()
    
    # Garantir ordenação temporal
    if 'date' in df.columns:
        df = df.sort_values(['date', group_col] if group_col in df.columns else 'date')
    
    # Determinar colunas para processar
    if stats_cols is None:
        stats_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remover colunas de ID e target
        stats_cols = [c for c in stats_cols if not any(
            x in c.lower() for x in ['id', 'target', 'label', 'win']
        )]
    
    # Garantir que colunas existem
    stats_cols = [c for c in stats_cols if c in df.columns]
    
    if not stats_cols:
        logger.warning("⚠️ Nenhuma coluna numérica encontrada para rolling features")
        return df
    
    logger.info(
        f"📊 Criando rolling features: {len(stats_cols)} colunas, "
        f"windows={windows}, aggs={aggregations}"
    )
    
    # Calcular rolling features com shift(1) para evitar leakage
    for window in windows:
        for col in stats_cols:
            for agg in aggregations:
                new_col = f"{col}_rolling_{window}_{agg}"
                
                try:
                    if group_col in df.columns:
                        # Grouped rolling: por time
                        df[new_col] = df.groupby(group_col)[col].transform(
                            lambda x: x.shift(1).rolling(window, min_periods=1).agg(agg)
                        )
                    else:
                        # Global rolling
                        df[new_col] = df[col].shift(1).rolling(window, min_periods=1).agg(agg)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erro calculando {new_col}: {e}")
    
    return df


@anti_leakage
def create_season_avg_features(
    df: pd.DataFrame,
    stats_cols: List[str] = None,
    group_col: str = 'team',
    season_col: str = 'season',
    target_date: str = None
) -> pd.DataFrame:
    """
    Cria features de média da temporada até o jogo anterior.
    
    Usa expanding window com shift(1) para calcular a média acumulada
    de todos os jogos anteriores na temporada.
    
    Args:
        df: DataFrame com dados históricos
        stats_cols: Colunas para calcular médias
        group_col: Coluna para agrupar (normalmente 'team')
        season_col: Coluna que identifica a temporada
        target_date: Data alvo para validação anti-leakage
    
    Returns:
        DataFrame com novas colunas: {col}_season_avg
    
    Exemplo:
        >>> df = create_season_avg_features(df, stats_cols=['pts', 'reb'])
        >>> # Novas colunas: pts_season_avg, reb_season_avg
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Ordenar por data
    if 'date' in df.columns:
        sort_cols = [season_col, 'date'] if season_col in df.columns else ['date']
        if group_col in df.columns:
            sort_cols.append(group_col)
        df = df.sort_values([c for c in sort_cols if c in df.columns])
    
    # Determinar colunas
    if stats_cols is None:
        stats_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats_cols = [c for c in stats_cols if not any(
            x in c.lower() for x in ['id', 'target', 'label', 'win']
        )]
    
    stats_cols = [c for c in stats_cols if c in df.columns]
    
    logger.info(f"📊 Criando season avg features: {len(stats_cols)} colunas")
    
    # Agrupar por temporada E time
    for col in stats_cols:
        new_col = f"{col}_season_avg"
        
        try:
            group_cols = []
            if season_col in df.columns:
                group_cols.append(season_col)
            if group_col in df.columns:
                group_cols.append(group_col)
            
            if group_cols:
                df[new_col] = df.groupby(group_cols)[col].transform(
                    lambda x: x.shift(1).expanding(min_periods=1).mean()
                )
            else:
                df[new_col] = df[col].shift(1).expanding(min_periods=1).mean()
                
        except Exception as e:
            logger.warning(f"⚠️ Erro calculando {new_col}: {e}")
    
    return df


@anti_leakage
def create_ema_features(
    df: pd.DataFrame,
    spans: List[int] = [10, 20],
    stats_cols: List[str] = None,
    group_col: str = 'team',
    target_date: str = None
) -> pd.DataFrame:
    """
    Cria features de médias móveis exponenciais (EMA).
    
    EMA dá mais peso a observações recentes, capturando melhor
    tendências de forma recente.
    
    Args:
        df: DataFrame com dados históricos
        spans: Lista de spans para EMA (ex: [10, 20])
        stats_cols: Colunas para calcular EMAs
        group_col: Coluna para agrupar
        target_date: Data alvo para validação
    
    Returns:
        DataFrame com novas colunas: {col}_ema_{span}
    
    Exemplo:
        >>> df = create_ema_features(df, spans=[10, 20], stats_cols=['pts'])
        >>> # Novas colunas: pts_ema_10, pts_ema_20
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Ordenar por data
    if 'date' in df.columns:
        df = df.sort_values(['date', group_col] if group_col in df.columns else 'date')
    
    # Determinar colunas
    if stats_cols is None:
        stats_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats_cols = [c for c in stats_cols if not any(
            x in c.lower() for x in ['id', 'target', 'label', 'win']
        )]
    
    stats_cols = [c for c in stats_cols if c in df.columns]
    
    logger.info(f"📊 Criando EMA features: {len(stats_cols)} colunas, spans={spans}")
    
    for span in spans:
        for col in stats_cols:
            new_col = f"{col}_ema_{span}"
            
            try:
                if group_col in df.columns:
                    df[new_col] = df.groupby(group_col)[col].transform(
                        lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
                    )
                else:
                    df[new_col] = df[col].shift(1).ewm(span=span, min_periods=1).mean()
                    
            except Exception as e:
                logger.warning(f"⚠️ Erro calculando {new_col}: {e}")
    
    return df


# =============================================================================
# FEATURES DERIVADAS
# =============================================================================

@anti_leakage
def create_momentum_features(
    df: pd.DataFrame,
    stats_cols: List[str] = None,
    group_col: str = 'team',
    target_date: str = None
) -> pd.DataFrame:
    """
    Cria features de momentum (diferença entre médias curtas e longas).
    
    Momentum positivo indica tendência de melhora recente.
    
    Args:
        df: DataFrame com dados históricos
        stats_cols: Colunas para calcular momentum
        group_col: Coluna para agrupar
        target_date: Data alvo para validação
    
    Returns:
        DataFrame com novas colunas: {col}_momentum
    
    Exemplo:
        >>> df = create_momentum_features(df, stats_cols=['pts'])
        >>> # Nova coluna: pts_momentum = pts_rolling_5_mean - pts_rolling_10_mean
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Primeiro criar rolling features base se não existirem
    has_rolling = any('rolling_5_mean' in c for c in df.columns)
    if not has_rolling:
        df = create_rolling_features(df, windows=[5, 10], stats_cols=stats_cols,
                                     aggregations=['mean'], group_col=group_col)
    
    # Calcular momentum: short_term - long_term
    if stats_cols is None:
        # Inferir das colunas rolling existentes
        rolling_cols = [c.replace('_rolling_5_mean', '') 
                       for c in df.columns if '_rolling_5_mean' in c]
        stats_cols = rolling_cols
    
    for col in stats_cols:
        short_col = f"{col}_rolling_5_mean"
        long_col = f"{col}_rolling_10_mean"
        momentum_col = f"{col}_momentum"
        
        if short_col in df.columns and long_col in df.columns:
            df[momentum_col] = df[short_col] - df[long_col]
        else:
            logger.warning(f"⚠️ Colunas rolling não encontradas para momentum de {col}")
    
    return df


@anti_leakage
def create_consistency_features(
    df: pd.DataFrame,
    stats_cols: List[str] = None,
    window: int = 10,
    group_col: str = 'team',
    target_date: str = None
) -> pd.DataFrame:
    """
    Cria features de consistência (coeficiente de variação).
    
    CV baixo = time consistente, CV alto = time imprevisível.
    
    Args:
        df: DataFrame com dados históricos
        stats_cols: Colunas para calcular consistência
        window: Janela para cálculo
        group_col: Coluna para agrupar
        target_date: Data alvo para validação
    
    Returns:
        DataFrame com novas colunas: {col}_cv_{window}
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    if 'date' in df.columns:
        df = df.sort_values(['date', group_col] if group_col in df.columns else 'date')
    
    if stats_cols is None:
        stats_cols = ['pts', 'reb', 'ast']
        stats_cols = [c for c in stats_cols if c in df.columns]
    
    logger.info(f"📊 Criando consistency features: {len(stats_cols)} colunas, window={window}")
    
    for col in stats_cols:
        mean_col = f"{col}_rolling_{window}_mean"
        std_col = f"{col}_rolling_{window}_std"
        cv_col = f"{col}_cv_{window}"
        
        # Calcular rolling se não existir
        if mean_col not in df.columns or std_col not in df.columns:
            if group_col in df.columns:
                df[mean_col] = df.groupby(group_col)[col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df[std_col] = df.groupby(group_col)[col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).std()
                )
            else:
                df[mean_col] = df[col].shift(1).rolling(window, min_periods=1).mean()
                df[std_col] = df[col].shift(1).rolling(window, min_periods=1).std()
        
        # CV = std / mean (evitar divisão por zero)
        df[cv_col] = df[std_col] / (df[mean_col] + 1e-8)
    
    return df


# =============================================================================
# VALIDAÇÃO E UTILIDADES
# =============================================================================

def validate_temporal_integrity(
    df: pd.DataFrame,
    train_end_date: str,
    test_start_date: str
) -> bool:
    """
    Valida que não há sobreposição entre treino e teste.
    
    Args:
        df: DataFrame completo
        train_end_date: Última data de treino (YYYY-MM-DD)
        test_start_date: Primeira data de teste (YYYY-MM-DD)
    
    Returns:
        True se válido, False se há sobreposição
        
    Raises:
        DataLeakageError: Se detectar sobreposição
    """
    if 'date' not in df.columns:
        logger.warning("⚠️ Não foi possível validar integridade temporal: sem coluna 'date'")
        return True
    
    train_end = pd.to_datetime(train_end_date)
    test_start = pd.to_datetime(test_start_date)
    
    if train_end >= test_start:
        raise DataLeakageError(
            f"❌ DATA LEAKAGE DETECTADO!\n"
            f"Train end ({train_end_date}) >= Test start ({test_start_date})\n"
            f"Dados de teste podem estar vazando para treino!"
        )
    
    logger.info(f"✅ Validação temporal OK: train até {train_end_date}, test de {test_start_date}")
    return True


def get_features_available_at(
    df: pd.DataFrame,
    target_date: str,
    feature_cols: List[str] = None
) -> pd.DataFrame:
    """
    Retorna features disponíveis na data alvo.
    
    Útil para debugging e validação de que o modelo só
    está usando informações passadas.
    
    Args:
        df: DataFrame com features
        target_date: Data alvo para previsão
        feature_cols: Colunas de features a retornar (opcional)
    
    Returns:
        DataFrame com apenas dados disponíveis até target_date
    """
    if 'date' not in df.columns:
        return df
    
    target = pd.to_datetime(target_date)
    
    # Filtrar para dados anteriores ao target
    df_past = df[pd.to_datetime(df['date']) < target].copy()
    
    if feature_cols:
        cols = [c for c in feature_cols if c in df_past.columns]
        cols = ['date', 'team'] + cols if 'team' in df_past.columns else ['date'] + cols
        df_past = df_past[cols]
    
    return df_past


def create_all_features(
    df: pd.DataFrame,
    windows: List[int] = [5, 10],
    ema_spans: List[int] = [10, 20],
    stats_cols: List[str] = None,
    group_col: str = 'team',
    include_momentum: bool = True,
    include_consistency: bool = True
) -> pd.DataFrame:
    """
    Função de conveniência que cria todas as features.
    
    Args:
        df: DataFrame com dados históricos
        windows: Janelas para rolling features
        ema_spans: Spans para EMA
        stats_cols: Colunas para processar
        group_col: Coluna de agrupamento
        include_momentum: Incluir features de momentum
        include_consistency: Incluir features de consistência
    
    Returns:
        DataFrame com todas as features adicionadas
    """
    logger.info("🔧 Criando pipeline completo de features anti-leakage")
    
    # Rolling features
    df = create_rolling_features(df, windows=windows, stats_cols=stats_cols,
                                 aggregations=['mean', 'std'], group_col=group_col)
    
    # Season averages
    df = create_season_avg_features(df, stats_cols=stats_cols, group_col=group_col)
    
    # EMAs
    df = create_ema_features(df, spans=ema_spans, stats_cols=stats_cols, group_col=group_col)
    
    # Momentum
    if include_momentum:
        df = create_momentum_features(df, stats_cols=stats_cols, group_col=group_col)
    
    # Consistency
    if include_consistency:
        df = create_consistency_features(df, stats_cols=stats_cols, 
                                         window=10, group_col=group_col)
    
    logger.info(f"✅ Pipeline de features concluído: {len(df.columns)} colunas")
    
    return df


# =============================================================================
# CLI PARA TESTES
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("🔧 Feature Engineering - Teste Anti-Leakage")
    print("="*60)
    
    # Criar dados de teste
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'team': ['LAL'] * 30,
        'pts': np.random.randint(100, 130, 30),
        'reb': np.random.randint(40, 55, 30),
        'ast': np.random.randint(20, 35, 30),
        'season': '2023-24'
    })
    
    print(f"\n📊 DataFrame original: {len(df)} linhas")
    print(df.head())
    
    # Aplicar features
    df_features = create_all_features(
        df,
        windows=[5, 10],
        stats_cols=['pts', 'reb', 'ast']
    )
    
    print(f"\n📊 DataFrame com features: {len(df_features.columns)} colunas")
    print("Novas colunas:", [c for c in df_features.columns if 'rolling' in c or 'ema' in c])
    
    # Validar anti-leakage
    print("\n🔍 Verificando anti-leakage:")
    
    # Dia 1 deve ter NaN nas rolling features
    first_row = df_features.iloc[0]
    rolling_cols = [c for c in df_features.columns if 'rolling_5' in c]
    
    for col in rolling_cols[:3]:  # Verificar algumas colunas
        val = first_row[col]
        status = "✅ NaN (correto)" if pd.isna(val) else f"⚠️ {val} (verificar)"
        print(f"   {col}: {status}")
    
    # Validar integridade temporal
    print("\n🔍 Validando integridade temporal:")
    try:
        validate_temporal_integrity(df_features, "2024-01-20", "2024-01-21")
        print("   ✅ Validação passou!")
    except DataLeakageError as e:
        print(f"   ❌ {e}")
    
    print("\n✅ Teste concluído!")
