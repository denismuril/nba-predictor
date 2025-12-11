"""
Opponent-Adjusted Stats - Correção de Strength of Schedule
===========================================================

PROBLEMA CRÍTICO que este módulo resolve:
- Médias móveis brutas são enganosas quando um time joga contra
  oponentes fracos/fortes
- Exemplo: Lakers 115 pts/jogo vs defesas fracas != 115 vs defesas elite

SOLUÇÃO (Padrão Vegas):
- Ajustar stats ofensivos pela força defensiva dos oponentes enfrentados
- Ajustar stats defensivos pela força ofensiva dos oponentes enfrentados

Fórmula:
    ORtg_Ajustado = ORtg_Bruto - Liga_Avg_DRtg + Opp_Avg_DRtg

Se time marcou 115 vs defesas de 120 DRtg (fracos), mas liga tem 112 DRtg:
    ORtg_Ajustado = 115 - 112 + 120 = 123 (inflado, jogou contra fracos)

Se time marcou 110 vs defesas de 105 DRtg (elite):
    ORtg_Ajustado = 110 - 112 + 105 = 103 (ajustado para baixo)

Autor: NBA Predictor Team
Data: 2025-12-03
Padrão: Vegas/Pinnacle
"""

import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)


def calcular_stats_ajustados_oponente(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona métricas ajustadas por força do oponente.

    CRÍTICO: Chama esta função APÓS calcular stats básicos (Four Factors)
    mas ANTES de criar rolling averages.

    Args:
        df: DataFrame com colunas básicas:
            - home_off_rating, away_off_rating (Offensive Rating bruto)
            - home_def_rating, away_def_rating (Defensive Rating bruto)
            - date
            - (opcional) season

    Returns:
        DataFrame com colunas adicionadas:
            - home_ortg_adj, away_ortg_adj (Offensive Rating ajustado)
            - home_drtg_adj, away_drtg_adj (Defensive Rating ajustado)

    Exemplo:
        >>> df['home_off_rating'] = [115, 110, 120]
        >>> df['away_def_rating'] = [120, 105, 110]  # Fraco, Elite, Médio
        >>> df = calcular_stats_ajustados_oponente(df)
        >>> # home_ortg_adj será ajustado pela força da defesa enfrentada
    """
    logger.info("🎯 Calculando métricas ajustadas por oponente...")

    # Verificar colunas de ratings
    rating_cols = [
        'home_off_rating', 'away_off_rating',
        'home_def_rating', 'away_def_rating'
    ]
    missing_ratings = [col for col in rating_cols if col not in df.columns]

    if missing_ratings:
        if 'home_score' in df.columns and 'away_score' in df.columns:
            logger.info("   📊 Calculando ratings a partir dos scores...")
            df['home_off_rating'] = df['home_score'].fillna(0).astype(float)
            df['away_off_rating'] = df['away_score'].fillna(0).astype(float)
            df['home_def_rating'] = df['away_score'].fillna(0).astype(float)
            df['away_def_rating'] = df['home_score'].fillna(0).astype(float)

            zero_mask = (
                (df['home_off_rating'] == 0) | (df['away_off_rating'] == 0)
            )
            if zero_mask.any():
                df.loc[zero_mask, 'home_off_rating'] = 112.0
                df.loc[zero_mask, 'away_off_rating'] = 112.0
                df.loc[zero_mask, 'home_def_rating'] = 112.0
                df.loc[zero_mask, 'away_def_rating'] = 112.0
        else:
            logger.warning("⚠️ Colunas de score faltando. Usando fallback 112.0")
            df['home_off_rating'] = 112.0
            df['away_off_rating'] = 112.0
            df['home_def_rating'] = 112.0
            df['away_def_rating'] = 112.0

    # Verificar se date existe
    if 'date' not in df.columns:
        logger.warning("⚠️ Coluna 'date' faltando. Pulando ajuste de oponente.")
        df['home_ortg_adj'] = df['home_off_rating']
        df['away_ortg_adj'] = df['away_off_rating']
        df['home_drtg_adj'] = df['home_def_rating']
        df['away_drtg_adj'] = df['away_def_rating']
        return df

    # Inferir temporada se não existir
    if 'season' not in df.columns:
        df['season'] = df['date'].apply(
            lambda x: f"{x.year}-{x.year+1}" if x.month >= 10
            else f"{x.year-1}-{x.year}"
        )
        logger.info("   Temporada inferida a partir da data")

    # V21 FIX: Usar Expanding Window (shift(1)) para evitar vazamento temporal
    # ANTES: groupby('season').mean() vazava dados do futuro
    # AGORA: Para cada jogo, usa APENAS dados de jogos anteriores
    logger.info("   V21 FIX: Calculando médias com Expanding Window...")

    df = df.sort_values('date').reset_index(drop=True)

    # Calcular média geral de ORtg e DRtg (home + away combinados)
    df['_game_ortg'] = (df['home_off_rating'] + df['away_off_rating']) / 2
    df['_game_drtg'] = (df['home_def_rating'] + df['away_def_rating']) / 2

    # V21 FIX: Expanding mean com shift(1)
    df['liga_ortg_avg'] = (
        df['_game_ortg'].expanding(min_periods=10).mean().shift(1)
    )
    df['liga_drtg_avg'] = (
        df['_game_drtg'].expanding(min_periods=10).mean().shift(1)
    )

    # V21 FIX: Fallback conservador para primeiros jogos
    df['liga_ortg_avg'] = df['liga_ortg_avg'].fillna(112.0)
    df['liga_drtg_avg'] = df['liga_drtg_avg'].fillna(112.0)

    # Limpar colunas temporárias
    df = df.drop(columns=['_game_ortg', '_game_drtg'], errors='ignore')

    # Log estatísticas
    ortg_min, ortg_max = df['liga_ortg_avg'].min(), df['liga_ortg_avg'].max()
    drtg_min, drtg_max = df['liga_drtg_avg'].min(), df['liga_drtg_avg'].max()
    logger.info(f"   V21 FIX: Liga ORtg avg: {ortg_min:.1f} - {ortg_max:.1f}")
    logger.info(f"   V21 FIX: Liga DRtg avg: {drtg_min:.1f} - {drtg_max:.1f}")

    # AJUSTAR OFFENSIVE RATING
    df['home_ortg_adj'] = (
        df['home_off_rating'] -
        df['liga_drtg_avg'] +
        df['away_def_rating']
    )

    df['away_ortg_adj'] = (
        df['away_off_rating'] -
        df['liga_drtg_avg'] +
        df['home_def_rating']
    )

    # AJUSTAR DEFENSIVE RATING
    df['home_drtg_adj'] = (
        df['home_def_rating'] -
        df['liga_ortg_avg'] +
        df['away_off_rating']
    )

    df['away_drtg_adj'] = (
        df['away_def_rating'] -
        df['liga_ortg_avg'] +
        df['home_off_rating']
    )

    # SANITY CHECKS - Clipar valores extremos
    adj_cols = ['home_ortg_adj', 'away_ortg_adj', 'home_drtg_adj', 'away_drtg_adj']
    for col in adj_cols:
        df[col] = df[col].clip(lower=80, upper=140)
        mean_val = df[col].mean()
        std_val = df[col].std()
        logger.info(f"   {col}: μ={mean_val:.1f}, σ={std_val:.1f}")

    logger.info("✅ Stats ajustados por oponente calculados")
    logger.info("   Colunas adicionadas: home/away_ortg_adj, home/away_drtg_adj")

    return df


def add_opponent_adjusted_rolling(
    df: pd.DataFrame,
    windows: List[int] = [10, 20]
) -> pd.DataFrame:
    """
    Adiciona rolling averages das stats AJUSTADAS por oponente.

    IMPORTANTE: Chama esta função DEPOIS de calcular_stats_ajustados_oponente()

    Args:
        df: DataFrame com stats ajustados (home_ortg_adj, etc.)
        windows: Janelas de rolling (recomendado: [10, 20])

    Returns:
        DataFrame com rolling features ajustadas
    """
    logger.info(f"📊 Calculando rolling stats ajustados (windows={windows})...")

    required = ['home_ortg_adj', 'away_ortg_adj', 'home_drtg_adj', 'away_drtg_adj']
    missing = [col for col in required if col not in df.columns]

    if missing:
        logger.error(f"❌ Stats ajustados faltando: {missing}")
        logger.error("Execute calcular_stats_ajustados_oponente() primeiro!")
        return df

    # Criar DataFrame longo para rolling por team
    home_df = df[['date', 'home_team', 'home_ortg_adj', 'home_drtg_adj']].copy()
    home_df.columns = ['date', 'team', 'ortg_adj', 'drtg_adj']

    away_df = df[['date', 'away_team', 'away_ortg_adj', 'away_drtg_adj']].copy()
    away_df.columns = ['date', 'team', 'ortg_adj', 'drtg_adj']

    long_df = pd.concat([home_df, away_df]).sort_values(
        ['team', 'date']
    ).reset_index(drop=True)

    # Calcular rolling stats (SHIFTED para evitar data leakage!)
    for window in windows:
        for metric in ['ortg_adj', 'drtg_adj']:
            col_name = f'rolling_{window}_{metric}'
            long_df[col_name] = long_df.groupby('team')[metric].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=3).mean()
            )

    # Preencher NaN com valores brutos (fallback)
    for window in windows:
        long_df[f'rolling_{window}_ortg_adj'] = (
            long_df[f'rolling_{window}_ortg_adj'].fillna(long_df['ortg_adj'])
        )
        long_df[f'rolling_{window}_drtg_adj'] = (
            long_df[f'rolling_{window}_drtg_adj'].fillna(long_df['drtg_adj'])
        )

    # Separar de volta em home e away
    long_home = long_df.merge(
        df[['date', 'home_team']].drop_duplicates(),
        left_on=['date', 'team'],
        right_on=['date', 'home_team'],
        how='inner'
    )

    long_away = long_df.merge(
        df[['date', 'away_team']].drop_duplicates(),
        left_on=['date', 'team'],
        right_on=['date', 'away_team'],
        how='inner'
    )

    # Renomear colunas
    roll_cols = [c for c in long_df.columns if 'rolling_' in c]

    long_home = long_home[['date', 'team'] + roll_cols].rename(
        columns={c: f'home_{c}' for c in roll_cols}
    )

    long_away = long_away[['date', 'team'] + roll_cols].rename(
        columns={c: f'away_{c}' for c in roll_cols}
    )

    # Merge de volta
    df = df.merge(
        long_home.drop(columns=['team']),
        left_on=['date', 'home_team'],
        right_on=['date', 'home_team'],
        how='left',
        suffixes=('', '_dup')
    )

    df = df.merge(
        long_away.drop(columns=['team']),
        left_on=['date', 'away_team'],
        right_on=['date', 'away_team'],
        how='left',
        suffixes=('', '_dup')
    )

    # Remover colunas duplicadas
    df = df.loc[:, ~df.columns.str.endswith('_dup')]

    logger.info(f"✅ Rolling stats ajustados ({len(roll_cols)*2} features)")

    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("🎯 Opponent-Adjusted Stats - Demo\n")

    demo_df = pd.DataFrame({
        'date': pd.date_range('2024-10-22', periods=20),
        'home_team': ['LAL', 'BOS', 'DEN', 'GSW', 'MIA'] * 4,
        'away_team': ['DET', 'WAS', 'CHA', 'POR', 'SAC'] * 4,
        'home_off_rating': [115, 118, 120, 112, 114] * 4,
        'away_off_rating': [108, 110, 105, 115, 112] * 4,
        'home_def_rating': [110, 108, 105, 112, 109] * 4,
        'away_def_rating': [120, 115, 118, 110, 113] * 4,
        'season': ['2024-25'] * 20
    })

    print("📊 Antes do ajuste:")
    print(demo_df[['home_team', 'home_off_rating', 'away_def_rating']].head(3))

    demo_df = calcular_stats_ajustados_oponente(demo_df)

    print("\n📊 Depois do ajuste:")
    cols = ['home_team', 'home_off_rating', 'home_ortg_adj', 'away_def_rating']
    print(demo_df[cols].head(3))

    demo_df = add_opponent_adjusted_rolling(demo_df, windows=[5, 10])

    print("\n📊 Com rolling ajustado:")
    cols = ['home_team', 'home_rolling_5_ortg_adj', 'home_rolling_10_ortg_adj']
    print(demo_df[cols].tail(5))

    print("\n✅ Demo completo!")
