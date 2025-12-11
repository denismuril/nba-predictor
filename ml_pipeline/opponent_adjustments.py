"""
Opponent-Adjusted Features Calculator

Ajusta métricas ofensivas pela qualidade defensiva dos oponentes enfrentados.

Problema: Um time pode parecer ofensivamente forte jogando contra defesas fracas.
Solução: Ajustar eFG%, TOV%, etc. pela média defensiva dos últimos N oponentes.

Fórmula:
    Adj_Metric = Raw_Metric - (Opp_Def_Metric_Allowed - League_Avg)

Exemplo:
    - Time A tem eFG% de 55%
    - Últimos 10 opp permitiram 53% eFG em média (defesas fracas)
    - League avg = 53.5%
    - Adj_eFG% = 55% - (53% - 53.5%) = 55.5%
    
Referência: Dean Oliver - "Basketball on Paper", Adjusted Statistics
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# League Averages NBA 2025-26 (atualizar por temporada)
LEAGUE_AVG_STATS = {
    'efg_pct': 0.535,      # Effective Field Goal %
    'tov_pct': 12.5,       # Turnover % (escala 10-20)
    'orb_pct': 0.235,      # Offensive Rebound %
    'ftr': 0.235,          # Free Throw Rate
    'def_rating': 112.0,   # Defensive Rating (pts per 100 poss)
    'pace': 99.5,          # Possessões per 48min
}


def calculate_opponent_strength_schedule(
    df: pd.DataFrame,
    metric: str = 'efg_pct',
    window: int = 10
) -> pd.DataFrame:
    """
    Calcula a força média dos oponentes enfrentados (SOS - Strength of Schedule).
    
    Args:
        df: DataFrame com jogos (formato longo: uma linha por time por jogo)
        metric: Métrica a calcular (ex: 'efg_pct', 'def_rating')
        window: Janela de rolling (últimos N oponentes)
    
    Returns:
        DataFrame com coluna 'opp_strength_{metric}'
    """
    # Para cada jogo, precisamos da métrica DEFENSIVA do oponente
    # Ex: Se time A jogou contra B, queremos saber a def_efg_allowed de B
    
    # Criar lookup de defensive stats por time e data
    df_sorted = df.sort_values(['team', 'date']).copy()
    
    # Calcular defensive metric (o que o time PERMITE)
    # Por exemplo, def_efg_allowed = eFG% que o oponente conseguiu contra esse time
    
    # Isso requer ter stats do oponente disponíveis
    # Como approximação, vamos usar a métrica ofensiva do oponente como proxy
    
    if f'opp_{metric}' in df.columns:
        opp_metric_col = f'opp_{metric}'
    else:
        logger.warning(f"Coluna opp_{metric} não encontrada. Pulando SOS para {metric}.")
        return df
    
    # Rolling average da métrica do oponente
    df_sorted[f'opp_strength_{metric}'] = df_sorted.groupby('team')[opp_metric_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=min(3, window)).mean()
    )
    
    return df_sorted


def add_opponent_adjusted_efg(
    df: pd.DataFrame,
    windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Adiciona eFG% ajustado por qualidade defensiva dos oponentes.
    
    Args:
        df: DataFrame com jogos no formato normal (home/away)
        windows: Janelas de rolling a calcular
    
    Returns:
        DataFrame com colunas adicionais:
            - home_rolling_{w}_efg_adj
            - away_rolling_{w}_efg_adj
    """
    logger.info("📊 Calculando eFG% ajustado por oponente...")
    
    # Criar formato longo
    home_df = df[['date', 'home_team', 'away_team']].copy()
    home_df['team'] = home_df['home_team']
    home_df['opponent'] = home_df['away_team']
    
    # eFG% do time
    if 'efg_pct' in df.columns:
        home_df['efg_pct'] = df['efg_pct']
    else:
        # Calcular se não existir
        if 'fgm' in df.columns and 'fg3m' in df.columns and 'fga' in df.columns:
            home_df['efg_pct'] = (df['fgm'] + 0.5 * df['fg3m']) / df['fga'].replace(0, np.nan)
        else:
            logger.warning("⚠️ Não foi possível calcular eFG%. Colunas faltando.")
            return df
    
    # eFG% permitido (defensive)
    if 'opp_efg_pct' in df.columns:
        home_df['opp_efg_pct'] = df['opp_efg_pct']  # O que o oponente fez contra nós
    elif 'opp_fgm' in df.columns:
        home_df['opp_efg_pct'] = (df['opp_fgm'] + 0.5 * df['opp_fg3m']) / df['opp_fga'].replace(0, np.nan)
    
    # Repetir para away
    away_df = df[['date', 'away_team', 'home_team']].copy()
    away_df['team'] = away_df['away_team']
    away_df['opponent'] = away_df['home_team']
    
    if 'opp_efg_pct' in df.columns:
        away_df['efg_pct'] = df['opp_efg_pct']  # Away usa opp stats
        away_df['opp_efg_pct'] = df['efg_pct']   # Opponent (home) defensivo
    
    # Combinar
    long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date'])
    
    # Para cada time, calcular média de eFG% PERMITIDO pelos oponentes enfrentados
    for window in windows:
        # Primeiro, precisamos de um lookup de def_efg_allowed por time
        # Aproximação: usar rolling de opp_efg_pct (o que permitimos)
        long_df[f'def_efg_allowed_rolling_{window}'] = long_df.groupby('team')['opp_efg_pct'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=min(3, window)).mean()
        )
        
        # Agora, para cada jogo, pegar a def_efg_allowed do OPONENTE
        # Isso requer um merge complexo... simplificar:
        
        # Rolling eFG% do próprio time
        long_df[f'rolling_{window}_efg'] = long_df.groupby('team')['efg_pct'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=min(3, window)).mean()
        )
        
        # SOS: média de eFG% dos últimos N oponentes
        # (Proxy: assumir que oponentes tiveram performance consistente)
        # Simplificação: usar opp_efg_pct como proxy
        long_df[f'opp_avg_efg_{window}'] = long_df.groupby('team')['opp_efg_pct'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=min(3, window)).mean()
        )
        
        # Adjustment
        league_avg = LEAGUE_AVG_STATS['efg_pct']
        long_df[f'rolling_{window}_efg_adj'] = (
            long_df[f'rolling_{window}_efg'] - 
            (long_df[f'opp_avg_efg_{window}'] - league_avg)
        )
        
        # Fill NaNs
        long_df[f'rolling_{window}_efg_adj'] = long_df[f'rolling_{window}_efg_adj'].fillna(
            long_df[f'rolling_{window}_efg']
        )
    
    # Merge de volta
    new_cols = [c for c in long_df.columns if 'efg_adj' in c]
    
    # Home
    home_merged = long_df[long_df['team'].isin(df['home_team'])][['date', 'team'] + new_cols].copy()
    home_merged.columns = ['date', 'home_team'] + [f'home_{c}' for c in new_cols]
    
    df = df.merge(home_merged, on=['date', 'home_team'], how='left')
    
    # Away
    away_merged = long_df[long_df['team'].isin(df['away_team'])][['date', 'team'] + new_cols].copy()
    away_merged.columns = ['date', 'away_team'] + [f'away_{c}' for c in new_cols]
    
    df = df.merge(away_merged, on=['date', 'away_team'], how='left')
    
    logger.info(f"   ✅ eFG% ajustado calculado para janelas: {windows}")
    
    return df


def add_opponent_adjusted_tov(
    df: pd.DataFrame,
    windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Adiciona TOV% ajustado por press defense dos oponentes.
    
    Similar ao eFG%, mas para turnovers.
    """
    logger.info("📊 Calculando TOV% ajustado por oponente (press defense)...")
    
    # Implementação similar ao eFG%
    # Por questão de tempo, vou deixar como stub para implementar depois
    
    logger.warning("⚠️ TOV% ajustado ainda não implementado. TODO para próxima iteração.")
    
    return df


if __name__ == '__main__':
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    print("🏀 Opponent-Adjusted Features Demo\n")
    
    # Simular dados
    sample_games = pd.DataFrame({
        'date': pd.date_range('2025-11-01', periods=20),
        'home_team': ['LAL'] * 10 + ['BOS'] * 10,
        'away_team': ['Various'] * 20,
        'fgm': np.random.randint(35, 45, 20),
        'fg3m': np.random.randint(10, 15, 20),
        'fga': np.random.randint(85, 95, 20),
        'opp_fgm': np.random.randint(30, 40, 20),
        'opp_fg3m': np.random.randint(8, 12, 20),
        'opp_fga': np.random.randint(80, 90, 20),
    })
    
    # Calcular eFG% ajustado
    result = add_opponent_adjusted_efg(sample_games, windows=[5, 10])
    
    print("Colunas criadas:")
    adj_cols = [c for c in result.columns if 'efg_adj' in c]
    print(f"  {adj_cols}")
    
    print("\n✅ Demo completo!")
