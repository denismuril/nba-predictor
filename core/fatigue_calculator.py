"""
Fatigue Calculator - Índice Composto de Fadiga (v19.0 GeoSpatial)

Calcula fadiga de times baseado em múltiplos fatores:
1. Rest days (dias de descanso)
2. Travel distance (milhas viajadas últimos 7 dias) - CÁLCULO GEODÉSICO REAL
3. Games density (jogos nos últimos 5 dias)
4. Back-to-back-to-back scenarios

Índice Final (0-1):
    Fatigue = 0.4*rest + 0.3*travel + 0.2*density + 0.1*b2b2b
    
    0.0 = Sem fadiga (descansado, sem viagem)
    1.0 = Fadiga máxima (0 dias rest, 5000mi viajadas, 5 jogos em 5 dias)
    
Referência: Sports Science research on NBA fatigue and performance
v19.0: Implementação de cálculo geodésico real usando Haversine formula
"""
import math
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Coordenadas EXATAS das arenas NBA (Lat, Lon) - v19.0 GeoSpatial
# Fonte: Verificado manualmente contra dados oficiais
NBA_ARENA_COORDS: Dict[str, Tuple[float, float]] = {
    'ATL': (33.7573, -84.3963), 'BOS': (42.3662, -71.0621), 'BKN': (40.6826, -73.9754),
    'CHA': (35.2251, -80.8392), 'CHI': (41.8807, -87.6742), 'CLE': (41.4966, -81.6885),
    'DAL': (32.7905, -96.8103), 'DEN': (39.7487, -105.0076), 'DET': (42.3411, -83.0553),
    'GSW': (37.7680, -122.3877), 'HOU': (29.7508, -95.3621), 'IND': (39.7640, -86.1555),
    'LAC': (33.9455, -118.3417), 'LAL': (34.0430, -118.2673), 'MEM': (35.1382, -90.0506),
    'MIA': (25.7814, -80.1870), 'MIL': (43.0451, -87.9174), 'MIN': (44.9795, -93.2761),
    'NOP': (29.9490, -90.0821), 'NYK': (40.7505, -73.9934), 'OKC': (35.4634, -97.5151),
    'ORL': (28.5392, -81.3839), 'PHI': (39.9012, -75.1720), 'PHX': (33.4457, -112.0712),
    'POR': (45.5316, -122.6668), 'SAC': (38.5802, -121.4997), 'SAS': (29.4270, -98.4375),
    'TOR': (43.6435, -79.3791), 'UTA': (40.7683, -111.9011), 'WAS': (38.8982, -77.0209),
    # Aliases para compatibilidade
    'BRK': (40.6826, -73.9754), 'CHO': (35.2251, -80.8392), 'PHO': (33.4457, -112.0712),
}

# Tentar importar do config (fallback para constantes locais)
try:
    from config.arena_constants import NBA_ARENA_LOCATIONS
    # Merge com coordenadas locais (prioridade para config)
    NBA_ARENA_COORDS.update(NBA_ARENA_LOCATIONS)
except ImportError:
    logger.debug("config.arena_constants não encontrado, usando coordenadas built-in.")


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calcula distância geodésica entre dois pontos usando Fórmula de Haversine.
    
    Math:
        a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        c = 2 * atan2(√a, √(1-a))
        d = R * c
    
    Args:
        coord1: (latitude, longitude) do ponto 1
        coord2: (latitude, longitude) do ponto 2
    
    Returns:
        Distância em milhas
    
    Example:
        >>> haversine_distance((40.7505, -73.9934), (34.0430, -118.2673))
        2451.0  # NYC → LA ≈ 2451 milhas
    """
    R = 3959.0  # Raio da Terra em milhas
    
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Converter para radianos
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (
        math.sin(delta_lat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) *
        math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def calculate_rest_fatigue(rest_days: pd.Series) -> pd.Series:
    """
    Converte rest days em índice de fadiga (0-1).
    
    Args:
        rest_days: Dias desde último jogo
    
    Returns:
        Fatigue index: 0 = descansado (3+ dias), 1 = sem descanso (0 dias)
    """
    # Clipar rest_days
    rest_clipped = rest_days.clip(0, 3)
    
    # Inverter: 3 dias = 0 fadiga, 0 dias = 1.0 fadiga
    fatigue = (3 - rest_clipped) / 3.0
    
    return fatigue


def calculate_travel_fatigue(
    df: pd.DataFrame,
    team_col: str = 'team',
    date_col: str = 'date',
    is_home_col: str = 'is_home',
    opponent_col: str = 'opponent'
) -> pd.Series:
    """
    Calcula fadiga por viagem (últimos 7 dias) usando distância geodésica real.

    v19.0 GeoSpatial: Implementação com Haversine formula para distâncias reais.

    Lógica:
        1. Para cada jogo, identificar arena atual (home do time ou do oponente)
        2. Calcular distância entre arena do jogo anterior e arena atual
        3. Somar distâncias em rolling window de 7 dias
        4. Normalizar para índice 0-1 (max ~5000mi/semana)

    Args:
        df: DataFrame com jogos (long format)
        team_col: Nome da coluna de time
        date_col: Nome da coluna de data
        is_home_col: Coluna indicando se o time está em casa (bool/int)
        opponent_col: Coluna com o oponente (para determinar arena away)

    Returns:
        Travel fatigue index (0-1)
    """
    df = df.sort_values([team_col, date_col]).copy()

    # Garantir que date é datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    travel_miles = []

    for idx, row in df.iterrows():
        team = row[team_col]

        # Determinar arena do jogo ATUAL
        # Se is_home = True → arena do próprio time
        # Se is_home = False → arena do oponente
        is_home = row.get(is_home_col, True)
        opponent = row.get(opponent_col, None)

        if is_home:
            current_arena_team = team
        else:
            current_arena_team = opponent if opponent else team

        current_arena = NBA_ARENA_COORDS.get(current_arena_team)

        if current_arena is None:
            travel_miles.append(0.0)
            continue

        # Pegar jogo ANTERIOR do mesmo time
        mask = (df[team_col] == team) & (df[date_col] < row[date_col])
        prev_games = df.loc[mask]

        if prev_games.empty:
            # Primeiro jogo da temporada - sem viagem
            travel_miles.append(0.0)
            continue

        # Último jogo
        last_game = prev_games.iloc[-1]

        # Determinar arena do jogo ANTERIOR
        last_is_home = last_game.get(is_home_col, True)
        last_opponent = last_game.get(opponent_col, None)

        if last_is_home:
            last_arena_team = team
        else:
            last_arena_team = last_opponent if last_opponent else team

        last_arena = NBA_ARENA_COORDS.get(last_arena_team)

        if last_arena is None:
            travel_miles.append(0.0)
            continue

        # Calcular distância geodésica (Haversine)
        miles = haversine_distance(last_arena, current_arena)
        travel_miles.append(miles)

    df['travel_miles'] = travel_miles

    # Rolling 7 dias por time
    df['travel_miles_7d'] = df.groupby(team_col)['travel_miles'].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    )

    # Normalizar (0-1): max ~5000 milhas/semana
    # Referência: Coast-to-coast trip (NYC→LA) ≈ 2500mi × 2 = 5000mi
    fatigue = (df['travel_miles_7d'] / 5000.0).clip(0, 1)

    logger.debug(f"   Travel miles calculadas: mean={df['travel_miles'].mean():.0f}mi, "
                 f"max_7d={df['travel_miles_7d'].max():.0f}mi")

    return fatigue


def calculate_games_density_fatigue(
    df: pd.DataFrame,
    team_col: str = 'team',
    date_col: str = 'date',
    window_days: int = 5
) -> pd.Series:
    """
    Calcula fadiga por densidade de jogos (games in last N days).
    
    Args:
        df: DataFrame com jogos
        team_col: Coluna de time
        date_col: Coluna de data
        window_days: Janela em dias (default: 5)
    
    Returns:
        Density fatigue (0-1)
    """
    df = df.sort_values([team_col, date_col]).copy()
    
    # Contar jogos nos últimos N dias
    df[f'games_last_{window_days}d'] = df.groupby(team_col)[date_col].transform(
        lambda x: x.rolling(f'{window_days}D').count()
    )
    
    # Normalizar: 1 jogo = 0 fadiga, 5 jogos em 5 dias = 1.0 fadiga
    # (Tecnicamente impossível ter >5 jogos em 5 dias, mas clipar por segurança)
    fatigue = ((df[f'games_last_{window_days}d'] - 1) / 4).clip(0, 1)
    
    return fatigue


def calculate_b2b2b_penalty(
    df: pd.DataFrame,
    team_col: str = 'team',
    date_col: str = 'date'
) -> pd.Series:
    """
    Detecta back-to-back-to-back scenarios (3 jogos consecutivos).
    
    Returns:
        Binary: 1.0 se b2b2b, 0.0 caso contrário
    """
    df = df.sort_values([team_col, date_col]).copy()
    
    # Calcular dias desde último jogo
    df['days_since_last'] = df.groupby(team_col)[date_col].diff().dt.days
    
    # Back-to-back = 1 dia
    df['is_b2b'] = (df['days_since_last'] <= 1).astype(int)
    
    # Back-to-back-to-back = 2 jogos b2b consecutivos
    df['is_b2b2b'] = (
        df['is_b2b'] & 
        df.groupby(team_col)['is_b2b'].shift(1).fillna(0).astype(bool)
    ).astype(float)
    
    return df['is_b2b2b']


def calculate_comprehensive_fatigue_index(
    df: pd.DataFrame,
    team_col: str = 'team',
    date_col: str = 'date',
    rest_col: str = 'rest_days',
    is_home_col: str = 'is_home',
    opponent_col: str = 'opponent'
) -> pd.DataFrame:
    """
    Calcula índice composto de fadiga (0-1).

    v19.0 GeoSpatial: Agora usa cálculo geodésico real para travel fatigue.

    Formula:
        Fatigue = 0.4*rest + 0.3*travel + 0.2*density + 0.1*b2b2b

    Args:
        df: DataFrame com jogos (long format: uma linha por time por jogo)
        team_col: Nome da coluna de time
        date_col: Nome da coluna de data
        rest_col: Coluna com rest_days (se existir)
        is_home_col: Coluna indicando se o time joga em casa
        opponent_col: Coluna com o oponente

    Returns:
        DataFrame com coluna 'fatigue_index'
    """
    logger.info("💪 Calculando Fatigue Index composto (v19.0 GeoSpatial)...")

    df = df.copy()

    # 1. Rest fatigue
    if rest_col in df.columns:
        fatigue_rest = calculate_rest_fatigue(df[rest_col])
    else:
        # Calcular rest_days
        df = df.sort_values([team_col, date_col])
        df['rest_days'] = df.groupby(team_col)[date_col].diff().dt.days
        df['rest_days'] = df['rest_days'].fillna(3).clip(0, 7)
        fatigue_rest = calculate_rest_fatigue(df['rest_days'])

    # 2. Travel fatigue (v19.0 - CÁLCULO GEODÉSICO REAL)
    has_travel_cols = is_home_col in df.columns and opponent_col in df.columns
    if has_travel_cols:
        fatigue_travel = calculate_travel_fatigue(
            df, team_col, date_col, is_home_col, opponent_col
        )
        logger.info("   ✅ Travel fatigue calculada com geolocalização real")
    else:
        # Fallback: estimar baseado em se é home/away (se pelo menos is_home existir)
        if is_home_col in df.columns:
            # Away games = assume viagem média (~1000mi)
            fatigue_travel = (~df[is_home_col].astype(bool)).astype(float) * 0.2
            logger.warning("   ⚠️ Travel fatigue estimada (sem coluna opponent)")
        else:
            fatigue_travel = pd.Series(0.0, index=df.index)
            logger.warning("   ⚠️ Travel fatigue zerada (sem colunas is_home/opponent)")

    # 3. Games density fatigue
    fatigue_density = calculate_games_density_fatigue(df, team_col, date_col)

    # 4. B2B2B penalty
    fatigue_b2b2b = calculate_b2b2b_penalty(df, team_col, date_col)

    # 5. Composite index (PESOS ORIGINAIS RESTAURADOS)
    # v19.0: Travel fatigue agora é calculada corretamente, pesos restaurados
    df['fatigue_index'] = (
        0.40 * fatigue_rest +       # Rest days (maior fator)
        0.30 * fatigue_travel +     # Travel distance (reativado!)
        0.20 * fatigue_density +    # Game density
        0.10 * fatigue_b2b2b        # B2B2B penalty
    )

    # Clip to [0, 1]
    df['fatigue_index'] = df['fatigue_index'].clip(0, 1)

    logger.info(f"   ✅ Fatigue Index: média={df['fatigue_index'].mean():.3f}, "
                f"max={df['fatigue_index'].max():.3f}")

    return df


if __name__ == '__main__':
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    print("💪 Fatigue Calculator Demo\n")
    
    # Simular temporada
    dates = pd.date_range('2025-11-01', periods=30, freq='2D')  # Jogo a cada 2 dias
    
    sample_games = pd.DataFrame({
        'date': dates,
        'team': ['LAL'] * 30,
        'rest_days': [2, 1, 1, 0, 2, 3, 1, 1, 0, 1] * 3,  # Variado
    })
    
    # Calcular fatigue
    result = calculate_comprehensive_fatigue_index(
        sample_games,
        team_col='team',
        date_col='date',
        rest_col='rest_days'
    )
    
    print("Primeiros 10 jogos:")
    print(result[['date', 'rest_days', 'fatigue_index']].head(10))
    
    print(f"\nFatigue médio: {result['fatigue_index'].mean():.3f}")
    print(f"Max fatigue: {result['fatigue_index'].max():.3f}")
    print(f"Jogos com alta fadiga (>0.5): {(result['fatigue_index'] > 0.5).sum()}")
    
    print("\n✅ Demo completo!")
