"""
Props Quantum Features - Engenharia de Features "Inumanas"

Módulo de feature engineering avançado para Player Props.
Implementa contexto multidimensional que humanos e modelos básicos ignoram.

Autor: Lead Quant Researcher & AI Architect
Versão: 1.0.0 - Quantum Edition
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

# Configuração
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# =============================================================================
# ARENA LOCATIONS & TIMEZONE DATA
# =============================================================================

# Altitudes em metros (importantes para fadiga)
ARENA_ALTITUDES = {
    'DEN': 1609,   # Denver - Mile High City
    'UTA': 1288,   # Salt Lake City
    'PHX': 331,    # Phoenix
    'DAL': 131,    # Dallas
    'LAL': 71,     # Los Angeles
    'LAC': 71,     # Los Angeles
    'GSW': 0,      # San Francisco (bay level)
    'SAC': 9,      # Sacramento
    'POR': 15,     # Portland
    'SEA': 0,      # Seattle
    'MIN': 253,    # Minneapolis
    'OKC': 366,    # Oklahoma City
    'SAS': 198,    # San Antonio
    'HOU': 12,     # Houston
    'MEM': 102,    # Memphis
    'NOP': 0,      # New Orleans
    'MIL': 188,    # Milwaukee
    'CHI': 176,    # Chicago
    'IND': 218,    # Indianapolis
    'DET': 183,    # Detroit
    'CLE': 199,    # Cleveland
    'TOR': 76,     # Toronto
    'BOS': 6,      # Boston
    'NYK': 10,     # New York
    'BKN': 10,     # Brooklyn
    'PHI': 12,     # Philadelphia
    'WAS': 0,      # Washington DC
    'CHA': 229,    # Charlotte
    'ATL': 320,    # Atlanta
    'MIA': 0,      # Miami
    'ORL': 25,     # Orlando
}

# Fusos horários (UTC offset)
ARENA_TIMEZONES = {
    # Pacific Time (UTC-8)
    'LAL': -8, 'LAC': -8, 'GSW': -8, 'SAC': -8, 'POR': -8,
    # Mountain Time (UTC-7)
    'DEN': -7, 'UTA': -7, 'PHX': -7,
    # Central Time (UTC-6)
    'DAL': -6, 'HOU': -6, 'SAS': -6, 'MEM': -6, 'NOP': -6, 
    'OKC': -6, 'MIN': -6, 'MIL': -6, 'CHI': -6,
    # Eastern Time (UTC-5)
    'IND': -5, 'DET': -5, 'CLE': -5, 'TOR': -5, 'BOS': -5,
    'NYK': -5, 'BKN': -5, 'PHI': -5, 'WAS': -5, 'CHA': -5,
    'ATL': -5, 'MIA': -5, 'ORL': -5,
}

# Coordenadas para cálculo de distância
ARENA_COORDS = None  # Carregado do JSON

def _load_arena_coords():
    """Carrega coordenadas das arenas do JSON."""
    global ARENA_COORDS
    if ARENA_COORDS is None:
        arena_file = DATA_DIR / "arena_locations.json"
        if arena_file.exists():
            with open(arena_file, 'r') as f:
                raw = json.load(f)
            # Converter para abreviações
            from utils.team_normalization import normalize_team
            ARENA_COORDS = {}
            for team_name, data in raw.items():
                abbr = normalize_team(team_name)
                ARENA_COORDS[abbr] = (data['lat'], data['lon'])
        else:
            logger.warning("arena_locations.json não encontrado")
            ARENA_COORDS = {}
    return ARENA_COORDS


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distância em km entre dois pontos."""
    R = 6371  # Raio da Terra em km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


# =============================================================================
# FASE 1.1: MICRO-MATCHUPS DEFENSIVOS (DvP 2.0)
# =============================================================================

def calculate_shooting_foul_frequency(df_boxscores: pd.DataFrame, opponent_team: str) -> Dict[str, float]:
    """
    Calcula frequência de faltas de arremesso por posição do oponente.
    
    Jogadores que sofrem mais faltas = mais free throws.
    Times que cometem mais faltas em alas = bom para alas adversários.
    
    Args:
        df_boxscores: DataFrame com boxscores históricos
        opponent_team: Time oponente (abreviação)
        
    Returns:
        Dict com frequência de faltas por posição
    """
    if df_boxscores is None or df_boxscores.empty:
        return {
            'foul_freq_guards': 0.5,
            'foul_freq_forwards': 0.5,
            'foul_freq_centers': 0.5,
            'total_pf_per_game': 20.0
        }
    
    try:
        # Filtrar jogos do oponente
        opp_games = df_boxscores[
            (df_boxscores['team'] == opponent_team) | 
            (df_boxscores['opponent'] == opponent_team)
        ]
        
        if opp_games.empty:
            return {
                'foul_freq_guards': 0.5,
                'foul_freq_forwards': 0.5,
                'foul_freq_centers': 0.5,
                'total_pf_per_game': 20.0
            }
        
        # Calcular faltas totais por jogo
        total_pf = opp_games['PF'].mean() if 'PF' in opp_games.columns else 20.0
        
        # Estimar distribuição por posição (aproximação)
        # Guards: ~35%, Forwards: ~40%, Centers: ~25%
        return {
            'foul_freq_guards': 0.35 * (total_pf / 20.0),
            'foul_freq_forwards': 0.40 * (total_pf / 20.0),
            'foul_freq_centers': 0.25 * (total_pf / 20.0),
            'total_pf_per_game': total_pf
        }
        
    except Exception as e:
        logger.warning(f"Erro ao calcular shooting_foul_freq: {e}")
        return {
            'foul_freq_guards': 0.5,
            'foul_freq_forwards': 0.5,
            'foul_freq_centers': 0.5,
            'total_pf_per_game': 20.0
        }


def calculate_allowed_rebound_rate(df_boxscores: pd.DataFrame, opponent_team: str) -> Dict[str, float]:
    """
    Calcula taxa de rebotes cedidos por zona (curto vs longo).
    
    Rebotes curtos: favorece centers
    Rebotes longos: favorece guards/forwards atléticos
    
    Args:
        df_boxscores: DataFrame com boxscores
        opponent_team: Time oponente
        
    Returns:
        Dict com taxas de rebote
    """
    defaults = {
        'allowed_oreb_rate': 0.25,  # Offensive rebounds cedidos
        'allowed_dreb_rate': 0.75,  # Defensive rebounds cedidos
        'long_reb_ratio': 0.30,     # Proporção de rebotes longos
        'short_reb_ratio': 0.70     # Proporção de rebotes curtos
    }
    
    if df_boxscores is None or df_boxscores.empty:
        return defaults
    
    try:
        opp_games = df_boxscores[df_boxscores['opponent'] == opponent_team]
        
        if opp_games.empty or 'OREB' not in opp_games.columns:
            return defaults
        
        total_reb = opp_games['OREB'].sum() + opp_games['DREB'].sum()
        if total_reb == 0:
            return defaults
            
        oreb_rate = opp_games['OREB'].sum() / total_reb
        dreb_rate = opp_games['DREB'].sum() / total_reb
        
        # Estimativa de rebotes longos vs curtos (baseado em métricas típicas)
        # Rebotes longos são ~30% dos rebotes defensivos
        long_reb = 0.30 * dreb_rate
        short_reb = oreb_rate + (0.70 * dreb_rate)
        
        return {
            'allowed_oreb_rate': oreb_rate,
            'allowed_dreb_rate': dreb_rate,
            'long_reb_ratio': long_reb,
            'short_reb_ratio': short_reb
        }
        
    except Exception as e:
        logger.warning(f"Erro ao calcular allowed_rebound_rate: {e}")
        return defaults


def calculate_dvp_advanced(
    df_boxscores: pd.DataFrame, 
    opponent_team: str,
    player_position: str = 'G'
) -> Dict[str, float]:
    """
    DvP 2.0 - Defense vs Position Avançado.
    
    Mapeia zonas de fraqueza defensiva por posição.
    
    Args:
        df_boxscores: DataFrame com boxscores
        opponent_team: Time oponente
        player_position: Posição do jogador (G, F, C)
        
    Returns:
        Dict com métricas DvP avançadas
    """
    foul_freq = calculate_shooting_foul_frequency(df_boxscores, opponent_team)
    reb_rates = calculate_allowed_rebound_rate(df_boxscores, opponent_team)
    
    # Mapear posição para frequência de faltas
    position_foul_map = {
        'G': foul_freq['foul_freq_guards'],
        'F': foul_freq['foul_freq_forwards'],
        'C': foul_freq['foul_freq_centers']
    }
    
    # Mapear posição para vantagem de rebote
    position_reb_advantage = {
        'G': reb_rates['long_reb_ratio'],      # Guards se beneficiam de rebotes longos
        'F': (reb_rates['long_reb_ratio'] + reb_rates['short_reb_ratio']) / 2,
        'C': reb_rates['short_reb_ratio']       # Centers dominam rebotes curtos
    }
    
    position = player_position[0].upper() if player_position else 'G'
    
    return {
        'dvp_foul_advantage': position_foul_map.get(position, 0.5),
        'dvp_reb_advantage': position_reb_advantage.get(position, 0.5),
        'dvp_total_pf': foul_freq['total_pf_per_game'],
        'dvp_oreb_allowed': reb_rates['allowed_oreb_rate'],
        **foul_freq,
        **reb_rates
    }


# =============================================================================
# FASE 1.2: MODELO DE FADIGA BIOLÓGICA & LOGÍSTICA
# =============================================================================

def calculate_circadian_disruption(
    player_team: str,
    opponent_team: str,
    is_away: bool
) -> float:
    """
    Calcula disrupção circadiana baseada em diferença de fuso horário.
    
    Time do Leste jogando no Oeste às 22h (hora local) = jogo às 01h (corpo do jogador).
    
    Returns:
        Score de disrupção (0-3 horas típico, pode ser maior)
    """
    coords = _load_arena_coords()
    
    home_tz = ARENA_TIMEZONES.get(opponent_team if is_away else player_team, -5)
    away_tz = ARENA_TIMEZONES.get(player_team if is_away else opponent_team, -5)
    
    if is_away:
        # Jogador viajando: diferença entre seu fuso e o do jogo
        disruption = abs(away_tz - home_tz)
    else:
        disruption = 0  # Jogando em casa
    
    return disruption


def calculate_altitude_impact(game_location: str, player_team: str) -> float:
    """
    Calcula impacto de altitude para times visitantes.
    
    Denver (1609m) e Utah (1288m) causam fadiga significativa.
    Jogadores precisam de ~2-3 jogos para aclimatar.
    
    Returns:
        Score de impacto (0.0-1.0, onde 1.0 = máximo impacto)
    """
    game_altitude = ARENA_ALTITUDES.get(game_location, 0)
    home_altitude = ARENA_ALTITUDES.get(player_team, 0)
    
    # Diferença de altitude
    altitude_diff = game_altitude - home_altitude
    
    # Impacto significativo apenas acima de 1000m de diferença
    if altitude_diff > 1000:
        return min(1.0, altitude_diff / 1600)  # Normalizado para Denver
    elif altitude_diff > 500:
        return min(0.5, altitude_diff / 1000)
    else:
        return 0.0


def calculate_thirst_index(
    minutes_last_game: float,
    travel_distance_km: float,
    is_b2b: bool
) -> float:
    """
    Índice de "sede" - combinação de minutos + viagem + B2B.
    
    Jogadores com >36min + viagem longa em B2B estão em desvantagem.
    
    Returns:
        Score de fadiga (0-100)
    """
    # Base: minutos jogados (normalizado para 36 como "normal")
    minutes_factor = max(0, (minutes_last_game - 30) / 10) * 25  # 0-25 pontos
    
    # Viagem: cada 500km = 5 pontos de fadiga
    travel_factor = min(25, (travel_distance_km / 500) * 5)
    
    # B2B: +30 pontos fixos
    b2b_factor = 30 if is_b2b else 0
    
    # Road trip: fadiga acumulada
    # (seria calculado com histórico de jogos recentes)
    
    return min(100, minutes_factor + travel_factor + b2b_factor)


def calculate_travel_distance(
    from_team: str,
    to_team: str
) -> float:
    """
    Calcula distância de viagem entre duas cidades.
    
    Returns:
        Distância em km
    """
    coords = _load_arena_coords()
    
    from_coords = coords.get(from_team)
    to_coords = coords.get(to_team)
    
    if from_coords and to_coords:
        return haversine_distance(
            from_coords[0], from_coords[1],
            to_coords[0], to_coords[1]
        )
    else:
        return 0.0


def calculate_fatigue_index(
    player_team: str,
    opponent_team: str,
    is_away: bool,
    minutes_last_game: float = 30.0,
    is_b2b: bool = False,
    games_in_last_5_days: int = 1
) -> Dict[str, float]:
    """
    Modelo completo de fadiga biológica & logística.
    
    Combina todos os fatores de fadiga em um índice unificado.
    
    Returns:
        Dict com todas as métricas de fadiga
    """
    game_location = opponent_team if is_away else player_team
    
    # Calcular componentes
    circadian = calculate_circadian_disruption(player_team, opponent_team, is_away)
    altitude = calculate_altitude_impact(game_location, player_team)
    travel_km = calculate_travel_distance(player_team, game_location) if is_away else 0
    thirst = calculate_thirst_index(minutes_last_game, travel_km, is_b2b)
    
    # Score composto de fadiga (0-100)
    fatigue_score = (
        (circadian * 10) +      # Disrupção circadiana: até 30 pontos
        (altitude * 20) +        # Altitude: até 20 pontos
        thirst                   # Thirst index: até 100 pontos
    )
    fatigue_score = min(100, fatigue_score)
    
    # Ajuste para sequência de jogos
    if games_in_last_5_days >= 4:
        fatigue_score = min(100, fatigue_score * 1.3)
    elif games_in_last_5_days >= 3:
        fatigue_score = min(100, fatigue_score * 1.15)
    
    return {
        'circadian_disruption': circadian,
        'altitude_impact': altitude,
        'travel_distance_km': travel_km,
        'thirst_index': thirst,
        'fatigue_score': fatigue_score,
        'is_high_fatigue': fatigue_score > 60,
        'games_in_5_days': games_in_last_5_days
    }


# =============================================================================
# FASE 1.3: GAME SCRIPT & BLOWOUT RISK
# =============================================================================

def calculate_blowout_risk(
    spread_prediction: float,
    home_team: str,
    away_team: str
) -> Dict[str, any]:
    """
    Avalia risco de blowout e ajusta projeções de minutos.
    
    Casas de apostas ERRAM ao não ajustar linhas para blowouts.
    Estrelas jogam menos, banco joga mais.
    
    Args:
        spread_prediction: Spread previsto (negativo = home favorito)
        home_team: Time da casa
        away_team: Time visitante
        
    Returns:
        Dict com métricas de blowout
    """
    abs_spread = abs(spread_prediction)
    
    if abs_spread >= 18:
        risk_level = 'EXTREME'
        star_minutes_adj = 0.82  # -18% minutos
        bench_minutes_adj = 1.25  # +25% minutos
    elif abs_spread >= 15:
        risk_level = 'HIGH'
        star_minutes_adj = 0.88  # -12% minutos
        bench_minutes_adj = 1.15  # +15% minutos
    elif abs_spread >= 10:
        risk_level = 'MEDIUM'
        star_minutes_adj = 0.94  # -6% minutos
        bench_minutes_adj = 1.08  # +8% minutos
    elif abs_spread >= 6:
        risk_level = 'LOW'
        star_minutes_adj = 0.97  # -3% minutos
        bench_minutes_adj = 1.03  # +3% minutos
    else:
        risk_level = 'MINIMAL'
        star_minutes_adj = 1.0
        bench_minutes_adj = 1.0
    
    # Determinar quem é favorito/underdog
    home_is_favorite = spread_prediction < 0
    favorite_team = home_team if home_is_favorite else away_team
    underdog_team = away_team if home_is_favorite else home_team
    
    return {
        'blowout_risk': risk_level,
        'abs_spread': abs_spread,
        'star_minutes_adj': star_minutes_adj,
        'bench_minutes_adj': bench_minutes_adj,
        'favorite_team': favorite_team,
        'underdog_team': underdog_team,
        'home_is_favorite': home_is_favorite
    }


def adjust_minutes_for_blowout(
    base_minutes: float,
    is_star_player: bool,
    blowout_data: Dict,
    is_on_favorite: bool
) -> float:
    """
    Ajusta projeção de minutos baseado em risco de blowout.
    
    Args:
        base_minutes: Projeção base de minutos
        is_star_player: Se é jogador estrela (USG% > 25, MIN > 30)
        blowout_data: Dict retornado por calculate_blowout_risk
        is_on_favorite: Se está no time favorito
        
    Returns:
        Minutos ajustados
    """
    if is_on_favorite:
        # Favorito vence: estrelas descansam, banco joga
        if is_star_player:
            return base_minutes * blowout_data['star_minutes_adj']
        else:
            return base_minutes * blowout_data['bench_minutes_adj']
    else:
        # Underdog perde: estrelas podem jogar mais tentando virar, ou serem poupados
        # Em blowouts extremos, todos jogam menos
        if blowout_data['blowout_risk'] in ['EXTREME', 'HIGH']:
            return base_minutes * 0.92  # Todos jogam menos
        else:
            return base_minutes  # Sem ajuste


# =============================================================================
# FASE 1.4: CORRELAÇÃO DE ELENCO (DYNAMIC USAGE)
# =============================================================================

def identify_alpha_players(
    df_player_stats: pd.DataFrame,
    team: str,
    usg_threshold: float = 28.0,
    min_minutes: float = 28.0
) -> List[Dict]:
    """
    Identifica jogadores "Alpha" de um time.
    
    Alpha = USG% > 28 OU minutos > 28
    
    Returns:
        Lista de dicts com info dos alphas
    """
    if df_player_stats is None or df_player_stats.empty:
        return []
    
    try:
        team_players = df_player_stats[df_player_stats['TEAM'] == team]
        
        alphas = team_players[
            (team_players.get('USG_PCT', 0) > usg_threshold) |
            (team_players.get('MIN', 0) > min_minutes)
        ]
        
        return alphas.to_dict('records')
    except Exception as e:
        logger.warning(f"Erro ao identificar alphas: {e}")
        return []


def calculate_usage_covariance(
    df_boxscores: pd.DataFrame,
    player: str,
    alpha_out: str,
    n_games: int = 20
) -> float:
    """
    Calcula covariância de usage quando um alpha está fora.
    
    Quem absorve os arremessos quando LeBron não joga?
    
    Args:
        df_boxscores: Histórico de boxscores
        player: Jogador sendo analisado
        alpha_out: Jogador alpha que está fora
        n_games: Número de jogos para análise
        
    Returns:
        Fator de boost de usage (1.0 = sem mudança)
    """
    if df_boxscores is None or df_boxscores.empty:
        return 1.0
    
    try:
        # Jogos com alpha presente
        games_with_alpha = df_boxscores[
            df_boxscores['player'].str.contains(alpha_out, case=False, na=False)
        ]
        
        # Jogos sem alpha
        games_without_alpha = df_boxscores[
            ~df_boxscores['player'].str.contains(alpha_out, case=False, na=False)
        ]
        
        # Stats do jogador em cada cenário
        player_with = games_with_alpha[
            games_with_alpha['player'].str.contains(player, case=False, na=False)
        ]
        player_without = games_without_alpha[
            games_without_alpha['player'].str.contains(player, case=False, na=False)
        ]
        
        if player_with.empty or player_without.empty:
            return 1.0
        
        # Calcular boost (FGA ou USG como proxy)
        if 'FGA' in player_with.columns:
            fga_with = player_with['FGA'].mean()
            fga_without = player_without['FGA'].mean()
            return fga_without / max(fga_with, 1)
        else:
            return 1.0
            
    except Exception as e:
        logger.warning(f"Erro ao calcular covariância: {e}")
        return 1.0


def calculate_dynamic_usage(
    player: str,
    player_team: str,
    injury_report: List[Dict],
    df_boxscores: pd.DataFrame,
    df_player_stats: pd.DataFrame
) -> Dict[str, float]:
    """
    Calcula boost de usage quando jogadores-chave estão fora.
    
    Args:
        player: Nome do jogador
        player_team: Time do jogador
        injury_report: Lista de jogadores lesionados
        df_boxscores: Histórico de boxscores
        df_player_stats: Stats atuais dos jogadores
        
    Returns:
        Dict com métricas de usage dinâmico
    """
    result = {
        'projected_usage_boost': 1.0,
        'projected_assist_boost': 1.0,
        'projected_reb_boost': 1.0,
        'teammate_out_impact': 0,
        'alphas_out': []
    }
    
    if not injury_report:
        return result
    
    try:
        # Identificar alphas do time
        alphas = identify_alpha_players(df_player_stats, player_team)
        alpha_names = [a.get('PLAYER', '') for a in alphas]
        
        # Verificar quais alphas estão no injury report
        injured_names = [
            i.get('player', i.get('name', '')) 
            for i in injury_report 
            if i.get('status', '').upper() in ['OUT', 'DOUBTFUL']
        ]
        
        alphas_out = [a for a in alpha_names if any(n in a for n in injured_names)]
        result['alphas_out'] = alphas_out
        
        if not alphas_out:
            return result
        
        # Calcular boost para cada alpha fora
        total_usage_boost = 1.0
        total_assist_boost = 1.0
        
        for alpha in alphas_out:
            usage_cov = calculate_usage_covariance(df_boxscores, player, alpha)
            total_usage_boost *= usage_cov
        
        # Limitar boost máximo (não pode dobrar usage)
        result['projected_usage_boost'] = min(1.5, total_usage_boost)
        result['projected_assist_boost'] = min(1.3, total_usage_boost * 0.8)  # Assists crescem menos
        result['projected_reb_boost'] = min(1.2, 1 + (len(alphas_out) * 0.05))
        result['teammate_out_impact'] = min(100, len(alphas_out) * 30)
        
        return result
        
    except Exception as e:
        logger.warning(f"Erro ao calcular dynamic usage: {e}")
        return result


# =============================================================================
# FUNÇÃO PRINCIPAL: GERAR TODAS AS FEATURES QUANTUM
# =============================================================================

def generate_quantum_features(
    player: str,
    player_team: str,
    player_position: str,
    opponent_team: str,
    is_away: bool,
    spread_prediction: float,
    injury_report: List[Dict] = None,
    df_boxscores: pd.DataFrame = None,
    df_player_stats: pd.DataFrame = None,
    minutes_last_game: float = 30.0,
    is_b2b: bool = False,
    games_in_last_5_days: int = 1
) -> Dict[str, any]:
    """
    Gera TODAS as features Quantum para um jogador/jogo.
    
    Esta é a função principal que orquestra todas as outras.
    
    Args:
        player: Nome do jogador
        player_team: Time do jogador (abreviação)
        player_position: Posição (G/F/C)
        opponent_team: Time oponente (abreviação)
        is_away: Se é jogo fora de casa
        spread_prediction: Spread previsto pelo modelo
        injury_report: Lista de lesionados
        df_boxscores: Histórico de boxscores
        df_player_stats: Stats atuais
        minutes_last_game: Minutos no último jogo
        is_b2b: Se é back-to-back
        games_in_last_5_days: Jogos nos últimos 5 dias
        
    Returns:
        Dict com TODAS as features quantum
    """
    logger.info(f"🔬 Gerando features quantum para {player} vs {opponent_team}")
    
    # 1. DvP 2.0
    dvp_features = calculate_dvp_advanced(df_boxscores, opponent_team, player_position)
    
    # 2. Fatigue Index
    fatigue_features = calculate_fatigue_index(
        player_team, opponent_team, is_away,
        minutes_last_game, is_b2b, games_in_last_5_days
    )
    
    # 3. Blowout Risk
    blowout_features = calculate_blowout_risk(spread_prediction, 
        opponent_team if is_away else player_team,
        player_team if is_away else opponent_team
    )
    
    # 4. Dynamic Usage
    usage_features = calculate_dynamic_usage(
        player, player_team, injury_report or [],
        df_boxscores, df_player_stats
    )
    
    # Combinar tudo
    all_features = {
        'player': player,
        'team': player_team,
        'opponent': opponent_team,
        'is_away': is_away,
        **{f'dvp_{k}': v for k, v in dvp_features.items()},
        **{f'fatigue_{k}': v for k, v in fatigue_features.items()},
        **{f'blowout_{k}': v for k, v in blowout_features.items()},
        **{f'usage_{k}': v for k, v in usage_features.items()},
    }
    
    logger.debug(f"Features geradas: {len(all_features)} campos")
    
    return all_features


def generate_quantum_features_batch(
    players_data: List[Dict],
    df_boxscores: pd.DataFrame = None,
    df_player_stats: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Gera features quantum para múltiplos jogadores em batch.
    
    Args:
        players_data: Lista de dicts com dados dos jogadores
        df_boxscores: Histórico compartilhado
        df_player_stats: Stats compartilhados
        
    Returns:
        DataFrame com todas as features
    """
    results = []
    
    for player_data in players_data:
        try:
            features = generate_quantum_features(
                player=player_data.get('player', ''),
                player_team=player_data.get('team', ''),
                player_position=player_data.get('position', 'G'),
                opponent_team=player_data.get('opponent', ''),
                is_away=player_data.get('is_away', False),
                spread_prediction=player_data.get('spread', 0.0),
                injury_report=player_data.get('injuries', []),
                df_boxscores=df_boxscores,
                df_player_stats=df_player_stats,
                minutes_last_game=player_data.get('last_min', 30.0),
                is_b2b=player_data.get('is_b2b', False),
                games_in_last_5_days=player_data.get('games_5d', 1)
            )
            results.append(features)
        except Exception as e:
            logger.error(f"Erro ao gerar features para {player_data.get('player')}: {e}")
    
    return pd.DataFrame(results)


# =============================================================================
# TESTES & VALIDAÇÃO
# =============================================================================

def test_all_features():
    """Testa todas as funções de feature engineering."""
    print("🧪 Testando Props Quantum Features...")
    
    # Teste 1: Circadian Disruption
    circ = calculate_circadian_disruption('BOS', 'LAL', is_away=True)
    print(f"✅ Circadian Disruption (BOS@LAL): {circ} horas")
    
    # Teste 2: Altitude Impact
    alt = calculate_altitude_impact('DEN', 'MIA')
    print(f"✅ Altitude Impact (Miami@Denver): {alt:.2f}")
    
    # Teste 3: Travel Distance
    dist = calculate_travel_distance('BOS', 'LAL')
    print(f"✅ Travel Distance (BOS→LAL): {dist:.0f} km")
    
    # Teste 4: Thirst Index
    thirst = calculate_thirst_index(38, 3000, True)
    print(f"✅ Thirst Index (38min, 3000km, B2B): {thirst:.0f}")
    
    # Teste 5: Blowout Risk
    blowout = calculate_blowout_risk(-17.5, 'BOS', 'WAS')
    print(f"✅ Blowout Risk (BOS -17.5 vs WAS): {blowout['blowout_risk']}")
    
    # Teste 6: Fatigue Index completo
    fatigue = calculate_fatigue_index('MIA', 'DEN', True, 36, True, 3)
    print(f"✅ Fatigue Score (MIA@DEN, B2B, 36min): {fatigue['fatigue_score']:.0f}")
    
    print("\n✅ Todos os testes passaram!")


if __name__ == "__main__":
    test_all_features()
