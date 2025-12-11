"""
Gerador de Combos e Parlays de Apostas
========================================

Módulo para gerar combinações inteligentes de apostas incluindo:
- Time vencedor + Player props
- Parlays multi-time
- Combos complexos customizados

Uso:
    from betting.combo_generator import generate_smart_combos, generate_multi_team_parlays
    
    combos = generate_smart_combos(daily_games, player_props, min_ev=5.0)
    parlays = generate_multi_team_parlays(daily_games, parlay_size=3, min_ev=10.0)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


def calculate_parlay_probability(probabilities: List[float]) -> float:
    """
    Calcula a probabilidade de um parlay (produto das probabilidades individuais).
    
    Assume independência entre eventos.
    
    Args:
        probabilities: Lista de probabilidades individuais (0-1)
    
    Returns:
        Probabilidade combinada do parlay
    """
    if not probabilities:
        return 0.0
    
    prob = 1.0
    for p in probabilities:
        prob *= p
    
    return prob


def calculate_parlay_odds(odds_list: List[float]) -> float:
    """
    Calcula a odd combinada de um parlay (produto das odds individuais).
    
    Args:
        odds_list: Lista de odds decimais
    
    Returns:
        Odd combinada do parlay
    """
    if not odds_list:
        return 0.0
    
    combined_odd = 1.0
    for odd in odds_list:
        combined_odd *= odd
    
    return combined_odd


def calculate_parlay_ev(probability: float, combined_odd: float) -> float:
    """
    Calcula o Expected Value (EV) de um parlay.
    
    EV = (Probabilidade * Odd - 1) * 100
    
    Args:
        probability: Probabilidade combinada do parlay (0-1)
        combined_odd: Odd combinada do parlay
    
    Returns:
        EV em porcentagem
    """
    return (probability * combined_odd - 1) * 100


def generate_team_player_combos(
    daily_games: pd.DataFrame,
    player_props: pd.DataFrame,
    min_ev: float = 5.0,
    max_combos: int = 20
) -> List[Dict[str, Any]]:
    """
    Gera combos de Time Vencedor + Player Prop do mesmo time.
    
    Args:
        daily_games: DataFrame com previsões de jogos
        player_props: DataFrame com previsões de player props
        min_ev: EV mínimo para incluir combo (default: 5%)
        max_combos: Número máximo de combos a retornar
    
    Returns:
        Lista de combos ordenados por EV
    """
    combos = []
    
    try:
        for _, game in daily_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Probabilidades e odds dos times
            prob_home = game.get('prob_home', 0) / 100.0  # Converter para 0-1
            prob_away = game.get('prob_away', 0) / 100.0
            
            odds_home = game.get('odds_home', 0)
            odds_away = game.get('odds_away', 0)
            
            # Se odds não disponível, estimar com fair value
            if odds_home == 0 and prob_home > 0:
                odds_home = 1 / prob_home if prob_home > 0.01 else 100
            if odds_away == 0 and prob_away > 0:
                odds_away = 1 / prob_away if prob_away > 0.01 else 100
            
            # Filtrar player props deste jogo (home team)
            if 'team' in player_props.columns:
                home_props = player_props[player_props['team'] == home_team]
                away_props = player_props[player_props['team'] == away_team]
                
                # Combos: Home team vence + Home player prop
                for _, prop in home_props.iterrows():
                    player_name = prop.get('player', 'Unknown')
                    stat_type = prop.get('stat_type', 'PTS')
                    line = prop.get('line', 0)
                    prob_over = prop.get('prob_over', 50) / 100.0
                    
                    # Odd estimada para player prop (simplificado)
                    odd_prop = 1 / prob_over if prob_over > 0.01 else 2.0
                    
                    # Calcular combo
                    combined_prob = prob_home * prob_over
                    combined_odd = odds_home * odd_prop
                    ev = calculate_parlay_ev(combined_prob, combined_odd)
                    
                    if ev >= min_ev:
                        combos.append({
                            'type': 'team_player',
                            'components': [
                                {
                                    'event': f'{home_team} vence',
                                    'prob': prob_home,
                                    'odd': odds_home
                                },
                                {
                                    'event': f'{player_name} {stat_type} Over {line:.1f}',
                                    'prob': prob_over,
                                    'odd': odd_prop
                                }
                            ],
                            'combined_prob': combined_prob,
                            'combined_odd': combined_odd,
                            'ev': ev,
                            'description': f'{home_team} vence + {player_name} {stat_type} Over {line:.1f}'
                        })
                
                # Combos: Away team vence + Away player prop
                for _, prop in away_props.iterrows():
                    player_name = prop.get('player', 'Unknown')
                    stat_type = prop.get('stat_type', 'PTS')
                    line = prop.get('line', 0)
                    prob_over = prop.get('prob_over', 50) / 100.0
                    
                    odd_prop = 1 / prob_over if prob_over > 0.01 else 2.0
                    
                    combined_prob = prob_away * prob_over
                    combined_odd = odds_away * odd_prop
                    ev = calculate_parlay_ev(combined_prob, combined_odd)
                    
                    if ev >= min_ev:
                        combos.append({
                            'type': 'team_player',
                            'components': [
                                {
                                    'event': f'{away_team} vence',
                                    'prob': prob_away,
                                    'odd': odds_away
                                },
                                {
                                    'event': f'{player_name} {stat_type} Over {line:.1f}',
                                    'prob': prob_over,
                                    'odd': odd_prop
                                }
                            ],
                            'combined_prob': combined_prob,
                            'combined_odd': combined_odd,
                            'ev': ev,
                            'description': f'{away_team} vence + {player_name} {stat_type} Over {line:.1f}'
                        })
    
    except Exception as e:
        logger.error(f"Erro ao gerar team+player combos: {e}")
    
    # Ordenar por EV e limitar
    combos.sort(key=lambda x: x['ev'], reverse=True)
    return combos[:max_combos]


def generate_multi_team_parlays(
    daily_games: pd.DataFrame,
    parlay_size: int = 3,
    min_ev: float = 10.0,
    min_prob_per_game: float = 0.55,
    max_parlays: int = 15
) -> List[Dict[str, Any]]:
    """
    Gera parlays de múltiplos times.
    
    Args:
        daily_games: DataFrame com previsões de jogos
        parlay_size: Número de times no parlay (2, 3, 4, etc.)
        min_ev: EV mínimo para incluir parlay
        min_prob_per_game: Probabilidade mínima por jogo individual (evita azarões)
        max_parlays: Número máximo de parlays a retornar
    
    Returns:
        Lista de parlays ordenados por EV
    """
    parlays = []
    
    try:
        # Preparar lista de picks (favoritos com prob > min_prob_per_game)
        picks = []
        
        for _, game in daily_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            prob_home = game.get('prob_home', 0) / 100.0
            prob_away = game.get('prob_away', 0) / 100.0
            odds_home = game.get('odds_home', 0)
            odds_away = game.get('odds_away', 0)
            
            # Estimar odds se não disponível
            if odds_home == 0 and prob_home > 0:
                odds_home = 1 / prob_home if prob_home > 0.01 else 100
            if odds_away == 0 and prob_away > 0:
                odds_away = 1 / prob_away if prob_away > 0.01 else 100
            
            # Adicionar home team se probabilidade suficiente
            if prob_home >= min_prob_per_game:
                picks.append({
                    'team': home_team,
                    'opponent': away_team,
                    'is_home': True,
                    'prob': prob_home,
                    'odd': odds_home,
                    'game_date': game.get('date', '')
                })
            
            # Adicionar away team se probabilidade suficiente
            if prob_away >= min_prob_per_game:
                picks.append({
                    'team': away_team,
                    'opponent': home_team,
                    'is_home': False,
                    'prob': prob_away,
                    'odd': odds_away,
                    'game_date': game.get('date', '')
                })
        
        # Gerar todas as combinações possíveis
        if len(picks) >= parlay_size:
            for combo in combinations(picks, parlay_size):
                # Verificar que não há times duplicados (conflito)
                teams_in_parlay = set()
                valid = True
                
                for pick in combo:
                    if pick['team'] in teams_in_parlay or pick['opponent'] in teams_in_parlay:
                        valid = False
                        break
                    teams_in_parlay.add(pick['team'])
                
                if not valid:
                    continue
                
                # Calcular probabilidade e odds combinadas
                probs = [pick['prob'] for pick in combo]
                odds = [pick['odd'] for pick in combo]
                
                combined_prob = calculate_parlay_probability(probs)
                combined_odd = calculate_parlay_odds(odds)
                ev = calculate_parlay_ev(combined_prob, combined_odd)
                
                if ev >= min_ev:
                    components = []
                    description_parts = []
                    
                    for pick in combo:
                        location = 'vs' if pick['is_home'] else '@'
                        event_desc = f"{pick['team']} vence ({location} {pick['opponent']})"
                        
                        components.append({
                            'event': event_desc,
                            'prob': pick['prob'],
                            'odd': pick['odd']
                        })
                        description_parts.append(f"{pick['team']} vence")
                    
                    parlays.append({
                        'type': f'{parlay_size}_team_parlay',
                        'components': components,
                        'combined_prob': combined_prob,
                        'combined_odd': combined_odd,
                        'ev': ev,
                        'description': ' + '.join(description_parts)
                    })
        
    except Exception as e:
        logger.error(f"Erro ao gerar parlays multi-time: {e}")
    
    # Ordenar por EV e limitar
    parlays.sort(key=lambda x: x['ev'], reverse=True)
    return parlays[:max_parlays]


def generate_smart_combos(
    daily_games: pd.DataFrame,
    player_props: Optional[pd.DataFrame] = None,
    min_ev: float = 5.0,
    include_team_player: bool = True,
    include_parlays: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Gera todos os tipos de combos inteligentes automaticamente.
    
    Args:
        daily_games: DataFrame com previsões de jogos
        player_props: DataFrame com previsões de player props (opcional)
        min_ev: EV mínimo para incluir combos
        include_team_player: Se True, inclui combos team+player
        include_parlays: Se True, inclui parlays multi-time
    
    Returns:
        Dicionário com diferentes tipos de combos:
        {
            'team_player': [...],
            'parlay_2': [...],
            'parlay_3': [...],
            'parlay_4': [...]
        }
    """
    result = {
        'team_player': [],
        'parlay_2': [],
        'parlay_3': [],
        'parlay_4': []
    }
    
    try:
        # Team + Player combos
        if include_team_player and player_props is not None and not player_props.empty:
            result['team_player'] = generate_team_player_combos(
                daily_games, 
                player_props, 
                min_ev=min_ev,
                max_combos=10
            )
        
        # Multi-team parlays
        if include_parlays and len(daily_games) >= 2:
            # 2-team parlays (menos restritivo)
            result['parlay_2'] = generate_multi_team_parlays(
                daily_games,
                parlay_size=2,
                min_ev=min_ev,
                min_prob_per_game=0.45,  # Reduzido de 0.55 para aceitar mais combos
                max_parlays=10
            )
            
            # 3-team parlays
            if len(daily_games) >= 3:
                result['parlay_3'] = generate_multi_team_parlays(
                    daily_games,
                    parlay_size=3,
                    min_ev=min_ev,  # EV igual ao mínimo
                    min_prob_per_game=0.50,  # Reduzido de 0.60
                    max_parlays=8
                )
            
            # 4-team parlays
            if len(daily_games) >= 4:
                result['parlay_4'] = generate_multi_team_parlays(
                    daily_games,
                    parlay_size=4,
                    min_ev=min_ev,  # EV igual ao mínimo
                    min_prob_per_game=0.55,  # Reduzido de 0.65
                    max_parlays=5
                )
    
    except Exception as e:
        logger.error(f"Erro ao gerar smart combos: {e}")
    
    return result


def create_custom_combo(
    selected_picks: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Cria um combo customizado a partir de picks selecionados pelo usuário.
    
    Args:
        selected_picks: Lista de picks selecionados, cada um com 'event', 'prob', 'odd'
    
    Returns:
        Combo combinado ou None se inválido
    """
    if not selected_picks or len(selected_picks) < 2:
        return None
    
    try:
        probs = [pick['prob'] for pick in selected_picks]
        odds = [pick['odd'] for pick in selected_picks]
        
        combined_prob = calculate_parlay_probability(probs)
        combined_odd = calculate_parlay_odds(odds)
        ev = calculate_parlay_ev(combined_prob, combined_odd)
        
        descriptions = [pick['event'] for pick in selected_picks]
        
        return {
            'type': 'custom',
            'components': selected_picks,
            'combined_prob': combined_prob,
            'combined_odd': combined_odd,
            'ev': ev,
            'description': ' + '.join(descriptions)
        }
    
    except Exception as e:
        logger.error(f"Erro ao criar combo customizado: {e}")
        return None
