"""
Roster Manager: Calcula força do elenco disponível baseado em lesões.

Conceito:
- Lakers com LeBron ≠ Lakers sem LeBron
- Calcula roster_strength = soma(BPM dos jogadores disponíveis)
"""
import logging
import pandas as pd
from config.constants import TEAM_ABBREV_MAP

logger = logging.getLogger(__name__)

def normalize_player_name(name):
    """Normaliza nome de jogador para matching"""
    if not name:
        return ""
    # Remover sufixos (Jr., III, etc.)
    name = name.replace(" Jr.", "").replace(" III", "").replace(" II", "")
    # Remover pontos e converter para lowercase
    name = name.replace(".", "").lower().strip()
    return name

def calculate_roster_strength(team_abbr, injury_report, player_stats):
    """
    Calcula a força do roster de um time baseado nos jogadores DISPONÍVEIS.
    
    Parameters:
    -----------
    team_abbr : str
        Abreviação do time (ex: 'LAL')
    injury_report : dict
        {team: [{'player': 'LeBron James', 'status': 'Out'}, ...]}
    player_stats : dict
        {'bpm': DataFrame, 'rapm': DataFrame, etc.}
        
    Returns:
    --------
    dict : {
        'available_bpm': float (soma BPM dos disponíveis),
        'injured_bpm': float (soma BPM dos lesionados),
        'roster_strength': float (available - injured),
        'injury_impact': float (-injured_bpm, sempre negativo)
    }
    """
    
    # Verificar se temos dados de BPM
    if 'bpm' not in player_stats or 'BBALL_REF' not in player_stats:
        logger.warning(f"⚠️  Sem dados de BPM para calcular roster strength")
        return {
            'available_bpm': 0.0,
            'injured_bpm': 0.0,
            'roster_strength': 0.0,
            'injury_impact': 0.0
        }
    
    # Pegar DataFrame de BPM (Basketball Reference)
    bpm_df = player_stats.get('BBALL_REF')
    if bpm_df is None or bpm_df.empty:
        return {
            'available_bpm': 0.0,
            'injured_bpm': 0.0,
            'roster_strength': 0.0,
            'injury_impact': 0.0
        }
    
    # Filtrar jogadores do time
    # Tentar por Team coluna
    if 'Team' in bpm_df.columns:
        team_players = bpm_df[bpm_df['Team'] == team_abbr].copy()
    elif 'Tm' in bpm_df.columns:
        team_players = bpm_df[bpm_df['Tm'] == team_abbr].copy()
    else:
        logger.warning(f"⚠️  Coluna de time não encontrada no DataFrame de BPM")
        return {
            'available_bpm': 0.0,
            'injured_bpm': 0.0,
            'roster_strength': 0.0,
            'injury_impact': 0.0
        }
    
    if team_players.empty:
        logger.debug(f"Nenhum jogador encontrado para {team_abbr}")
        return {
            'available_bpm': 0.0,
            'injured_bpm': 0.0,
            'roster_strength': 0.0,
            'injury_impact': 0.0
        }
    
    # Normalizar nomes de jogadores
    team_players['player_normalized'] = team_players['Player'].apply(normalize_player_name)
    
    # Pegar lista de lesionados do time
    team_name_full = [k for k, v in TEAM_ABBREV_MAP.items() if v == team_abbr]
    injured_players = []
    
    if team_name_full and injury_report:
        team_injuries = injury_report.get(team_name_full[0], [])
        injured_players = [
            normalize_player_name(inj.get('player', '')) 
            for inj in team_injuries 
            if inj.get('status', '').upper() in ['OUT', 'DOUBTFUL']
        ]
    
    # Calcular BPMs
    total_bpm = 0.0
    injured_bpm = 0.0
    
    for _, player in team_players.iterrows():
        player_name_norm = player['player_normalized']
        bpm_value = float(player.get('BPM', 0))
        
        if player_name_norm in injured_players:
            injured_bpm += bpm_value
        else:
            total_bpm += bpm_value
    
    return {
        'available_bpm': round(total_bpm, 2),
        'injured_bpm': round(injured_bpm, 2),
        'roster_strength': round(total_bpm - abs(injured_bpm), 2),  # Força líquida
        'injury_impact': round(-abs(injured_bpm), 2)  # Sempre negativo
    }


def get_roster_strengths_for_game(home_team, away_team, injury_report, player_stats):
    """
    Calcula roster strength para ambos os times de um jogo.
    
    Returns:
    --------
    dict : {
        'home_roster_strength': float,
        'away_roster_strength': float,
        'home_injury_impact': float,
        'away_injury_impact': float
    }
    """
    home_roster = calculate_roster_strength(home_team, injury_report, player_stats)
    away_roster = calculate_roster_strength(away_team, injury_report, player_stats)
    
    return {
        'home_roster_strength': home_roster['roster_strength'],
        'away_roster_strength': away_roster['roster_strength'],
        'home_injury_impact': home_roster['injury_impact'],
        'away_injury_impact': away_roster['injury_impact'],
        'home_available_bpm': home_roster['available_bpm'],
        'away_available_bpm': away_roster['available_bpm']
    }
