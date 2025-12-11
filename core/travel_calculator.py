"""
Travel Distance & Fatigue Calculator
Calcula pontuação de fadiga baseada em:
1. Distância viajada (km)
2. Mudança de fuso horário (Jet Lag)
3. Densidade de jogos (Back-to-Backs)
"""
import json
import logging
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from datetime import timedelta

logger = logging.getLogger(__name__)

# Load arena locations
ARENA_FILE = Path(__file__).parent.parent / 'data' / 'arena_locations.json'

try:
    with open(ARENA_FILE, 'r') as f:
        ARENA_LOCATIONS = json.load(f)
except Exception as e:
    logger.warning(f"⚠️  Erro ao carregar arena_locations.json: {e}. Usando coordenadas zeradas.")
    ARENA_LOCATIONS = {}

def haversine(lat1, lon1, lat2, lon2):
    """Calcula distância em KM entre duas coordenadas."""
    R = 6371  # Raio da Terra em km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def get_arena_info(team_name):
    """Retorna dict com lat, lon, timezone para um time."""
    # Tentar match direto ou parcial
    if team_name in ARENA_LOCATIONS:
        return ARENA_LOCATIONS[team_name]
    
    # Fallback para busca parcial se necessário
    for key, val in ARENA_LOCATIONS.items():
        if key in team_name or team_name in key:
            return val
            
    return {'lat': 0, 'lon': 0, 'timezone': 0} # Default

def calculate_fatigue_score(distance_km, timezone_diff, games_in_72h, is_b2b):
    """
    Calcula Score de Fadiga (0 a 100).
    Fórmula empírica baseada em ciência do esporte.
    """
    score = 0
    
    # 1. Distância (aprox 1 ponto a cada 300km)
    score += (distance_km / 300)
    
    # 2. Jet Lag (3 pontos por hora de diferença)
    score += (abs(timezone_diff) * 3)
    
    # 3. Carga de Jogos
    if games_in_72h >= 3:
        score += 20  # 3 jogos em 3 noites (raro) ou 4 em 5
    elif games_in_72h == 2:
        score += 5
        
    # 4. Back-to-Back direto
    if is_b2b:
        score += 15
        
    # Cap em 100
    return min(100, score)

def calculate_schedule_fatigue(df):
    """
    Processa o calendário e calcula fadiga acumulada para cada jogo.
    Retorna DataFrame enriquecido com colunas 'home_fatigue_score' e 'away_fatigue_score'.
    """
    logger.info("✈️  Calculando métricas de fadiga de viagem...")
    
    df = df.sort_values('date').copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Estrutura para rastrear estado atual de cada time
    team_state = {} # {team: {'last_date': date, 'last_lat': lat, 'last_lon': lon, 'last_tz': tz}}
    
    # Inicializar colunas
    df['home_fatigue_score'] = 0.0
    df['away_fatigue_score'] = 0.0
    df['home_distance_km'] = 0.0
    df['away_distance_km'] = 0.0
    
    # Processar cronologicamente
    for idx, row in df.iterrows():
        game_date = row['date']
        
        for side in ['home', 'away']:
            team = row[f'{side}_team']
            arena = get_arena_info(team if side == 'home' else row['home_team']) # Se away, joga na arena do home
            
            # Estado anterior do time
            if team not in team_state:
                # Assumir que começa em casa descansado
                home_arena = get_arena_info(team)
                team_state[team] = {
                    'last_date': game_date - timedelta(days=5),
                    'last_lat': home_arena['lat'],
                    'last_lon': home_arena['lon'],
                    'last_tz': home_arena.get('timezone', -5),
                    'games_72h': []
                }
            
            state = team_state[team]
            
            # 1. Calcular Distância da última localização até o jogo atual
            # Se for home team, o jogo é na sua arena. Se for away, é na arena do home.
            current_lat = arena['lat']
            current_lon = arena['lon']
            current_tz = arena.get('timezone', -5)
            
            dist = haversine(state['last_lat'], state['last_lon'], current_lat, current_lon)
            tz_diff = current_tz - state['last_tz']
            
            # 2. Calcular Jogos nas últimas 72h
            # Limpar jogos antigos da janela
            state['games_72h'] = [d for d in state['games_72h'] if (game_date - d).days <= 3]
            games_in_72h = len(state['games_72h'])
            
            # 3. Verificar B2B (jogo ontem)
            days_since_last = (game_date - state['last_date']).days
            is_b2b = days_since_last == 1
            
            # 4. Calcular Score Final
            fatigue = calculate_fatigue_score(dist, tz_diff, games_in_72h, is_b2b)
            
            # Atualizar DataFrame
            df.at[idx, f'{side}_fatigue_score'] = round(fatigue, 1)
            df.at[idx, f'{side}_distance_km'] = round(dist, 1)
            
            # Atualizar Estado do Time
            state['last_date'] = game_date
            state['last_lat'] = current_lat
            state['last_lon'] = current_lon
            state['last_tz'] = current_tz
            state['games_72h'].append(game_date)
            
    logger.info("✅ Fadiga calculada com sucesso.")
    return df

if __name__ == "__main__":
    # Teste rápido
    print("Teste de Cálculo de Fadiga:")
    print(f"B2B + Viagem Curta (500km): {calculate_fatigue_score(500, 0, 1, True)}")
    print(f"Descansado + Viagem Longa (3000km + 3h fuso): {calculate_fatigue_score(3000, 3, 0, False)}")
    print(f"Inferno (B2B + 2000km + 3º jogo em 4 noites): {calculate_fatigue_score(2000, 1, 3, True)}")
