"""
Arena Constants - Coordenadas e Multiplicadores de Home Court Advantage

Baseado em análise de dados históricos e características únicas de cada arena.

Multiplicadores refletem:
1. Altitude (DEN, UTA)
2. Crowd intensity (POR, MIA, BOS)
3. Travel difficulty (remote locations)
4. Historical home advantage

Base HCA League (2025-26): ~2.8 pontos
"""

# Coordenadas das Arenas NBA (Latitude, Longitude)
NBA_ARENA_LOCATIONS = {
    # Eastern Conference
    'ATL': (33.7573, -84.3963),   # State Farm Arena
    'BOS': (42.3662, -71.0621),   # TD Garden
    'BRK': (40.6826, -73.9754),   # Barclays Center
    'CHA': (35.2251, -80.8392),   # Spectrum Center
    'CHI': (41.8807, -87.6742),   # United Center
    'CLE': (41.4965, -81.6882),   # Rocket Mortgage FieldHouse
    'DET': (42.3410, -83.0550),   # Little Caesars Arena
    'IND': (39.7640, -86.1555),   # Gainbridge Fieldhouse
    'MIA': (25.7814, -80.1870),   # FTX Arena (Miami)
    'MIL': (43.0435, -87.9170),   # Fiserv Forum
    'NYK': (40.7505, -73.9934),   # Madison Square Garden
    'ORL': (28.5392, -81.3839),   # Amway Center
    'PHI': (39.9012, -75.1720),   # Wells Fargo Center
    'TOR': (43.6435, -79.3791),   # Scotiabank Arena
    'WAS': (38.8981, -77.0209),   # Capital One Arena
    
    # Western Conference
    'DAL': (32.7905, -96.8103),   # American Airlines Center
    'DEN': (39.7487, -105.0077),  # Ball Arena (Altitude 5280ft)
    'GSW': (37.7680, -122.3877),  # Chase Center
    'HOU': (29.7508, -95.3621),   # Toyota Center
    'LAC': (34.0430, -118.2673),  # Crypto.com Arena (shared with LAL)
    'LAL': (34.0430, -118.2673),  # Crypto.com Arena
    'MEM': (35.1382, -90.0506),   # FedExForum
    'MIN': (44.9795, -93.2760),   # Target Center
    'NOP': (29.9490, -90.0821),   # Smoothie King Center
    'OKC': (35.4634, -97.5151),   # Paycom Center
    'PHO': (33.4457, -112.0712),  # Footprint Center
    'POR': (45.5316, -122.6668),  # Moda Center
    'SAC': (38.5802, -121.4997),  # Golden 1 Center
    'SAS': (29.4270, -98.4375),   # AT&T Center
    'UTA': (40.7683, -111.9011),  # Delta Center (Altitude 4226ft)
}

# Aliases para compatibilidade
NBA_ARENA_LOCATIONS['CHO'] = NBA_ARENA_LOCATIONS['CHA']  # Charlotte
NBA_ARENA_LOCATIONS['PHX'] = NBA_ARENA_LOCATIONS['PHO']  # Phoenix
NBA_ARENA_LOCATIONS['BKN'] = NBA_ARENA_LOCATIONS['BRK']  # Brooklyn

# Home Court Advantage Multipliers (baseado em dados históricos)
# Multiplicador aplicado sobre base HCA (~2.8 pts)
ARENA_HCA_MULTIPLIERS = {
    # Tier 1: Elite Home Advantage (>1.20x)
    'DEN': 1.35,  # Altitude + Elite home record
    'UTA': 1.28,  # Altitude + Historically strong
    'POR': 1.22,  # Loud crowd, tough travel
    
    # Tier 2: Above Average (1.10x - 1.20x)
    'MIA': 1.18,  # Heat culture + crowd
    'BOS': 1.16,  # TD Garden atmosphere
    'GSW': 1.15,  # Warriors bandwagon still strong
    'PHI': 1.12,  # Passionate fanbase
    'MIN': 1.10,  # Target Center noise
    
    # Tier 3: Average (1.00x - 1.10x)
    'MIL': 1.08,
    'PHO': 1.07,  # Desert heat factor
    'DAL': 1.05,
    'MEM': 1.04,
    'OKC': 1.03,
    'SAC': 1.02,
    'ORL': 1.02,
    'IND': 1.01,
    'CLE': 1.00,
    'TOR': 1.00,
    'NOP': 1.00,
    'CHI': 1.00,
    'SAS': 1.00,
    
    # Tier 4: Below Average (0.90x - 1.00x)
    'ATL': 0.98,  # Inconsistent crowds
    'HOU': 0.97,
    'DET': 0.96,
    'WAS': 0.95,
    'CHA': 0.94,
    
    # Tier 5: Weak Home Advantage (<0.90x)
    'LAL': 0.92,  # Many opponent fans (tourism)
    'LAC': 0.90,  # Shares arena, less identity
    'NYK': 0.92,  # MSG is iconic but many road fans
    'BRK': 0.88,  # Still building home culture
}

# Aliases
ARENA_HCA_MULTIPLIERS['CHO'] = ARENA_HCA_MULTIPLIERS['CHA']
ARENA_HCA_MULTIPLIERS['PHX'] = ARENA_HCA_MULTIPLIERS['PHO']
ARENA_HCA_MULTIPLIERS['BKN'] = ARENA_HCA_MULTIPLIERS['BRK']

# Altitude Advantage (feet above sea level)
ARENA_ALTITUDE = {
    'DEN': 5280,  # Mile High City
    'UTA': 4226,  # Salt Lake City
    'PHO': 1117,
    'SAS': 650,
    # Outros times < 500ft (negligível)
}

# League Average HCA por temporada
# Math-Fix: HCA atualizado para refletir tendência de queda pós-2020
LEAGUE_AVG_HCA_BY_SEASON = {
    '2019-20': 3.2,   # Pré-COVID
    '2020-21': 1.8,   # Bubble + sem torcida
    '2021-22': 2.2,   # Retorno gradual
    '2022-23': 2.1,   # Normalização
    '2023-24': 2.05,  # Math-Fix: Atualizado (era 2.8)
    '2024-25': 2.05,  # Math-Fix: Alinhado com Vegas/Pinnacle
    '2025-26': 2.05,  # Projeção conservadora
}

# Current season
CURRENT_SEASON = '2025-26'
BASE_HCA = LEAGUE_AVG_HCA_BY_SEASON.get(CURRENT_SEASON, 2.8)
