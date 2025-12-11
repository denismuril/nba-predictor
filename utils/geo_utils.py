"""
Geographic Utilities - Cálculo de Distâncias entre Arenas

Usado para calcular Travel Distance e Fatigue Index.
"""
import math
from typing import Tuple


def haversine_distance(
    coord1: Tuple[float, float], 
    coord2: Tuple[float, float]
) -> float:
    """
    Calcula distância entre dois pontos na Terra usando Haversine formula.
    
    Args:
        coord1: (latitude, longitude) do ponto 1
        coord2: (latitude, longitude) do ponto 2
    
    Returns:
        Distância em milhas
    
    Example:
        >>> coord_nyc = (40.7505, -73.9934)
        >>> coord_la = (34.0430, -118.2673)
        >>> distance = haversine_distance(coord_nyc, coord_la)
        >>> print(f"{distance:.0f} miles")  # ~2451 miles
    """
    # Raio da Terra em milhas
    R = 3959.0
    
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
    
    distance = R * c
    
    return distance


def calculate_time_zone_change(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> int:
    """
    Estima mudança de timezone baseada na longitude.
    
    Aproximação: Cada 15° de longitude ≈ 1 hora de timezone.
    
    Args:
        coord1: (lat, lon) origem
        coord2: (lat, lon) destino
    
    Returns:
        Mudança de timezone em horas (absoluto)
    
    Example:
        >>> coord_bos = (42.3662, -71.0621)  # Eastern
        >>> coord_por = (45.5316, -122.6668)  # Pacific
        >>> tz_change = calculate_time_zone_change(coord_bos, coord_por)
        >>> print(f"{tz_change} hours")  # 3 hours
    """
    _, lon1 = coord1
    _, lon2 = coord2
    
    # Diferença em longitude
    delta_lon = abs(lon2 - lon1)
    
    # Converter para horas de diferença (15° = 1h)
    tz_change = int(round(delta_lon / 15.0))
    
    return tz_change


if __name__ == '__main__':
    # Demo
    from config.arena_constants import NBA_ARENA_LOCATIONS
    
    print("🗺️  Geo Utils Demo - NBA Travel Distances\n")
    
    # Exemplo: Road trip Lakers
    lal_coord = NBA_ARENA_LOCATIONS['LAL']
    bos_coord = NBA_ARENA_LOCATIONS['BOS']
    mia_coord = NBA_ARENA_LOCATIONS['MIA']
    
    # LAL → BOS
    distance_to_boston = haversine_distance(lal_coord, bos_coord)
    tz_change_boston = calculate_time_zone_change(lal_coord, bos_coord)
    
    print(f"Lakers → Boston:")
    print(f"  Distance: {distance_to_boston:.0f} miles")
    print(f"  Timezone change: {tz_change_boston} hours\n")
    
    # BOS → MIA
    distance_to_miami = haversine_distance(bos_coord, mia_coord)
    tz_change_miami = calculate_time_zone_change(bos_coord, mia_coord)
    
    print(f"Boston → Miami:")
    print(f"  Distance: {distance_to_miami:.0f} miles")
    print(f"  Timezone change: {tz_change_miami} hours\n")
    
    # Total road trip
    total_miles = distance_to_boston + distance_to_miami
    print(f"Total 2-game road trip: {total_miles:.0f} miles")
    
    print("\n✅ Demo completo!")
