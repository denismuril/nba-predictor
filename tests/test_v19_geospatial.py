#!/usr/bin/env python3
"""
v19.0 GeoSpatial Test Script
Verifica implementação de cálculo geodésico em fatigue_calculator.py
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

def test_haversine():
    from core.fatigue_calculator import haversine_distance, NBA_ARENA_COORDS
    
    print("=== Test 1: Verificar Coordenadas ===")
    print(f"Total arenas: {len(NBA_ARENA_COORDS)}")
    assert len(NBA_ARENA_COORDS) >= 30, "Deve ter pelo menos 30 arenas"
    print("✅ 30+ arenas carregadas")
    
    print("\n=== Test 2: Haversine LAL -> BOS ===")
    lal = NBA_ARENA_COORDS['LAL']
    bos = NBA_ARENA_COORDS['BOS']
    dist = haversine_distance(lal, bos)
    print(f"LAL {lal} -> BOS {bos}: {dist:.0f} miles")
    assert 2400 < dist < 2700, f"Distância LAL-BOS deve ser ~2500mi, got {dist}"
    print("✅ Haversine funcionando corretamente")
    
    print("\n=== Test 3: Mesma Arena ===")
    same = haversine_distance(lal, lal)
    print(f"LAL -> LAL: {same:.0f} miles")
    assert same == 0, "Mesma arena deve ter distância 0"
    print("✅ Distância zero para mesma arena")
    
    print("\n=== Test 4: Coast to Coast GSW -> NYK ===")
    gsw = NBA_ARENA_COORDS['GSW']
    nyk = NBA_ARENA_COORDS['NYK']
    dist2 = haversine_distance(gsw, nyk)
    print(f"GSW {gsw} -> NYK {nyk}: {dist2:.0f} miles")
    assert 2500 < dist2 < 3000, f"Distância GSW-NYK deve ser ~2550mi"
    print("✅ Coast-to-coast correto")
    
    print("\n🎉 HAVERSINE TESTS PASSED!")

if __name__ == "__main__":
    test_haversine()
