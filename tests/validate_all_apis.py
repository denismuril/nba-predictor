"""
Multi-API NBA Advanced Stats Scraper with Fallback

Testa e usa 3 APIs em ordem de prioridade:
1. API-Football/API-Sports (já validado)
2. SportsBlaze 
3. SportData.io

Usage:
    python tests/validate_all_apis.py
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json

print("🏀 Validando TODAS as APIs para NBA Advanced Stats\n")
print("="*60)

# ============================================================================
# API 1: API-Football/API-Sports (JÁ VALIDADO ✅)
# ============================================================================
print("\n1️⃣ API-FOOTBALL/API-SPORTS")
print("-" * 60)

api_football_key = '01eee81ebe305e3e88ced3e2de4905c1'

try:
    url = "https://v2.nba.api-sports.io/games/statistics"
    headers = {'x-apisports-key': api_football_key}
    params = {'id': 10403}
    
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('results', 0) > 0:
            stats = data['response'][0]['statistics'][0]
            print(f"✅ API-Football: FUNCIONANDO!")
            print(f"   Fast Break: {stats.get('fastBreakPoints', 'N/A')}")
            print(f"   Second Chance: {stats.get('secondChancePoints', 'N/A')}")
            print(f"   Paint: {stats.get('pointsInPaint', 'N/A')}")
            api_football_ok = True
        else:
            print(f"⚠️ API-Football: Sem resultados")
            api_football_ok = False
    else:
        print(f"❌ API-Football: Status {response.status_code}")
        api_football_ok = False
except Exception as e:
    print(f"❌ API-Football: Erro - {e}")
    api_football_ok = False

# ============================================================================
# API 2: SportsBlaze
# ============================================================================
print("\n2️⃣ SPORTSBLAZE")
print("-" * 60)

sportsblaze_key = 'sbfxqpy6v6fjljvobf61a5o'

try:
    # Testar endpoint de boxscores diários (exemplo com data)
    url = "https://api.sportsblaze.com/nba/v1/boxscores/daily/2025-02-09.json"
    params = {'key': sportsblaze_key}
    
    response = requests.get(url, headers={}, params=params, timeout=10)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SportsBlaze: CONECTADO!")
        print(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else 'Lista'}")
        
        # Tentar encontrar stats detalhadas
        if isinstance(data, dict) and 'games' in data:
            games = data.get('games', [])
            if games and len(games) > 0:
                game = games[0]
                print(f"   Sample game keys: {list(game.keys())[:5]}")
                # Verificar se tem second_chance, fast_break, etc
                team_stats = game.get('home_team', {}) or game.get('teams', {})
                print(f"   Team stats sample: {list(team_stats.keys())[:10] if isinstance(team_stats, dict) else 'N/A'}")
        
        sportsblaze_ok = True
    else:
        print(f"⚠️ SportsBlaze: Status {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        sportsblaze_ok = False
        
except Exception as e:
    print(f"❌ SportsBlaze: Erro - {e}")
    sportsblaze_ok = False

# ============================================================================
# API 3: SportData.io
# ============================================================================
print("\n3️⃣ SPORTDATA.IO")
print("-" * 60)

sportdata_key = 'bc2194faba594d67b396b5fc52d42bd4'

try:
    # Testar endpoint de box scores
    # Formato: https://api.sportsdata.io/v3/nba/stats/json/BoxScores/2024-FEB-09
    url = "https://api.sportsdata.io/v3/nba/stats/json/BoxScoresByDate/2024-02-09"
    params = {'key': sportdata_key}
    
    response = requests.get(url, headers={}, params=params, timeout=10)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SportData.io: CONECTADO!")
        
        if isinstance(data, list) and len(data) > 0:
            game = data[0]
            print(f"   Game keys: {list(game.keys())[:10]}")
            
            # Procurar por stats avançadas
            if 'TeamStats' in game:
                team_stats = game['TeamStats'][0] if game['TeamStats'] else {}
                print(f"   TeamStats keys: {list(team_stats.keys())[:15]}")
                
                # Verificar campos específicos
                fb = team_stats.get('FastBreakPoints')
                sc = team_stats.get('SecondChancePoints') 
                paint = team_stats.get('PointsInThePaint')
                
                print(f"   Fast Break: {fb if fb is not None else 'N/A'}")
                print(f"   Second Chance: {sc if sc is not None else 'N/A'}")
                print(f"   Paint: {paint if paint is not None else 'N/A'}")
        
        sportdata_ok = True
    else:
        print(f"⚠️ SportData.io: Status {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        sportdata_ok = False
        
except Exception as e:
    print(f"❌ SportData.io: Erro - {e}")
    sportdata_ok = False

# ============================================================================
# RESUMO
# ============================================================================
print("\n" + "="*60)
print("📊 RESUMO DA VALIDAÇÃO")
print("="*60)
print(f"API-Football:  {'✅ OK' if api_football_ok else '❌ FALHOU'}")
print(f"SportsBlaze:   {'✅ OK' if sportsblaze_ok else '❌ FALHOU'}")
print(f"SportData.io:  {'✅ OK' if sportdata_ok else '❌ FALHOU'}")

total_ok = sum([api_football_ok, sportsblaze_ok, sportdata_ok])
print(f"\nTotal funcionando: {total_ok}/3")

if total_ok >= 2:
    print("✅ Sistema de fallback pode ser implementado!")
elif total_ok == 1:
    print("⚠️ Apenas 1 API funcional - fallback limitado")
else:
    print("❌ Nenhuma API funcional - usar fallbacks sintéticos")

print("\n✅ Validação completa!")
