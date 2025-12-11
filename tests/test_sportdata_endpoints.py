"""
Test SportData.io NBA API com endpoint correto

Baseado na documentação oficial.
"""
import requests
import json

print("🏀 Testando SportData.io NBA API\n")

sportdata_key = 'bc2194faba594d67b396b5fc52d42bd4'

# Tentar diferentes formatos de endpoint
endpoints_to_try = [
    # Formato 1: v3/nba/scores
    "https://api.sportsdata.io/v3/nba/scores/json/BoxScoresByDate/2024-02-09",
    # Formato 2: v3/nba/stats  
    "https://api.sportsdata.io/v3/nba/stats/json/BoxScoresByDate/2024-02-09",
    # Formato 3: Sem date específica, tentar games ativos
    "https://api.sportsdata.io/v3/nba/scores/json/GamesByDate/2024-02-09",
    # Formato 4: Player stats
    "https://api.sportsdata.io/v3/nba/stats/json/PlayerGameStatsByDate/2024-02-09",
]

for i, url in enumerate(endpoints_to_try, 1):
    print(f"\n{i}. Testando: {url.split('json/')[-1]}")
    print("-" * 60)
    
    try:
        response = requests.get(f"{url}?key={sportdata_key}", timeout=10)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCESSO!")
            
            if isinstance(data, list):
                print(f"   Records: {len(data)}")
                if len(data) > 0:
                    print(f"   Keys: {list(data[0].keys())[:10]}")
                    
                    # Procurar por stats avançadas
                    game = data[0]
                    if 'HomeTeamScore' in game:
                        print(f"   Home Score: {game.get('HomeTeamScore')}")
                    if 'AwayTeamScore' in game:
                        print(f"   Away Score: {game.get('AwayTeamScore')}")
                        
            elif isinstance(data, dict):
                print(f"   Keys: {list(data.keys())[:10]}")
            
            print(f"\n🎉 ENDPOINT FUNCIONAL ENCONTRADO!")
            print(f"   URL: {url}")
            break
            
        elif response.status_code == 401:
            print(f"❌ Unauthorized - API key inválida ou sem permissão")
        elif response.status_code == 404:
            print(f"⚠️ Endpoint não encontrado")
        else:
            print(f"⚠️ Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Erro: {e}")

print("\n✅ Teste completo!")
