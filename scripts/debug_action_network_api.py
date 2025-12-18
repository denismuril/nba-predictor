#!/usr/bin/env python
"""Detailed debug for action network API structure."""
import requests
from datetime import datetime
import json

date_str = datetime.now().strftime("%Y%m%d")

url = "https://api.actionnetwork.com/web/v2/leagues/4/projections/available"
params = {
    "date": date_str,
    "isLive": "false",
    "limit": "100"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.actionnetwork.com/"
}

print(f"Calling API for date {date_str}...")
response = requests.get(url, params=params, headers=headers, timeout=15)
print(f"Status: {response.status_code}")

data = response.json()
print(f"\n=== Top-level keys ===")
print(list(data.keys()))

# Players structure
print(f"\n=== Players Structure ===")
if "players" in data:
    players = data["players"]
    print(f"Type: {type(players)}")
    print(f"Count: {len(players)}")
    if isinstance(players, dict):
        # Show one entry
        first_key = list(players.keys())[0]
        print(f"First key: {first_key}")
        print(f"First value: {players[first_key]}")
    elif isinstance(players, list) and players:
        print(f"First item: {players[0]}")
else:
    print("No 'players' key!")

# PlayerProps structure
print(f"\n=== PlayerProps Structure ===")
if "playerProps" in data:
    props = data["playerProps"]
    print(f"Type: {type(props)}")
    print(f"Count: {len(props)}")
    if props:
        print(f"\nFirst prop keys: {list(props[0].keys())}")
        print(f"\nFirst prop full:")
        print(json.dumps(props[0], indent=2))
else:
    print("No 'playerProps' key!")


print()

# Teste 3: Markets sem bookIds
print("=" * 70)
print("TESTE 3: Markets sem bookIds")
print("=" * 70)
url3 = "https://api.actionnetwork.com/web/v2/scoreboard/nba/markets"
params3 = {
    "date": date_str
}

try:
    response = requests.get(url3, params=params3, headers=headers, timeout=15)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCESSO!")
        print(f"Keys: {list(data.keys())}")
    else:
        print(f"❌ Falhou: {response.status_code}")
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Erro: {e}")

print()

# Teste 4: Props endpoint alternativo
print("=" * 70)
print("TESTE 4: Props endpoint alternativo")
print("=" * 70)
url4 = "https://api.actionnetwork.com/web/v2/props"
params4 = {
    "date": date_str,
    "league": "nba"
}

try:
    response = requests.get(url4, params=params4, headers=headers, timeout=15)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCESSO!")
        print(f"Keys: {list(data.keys())}")
    else:
        print(f"❌ Falhou: {response.status_code}")
except Exception as e:
    print(f"❌ Erro: {e}")

print("\n" + "=" * 70)
print("Fim dos testes")
print("=" * 70)
