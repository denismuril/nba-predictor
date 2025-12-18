#!/usr/bin/env python
"""Save API response to file for analysis."""
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

# Save to file
with open("api_response.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved to api_response.json")

# Show summary
print(f"Keys: {list(data.keys())}")
print(f"players count: {len(data.get('players', []))}")
print(f"playerProps count: {len(data.get('playerProps', []))}")

if data.get('playerProps'):
    p = data['playerProps'][0]
    print(f"\nFirst prop keys: {list(p.keys())}")
    print(f"First prop: {json.dumps(p, indent=2)}")
