
import requests
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

print("--- DEBUGGING STANDINGS MAPPING ---")

TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
}

def load_standings_debug():
    url = "http://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        standings = {}
        children = data.get('children', [])
        for child in children:
            entries = child.get('standings', {}).get('entries', [])
            for entry in entries:
                team_name = entry['team']['displayName']
                wins = 0
                losses = 0
                for stat in entry.get('stats', []):
                    if stat['name'] == 'wins': wins = int(stat['value'])
                    elif stat['name'] == 'losses': losses = int(stat['value'])
                standings[team_name] = {'wins': wins, 'losses': losses}
        
        print(f"Raw Teams Found: {len(standings)}")
        if len(standings) > 0:
            print(f"Sample Key: {list(standings.keys())[0]}")
        
        # Check mapping
        found_abbrs = []
        for team_name in standings.keys():
            abbr = TEAM_MAP.get(team_name)
            if abbr:
                found_abbrs.append(abbr)
            else:
                print(f"⚠️ MISSING MAPPING FOR: '{team_name}'")
        
        print(f"Mapped Abbreviations ({len(found_abbrs)}): {sorted(found_abbrs)}")
        
        return standings
    except Exception as e:
        print(f"Error fetching standings: {e}")

load_standings_debug()


print("\n--- DEBUGGING RESULTS API (NBA_API) ---")
try:
    # Try fetching results for a specific past date
    board = scoreboardv2.ScoreboardV2(game_date='2025-12-10')
    games = board.game_header.get_dict()['data']
    lines = board.line_score.get_dict()['data']
    
    print(f"Games found for 2025-12-10: {len(games)}")
    for game in games:
        game_id = game[2]
        home_team_id = game[6]
        print(f"Game: {game_id} - Status: {game[4]}")  # 4 is Status
        
    # Check if we can get scores
    if lines:
        print("Line Scores available (Sample):")
        print(lines[0])
except Exception as e:
    print(f"NBA API Error: {e}")
