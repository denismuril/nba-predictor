
import json
import glob
import os
from collections import defaultdict

def merge():
    base = 'data/injuries_historical'
    files = glob.glob(os.path.join(base, 'injuries_historical_*.json'))
    print(f'Found {len(files)} files')
    
    data = defaultdict(dict)
    
    # Mapeamento de nomes de times para abreviações (simplificado)
    # Assumindo que o pipeline lida com nomes completos ou que o json já tem abreviações se vier do nbainjuries (mas vem do PDF com nomes completos)
    # Vou usar uma heurística simples: se o nome tem espaço, usa o mapeamento.
    
    TEAM_TO_ABBR = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }

    count = 0
    for f in files:
        try:
            with open(f) as fp:
                content = json.load(fp)
                if not isinstance(content, list): continue
                
                for r in content:
                    d = r.get('date')
                    if not d: continue
                    
                    inj = r.get('injuries', {})
                    for t, player_map in inj.items():
                        # Converter nome time
                        abbr = TEAM_TO_ABBR.get(t, t)
                        
                        # Converter players dict -> list
                        plist = []
                        for p_name, p_status in player_map.items():
                            plist.append({'player': p_name, 'status': p_status})
                        
                        if plist:
                            data[d][abbr] = plist
        except Exception as e:
            print(f'Error reading {f}: {e}')

    out = os.path.join(base, 'injury_date_mapping.json')
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f'Saved {len(data)} dates to {out}')

if __name__ == '__main__':
    merge()
