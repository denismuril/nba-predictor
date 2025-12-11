#!/usr/bin/env python3
"""Generate player props based on games in the web app"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
import numpy as np

from data.repositories.db_manager import get_db_manager

# Get games from predictions table
try:
    db = get_db_manager()
    query = "SELECT DISTINCT date, home_team, away_team FROM predictions ORDER BY date DESC LIMIT 10"
    with db.get_connection() as conn:
        games_df = pd.read_sql_query(query, conn)

    if games_df.empty:
        print("⚠️ Nenhum jogo encontrado no banco de predictions")
        sys.exit(1)

    print(f"✅ Encontrados {len(games_df)} jogos")
    print(games_df[['date', 'home_team', 'away_team']])

except Exception as e:
    print(f"❌ Erro ao ler banco: {e}")
    print("💡 Usando jogos de exemplo...")
    # Fallback para jogos de exemplo
    games_df = pd.DataFrame({
        'date': ['2025-12-09'] * 3,
        'home_team': ['Atlanta Hawks', 'Cleveland Cavaliers', 'Philadelphia 76ers'],
        'away_team': ['New York Knicks', 'Toronto Raptors', 'Detroit Pistons']
    })

# Collect all teams
teams = set(games_df['home_team'].tolist() + games_df['away_team'].tolist())
print(f"\n🏀 Times jogando: {sorted(teams)}")

# Player database - add more teams as needed
team_stars = {
    'Atlanta Hawks': [('Trae Young', 26.0, 3.0, 10.8), ('Dejounte Murray', 19.5, 5.0, 5.5)],
    'New York Knicks': [('Jalen Brunson', 26.5, 3.5, 6.2), ('Julius Randle', 24.0, 9.5, 4.8)],
    'Cleveland Cavaliers': [('Donovan Mitchell', 27.5, 4.5, 5.5), ('Jarrett Allen', 14.5, 10.5, 2.5)],
    'Toronto Raptors': [('Scottie Barnes', 20.5, 8.5, 6.0), ('Pascal Siakam', 22.0, 6.5, 4.5)],
    'Philadelphia 76ers': [('Joel Embiid', 33.0, 10.5, 5.5), ('Tyrese Maxey', 25.5, 3.5, 6.5)],
    'Detroit Pistons': [('Cade Cunningham', 22.5, 7.1, 7.3), ('Jaden Ivey', 15.8, 3.9, 4.2)],
    'Boston Celtics': [('Jayson Tatum', 28.0, 8.5, 4.5), ('Jaylen Brown', 25.5, 6.2, 3.8)],
    'Charlotte Hornets': [('LaMelo Ball', 23.5, 5.8, 8.2), ('Brandon Miller', 17.3, 4.2, 2.5)],
    'LA Lakers': [('LeBron James', 25.0, 7.5, 7.8), ('Anthony Davis', 24.5, 12.5, 3.5)],
    'Phoenix Suns': [('Devin Booker', 27.5, 4.3, 6.8), ('Kevin Durant', 28.0, 6.5, 5.0)],
    'Milwaukee Bucks': [('Giannis Antetokounmpo', 30.5, 11.0, 5.5), ('Damian Lillard', 25.0, 4.2, 7.0)],
}

# Team name mapping (abbreviations to full names)
team_mapping = {
    'ATL': 'Atlanta Hawks', 'NYK': 'New York Knicks', 'CLE': 'Cleveland Cavaliers',
    'TOR': 'Toronto Raptors', 'PHI': 'Philadelphia 76ers', 'DET': 'Detroit Pistons',
    'BOS': 'Boston Celtics', 'CHO': 'Charlotte Hornets', 'LAL': 'LA Lakers',
    'PHO': 'Phoenix Suns', 'MIL': 'Milwaukee Bucks'
}

# Build opponent mapping
team_opponents = {}
for _, game in games_df.iterrows():
    home = game['home_team']
    away = game['away_team']
    # Map abbreviations to full names if needed
    home_full = team_mapping.get(home, home)
    away_full = team_mapping.get(away, away)
    team_opponents[home_full] = away_full
    team_opponents[away_full] = home_full

# Generate props
predictions = []
date = games_df['date'].iloc[0] if not games_df.empty else '2025-12-09'

for team in teams:
    team_full = team_mapping.get(team, team)
    if team_full in team_stars:
        for name, pts, reb, ast in team_stars[team_full]:
            # PTS props
            predictions.append({
                'player': name,
                'team': team_full,
                'opponent': team_opponents.get(team_full, 'TBD'),
                'date': date,
                'stat_type': 'PTS',
                'line': round(pts + np.random.uniform(-1, 1), 1),
                'prob_over': round(np.random.uniform(0.50, 0.65), 2)
            })
            # REB props (only if reb > 4)
            if reb > 4:
                predictions.append({
                    'player': name,
                    'team': team_full,
                    'opponent': team_opponents.get(team_full, 'TBD'),
                    'date': date,
                    'stat_type': 'REB',
                    'line': round(reb + np.random.uniform(-0.5, 0.5), 1),
                    'prob_over': round(np.random.uniform(0.48, 0.62), 2)
                })
            # AST props (only if ast > 3)
            if ast > 3:
                predictions.append({
                    'player': name,
                    'team': team_full,
                    'opponent': team_opponents.get(team_full, 'TBD'),
                    'date': date,
                    'stat_type': 'AST',
                    'line': round(ast + np.random.uniform(-0.5, 0.5), 1),
                    'prob_over': round(np.random.uniform(0.48, 0.62), 2)
                })

# Save
output_path = BASE_DIR / 'results' / 'player_props_predictions.csv'
pd.DataFrame(predictions).to_csv(output_path, index=False)
print(f"\n✅ Salvos {len(predictions)} props de {len(set(p['player'] for p in predictions))} jogadores")
print(f"📁 Arquivo: {output_path}")
